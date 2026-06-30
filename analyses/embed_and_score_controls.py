"""Embed a composition-control set through Evo2 and score within-family recovery.

The companion to make_control_sequences.py. For one control set (a mirror data dir under
data/evo2_gene_families/controls/<control>/), embed every control CDS through the SAME
Evo2 tap as the natural run (blocks.24, pool last EMBED_BP), build the k-NN geodesic, and
compute the per-family within-family Spearman ρ of the geodesic vs:
  - taxonomy  — the FIXED, clade-structured ground truth (host organism rank distance);
                this is the confounder the controls test. If a control retains this ρ,
                Evo2's within-family "phylogeny" signal is a composition artifact.
  - k-mer     — the control's OWN nucleotide composition (sanity: did the embedding keep
                any structure at all).

Writes <control>/axisB_control_scores.csv. The natural baseline to compare against is
results/<date>_evo2-gene-families/axisB_within_family_correlations.csv
(spearman_geodesic_taxonomy). Embeddings cache per control under <control>/embeddings/,
keyed by org_gene, so re-runs resume.

In addition to RECOVERY (geodesic vs taxonomy) this also reports RECONSTRUCTION
(PRESERVATION): the Spearman ρ of each control's geodesic vs the NATURAL run's geodesic,
both within-family (per-gene submatrices) and between-family (centroid). ρ→1 means the
control reconstructed the natural geometry (that aspect was not load-bearing); ρ falling
means the ablated aspect carried real structure.

Outputs (all under <natural-run>/controls/):
  control_within_scores.csv   per-family: spearman_geodesic_taxonomy + rho_geodesic_vs_natural
  control_between_scores.csv  per-control: rho_vs_natural_centroid (+ rho_centroid_vs_pfamjsd)

Usage:
    # cross-kingdom (ortholog) panel — recovery + within/between preservation
    uv run python analyses/embed_and_score_controls.py --control all --axis within  --layer 15 --natural-run <run>
    uv run python analyses/embed_and_score_controls.py --control all --axis between --layer 15 --natural-run <run>
    # matched-human panel — read the sibling control run dirs, preservation only
    uv run python analyses/embed_and_score_controls.py --control all --panel human  --layer 13 --natural-run <run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# This script lives in analyses/; its helper modules live under scripts/, scripts/evo2/, scripts/baselines/.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "scripts" / "evo2"))
sys.path.insert(0, str(_REPO / "scripts" / "baselines"))
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402
from taxonomic_distance import taxonomic_distance_matrix  # noqa: E402
from geodesic_utils import (  # noqa: E402
    between_preservation_rho,
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
    upper_triangle,
    within_preservation_rho,
)

# Reuse the exact embedding op + tap from the natural pipeline (shared Evo2 engine).
from evo2_embedding import EMBED_LAYER, embed_one, load_model  # noqa: E402

CONTROL_ROOT = Path("data/evo2_gene_families/controls")
ALL_CONTROLS = ["dinuc_shuffle", "codon_shuffle", "synonymous_recode", "gc_match",
                "kmer4_shuffle", "kmer6_shuffle"]
# Controls applicable to a genomic span (no reading frame): the matched-human panel set.
HUMAN_CONTROLS = ["gc_match", "dinuc_shuffle", "kmer6_shuffle"]


def load_control_sequences(control_dir: Path, families: list[str]) -> dict[str, str]:
    """org_gene -> control CDS, from the control dir's per-family FASTAs."""
    seqs: dict[str, str] = {}
    for fam in sorted(set(families)):
        fasta = control_dir / f"{fam}.fasta"
        cur_id, cur = None, []
        for line in fasta.read_text().splitlines():
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur)
                cur_id, cur = line[1:].split("|")[0], []
            elif line.strip():
                cur.append(line.strip().upper())
        if cur_id:
            seqs[cur_id] = "".join(cur)
    return seqs


def embed_control(control_dir: Path, device: str, layer: str = EMBED_LAYER) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed every CDS in a control dir (resumable per-gene cache), at residual tap `layer`."""
    manifest = pd.read_csv(control_dir / "manifest.csv")
    seqs = load_control_sequences(control_dir, manifest["family"].tolist())
    # Layer-tagged cache so blocks.24 and blocks.15 control embeddings never collide.
    tag = layer.replace("blocks.", "blocks")
    emb_dir = control_dir / ("embeddings" if layer == EMBED_LAYER else f"embeddings_{tag}")
    emb_dir.mkdir(parents=True, exist_ok=True)
    npy, meta_csv = emb_dir / "embeddings.npy", emb_dir / "metadata.csv"

    cache: dict[str, np.ndarray] = {}
    if npy.exists() and meta_csv.exists():
        prev, pm = np.load(npy), pd.read_csv(meta_csv)
        if len(prev) == len(pm):
            cache = {g: prev[i] for i, g in enumerate(pm["org_gene"])}
            print(f"  resuming: {len(cache)} cached embeddings")

    org_genes = manifest["org_gene"].tolist()
    todo = sum(1 for g in org_genes if g not in cache)
    print(f"  embedding {todo} / {len(org_genes)} CDS")
    model = None
    embs, rows = [], []
    for i, (_, row) in enumerate(tqdm(manifest.iterrows(), total=len(manifest), desc="embed")):
        og = row["org_gene"]
        if og in cache:
            emb = cache[og]
        else:
            if model is None:
                model = load_model()
            emb = embed_one(seqs[og], model, device, layer=layer)
        embs.append(emb)
        rows.append(row.to_dict())
        if (i + 1) % 200 == 0:
            np.save(npy, np.stack(embs))
            pd.DataFrame(rows).to_csv(meta_csv, index=False)
    E = np.stack(embs)
    meta = pd.DataFrame(rows)
    np.save(npy, E)
    meta.to_csv(meta_csv, index=False)
    return E, meta


def score_within_family(geodesic: np.ndarray, meta: pd.DataFrame, seqs: dict[str, str]) -> pd.DataFrame:
    """Per-family within-family geodesic ρ vs taxonomy (fixed) and vs control k-mer."""
    families = meta["family"].to_numpy()
    org_genes = meta["org_gene"].tolist()
    tax = taxonomic_distance_matrix(meta)
    kmer = kmer_distance_matrix([seqs[g] for g in org_genes], k=6)
    rows = []
    for fam in sorted(set(families)):
        idx = np.where(families == fam)[0]
        if len(idx) < 4:
            continue
        geo = upper_triangle(geodesic[np.ix_(idx, idx)])
        tx = upper_triangle(tax[np.ix_(idx, idx)])
        km = upper_triangle(kmer[np.ix_(idx, idx)])
        rt, pt = (spearmanr(geo, tx) if np.ptp(tx) > 0 else (float("nan"), float("nan")))
        rk, pk = spearmanr(geo, km)
        rows.append({
            "family": fam, "n_members": len(idx),
            "spearman_geodesic_taxonomy": rt, "p_taxonomy": pt,
            "spearman_geodesic_kmer": rk, "p_kmer": pk,
        })
    return pd.DataFrame(rows)


def family_centroid_distances(geodesic: np.ndarray, families: np.ndarray, order: list[str]) -> np.ndarray:
    """(F, F) mean pairwise geodesic: within-family on the diagonal, between off-diagonal."""
    F = len(order)
    C = np.zeros((F, F))
    idx = {f: np.where(families == f)[0] for f in order}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                sub = geodesic[np.ix_(idx[a], idx[a])]
                C[i, j] = sub[np.triu_indices(len(idx[a]), 1)].mean() if len(idx[a]) > 1 else 0.0
            else:
                C[i, j] = geodesic[np.ix_(idx[a], idx[b])].mean()
    return C


def run_one_within(control: str, device: str, layer: str):
    """Embed a control, build its geodesic, score within-family recovery vs taxonomy/k-mer,
    and write the per-control axisB_control_scores.csv. Returns (geodesic, meta, scores)."""
    control_dir = CONTROL_ROOT / control
    if not (control_dir / "manifest.csv").exists():
        sys.exit(f"No control set at {control_dir} — run make_control_sequences.py first")
    print(f"\n=== {control} ({layer}) within ===")
    E, meta = embed_control(control_dir, device, layer=layer)
    k_opt, W = find_min_connected_k(E, k_min=3)
    geodesic = compute_geodesic(W)
    seqs = load_control_sequences(control_dir, meta["family"].tolist())
    scores = score_within_family(geodesic, meta, seqs)
    tag = layer.replace("blocks.", "blocks")
    out = control_dir / ("axisB_control_scores.csv" if layer == EMBED_LAYER
                         else f"axisB_control_scores_{tag}.csv")
    scores.to_csv(out, index=False)
    print(f"  k={k_opt}; saved {out}")
    return geodesic, meta, scores


def run_within(controls: list[str], device: str, layer: str, nat_dir: Path) -> None:
    """WITHIN-family control scoring for the cross-kingdom (ortholog) panel. Per control:
    the RECOVERY view (geodesic vs taxonomy, from run_one_within) AND the PRESERVATION view
    (control per-gene geodesic vs the NATURAL per-gene geodesic, per family). Writes the
    consolidated control_within_scores.csv into <nat_dir>/controls/ (the recovery-only
    per-control axisB CSVs are still written under each control dir for back-compat)."""
    nat_geo_df = pd.read_csv(nat_dir / "evo2_gene_family_geodesic_labeled.csv", index_col=0)
    family_order = pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0).index.tolist()
    rows = []
    for control in controls:
        geodesic, meta, scores = run_one_within(control, device, layer=layer)
        # PRESERVATION: align control + natural geodesics on their shared genes, per family.
        ctrl_geo_df = pd.DataFrame(geodesic, index=meta["org_gene"], columns=meta["org_gene"])
        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fam_of = dict(zip(meta["org_gene"], meta["family"]))
        fams = np.array([fam_of[g] for g in common])
        pres = dict((f, rho) for f, _, rho in within_preservation_rho(
            nat_geo_df.loc[common, common].values, ctrl_geo_df.loc[common, common].values,
            fams, family_order)[0])
        for _, s in scores.iterrows():
            rows.append({"condition": control, "family": s["family"],
                         "n_members": int(s["n_members"]),
                         "spearman_geodesic_taxonomy": s["spearman_geodesic_taxonomy"],
                         "rho_geodesic_vs_natural": pres.get(s["family"], np.nan)})
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "control_within_scores.csv", index=False)
    pres_summ = df.groupby("condition", sort=False)["rho_geodesic_vs_natural"].mean()
    print("\nMean within-family geodesic-vs-natural ρ per control (PRESERVATION):")
    print(pres_summ.round(3).to_string())
    print(f"Saved {out_dir}/control_within_scores.csv")


def run_human(controls: list[str], layer: int, nat_dir: Path) -> None:
    """WITHIN + BETWEEN control preservation for the Evo2 matched-HUMAN panel. The human
    controls are full re-embed run dirs (scripts/evo2/embed_and_geodesic_paralog.py --control),
    so we read each sibling run's saved geodesic + centroid (no taxonomy ground truth for human
    paralogs — preservation only) and score control-vs-natural at both axes, writing
    control_within_scores.csv + control_between_scores.csv into <nat_dir>/controls/."""
    repo = Path(__file__).resolve().parent.parent
    nat_geo_df = pd.read_csv(nat_dir / "evo2_human_geodesic_labeled.csv", index_col=0)
    nat_cen_df = pd.read_csv(nat_dir / "evo2_human_centroid_distances.csv", index_col=0)
    family_order = nat_cen_df.index.tolist()
    nat_meta = pd.read_csv(nat_dir / "metadata.csv")
    fam_of = dict(zip(nat_meta["gene"], nat_meta["family"]))
    within_rows, between_rows = [], [{"condition": "natural", "rho_vs_natural_centroid": 1.0}]
    for control in controls:
        sibs = sorted(repo.glob(f"results/*_evo2-human-panel-blocks{layer}-{control}"))
        if not sibs:
            print(f"  SKIP {control}: no run dir results/*-blocks{layer}-{control}")
            continue
        cdir = sibs[-1]
        ctrl_geo_df = pd.read_csv(cdir / "evo2_human_geodesic_labeled.csv", index_col=0)
        ctrl_cen_df = pd.read_csv(cdir / "evo2_human_centroid_distances.csv", index_col=0)
        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fams = np.array([fam_of.get(g, "NA") for g in common])
        prows, _ = within_preservation_rho(
            nat_geo_df.loc[common, common].values, ctrl_geo_df.loc[common, common].values,
            fams, family_order)
        for fam, n, rho in prows:
            within_rows.append({"condition": control, "family": fam, "n_members": n,
                                "rho_geodesic_vs_natural": rho})
        brho = between_preservation_rho(
            nat_cen_df.reindex(index=family_order, columns=family_order).values,
            ctrl_cen_df.reindex(index=family_order, columns=family_order).values)
        between_rows.append({"condition": control, "rho_vs_natural_centroid": brho})
        print(f"  {control:16} within(mean)={np.mean([r[2] for r in prows]):.3f}  between={brho:.3f}  ({cdir.name})")
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(within_rows).to_csv(out_dir / "control_within_scores.csv", index=False)
    pd.DataFrame(between_rows).to_csv(out_dir / "control_between_scores.csv", index=False)
    print(f"Saved {out_dir}/control_within_scores.csv + control_between_scores.csv")


def run_between(controls: list[str], device: str, layer: str, nat_dir: Path) -> None:
    """Mirror control: BETWEEN-family centroid geodesic of each control vs the NATURAL
    centroid geometry and vs Pfam-JSD (parallel to the GPN MSA between-control)."""
    order = pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0).index.tolist()
    nat_cen = pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0).reindex(
        index=order, columns=order).values
    jsd_p = nat_dir / "pfam_jsd_distances.csv"
    jsd = (pd.read_csv(jsd_p, index_col=0).reindex(index=order, columns=order).values
           if jsd_p.exists() else None)
    iu = np.triu_indices(len(order), 1)

    def rho(C, ref):
        if ref is None:
            return float("nan")
        c, r = C[iu], ref[iu]
        ok = np.isfinite(c) & np.isfinite(r)
        return float(spearmanr(c[ok], r[ok])[0]) if ok.sum() >= 4 and np.ptp(r[ok]) > 0 else float("nan")

    rows = [{"condition": "natural", "rho_vs_natural_centroid": 1.0,
             "rho_centroid_vs_pfamjsd": rho(nat_cen, jsd)}]
    for control in controls:
        control_dir = CONTROL_ROOT / control
        print(f"\n=== {control} ({layer}) between ===")
        E, meta = embed_control(control_dir, device, layer=layer)
        C = compute_centroid_geodesic(E, meta["family"].to_numpy(), order)
        rows.append({"condition": control,
                     "rho_vs_natural_centroid": rho(C, nat_cen),
                     "rho_centroid_vs_pfamjsd": rho(C, jsd)})
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "control_between_scores.csv", index=False)
    print(f"\nBETWEEN-family control (centroid geodesic vs natural / Pfam-JSD):")
    print(df.round(3).to_string(index=False))
    print(f"Saved {out_dir}/control_between_scores.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, choices=ALL_CONTROLS + ["all"])
    ap.add_argument("--axis", choices=["within", "between"], default="within",
                    help="within: per-family geodesic vs taxonomy + vs-natural preservation. "
                         "between: family-centroid geodesic vs natural + Pfam-JSD.")
    ap.add_argument("--panel", choices=["ortholog", "human"], default="ortholog",
                    help="ortholog: cross-kingdom KEGG controls (re-embed from data/ + taxonomy). "
                         "human: matched-human panel — read the sibling control run dirs and score "
                         "control-vs-natural preservation only (no taxonomy for human paralogs).")
    ap.add_argument("--natural-run", default=None,
                    help="Natural run dir. Required for --axis between, --panel human, and to add "
                         "the within-family preservation column on --panel ortholog.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Evo2 block index to tap (default: the production EMBED_LAYER). e.g. 15.")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = EMBED_LAYER if args.layer is None else f"blocks.{args.layer}"
    print(f"Device: {device} | tap: {layer} | panel: {args.panel} | axis: {args.axis}")
    controls = ALL_CONTROLS if args.control == "all" else [args.control]

    if args.panel == "human":
        # Human controls are full re-embed run dirs; both axes are pure preservation.
        if args.control == "all":
            controls = HUMAN_CONTROLS
        if not args.natural_run:
            sys.exit("--panel human requires --natural-run")
        if args.layer is None:
            sys.exit("--panel human requires --layer (the block the human run dirs were built at)")
        run_human(controls, args.layer, Path(args.natural_run))
    elif args.axis == "between":
        if not args.natural_run:
            sys.exit("--axis between requires --natural-run")
        run_between(controls, device, layer, Path(args.natural_run))
    elif args.natural_run:
        run_within(controls, device, layer, Path(args.natural_run))
    else:  # back-compat: recovery-only per-control axisB, no preservation aggregation
        for c in controls:
            run_one_within(c, device, layer=layer)
    print("\nDone.")


if __name__ == "__main__":
    main()
