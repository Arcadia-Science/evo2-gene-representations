"""Embed a composition-control set through Evo2 and score within-family recovery."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "scripts" / "evo2"))
sys.path.insert(0, str(_REPO / "scripts" / "baselines"))
import arcadia_style as acs  # noqa: E402
from control_tables import write_control_table  # noqa: E402

# Reuse the exact embedding op + tap from the natural pipeline (shared Evo2 engine).
from evo2_embedding import EMBED_LAYER, embed_one, load_model  # noqa: E402
from geodesic_utils import (  # noqa: E402
    between_preservation_rho,
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
    upper_triangle,
    within_preservation_rho,
)
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402
from taxonomic_distance import taxonomic_distance_matrix  # noqa: E402

CONTROL_ROOT = Path("data/evo2_gene_families/controls")
ALL_CONTROLS = [
    "dinuc_shuffle",
    "codon_shuffle",
    "synonymous_recode",
    "gc_match",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "missense_subset",
]
# Controls applicable to a genomic span (no reading frame): the matched-human panel set.
HUMAN_CONTROLS = ["gc_match", "dinuc_shuffle", "kmer4_shuffle", "kmer6_shuffle"]
# On CDS input the reading frame exists, so the protein/codon controls the cross-kingdom pipeline
# uses become meaningful too: codon_shuffle (scrambles protein, keeps codon counts),
# synonymous_recode (keeps protein, resamples synonymous codons) and its nonsynonymous counterpart
# missense_subset (edits a strict SUBSET of the recode's changed bases, damaging the protein).
HUMAN_CDS_CONTROLS = HUMAN_CONTROLS + ["codon_shuffle", "synonymous_recode", "missense_subset"]
# Display order/labels/colours for the human-panel summary figure. Composition gradient:
# 6-mer (most preserved) → 4-mer → codon → dinucleotide → GC-only, then the nested pair:
# synonymous_recode (protein kept) and missense_subset, which edits a strict SUBSET of the recode's
# changed bases — fewer nucleotides moved, protein damaged. Colours match control_comparison_figure.
# figure_human_controls filters this to the controls actually present, so genomic runs (4 controls)
# render unchanged.
_C = acs.CONTROL_COLORS
HUMAN_CONTROL_DISPLAY = [
    ("kmer6_shuffle", "6-mer shuffle", _C["kmer6_shuffle"]),
    ("kmer4_shuffle", "4-mer shuffle", _C["kmer4_shuffle"]),
    ("codon_shuffle", "Codon shuffle", _C["codon_shuffle"]),
    ("dinuc_shuffle", "Dinucleotide shuffle", _C["dinuc_shuffle"]),
    ("gc_match", "GC-matched random", _C["gc_match"]),
    ("synonymous_recode", "Synonymous recode (protein kept)", _C["synonymous_recode"]),
    ("missense_subset", "Missense at recode sites (protein damaged)", _C["missense_subset"]),
]


def resolve_latest_run(pattern: str) -> Path | None:
    """Return the newest run whose directory name matches the layer suffix exactly."""
    import glob as _glob

    hits = sorted(_glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(hits[0]) if hits else None


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


def embed_control(
    control_dir: Path, device: str, layer: str = EMBED_LAYER
) -> tuple[np.ndarray, pd.DataFrame]:
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


def embed_control_stack(control: str, device: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed every control CDS once and cache the full layer stack."""
    from evo2_embedding import N_BLOCKS, embed_all_blocks  # lazy: no GPU when fully cached

    control_dir = CONTROL_ROOT / control
    if not (control_dir / "manifest.csv").exists():
        sys.exit(f"No control set at {control_dir} — run make_control_sequences.py first")
    manifest = pd.read_csv(control_dir / "manifest.csv")
    org_genes = manifest["org_gene"].tolist()
    N = len(org_genes)
    cache_dir = Path("data/cache") / f"evo2_layer_sweep_{control}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stack_path, meta_path = cache_dir / "layer_stack.npy", cache_dir / "metadata.csv"
    done_path = cache_dir / "genes_done.txt"

    # Fully cached?
    if stack_path.exists() and meta_path.exists() and done_path.exists():
        if (
            pd.read_csv(meta_path)["org_gene"].tolist() == org_genes
            and done_path.read_text().splitlines() == org_genes
        ):
            print(f"  [{control}] reusing cached layer stack {stack_path}")
            return np.load(stack_path), manifest

    seqs = load_control_sequences(control_dir, manifest["family"].tolist())
    # Resume a partial build: reload the stack + how many manifest-order genes are done.
    stack = np.zeros((N_BLOCKS, N, 4096), dtype=np.float32)
    done: list[str] = []
    if stack_path.exists() and done_path.exists():
        prev = np.load(stack_path)
        prev_done = done_path.read_text().splitlines()
        if prev.shape == stack.shape and prev_done == org_genes[: len(prev_done)]:
            stack, done = prev, prev_done
            print(f"  [{control}] resuming: {len(done)}/{N} CDS already embedded")
    model = load_model()
    start = len(done)
    for i in tqdm(
        range(start, N), desc=f"{control} (all {N_BLOCKS} blocks)", total=N, initial=start
    ):
        stack[:, i, :] = embed_all_blocks(seqs[org_genes[i]], model, device)
        done.append(org_genes[i])
        if (i + 1) % 200 == 0 or i == N - 1:
            tmp = stack_path.with_suffix(".tmp.npy")
            np.save(tmp, stack)
            tmp.replace(stack_path)
            done_path.write_text("\n".join(done))
    manifest.to_csv(meta_path, index=False)
    print(f"  [{control}] saved layer stack {stack.shape} -> {stack_path}")
    return stack, manifest


def score_within_family(
    geodesic: np.ndarray, meta: pd.DataFrame, seqs: dict[str, str]
) -> pd.DataFrame:
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
        rt, pt = spearmanr(geo, tx) if np.ptp(tx) > 0 else (float("nan"), float("nan"))
        rk, pk = spearmanr(geo, km)
        rows.append(
            {
                "family": fam,
                "n_members": len(idx),
                "spearman_geodesic_taxonomy": rt,
                "p_taxonomy": pt,
                "spearman_geodesic_kmer": rk,
                "p_kmer": pk,
            }
        )
    return pd.DataFrame(rows)


def family_centroid_distances(
    geodesic: np.ndarray, families: np.ndarray, order: list[str]
) -> np.ndarray:
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
    out = control_dir / (
        "axisB_control_scores.csv" if layer == EMBED_LAYER else f"axisB_control_scores_{tag}.csv"
    )
    scores.to_csv(out, index=False)
    print(f"  k={k_opt}; saved {out}")
    return geodesic, meta, scores


def run_within(controls: list[str], device: str, layer: str, nat_dir: Path) -> None:
    """WITHIN-family control scoring for the cross-kingdom (ortholog) panel."""
    nat_geo_df = pd.read_csv(nat_dir / "evo2_gene_family_geodesic_labeled.csv", index_col=0)
    family_order = pd.read_csv(
        nat_dir / "family_centroid_distances.csv", index_col=0
    ).index.tolist()
    rows = []
    for control in controls:
        geodesic, meta, scores = run_one_within(control, device, layer=layer)
        # PRESERVATION: align control + natural geodesics on their shared genes, per family.
        ctrl_geo_df = pd.DataFrame(geodesic, index=meta["org_gene"], columns=meta["org_gene"])
        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fam_of = dict(zip(meta["org_gene"], meta["family"], strict=False))
        fams = np.array([fam_of[g] for g in common])
        pres = dict(
            (f, rho)
            for f, _, rho in within_preservation_rho(
                nat_geo_df.loc[common, common].values,
                ctrl_geo_df.loc[common, common].values,
                fams,
                family_order,
            )[0]
        )
        for _, s in scores.iterrows():
            rows.append(
                {
                    "condition": control,
                    "family": s["family"],
                    "n_members": int(s["n_members"]),
                    "spearman_geodesic_taxonomy": s["spearman_geodesic_taxonomy"],
                    "rho_geodesic_vs_natural": pres.get(s["family"], np.nan),
                }
            )
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_control_table(
        out_dir / "control_within_scores.csv",
        df,
        generator="embed_and_score_controls:consolidate_within",
    )
    pres_summ = df.groupby("condition", sort=False)["rho_geodesic_vs_natural"].mean()
    print("\nMean within-family geodesic-vs-natural ρ per control (PRESERVATION):")
    print(pres_summ.round(3).to_string())
    print(f"Saved {out_dir}/control_within_scores.csv")


def _labeled_geodesic_path(nat_dir: Path) -> Path:
    """The natural cross-kingdom labeled geodesic, gzipped in sweep run dirs. pandas reads
    either transparently; return whichever exists (.csv preferred, else .csv.gz)."""
    plain = nat_dir / "evo2_gene_family_geodesic_labeled.csv"
    return plain if plain.exists() else nat_dir / "evo2_gene_family_geodesic_labeled.csv.gz"


def _score_ortholog_layer(
    controls: list[str],
    layer_idx: int,
    nat_dir: Path,
    emb_of: dict[str, tuple[np.ndarray, pd.DataFrame]],
) -> None:
    """WITHIN + BETWEEN control scoring for the cross-kingdom panel at one block, from precomputed
    embeddings emb_of[control] = (E, meta) (a stack slice).
    """
    nat_geo_df = pd.read_csv(_labeled_geodesic_path(nat_dir), index_col=0)
    order = pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0).index.tolist()
    nat_cen = (
        pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0)
        .reindex(index=order, columns=order)
        .values
    )
    jsd_p = nat_dir / "pfam_jsd_distances.csv"
    jsd = (
        pd.read_csv(jsd_p, index_col=0).reindex(index=order, columns=order).values
        if jsd_p.exists()
        else None
    )
    iu = np.triu_indices(len(order), 1)

    def rho(C, ref):
        if ref is None:
            return float("nan")
        c, r = C[iu], ref[iu]
        ok = np.isfinite(c) & np.isfinite(r)
        return (
            float(spearmanr(c[ok], r[ok])[0])
            if ok.sum() >= 4 and np.ptp(r[ok]) > 0
            else float("nan")
        )

    within_rows = []
    between_rows = [
        {
            "condition": "natural",
            "rho_vs_natural_centroid": 1.0,
            "rho_centroid_vs_pfamjsd": rho(nat_cen, jsd),
        }
    ]
    for control in controls:
        E, meta = emb_of[control]
        seqs = load_control_sequences(CONTROL_ROOT / control, meta["family"].tolist())
        _, W = find_min_connected_k(E, k_min=3)
        geodesic = compute_geodesic(W)
        scores = score_within_family(geodesic, meta, seqs)  # recovery vs taxonomy / k-mer
        # WITHIN preservation: control per-gene geodesic vs the natural per-gene geodesic.
        ctrl_geo_df = pd.DataFrame(geodesic, index=meta["org_gene"], columns=meta["org_gene"])
        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fam_of = dict(zip(meta["org_gene"], meta["family"], strict=False))
        fams = np.array([fam_of[g] for g in common])
        pres = dict(
            (f, r)
            for f, _, r in within_preservation_rho(
                nat_geo_df.loc[common, common].values,
                ctrl_geo_df.loc[common, common].values,
                fams,
                order,
            )[0]
        )
        for _, s in scores.iterrows():
            within_rows.append(
                {
                    "condition": control,
                    "family": s["family"],
                    "n_members": int(s["n_members"]),
                    "spearman_geodesic_taxonomy": s["spearman_geodesic_taxonomy"],
                    "rho_geodesic_vs_natural": pres.get(s["family"], np.nan),
                }
            )
        # BETWEEN preservation: control centroid geodesic vs natural centroid + Pfam-JSD.
        C = compute_centroid_geodesic(E, meta["family"].to_numpy(), order)
        between_rows.append(
            {
                "condition": control,
                "rho_vs_natural_centroid": rho(C, nat_cen),
                "rho_centroid_vs_pfamjsd": rho(C, jsd),
            }
        )
        wmean = np.nanmean([pres.get(s["family"], np.nan) for _, s in scores.iterrows()])
        print(
            f"  {control:18} within(mean)={wmean:.3f}  "
            f"between={between_rows[-1]['rho_vs_natural_centroid']:.3f}"
        )

    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_control_table(
        out_dir / "control_within_scores.csv",
        pd.DataFrame(within_rows),
        generator="embed_and_score_controls:human_all_layers",
    )
    write_control_table(
        out_dir / "control_between_scores.csv",
        pd.DataFrame(between_rows),
        generator="embed_and_score_controls:human_all_layers",
    )


def run_ortholog_all_layers(controls: list[str], nat_dir_for, device: str) -> None:
    """Cross-kingdom controls at EVERY block."""
    stacks: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for control in controls:
        print(f"\n=== building {control} layer stack ===")
        stacks[control] = embed_control_stack(control, device)
    n_layers = next(iter(stacks.values()))[0].shape[0]
    for L in range(n_layers):
        nat_dir = nat_dir_for(L)
        if nat_dir is None or not nat_dir.exists():
            print(f"\n########## blocks.{L}: SKIP (no natural run dir) ##########")
            continue
        print(f"\n########## blocks.{L} -> {nat_dir} ##########")
        emb_of = {c: (stack[L], manifest) for c, (stack, manifest) in stacks.items()}
        _score_ortholog_layer(controls, L, nat_dir, emb_of)


def figure_human_controls(out_dir: Path) -> None:
    """Summary figure for the Evo2 matched-HUMAN panel controls, reading the consolidated
    control_within_scores.csv (+ control_between_scores.csv) in `out_dir`.
    """
    import matplotlib

    matplotlib.use("Agg")
    from plot_utils import control_preservation_figure  # scripts/ already on sys.path

    within = pd.read_csv(out_dir / "control_within_scores.csv")
    between_p = out_dir / "control_between_scores.csv"
    between = pd.read_csv(between_p) if between_p.exists() else None
    # Family order: largest families first (stable across controls).
    fam_order = (
        within.drop_duplicates("family")
        .sort_values("n_members", ascending=False)["family"]
        .tolist()
    )
    within_by = {
        k: dict(zip(g["family"], g["rho_geodesic_vs_natural"], strict=False))
        for k, g in within.groupby("condition")
    }
    between_by = (
        dict(zip(between["condition"], between["rho_vs_natural_centroid"], strict=False))
        if between is not None
        else None
    )
    # Render only the controls actually scored (genomic runs = 4; CDS runs = 6).
    present = set(within["condition"])
    display = [(k, lbl, c) for k, lbl, c in HUMAN_CONTROL_DISPLAY if k in present]
    ok = control_preservation_figure(
        out_dir / "control_preservation",
        fam_order,
        display,
        within_by,
        between_by,
        title_left="Evo2-human composition controls: is the within-family geometry preserved?",
    )
    if ok:
        print(f"  Saved {out_dir}/control_preservation.{{pdf,png}}")
    else:
        print("  [figure] no recognised controls in control_within_scores.csv — skipping figure")


def _human_setup(
    controls: list[str], device: str, force_reembed: bool, input_mode: str = "genomic"
):
    """Load the human panel and cached control embeddings."""
    # Import lazily to keep the cross-kingdom path lightweight.
    import embed_and_geodesic_paralog as ev  # noqa: E402
    import sample_human_genes as ss  # noqa: E402

    # CDS panel is Evo2-only and needs no genomic locus, so load it the SAME way the CDS embedder
    # did (require_both=False, resolve_loci=False) — matches the natural run's 1122 genes exactly
    # and
    # avoids per-gene Ensembl REST. Genomic keeps the matched-both, locus-resolved behavior.
    genes, fams, locus_of, fam_order = ss.load_matched_panel(
        None, require_both=(input_mode != "cds"), resolve_loci=(input_mode != "cds")
    )
    if input_mode == "cds":
        # Same gene subset the CDS embedder keeps (genes with a cached canonical CDS), same order.
        cds = json.loads(Path("data/cache/cds_sequences.json").read_text())
        keep = [(g, f) for g, f in zip(genes, fams, strict=False) if g in cds]
        genes, fams = [g for g, _ in keep], [f for _, f in keep]
        fam_order = [f for f in fam_order if f in set(fams)]
        seqs_natural = {g: cds[g] for g in genes}
        base = ev.CACHE_DIR.parent / "evo2_human_layer_sweep_cds"
    else:
        seqs_natural = ev.load_or_fetch_genomic(genes, locus_of, ev.DEFAULT_MAX_LEN)
        base = ev.CACHE_DIR  # data/cache/evo2_human_layer_sweep
    fams_arr = np.array(fams)
    fam_of = dict(zip(genes, fams, strict=False))
    stacks: dict[str, np.ndarray] = {}
    for control in controls:
        print(f"\n=== loading {control} layer stack ({input_mode}) ===")
        seqs = ev.apply_control(seqs_natural, control, fam_of=fam_of)
        cache_dir = base.parent / f"{base.name}_{control}"
        stacks[control] = ev.run_sweep(
            genes,
            fams,
            seqs,
            device,
            force_reembed,
            50,
            ev.EVO2_WINDOW,
            cache_dir=cache_dir,
            control=control,
        )
    return stacks, genes, fams_arr, fam_order


def _human_patristic_by_family(
    genes: list[str], fam_of: dict[str, str], cache: Path = Path("data/cache/human_patristic")
):
    """Load each family's members and patristic submatrix from cache."""
    import json as _json

    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for fam in sorted(set(fam_of.values())):
        npy, idsj = cache / f"{fam}.npy", cache / f"{fam}.ids.json"
        if not (npy.exists() and idsj.exists()):
            continue
        pos = {g: i for i, g in enumerate(_json.loads(idsj.read_text()))}
        fam_genes = [g for g in genes if fam_of.get(g) == fam and g in pos]
        if len(fam_genes) < 4:
            continue
        idx = [pos[g] for g in fam_genes]
        out[fam] = (fam_genes, np.load(npy)[np.ix_(idx, idx)])
    return out


def _patristic_recovery(geo_df: pd.DataFrame, pat_by_fam: dict) -> dict[str, float]:
    """Per-family Spearman(control within-family geodesic, patristic) — RECOVERY vs the gene tree
    (the matched-human analogue of the cross-kingdom taxonomy recovery)."""
    from scipy.stats import spearmanr as _sp

    rec: dict[str, float] = {}
    for fam, (fg, pat) in pat_by_fam.items():
        present = [g for g in fg if g in geo_df.index]
        if len(present) < 4:
            continue
        idx = [fg.index(g) for g in present]
        patsub = pat[np.ix_(idx, idx)]
        gg = geo_df.loc[present, present].values
        iu = np.triu_indices(len(present), 1)
        c, r = gg[iu], patsub[iu]
        ok = np.isfinite(c) & np.isfinite(r)
        rec[fam] = float(_sp(c[ok], r[ok])[0]) if ok.sum() >= 4 and np.ptp(r[ok]) > 0 else np.nan
    return rec


def _score_human_layer(
    layer: int,
    nat_dir: Path,
    control_stacks: dict[str, np.ndarray],
    genes: list[str],
    fams_arr: np.ndarray,
    fam_order: list[str],
) -> None:
    """Score control-vs-natural preservation at one block for the matched-HUMAN panel and write
    <nat_dir>/controls/.
    """
    nat_geo_df = pd.read_csv(nat_dir / "evo2_human_geodesic_labeled.csv", index_col=0)
    nat_cen_df = pd.read_csv(nat_dir / "evo2_human_centroid_distances.csv", index_col=0)
    family_order = nat_cen_df.index.tolist()
    nat_meta = pd.read_csv(nat_dir / "metadata.csv")
    fam_of = dict(zip(nat_meta["gene"], nat_meta["family"], strict=False))
    pat_by_fam = _human_patristic_by_family(genes, fam_of)

    within_rows, between_rows = [], [{"condition": "natural", "rho_vs_natural_centroid": 1.0}]
    for control, stack in control_stacks.items():
        emb = stack[layer]
        # Same geometry the natural run / write_run_dir builds: angular k-NN geodesic + centroid.
        _, W = find_min_connected_k(emb, k_min=3)
        geo = compute_geodesic(W)
        cen = compute_centroid_geodesic(emb, fams_arr, fam_order)
        ctrl_geo_df = pd.DataFrame(geo, index=genes, columns=genes)
        ctrl_cen_df = pd.DataFrame(cen, index=fam_order, columns=fam_order)

        common = [g for g in nat_geo_df.index if g in ctrl_geo_df.index]
        fams_common = np.array([fam_of.get(g, "NA") for g in common])
        prows, _ = within_preservation_rho(
            nat_geo_df.loc[common, common].values,
            ctrl_geo_df.loc[common, common].values,
            fams_common,
            family_order,
        )
        rec = _patristic_recovery(ctrl_geo_df, pat_by_fam)
        for fam, n, rho in prows:
            within_rows.append(
                {
                    "condition": control,
                    "family": fam,
                    "n_members": n,
                    "rho_geodesic_patristic": rec.get(fam, np.nan),
                    "rho_geodesic_vs_natural": rho,
                }
            )
        brho = between_preservation_rho(
            nat_cen_df.reindex(index=family_order, columns=family_order).values,
            ctrl_cen_df.reindex(index=family_order, columns=family_order).values,
        )
        between_rows.append({"condition": control, "rho_vs_natural_centroid": brho})
        print(
            f"  {control:16} within(mean)={np.mean([r[2] for r in prows]):.3f}  between={brho:.3f}"
        )

    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_control_table(
        out_dir / "control_within_scores.csv",
        pd.DataFrame(within_rows),
        generator="embed_and_score_controls:ortholog_all_layers",
    )
    write_control_table(
        out_dir / "control_between_scores.csv",
        pd.DataFrame(between_rows),
        generator="embed_and_score_controls:ortholog_all_layers",
    )
    print(f"Saved {out_dir}/control_within_scores.csv + control_between_scores.csv")
    figure_human_controls(out_dir)


def run_human(
    controls: list[str],
    layer: int,
    nat_dir: Path,
    device: str = "cpu",
    force_reembed: bool = False,
    input_mode: str = "genomic",
) -> None:
    """Score within- and between-family control preservation for one human-panel block."""
    stacks, genes, fams_arr, fam_order = _human_setup(controls, device, force_reembed, input_mode)
    print(f"\n=== scoring blocks.{layer} ===")
    _score_human_layer(layer, nat_dir, stacks, genes, fams_arr, fam_order)


def run_human_all_layers(
    controls: list[str],
    nat_dir_for,
    device: str = "cpu",
    force_reembed: bool = False,
    input_mode: str = "genomic",
) -> None:
    """Every-block control preservation for the matched-HUMAN panel."""
    stacks, genes, fams_arr, fam_order = _human_setup(controls, device, force_reembed, input_mode)
    n_layers = next(iter(stacks.values())).shape[0]
    for layer in range(n_layers):
        nat_dir = nat_dir_for(layer)
        if nat_dir is None or not nat_dir.exists():
            print(f"\n########## blocks.{layer}: SKIP (no natural run dir) ##########")
            continue
        print(f"\n########## blocks.{layer} -> {nat_dir} ##########")
        _score_human_layer(layer, nat_dir, stacks, genes, fams_arr, fam_order)


def run_between(controls: list[str], device: str, layer: str, nat_dir: Path) -> None:
    """Score control centroid geometry against natural geometry and Pfam JSD."""
    order = pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0).index.tolist()
    nat_cen = (
        pd.read_csv(nat_dir / "family_centroid_distances.csv", index_col=0)
        .reindex(index=order, columns=order)
        .values
    )
    jsd_p = nat_dir / "pfam_jsd_distances.csv"
    jsd = (
        pd.read_csv(jsd_p, index_col=0).reindex(index=order, columns=order).values
        if jsd_p.exists()
        else None
    )
    iu = np.triu_indices(len(order), 1)

    def rho(C, ref):
        if ref is None:
            return float("nan")
        c, r = C[iu], ref[iu]
        ok = np.isfinite(c) & np.isfinite(r)
        return (
            float(spearmanr(c[ok], r[ok])[0])
            if ok.sum() >= 4 and np.ptp(r[ok]) > 0
            else float("nan")
        )

    rows = [
        {
            "condition": "natural",
            "rho_vs_natural_centroid": 1.0,
            "rho_centroid_vs_pfamjsd": rho(nat_cen, jsd),
        }
    ]
    for control in controls:
        control_dir = CONTROL_ROOT / control
        print(f"\n=== {control} ({layer}) between ===")
        E, meta = embed_control(control_dir, device, layer=layer)
        C = compute_centroid_geodesic(E, meta["family"].to_numpy(), order)
        rows.append(
            {
                "condition": control,
                "rho_vs_natural_centroid": rho(C, nat_cen),
                "rho_centroid_vs_pfamjsd": rho(C, jsd),
            }
        )
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_control_table(
        out_dir / "control_between_scores.csv", df, generator="embed_and_score_controls:between"
    )
    print("\nBETWEEN-family control (centroid geodesic vs natural / Pfam-JSD):")
    print(df.round(3).to_string(index=False))
    print(f"Saved {out_dir}/control_between_scores.csv")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--control", required=True, choices=ALL_CONTROLS + ["all"])
    ap.add_argument(
        "--axis",
        choices=["within", "between"],
        default="within",
        help="within: per-family geodesic vs taxonomy + vs-natural preservation. "
        "between: family-centroid geodesic vs natural + Pfam-JSD.",
    )
    ap.add_argument(
        "--panel",
        choices=["ortholog", "human"],
        default="ortholog",
        help="ortholog: cross-kingdom KEGG controls (re-embed from data/ + taxonomy). "
        "human: matched-human panel — read the sibling control run dirs and score "
        "control-vs-natural preservation only (no taxonomy for human paralogs).",
    )
    ap.add_argument(
        "--natural-run",
        default=None,
        help="Natural run dir. Required for --axis between, --panel human, and to add "
        "the within-family preservation column on --panel ortholog.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Evo2 block index to tap (default: the production EMBED_LAYER). e.g. 15.",
    )
    ap.add_argument(
        "--force-reembed",
        action="store_true",
        help="(--panel human) Re-embed controls even if the cached layer stack exists.",
    )
    ap.add_argument(
        "--figure-only",
        action="store_true",
        help="For --panel human, regenerate the control-preservation figure only "
        "from the existing control_*_scores.csv — no embedding or scoring.",
    )
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Score controls at EVERY block, writing each layer's <run>/controls/. "
        "Resolves each layer's natural run dir automatically (ignores --layer/"
        "--natural-run). --panel human: reuse cached stacks (no GPU). "
        "--panel ortholog: build a full layer stack per control once, then score "
        "all layers.",
    )
    ap.add_argument(
        "--input",
        choices=["genomic", "cds"],
        default="genomic",
        help="(--panel human) What Evo2 read: 'genomic' transcript span (default) or "
        "'cds' spliced coding sequence (matches the cross-kingdom ortholog input; "
        "controls embed CDS and score against the -cds natural run dirs).",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = EMBED_LAYER if args.layer is None else f"blocks.{args.layer}"
    print(f"Device: {device} | tap: {layer} | panel: {args.panel} | axis: {args.axis}")
    controls = ALL_CONTROLS if args.control == "all" else [args.control]

    if args.panel == "human":
        # Human controls embed in-process into <nat>/controls/; both axes are pure preservation.
        if args.control == "all":
            controls = HUMAN_CDS_CONTROLS if args.input == "cds" else HUMAN_CONTROLS
        cds_only = {"codon_shuffle", "synonymous_recode"}
        if args.input != "cds" and any(c in cds_only for c in controls):
            sys.exit(
                "codon_shuffle / synonymous_recode require --input cds (they need a reading frame)"
            )
        run_suffix = "-cds" if args.input == "cds" else ""
        if args.all_layers:
            # Sweep layers are nested one-per-block under a single parent run dir
            # (…_evo2-human-panel[-cds]/blocks<L>), so the layer token is the leaf.
            nat_dir_for = lambda L: resolve_latest_run(  # noqa: E731
                f"results/*_evo2-human-panel{run_suffix}/blocks{L}"
            )
            run_human_all_layers(controls, nat_dir_for, device, args.force_reembed, args.input)
            print("\nDone.")
            return
        if not args.natural_run:
            sys.exit("--panel human requires --natural-run")
        if args.figure_only:
            figure_human_controls(Path(args.natural_run) / "controls")
            return
        if args.layer is None:
            sys.exit("--panel human requires --layer (the block the controls are embedded at)")
        run_human(
            controls, args.layer, Path(args.natural_run), device, args.force_reembed, args.input
        )
    elif args.all_layers:  # ortholog panel, every block
        nat_dir_for = lambda L: resolve_latest_run(f"results/*_evo2-gene-families/blocks{L}")  # noqa: E731
        run_ortholog_all_layers(controls, nat_dir_for, device)
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
