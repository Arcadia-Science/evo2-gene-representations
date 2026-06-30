"""MSA-based composition controls for the GPN-Star gene-family pipeline (Section 3).

The GPN-Star analog of the Evo2 sequence controls. GPN-Star embeds an MSA window — an
(L, N) array of L genomic positions × N aligned species, where column 0 is the human
reference and the model embeds that reference track contextualized by the cross-species
alignment. A "position" is therefore a COLUMN (the conservation pattern at that site), so
the control that mirrors the Evo2 sequence shuffle is a COLUMN shuffle. We also add the
control unique to GPN-Star — ablating the cross-species conservation signal:

  column_shuffle      permute the L axis: preserves the reference base composition AND the
                      multiset of conservation columns, destroys order/motifs/codons.
                      (Mirrors Evo2's composition-preserving shuffle.)
  ref_dinuc_column    permute columns so the reference base sequence is a dinucleotide-
                      preserving (Altschul–Erikson) shuffle — preserves reference 2-mer
                      composition while repositioning columns. (Graded rung. Can't preserve
                      adjacent-COLUMN pairs — columns are ~unique — so the gradient is on
                      the reference base k-mers, carrying each column within its base pool.)
  conservation_ablation  keep the reference (col 0) and all positions, but independently
                      shuffle each non-reference species row along L: preserves each
                      species' base composition, destroys the per-site conservation. Tests
                      whether the within-family signal is the MSA or the bare human sequence.

Ground truth is the FIXED within-family CDS sequence identity (1 − identity); GPN-Star
families are all human paralogs so taxonomy is meaningless. Scores per-family geodesic-vs-
identity ρ for each control and the natural run, writes msa_control_scores.csv + a figure
under <natural-run>/msa_controls/.

Usage:
    uv run python analyses/embed_and_score_msa_controls.py \
        --natural-run results/2026-06-19_gpnstar-vertebrate
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# This script lives in analyses/; its helper modules live alongside it and under scripts/.
HERE = Path(__file__).resolve().parent
_REPO = HERE.parent
sys.path.insert(0, str(HERE))                          # make_control_sequences (sibling)
sys.path.insert(0, str(_REPO / "scripts"))             # geodesic_utils, plot_utils
sys.path.insert(0, str(_REPO / "scripts" / "gpnstar")) # embed_and_geodesic
from embed_and_geodesic import (  # noqa: E402
    MAX_WINDOWS,
    MODELS,
    load_base_model,
    load_or_fetch_coords,
    resolve_model_path,
    sample_span_windows,
)
from gene_families import family_members, family_order as _family_order  # noqa: E402

GENE_FAMILIES = family_members("human")
FAMILY_ORDER = _family_order("human")
from gpn.star.data import GenomeMSA  # noqa: E402
from geodesic_utils import (  # noqa: E402
    compute_geodesic,
    find_min_connected_k,
    within_preservation_rho,
)
from make_control_sequences import _euler_shuffle  # noqa: E402
from plot_utils import grouped_rho_bars, per_family_rho, set_pub_style  # noqa: E402

COORD_CACHE = Path("data/cache/gene_coords.json")
CONTROLS = ["column_shuffle", "ref_dinuc_column", "ref_kmer6_column", "conservation_ablation"]


def _rng(control: str, gene: str) -> random.Random:
    """Deterministic per-(control, gene) RNG (stable across processes)."""
    h = hashlib.md5(f"{control}:{gene}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# ── MSA transforms: operate on an (L, N) int array, column 0 = reference ──────


def t_column_shuffle(msa: np.ndarray, rng: random.Random) -> np.ndarray:
    perm = list(range(msa.shape[0]))
    rng.shuffle(perm)
    return msa[perm]


def t_conservation_ablation(msa: np.ndarray, rng: random.Random) -> np.ndarray:
    out = msa.copy()
    for j in range(1, msa.shape[1]):  # leave reference (col 0) and positions intact
        idx = list(range(msa.shape[0]))
        rng.shuffle(idx)
        out[:, j] = msa[idx, j]
    return out


def _kmer_column_perm(ref: list[int], k: int, rng: random.Random) -> list[int]:
    """Column permutation realising a k-mer-preserving (Altschul–Erikson) shuffle of the
    reference token sequence; each column is carried within its single-token pool, so the
    permuted reference has the EXACT k-mer spectrum of the original (k=2 → dinucleotide)."""
    n = len(ref)
    if n < k + 1:
        return list(range(n))
    toks = [tuple(ref[i : i + k - 1]) for i in range(n - (k - 1) + 1)]  # overlapping (k-1)-mers
    shuf = _euler_shuffle(toks, rng)  # preserves adjacent (k-1)-mer pairs = k-mer counts
    rec = list(shuf[0]) + [t[-1] for t in shuf[1:]]  # reconstruct token seq, k-mers preserved
    pools: dict[int, deque] = {}
    bypos: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(ref):
        bypos[t].append(i)
    for t in bypos:
        rng.shuffle(bypos[t])
        pools[t] = deque(bypos[t])
    return [pools[t].popleft() for t in rec]  # exact: pool counts == token counts in rec


def t_ref_dinuc_column(msa: np.ndarray, rng: random.Random) -> np.ndarray:
    """Column permutation preserving the reference DINUCLEOTIDE (2-mer) spectrum."""
    return msa[_kmer_column_perm([int(x) for x in msa[:, 0]], 2, rng)]


def t_ref_kmer6_column(msa: np.ndarray, rng: random.Random) -> np.ndarray:
    """Column permutation preserving the reference 6-mer spectrum (matches the k-mer
    baseline / the Evo2 kmer6 control / GPN's kNN connectivity k=6)."""
    return msa[_kmer_column_perm([int(x) for x in msa[:, 0]], 6, rng)]


TRANSFORMS = {
    "column_shuffle": t_column_shuffle,
    "ref_dinuc_column": t_ref_dinuc_column,
    "ref_kmer6_column": t_ref_kmer6_column,
    "conservation_ablation": t_conservation_ablation,
}


def embed_gene_transformed(genome_msa, model, coord, windows, device, transform, rng,
                           layer_idx: int) -> np.ndarray:
    """Multi-window embed with an MSA transform applied to each (L, N) window before the model.

    Full-transcript-span multi-window (matching the natural run / layer-selection sweep), tapping
    hidden_state `layer_idx` (output_hidden_states); the transform is applied independently to
    every window's MSA. Mean-pooled over positions then windows.
    """
    chrom, strand = coord["chrom"], coord["strand"]
    accum = None
    for win_start, win_end in windows:
        msa = genome_msa.get_msa(chrom, win_start, win_end, strand=strand, tokenize=True)  # (L, N)
        if transform is not None:
            msa = transform(np.asarray(msa), rng)
        msa_t = torch.from_numpy(np.asarray(msa).astype(np.int64)).to(device)
        input_ids = msa_t[:, :1].unsqueeze(0)
        source_ids = msa_t.unsqueeze(0)
        target_species = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, source_ids=source_ids,
                        target_species=target_species, output_hidden_states=True)
        vec = out.hidden_states[layer_idx][:, :, 0, :].mean(dim=1).squeeze(0).cpu().float().numpy()
        accum = vec if accum is None else accum + vec
    return (accum / len(windows)).astype(np.float32)


def embed_control(control, genome_msa, model, max_window, coords, genes, device, cache_dir,
                  layer_idx: int) -> np.ndarray:
    """Embed every gene under one control (resumable cache, keyed by gene)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy, meta = cache_dir / "embeddings.npy", cache_dir / "genes.txt"
    if npy.exists() and meta.exists():
        prev_genes = meta.read_text().splitlines()
        if prev_genes == genes:
            print(f"  [{control}] resuming from cache")
            return np.load(npy)
    transform = TRANSFORMS[control]
    embs = []
    for g in tqdm(genes, desc=control):
        c = coords[g]
        windows = sample_span_windows(c["tx_start"] - 1, c["tx_end"], max_window, MAX_WINDOWS)
        embs.append(embed_gene_transformed(genome_msa, model, c, windows, device,
                                           transform, _rng(control, g), layer_idx))
    E = np.stack(embs)
    np.save(npy, E)
    meta.write_text("\n".join(genes))
    return E


def family_rho_vs_identity(geodesic, genes, families_arr, seqid_dist) -> list[float]:
    return per_family_rho(geodesic, seqid_dist, families_arr, FAMILY_ORDER)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--natural-run", required=True)
    ap.add_argument("--model", default="vertebrate", choices=list(MODELS))
    ap.add_argument("--controls", nargs="+", default=CONTROLS, choices=CONTROLS)
    ap.add_argument("--layer", type=int, default=None,
                    help="GPN-Star hidden-state index to tap (default: last). Point --natural-run "
                         "at the matching layer-selection run so control-vs-natural is consistent.")
    ap.add_argument("--axis", choices=["within", "between"], default="between",
                    help="between: family-centroid geodesic vs natural (GPN's signal axis). "
                         "within: per-family geodesic vs patristic (mirror control; use --layer 8).")
    args = ap.parse_args()
    run_dir = Path(args.natural_run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_idx = -1 if args.layer is None else args.layer
    print(f"Device: {device} | hidden_state tap: {layer_idx} | full-transcript multiwindow")

    # Gene/family lists from the natural run's metadata, then restrict to genes with a resolvable
    # transcript span (the same full-transcript multi-window scheme as the sweep / the natural run).
    nat_meta = pd.read_csv(run_dir / "metadata.csv")
    genes = nat_meta["gene"].tolist()
    coords = load_or_fetch_coords(genes, COORD_CACHE)
    genes = [g for g in genes if g in coords]
    families_arr = nat_meta.set_index("gene").reindex(genes)["family"].to_numpy()
    print(f"{len(genes)} genes across {len(set(families_arr))} families (transcript-mappable)")

    # ── Mirror control: WITHIN-family axis (GPN's near-null axis) at the within layer ──────
    if args.axis == "within":
        import json as _json
        from scipy.stats import spearmanr  # noqa: F401
        # within-family patristic ground truth (cached, layer-independent) as a block-diag NxN
        cache = Path("data/cache/gpnstar_patristic")
        pos = {g: i for i, g in enumerate(genes)}
        Dpat = np.full((len(genes), len(genes)), np.nan)
        for fam in set(families_arr):
            npy, idsj = cache / f"{fam}.npy", cache / f"{fam}.ids.json"
            if not (npy.exists() and idsj.exists()):
                continue
            ids = _json.loads(idsj.read_text())
            M = np.load(npy)
            keep = [(a, pos[ids[a]]) for a in range(len(ids)) if ids[a] in pos]
            for a, pa in keep:
                for b, pb in keep:
                    Dpat[pa, pb] = M[a, b]

        def per_fam_rho(geo):
            return per_family_rho(geo, Dpat, families_arr, FAMILY_ORDER)[0]

        # Align the natural per-gene geodesic to the `genes` order (== families_arr / Dpat),
        # so both the recovery (vs patristic) and preservation (vs natural) submatrices index
        # the same genes even if some metadata genes were dropped for missing coords.
        nat_lab = pd.read_csv(run_dir / f"gpnstar_{args.model}_geodesic_labeled.csv", index_col=0)
        nat_geo = nat_lab.reindex(index=genes, columns=genes).values

        # Two within-family control views, both written to control_within_scores.csv:
        #   rho_geodesic_patristic  — RECOVERY: does the control still track the phylogeny?
        #   rho_geodesic_vs_natural — PRESERVATION: did the control reconstruct the natural
        #                             within-family geometry (control geodesic vs natural)?
        rows = [{"condition": "natural", "family": f,
                 "rho_geodesic_patristic": r, "rho_geodesic_vs_natural": 1.0}
                for f, r in zip(FAMILY_ORDER, per_fam_rho(nat_geo))]
        ctrl_root = run_dir / "msa_controls"
        cfg = MODELS[args.model]
        genome_msa = GenomeMSA(str(cfg["msa_path"]), n_species=cfg["n_species"])
        model, max_window = load_base_model(resolve_model_path(args.model, Path("models")), device)
        for control in args.controls:
            E = embed_control(control, genome_msa, model, max_window, coords, genes, device,
                              ctrl_root / control, layer_idx)
            _, W = find_min_connected_k(E, k_min=3)
            ctrl_geo = compute_geodesic(W)
            pres = dict((f, rho) for f, _, rho in
                        within_preservation_rho(nat_geo, ctrl_geo, families_arr, FAMILY_ORDER)[0])
            for f, r in zip(FAMILY_ORDER, per_fam_rho(ctrl_geo)):
                rows.append({"condition": control, "family": f, "rho_geodesic_patristic": r,
                             "rho_geodesic_vs_natural": pres.get(f, np.nan)})
        out_dir = run_dir / "controls"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "control_within_scores.csv", index=False)
        summ = df.groupby("condition", sort=False)["rho_geodesic_patristic"].mean()
        # simple mean-per-condition bar (mirror of the Evo2 within control figure)
        set_pub_style(title_size=10, tick_size=8)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        order = ["natural"] + list(args.controls)
        vals = [summ.get(c, np.nan) for c in order]
        ax.bar(range(len(order)), vals, color=["#2A6F4E"] + ["#1F4E79"] * len(args.controls),
               edgecolor="black", linewidth=0.5, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, (v or 0) + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Mean within-family ρ (geodesic vs patristic)")
        ax.set_title(f"GPN-Star WITHIN-family control @ hidden_state.{layer_idx} (its near-null axis)")
        fig.tight_layout()
        fig.savefig(out_dir / "control_within.png", dpi=200)
        fig.savefig(out_dir / "control_within.pdf", dpi=200)
        plt.close(fig)
        print("\nMean within-family geodesic-vs-patristic ρ per condition (RECOVERY):")
        print(summ.round(3).to_string())
        pres_summ = df[df["condition"] != "natural"].groupby(
            "condition", sort=False)["rho_geodesic_vs_natural"].mean()
        print("\nMean within-family geodesic-vs-natural ρ per control (PRESERVATION):")
        print(pres_summ.round(3).to_string())
        print(f"Saved {out_dir}/control_within_scores.csv + control_within.png")
        return

    # The control test compares each control's BETWEEN-family centroid geodesic DIRECTLY to
    # the natural (unchanged) centroid geodesic — Spearman ρ of the two 9×9 upper triangles.
    # ρ→1 means the ablation left the geometry intact (that aspect didn't matter); ρ falling
    # means the ablated aspect was load-bearing for GPN-Star's between-family structure.
    from scipy.stats import spearmanr  # noqa: E402
    from geodesic_utils import compute_centroid_geodesic  # noqa: E402

    iu = np.triu_indices(len(FAMILY_ORDER), k=1)

    def centroid_from_emb(E: np.ndarray) -> np.ndarray:
        return compute_centroid_geodesic(E, families_arr, FAMILY_ORDER)

    # Natural reference centroid geodesic (saved by the natural run; same code path).
    nat_centroid = pd.read_csv(
        run_dir / f"gpnstar_{args.model}_centroid_distances.csv", index_col=0
    ).reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER).values

    # Pfam JSD kept only as a SUPPLEMENTARY column in the CSV (not the figure axis): it
    # corroborates what the control preserves, but the headline metric is vs-natural.
    jsd_p = run_dir / "pfam_jsd_distances.csv"
    jsd = (pd.read_csv(jsd_p, index_col=0).reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER).values
           if jsd_p.exists() else None)

    def rho_vs(a, b):
        return float(spearmanr(a[iu], b[iu])[0])

    # Load the model only if a control's embeddings aren't already cached.
    ctrl_root = run_dir / "msa_controls"
    need_model = any(not (ctrl_root / c / "embeddings.npy").exists() for c in args.controls)
    genome_msa = model = None
    max_window = 0
    if need_model:
        print("Loading GenomeMSA + model...")
        cfg = MODELS[args.model]
        genome_msa = GenomeMSA(str(cfg["msa_path"]), n_species=cfg["n_species"])
        model, max_window = load_base_model(resolve_model_path(args.model, Path("models")), device)

    rows = [{"condition": "natural", "rho_vs_natural_geodesic": 1.0,
             "rho_geodesic_vs_pfamjsd": rho_vs(nat_centroid, jsd) if jsd is not None else np.nan}]
    ctrl_rho = {}
    for control in args.controls:
        E = embed_control(control, genome_msa, model, max_window, coords, genes, device,
                          ctrl_root / control, layer_idx)
        cen = centroid_from_emb(E)
        ctrl_rho[control] = rho_vs(cen, nat_centroid)
        rows.append({"condition": control,
                     "rho_vs_natural_geodesic": ctrl_rho[control],
                     "rho_geodesic_vs_pfamjsd": rho_vs(cen, jsd) if jsd is not None else np.nan})

    scores = pd.DataFrame(rows)
    scores.to_csv(ctrl_root / "msa_control_between_scores.csv", index=False)
    print("\nControl between-family centroid geodesic vs NATURAL (Spearman ρ):")
    print(scores.round(3).to_string(index=False))

    # Figure: x = control, y = ρ(control centroid geodesic vs natural). Dashed line at 1 = natural.
    set_pub_style(title_size=10, tick_size=8)
    palette = {"column_shuffle": "#1F4E79", "ref_dinuc_column": "#6FA8DC",
               "ref_kmer6_column": "#3B7DB5", "conservation_ablation": "#B5651D"}
    labels = {"column_shuffle": "Column\nshuffle", "ref_dinuc_column": "Ref-dinucleotide\ncolumn shuffle",
              "ref_kmer6_column": "Ref-6-mer\ncolumn shuffle", "conservation_ablation": "Conservation\nablation"}
    order = [c for c in ["column_shuffle", "ref_dinuc_column", "ref_kmer6_column", "conservation_ablation"]
             if c in ctrl_rho]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    xs = np.arange(len(order))
    ax.bar(xs, [ctrl_rho[c] for c in order], color=[palette[c] for c in order],
           edgecolor="black", linewidth=0.5, zorder=3)
    for x, c in zip(xs, order):
        ax.text(x, ctrl_rho[c] + (0.02 if ctrl_rho[c] >= 0 else -0.06), f"{ctrl_rho[c]:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="#2A6F4E", linewidth=1.5, linestyle="--", zorder=2,
               label="natural (self = 1.0)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[c] for c in order], fontsize=9)
    ax.set_ylim(-0.3, 1.05)
    ax.set_ylabel("Spearman ρ of control vs natural\nbetween-family centroid geodesic")
    ax.set_title("GPN-Star: which MSA controls preserve the between-family geodesic?\n"
                 "(ρ→1 = geometry intact; falling = that aspect was load-bearing)",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(ctrl_root / "msa_control_between.pdf", dpi=200)
    fig.savefig(ctrl_root / "msa_control_between.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved {ctrl_root}/msa_control_between_scores.csv and msa_control_between.{{pdf,png}}")
    print("Done.")


if __name__ == "__main__":
    main()
