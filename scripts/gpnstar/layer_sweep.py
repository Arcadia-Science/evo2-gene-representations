"""Layer-depth sweep + cosine-ceiling diagnostic for the GPN-Star gene pipeline,
with EXON-AWARE MULTI-WINDOW embedding and a WITHIN-family evaluation metric.

Two learnings drove this:
  1. (from the Evo2 species sweep) The best layer is not the default tap; for a masked-LM
     the last layers specialise toward the reconstruction head and shed family structure.
     GPN-Star is a bidirectional RoFormer MLM, so `output_hidden_states=True` returns all
     (num_hidden_layers + 1) states in one forward pass — sweeping layers is ~free.
  2. (diagnosed here) The old pipeline embedded ONE window at the genomic CDS midpoint.
     But 89% of these genes span > one window (median 5×, up to 144 kb), so that lone window
     rarely covered the gene. Fix: tile up to MAX_WINDOWS windows over the FULL genomic
     transcript span [tx_start, tx_end] (5'UTR + exons + introns + 3'UTR — §1's shared locus,
     matching Evo2's genomic gene-body input) and mean-pool. (Bidirectional model → no burn-in;
     pool all positions in each window, then across windows.) Shares the windowing helpers with
     the production embedder (embed_and_geodesic), so the two stay byte-for-byte aligned.

Evaluation is WITHIN-family (genes in the same Pfam family) using the gene-level ground
truths (Compara paralog %identity, CDS identity). Pfam JSD is inherently between-family
(one profile per family) so it is kept only as a reference column.

Usage:
    uv run python scripts/gpnstar/layer_sweep.py
    uv run python scripts/gpnstar/layer_sweep.py --force-reembed   # default --max-windows fully covers each gene
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gpn.star.data import GenomeMSA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embed_and_geodesic import (  # noqa: E402
    MAX_WINDOWS,
    MODELS,
    embed_gene_multiwindow,
    load_base_model,
    load_or_fetch_coords,
    resolve_model_path,
    sample_span_windows,
)
from gene_families import family_members, family_order as _family_order  # noqa: E402

GENE_FAMILIES = family_members("human")
FAMILY_ORDER = _family_order("human")
from geodesic_utils import (  # noqa: E402
    build_knn_graph,
    compute_centroid_geodesic,
    compute_geodesic,
    upper_triangle,
)

COORD_CACHE = Path("data/cache/gene_coords.json")  # shared with embed_and_geodesic
EMB_CACHE_DIR = Path("data/cache/gpnstar_layer_sweep")
DEFAULT_MAX_WINDOWS = MAX_WINDOWS


# The multi-window embedding helper (embed_gene_multiwindow) now lives in embed_and_geodesic.py
# (imported above) so the sweep and the production embedder share one definition byte-for-byte.


# ── distance / correlation helpers ───────────────────────────────────────────────


def angular_distance_matrix(emb: np.ndarray) -> np.ndarray:
    normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = normed @ normed.T
    return np.arccos(np.clip(sim, -1.0, 1.0))


def lowest_connected_k(emb: np.ndarray, k_min: int = 3):
    N = len(emb)
    for k in range(k_min, N):
        W = build_knn_graph(emb, k)
        n_comp, _ = connected_components(csr_matrix(W), directed=False)
        if n_comp == 1:
            return k, W
    raise ValueError("graph never connects")


def corr(dist_emb, dist_ref, pair_mask=None):
    """Spearman, n_pairs over the strict upper triangle; optionally restrict to pair_mask."""
    a, b = upper_triangle(dist_emb), upper_triangle(dist_ref)
    valid = np.isfinite(a) & np.isfinite(b)
    if pair_mask is not None:
        valid &= pair_mask
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return float("nan"), int(len(a))
    return float(spearmanr(a, b)[0]), int(len(a))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="vertebrate", choices=list(MODELS))
    p.add_argument("--models-dir", default="models")
    p.add_argument(
        "--truth-dir",
        default=None,
        help="Results folder with the geodesic/seq-identity baselines "
        "(default: newest results/*_gpnstar-<model>)",
    )
    p.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--force-reembed", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.truth_dir:
        truth = Path(args.truth_dir)
    else:
        matches = sorted(Path("results").glob(f"*_gpnstar-{args.model}"))
        if not matches:
            sys.exit(
                f"ERROR: no results/*_gpnstar-{args.model} run found. Run "
                f"embed_and_geodesic.py first, or pass --truth-dir."
            )
        truth = matches[-1]
    print(f"truth-dir: {truth}")

    gene_list, family_labels = [], []
    for fam in FAMILY_ORDER:
        for g in GENE_FAMILIES[fam]:
            gene_list.append(g)
            family_labels.append(fam)
    print(
        f"Genes: {len(gene_list)} across {len(FAMILY_ORDER)} families | "
        f"model: {args.model} | device: {device}"
    )

    coords = load_or_fetch_coords(gene_list, COORD_CACHE)

    # Drop genes Ensembl couldn't map, keeping gene_list/family_labels aligned.
    kept = [(g, f) for g, f in zip(gene_list, family_labels, strict=False) if g in coords]
    gene_list = [g for g, _ in kept]
    family_labels = [f for _, f in kept]
    families_arr = np.array(family_labels)
    N = len(gene_list)
    print(f"Genes with transcript span: {N} across {len(set(family_labels))} families")

    # Drop genes whose chromosome is absent from the MSA (e.g. olfactory receptors on
    # novel-patch contigs, which are Evo2-only per the manifest); get_msa would KeyError on
    # them. Load the MSA once here so the kept gene set is fixed before the cache key is built.
    genome_msa = GenomeMSA(
        str(MODELS[args.model]["msa_path"]), n_species=MODELS[args.model]["n_species"]
    )

    def _chrom_in_msa(chrom: str) -> bool:
        try:
            genome_msa.data[chrom]
            return True
        except KeyError:
            return False

    in_msa = [_chrom_in_msa(coords[g]["chrom"]) for g in gene_list]
    dropped_msa = [g for g, ok in zip(gene_list, in_msa, strict=False) if not ok]
    if dropped_msa:
        print(f"  {len(dropped_msa)} gene(s) on MSA-absent contigs dropped (Evo2-only): {dropped_msa}")
        gene_list = [g for g, ok in zip(gene_list, in_msa, strict=False) if ok]
        family_labels = [f for f, ok in zip(family_labels, in_msa, strict=False) if ok]
        families_arr = np.array(family_labels)
        N = len(gene_list)
        print(f"Genes embeddable by GPN-Star (chrom in MSA): {N}")

    # ── embed all layers over full-transcript-span multi-windows (cached) ──
    config = {
        "model": args.model,
        "genes": gene_list,
        "max_windows": args.max_windows,
        "scheme": "transcript-multiwindow",
    }
    cfg_path = EMB_CACHE_DIR / "config.json"
    stack_path = EMB_CACHE_DIR / f"layer_stack_{args.model}.npy"  # (n_states, N, H)
    if (
        not args.force_reembed
        and cfg_path.exists()
        and stack_path.exists()
        and json.loads(cfg_path.read_text()) == config
    ):
        print("  Reusing cached layer-stack embeddings.")
        layer_stack = np.load(stack_path)
    else:
        model_path = resolve_model_path(args.model, Path(args.models_dir))
        print(f"  Loading model from {model_path} (MSA already open)")
        model, max_window = load_base_model(model_path, device)
        n_states = model.config.num_hidden_layers + 1

        n_win_used = []
        per_gene = []
        for g in tqdm(
            gene_list,
            desc=f"Embedding genes (≤{args.max_windows} transcript windows × {n_states} layers)",
        ):
            c = coords[g]
            windows = sample_span_windows(
                c["tx_start"] - 1, c["tx_end"], max_window, args.max_windows
            )
            n_win_used.append(len(windows))
            per_gene.append(
                embed_gene_multiwindow(genome_msa, model, c, windows, device, n_states)
            )
        layer_stack = np.stack(per_gene, axis=1)  # (n_states, N, H)
        EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(stack_path, layer_stack)
        cfg_path.write_text(json.dumps(config, indent=2))
        print(
            f"  Windows/gene: min={min(n_win_used)} "
            f"median={int(np.median(n_win_used))} max={max(n_win_used)}"
        )
        print(f"  Saved {layer_stack.shape} layer-stack to {stack_path}")

    n_states = layer_stack.shape[0]
    print(f"  {n_states} hidden states (0=embeddings … {n_states - 1}=last_hidden_state)\n")

    # ── ground truth, aligned to gene_list / FAMILY_ORDER ──
    seqid = pd.read_csv(truth / "sequence_identity_genes.csv", index_col=0).reindex(
        index=gene_list, columns=gene_list
    )
    seqid_dist = 1.0 - seqid.values.astype(float)
    para = pd.read_csv(truth / "ensembl_paralog_identity_genes.csv", index_col=0).reindex(
        index=gene_list, columns=gene_list
    )
    para_dist = 100.0 - para.values.astype(float)  # NaN preserved → masked in corr()
    jsd = (
        pd.read_csv(truth / "pfam_jsd_distances.csv", index_col=0)
        .reindex(index=FAMILY_ORDER, columns=FAMILY_ORDER)
        .values.astype(float)
    )

    # within-family pair mask (the metric of interest)
    gi, gj = np.triu_indices(N, k=1)
    within = families_arr[gi] == families_arr[gj]
    n_within_para = int((np.isfinite(upper_triangle(para_dist)) & within).sum())
    print(
        f"  Within-family pairs: {int(within.sum())} (seq-id); {n_within_para} have paralog data. "
        f"[between-family Pfam JSD kept as reference]\n"
    )

    # ── per-layer correlations: WITHIN-family primary ──
    print(
        f"  {'layer':>5}  {'k':>3} |  {'WITHIN: ANG·seqid':>17}  {'ANG·para':>9}  "
        f"{'GEO·seqid':>9}  {'GEO·para':>9} | {'ref GEO·JSD':>11}"
    )
    rows = []
    for li in range(n_states):
        emb = layer_stack[li]
        ang = angular_distance_matrix(emb)
        k_opt, W = lowest_connected_k(emb, k_min=args.k_min)
        geo = compute_geodesic(W)

        w_ang_seq, _ = corr(ang, seqid_dist, within)
        w_ang_par, _ = corr(ang, para_dist, within)
        w_geo_seq, _ = corr(geo, seqid_dist, within)
        w_geo_par, _ = corr(geo, para_dist, within)
        geo_centroid = compute_centroid_geodesic(emb, families_arr, FAMILY_ORDER)
        geo_jsd, _ = corr(geo_centroid, jsd)

        print(
            f"  {li:>5}  {k_opt:>3} |  {w_ang_seq:>+17.3f}  {w_ang_par:>+9.3f}  "
            f"{w_geo_seq:>+9.3f}  {w_geo_par:>+9.3f} | {geo_jsd:>+11.3f}"
        )
        rows.append(
            {
                "layer": li,
                "k": k_opt,
                "within_ang_seqid_spearman": w_ang_seq,
                "within_ang_paralog_spearman": w_ang_par,
                "within_geo_seqid_spearman": w_geo_seq,
                "within_geo_paralog_spearman": w_geo_par,
                "ref_between_geo_jsd_spearman": geo_jsd,
            }
        )

    df = pd.DataFrame(rows)
    date_str = datetime.date.today().isoformat()
    out_dir = Path("results") / f"{date_str}_gpnstar-layer-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"layer_sweep_{args.model}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    cur = df[df["layer"] == n_states - 1].iloc[0]
    best = df.loc[df["within_ang_seqid_spearman"].idxmax()]
    best_par = df.loc[df["within_ang_paralog_spearman"].idxmax()]
    print(
        f"\nCurrent tap (last layer {int(cur['layer'])}): "
        f"within ANG·seqid ρ={cur['within_ang_seqid_spearman']:+.3f}, "
        f"ANG·para ρ={cur['within_ang_paralog_spearman']:+.3f}"
    )
    print(
        f"Best within ANG·seqid: layer {int(best['layer'])} "
        f"ρ={best['within_ang_seqid_spearman']:+.3f}"
    )
    print(
        f"Best within ANG·para:  layer {int(best_par['layer'])} "
        f"ρ={best_par['within_ang_paralog_spearman']:+.3f}"
    )


if __name__ == "__main__":
    main()
