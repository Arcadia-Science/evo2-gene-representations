"""Embed cross-kingdom gene-family CDS with Evo2 and test two structures at once.

Companion to embed_and_geodesic_species.py (which embeds whole-genome windows). Here
each manifest row is ONE coding sequence pulled from KEGG across the tree of life
(scripts/gene_families.py build-ortholog), and we ask the two questions the design
doc (gene_family_design.md) frames:

  Axis A — between-family.  Do globins, P450s, opsins, … form separable clusters on
    the Evo2 manifold? Family-centroid geodesic matrix + within/between-family
    distance ratio with a label-permutation p-value. (The GPN-Star Figure-2 analog,
    but cross-kingdom.)

  Axis B — within-family phylogeny.  *Inside* one family, does the embedding geodesic
    recapitulate sequence divergence and host taxonomy (bacterial → plant →
    vertebrate globin)? Per-family Spearman ρ of the within-family geodesic submatrix
    vs k-mer sequence divergence and vs taxonomic-rank distance. This is the new
    contribution — no GPN-Star configuration can pose it.

Embedding strategy is identical to the species pipeline: Evo2 7B, residual-stream tap
blocks.24, autoregressive burn-in (pool only the final EMBED_BP positions). A standalone
CDS (median ~0.5–2 kb) is one window, so we pool the whole coding sequence.

Pipeline:
  1. Load manifest + per-family FASTAs.
  2. Embed each CDS (Evo2 blocks.24) -> one 4096-d vector.
  3. k-NN graph (angular distance), lowest connected k, all-pairs geodesic.
  4. Axis A: family-centroid distances + within/between-family ratio + Mantel.
  5. Axis B: per-family geodesic vs k-mer divergence and vs taxonomic distance.
  6. Save labeled CSVs + summaries.

Usage:
    uv run python scripts/evo2/embed_and_geodesic_ortholog.py
    uv run python scripts/evo2/embed_and_geodesic_ortholog.py --force-reembed
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # evo2/ siblings (evo2_embedding)
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402  (shared baseline kernel)
from taxonomic_distance import taxonomic_distance_matrix  # noqa: E402
from evo2_embedding import EMBED_LAYER, embed_one, load_model  # noqa: E402
from geodesic_utils import (  # noqa: E402
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
    upper_triangle,
    within_between_analysis,
)

# ── Constants ──────────────────────────────────────────────────────────────────
# MODEL_NAME / EMBED_LAYER + the Evo2 embedding mechanics (second-half pooling, forward pass) live
# in the shared evo2_embedding.py — also used by the paralog embedder and the layer sweep.

DATA_DIR = Path("data/evo2_gene_families")
MANIFEST_PATH = DATA_DIR / "manifest.csv"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_NPY = EMBED_DIR / "evo2_gene_family_embeddings.npy"
EMBED_META = EMBED_DIR / "metadata.csv"
EMBED_NPY_PARTIAL = EMBED_DIR / "embeddings_partial.npy"
EMBED_META_PARTIAL = EMBED_DIR / "metadata_partial.csv"
CHECKPOINT_EVERY = 100

KMER_K = 6

# ── FASTA loading ────────────────────────────────────────────────────────────


def load_family_sequences(families: list[str]) -> dict[str, str]:
    """Map org_gene -> CDS from each family's FASTA (header: org_gene|family|ko|label)."""
    seqs: dict[str, str] = {}
    for family in sorted(set(families)):
        fasta = DATA_DIR / f"{family}.fasta"
        if not fasta.exists():
            raise FileNotFoundError(f"Family FASTA not found: {fasta}")
        cur_id, cur_seq = None, []
        with open(fasta) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_id:
                        seqs[cur_id] = "".join(cur_seq)
                    cur_id = line[1:].split("|")[0]  # org_gene
                    cur_seq = []
                else:
                    cur_seq.append(line.upper())
        if cur_id:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


# ── Evo2 embedding (mechanics shared via evo2_embedding.embed_one / load_model) ──────


def compute_embeddings(
    manifest: pd.DataFrame,
    seqs: dict[str, str],
    device: str,
    cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed every manifest CDS in manifest order, reusing `cache` per org_gene.

    `cache` maps org_gene -> a previously-computed embedding (same blocks.24/EMBED_BP
    config). Rows already in the cache are reused; only genuinely new org_genes hit
    Evo2, so adding families re-embeds just the new CDS. The Evo2 model is loaded
    lazily, so a 100%-cache-hit run never pays the ~14 GB load. Checkpoints cumulatively.
    """
    cache = cache or {}
    org_genes = manifest["org_gene"].tolist()
    n_reuse = sum(1 for g in org_genes if g in cache)
    print(f"  Reusing {n_reuse} cached embeddings; embedding {len(org_genes) - n_reuse} new CDS")

    model = None  # lazy: only load Evo2 if something actually needs embedding
    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []

    def flush() -> None:
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = EMBED_NPY_PARTIAL.with_suffix(".tmp.npy")
        np.save(tmp, np.stack(embeddings, axis=0))
        tmp.replace(EMBED_NPY_PARTIAL)
        mtmp = EMBED_META_PARTIAL.with_suffix(".tmp")
        pd.DataFrame(meta_rows).to_csv(mtmp, index=False)
        mtmp.replace(EMBED_META_PARTIAL)
        tqdm.write(f"    [checkpoint] flushed {len(embeddings)} embeddings")

    for i, (_, row) in enumerate(
        tqdm(manifest.iterrows(), total=len(manifest), desc="Embedding CDS")
    ):
        og = row["org_gene"]
        if og in cache:
            emb = cache[og]
        else:
            if model is None:
                model = load_model()
            emb = embed_one(seqs[og], model, device)
        embeddings.append(emb)
        meta_rows.append(row.to_dict())
        if (i + 1) % CHECKPOINT_EVERY == 0:
            flush()

    return np.stack(embeddings, axis=0), pd.DataFrame(meta_rows)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument(
        "--from-sweep-layer",
        type=int,
        default=None,
        help="Skip embedding; pull this block's embeddings from the dense-sweep cache "
        "(data/cache/evo2_layer_sweep). Use the layer-selection recommended layer to "
        "regenerate the gene figures. Output dir is suffixed -blocks<N>.",
    )
    p.add_argument("--kmer-k", type=int, default=KMER_K)
    p.add_argument(
        "--n-perms",
        type=int,
        default=999,
        help="Permutations for the Axis-A within/between separability test "
        "(999 gives p-resolution 1e-3; bump to 9999 for a final run).",
    )
    p.add_argument(
        "--distances-from",
        default=None,
        help="Reuse the Axis-B distance matrices (kmer_distance.npy, taxonomic_distance.npy) "
        "from this donor run dir instead of recomputing the O(N²) k-mer/taxonomy distances. "
        "They are sequence/taxonomy-derived and layer-independent — for an all-layer sweep. "
        "The big matrices are NOT re-saved into this run dir (they live in the donor).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def load_layer_from_sweep(manifest: pd.DataFrame, layer_idx: int) -> np.ndarray:
    """Pull one layer's embeddings from the dense-sweep cache, aligned to manifest order.

    The sweep (scripts/evo2/layer_sweep.py) used the SAME blocks-residual / EMBED_BP pooling
    as this pipeline, so layer `EMBED_LAYER`'s slice is byte-identical to a fresh embed — only
    the layer index differs. Lets us regenerate the gene figures at the layer-selection
    recommended layer with no re-embedding.
    """
    cache = Path("data/cache/evo2_layer_sweep")
    stack = np.load(cache / "layer_stack.npy")  # (n_blocks, N, H)
    sweep_meta = pd.read_csv(cache / "metadata.csv")
    if not (0 <= layer_idx < stack.shape[0]):
        raise ValueError(f"layer_idx {layer_idx} out of range [0,{stack.shape[0]})")
    pos = {g: i for i, g in enumerate(sweep_meta["org_gene"].tolist())}
    missing = [g for g in manifest["org_gene"] if g not in pos]
    if missing:
        raise ValueError(f"{len(missing)} manifest genes absent from sweep cache: {missing[:5]}")
    idx = [pos[g] for g in manifest["org_gene"]]
    return stack[layer_idx][idx].astype(np.float32)


def main() -> None:
    args = parse_args()
    date_str = datetime.date.today().isoformat()
    layer_tag = f"-blocks{args.from_sweep_layer}" if args.from_sweep_layer is not None else ""
    OUT_DIR = Path("results") / f"{date_str}_evo2-gene-families{layer_tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Manifest + sequences ────────────────────────────────────────────────
    print("\n[1] Loading manifest and sequences")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  {len(manifest)} CDS across {manifest['family'].nunique()} families")
    print(f"  Per family: {manifest['family'].value_counts().to_dict()}")
    seqs = load_family_sequences(manifest["family"].tolist())
    missing = [g for g in manifest["org_gene"] if g not in seqs]
    if missing:
        raise ValueError(f"{len(missing)} manifest org_genes missing from FASTA: {missing[:5]}")

    org_genes = manifest["org_gene"].tolist()
    families = np.array(manifest["family"].tolist())
    family_order = sorted(manifest["family"].unique())
    N = len(manifest)

    # ── 2. Evo2 embedding ──────────────────────────────────────────────────────
    if args.from_sweep_layer is not None:
        print(f"\n[2] Using sweep-cache layer blocks.{args.from_sweep_layer} (no re-embed)")
        embeddings = load_layer_from_sweep(manifest, args.from_sweep_layer)
        meta_df = manifest.copy()
        print(f"  Loaded embeddings from cache: shape={embeddings.shape}")
    else:
        print(f"\n[2] Evo2 embedding ({EMBED_LAYER})")
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        # Per-gene cache: load any prior embeddings as {org_gene: vector} and reuse them,
        # so adding families only embeds the new CDS. --force-reembed empties the cache
        # (use it if the embedding config — layer/EMBED_BP — or a gene's CDS changed).
        cache: dict[str, np.ndarray] = {}
        if EMBED_NPY.exists() and EMBED_META.exists() and not args.force_reembed:
            prev = np.load(EMBED_NPY)
            prev_meta = pd.read_csv(EMBED_META)
            if len(prev) == len(prev_meta):
                cache = {g: prev[i] for i, g in enumerate(prev_meta["org_gene"].tolist())}
                print(f"  Loaded {len(cache)} cached per-gene embeddings from {EMBED_NPY}")
        embeddings, meta_df = compute_embeddings(manifest, seqs, device, cache)
        np.save(EMBED_NPY, embeddings)
        meta_df.to_csv(EMBED_META, index=False)
        print(f"  Saved embeddings : {EMBED_NPY}  shape={embeddings.shape}")

    assert embeddings.shape == (N, 4096), f"Expected ({N}, 4096), got {embeddings.shape}"

    # ── 3. k-NN graph + geodesic ───────────────────────────────────────────────
    print("\n[3] k-NN graph (lowest connected k) + all-pairs geodesic")
    k_opt, W = find_min_connected_k(embeddings, k_min=3)
    geodesic = compute_geodesic(W)
    np.save(OUT_DIR / "evo2_gene_family_geodesic.npy", geodesic)
    pd.DataFrame(geodesic, index=org_genes, columns=org_genes).to_csv(
        OUT_DIR / "evo2_gene_family_geodesic_labeled.csv"
    )
    print(f"  k={k_opt}, geodesic range [{geodesic.min():.4f}, {geodesic.max():.4f}]")

    # ── 4. Axis A: between-family separability ─────────────────────────────────
    # Between-family distance is the geodesic BETWEEN family centroids (one node per
    # family), so it does not depend on per-family member counts. The gene-level
    # `geodesic` above is still used for the within-family analysis below.
    print("\n[4] Axis A — between-family separability")
    F = len(family_order)
    centroid = compute_centroid_geodesic(embeddings, families, family_order).astype(np.float32)
    pd.DataFrame(centroid, index=family_order, columns=family_order).to_csv(
        OUT_DIR / "family_centroid_distances.csv"
    )
    print("  Family mean geodesic (within diagonal / between off-diagonal):")
    print(pd.DataFrame(centroid, index=family_order, columns=family_order).round(3).to_string())

    within, between, ratio, wb_p = within_between_analysis(
        geodesic, families, n_perms=args.n_perms
    )
    print(f"\n  Mean within-family  : {within.mean():.4f}  (n={len(within)} pairs)")
    print(f"  Mean between-family : {between.mean():.4f}  (n={len(between)} pairs)")
    print(f"  Between/within ratio: {ratio:.4f}  (permutation p = {wb_p:.4e})")
    pd.DataFrame(
        [
            {
                "mean_within": float(within.mean()),
                "mean_between": float(between.mean()),
                "ratio": ratio,
                "p_value": wb_p,
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
            }
        ]
    ).to_csv(OUT_DIR / "axisA_within_between_summary.csv", index=False)

    # ── 5. Axis B: within-family phylogeny ─────────────────────────────────────
    print("\n[5] Axis B — within-family geodesic vs sequence divergence & taxonomy")
    if args.distances_from:
        donor = Path(args.distances_from)
        kmer_dist = np.load(donor / "kmer_distance.npy")
        tax_dist = np.load(donor / "taxonomic_distance.npy")
        # Symlink (not copy) the layer-independent matrices so the within-family heatmaps
        # still find them, without duplicating ~107 MB each per layer.
        for name in ("kmer_distance.npy", "taxonomic_distance.npy"):
            link = OUT_DIR / name
            link.unlink(missing_ok=True)
            link.symlink_to((donor / name).resolve())
        print(f"  Reusing k-mer + taxonomic matrices from {donor} (symlinked, not copied)")
    else:
        seq_list = [seqs[g] for g in org_genes]
        kmer_dist = kmer_distance_matrix(seq_list, k=args.kmer_k)
        tax_dist = taxonomic_distance_matrix(manifest)
        np.save(OUT_DIR / "kmer_distance.npy", kmer_dist)
        np.save(OUT_DIR / "taxonomic_distance.npy", tax_dist)

    rows = []
    print(f"  {'family':<22} {'n':>4} {'pairs':>6} {'ρ_kmer':>8} {'p':>9}   {'ρ_tax':>7} {'p':>9}")
    for fam in family_order:
        idx = np.where(families == fam)[0]
        n = len(idx)
        if n < 4:
            print(f"  {fam:<22} {n:>4}  (too few for within-family correlation)")
            continue
        geo_f = upper_triangle(geodesic[np.ix_(idx, idx)])
        kmer_f = upper_triangle(kmer_dist[np.ix_(idx, idx)])
        tax_f = upper_triangle(tax_dist[np.ix_(idx, idx)])
        rho_k, p_k = spearmanr(geo_f, kmer_f)
        # taxonomic distance is highly tied (4 levels); guard the degenerate case.
        if np.ptp(tax_f) == 0:
            rho_t, p_t = float("nan"), float("nan")
        else:
            rho_t, p_t = spearmanr(geo_f, tax_f)
        rows.append(
            {
                "family": fam,
                "n_members": n,
                "n_pairs": len(geo_f),
                "spearman_geodesic_kmer": rho_k,
                "p_kmer": p_k,
                "spearman_geodesic_taxonomy": rho_t,
                "p_taxonomy": p_t,
            }
        )
        print(
            f"  {fam:<22} {n:>4} {len(geo_f):>6} {rho_k:>+8.3f} {p_k:>9.2e}   "
            f"{rho_t:>+7.3f} {p_t:>9.2e}"
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "axisB_within_family_correlations.csv", index=False)

    # Pooled within-family Mantel (block-diagonal): does divergence track geodesic
    # across all families at once, ignoring between-family pairs?
    within_mask = families[:, None] == families[None, :]
    iu = np.triu_indices(N, k=1)
    wpairs = within_mask[iu]
    geo_all, kmer_all = upper_triangle(geodesic), upper_triangle(kmer_dist)
    rho_pool, p_pool = spearmanr(geo_all[wpairs], kmer_all[wpairs])
    print(
        f"\n  Pooled within-family ρ (geodesic vs k-mer): {rho_pool:+.4f} "
        f"(p = {p_pool:.4e}, n = {int(wpairs.sum())} pairs)"
    )

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print(f"\n[6] Outputs written to {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
