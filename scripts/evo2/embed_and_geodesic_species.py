"""
Embed 500 bacterial species with Evo2 and compute geodesic distances.

Lite replication of https://www.goodfire.ai/research/phylogeny-manifold using 500 species.

Pipeline:
  1. Load manifest CSV (data/species/gtdb_500_manifest.csv).
  2. Load FASTA sequences from data/species/sequences_5pct/.
  3. Embed each species with Evo2 7B (mean-pool over sampled windows).
  4. Build k-NN graph with angular-distance edge weights.
  5. Compute all-pairs geodesic distances via Dijkstra.
  6. Download + parse GTDB bac120.tree, compute 500x500 patristic distance matrix.
  7. Spearman rho + Mantel test: geodesic vs patristic.
  8. Within-phylum vs between-phylum geodesic analysis.
  9. Save all outputs with labeled CSVs.

Usage:
    uv run python scripts/embed_and_geodesic_species.py
    uv run python scripts/embed_and_geodesic_species.py --force-reembed
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

# Shared geodesic helpers live in scripts/geodesic_utils.py (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Phylogenetic baseline (GTDB patristic distances) lives in a sibling module.
from calculate_species_baselines import compute_patristic_distances, download_tree  # noqa: E402
from geodesic_utils import (  # noqa: E402
    build_knn_graph,
    compute_geodesic,
    k_sweep_correlations,
    mantel_test,
    upper_triangle,
    within_between_analysis,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "evo2_7b"
# Tap the RESIDUAL STREAM leaving block 24, not the MLP sublayer delta. A block's
# forward returns (u, None) where u is the residual stream, so hooking the block
# module `blocks.24` (vs `blocks.24.mlp.l3`, the pre-add MLP write) yields the
# accumulated representation. The layer/tensor sweep (scripts/evo2/troubleshooting/layer_sweep.py)
# showed this lifts geodesic-vs-phylo Pearson from ~0.36 → ~0.72, ~matching Goodfire;
# `mlp.l3` carries a much weaker cosine geometry. Layer 24 of 32 confirmed (24 > 23).
EMBED_LAYER = "blocks.24"

MANIFEST_PATH = Path("data/species/gtdb_500_manifest.csv")
SEQUENCES_DIR = Path("data/species/sequences_5pct")
TREE_PATH = Path("data/species/bac120.tree")
EMBED_DIR = Path("data/species/embeddings")
EMBED_NPY = EMBED_DIR / "evo2_species_embeddings.npy"
EMBED_META = EMBED_DIR / "metadata.csv"
# Partial checkpoint flushed during embedding (cumulative: rewritten with all species
# done so far, every CHECKPOINT_EVERY) so a subset analysis can run on completed
# species while the full run continues. See preview_subset.py.
EMBED_NPY_PARTIAL = EMBED_DIR / "embeddings_partial.npy"
EMBED_META_PARTIAL = EMBED_DIR / "metadata_partial.csv"
CHECKPOINT_EVERY = 25

# Each stored window is FETCH_BP long; the first (FETCH_BP - EMBED_BP) bp prime the
# autoregressive model (burn-in context) and only the final EMBED_BP positions are
# pooled into the embedding, so the pooled tokens all have sufficient left-context
# (Goodfire phylogeny-manifold method). Must match KEEP_BP in download_species_sequences.py.
EMBED_BP = 2000

# Goodfire's rule for the k-NN graph is "use the smallest K giving a single connected
# component". That K scales with N — they report K=27 for 2400+ species — so hard-coding
# their value over-densifies our 499-species graph, collapsing geodesics toward direct
# angular distances and dragging the geodesic↔patristic correlation from ~0.50 (at the
# true minimum K) down to ~0.28 (at K=27). Instead the pipeline sweeps these small K and
# picks the lowest one that connects the graph at run time (the shared
# geodesic_utils.k_sweep_correlations, also used by
# scripts/evo2/troubleshooting/k_sweep.py). The few K
# past the minimum are kept only as a sensitivity record in k_sweep.csv.
K_SWEEP_VALUES = list(range(2, 11))

# ── Step 2: Read FASTA sequences ───────────────────────────────────────────────


def read_fasta_sequences(fasta_path: Path) -> list[str]:
    """Parse all sequences from a FASTA file. Returns list of sequence strings."""
    sequences = []
    current_seq: list[str] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.upper())
    if current_seq:
        sequences.append("".join(current_seq))
    return sequences


# ── Step 3: Evo2 embedding ─────────────────────────────────────────────────────


def embed_species(fasta_path: Path, model, device: str) -> tuple[np.ndarray, int]:
    """Embed a single species by mean-pooling sampled windows.

    For each sequence in fasta_path:
      - tokenize the full window -> forward pass with return_embeddings=True
      - mean-pool over only the final EMBED_BP token positions -> (4096,) float32 vector
        (the leading positions are burn-in context, not pooled)
    Then average the per-window vectors -> (4096,) species vector.
    """
    sequences = read_fasta_sequences(fasta_path)
    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {fasta_path}")

    window_vecs: list[np.ndarray] = []
    for seq_str in sequences:
        input_ids = (
            torch.tensor(model.tokenizer.tokenize(seq_str), dtype=torch.int).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            _, emb_dict = model(input_ids, return_embeddings=True, layer_names=[EMBED_LAYER])
        # (L, 4096): pool only the final EMBED_BP positions; the rest are burn-in
        # context. Evo2 uses single-nucleotide tokens, so token index ≈ bp index.
        # If a window is shorter than EMBED_BP (e.g. legacy 2000 bp data), [-EMBED_BP:]
        # safely keeps all positions.
        win_emb = emb_dict[EMBED_LAYER][0].float()  # (L, 4096)
        emb = win_emb[-EMBED_BP:].mean(dim=0).cpu().numpy()  # (4096,)
        window_vecs.append(emb)

    species_vec = np.stack(window_vecs, axis=0).mean(axis=0).astype(np.float32)
    return species_vec, len(sequences)


def compute_embeddings(
    manifest: pd.DataFrame,
    sequences_dir: Path,
    device: str,
    checkpoint_npy: Path | None = None,
    checkpoint_meta: Path | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed all species and return (embeddings array, metadata DataFrame).

    If checkpoint paths are given, the partial embeddings + metadata (all species
    completed so far) are flushed to disk every `checkpoint_every` species via an
    atomic write, so a subset analysis can run while the full embedding continues.
    Flushing is cumulative — each write contains every species done so far and never
    drops earlier ones.
    """
    from evo2 import Evo2

    print(f"  Loading {MODEL_NAME} (downloads on first run ~14 GB)...")
    # Evo2 is a thin wrapper, not an nn.Module: it places its StripedHyena on GPU
    # during construction and runs in inference_mode, so there is no .to()/.eval().
    model = Evo2(MODEL_NAME)

    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []

    def flush_checkpoint() -> None:
        if checkpoint_npy is None or checkpoint_meta is None:
            return
        checkpoint_npy.parent.mkdir(parents=True, exist_ok=True)
        npy_tmp = checkpoint_npy.with_name(checkpoint_npy.name + ".tmp.npy")
        np.save(npy_tmp, np.stack(embeddings, axis=0))
        npy_tmp.replace(checkpoint_npy)  # atomic
        meta_tmp = checkpoint_meta.with_name(checkpoint_meta.name + ".tmp")
        pd.DataFrame(meta_rows).to_csv(meta_tmp, index=False)
        meta_tmp.replace(checkpoint_meta)
        tqdm.write(f"    [checkpoint] flushed {len(embeddings)} embeddings → {checkpoint_npy}")

    for i, (_, row) in enumerate(
        tqdm(manifest.iterrows(), total=len(manifest), desc="Embedding species")
    ):
        ncbi_acc = row["ncbi_accession"]
        fasta_path = sequences_dir / f"{ncbi_acc}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        emb, n_windows = embed_species(fasta_path, model, device)
        embeddings.append(emb)
        meta_rows.append(
            {
                "ncbi_accession": ncbi_acc,
                "gtdb_accession": row["gtdb_accession"],
                "species_name": row["species_name"],
                "gtdb_phylum": row["gtdb_phylum"],
                "gtdb_genus": row["gtdb_genus"],
                "n_windows": n_windows,
                "sequences_dir": str(sequences_dir),
            }
        )
        if (i + 1) % checkpoint_every == 0:
            flush_checkpoint()

    emb_array = np.stack(embeddings, axis=0)  # (N, 4096)
    meta_df = pd.DataFrame(meta_rows)
    return emb_array, meta_df


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models-dir",
        default="models/",
        help="Directory for model cache (default: models/)",
    )
    p.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-run embedding even if cached embeddings exist",
    )
    p.add_argument(
        "--sequences-dir",
        default=str(SEQUENCES_DIR),
        help="Directory containing per-species FASTA files (default: data/species/sequences_5pct/)",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    sequences_dir = Path(args.sequences_dir)
    date_str = datetime.date.today().isoformat()
    OUT_DIR = Path("results") / f"{date_str}_evo2-species"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load manifest ──────────────────────────────────────────────────────
    print("\n[1] Loading manifest")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  {len(manifest)} species in {MANIFEST_PATH}")

    # Not every species downloaded successfully — some assemblies are fragmented
    # drafts whose longest contig is shorter than one window. Keep only species
    # that actually have a FASTA on disk so every downstream array aligns.
    has_fasta = manifest["ncbi_accession"].apply(lambda a: (sequences_dir / f"{a}.fasta").exists())
    n_missing = int((~has_fasta).sum())
    if n_missing:
        missing = manifest.loc[~has_fasta, "ncbi_accession"].tolist()
        print(
            f"  Dropping {n_missing} species with no downloaded FASTA: "
            f"{missing[:5]}{'...' if n_missing > 5 else ''}"
        )
        manifest = manifest[has_fasta].reset_index(drop=True)
    print(f"  {len(manifest)} species with sequences will be embedded")
    N = len(manifest)
    ncbi_accessions = manifest["ncbi_accession"].tolist()
    gtdb_accessions = manifest["gtdb_accession"].tolist()
    manifest["species_name"].tolist()
    phyla = np.array(manifest["gtdb_phylum"].tolist())

    phylum_counts = manifest["gtdb_phylum"].value_counts()
    print(f"  {phylum_counts.shape[0]} unique phyla")
    print(f"  Top 5: {phylum_counts.head(5).to_dict()}")

    # ── 2+3. FASTA loading and Evo2 embedding ─────────────────────────────────
    print("\n[2+3] FASTA loading and Evo2 embedding")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    if EMBED_NPY.exists() and EMBED_META.exists() and not args.force_reembed:
        print(f"  Loading cached embeddings from {EMBED_NPY}")
        embeddings = np.load(EMBED_NPY)
        meta_df = pd.read_csv(EMBED_META)
        # The cache lives at a fixed path, not keyed to the manifest. Downstream code
        # matches embedding row i to ncbi_accessions[i] purely by position, so a cache
        # built from a different or reordered manifest would silently mislabel every
        # species. Refuse to reuse it unless the accessions match exactly (same set,
        # same order). The row-count assert below is only a coarse backstop.
        cached_accessions = meta_df["ncbi_accession"].tolist()
        if cached_accessions != ncbi_accessions:
            raise ValueError(
                f"Cached embeddings in {EMBED_NPY} do not match the current manifest "
                f"(cached {len(cached_accessions)} species vs current {len(ncbi_accessions)}, "
                "or same count in a different order/with different accessions). "
                "Re-run with --force-reembed to rebuild the cache."
            )
        expected_sequences_dir = str(sequences_dir)
        if "sequences_dir" not in meta_df.columns:
            raise ValueError(
                f"Cached embeddings in {EMBED_NPY} do not record sequences_dir. "
                "Re-run with --force-reembed to rebuild the cache."
            )
        cached_sequence_dirs = meta_df["sequences_dir"].astype(str).tolist()
        if cached_sequence_dirs != [expected_sequences_dir] * len(cached_accessions):
            raise ValueError(
                f"Cached embeddings in {EMBED_NPY} were built from a different "
                "sequence directory. Re-run with --force-reembed to rebuild the cache."
            )
    else:
        embeddings, meta_df = compute_embeddings(
            manifest,
            sequences_dir,
            device,
            checkpoint_npy=EMBED_NPY_PARTIAL,
            checkpoint_meta=EMBED_META_PARTIAL,
        )
        np.save(EMBED_NPY, embeddings)
        meta_df.to_csv(EMBED_META, index=False)
        print(f"  Saved embeddings : {EMBED_NPY}  shape={embeddings.shape}")
        print(f"  Saved metadata   : {EMBED_META}")

    print(f"  Embeddings shape : {embeddings.shape}")
    assert embeddings.shape[0] == N, f"Expected {N} rows, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 4096, f"Expected dim 4096, got {embeddings.shape[1]}"

    # ── 4. GTDB patristic distances (needed to score the k-sweep below) ─────────
    print("\n[4] GTDB patristic distances")
    if not TREE_PATH.exists():
        download_tree(TREE_PATH)
    else:
        print(f"  Using cached tree: {TREE_PATH}")

    patristic_cache = OUT_DIR / "gtdb_patristic_distances.csv"
    if patristic_cache.exists():
        print(f"  Loading cached patristic matrix from {patristic_cache}")
        pat_df = pd.read_csv(patristic_cache, index_col=0)
        # Rows are matched to gtdb_accessions[i] by position, so reject a cache whose
        # index doesn't match the current manifest (same staleness risk as embeddings).
        if pat_df.index.astype(str).tolist() != gtdb_accessions:
            raise ValueError(
                f"Cached patristic matrix in {patristic_cache} does not match the current "
                "manifest (different/reordered accessions). Delete it to rebuild."
            )
        patristic = pat_df.values.astype(np.float32)
        present_mask = (patristic != 0).any(axis=1)
    else:
        patristic, present_mask = compute_patristic_distances(TREE_PATH, gtdb_accessions)
        df_pat = pd.DataFrame(patristic, index=gtdb_accessions, columns=gtdb_accessions)
        df_pat.to_csv(patristic_cache)
        print(f"  Saved patristic matrix : {patristic_cache}  shape={patristic.shape}")

    print(f"  Patristic range : [{patristic.min():.4f}, {patristic.max():.4f}]")
    n_present = int(present_mask.sum())
    if n_present < N:
        print(f"  {n_present}/{N} species are present in the GTDB tree")

    # ── 5. k-NN graph: sweep k, pick the LOWEST connected k (Goodfire's rule) ────
    print("\n[5] k-NN k-sweep (geodesic↔patristic vs k; choosing lowest connected k)")
    sweep_df = k_sweep_correlations(
        embeddings, patristic, K_SWEEP_VALUES, present_mask=present_mask
    )
    sweep_df.to_csv(OUT_DIR / "k_sweep.csv", index=False)
    connected_ks = sweep_df[sweep_df["connected"]]
    if connected_ks.empty:
        raise ValueError(f"No k in {K_SWEEP_VALUES} connects the graph; widen K_SWEEP_VALUES.")
    k_opt = int(connected_ks["k"].min())
    print(f"  {'k':>3}  {'comp':>4}  {'%finite':>7}  {'Pearson':>8}  {'Spearman':>9}")
    for _, r in sweep_df.iterrows():
        marker = (
            "  <- chosen"
            if int(r["k"]) == k_opt
            else ("" if r["connected"] else "  (disconnected)")
        )
        print(
            f"  {int(r['k']):>3}  {int(r['n_components']):>4}  "
            f"{r['frac_finite_pairs'] * 100:>6.1f}%  "
            f"{r['pearson_geodesic_phylo']:>8.4f}  {r['spearman_geodesic_phylo']:>9.4f}{marker}"
        )
    print(f"  Saved k-sweep table : {OUT_DIR / 'k_sweep.csv'}")
    print(f"  Lowest connected k (non-self neighbors): {k_opt}")

    # ── 6. Geodesic distances at the chosen k ───────────────────────────────────
    print(f"\n[6] All-pairs geodesic distances at k={k_opt}")
    W = build_knn_graph(embeddings, k_opt)
    geodesic = compute_geodesic(W)
    geo_npy = OUT_DIR / "evo2_species_geodesic.npy"
    np.save(geo_npy, geodesic)
    df_geo = pd.DataFrame(geodesic, index=ncbi_accessions, columns=ncbi_accessions)
    df_geo.to_csv(OUT_DIR / "evo2_species_geodesic_labeled.csv")
    print(f"  Saved geodesic matrix : {geo_npy}  shape={geodesic.shape}")
    print(f"  Geodesic range        : [{geodesic.min():.4f}, {geodesic.max():.4f}]")

    # Restrict every downstream comparison to species that are leaves in the tree.
    # Species absent from the tree have all-zero patristic rows and would otherwise
    # swamp the correlation with meaningless zeros.
    if n_present < N:
        print(
            f"  Restricting correlation/within-between analysis to {n_present}/{N} "
            f"species present in the GTDB tree"
        )
    geodesic_an = geodesic[np.ix_(present_mask, present_mask)]
    patristic_an = patristic[np.ix_(present_mask, present_mask)]
    phyla_an = phyla[present_mask]

    # ── 7. Spearman rho + Mantel test ─────────────────────────────────────────
    print("\n[7] Spearman rho and Mantel test (geodesic vs patristic)")
    geo_flat = upper_triangle(geodesic_an)
    pat_flat = upper_triangle(patristic_an)

    rho_sp, pval_sp = spearmanr(geo_flat, pat_flat)
    print(f"  Spearman rho (geodesic vs patristic) = {rho_sp:.4f}  (p = {pval_sp:.4e})")

    print("  Running Mantel test (9999 permutations)...")
    rho_mantel, p_mantel = mantel_test(geodesic_an, patristic_an, n_perms=9999)
    print(f"  Mantel rho = {rho_mantel:.4f}  (p = {p_mantel:.4e})")

    pd.DataFrame(
        [
            {
                "spearman_rho": rho_sp,
                "spearman_p": pval_sp,
                "mantel_rho": rho_mantel,
                "mantel_p": p_mantel,
                "n_pairs": len(geo_flat),
                "n_species_present": n_present,
                "n_species_total": N,
            }
        ]
    ).to_csv(OUT_DIR / "correlation_summary.csv", index=False)

    # ── 8. Within vs. between phylum analysis ─────────────────────────────────
    print("\n[8] Within vs. between phylum distances")
    within, between, ratio, wb_p = within_between_analysis(geodesic_an, phyla_an, n_perms=9999)
    print(f"  Mean within-phylum  : {within.mean():.4f}  (n={len(within)} pairs)")
    print(f"  Mean between-phylum : {between.mean():.4f}  (n={len(between)} pairs)")
    print(f"  Between/within ratio: {ratio:.4f}")
    print(f"  Permutation p-value : {wb_p:.4e}")

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
    ).to_csv(OUT_DIR / "within_between_summary.csv", index=False)

    # Per-phylum summary
    phylum_list = sorted(set(phyla.tolist()))
    phylum_rows = []
    for ph in phylum_list:
        idx = np.where(phyla == ph)[0]
        if len(idx) < 2:
            continue
        sub = geodesic[np.ix_(idx, idx)]
        tri = sub[np.triu_indices(len(idx), k=1)]
        phylum_rows.append(
            {
                "gtdb_phylum": ph,
                "n_species": len(idx),
                "mean_within_geodesic": float(tri.mean()),
                "min_within_geodesic": float(tri.min()),
                "max_within_geodesic": float(tri.max()),
            }
        )
    df_phylum = pd.DataFrame(phylum_rows).sort_values("gtdb_phylum")
    df_phylum.to_csv(OUT_DIR / "per_phylum_geodesic_summary.csv", index=False)
    print("\n  Per-phylum within-geodesic summary:")
    print(df_phylum.to_string(index=False))

    # ── 9. Summary print ──────────────────────────────────────────────────────
    print(f"\n[9] All outputs saved to {OUT_DIR}/")
    print("  Files written:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
