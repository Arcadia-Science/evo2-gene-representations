"""
Embed 500 bacterial species with Evo2 and compute geodesic distances.

Lite replication of https://www.goodfire.ai/research/phylogeny-manifold using 500 species.

Pipeline:
  1. Load manifest CSV (data/species/gtdb_500_manifest.csv).
  2. Load FASTA sequences from data/species/sequences/.
  3. Embed each species with Evo2 7B (mean-pool over 10 windows).
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
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "evo2_7b"
EMBED_LAYER = "blocks.24.mlp.l3"

MANIFEST_PATH = Path("data/species/gtdb_500_manifest.csv")
SEQUENCES_DIR = Path("data/species/sequences")
TREE_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120.tree"
TREE_PATH = Path("data/species/bac120.tree")
EMBED_DIR = Path("data/species/embeddings")
EMBED_NPY = EMBED_DIR / "evo2_species_embeddings.npy"
EMBED_META = EMBED_DIR / "metadata.csv"

N_WINDOWS = 10  # sequences per species

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


def embed_species(fasta_path: Path, model, device: str) -> np.ndarray:
    """Embed a single species by mean-pooling 10 windows.

    For each of the 10 sequences in fasta_path:
      - tokenize -> forward pass with return_embeddings=True
      - mean-pool over sequence length -> (4096,) float32 vector
    Then average the 10 per-window vectors -> (4096,) species vector.
    """
    sequences = read_fasta_sequences(fasta_path)
    if len(sequences) == 0:
        raise ValueError(f"No sequences found in {fasta_path}")

    window_vecs: list[np.ndarray] = []
    for seq_str in sequences:
        input_ids = (
            torch.tensor(model.tokenizer.tokenize(seq_str), dtype=torch.int)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            _, emb_dict = model(
                input_ids, return_embeddings=True, layer_names=[EMBED_LAYER]
            )
        emb = emb_dict[EMBED_LAYER][0].float().mean(dim=0).cpu().numpy()  # (4096,)
        window_vecs.append(emb)

    return np.stack(window_vecs, axis=0).mean(axis=0).astype(np.float32)  # (4096,)


def compute_embeddings(
    manifest: pd.DataFrame,
    sequences_dir: Path,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed all species and return (embeddings array, metadata DataFrame)."""
    from evo2 import Evo2

    print(f"  Loading {MODEL_NAME} (downloads on first run ~14 GB)...")
    # Evo2 is a thin wrapper, not an nn.Module: it places its StripedHyena on GPU
    # during construction and runs in inference_mode, so there is no .to()/.eval().
    model = Evo2(MODEL_NAME)

    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Embedding species"):
        ncbi_acc = row["ncbi_accession"]
        fasta_path = sequences_dir / f"{ncbi_acc}.fasta"
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        emb = embed_species(fasta_path, model, device)
        embeddings.append(emb)
        meta_rows.append(
            {
                "ncbi_accession": ncbi_acc,
                "species_name": row["species_name"],
                "gtdb_phylum": row["gtdb_phylum"],
                "gtdb_genus": row["gtdb_genus"],
            }
        )

    emb_array = np.stack(embeddings, axis=0)  # (500, 4096)
    meta_df = pd.DataFrame(meta_rows)
    return emb_array, meta_df


# ── Step 4: k-NN graph with angular distances ──────────────────────────────────


def cosine_to_angular(cosine_dist: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(1.0 - cosine_dist, -1.0, 1.0))


def build_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    N = len(embeddings)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    cosine_dists, indices = nn.kneighbors(embeddings)
    angular_dists = cosine_to_angular(cosine_dists)

    W = np.zeros((N, N))
    for i in range(N):
        for j_pos in range(k):
            j = indices[i, j_pos]
            w = angular_dists[i, j_pos]
            if W[j, i] > 0:
                sym_w = min(w, W[j, i])
                W[i, j] = sym_w
                W[j, i] = sym_w
            else:
                W[i, j] = w
    return W


def find_min_connected_k(embeddings: np.ndarray, k_min: int = 3) -> tuple[int, np.ndarray]:
    N = len(embeddings)
    for k in range(k_min, N):
        W = build_knn_graph(embeddings, k)
        n_components, _ = connected_components(
            csgraph=csr_matrix(W), directed=False, return_labels=True
        )
        print(f"  k={k}: {n_components} component(s)")
        if n_components == 1:
            print(f"  => Fully connected at k={k}")
            return k, W
    raise ValueError(f"Graph not connected even at k={N - 1}")


# ── Step 5: All-pairs geodesic distances ───────────────────────────────────────


def compute_geodesic(W: np.ndarray) -> np.ndarray:
    geo = shortest_path(csr_matrix(W), method="auto", directed=False)
    assert not np.any(np.isinf(geo)), "Geodesic matrix has inf — graph is not fully connected"
    return geo


# ── Step 6: GTDB patristic distances ───────────────────────────────────────────


def download_tree(tree_path: Path) -> None:
    """Download bac120.tree if not already present."""
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading GTDB tree from {TREE_URL} ...")
    urllib.request.urlretrieve(TREE_URL, str(tree_path))
    print(f"  Saved to {tree_path}  ({tree_path.stat().st_size / 1e6:.1f} MB)")


def compute_patristic_distances(
    tree_path: Path,
    gtdb_accessions: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise patristic distances for the given GTDB accessions.

    Returns (D, present_mask) where present_mask[i] is True iff gtdb_accessions[i]
    is a leaf in the tree. Distances for absent taxa are left as 0 and the caller
    should restrict any downstream comparison to present taxa via present_mask.

    The GTDB bac120 tree has ~100K leaves. dendropy's phylogenetic_distance_matrix()
    materializes ALL pairwise distances among every leaf (~10^10 pairs for the full
    tree), which exhausts memory. So we first prune the tree down to just our target
    taxa, then compute the (small) distance matrix on the pruned tree. Tree leaf
    names match gtdb_accession (e.g. "GB_GCA_000001405.15").
    """
    import dendropy

    print("  Parsing GTDB tree with dendropy...")
    # preserve_underscores=True: Newick treats unquoted underscores as spaces by
    # default, which would turn "RS_GCF_..." leaf labels into "RS GCF ..." and break
    # matching against our gtdb_accession values.
    tree = dendropy.Tree.get(
        path=str(tree_path), schema="newick", preserve_underscores=True
    )

    target_set = set(gtdb_accessions)
    all_labels = {t.label for t in tree.taxon_namespace}
    present = sorted(target_set & all_labels)
    missing = target_set - all_labels
    if missing:
        print(
            f"  WARNING: {len(missing)}/{len(target_set)} accessions not found in "
            f"tree (their patristic distances left as 0): "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    print(f"  Pruning tree to {len(present)} target taxa...")
    tree.retain_taxa_with_labels(present)

    print("  Computing patristic distance matrix on pruned tree...")
    pdm = tree.phylogenetic_distance_matrix()

    # taxon_namespace still references the original taxa; map labels -> taxon objects
    # that survive on the pruned tree.
    taxon_map = {t.label: t for t in tree.taxon_namespace if t.label in target_set}

    N = len(gtdb_accessions)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        t_i = taxon_map.get(gtdb_accessions[i])
        if t_i is None:
            continue
        for j in range(i + 1, N):
            t_j = taxon_map.get(gtdb_accessions[j])
            if t_j is None:
                continue
            d = pdm.patristic_distance(t_i, t_j)
            D[i, j] = D[j, i] = max(0.0, d)

    present_mask = np.array([a in taxon_map for a in gtdb_accessions], dtype=bool)
    return D, present_mask


# ── Step 7: Spearman rho + Mantel test ─────────────────────────────────────────


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def mantel_test(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = mat_a.shape[0]
    b_flat = upper_triangle(mat_b)
    obs_rho, _ = spearmanr(upper_triangle(mat_a), b_flat)
    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(N)
        perm_flat = upper_triangle(mat_a[np.ix_(perm, perm)])
        perm_rho, _ = spearmanr(perm_flat, b_flat)
        if perm_rho >= obs_rho:
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_perms + 1)
    return float(obs_rho), float(p_value)


# ── Step 8: Within vs. between phylum analysis ─────────────────────────────────


def within_between_analysis(
    geodesic: np.ndarray,
    groups: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compare within-phylum vs between-phylum geodesic distances via permutation test."""
    N = len(groups)
    pairs_i, pairs_j = np.triu_indices(N, k=1)
    pair_dists = geodesic[pairs_i, pairs_j]
    same = groups[pairs_i] == groups[pairs_j]

    within = pair_dists[same]
    between = pair_dists[~same]
    obs_ratio = between.mean() / within.mean()

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        pf = rng.permutation(groups)
        ps = pf[pairs_i] == pf[pairs_j]
        if ps.any() and (~ps).any():
            if pair_dists[~ps].mean() / pair_dists[ps].mean() >= obs_ratio:
                count += 1
    p_val = (count + 1) / (n_perms + 1)
    return within, between, float(obs_ratio), float(p_val)


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
        default="data/species/sequences",
        help="Directory containing per-species FASTA files (default: data/species/sequences/)",
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
    has_fasta = manifest["ncbi_accession"].apply(
        lambda a: (sequences_dir / f"{a}.fasta").exists()
    )
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
    species_names = manifest["species_name"].tolist()
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
    else:
        embeddings, meta_df = compute_embeddings(manifest, sequences_dir, device)
        np.save(EMBED_NPY, embeddings)
        meta_df.to_csv(EMBED_META, index=False)
        print(f"  Saved embeddings : {EMBED_NPY}  shape={embeddings.shape}")
        print(f"  Saved metadata   : {EMBED_META}")

    print(f"  Embeddings shape : {embeddings.shape}")
    assert embeddings.shape[0] == N, f"Expected {N} rows, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == 4096, f"Expected dim 4096, got {embeddings.shape[1]}"

    # ── 4. k-NN graph ─────────────────────────────────────────────────────────
    print("\n[4] Building k-NN graph with angular distances")
    k_opt, W = find_min_connected_k(embeddings, k_min=3)
    print(f"  Optimal k: {k_opt}")

    # ── 5. Geodesic distances ─────────────────────────────────────────────────
    print("\n[5] Computing all-pairs geodesic distances")
    geodesic = compute_geodesic(W)
    geo_npy = OUT_DIR / "evo2_species_geodesic.npy"
    np.save(geo_npy, geodesic)
    df_geo = pd.DataFrame(geodesic, index=ncbi_accessions, columns=ncbi_accessions)
    df_geo.to_csv(OUT_DIR / "evo2_species_geodesic_labeled.csv")
    print(f"  Saved geodesic matrix : {geo_npy}  shape={geodesic.shape}")
    print(f"  Geodesic range        : [{geodesic.min():.4f}, {geodesic.max():.4f}]")

    # ── 6. GTDB patristic distances ────────────────────────────────────────────
    print("\n[6] GTDB patristic distances")
    if not TREE_PATH.exists():
        download_tree(TREE_PATH)
    else:
        print(f"  Using cached tree: {TREE_PATH}")

    patristic_cache = OUT_DIR / "gtdb_patristic_distances.csv"
    if patristic_cache.exists():
        print(f"  Loading cached patristic matrix from {patristic_cache}")
        patristic = pd.read_csv(patristic_cache, index_col=0).values.astype(np.float32)
        present_mask = (patristic != 0).any(axis=1)
    else:
        patristic, present_mask = compute_patristic_distances(TREE_PATH, gtdb_accessions)
        df_pat = pd.DataFrame(patristic, index=gtdb_accessions, columns=gtdb_accessions)
        df_pat.to_csv(patristic_cache)
        print(f"  Saved patristic matrix : {patristic_cache}  shape={patristic.shape}")

    print(f"  Patristic range : [{patristic.min():.4f}, {patristic.max():.4f}]")

    # Restrict every downstream comparison to species that are leaves in the tree.
    # Species absent from the tree have all-zero patristic rows and would otherwise
    # swamp the correlation with meaningless zeros.
    n_present = int(present_mask.sum())
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
