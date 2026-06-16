"""Shared geodesic-manifold helpers for the gpnstar (gene) and evo2 (species) pipelines.

These functions are pipeline-agnostic: they operate on an (N, D) embedding matrix
or an (N, N) distance matrix and know nothing about genes vs. species. Both
``scripts/gpnstar/embed_and_geodesic_genes.py`` and
``scripts/evo2/embed_and_geodesic_species.py`` import from here so the k-NN graph,
geodesic, and statistical-test logic lives in exactly one place.

Scripts run as ``uv run python scripts/<dir>/<script>.py`` add ``scripts/`` to
sys.path before importing this module (see the bootstrap at the top of each script).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

# ── k-NN graph with angular distances ──────────────────────────────────────────


def cosine_to_angular(cosine_dist: np.ndarray) -> np.ndarray:
    """Convert cosine distance (1 - cosine similarity) to angular distance in radians."""
    return np.arccos(np.clip(1.0 - cosine_dist, -1.0, 1.0))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity (N, N) via L2-normalized embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norms, 1e-12, None)
    return unit @ unit.T


def build_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetric k-NN adjacency matrix with angular-distance edge weights.

    ``k`` is the number of non-self neighbors per point. scikit-learn's
    kneighbors result includes the query point itself, so we ask for one extra
    neighbor and skip ``j == i`` below.

    For mutual neighbors the smaller of the two angular distances is kept.
    """
    N = len(embeddings)
    if k < 1 or k >= N:
        raise ValueError(f"k must be in [1, {N - 1}], got {k}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    cosine_dists, indices = nn.kneighbors(embeddings)
    angular_dists = cosine_to_angular(cosine_dists)

    W = np.zeros((N, N))
    for i in range(N):
        n_added = 0
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i:
                continue
            w = angular_dists[i, j_pos]
            if W[j, i] > 0:
                sym_w = min(w, W[j, i])
                W[i, j] = sym_w
                W[j, i] = sym_w
            else:
                W[i, j] = w
            n_added += 1
            if n_added == k:
                break
        if n_added != k:
            raise ValueError(f"Only found {n_added} non-self neighbors for row {i}; expected {k}")
    return W


def find_min_connected_k(embeddings: np.ndarray, k_min: int = 3) -> tuple[int, np.ndarray]:
    """Increase k until the k-NN graph is fully connected; return (k, adjacency matrix)."""
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


# ── All-pairs geodesic distances ───────────────────────────────────────────────


def compute_geodesic(W: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path (geodesic) distances over the undirected k-NN graph."""
    geo = shortest_path(csr_matrix(W), method="auto", directed=False)
    assert not np.any(np.isinf(geo)), "Geodesic matrix has inf — graph is not fully connected"
    return geo


def k_sweep_correlations(
    embeddings: np.ndarray,
    reference: np.ndarray,
    k_values,
    present_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep k-NN ``k`` and correlate the resulting geodesic against ``reference``.

    For each k: build the angular-distance k-NN graph, count connected components,
    compute all-pairs geodesics, then correlate geodesic-vs-reference (e.g. patristic)
    over the strict upper triangle, using only finite pairs (an unconnected graph leaves
    some geodesics infinite). If ``present_mask`` is given, both matrices are restricted
    to those indices first (e.g. to drop species absent from the reference tree).

    Single source of truth for both ``scripts/evo2/troubleshooting/k_sweep.py`` (sensitivity sweep) and
    ``embed_and_geodesic_species.py`` (which picks the lowest connected k from the result).
    Returns one row per k with columns: k, n_components, connected, frac_finite_pairs,
    pearson_geodesic_phylo, spearman_geodesic_phylo.
    """
    # The k-NN graph and connectivity are always over ALL points; only the correlation
    # is restricted to present_mask (so connectivity reflects the real graph).
    ref_mat = reference if present_mask is None else reference[np.ix_(present_mask, present_mask)]
    pat_flat = upper_triangle(ref_mat)

    rows = []
    for k in k_values:
        W = build_knn_graph(embeddings, k)
        n_comp, _ = connected_components(csr_matrix(W), directed=False)
        geo = shortest_path(csr_matrix(W), method="auto", directed=False)
        if present_mask is not None:
            geo = geo[np.ix_(present_mask, present_mask)]

        geo_flat = upper_triangle(geo)
        finite = np.isfinite(geo_flat)
        frac_finite = float(finite.mean())
        gf, pf = geo_flat[finite], pat_flat[finite]
        pe, _ = pearsonr(gf, pf)
        sp, _ = spearmanr(gf, pf)
        rows.append(
            {
                "k": int(k),
                "n_components": int(n_comp),
                "connected": n_comp == 1,
                "frac_finite_pairs": frac_finite,
                "pearson_geodesic_phylo": float(pe),
                "spearman_geodesic_phylo": float(sp),
            }
        )
    return pd.DataFrame(rows)


# ── Matrix correlation: Spearman + Mantel ──────────────────────────────────────


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle (k=1) of a square matrix to a 1-D vector."""
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def mantel_test(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    """One-sided Mantel test: Spearman rho between two distance matrices with a
    row/column permutation null. Returns (observed_rho, p_value)."""
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


# ── Within- vs. between-group geodesic analysis ────────────────────────────────


def within_between_analysis(
    geodesic: np.ndarray,
    groups: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compare within-group vs. between-group geodesic distances via permutation test.

    ``groups`` is a per-row label array (e.g. phylum for species, family for genes).
    Returns (within_distances, between_distances, between/within ratio, p_value).
    """
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
