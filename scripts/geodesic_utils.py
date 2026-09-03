"""Shared k-nearest-neighbor geodesic and matrix-correlation helpers."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return a symmetric cosine-distance k-nearest-neighbor adjacency matrix."""
    n = len(embeddings)
    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    graph = np.zeros((n, n))
    for i in range(n):
        added = 0
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i:
                continue
            weight = max(distances[i, j_pos], graph[i, j])
            graph[i, j] = weight
            graph[j, i] = weight
            added += 1
            if added == k:
                break
        if added != k:
            raise ValueError(f"Only found {added} non-self neighbors for row {i}; expected {k}")
    return graph


def find_min_connected_k(embeddings: np.ndarray, k_min: int = 3) -> tuple[int, np.ndarray]:
    """Return the smallest connected k-nearest-neighbor graph at or above ``k_min``."""
    n = len(embeddings)
    if k_min >= n:
        raise ValueError(f"k_min={k_min} must be < N={n}")

    def connected(k: int) -> tuple[bool, np.ndarray]:
        graph = build_knn_graph(embeddings, k)
        n_components, _ = connected_components(
            csgraph=csr_matrix(graph), directed=False, return_labels=True
        )
        print(f"  k={k}: {n_components} component(s)")
        return n_components == 1, graph

    lo = hi = k_min
    ok, graph_hi = connected(hi)
    while not ok:
        if hi >= n - 1:
            raise ValueError(f"Graph not connected even at k={n - 1}")
        lo, hi = hi + 1, min(hi * 2, n - 1)
        ok, graph_hi = connected(hi)

    while lo < hi:
        mid = (lo + hi) // 2
        ok, graph_mid = connected(mid)
        if ok:
            hi, graph_hi = mid, graph_mid
        else:
            lo = mid + 1

    print(f"  => Fully connected at k={hi}")
    return hi, graph_hi


def compute_geodesic(graph: np.ndarray) -> np.ndarray:
    """Return all-pairs shortest-path distances over an undirected connected graph."""
    geodesic = shortest_path(csr_matrix(graph), method="auto", directed=False)
    if np.any(np.isinf(geodesic)):
        raise ValueError("Geodesic matrix contains infinity; the graph is disconnected")
    return geodesic


def family_centroids(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
) -> np.ndarray:
    """Return mean L2-normalized embeddings for groups in ``group_order``."""
    centroids = []
    for group in group_order:
        idx = np.where(groups == group)[0]
        if len(idx) == 0:
            raise ValueError(f"No members for group {group!r}")
        unit = embeddings[idx] / np.clip(
            np.linalg.norm(embeddings[idx], axis=1, keepdims=True), 1e-12, None
        )
        centroids.append(unit.mean(axis=0))
    return np.vstack(centroids)


def compute_centroid_geodesic(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
    k_min: int = 3,
) -> np.ndarray:
    """Return a connected k-nearest-neighbor geodesic over group centroids."""
    centroids = family_centroids(embeddings, groups, group_order)
    _, graph = find_min_connected_k(centroids, k_min=min(k_min, len(group_order) - 1))
    return compute_geodesic(graph)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a square matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def mantel_test(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    """Return one-sided Spearman Mantel rho and permutation p-value."""
    rng = np.random.default_rng(seed)
    n = mat_a.shape[0]
    b_flat = upper_triangle(mat_b)
    observed = spearmanr(upper_triangle(mat_a), b_flat).statistic
    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        permuted = upper_triangle(mat_a[np.ix_(perm, perm)])
        if spearmanr(permuted, b_flat).statistic >= observed:
            count_extreme += 1
    return float(observed), float((count_extreme + 1) / (n_perms + 1))
