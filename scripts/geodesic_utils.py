"""Shared geodesic helpers, pipeline-agnostic: they take an (N, D) embedding or an (N, N) distance
matrix and know nothing about genes vs species.
"""

import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

# ── k-NN graph with angular distances


def cosine_to_angular(cosine_dist: np.ndarray) -> np.ndarray:
    """Convert cosine distance (1 - cosine similarity) to angular distance in radians."""
    return np.arccos(np.clip(1.0 - cosine_dist, -1.0, 1.0))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity (N, N) via L2-normalized embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norms, 1e-12, None)
    return unit @ unit.T


def build_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Symmetric k-NN adjacency with cosine-distance edge weights."""
    N = len(embeddings)
    if k < 1 or k >= N:
        raise ValueError(f"k must be in [1, {N - 1}], got {k}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    cosine_dists, indices = nn.kneighbors(embeddings)

    W = np.zeros((N, N))
    for i in range(N):
        n_added = 0
        for j_pos in range(k + 1):
            j = indices[i, j_pos]
            if j == i:
                continue
            w = cosine_dists[i, j_pos]
            # Union-symmetrize: for a symmetric metric the two directed weights are equal,
            # so this unions the edge sets without altering weights.
            sym_w = max(w, W[i, j])
            W[i, j] = sym_w
            W[j, i] = sym_w
            n_added += 1
            if n_added == k:
                break
        if n_added != k:
            raise ValueError(f"Only found {n_added} non-self neighbors for row {i}; expected {k}")
    return W


def find_min_connected_k(embeddings: np.ndarray, k_min: int = 3) -> tuple[int, np.ndarray]:
    """Smallest k >= k_min whose k-NN graph is connected; returns (k, adjacency)."""
    N = len(embeddings)
    if k_min >= N:
        raise ValueError(f"k_min={k_min} must be < N={N}")

    def connected(k: int):
        W = build_knn_graph(embeddings, k)
        n_components, _ = connected_components(
            csgraph=csr_matrix(W), directed=False, return_labels=True
        )
        print(f"  k={k}: {n_components} component(s)")
        return n_components == 1, W

    # Phase 1 — double until connected, to bracket the answer in (lo, hi].
    lo, hi = k_min, k_min  # invariant: lo-1 known disconnected (or lo == k_min)
    ok, W_hi = connected(hi)
    while not ok:
        if hi >= N - 1:
            raise ValueError(f"Graph not connected even at k={N - 1}")
        lo, hi = hi + 1, min(hi * 2, N - 1)
        ok, W_hi = connected(hi)

    # Phase 2 — bisect [lo, hi]. hi is always a connected candidate and W_hi its graph.
    while lo < hi:
        mid = (lo + hi) // 2
        ok, W_mid = connected(mid)
        if ok:
            hi, W_hi = mid, W_mid
        else:
            lo = mid + 1

    print(f"  => Fully connected at k={hi}")
    return hi, W_hi


# ── All-pairs geodesic distances


def compute_geodesic(W: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path (geodesic) distances over the undirected k-NN graph."""
    geo = shortest_path(csr_matrix(W), method="auto", directed=False)
    assert not np.any(np.isinf(geo)), "Geodesic matrix has inf — graph is not fully connected"
    return geo


# ── Between-family distances over family centroids


def family_centroids(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
) -> np.ndarray:
    """
    Per-family centroid = mean of the L2-normalized members, so a family collapses to one direction
    independent of its size.
    """
    cents = []
    for g in group_order:
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            raise ValueError(f"No members for group {g!r}")
        unit = embeddings[idx] / np.clip(
            np.linalg.norm(embeddings[idx], axis=1, keepdims=True), 1e-12, None
        )
        cents.append(unit.mean(axis=0))
    return np.vstack(cents)


# ── centroid-graph k rule
# Use ceil(sqrt(F)) when callers request a fixed centroid-graph density.
# Published between-family analyses use graph-free Wasserstein distances.
def centroid_graph_k(F: int, k_min: int = 3) -> int:
    """k for the F-centroid k-NN graph: ceil(sqrt(F)), never below k_min, never above F-1."""
    return int(min(max(k_min, math.ceil(math.sqrt(F))), F - 1))


def compute_centroid_geodesic(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
    k_min: int = 3,
) -> np.ndarray:
    """Between-family geodesic over family centroids rather than member genes, so each family is one
    node and the result is independent of membership counts.
    """
    # See `centroid_graph_k` above for why k is not raised here.
    cents = family_centroids(embeddings, groups, group_order)
    F = len(group_order)
    _, W = find_min_connected_k(cents, k_min=min(k_min, F - 1))
    return compute_geodesic(W)


def k_sweep_correlations(
    embeddings: np.ndarray,
    reference: np.ndarray,
    k_values,
    present_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep k and correlate the resulting geodesic against `reference`, one row per k.
    Finite pairs only, since an unconnected graph leaves some geodesics infinite.
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


# ── Matrix correlation: Spearman + Mantel


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


def chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee's rank correlation xi in [0, 1]: 0 for independence, 1 when Y is a.s."""
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    n = x.size
    if n < 2:
        return float("nan")

    order = np.argsort(x, kind="mergesort")  # stable sort: X-ties keep input order
    y_sorted = y[order]
    r = rankdata(y_sorted, method="average")  # ranks of Y, average rank for ties

    num = np.sum(np.abs(np.diff(r)))  # Σ |r_{i+1} − r_i|
    ell = n + 1 - r  # l_i = #{j : Y_j ≥ Y_i}, derived from ranks
    denom = 2 * np.sum(ell * (n - ell))  # ties-robust denominator
    xi = 1 - n * num / denom
    return float(np.clip(xi, 0.0, 1.0))


# ── Distance-uniform pair sampling


def uniform_distance_bin_edges(
    ref_flat: np.ndarray,
    n_bins: int,
    span: tuple[float, float] | None = None,
) -> np.ndarray:
    """Equal-WIDTH bin edges over the reference-distance range."""
    lo, hi = (float(ref_flat.min()), float(ref_flat.max())) if span is None else span
    if not hi > lo:
        raise ValueError(f"Degenerate reference range [{lo}, {hi}]")
    return np.linspace(lo, hi, n_bins + 1)


def uniform_distance_pair_indices(
    ref_flat: np.ndarray,
    n_bins: int,
    per_bin: int,
    rng: np.random.Generator,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """
    Indices of pairs sampled ~uniformly along the reference distance axis. Underfilled bins
    contribute
    what they have rather than padding with replacement.
    """
    if edges is None:
        edges = uniform_distance_bin_edges(ref_flat, n_bins)
    # digitize on the interior edges => bin index in [0, n_bins - 1], max value included
    bin_idx = np.clip(np.digitize(ref_flat, edges[1:-1]), 0, len(edges) - 2)
    picks = []
    for b in range(len(edges) - 1):
        members = np.flatnonzero(bin_idx == b)
        if members.size == 0:
            continue
        take = min(per_bin, members.size)
        picks.append(rng.choice(members, size=take, replace=False))
    return np.concatenate(picks)


def uniform_distance_bin_occupancy(
    ref_flat: np.ndarray,
    n_bins: int,
    per_bin: int,
    edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Per-bin pair counts and how many a `per_bin` draw can take — underfilled near-zero bins mean the
    design is uniform only over the range the species sample covers.
    """
    if edges is None:
        edges = uniform_distance_bin_edges(ref_flat, n_bins)
    bin_idx = np.clip(np.digitize(ref_flat, edges[1:-1]), 0, len(edges) - 2)
    counts = np.bincount(bin_idx, minlength=len(edges) - 1)
    return pd.DataFrame(
        {
            "bin": np.arange(len(counts)),
            "lo": edges[:-1],
            "hi": edges[1:],
            "n_pairs": counts,
            "n_sampled": np.minimum(counts, per_bin),
            "underfilled": (counts < per_bin) & (counts > 0),
            "empty": counts == 0,
        }
    )


def uniform_pair_correlations(
    x_flat: np.ndarray,
    ref_flat: np.ndarray,
    n_bins: int = 20,
    per_bin: int = 100,
    n_boot: int = 500,
    seed: int = 42,
    edges: np.ndarray | None = None,
) -> dict:
    """Pearson / Spearman / xi over pairs sampled uniformly in `ref_flat`, averaged over `n_boot`
    draws.
    """
    x_flat = np.asarray(x_flat, dtype=np.float64)
    ref_flat = np.asarray(ref_flat, dtype=np.float64)
    finite = np.isfinite(x_flat) & np.isfinite(ref_flat)
    x_flat, ref_flat = x_flat[finite], ref_flat[finite]

    if edges is None:
        edges = uniform_distance_bin_edges(ref_flat, n_bins)
    occ = uniform_distance_bin_occupancy(ref_flat, n_bins, per_bin, edges=edges)

    rng = np.random.default_rng(seed)
    pearsons, spearmans, xis, sizes = [], [], [], []
    for _ in range(n_boot):
        idx = uniform_distance_pair_indices(ref_flat, n_bins, per_bin, rng, edges=edges)
        xs, rs = x_flat[idx], ref_flat[idx]
        pearsons.append(pearsonr(xs, rs)[0])
        spearmans.append(spearmanr(xs, rs)[0])
        xis.append(chatterjee_xi(xs, rs))
        sizes.append(idx.size)

    return {
        "n_bins": n_bins,
        "per_bin": per_bin,
        "n_boot": n_boot,
        "mean_pairs_per_draw": float(np.mean(sizes)),
        "n_bins_occupied": int((~occ["empty"]).sum()),
        "n_bins_underfilled": int(occ["underfilled"].sum()),
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_sd": float(np.std(pearsons, ddof=1)),
        "spearman_mean": float(np.mean(spearmans)),
        "spearman_sd": float(np.std(spearmans, ddof=1)),
        "chatterjee_xi_mean": float(np.mean(xis)),
        "chatterjee_xi_sd": float(np.std(xis, ddof=1)),
    }


# ── Within- vs. between-group geodesic analysis


def within_between_analysis(
    geodesic: np.ndarray,
    groups: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Within-group vs between-group geodesic distances by permutation test.
    Returns (within, between, ratio, p_value).
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


# ── Control preservation: rho -> 1 means the ablated aspect was not load-bearing;
# rho falling means it carried real structure.


def between_preservation_rho(nat_centroid: np.ndarray, ctrl_centroid: np.ndarray) -> float:
    """Spearman rho between the upper triangles of two aligned F x F centroid-distance
    matrices (natural vs a control). Caller aligns both to the same family order. NaN-safe."""
    a, b = upper_triangle(nat_centroid), upper_triangle(ctrl_centroid)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def within_preservation_rho(
    nat_geo: np.ndarray,
    ctrl_geo: np.ndarray,
    families: np.ndarray,
    family_order: list[str] | None = None,
    min_members: int = 4,
) -> tuple[list[tuple[str, int, float]], float]:
    """Per-family Spearman between two per-gene geodesics sharing the same gene order.
    Families below `min_members` or with zero variance are skipped.
    """
    order = family_order if family_order is not None else sorted(set(map(str, families)))
    rows: list[tuple[str, int, float]] = []
    for fam in order:
        idx = np.where(families == fam)[0]
        if len(idx) < min_members:
            continue
        a = upper_triangle(nat_geo[np.ix_(idx, idx)])
        b = upper_triangle(ctrl_geo[np.ix_(idx, idx)])
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        rows.append((str(fam), int(len(idx)), float(spearmanr(a, b).correlation)))
    mean = float(np.mean([r[2] for r in rows])) if rows else float("nan")
    return rows, mean
