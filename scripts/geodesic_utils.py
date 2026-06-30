"""Shared geodesic-manifold helpers for the gpnstar (gene) and evo2 (species) pipelines.

These functions are pipeline-agnostic: they operate on an (N, D) embedding matrix
or an (N, N) distance matrix and know nothing about genes vs. species. Both
``scripts/gpnstar/embed_and_geodesic.py`` and
``scripts/evo2/embed_and_geodesic_species.py`` import from here so the k-NN graph,
geodesic, and statistical-test logic lives in exactly one place.

Scripts run as ``uv run python scripts/<dir>/<script>.py`` add ``scripts/`` to
sys.path before importing this module (see the bootstrap at the top of each script).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import pearsonr, rankdata, spearmanr
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
    """Build a symmetric k-NN adjacency matrix with cosine-distance edge weights.

    Follows the Goodfire tree-of-life reproduction (``tree_of_life_reproduction.zip``,
    ``METHODS_DECISIONS.md`` §4 / ``metric_probing/analysis/analysis_utils.py:knn_graph``):

      * Edges are weighted by **cosine distance** (``1 - cosine similarity``) rather
        than angular distance. The neighbor *sets* are identical to the angular case
        (``arccos`` is monotonic), but the geodesic edge weights — and therefore the
        shortest-path sums — differ.
      * The directed k-NN matrix is symmetrized by the **union** rule
        ``A = max(A, Aᵀ)``: an undirected edge is kept whenever *either* endpoint
        lists the other (the Isomap symmetric-kNN graph). This is *not* the mutual /
        intersection (``min``) graph, which keeps an edge only when *both* list each
        other, is sparser, and needs a larger k to connect. Because cosine distance
        is symmetric, ``max`` governs edge *presence*, not the weight value (the two
        stored directed distances are equal whenever both are present).

    ``k`` is the number of non-self neighbors per point. scikit-learn's kneighbors
    result includes the query point itself, so we ask for one extra neighbor and
    skip ``j == i`` below.
    """
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
            # Union / max-symmetrize (Goodfire `A.maximum(A.T)`): keep the larger of
            # the two directed weights. For a symmetric metric the two are equal, so
            # this simply unions the directed edge sets without altering weights.
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


# ── Between-family distances over family centroids ─────────────────────────────


def family_centroids(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
) -> np.ndarray:
    """Per-family centroid vector = mean of the L2-normalized member embeddings.

    Each family collapses to a single direction in embedding space, so the
    centroid is independent of how many members the family has.
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


def compute_centroid_geodesic(
    embeddings: np.ndarray,
    groups: np.ndarray,
    group_order: list[str],
    k_min: int = 3,
) -> np.ndarray:
    """Between-family geodesic built BETWEEN family centroids, not member genes.

    Collapses each family to one centroid (``family_centroids``), then runs the
    standard angular k-NN geodesic over those F centroid nodes only. The result
    is independent of per-family membership counts — a 400-member OR family and
    a 3-member NOS family each contribute exactly one node — unlike averaging
    member-pair geodesics over the gene-level graph, where a large family
    reshapes the manifold every path traverses. With F small the graph is
    near-complete, so the geodesic stays close to the direct centroid angular
    distance (there is little manifold to follow at the family level).
    """
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
    """Sweep k-NN ``k`` and correlate the resulting geodesic against ``reference``.

    For each k: build the angular-distance k-NN graph, count connected components,
    compute all-pairs geodesics, then correlate geodesic-vs-reference (e.g. patristic)
    over the strict upper triangle, using only finite pairs (an unconnected graph leaves
    some geodesics infinite). If ``present_mask`` is given, both matrices are restricted
    to those indices first (e.g. to drop species absent from the reference tree).

    Single source of truth for both ``scripts/evo2/troubleshooting/k_sweep.py``
    (sensitivity sweep) and ``embed_and_geodesic_species.py`` (which picks the
    lowest connected k from the result).
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


def chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    """Chatterjee's rank-correlation coefficient ξₙ (Chatterjee 2021).

    Ported verbatim from the Goodfire tree-of-life reproduction
    (``metric_probing/analysis/analysis_utils.py:compute_chatterjee_correlation``),
    which reports it alongside Spearman/Pearson. ξ lies in [0, 1]: 0 ⇔ independence,
    1 ⇔ Y is (a.s.) a measurable function of X. Unlike Spearman/Pearson it detects
    *non-monotonic* functional dependence, but it is **asymmetric** — ξ(x, y) ≠
    ξ(y, x) in general (it measures how well Y is predictable from X).

    Ties: X-ties keep NumPy's stable mergesort order; Y is ranked with average ranks
    (matching the R ``xicor`` package), using the ties-robust denominator so the
    estimate stays valid when Y has ties.
    """
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


# ── Control reconstruction (preservation): control geometry vs the natural geometry ──────
# Shared by the within/between control scorers (analyses/embed_and_score_controls.py and
# analyses/embed_and_score_msa_controls.py). The metric is rho->1 = the control reconstructed
# the natural geometry (that aspect of the sequence was not load-bearing); rho falling = the
# ablated aspect carried real structure.


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
    """Per-family Spearman(control submatrix, natural submatrix) of two per-gene geodesics
    that share the SAME gene order (rows/cols of both matrices index the same genes, with
    `families` the per-gene family label). Families with < `min_members` genes (or zero
    variance) are skipped. Returns ([(family, n_members, rho), ...], equal-weight mean)."""
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
