"""Formalized layer-selection benchmark for gene-family latent-space analyses.

Model-agnostic scoring engine. Given a per-model layer stack (n_layers, N, H) plus a
metadata table and the gene sequences, it scores EVERY layer on two components, reported
separately, and writes the deliverable table + a recommendation:

  1. within/between CLUSTERING RATIO (Euclidean) = within_dispersion / between_dispersion
       within_dispersion  = mean over families of the family's MEAN PAIRWISE Euclidean
                            distance among its members. Each family is averaged to ONE
                            number first, then those are averaged with equal weight, so a
                            large family cannot dominate (member→member, not member→centroid).
       between_dispersion = mean pairwise Euclidean distance between family CENTROIDS
                            (raw mean embedding); each family contributes exactly one node,
                            so it is independent of per-family membership counts.
       A lower ratio = tighter, better-separated family clusters.

  2. per-family PCA variance: for EACH family, run a PCA on that family's stacked member-gene
       embeddings (centred on the family's average gene) and report the fraction of the family's
       variance carried by its top 3 principal components; aggregate as the EQUAL-WEIGHT mean
       across families (and write a per-family breakdown CSV at the recommended layer). A high
       fraction means a family's member genes vary along only a few axes (low-rank family cloud).

The earlier composite S_within / S_between / Pfam-JSD-homology scoring (patristic
ground truth, blocked-CV family probes, nuisance subtraction) is preserved below but
COMMENTED OUT at its call sites — see the helper functions and the `main()` loop.

The layer stack itself is produced per model:
  * Evo2:      scripts/evo2/layer_sweep.py  (all 32 blocks, one forward pass per CDS)
  * GPN-Star:  scripts/gpnstar/layer_sweep.py  (all 17 hidden states, one forward pass)

Usage:
    uv run python scripts/layer_selection/layer_selection.py --model evo2
    uv run python scripts/layer_selection/layer_selection.py --model gpnstar
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))  # baseline modules
from geodesic_utils import (  # noqa: E402
    build_knn_graph,
    compute_centroid_geodesic,
    compute_geodesic,
    upper_triangle,
)
from kmer_sequence_divergence import cosine_distance_matrix, kmer_frequency_vector  # noqa: E402
from protein_alignment_patristic_seqid import (  # noqa: E402
    align_members,
    seqid_distance_matrix,
    tree_patristic,
)

# ── scoring constants ────────────────────────────────────────────────────────────

MIN_MEMBERS = 4          # families below this give a meaningless within-family rank corr
KNN_SUBFAMILY = 5        # neighbours examined for same-subfamily retrieval
N_BOOTSTRAP = 200        # member-resamples for dist-corr stability
N_SPLITS = 5             # blocked-CV folds
PCA_DIM = 100            # probe feature dim (speed + conditioning); <= N-1, H
RANDOM_STATE = 0
SEQID_CLUSTER_DIST = 0.7  # within-family single-linkage block threshold (= <30% identity)


# ── model loaders → a common ProblemData ─────────────────────────────────────────


@dataclass
class ProblemData:
    model: str
    stack: np.ndarray            # (L, N, H)
    layer_labels: list[str]      # human-readable per-layer tag (depth + type)
    ids: list[str]               # member id per column of stack
    family: np.ndarray           # (N,) family label
    subfamily: np.ndarray | None  # (N,) finer label, or None if unavailable
    clade: np.ndarray | None     # (N,) clade/group for blocked split, or None
    length: np.ndarray           # (N,) CDS length (nuisance)
    gc: np.ndarray               # (N,) GC fraction (nuisance)
    seqs: dict[str, str]         # id -> CDS, for the patristic ground truth


def _gc(seq: str) -> float:
    s = seq.upper()
    g = sum(s.count(b) for b in "GC")
    return g / max(len(s), 1)


def load_evo2() -> ProblemData:
    cache = Path("data/cache/evo2_layer_sweep")
    stack = np.load(cache / "layer_stack.npy")
    meta = pd.read_csv(cache / "metadata.csv")
    # scripts/evo2 (not the installed `evo2` model pkg) holds these sibling modules.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2"))
    from embed_and_geodesic_ortholog import load_family_sequences  # noqa: E402
    from evo2_embedding import layer_type  # noqa: E402

    seqs = load_family_sequences(meta["family"].tolist())
    ids = meta["org_gene"].tolist()
    labels = [f"blocks.{i} ({layer_type(i)})" for i in range(stack.shape[0])]
    return ProblemData(
        model="evo2",
        stack=stack,
        layer_labels=labels,
        ids=ids,
        family=meta["family"].to_numpy(),
        subfamily=meta["ko_label"].to_numpy(),
        clade=meta["group"].to_numpy(),
        length=meta["cds_len"].to_numpy().astype(float),
        gc=np.array([_gc(seqs[g]) for g in ids]),
        seqs=seqs,
    )


def load_gpnstar(model: str = "vertebrate") -> ProblemData:
    cache = Path("data/cache/gpnstar_layer_sweep")
    config = json.loads((cache / "config.json").read_text())
    stack = np.load(cache / f"layer_stack_{model}.npy")
    ids = config["genes"]
    seqs_all = json.loads(Path("data/cache/cds_sequences.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gene_families import family_members, family_order as _family_order  # noqa: E402

    GENE_FAMILIES = family_members("human")
    FAMILY_ORDER = _family_order("human")
    fam_of = {g: fam for fam in FAMILY_ORDER for g in GENE_FAMILIES[fam]}
    family = np.array([fam_of.get(g, "?") for g in ids])
    missing = [g for g in ids if g not in seqs_all]
    if missing:
        print(f"  WARNING: {len(missing)} genes lack a CDS (patristic will skip them): {missing[:5]}")
    seqs = {g: seqs_all[g] for g in ids if g in seqs_all}
    length = np.array([len(seqs.get(g, "")) for g in ids], dtype=float)
    gc = np.array([_gc(seqs[g]) if g in seqs else np.nan for g in ids])
    n_states = stack.shape[0]
    labels = [("embeddings" if i == 0 else f"hidden_state.{i}") for i in range(n_states)]
    return ProblemData(
        model="gpnstar",
        stack=stack,
        layer_labels=labels,
        ids=ids,
        family=family,
        subfamily=None,        # human-only panel: no clean per-gene subfamily; renormalized
        clade=None,            # human-only: block by within-family seq-id cluster instead
        length=length,
        gc=gc,
        seqs=seqs,
    )


def _human_problem(model: str, stack: np.ndarray, meta: pd.DataFrame, labels: list[str]) -> ProblemData:
    """ProblemData for a matched-manifold human Panel-1 sweep. Only stack + family + layer_labels
    feed the current clustering-ratio / per-family-PCA scoring; the patristic/probe fields are
    set empty (the old composite scoring is disabled)."""
    n = stack.shape[1]
    return ProblemData(
        model=model, stack=stack, layer_labels=labels,
        ids=meta["gene"].tolist(), family=meta["family"].to_numpy(),
        subfamily=None, clade=None,
        length=np.zeros(n), gc=np.full(n, np.nan), seqs={},
    )


def load_evo2_human() -> ProblemData:
    cache = Path("data/cache/evo2_human_layer_sweep")
    stack = np.load(cache / "layer_stack.npy")
    meta = pd.read_csv(cache / "metadata.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evo2"))
    from evo2_embedding import layer_type  # noqa: E402
    labels = [f"blocks.{i} ({layer_type(i)})" for i in range(stack.shape[0])]
    return _human_problem("evo2", stack, meta, labels)


def load_gpnstar_human(model: str = "vertebrate") -> ProblemData:
    cache = Path("data/cache/gpnstar_human_layer_sweep")
    stack = np.load(cache / f"layer_stack_{model}.npy")
    meta = pd.read_csv(cache / "metadata.csv")
    labels = [("embeddings" if i == 0 else f"hidden_state.{i}") for i in range(stack.shape[0])]
    return _human_problem("gpnstar", stack, meta, labels)


# ── patristic ground truth (computed once, reused across all layers) ──────────────


def build_patristic(pd_: ProblemData) -> dict[str, tuple[list[str], np.ndarray]]:
    """Per-family patristic distance matrix from one MAFFT→FastTree pass (cached on disk).

    Returns {family: (member_ids, D)} where D is the patristic distance over member_ids.
    Sequence-derived and layer-independent, so it is computed once per model.
    """
    cache = Path(f"data/cache/{pd_.model}_patristic")
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    import tempfile

    fams = sorted(set(pd_.family))
    print(f"[patristic] {len(fams)} families (cached at {cache})")
    with tempfile.TemporaryDirectory(dir="/opt/dlami/nvme/uv/tmp") as td:
        for fam in fams:
            members = [pd_.ids[i] for i in range(len(pd_.ids))
                       if pd_.family[i] == fam and pd_.ids[i] in pd_.seqs]
            if len(members) < MIN_MEMBERS:
                continue
            npy = cache / f"{fam}.npy"
            ids_json = cache / f"{fam}.ids.json"
            if npy.exists() and ids_json.exists() and json.loads(ids_json.read_text()) == members:
                out[fam] = (members, np.load(npy))
                continue
            wd = Path(td) / fam
            wd.mkdir(parents=True, exist_ok=True)
            res = align_members(members, pd_.seqs, wd)
            if res is None:
                print(f"  {fam}: alignment failed, skipped")
                continue
            aln, _ = res
            D = tree_patristic(aln, members, wd)
            if D is None:
                print(f"  {fam}: tree failed, skipped")
                continue
            np.save(npy, D)
            ids_json.write_text(json.dumps(members))
            out[fam] = (members, D)
            print(f"  {fam}: n={len(members)} patristic ready")
    return out


def build_seqid_clusters(pd_: ProblemData) -> np.ndarray:
    """Within-family single-linkage clusters at SEQID_CLUSTER_DIST, as global block labels.

    Used as the blocked-split grouping when there is no clade axis (GPN-Star, human-only).
    """
    import tempfile

    labels = np.array(["none"] * len(pd_.ids), dtype=object)
    cid = 0
    with tempfile.TemporaryDirectory(dir="/opt/dlami/nvme/uv/tmp") as td:
        for fam in sorted(set(pd_.family)):
            idx = [i for i in range(len(pd_.ids)) if pd_.family[i] == fam and pd_.ids[i] in pd_.seqs]
            members = [pd_.ids[i] for i in idx]
            if len(members) < 2:
                for i in idx:
                    labels[i] = f"c{cid}"; cid += 1
                continue
            wd = Path(td) / fam
            wd.mkdir(parents=True, exist_ok=True)
            res = align_members(members, pd_.seqs, wd)
            if res is None:
                for i in idx:
                    labels[i] = f"c{cid}"; cid += 1
                continue
            _, aligned = res
            D = seqid_distance_matrix(members, aligned)
            adj = np.nan_to_num(D, nan=1.0) < SEQID_CLUSTER_DIST
            n_comp, comp = connected_components(csr_matrix(adj), directed=False)
            for j, i in enumerate(idx):
                labels[i] = f"f{fam[:4]}_{cid + comp[j]}"
            cid += n_comp
    return labels


# ── per-layer geometry ───────────────────────────────────────────────────────────


def lowest_connected_k(emb: np.ndarray, k_min: int = 3):
    N = len(emb)
    for k in range(k_min, N):
        W = build_knn_graph(emb, k)
        n_comp, _ = connected_components(csr_matrix(W), directed=False)
        if n_comp == 1:
            return k, W
    raise ValueError("graph never connects")


def angular_matrix(emb: np.ndarray) -> np.ndarray:
    u = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return np.arccos(np.clip(u @ u.T, -1.0, 1.0))


def fisher_mean(rhos: list[float]) -> float:
    r = np.clip(np.array([x for x in rhos if np.isfinite(x)]), -0.999, 0.999)
    if len(r) == 0:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(r))))


# ── between-family STRUCTURE (homology) helpers ───────────────────────────────────


def family_centroid_distances(D: np.ndarray, family: np.ndarray, order: list[str]) -> np.ndarray:
    """(F, F) mean pairwise distance: within-family on the diagonal, between off-diagonal."""
    F = len(order)
    C = np.zeros((F, F))
    idx = {f: np.where(family == f)[0] for f in order}
    for i, a in enumerate(order):
        ia = idx[a]
        for j, b in enumerate(order):
            if i == j:
                sub = D[np.ix_(ia, ia)]
                C[i, j] = sub[np.triu_indices(len(ia), 1)].mean() if len(ia) > 1 else 0.0
            else:
                C[i, j] = D[np.ix_(ia, idx[b])].mean()
    return C


def family_kmer_distance(pd_: ProblemData, order: list[str], k: int = 4) -> np.ndarray:
    """(F, F) cosine distance between mean family k-mer frequency vectors (composition control)."""
    means = []
    for f in order:
        vs = [kmer_frequency_vector(pd_.seqs[g], k) for g in pd_.ids
              if pd_.family[pd_.ids.index(g)] == f and g in pd_.seqs]
        means.append(np.mean(vs, axis=0))
    return cosine_distance_matrix(np.array(means))


def family_gc_distance(pd_: ProblemData, order: list[str]) -> np.ndarray:
    """(F, F) |Δ mean GC| between families (composition control)."""
    g = np.array([np.nanmean(pd_.gc[pd_.family == f]) for f in order])
    return np.abs(g[:, None] - g[None, :])


def load_pfam_jsd(model: str, order: list[str]) -> np.ndarray | None:
    """Family-intrinsic Pfam-HMM-profile JSD matrix aligned to `order` (NaN for missing)."""
    tag = "evo2-gene-families" if model == "evo2" else "gpnstar-vertebrate"
    cands = sorted(Path("results").glob(f"*_{tag}/pfam_jsd_distances.csv"))
    if not cands:
        return None
    df = pd.read_csv(cands[-1], index_col=0).reindex(index=order, columns=order)
    return df.values.astype(float)


def between_homology(emb: np.ndarray, family: np.ndarray, order: list[str],
                     jsd, kmer_fam, gc_fam):
    """Spearman ρ of the between-family centroid geodesic vs Pfam-JSD / k-mer / GC.

    This is the between-family STRUCTURE metric (does the manifold arrange families by
    graded homology?) — complementary to, and distinct from, family CLASSIFIABILITY.

    The family-distance matrix is the geodesic built BETWEEN family centroids
    (``compute_centroid_geodesic``), so it is independent of per-family member counts —
    not the old mean of member-pair geodesics over a gene-level graph that a large
    family would dominate.
    """
    C = compute_centroid_geodesic(emb, family, order)
    iu = np.triu_indices(len(order), 1)
    c_u = C[iu]

    def _rho(ref):
        if ref is None:
            return float("nan"), float("nan")
        r_u = ref[iu]
        ok = np.isfinite(r_u) & np.isfinite(c_u)
        if ok.sum() < 4 or np.ptp(r_u[ok]) == 0:
            return float("nan"), float("nan")
        rho, p = spearmanr(c_u[ok], r_u[ok])
        return float(rho), float(p)

    rj, pj = _rho(jsd)
    rk, _ = _rho(kmer_fam)
    rg, _ = _rho(gc_fam)
    return rj, pj, rk, rg


# ── WITHIN-family score ──────────────────────────────────────────────────────────


def within_scores(pd_: ProblemData, geo: np.ndarray, ang: np.ndarray,
                  patristic: dict, rng: np.random.Generator):
    """(distance_correlation, subfamily_kNN_accuracy, bootstrap_stability, per_family_rho)."""
    id_pos = {g: i for i, g in enumerate(pd_.ids)}

    # distance_correlation: per-family geodesic-vs-patristic Spearman, Fisher-averaged.
    per_family: dict[str, float] = {}
    boot_family_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fam, (members, D) in patristic.items():
        gi = [id_pos[g] for g in members]
        geo_sub = geo[np.ix_(gi, gi)]
        iu = np.triu_indices(len(members), k=1)
        d_u, g_u = D[iu], geo_sub[iu]
        ok = np.isfinite(d_u) & np.isfinite(g_u)
        if ok.sum() < 6 or np.ptp(d_u[ok]) == 0:
            continue
        per_family[fam] = float(spearmanr(g_u[ok], d_u[ok])[0])
        boot_family_pairs[fam] = (np.array(gi), D)
    dist_corr = max(0.0, fisher_mean(list(per_family.values())))

    # subfamily kNN accuracy: only over families with >= 2 subfamilies (else trivially 1.0).
    sub_acc = float("nan")
    if pd_.subfamily is not None:
        accs = []
        for fam in set(pd_.family):
            idx = np.where(pd_.family == fam)[0]
            subs = pd_.subfamily[idx]
            if len(set(subs)) < 2 or len(idx) <= KNN_SUBFAMILY:
                continue
            sub_d = ang[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub_d, np.inf)
            for r in range(len(idx)):
                nn = np.argsort(sub_d[r])[:KNN_SUBFAMILY]
                accs.append(float(np.mean(subs[nn] == subs[r])))
        sub_acc = float(np.mean(accs)) if accs else float("nan")

    # bootstrap stability: resample family members (from existing geo+patristic), recompute
    # the Fisher-mean dist-corr; stability = 1 - std/|mean|, clipped to [0,1].
    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        fam_r = []
        for fam, (gi, D) in boot_family_pairs.items():
            m = len(gi)
            samp = rng.integers(0, m, m)
            gg = gi[samp]
            geo_sub = geo[np.ix_(gg, gg)]
            Dr = D[np.ix_(samp, samp)]
            iu = np.triu_indices(m, k=1)
            d_u, g_u = Dr[iu], geo_sub[iu]
            ok = np.isfinite(d_u) & np.isfinite(g_u)
            if ok.sum() >= 6 and np.ptp(d_u[ok]) > 0:
                fam_r.append(float(spearmanr(g_u[ok], d_u[ok])[0]))
        if fam_r:
            boot_rhos.append(fisher_mean(fam_r))
    if len(boot_rhos) > 1 and abs(np.mean(boot_rhos)) > 1e-6:
        stability = float(np.clip(1.0 - np.std(boot_rhos) / abs(np.mean(boot_rhos)), 0.0, 1.0))
    else:
        stability = float("nan")

    return dist_corr, sub_acc, stability, per_family


# ── BETWEEN-family score ─────────────────────────────────────────────────────────


def _cv_macro_f1(X, y, groups):
    """Macro-F1 of a LogReg family classifier under blocked (or stratified) CV."""
    classes = np.unique(y)
    if groups is not None and len(np.unique(groups)) >= N_SPLITS:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS)
        splits = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(X, y)
    f1s, nc_accs = [], []
    for tr, te in splits:
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        ).fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred, labels=classes, average="macro", zero_division=0))
        # nearest family centroid (cosine) trained on the same fold
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        cents = {c: Xn[tr][y[tr] == c].mean(0) for c in classes if (y[tr] == c).any()}
        cl = list(cents)
        C = np.stack([cents[c] for c in cl])
        C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-12, None)
        nc_pred = np.array(cl)[np.argmax(Xn[te] @ C.T, axis=1)]
        nc_accs.append(balanced_accuracy_score(y[te], nc_pred))
    return float(np.mean(f1s)), float(np.mean(nc_accs))


def _norm_clf(bal_acc: float, k: int) -> float:
    chance = 1.0 / k
    return float(np.clip((bal_acc - chance) / (1 - chance), 0.0, 1.0))


def _cv_reg_r2(X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    ybin = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    r2s = []
    for tr, te in cv.split(X, ybin):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[tr], y[tr])
        pred = reg.predict(X[te])
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - y[te].mean()) ** 2)) or 1.0
        r2s.append(1 - ss_res / ss_tot)
    return float(np.clip(np.mean(r2s), 0.0, 1.0))


def between_scores(pd_: ProblemData, emb: np.ndarray, block_groups: np.ndarray | None):
    X = PCA(n_components=min(PCA_DIM, emb.shape[0] - 1, emb.shape[1]),
            random_state=RANDOM_STATE).fit_transform(emb)
    y = pd_.family

    blocked_f1, nc_ret = _cv_macro_f1(X, y, block_groups)
    random_f1, _ = _cv_macro_f1(X, y, None)

    # nuisance: GC, length, clade leakage — each in [0,1], averaged.
    nuis = {}
    gc_ok = np.isfinite(pd_.gc)
    if gc_ok.sum() > 20:
        nuis["gc"] = _cv_reg_r2(X[gc_ok], pd_.gc[gc_ok])
    len_ok = pd_.length > 0
    if len_ok.sum() > 20:
        nuis["length"] = _cv_reg_r2(X[len_ok], np.log10(pd_.length[len_ok]))
    if pd_.clade is not None:
        ck = np.unique(pd_.clade)
        if len(ck) >= 2:
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            # collapse rare clades so stratification is valid
            vc = pd.Series(pd_.clade).value_counts()
            keep = pd_.clade if (vc >= N_SPLITS).all() else np.where(
                np.isin(pd_.clade, vc[vc >= N_SPLITS].index), pd_.clade, "other")
            accs = []
            for tr, te in cv.split(X, keep):
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000)).fit(X[tr], keep[tr])
                accs.append(balanced_accuracy_score(keep[te], clf.predict(X[te])))
            nuis["clade"] = _norm_clf(float(np.mean(accs)), len(np.unique(keep)))
    nuisance = float(np.mean(list(nuis.values()))) if nuis else 0.0
    return blocked_f1, nc_ret, nuisance, random_f1, nuis


# ── NEW two-component scoring: clustering ratio + centroid-PCA ─────────────────────


def standardize_emb(emb: np.ndarray, mode: str) -> np.ndarray:
    """Optionally rescale a layer's (N, H) embeddings before Euclidean distances/PCA.

    Raw Evo2/GPN activations vary by orders of magnitude in norm (Evo2 blows up at deep
    layers), so a raw-Euclidean ratio is dominated by a single global magnitude axis rather
    than clustering geometry. The modes remove that confound:
      * "none"   — raw embeddings (norm-sensitive; the magnitude axis dominates).
      * "zscore" — per-FEATURE standardization across genes (mean 0, std 1 per dim), so no
                   single high-variance coordinate dominates the Euclidean distance.
      * "l2"     — per-SAMPLE L2 normalization (project each gene onto the unit sphere), so
                   Euclidean distance becomes a monotone function of angular distance.
    """
    if mode == "none":
        return emb
    if mode == "l2":
        return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    if mode == "zscore":
        mu = emb.mean(axis=0, keepdims=True)
        sd = np.clip(emb.std(axis=0, keepdims=True), 1e-12, None)
        return (emb - mu) / sd
    raise ValueError(f"unknown standardize mode {mode!r}")


def clustering_ratio(emb: np.ndarray, family: np.ndarray, order: list[str]):
    """(within_dispersion, between_dispersion, within/between ratio), all Euclidean.

    within_dispersion: for each family, the MEAN PAIRWISE Euclidean distance among its
        members (member→member, not member→centroid); these per-family means are then
        averaged with EQUAL WEIGHT across families, so a large family contributes one
        number rather than O(n^2) pairs and cannot dominate. Singletons are skipped.
    between_dispersion: mean pairwise Euclidean distance between family CENTROIDS (the
        raw mean embedding of each family), so every family contributes exactly one node
        regardless of size.
    A lower ratio = tighter clusters relative to their separation.
    """
    idx = {f: np.where(family == f)[0] for f in order}
    within_per_fam = [float(pdist(emb[idx[f]]).mean()) for f in order if len(idx[f]) >= 2]
    within = float(np.mean(within_per_fam)) if within_per_fam else float("nan")
    cents = np.vstack([emb[idx[f]].mean(axis=0) for f in order])
    between = float(pdist(cents).mean()) if len(order) >= 2 else float("nan")
    ratio = within / between if between and np.isfinite(between) else float("nan")
    return within, between, ratio


def family_pca_variance(emb: np.ndarray, family: np.ndarray, order: list[str],
                        n_components: int = 3):
    """Per-family PCA: top-`n_components` explained-variance fraction WITHIN each family.

    For every family we stack its member-gene embeddings into (n_f, H), fit a PCA on that
    family's own gene cloud (sklearn centers it on the family mean — the "average of their
    genes"), and take the fraction of the family's variance captured by its top 3 PCs. This
    is a per-family intrinsic-dimensionality readout, NOT a single PCA over the F centroids.

    Returns (per_family: {family -> top3_fraction}, mean_top3, (mean_pc1, mean_pc2, mean_pc3))
    where the means are EQUAL-WEIGHT across families (matching the clustering ratio), so a
    large family does not dominate. Families with < 2 members (no variance) are skipped; note
    a family with n_f members yields only min(n_f-1, H) PCs, so families with <= 4 members are
    trivially ~1.0 (their gene cloud spans <= 3 dims) — interpret small families with care.
    """
    per_family: dict[str, float] = {}
    pcs: list[list[float]] = []
    for f in order:
        idx = np.where(family == f)[0]
        if len(idx) < 2:
            continue
        Xf = emb[idx]
        n_comp = min(n_components, Xf.shape[0] - 1, Xf.shape[1])
        if n_comp < 1:
            continue
        evr = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit(Xf).explained_variance_ratio_
        evr3 = list(evr) + [0.0] * (3 - len(evr))
        pcs.append([float(evr3[0]), float(evr3[1]), float(evr3[2])])
        per_family[f] = float(sum(evr[:3]))
    if not pcs:
        return {}, float("nan"), (float("nan"), float("nan"), float("nan"))
    mean_pc = np.mean(pcs, axis=0)
    return per_family, float(mean_pc.sum()), (float(mean_pc[0]), float(mean_pc[1]), float(mean_pc[2]))


# ── driver ───────────────────────────────────────────────────────────────────────


def augment_homology(model: str, gpn_model: str) -> None:
    """Add the between-family homology columns to an EXISTING layer_scores CSV.

    Recomputes only the per-layer geodesic (from cached embeddings) — no re-embedding,
    no patristic re-alignment, no probes. Used to retrofit the homology metric onto a
    run that predates it. Geodesics are cached so this is paid only once.
    """
    pd_ = load_evo2() if model == "evo2" else load_gpnstar(gpn_model)
    cands = sorted(Path("results").glob(f"*_layer-selection/layer_scores_{model}.csv"))
    if not cands:
        sys.exit(f"No existing layer_scores_{model}.csv to augment.")
    csv = cands[-1]
    df = pd.read_csv(csv)
    family_order = sorted(set(pd_.family))
    jsd = load_pfam_jsd(model, family_order)
    kmer_fam = family_kmer_distance(pd_, family_order)
    gc_fam = family_gc_distance(pd_, family_order)
    geo_cache = Path(f"data/cache/{model}_layer_geodesics")
    geo_cache.mkdir(parents=True, exist_ok=True)
    print(f"Augmenting {csv} (Pfam-JSD {'loaded' if jsd is not None else 'MISSING'})")
    cols = {c: [] for c in ["between_homology_pfamjsd_rho", "between_homology_pfamjsd_p",
                            "between_kmer_rho", "between_gc_rho"]}
    for li in range(pd_.stack.shape[0]):
        gp = geo_cache / f"geo_{li}.npy"
        if gp.exists():
            geo = np.load(gp)
        else:
            _, W = lowest_connected_k(pd_.stack[li])
            geo = compute_geodesic(W)
            np.save(gp, geo)
        rj, pj, rk, rg = between_homology(pd_.stack[li], pd_.family, family_order, jsd, kmer_fam, gc_fam)
        cols["between_homology_pfamjsd_rho"].append(round(rj, 4) if np.isfinite(rj) else np.nan)
        cols["between_homology_pfamjsd_p"].append(round(pj, 4) if np.isfinite(pj) else np.nan)
        cols["between_kmer_rho"].append(round(rk, 4) if np.isfinite(rk) else np.nan)
        cols["between_gc_rho"].append(round(rg, 4) if np.isfinite(rg) else np.nan)
        print(f"  L{li:>2} {pd_.layer_labels[li]:<22} ρ(geo,Pfam-JSD)={cols['between_homology_pfamjsd_rho'][-1]} "
              f"(k-mer ctrl {cols['between_kmer_rho'][-1]})")
    for c, v in cols.items():
        df[c] = v
    df.to_csv(csv, index=False)
    best = df["between_homology_pfamjsd_rho"].idxmax() if df["between_homology_pfamjsd_rho"].notna().any() else None
    if best is not None:
        print(f"\nBest between-family HOMOLOGY: L{best} {df['layer'][best]} "
              f"ρ={df['between_homology_pfamjsd_rho'][best]:+.3f} p={df['between_homology_pfamjsd_p'][best]:.3f}")
    print(f"Saved {csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["evo2", "gpnstar"])
    p.add_argument("--gpn-model", default="vertebrate")
    p.add_argument("--panel", default="default", choices=["default", "human"],
                   help="'default' = the model's primary sweep cache (Evo2 cross-kingdom / GPN "
                        "families); 'human' = the matched-manifold human Panel-1 sweep caches "
                        "(evo2_human_layer_sweep / gpnstar_human_layer_sweep). Writes "
                        "layer_scores_<model>_human[_<mode>].csv.")
    p.add_argument("--standardize", default="none", choices=["none", "zscore", "l2"],
                   help="Rescale each layer's embeddings before the Euclidean ratio + PCA: "
                        "'none' (raw, norm-sensitive), 'zscore' (per-feature), or 'l2' "
                        "(per-sample unit-norm). Non-'none' writes layer_scores_<model>_<mode>.csv.")
    p.add_argument("--augment", action="store_true",
                   help="Only add the between-family homology columns to the existing CSV "
                        "(recompute geodesics from cached embeddings; no re-embed/probes).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.augment:
        augment_homology(args.model, args.gpn_model)
        return
    rng = np.random.default_rng(RANDOM_STATE)  # noqa: F841 (used by the commented-out scoring)
    print(f"[1] Loading {args.model} layer stack (panel={args.panel})")
    if args.panel == "human":
        pd_ = load_evo2_human() if args.model == "evo2" else load_gpnstar_human(args.gpn_model)
    else:
        pd_ = load_evo2() if args.model == "evo2" else load_gpnstar(args.gpn_model)
    L, N, H = pd_.stack.shape
    print(f"  stack {pd_.stack.shape} | {N} genes | {len(set(pd_.family))} families | dim {H}")
    family_order = sorted(set(pd_.family))

    # ── OLD composite scoring (patristic / blocked-CV probe / Pfam-JSD homology) ──────
    # Preserved for reference but no longer run; the benchmark now reports the two
    # clustering components below. Re-enable this block to restore the old deliverable.
    #
    # print("[2] Patristic ground truth (once)")
    # patristic = build_patristic(pd_)
    # print(f"  {len(patristic)} families with usable patristic trees")
    # if pd_.clade is not None:
    #     block_groups = pd_.clade
    #     block_kind = "clade"
    # else:
    #     print("[2b] No clade axis → building within-family seq-id clusters for blocking")
    #     block_groups = build_seqid_clusters(pd_)
    #     block_kind = "seqid-cluster"
    # print(f"  blocked split by: {block_kind} ({len(np.unique(block_groups))} blocks)")
    # jsd = load_pfam_jsd(args.model, family_order)
    # kmer_fam = family_kmer_distance(pd_, family_order)
    # gc_fam = family_gc_distance(pd_, family_order)
    # geo_cache = Path(f"data/cache/{args.model}_layer_geodesics")
    # geo_cache.mkdir(parents=True, exist_ok=True)

    print(f"[2] Scoring {L} layers (within/between clustering ratio + per-family PCA; "
          f"standardize={args.standardize})")
    rows = []
    per_family_pca: list[dict[str, float]] = []  # per-layer {family -> top3 fraction}
    for li in range(L):
        emb = standardize_emb(pd_.stack[li], args.standardize)
        within, between, ratio = clustering_ratio(emb, pd_.family, family_order)
        fam_top3, top3, (pc1, pc2, pc3) = family_pca_variance(emb, pd_.family, family_order)
        per_family_pca.append(fam_top3)

        # ── OLD per-layer composite scoring (disabled; see note above) ────────────
        # k, W = lowest_connected_k(emb)
        # geo = compute_geodesic(W)
        # np.save(geo_cache / f"geo_{li}.npy", geo)
        # ang = angular_matrix(emb)
        # dist_corr, sub_acc, stability, _ = within_scores(pd_, geo, ang, patristic, rng)
        # bl_f1, nc_ret, nuisance, rand_f1, _ = between_scores(pd_, emb, block_groups)
        # bh_jsd, bh_jsd_p, bh_kmer, bh_gc = between_homology(
        #     emb, pd_.family, family_order, jsd, kmer_fam, gc_fam)
        # if np.isfinite(sub_acc):
        #     s_within = 0.4 * dist_corr + 0.4 * sub_acc + 0.2 * (stability if np.isfinite(stability) else 0)
        # else:
        #     s_within = (0.4 * dist_corr + 0.2 * (stability if np.isfinite(stability) else 0)) / 0.6
        # s_between = 0.5 * bl_f1 + 0.3 * nc_ret - 0.2 * nuisance

        rows.append({
            "model": args.model,
            "standardize": args.standardize,
            "layer_idx": li,
            "layer": pd_.layer_labels[li],
            "within_dispersion": round(within, 4),
            "between_dispersion": round(between, 4),
            "within_between_ratio": round(ratio, 4) if np.isfinite(ratio) else np.nan,
            "pca_family_pc1_mean": round(pc1, 4) if np.isfinite(pc1) else np.nan,
            "pca_family_pc2_mean": round(pc2, 4) if np.isfinite(pc2) else np.nan,
            "pca_family_pc3_mean": round(pc3, 4) if np.isfinite(pc3) else np.nan,
            "pca_family_top3_mean": round(top3, 4) if np.isfinite(top3) else np.nan,
        })
        r = rows[-1]
        print(f"  L{li:>2} {pd_.layer_labels[li]:<22} "
              f"within/between={r['within_between_ratio']:.4f} "
              f"(within={r['within_dispersion']:.3f} between={r['between_dispersion']:.3f}) | "
              f"per-family PCA top3 (mean)={r['pca_family_top3_mean']:.3f} "
              f"(PC1={r['pca_family_pc1_mean']:.3f} PC2={r['pca_family_pc2_mean']:.3f} PC3={r['pca_family_pc3_mean']:.3f})")

    df = pd.DataFrame(rows)
    # Lower within/between ratio = tighter, better-separated family clusters.
    best = df["within_between_ratio"].idxmin()
    use = ["" for _ in range(len(df))]
    use[best] = "best-clustering"
    df["recommended_use"] = use

    date = datetime.date.today().isoformat()
    out_dir = Path("results") / f"{date}_layer-selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_sfx = "_human" if args.panel == "human" else ""
    suffix = "" if args.standardize == "none" else f"_{args.standardize}"
    out_csv = out_dir / f"layer_scores_{args.model}{panel_sfx}{suffix}.csv"
    df.to_csv(out_csv, index=False)

    # Per-family PCA top-3 fraction at the recommended layer (one row per family, with member
    # count so trivially-low-rank small families are flagged).
    fam_counts = {f: int(np.sum(pd_.family == f)) for f in family_order}
    fam_df = pd.DataFrame(
        [{"family": f, "n_members": fam_counts[f], "pca_top3": round(v, 4)}
         for f, v in sorted(per_family_pca[best].items(), key=lambda kv: -kv[1])]
    )
    fam_csv = out_dir / f"family_pca_{args.model}{panel_sfx}{suffix}.csv"
    fam_df.to_csv(fam_csv, index=False)

    print(f"\n[3] Saved {out_csv}")
    print(f"  Best CLUSTERING (min within/between ratio): L{best} {df['layer'][best]}  "
          f"ratio={df['within_between_ratio'][best]:.4f}")
    print(f"  Per-family PCA top-3 (mean over families) at that layer: {df['pca_family_top3_mean'][best]:.3f}")
    print(f"  Saved per-family PCA breakdown → {fam_csv}")


if __name__ == "__main__":
    main()
