"""Validation tests for the centroid-free OT between-family metrics (ot_between_family).

Fast + embedding-free: builds a small synthetic multi-family panel (incl. a 2-gene tiny
family) so the whole suite runs in <1s without any cached model embeddings. Covers the
invariants the design requires: symmetry, finite/nonnegative values, ~zero diagonals,
gene-order invariance, deterministic repeatability, and the FGW(alpha=0)==W2^2 solver
sanity check. Also exercises the entropic solver, the tiny-family flag, and the subsample
sensitivity pass.

Run:  uv run pytest scripts/baselines/test_ot_between_family.py -q
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "baselines"))
from ot_between_family import (  # noqa: E402
    TINY_FAMILY_MAX,
    compute_ot_matrices,
    subsample_sensitivity,
    validate,
)


def _synthetic(seed: int = 0):
    """Four Gaussian family clouds of unequal size (one tiny 2-gene family)."""
    rng = np.random.default_rng(seed)
    fam_order = ["big", "mid", "small", "tiny"]
    sizes = {"big": 50, "mid": 20, "small": 8, "tiny": 2}
    embs, fams = [], []
    for f in fam_order:
        center = rng.standard_normal(24)
        embs.append(center + 0.4 * rng.standard_normal((sizes[f], 24)))
        fams += [f] * sizes[f]
    return np.vstack(embs), np.array(fams), fam_order


@pytest.fixture(scope="module")
def panel():
    emb, fams, fam_order = _synthetic()
    res = compute_ot_matrices(emb, fams, fam_order)
    return emb, fams, fam_order, res


def test_all_invariants(panel):
    """symmetry / finite / nonneg / zero-diag / determinism / order-invariance / alpha0."""
    emb, fams, fam_order, res = panel
    checks = validate(res, emb, fams, fam_order, verbose=False)
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, f"failed invariants: {failed}"


def test_matrix_keys(panel):
    _, _, _, res = panel
    assert "wasserstein" in res.matrices
    assert {"fgw_alpha0.25", "fgw_alpha0.50", "fgw_alpha0.75"} <= set(res.matrices)


def test_tiny_family_flagged(panel):
    _, _, _, res = panel
    assert "tiny" in res.meta["tiny_families"]
    assert all(res.meta["family_sizes"][f] <= TINY_FAMILY_MAX for f in res.meta["tiny_families"])


def test_uniform_mass_independent_of_size(panel):
    """Duplicating every gene in a family (doubling n_f) leaves the uniform-weight OT
    distances unchanged — mass is 1 per family regardless of size."""
    emb, fams, fam_order, res = panel
    dup_idx = np.arange(len(emb))
    dup = np.concatenate([dup_idx, np.where(fams == "mid")[0]])  # double the 'mid' family
    res_dup = compute_ot_matrices(emb[dup], fams[dup], fam_order)
    assert np.allclose(res.matrices["wasserstein"], res_dup.matrices["wasserstein"], atol=1e-6)


def test_entropic_solver_converges():
    """Entropic W2 records reg + small marginal residuals and stays close to exact."""
    emb, fams, fam_order = _synthetic()
    exact = compute_ot_matrices(emb, fams, fam_order)
    ent = compute_ot_matrices(emb, fams, fam_order, solver="entropic", reg=0.01)
    conv = ent.meta["convergence"]["wasserstein"]
    for pair_info in conv.values():
        assert pair_info["marginal_residual_row"] < 1e-3
        assert pair_info["marginal_residual_col"] < 1e-3
    # regularized W2 should track the exact ordering loosely (not identical, not silently mixed)
    assert ent.meta["solver"]["kind"] == "entropic"
    assert np.allclose(exact.matrices["wasserstein"], ent.matrices["wasserstein"], atol=0.05)


def test_fgw_init_stability_recorded():
    """With n_init>1 the metadata carries an init_spread per FGW pair (stability check)."""
    emb, fams, fam_order = _synthetic()
    res = compute_ot_matrices(emb, fams, fam_order, fgw_n_init=3)
    conv = res.meta["convergence"]["fgw_alpha0.50"]
    assert all("init_spread" in info for info in conv.values())
    assert max(info["init_spread"] for info in conv.values()) < 0.1  # small on clean clouds


def test_subsample_sensitivity_shape():
    emb, fams, fam_order = _synthetic()
    out = subsample_sensitivity(emb, fams, fam_order, size=2, n_repeats=5)
    assert out["size_per_family"] == 2
    F = len(fam_order)
    for stats in out["matrices"].values():
        assert stats["mean"].shape == (F, F)
        assert stats["std"].shape == (F, F)
