"""Centroid-free between-family proximity via optimal transport (POT / ``ot``)."""

from __future__ import annotations
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import ot

DEFAULT_ALPHAS: tuple[float, ...] = (0.25, 0.50, 0.75)
# Human families this small have too few genes to resolve within-family geometry, so the
# FGW quadratic (GW) term is poorly estimated for any pair that involves them. Flagged in
# the metadata (not dropped) so downstream reporting can caveat those comparisons.
TINY_FAMILY_MAX = 3


def _alpha_key(alpha: float) -> str:
    """Stable matrix key for an alpha value: fgw_alpha0.25, fgw_alpha0.50, ..."""
    return f"fgw_alpha{alpha:.2f}"


@dataclass
class OTResult:
    """Container for the centroid-free family-distance matrices + full provenance."""

    fam_order: list[str]
    matrices: dict[str, np.ndarray]  # name -> F×F symmetric distance matrix
    meta: dict = field(default_factory=dict)  # sizes, preprocessing, solver, convergence…

    def to_run_dir(self, run_dir: Path, prefix: str = "betweenfam_ot") -> None:
        """Write each matrix as a labelled F×F CSV plus one metadata JSON into ``run_dir``."""
        import pandas as pd

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, D in self.matrices.items():
            pd.DataFrame(D, index=self.fam_order, columns=self.fam_order).to_csv(
                run_dir / f"{prefix}_{name}_distances.csv"
            )
        (run_dir / f"{prefix}_metadata.json").write_text(json.dumps(self.meta, indent=2))


# ── preprocessing / cost


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (unit gene vectors), matching the centroid analysis."""
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def angular_distance(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """theta(x, y) = arccos(clip(cos_sim, -1, 1)) / pi for L2-normalized U (n×d), V (m×d)."""
    cos = np.clip(U @ V.T, -1.0, 1.0)
    return np.arccos(cos) / np.pi


def _preprocess(emb: np.ndarray, standardize: bool) -> np.ndarray:
    """Global preprocessing fitted ONCE across all genes, then per-gene L2-normalize."""
    emb = np.asarray(emb, dtype=np.float64)
    if standardize:
        mu = emb.mean(axis=0, keepdims=True)
        sd = np.clip(emb.std(axis=0, keepdims=True), 1e-12, None)
        emb = (emb - mu) / sd
    return l2_normalize(emb)


# ── solvers


def _wasserstein(
    M: np.ndarray, p: np.ndarray, q: np.ndarray, *, solver: str, reg: float | None
) -> tuple[float, dict]:
    """W2 between two families given squared-angular cost M. Returns (w2, info)."""
    if solver == "exact":
        w2sq, log = ot.emd2(p, q, M, log=True)
        info = {
            "solver": "emd2",
            "result_code": log.get("result_code"),
            "warning": log.get("warning"),
        }
    elif solver == "entropic":
        if reg is None:
            raise ValueError("entropic solver requires reg (regularization strength)")
        T = ot.sinkhorn(p, q, M, reg)
        w2sq = float(np.sum(T * M))
        info = {
            "solver": "sinkhorn",
            "reg": reg,
            "marginal_residual_row": float(np.abs(T.sum(1) - p).max()),
            "marginal_residual_col": float(np.abs(T.sum(0) - q).max()),
        }
    else:
        raise ValueError(f"unknown solver {solver!r} (want 'exact' or 'entropic')")
    w2 = float(np.sqrt(max(float(w2sq), 0.0)))
    info["w2_squared"] = float(w2sq)
    return w2, info


def _fgw(
    M: np.ndarray,
    C_f: np.ndarray,
    C_g: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
    *,
    solver: str,
    reg: float | None,
    n_init: int,
    seed: int,
    max_iter: int,
) -> tuple[float, dict]:
    """FGW distance for one alpha. Returns (fgw_value, convergence-info)."""
    vals, infos = [], []
    for i in range(max(1, n_init)):
        G0 = None
        if i > 0:  # first init = POT default (product coupling, deterministic); rest random
            rng = np.random.default_rng(seed + i)
            G0 = _random_coupling(p, q, rng)
        if solver == "exact":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # quad_loss/alpha divide-by-zero at alpha=0
                val, log = ot.gromov.fused_gromov_wasserstein2(
                    M,
                    C_f,
                    C_g,
                    p=p,
                    q=q,
                    loss_fun="square_loss",
                    alpha=alpha,
                    G0=G0,
                    log=True,
                    max_iter=max_iter,
                )
            infos.append(
                {
                    "result_code": log.get("result_code"),
                    "warning": log.get("warning"),
                    "n_cg_iter": len(log.get("loss", [])),
                    "quad_loss": float(log.get("quad_loss", np.nan)),
                    "lin_loss": float(log.get("lin_loss", np.nan)),
                }
            )
        elif solver == "entropic":
            if reg is None:
                raise ValueError("entropic FGW requires reg")
            val, log = ot.gromov.entropic_fused_gromov_wasserstein2(
                M,
                C_f,
                C_g,
                p=p,
                q=q,
                loss_fun="square_loss",
                alpha=alpha,
                epsilon=reg,
                G0=G0,
                log=True,
            )
            T = log.get("T")
            infos.append(
                {
                    "reg": reg,
                    "marginal_residual_row": float(np.abs(T.sum(1) - p).max())
                    if T is not None
                    else np.nan,
                    "marginal_residual_col": float(np.abs(T.sum(0) - q).max())
                    if T is not None
                    else np.nan,
                }
            )
        else:
            raise ValueError(f"unknown solver {solver!r}")
        vals.append(float(val))
    best = int(np.argmin(vals))
    info = dict(infos[best])
    info["n_init"] = len(vals)
    if len(vals) > 1:
        info["init_spread"] = float(np.ptp(vals))  # >0 flags a non-convex / unstable pair
    # FGW value can dip slightly below 0 from CG float noise; clamp for a valid distance.
    return max(vals[best], 0.0), info


def _random_coupling(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random transport plan with the correct marginals (Sinkhorn-projected)."""
    K = rng.random((len(p), len(q))) + 1e-3
    for _ in range(50):  # a few Sinkhorn iterations to match marginals p, q
        K *= (p / np.clip(K.sum(1), 1e-12, None))[:, None]
        K *= (q / np.clip(K.sum(0), 1e-12, None))[None, :]
    return K


# ── main entry point


def compute_ot_matrices(
    emb: np.ndarray,
    families: np.ndarray,
    fam_order: list[str],
    *,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    solver: str = "exact",
    reg: float | None = None,
    standardize: bool = False,
    fgw_n_init: int = 1,
    fgw_seed: int = 0,
    fgw_max_iter: int = 10000,
) -> OTResult:
    """Centroid-free family-distance matrices (Wasserstein + FGW per alpha)."""
    t0 = time.perf_counter()
    U = _preprocess(emb, standardize)
    families = np.asarray(families)
    idx = {f: np.where(families == f)[0] for f in fam_order}
    sizes = {f: int(len(idx[f])) for f in fam_order}
    for f in fam_order:
        if sizes[f] == 0:
            raise ValueError(f"family {f!r} has no members in the embedding")

    unit = {f: U[idx[f]] for f in fam_order}
    weights = {f: np.full(sizes[f], 1.0 / sizes[f]) for f in fam_order}
    # Within-family angular matrices C_f = A_ff (cached; reused across every pair + alpha).
    C = {f: angular_distance(unit[f], unit[f]) for f in fam_order}

    F = len(fam_order)
    W = np.zeros((F, F))
    FGW = {a: np.zeros((F, F)) for a in alphas}
    w_info: dict = {}
    fgw_info: dict = {a: {} for a in alphas}

    for i in range(F):
        for j in range(i + 1, F):
            fa, fb = fam_order[i], fam_order[j]
            A_fg = angular_distance(unit[fa], unit[fb])  # cross-family angular, n_fa×n_fb
            M = A_fg**2  # squared angular cross cost
            p, q = weights[fa], weights[fb]
            w2, winf = _wasserstein(M, p, q, solver=solver, reg=reg)
            W[i, j] = W[j, i] = w2
            w_info[f"{fa}|{fb}"] = winf
            for a in alphas:
                val, finf = _fgw(
                    M,
                    C[fa],
                    C[fb],
                    p,
                    q,
                    a,
                    solver=solver,
                    reg=reg,
                    n_init=fgw_n_init,
                    seed=fgw_seed,
                    max_iter=fgw_max_iter,
                )
                FGW[a][i, j] = FGW[a][j, i] = val
                fgw_info[a][f"{fa}|{fb}"] = finf

    matrices = {"wasserstein": W}
    for a in alphas:
        matrices[_alpha_key(a)] = FGW[a]

    tiny = [f for f in fam_order if sizes[f] <= TINY_FAMILY_MAX]
    meta = {
        "metric": "ot_between_family",
        "pot_version": ot.__version__,
        "fam_order": list(fam_order),
        "family_sizes": sizes,
        "n_genes": int(len(U)),
        "preprocessing": {
            "l2_normalize": True,
            "standardize": bool(standardize),
            "cost": "theta = arccos(clip(cos_sim,-1,1))/pi; wasserstein cost = theta**2; "
            "FGW cross cost M = theta**2, structure C_f = theta (A_ff)",
            "weights": "uniform within family (p_i = 1/n_f); total mass 1 per family",
        },
        "solver": {
            "kind": solver,
            "reg": reg,
            "fgw_n_init": fgw_n_init,
            "fgw_seed": fgw_seed,
            "fgw_max_iter": fgw_max_iter,
        },
        "alphas": list(alphas),
        "convergence": {"wasserstein": w_info, **{_alpha_key(a): fgw_info[a] for a in alphas}},
        "tiny_families": tiny,
        "tiny_family_note": (
            f"families with <= {TINY_FAMILY_MAX} genes have poorly resolved within-family "
            f"geometry; FGW comparisons involving them (its quadratic term) are unreliable."
        ),
        "runtime_seconds": round(time.perf_counter() - t0, 3),
    }
    return OTResult(fam_order=list(fam_order), matrices=matrices, meta=meta)


# ── optional sensitivity analysis


def subsample_sensitivity(
    emb: np.ndarray,
    families: np.ndarray,
    fam_order: list[str],
    *,
    size: int | None = None,
    n_repeats: int = 20,
    seed: int = 0,
    **kwargs,
) -> dict:
    """Repeated equal-size subsample robustness check (NOT the primary result)."""
    families = np.asarray(families)
    idx = {f: np.where(families == f)[0] for f in fam_order}
    if size is None:
        size = min(len(idx[f]) for f in fam_order)
    rng = np.random.default_rng(seed)
    stacks: dict[str, list[np.ndarray]] = {}
    for _ in range(n_repeats):
        pick = np.concatenate(
            [rng.choice(idx[f], size=size, replace=len(idx[f]) < size) for f in fam_order]
        )
        sub_fams = families[pick]
        res = compute_ot_matrices(emb[pick], sub_fams, fam_order, **kwargs)
        for name, D in res.matrices.items():
            stacks.setdefault(name, []).append(D)
    out = {}
    for name, mats in stacks.items():
        arr = np.stack(mats)
        out[name] = {"mean": arr.mean(0), "std": arr.std(0)}
    return {"size_per_family": size, "n_repeats": n_repeats, "matrices": out}


# ── validation (imported by the test module + runnable via --self-test)


def _solver_kwargs(meta: dict) -> dict:
    """Reconstruct the compute_ot_matrices solver kwargs from a result's metadata, so a
    validation re-run reproduces the same numerics (n_init, seed, solver, reg)."""
    s = meta["solver"]
    return dict(
        alphas=tuple(meta["alphas"]),
        solver=s["kind"],
        reg=s["reg"],
        standardize=meta["preprocessing"]["standardize"],
        fgw_n_init=s["fgw_n_init"],
        fgw_seed=s["fgw_seed"],
        fgw_max_iter=s["fgw_max_iter"],
    )


def validate(
    res: OTResult,
    emb: np.ndarray,
    families: np.ndarray,
    fam_order: list[str],
    *,
    atol_diag: float = 1e-9,
    verbose: bool = True,
) -> dict:
    """Structural + solver-sanity checks."""
    checks: dict[str, bool] = {}
    kw = _solver_kwargs(res.meta)

    def _log(name, ok):
        checks[name] = bool(ok)
        if verbose:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")

    for name, D in res.matrices.items():
        _log(f"{name}: symmetric", np.allclose(D, D.T, atol=1e-10))
        _log(f"{name}: finite", np.isfinite(D).all())
        _log(f"{name}: nonnegative", (D >= -1e-12).all())
        _log(f"{name}: ~zero diagonal", np.allclose(np.diag(D), 0.0, atol=atol_diag))

    # Determinism: identical inputs + identical settings -> identical matrices.
    res2 = compute_ot_matrices(emb, families, fam_order, **kw)
    for name in res.matrices:
        _log(
            f"{name}: deterministic repeat",
            np.allclose(res.matrices[name], res2.matrices[name], atol=1e-9),
        )

    # Invariance to gene ordering: permute rows -> same family matrices. Asserted only for
    # the deterministic FGW path (n_init=1); with restarts we skip (see docstring).
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(emb))
    res3 = compute_ot_matrices(np.asarray(emb)[perm], np.asarray(families)[perm], fam_order, **kw)
    for name in res.matrices:
        if name.startswith("fgw") and kw["fgw_n_init"] > 1:
            continue  # random restarts are arrangement-dependent by construction
        _log(
            f"{name}: gene-order invariant",
            np.allclose(res.matrices[name], res3.matrices[name], atol=1e-6),
        )

    # Solver sanity: FGW at alpha=0 reproduces the Wasserstein-squared objective.
    U = _preprocess(emb, res.meta["preprocessing"]["standardize"])
    fams = np.asarray(families)
    idx = {f: np.where(fams == f)[0] for f in fam_order}
    unit = {f: U[idx[f]] for f in fam_order}
    max_diff = 0.0
    for i in range(len(fam_order)):
        for j in range(i + 1, len(fam_order)):
            fa, fb = fam_order[i], fam_order[j]
            A = angular_distance(unit[fa], unit[fb])
            M = A**2
            p = np.full(len(idx[fa]), 1 / len(idx[fa]))
            q = np.full(len(idx[fb]), 1 / len(idx[fb]))
            Cf, Cg = angular_distance(unit[fa], unit[fa]), angular_distance(unit[fb], unit[fb])
            fgw0 = ot.gromov.fused_gromov_wasserstein2(
                M, Cf, Cg, p=p, q=q, loss_fun="square_loss", alpha=0.0
            )
            w2sq = ot.emd2(p, q, M)
            max_diff = max(max_diff, abs(float(fgw0) - float(w2sq)))
    _log(f"FGW(alpha=0) == W2^2 (max diff {max_diff:.2e})", max_diff < 1e-6)

    if verbose:
        n_fail = sum(not v for v in checks.values())
        print(f"  {'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} ({len(checks)} checks)")
    return checks


def _self_test() -> None:
    """Run validate() on a small synthetic panel with a mix of family sizes."""
    rng = np.random.default_rng(0)
    fam_order = ["big", "mid", "small", "tiny"]
    sizes = {"big": 60, "mid": 25, "small": 8, "tiny": 2}
    centers = {f: rng.standard_normal(32) for f in fam_order}
    embs, fams = [], []
    for f in fam_order:
        embs.append(centers[f] + 0.4 * rng.standard_normal((sizes[f], 32)))
        fams += [f] * sizes[f]
    emb = np.vstack(embs)
    fams = np.array(fams)
    print(f"compute_ot_matrices on synthetic 4-family panel (sizes {sizes}) ...")
    # Primary path (n_init=1): deterministic + order-invariant, all invariants asserted.
    res = compute_ot_matrices(emb, fams, fam_order)
    print(f"  runtime {res.meta['runtime_seconds']}s, tiny families {res.meta['tiny_families']}")
    checks = validate(res, emb, fams, fam_order)
    assert all(checks.values()), "self-test validation failed"

    # Separate diagnostic (NOT asserted): FGW stability across initializations.
    print("FGW multi-init stability (init_spread per pair; >0 = local-optimum sensitivity):")
    res_ms = compute_ot_matrices(emb, fams, fam_order, fgw_n_init=3)
    spreads = [
        inf.get("init_spread", 0.0)
        for a in res_ms.meta["alphas"]
        for inf in res_ms.meta["convergence"][_alpha_key(a)].values()
    ]
    print(f"  max init_spread across all FGW pairs/alphas = {max(spreads):.2e}")
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run synthetic validation")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    else:
        ap.print_help()
