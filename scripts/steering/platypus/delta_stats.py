"""Stage 2 — is there a shared human->platypus direction in the paired prefix activations?"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

N_SPLIT = 1000
N_PERM = 2000
SEED = 42


# --------------------------------------------------------------------------- Gram-based primitives
def _cos_from_grams(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """cos from squared norms and inner products, NaN where either vector is degenerate."""
    den = np.sqrt(np.maximum(xx, 0.0) * np.maximum(yy, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, xy / den, np.nan)
    return out


def loo_alignment(DD: np.ndarray, g: np.ndarray, ss: float, n: int) -> np.ndarray:
    """a_i = cos(D_i, v_-i) with v_-i = (S - D_i)/(n-1)."""
    dd = np.diag(DD)
    num = g - dd
    yy = ss - 2.0 * g + dd
    return _cos_from_grams(dd, yy, num)


def coherence(UU: np.ndarray, signs: np.ndarray | None = None) -> np.ndarray:
    """C = ||mean_i s_i u_i|| from the unit-vector Gram UU. `signs` is (P,N) of +-1, or None for +1."""
    n = UU.shape[0]
    if signs is None:
        return np.sqrt(max(UU.sum(), 0.0)) / n
    # ||sum_i s_i u_i||^2 = s^T UU s
    return np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", signs, UU, signs), 0.0)) / n


def _pair_cos(DD: np.ndarray, WA: np.ndarray, WB: np.ndarray) -> np.ndarray:
    """cos(v_A, v_B) for two sets of weight rows (split halves)."""
    num = np.einsum("pi,ij,pj->p", WA, DD, WB)
    xa = np.einsum("pi,ij,pj->p", WA, DD, WA)
    xb = np.einsum("pi,ij,pj->p", WB, DD, WB)
    return _cos_from_grams(xa, xb, num)


def _pval(real: float, null: np.ndarray, greater: bool = True) -> float:
    """Repo convention: (count_at_least_as_extreme + 1) / (n + 1)."""
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(real):
        return float("nan")
    k = np.sum(null >= real) if greater else np.sum(null <= real)
    return float((k + 1) / (null.size + 1))


# --------------------------------------------------------------------------- per-layer analysis
def analyse_layer(Hm: np.ndarray, Pm: np.ndarray, families: np.ndarray, rng: np.random.Generator):
    """All Stage-2 statistics for one layer. Hm/Pm are (N,H) float64."""
    n = Hm.shape[0]
    D = Pm - Hm
    dnorm = np.linalg.norm(D, axis=1)
    S = D.sum(axis=0)
    v = S / n
    vnorm = float(np.linalg.norm(v))

    DD = D @ D.T
    g = D @ S
    ss = float(S @ S)

    # unit-vector Gram for coherence; genes with a zero delta cannot be normalised
    ok = dnorm > 0
    U = np.zeros_like(D)
    U[ok] = D[ok] / dnorm[ok, None]
    UU = U @ U.T

    a = loo_alignment(DD, g, ss, n)
    C = float(coherence(UU))

    # influence of each gene on the mean: I_i = 1 - cos(v, v_-i)
    dd = np.diag(DD)
    infl = 1.0 - _cos_from_grams(np.full(n, ss), ss - 2.0 * g + dd, ss - g)

    # ---- split-half stability ------------------------------------------------
    half = n // 2
    WA = np.zeros((N_SPLIT, n))
    WB = np.zeros((N_SPLIT, n))
    for b in range(N_SPLIT):
        perm = rng.permutation(n)
        WA[b, perm[:half]] = 1.0
        WB[b, perm[half:2 * half]] = 1.0
    s_split = _pair_cos(DD, WA, WB)

    # ---- null 1: sign flip ---------------------------------------------------
    eps = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    # S_eps = eps @ D ; <D_i^eps, S_eps> = eps_i * (eps @ DD)_i ; ||D_i^eps||^2 = dd_i
    E = eps @ DD                       # (P,N) = sum_j eps_j DD[i,j]
    g_eps = eps * E                    # <eps_i D_i, S_eps>
    ss_eps = np.einsum("pi,pi->p", eps, E)
    yy = ss_eps[:, None] - 2.0 * g_eps + dd[None, :]
    a_sf = _cos_from_grams(np.broadcast_to(dd, yy.shape), yy, g_eps - dd[None, :])
    C_sf = coherence(UU, eps)

    # ---- null 2: mismatched pairs (platypus permuted) ------------------------
    PP, HH, PH = Pm @ Pm.T, Hm @ Hm.T, Pm @ Hm.T
    pS, hS = Pm @ S, Hm @ S            # for <D~_i, S>: S is invariant under the permutation
    a_mm, C_mm = _mismatch_null(PP, HH, PH, pS, hS, ss, families, rng, within_family=True)
    a_mm_pool, C_mm_pool = _mismatch_null(PP, HH, PH, pS, hS, ss, families, rng, within_family=False)

    return {
        "delta": D, "dnorm": dnorm, "v": v, "a": a, "infl": infl,
        "stats": {
            "v_norm": vnorm,
            "v_norm_over_mean_delta_norm": vnorm / float(dnorm.mean()) if dnorm.mean() else np.nan,
            "delta_norm_median": float(np.median(dnorm)),
            "delta_norm_cv": float(dnorm.std() / dnorm.mean()) if dnorm.mean() else np.nan,
            "loo_mean": float(np.nanmean(a)), "loo_median": float(np.nanmedian(a)),
            "loo_std": float(np.nanstd(a)), "loo_min": float(np.nanmin(a)),
            "loo_max": float(np.nanmax(a)),
            "loo_frac_pos": float(np.nanmean(a > 0)), "loo_frac_gt25": float(np.nanmean(a > 0.25)),
            "coherence": C, "coherence_isotropic_floor": 1.0 / np.sqrt(n),
            "split_median": float(np.nanmedian(s_split)),
            "split_p2.5": float(np.nanpercentile(s_split, 2.5)),
            "split_p97.5": float(np.nanpercentile(s_split, 97.5)),
            "null_signflip_loo_median": float(np.nanmedian(np.nanmean(a_sf, axis=1))),
            "null_signflip_coherence_median": float(np.nanmedian(C_sf)),
            "null_mismatch_loo_median": float(np.nanmedian(np.nanmean(a_mm, axis=1))),
            "null_mismatch_coherence_median": float(np.nanmedian(C_mm)),
            "null_mismatchpool_loo_median": float(np.nanmedian(np.nanmean(a_mm_pool, axis=1))),
            "null_mismatchpool_coherence_median": float(np.nanmedian(C_mm_pool)),
            "p_signflip_loo": _pval(float(np.nanmean(a)), np.nanmean(a_sf, axis=1)),
            "p_signflip_coherence": _pval(C, C_sf),
            "p_mismatch_loo": _pval(float(np.nanmean(a)), np.nanmean(a_mm, axis=1)),
            "p_mismatch_coherence": _pval(C, C_mm),
            "p_mismatchpool_loo": _pval(float(np.nanmean(a)), np.nanmean(a_mm_pool, axis=1)),
            "p_mismatchpool_coherence": _pval(C, C_mm_pool),
        },
        "dists": {"split": s_split, "sf_loo": np.nanmean(a_sf, axis=1),
                  "sf_coh": C_sf, "mm_loo": np.nanmean(a_mm, axis=1), "mm_coh": C_mm,
                  "mmpool_loo": np.nanmean(a_mm_pool, axis=1), "mmpool_coh": C_mm_pool},
    }


def _mismatch_null(PP, HH, PH, pS, hS, ss, families, rng, within_family: bool):
    """Permute which platypus goes with which human. S (hence v) is invariant; a_i and C are not."""
    n = PP.shape[0]
    groups = [np.where(families == f)[0] for f in np.unique(families)] if within_family else None
    a_out = np.empty((N_PERM, n))
    c_out = np.empty(N_PERM)
    ppd, hhd = np.diag(PP), np.diag(HH)
    ar = np.arange(n)
    for p in range(N_PERM):
        if within_family:
            pi = np.empty(n, dtype=int)
            for gidx in groups:
                pi[gidx] = rng.permutation(gidx)
        else:
            pi = rng.permutation(n)
        gg = pS[pi] - hS
        dd = ppd[pi] - 2.0 * PH[pi, ar] + hhd
        a_out[p] = _cos_from_grams(dd, ss - 2.0 * gg + dd, gg - dd)
        # coherence: full N x N delta Gram under this permutation
        Dg = PP[np.ix_(pi, pi)] - PH[np.ix_(pi, ar)] - PH[np.ix_(pi, ar)].T + HH
        nrm = np.sqrt(np.maximum(np.diag(Dg), 0.0))
        good = nrm > 0
        Ug = np.zeros_like(Dg)
        Ug[np.ix_(good, good)] = Dg[np.ix_(good, good)] / np.outer(nrm[good], nrm[good])
        c_out[p] = np.sqrt(max(Ug.sum(), 0.0)) / n
    return a_out, c_out


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results" / "2026-07-28_evo2-platypus-paired"
    ap.add_argument("--stage1-dir", type=Path, default=base / "stage1")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default stage2/ ; for --pooled, stage2_<representation>/")
    ap.add_argument("--pooled", action="store_true",
                    help="analyse a representation_sweep.py pooling instead of the Stage-1 "
                         "last-position activations")
    ap.add_argument("--pooled-npz", type=Path,
                    default=base / "stage6_representation" / "pooled_representations.npz")
    ap.add_argument("--representation", default="cds_mean")
    args = ap.parse_args()

    if args.pooled:
        # Run the same statistics for any selected pooling representation.
        d = np.load(args.pooled_npz, allow_pickle=True)
        if args.representation not in d.files:
            raise SystemExit(f"{args.representation!r} not in {args.pooled_npz.name}; have "
                             f"{[k for k in d.files if k not in ('genes', 'layers')]}")
        a = d[args.representation]                     # (N, 2, L, H)
        genes = np.array([str(g) for g in d["genes"]])
        with open(args.stage1_dir / "pairs.csv") as fh:
            rows = list(csv.DictReader(fh))
        if [r["gene"] for r in rows] != list(genes):
            raise AssertionError("gene order in the npz does not match pairs.csv")
        families = np.array([r["family"] for r in rows])
        layers = np.array([f"blocks.{i}" for i in range(a.shape[2])])
        acts = a
        out_dir = args.out_dir or base / f"stage2_{args.representation}"
        tag = args.representation
    else:
        d = np.load(args.stage1_dir / "activations_last_pos.npz", allow_pickle=True)
        acts = d["acts"]                      # (N, 2, L, H)
        genes, families, layers = d["genes"], d["families"], d["layers"]
        out_dir = args.out_dir or base / "stage2"
        tag = "prefix_last"
    args.out_dir = out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n, _, n_layers, _ = acts.shape
    print(f"representation={tag}  N={n} genes, {n_layers} layers, H={acts.shape[-1]}")

    rng = np.random.default_rng(SEED)
    rows, per_gene, vecs, dists = [], [], {}, {}
    for li in range(n_layers):
        Hm = acts[:, 0, li, :].astype(np.float64)
        Pm = acts[:, 1, li, :].astype(np.float64)
        res = analyse_layer(Hm, Pm, families, rng)
        rows.append({"layer": li, "layer_name": str(layers[li]), **res["stats"]})
        for i in range(n):
            per_gene.append({"layer": li, "gene": str(genes[i]), "family": str(families[i]),
                             "loo_cos": res["a"][i],
                             "delta_norm": res["dnorm"][i], "influence": res["infl"][i]})
        vecs[f"v_{li}"] = res["v"].astype(np.float32)
        nrm = np.linalg.norm(res["v"])
        vecs[f"vhat_{li}"] = (res["v"] / nrm).astype(np.float32) if nrm > 0 else res["v"].astype(np.float32)
        dists[f"split_{li}"] = res["dists"]["split"].astype(np.float32)
        for k in ("sf_loo", "sf_coh", "mm_loo", "mm_coh", "mmpool_loo", "mmpool_coh"):
            dists[f"{k}_{li}"] = np.asarray(res["dists"][k], dtype=np.float32)
        s = res["stats"]
        print(f"  L{li:2d} loo_med={s['loo_median']:+.4f} frac+={s['loo_frac_pos']:.2f} "
              f"C={s['coherence']:.4f} (floor {s['coherence_isotropic_floor']:.4f}) "
              f"split={s['split_median']:+.4f} |v|={s['v_norm']:.4g}", flush=True)

    pd.DataFrame(rows).to_csv(args.out_dir / "layer_stats.csv", index=False)
    pd.DataFrame(per_gene).to_csv(args.out_dir / "per_gene_by_layer.csv", index=False)
    np.savez_compressed(args.out_dir / "mean_vectors.npz", genes=genes, families=families,
                        layers=layers, **vecs)
    np.savez_compressed(args.out_dir / "null_distributions.npz", **dists)
    (args.out_dir / "stage2_config.json").write_text(json.dumps({
        "n_genes": int(n), "n_layers": int(n_layers), "n_split": N_SPLIT,
        "representation": tag,
        "n_perm": N_PERM, "seed": SEED,
        "dtype": "float64", "pval_rule": "(count_ge + 1)/(n_perm + 1)",
        "statistics": {
            "loo_alignment": {"null": 0.0},
            "coherence": {"null": "1/sqrt(n_genes)"},
            "split_half": {"null": 0.0},
        },
        "removed_2026_08_03": {
            "projections_q": "q_mean is algebraically identical to v_norm; q_frac_pos was not "
                             "leave-one-out corrected and has a null of ~1.0 at H >> N",
            "bootstrap_stability": "measures resample overlap; its own sign-flip null sat at 0.713 "
                                   "at all 32 layers vs 0.72-0.81 observed",
        },
    }, indent=2))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
