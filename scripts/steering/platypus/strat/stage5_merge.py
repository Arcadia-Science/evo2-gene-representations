"""Stage 5c — does the platypus direction, or its magnitude, covary with evolutionary rate? CPU."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PRIMARY = ["background_rate", "focal_residual", "human_residual"]
LEGACY = ["tree_len", "diameter", "focal_target_dist", "mean_target_dist", "treeness"]
NUISANCE = ["aln_len_trimmed", "n_taxa_tree", "log_cds_len", "retained_frac"]

# ---- H1a / H1b confirmatory set ----------------------------------------------------------------
# Use four confirmatory predictors spanning the effective rate-statistic axes.
# Report remaining predictors as exploratory.
CONFIRMATORY = ["dN_hp_yn", "dS_hp_yn", "focal_residual", "treeness"]
CONFIRM_ROLE = {"dN_hp_yn": "PC1 amino-acid divergence",
                "dS_hp_yn": "PC2 synonymous divergence (NEGATIVE CONTROL)",
                "focal_residual": "PC3 platypus-lineage acceleration",
                "treeness": "PC4 rate heterogeneity"}
# H1a is orientation, H1b is magnitude. They are SEPARATE hypotheses: a gene can carry a large
# difference vector that points the consensus way, or a small one that does not, and the two have
# opposite implications for steering. Holm runs over 4 predictors x 2 outcome families.
OUTCOMES_H1A = ["loo_cos", "loo_cos_pc1"]
OUTCOMES_H1B = ["delta_norm"]

# H2b tests dN and omega against dS as the neutral-divergence control.
# Keep codeml models separate and prioritize independently estimated yn00 dN/dS.
DNDS_PRIMARY = ["dN_background_yn", "omega_background_yn", "dN_hp_yn", "omega_hp_yn",
                "omega_m0", "omega_plat_m2", "omega_bg_m2",
                "dN_background_fr", "omega_platypus_fr"]
DNDS_NEGCTRL = ["dS_background_yn", "dS_hp_yn", "dS_background_fr", "tree_dS_m0"]


def resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    out = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < 5:
        return out
    b, a = np.polyfit(x[ok], y[ok], 1)
    out[ok] = y[ok] - (a + b * x[ok])
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """Spearman on ranks after linear regression on the (rank-transformed) covariates."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if ok.sum() < 10:
        return np.nan, np.nan
    rx = stats.rankdata(x[ok])
    ry = stats.rankdata(y[ok])
    rz = np.column_stack([stats.rankdata(Z[ok, j]) for j in range(Z.shape[1])])
    rz = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - rz @ np.linalg.lstsq(rz, rx, rcond=None)[0]
    ey = ry - rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def loo_pc1_cos(D: np.ndarray) -> np.ndarray:
    """cos(D_i, v_-i) with PC1 removed, PC1 and v_-i both estimated WITHOUT gene i."""
    n = D.shape[0]
    out = np.empty(n)
    for i in range(n):
        M = np.delete(D, i, axis=0)
        mu = M.mean(0)
        _u, _s, vt = np.linalg.svd(M - mu, full_matrices=False)
        p1 = vt[0]
        v = mu - (mu @ p1) * p1
        di = D[i] - (D[i] @ p1) * p1
        nv, nd = np.linalg.norm(v), np.linalg.norm(di)
        out[i] = float(di @ v / (nv * nd)) if nv > 0 and nd > 0 else np.nan
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved. NaNs pass through."""
    idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    out = [np.nan] * len(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    m, run = len(order), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        out[i] = min(run, 1.0)
    return out


def shape_fit(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> dict:
    """The three pre-registered shape statistics for H1a/H1b/H2a, from one OLS fit plus one contrast."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, strata = x[ok], y[ok], strata[ok]
    if len(x) < 20 or np.std(x) == 0:
        return {}
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones_like(z), z, z ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = len(x) - X.shape[1]
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_lin, t_quad = beta[1] / se[1], beta[2] / se[2]

    mid, ext = y[np.isin(strata, [2, 3])], y[np.isin(strata, [0, 4])]
    if len(mid) >= 5 and len(ext) >= 5:
        tt = stats.ttest_ind(mid, ext, equal_var=False)
        mid_ext, p_me = float(mid.mean() - ext.mean()), float(tt.pvalue)
    else:
        mid_ext, p_me = np.nan, np.nan

    return {"n": int(len(x)),
            "b_linear": round(float(beta[1]), 4),
            "p_linear": float(2 * stats.t.sf(abs(t_lin), dof)),
            "b_quad": round(float(beta[2]), 4),
            "p_quad": float(2 * stats.t.sf(abs(t_quad), dof)),
            "mid_minus_ext": round(mid_ext, 4) if np.isfinite(mid_ext) else np.nan,
            "p_mid_ext": p_me}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[25, 26, 27])
    ap.add_argument("--modes", nargs="+", default=["cds_mean"],
                    help="pooling modes; cds_mean is the only one any pre-registered test uses")
    args = ap.parse_args()

    pairs = pd.read_csv(args.run / "stage1" / "pairs.csv")
    tre = pd.read_csv(args.run / "stage5" / "tree_stats.csv")
    cov = pd.read_csv(args.run / "stage2" / "aligned_coverage.csv")
    df = pairs[["gene", "stratum", "perc_id_hp", "cds_len_human"]].merge(
        tre[tre.status == "ok"], on="gene", how="inner").merge(
        cov[["gene", "retained_frac"]], on="gene", how="left")
    df["log_cds_len"] = np.log10(df.cds_len_human)
    df["focal_residual"] = resid(df.platypus_branch.to_numpy(float),
                                 df.background_rate.to_numpy(float))
    df["human_residual"] = resid(df.human_branch.to_numpy(float),
                                 df.background_rate.to_numpy(float))
    print(f"n = {len(df)} genes with both geometry and a tree\n")

    # Stage 5d, if it has run. Absent columns stay out of the grid rather than filling with NaN, so
    # the test count reflects what was actually estimated.
    dnds_path = args.run / "stage5" / "dnds.csv"
    dnds_primary, dnds_negctrl = [], []
    if dnds_path.exists():
        dn = pd.read_csv(dnds_path)
        dn = dn[dn.status == "ok"]
        keep = [c for c in DNDS_PRIMARY + DNDS_NEGCTRL if c in dn.columns
                and dn[c].notna().sum() >= 20]
        df = df.merge(dn[["gene", *keep]], on="gene", how="left")
        dnds_primary = [c for c in DNDS_PRIMARY if c in keep]
        dnds_negctrl = [c for c in DNDS_NEGCTRL if c in keep]
        print(f"stage 5d dN/dS: {len(dn)} genes ok; predictors {dnds_primary}; "
              f"negative controls {dnds_negctrl}")
        if "dS_saturated" in dn.columns:
            print(f"  platypus dS saturated in {int(dn.dS_saturated.sum())}/{len(dn)} genes"
                  " -- read the negative control from BACKGROUND dS, not platypus dS")
        for c in keep:
            print(f"  {c:22s} n={int(df[c].notna().sum()):3d}  median={df[c].median():.4f}")
        print()
    else:
        print(f"stage 5d dN/dS not present ({dnds_path}) -- H2b dN/omega/dS arm SKIPPED\n")

    # sanity: the rate predictors should track the stratification variable
    print("rate predictor vs perc_id_hp (sanity, not a test):")
    for c in PRIMARY + LEGACY + dnds_primary + dnds_negctrl:
        r, p = stats.spearmanr(df.perc_id_hp, df[c], nan_policy="omit")
        print(f"  {c:22s} rho={r:+.3f}  p={p:.2g}")

    npz = np.load(args.run / "stage2" / "pooled_representations.npz", allow_pickle=True)
    genes_npz = [str(g) for g in npz["genes"]]
    rows, shape_rows = [], []
    for mode in args.modes:
        a = npz[mode]
        for li in args.layers:
            D = (a[:, 1, li, :] - a[:, 0, li, :]).astype(np.float64)
            per = pd.DataFrame({"gene": genes_npz})
            # raw loo_cos straight from the stage-2 per-gene table (same code as the headline stats)
            pg = pd.read_csv(args.run / f"geom_{mode}" / "per_gene_by_layer.csv")
            pg = pg[pg.layer == li][["gene", "loo_cos", "delta_norm"]]
            per = per.merge(pg, on="gene", how="left")
            per["loo_cos_pc1"] = loo_pc1_cos(D)
            m = df.merge(per, on="gene", how="inner")
            Z = m[NUISANCE].to_numpy(float)
            roles = {**{c: "primary" for c in PRIMARY},
                     **{c: "legacy" for c in LEGACY},
                     **{c: "dnds_primary" for c in dnds_primary},
                     **{c: "dnds_negative_control" for c in dnds_negctrl},
                     "perc_id_hp": "frame"}
            for pred in PRIMARY + LEGACY + dnds_primary + dnds_negctrl + ["perc_id_hp"]:
                for outc in ["loo_cos", "loo_cos_pc1", "delta_norm"]:
                    x = m[pred].to_numpy(float)
                    y = m[outc].to_numpy(float)
                    r, p = stats.spearmanr(x, y, nan_policy="omit")
                    pr, pp = partial_spearman(x, y, Z)
                    rows.append({"mode": mode, "layer": li, "predictor": pred,
                                 "role": roles[pred], "outcome": outc,
                                 "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
                                 "rho": round(float(r), 3), "p": float(p),
                                 "rho_partial": round(pr, 3), "p_partial": pp})
            # --- H1a / H1b: the confirmatory shape fits, separate for orientation and magnitude ---
            strata = m["stratum"].to_numpy(int)
            for pred in CONFIRMATORY:
                if pred not in m.columns:
                    continue
                for outc in OUTCOMES_H1A + OUTCOMES_H1B:
                    fit = shape_fit(m[pred].to_numpy(float), m[outc].to_numpy(float), strata)
                    if not fit:
                        continue
                    shape_rows.append({
                        "mode": mode, "layer": li, "predictor": pred,
                        "axis": CONFIRM_ROLE[pred],
                        "hypothesis": "H1b_magnitude" if outc in OUTCOMES_H1B else "H1a_orientation",
                        "outcome": outc, **fit})

    res = pd.DataFrame(rows)
    out = args.run / "stage5" / "rate_vs_direction.csv"
    res.to_csv(out, index=False)

    n_tests = res[res.role != "frame"].shape[0]
    thr = 0.05 / n_tests
    print(f"\n{n_tests} predictor x outcome x layer x mode tests; Bonferroni threshold {thr:.2g}")
    # Use item access because `mode` is also a DataFrame method.
    for mode in args.modes:
        for li in args.layers:
            s = res[(res["mode"] == mode) & (res.layer == li)]
            print(f"\n--- {mode}  L{li} " + "-" * 46)
            piv = s.pivot(index="predictor", columns="outcome", values="rho")
            pv = s.pivot(index="predictor", columns="outcome", values="p")
            show = piv.copy().astype(object)
            for i in piv.index:
                for c in piv.columns:
                    star = ("**" if pv.loc[i, c] < thr else
                            "*" if pv.loc[i, c] < 0.05 else "")
                    show.loc[i, c] = f"{piv.loc[i, c]:+.3f}{star}"
            print(show.to_string())
    strong = res[(res.p < thr) & (res.role != "frame")]
    print(f"\nsurviving Bonferroni: {len(strong)}")
    if len(strong):
        print(strong.sort_values("p")[["mode", "layer", "predictor", "role", "outcome", "rho", "p",
                                       "rho_partial", "p_partial"]].to_string(index=False))

    # ---- H1a / H1b: the confirmatory, shape-free geometry tests -----------------------------------
    if shape_rows:
        sh = pd.DataFrame(shape_rows)
        # Holm within each (mode, layer, hypothesis) family: 4 predictors x the outcomes in that
        # family, over the three shape statistics. Correcting per family rather than over the whole
        # grid keeps orientation and magnitude as the two separate hypotheses they are.
        sh["p_min"] = sh[["p_linear", "p_quad", "p_mid_ext"]].min(axis=1)
        sh["p_holm"] = np.nan
        for (_, _, _), g in sh.groupby(["mode", "layer", "hypothesis"]):
            sh.loc[g.index, "p_holm"] = holm(g.p_min.tolist())
        shp = args.run / "stage5" / "rate_vs_direction_shape.csv"
        sh.to_csv(shp, index=False)

        print("\n" + "=" * 100)
        print("H1a (orientation) / H1b (magnitude) -- CONFIRMATORY, four rate axes, three shape "
              "statistics each")
        print("  linear + quadratic from y ~ 1 + z + z^2; mid-vs-extreme is strata {2,3} vs {0,4}.")
        print("  A hump shows as ns linear + negative quadratic + positive mid-vs-extreme.")
        print("=" * 100)
        for mode in args.modes:
            for li in args.layers:
                for hyp in ["H1a_orientation", "H1b_magnitude"]:
                    s = sh[(sh["mode"] == mode) & (sh.layer == li) & (sh.hypothesis == hyp)]
                    if s.empty:
                        continue
                    print(f"\n--- {hyp}  {mode}  L{li} " + "-" * 40)
                    for _, r in s.iterrows():
                        flag = "**" if r.p_holm < 0.05 else ("*" if r.p_min < 0.05 else "  ")
                        print(f"  {flag} {r.predictor:16s} {r.outcome:12s} "
                              f"lin {r.b_linear:+.4f} (p={r.p_linear:.3g})  "
                              f"quad {r.b_quad:+.4f} (p={r.p_quad:.3g})  "
                              f"mid-ext {r.mid_minus_ext:+.4f} (p={r.p_mid_ext:.3g})  "
                              f"[Holm {r.p_holm:.3g}]  {r.axis}")
        print("\n  ** survives Holm within its (mode, layer, hypothesis) family; * nominal only.")

    # H2b is a CONTRAST, not a count of significant hits: dN/omega should beat dS on the same genes,
    # same layers, same outcome. Report the two side by side or the arm is unreadable.
    if dnds_primary or dnds_negctrl:
        print("\n" + "=" * 96)
        print("H2b -- dN/omega should predict, dS should NOT (dS = control, not a 4th test)")
        print("=" * 96)
        for outc in ["loo_cos", "loo_cos_pc1", "delta_norm"]:
            print(f"\n  outcome = {outc}")
            for role, preds in (("PREDICT ", dnds_primary), ("CONTROL ", dnds_negctrl)):
                for pred in preds:
                    s = res[(res.predictor == pred) & (res.outcome == outc) & (res.role != "frame")]
                    if not len(s):
                        continue
                    best = s.loc[s.p.idxmin()]
                    print(f"    {role}{pred:22s} best |rho| {best.rho:+.3f} "
                          f"(p={best.p:.2g}, partial {best.rho_partial:+.3f}) "
                          f"@ {best['mode']} L{int(best.layer)}   "
                          f"max|rho| across layers {s.rho.abs().max():.3f}")
        print("\n  Read this as: a dS row as strong as the dN/omega rows means the signal tracks "
              "neutral\n  divergence (composition), not protein change -- which would undercut the "
              "protein-level account.")
    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
