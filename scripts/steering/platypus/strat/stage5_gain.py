"""Stage 5e — does evolutionary rate predict steering gain? (H2a / H2b). CPU."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# (strat_metric import lives below, after HERE is on sys.path)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stage5_merge import (  # noqa: E402
    CONFIRMATORY,
    DNDS_NEGCTRL,
    DNDS_PRIMARY,
    LEGACY,
    NUISANCE,
    PRIMARY,
    holm,
    partial_spearman,
    resid,
)
from strat_metric import add_metric_args, load_scores  # noqa: E402

# Confirmatory tests use the CDS-mean representation only.
ARMS = ["stage4_cds_mean_blocks27", "stage4_cds_mean_blocks25_26_27"]
CONDS = ["add_a1.0", "add_own"]
EPS = 0.5  # Haldane-Anscombe, so a 0% or 100% cell has a finite logit


def overdispersion(d: pd.DataFrame, n_col: str) -> float:
    """
    Pooled ratio of observed between-sample variance to its binomial expectation, floored at 1.
    """
    num, den = [], []
    for (_g, _c), s in d.groupby(["gene", "condition"], sort=False):
        if len(s) < 3:
            continue
        p = (s["metric"] / 100.0).to_numpy(float)
        n = float(s[n_col].iloc[0])
        pbar = p.mean()
        if not (0 < pbar < 1) or n <= 0 or not np.isfinite(n) or not np.isfinite(pbar):
            continue
        num.append(p.var(ddof=1))
        den.append(pbar * (1 - pbar) / n)
    if not num:
        return 1.0
    return max(1.0, float(np.sum(num) / np.sum(den)))


def per_gene_gain(d: pd.DataFrame, cond: str, phi: float, n_col: str) -> pd.DataFrame:
    """Per-gene logit gain of `cond` vs unsteered, with its inverse-variance weight."""
    g = (
        d.groupby(["gene", "condition"])
        .agg(pct=("metric", "mean"), n_sites=(n_col, "first"), n_samp=("sample", "count"))
        .reset_index()
    )
    b = g[g.condition == "unsteered"].set_index("gene")
    s = g[g.condition == cond].set_index("gene")
    ix = s.index.intersection(b.index)
    rows = []
    for gene in ix:
        # A gene with no evidence for this site set has a NaN percentage and a zero site count; it
        # cannot contribute a logit gain or a weight, and letting it through would put NaN into the
        # WLS design matrix (which does not skip it -- it fails).
        if not np.isfinite(float(s.loc[gene, "pct"])) or not np.isfinite(float(b.loc[gene, "pct"])):
            continue
        if float(s.loc[gene, "n_sites"]) <= 0 or float(b.loc[gene, "n_sites"]) <= 0:
            continue
        out = {}
        for tag, src in (("s", s), ("u", b)):
            n = float(src.loc[gene, "n_sites"]) * float(src.loc[gene, "n_samp"])
            hits = float(src.loc[gene, "pct"]) / 100.0 * n
            p = (hits + EPS) / (n + 2 * EPS)
            out[f"p_{tag}"] = p
            out[f"n_{tag}"] = n
            out[f"v_{tag}"] = phi / (n * p * (1 - p))
        rows.append(
            {
                "gene": gene,
                "gain_pp": float(s.loc[gene, "pct"]) - float(b.loc[gene, "pct"]),
                "logit_gain": np.log(out["p_s"] / (1 - out["p_s"]))
                - np.log(out["p_u"] / (1 - out["p_u"])),
                "var": out["v_s"] + out["v_u"],
                "n_trials": out["n_s"],
            }
        )
    return pd.DataFrame(rows)


def wls_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Inverse-variance weighted fit of y on standardised x, LINEAR AND QUADRATIC."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    nan = {
        "n": int(len(x)),
        "b0": np.nan,
        "b1": np.nan,
        "se1": np.nan,
        "p1": np.nan,
        "b2": np.nan,
        "se2": np.nan,
        "p2": np.nan,
    }
    if len(x) < 15 or np.std(x) == 0:
        return nan
    z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones_like(z), z, z**2])
    W = np.diag(w)
    XtW = X.T @ W
    cov = np.linalg.inv(XtW @ X)
    beta = cov @ (XtW @ y)
    # residual scale: if the model already explains the scatter this is ~1; >1 inflates the SE, so
    # a predictor cannot look significant merely because the weights are optimistic
    r = y - X @ beta
    chi2 = float(r @ W @ r) / max(len(x) - X.shape[1], 1)
    scale = max(chi2, 1.0)
    se1 = float(np.sqrt(cov[1, 1] * scale))
    se2 = float(np.sqrt(cov[2, 2] * scale))
    return {
        "n": int(len(x)),
        "b0": float(beta[0]),
        "b1": float(beta[1]),
        "se1": se1,
        "p1": float(2 * stats.norm.sf(abs(beta[1] / se1))) if se1 > 0 else np.nan,
        "b2": float(beta[2]),
        "se2": se2,
        "p2": float(2 * stats.norm.sf(abs(beta[2] / se2))) if se2 > 0 else np.nan,
    }


def mid_vs_extreme(gain: pd.DataFrame) -> dict:
    """Strata {2,3} vs {0,4} on per-gene gain: the shape-free third statistic of H2a."""
    mid = gain[gain.stratum.isin([2, 3])]
    ext = gain[gain.stratum.isin([0, 4])]
    if len(mid) < 5 or len(ext) < 5:
        return {}
    u = stats.mannwhitneyu(mid.gain_pp, ext.gain_pp)
    wm = float(np.sum(mid.gain_pp / mid["var"]) / np.sum(1 / mid["var"]))
    we = float(np.sum(ext.gain_pp / ext["var"]) / np.sum(1 / ext["var"]))
    return {
        "n_mid": int(len(mid)),
        "n_ext": int(len(ext)),
        "mid_pp": round(float(mid.gain_pp.mean()), 3),
        "ext_pp": round(float(ext.gain_pp.mean()), 3),
        "mid_minus_ext_pp": round(float(mid.gain_pp.mean() - ext.gain_pp.mean()), 3),
        "mwu_p": float(u.pvalue),
        "wtd_mid_pp": round(wm, 3),
        "wtd_ext_pp": round(we, 3),
        "wtd_mid_minus_ext_pp": round(wm - we, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    add_metric_args(ap)
    args = ap.parse_args()

    # ---- predictors: built exactly as in stage5_merge, by importing its constants ---------------
    pairs = pd.read_csv(args.run / "stage1" / "pairs.csv")
    tre = pd.read_csv(args.run / "stage5" / "tree_stats.csv")
    cov = pd.read_csv(args.run / "stage2" / "aligned_coverage.csv")
    df = (
        pairs[["gene", "stratum", "perc_id_hp", "cds_len_human"]]
        .merge(tre[tre.status == "ok"], on="gene", how="inner")
        .merge(cov[["gene", "retained_frac"]], on="gene", how="left")
    )
    df["log_cds_len"] = np.log10(df.cds_len_human)
    df["focal_residual"] = resid(
        df.platypus_branch.to_numpy(float), df.background_rate.to_numpy(float)
    )
    df["human_residual"] = resid(
        df.human_branch.to_numpy(float), df.background_rate.to_numpy(float)
    )

    dnds_primary: list[str] = []
    dnds_negctrl: list[str] = []
    dnds_path = args.run / "stage5" / "dnds.csv"
    if dnds_path.exists():
        dn = pd.read_csv(dnds_path)
        dn = dn[dn.status == "ok"]
        keep = [
            c for c in DNDS_PRIMARY + DNDS_NEGCTRL if c in dn.columns and dn[c].notna().sum() >= 20
        ]
        df = df.merge(dn[["gene", *keep]], on="gene", how="left")
        dnds_primary = [c for c in DNDS_PRIMARY if c in keep]
        dnds_negctrl = [c for c in DNDS_NEGCTRL if c in keep]
        print(f"dN/dS: {len(dn)} genes ok; predict {dnds_primary}; control {dnds_negctrl}")
    else:
        print("dN/dS absent -- H2b arm skipped")
    preds = PRIMARY + LEGACY + dnds_primary + dnds_negctrl
    print(f"n = {len(df)} genes with a tree; {len(preds)} rate predictors\n")

    rows: list[dict] = []
    shape_rows: list[dict] = []
    for arm in args.arms:
        # One arm lacking the rescored site set must not kill the whole run -- report and move on.
        try:
            d, spec = load_scores(
                args.run, arm, scores=args.scores, metric=args.metric, min_voters=args.min_voters
            )
        except SystemExit as e:
            print(f"  (skip {arm}: {str(e).splitlines()[0]})")
            continue
        phi = overdispersion(d, spec["n_col"])
        print("=" * 96)
        print(f"{arm}   overdispersion phi = {phi:.2f}  (nominal trials divided by this)")
        print("=" * 96)
        for cond in CONDS:
            if cond not in set(d.condition):
                continue
            gain = per_gene_gain(d, cond, phi, spec["n_col"])
            m = df.merge(gain, on="gene", how="inner")
            w = 1.0 / m["var"].to_numpy(float)
            y = m.logit_gain.to_numpy(float)

            # b0 sanity check: does the intervention work at all on this arm?
            mu = float(np.sum(w * y) / np.sum(w))
            mu_se = float(np.sqrt(1.0 / np.sum(w)))
            print(
                f"\n  {cond}: n={len(m)}  weighted mean logit gain "
                f"{mu:+.4f} +/- {mu_se:.4f} (z={mu / mu_se:+.1f});  "
                f"unweighted raw gain {np.mean(m.gain_pp):+.2f} pp"
            )

            Z = m[NUISANCE].to_numpy(float)
            me = mid_vs_extreme(m)
            if me:
                shape_rows.append({"arm": arm, "condition": cond, **me})
            for pred in preds + ["perc_id_hp"]:
                x = m[pred].to_numpy(float)
                fit = wls_slope(x, y, w)
                b1, se1, p1 = fit["b1"], fit["se1"], fit["p1"]
                rho, prho = stats.spearmanr(x, m.gain_pp, nan_policy="omit")
                prho_partial = partial_spearman(x, m.gain_pp.to_numpy(float), Z)
                role = (
                    "primary"
                    if pred in PRIMARY
                    else "legacy"
                    if pred in LEGACY
                    else "dnds_primary"
                    if pred in dnds_primary
                    else "dnds_negative_control"
                    if pred in dnds_negctrl
                    else "frame"
                )
                rows.append(
                    {
                        "arm": arm,
                        "condition": cond,
                        "predictor": pred,
                        "role": role,
                        "n": int(np.isfinite(x).sum()),
                        "wls_beta_per_sd": round(b1, 4),
                        "wls_se": round(se1, 4),
                        "wls_p": p1,
                        "wls_quad_per_sd2": round(fit["b2"], 4),
                        "wls_quad_se": round(fit["se2"], 4),
                        "wls_quad_p": fit["p2"],
                        "confirmatory": pred in CONFIRMATORY,
                        "spearman_rho": round(float(rho), 3),
                        "spearman_p": float(prho),
                        "spearman_rho_partial": round(prho_partial[0], 3),
                        "spearman_p_partial": prho_partial[1],
                    }
                )

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no arms scored")
    out = args.run / "stage5" / "rate_vs_gain.csv"
    res.to_csv(out, index=False)

    tested = res[res.role != "frame"]
    thr = 0.05 / len(tested)
    print("\n" + "=" * 96)
    print(f"H2a/H2b -- rate vs STEERING GAIN.  {len(tested)} tests, Bonferroni {thr:.2g}")
    print("=" * 96)
    for arm in res.arm.unique():
        for cond in CONDS:
            s = res[(res.arm == arm) & (res.condition == cond)]
            if s.empty:
                continue
            print(f"\n--- {arm}  {cond} " + "-" * max(0, 40 - len(cond)))
            t = s[
                [
                    "predictor",
                    "role",
                    "n",
                    "wls_beta_per_sd",
                    "wls_se",
                    "wls_p",
                    "spearman_rho",
                    "spearman_p",
                ]
            ].copy()
            t["sig"] = np.where(t.wls_p < thr, "**", np.where(t.wls_p < 0.05, "*", ""))
            print(t.to_string(index=False))

    strong = tested[tested.wls_p < thr]
    print(f"\nsurviving Bonferroni on the weighted test: {len(strong)}")
    if len(strong):
        print(strong.sort_values("wls_p").to_string(index=False))
    nom = tested[tested.wls_p < 0.05]
    print(
        f"nominally p<0.05 (uncorrected): {len(nom)} of {len(tested)} "
        f"-- expected by chance {0.05 * len(tested):.1f}"
    )
    if len(nom):
        print(
            nom.sort_values("wls_p")[
                [
                    "arm",
                    "condition",
                    "predictor",
                    "role",
                    "wls_beta_per_sd",
                    "wls_p",
                    "spearman_rho",
                ]
            ].to_string(index=False)
        )

    if dnds_primary or dnds_negctrl:
        print("\n" + "=" * 96)
        print("H2b -- dN/omega should predict GAIN, dS should NOT")
        print("=" * 96)
        for role, pl in (("PREDICT ", dnds_primary), ("CONTROL ", dnds_negctrl)):
            for pred in pl:
                s = tested[tested.predictor == pred]
                if s.empty:
                    continue
                best = s.loc[s.wls_p.idxmin()]
                print(
                    f"  {role}{pred:22s} best beta {best.wls_beta_per_sd:+.4f}/SD "
                    f"(p={best.wls_p:.3g}, spearman {best.spearman_rho:+.3f}) "
                    f"@ {best.arm.replace('stage4_', '')} {best.condition}"
                )

    # ---- H2a: the three pre-registered shape statistics ------------------------------------------
    # Report linear, quadratic, and shape-free contrasts because a slope cannot detect a hump.
    print("\n" + "=" * 96)
    print("H2a -- rate-dependence of gain, PRE-REGISTERED shape-free: linear + quadratic +")
    print("mid-vs-extreme. Confirmatory predictors only; Holm within each (arm, condition).")
    print("=" * 96)
    conf = res[res.confirmatory].copy()
    if not conf.empty:
        conf["p_min"] = conf[["wls_p", "wls_quad_p"]].min(axis=1)
        conf["p_holm"] = np.nan
        for _, g in conf.groupby(["arm", "condition"]):
            conf.loc[g.index, "p_holm"] = holm(g.p_min.tolist())
        for arm in conf.arm.unique():
            for cond in CONDS:
                s = conf[(conf.arm == arm) & (conf.condition == cond)]
                if s.empty:
                    continue
                print(f"\n--- {arm.replace('stage4_', '')}  {cond} " + "-" * 30)
                for _, r in s.iterrows():
                    flag = "**" if r.p_holm < 0.05 else ("*" if r.p_min < 0.05 else "  ")
                    print(
                        f"  {flag} {r.predictor:22s} lin {r.wls_beta_per_sd:+.4f}/SD "
                        f"(p={r.wls_p:.3g})   quad {r.wls_quad_per_sd2:+.4f}/SD2 "
                        f"(p={r.wls_quad_p:.3g})   [Holm {r.p_holm:.3g}]"
                    )
        conf.to_csv(args.run / "stage5" / "rate_vs_gain_confirmatory.csv", index=False)

    if shape_rows:
        sh = pd.DataFrame(shape_rows)
        sh.to_csv(args.run / "stage5" / "gain_mid_vs_extreme.csv", index=False)
        print("\n  mid-vs-extreme contrast (strata 2,3 vs 0,4) -- the third H2a statistic:")
        for _, r in sh.iterrows():
            print(
                f"    {r.arm.replace('stage4_', ''):28s} {r.condition:9s} "
                f"mid {r.mid_pp:+.2f} vs ext {r.ext_pp:+.2f} pp  "
                f"diff {r.mid_minus_ext_pp:+.2f}  MWU p={r.mwu_p:.4f}   "
                f"[wtd {r.wtd_mid_pp:+.2f} vs {r.wtd_ext_pp:+.2f}]"
            )

    print(f"\n-> {out}")


if __name__ == "__main__":
    sys.exit(main())
