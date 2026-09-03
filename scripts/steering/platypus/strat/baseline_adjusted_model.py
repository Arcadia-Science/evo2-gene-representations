"""Baseline-adjusted test: does conservation predict steering response at matched baseline? No GPU.

The headroom analysis rescaled each delta by the gene's own room, which is unstable for the
amino-acid readout because its denominator (the unsteered identity) reaches into the single
digits. This asks the same question without any ratio: model the STEERED OUTCOME and put the
unsteered value in as a covariate.

    primary    S ~ U + conservation + C(alpha) + conservation:C(alpha) + (1 | gene)
    secondary  D ~ U + conservation + C(alpha) + conservation:C(alpha) + (1 | gene),  D = S - U

alpha is categorical so no linear dose-response is assumed. Conservation is continuous
human-platypus protein identity, centred and expressed per 10 percentage points; the unsteered
covariate is centred, so each C(alpha) coefficient reads at the panel's mean gene.

The secondary model is reported only because it was asked for. D = S - U, so regressing D on U
is the primary model with the U coefficient shifted by exactly -1 and every other coefficient
unchanged; the script asserts that identity rather than presenting it as new evidence.

Nothing about generation or scoring changes -- S and U are the per-gene condition means the
publication figures use, read from the tracked figure_data table.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module
import arcadia_style as acs  # noqa: E402
from strat_metric import ALIASES, DEFAULT_METRIC, METRICS  # noqa: E402

CONS_CMAP = LinearSegmentedColormap.from_list(
    "cons_seq", [acs.apc.sky, acs.apc.vital, acs.apc.aegean, acs.SERIES_PRIMARY]
)
STRATUM_POS = [0.12, 0.34, 0.55, 0.76, 0.96]
GREY = acs.SERIES_MUTED
AA_COL = "aa_id_to_target"
ALPHAS = [0.5, 1.0, 2.0, 3.0, 4.0]
PRE_RUNAWAY = [0.5, 1.0, 2.0]  # doses below the strongest GC/compositional runaway
FORMULA = "y ~ U_c + cons10 * C(alpha)"


def long_frame(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """One row per gene x alpha: steered outcome, its own unsteered baseline, conservation."""
    u = d[d.condition == "unsteered"].set_index("gene")
    rows = []
    for a in ALPHAS:
        s = d[d.condition == f"add_a{a}"].set_index("gene")
        genes = s.index.intersection(u.index)
        rows.append(
            pd.DataFrame(
                {
                    "gene": genes,
                    "alpha": a,
                    "y": s.loc[genes, col].to_numpy(float),
                    "U": u.loc[genes, col].to_numpy(float),
                    "cons": u.loc[genes, "perc_id_hp"].to_numpy(float),
                    "stratum": u.loc[genes, "stratum"].astype(int).to_numpy(),
                }
            )
        )
    g = pd.concat(rows, ignore_index=True)
    g["delta"] = g.y - g.U
    # Centred so every C(alpha) coefficient reads at the panel's mean gene, and conservation is
    # per 10 pp of identity rather than per 1 pp (which would print as 0.0x for a real effect).
    g["U_c"] = g.U - g.U.mean()
    g["cons10"] = (g.cons - g.cons.mean()) / 10.0
    return g


def fit(g: pd.DataFrame, response: str):
    """Random-intercept model. `response` is 'y' (primary) or 'delta' (secondary)."""
    data = g.rename(columns={response: "y"}) if response != "y" else g
    if response == "delta":
        data = g.drop(columns=["y"]).rename(columns={"delta": "y"})
    return smf.mixedlm(FORMULA, data, groups=data["gene"]).fit(reml=True)


def coef_table(res, model: str, metric: str) -> pd.DataFrame:
    ci = res.conf_int()
    t = pd.DataFrame(
        {
            "metric": metric,
            "model": model,
            "term": res.params.index,
            "estimate": res.params.to_numpy(),
            "se": res.bse.to_numpy(),
            "ci_lo": ci[0].to_numpy(),
            "ci_hi": ci[1].to_numpy(),
            "p": res.pvalues.to_numpy(),
        }
    )
    return t.round({"estimate": 4, "se": 4, "ci_lo": 4, "ci_hi": 4})


def simple_slopes(res, metric: str, model: str) -> pd.DataFrame:
    """Conservation slope AT each alpha = main effect + that alpha's interaction.

    Reported as a linear combination of the fitted coefficients with its own standard error, not
    as the interaction term alone: the interaction is a DIFFERENCE from the reference dose, and
    reading it as "the effect at alpha" is the usual way this model gets misquoted.
    """
    names = [n for n in res.params.index if n != "Group Var"]
    cov = res.cov_params().loc[names, names].to_numpy()
    beta = res.params[names].to_numpy()
    rows = []
    for a in ALPHAS:
        c = np.zeros(len(names))
        c[names.index("cons10")] = 1.0
        inter = f"cons10:C(alpha)[T.{a}]"
        if inter in names:
            c[names.index(inter)] = 1.0
        est = float(c @ beta)
        se = float(np.sqrt(c @ cov @ c))
        z = est / se if se > 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "model": model,
                "alpha": a,
                "term": "conservation slope (per +10 pp identity)",
                "estimate": round(est, 4),
                "se": round(se, 4),
                "ci_lo": round(est - 1.96 * se, 4),
                "ci_hi": round(est + 1.96 * se, 4),
                "p": float(2 * stats.norm.sf(abs(z))),
            }
        )
    return pd.DataFrame(rows)


def predicted(res, g: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Model-implied outcome across conservation at each alpha, baseline held at the panel mean.

    Holding U at its mean is what makes this a BASELINE-ADJUSTED view: every point on every curve
    describes the same hypothetical gene except for its conservation.
    """
    names = [n for n in res.params.index if n != "Group Var"]
    cov = res.cov_params().loc[names, names].to_numpy()
    beta = res.params[names].to_numpy()
    grid = np.arange(np.floor(g.cons.min()), np.ceil(g.cons.max()) + 0.5, 1.0)
    rows = []
    for a in ALPHAS:
        for cons in grid:
            c = np.zeros(len(names))
            c[names.index("Intercept")] = 1.0
            c[names.index("U_c")] = 0.0  # baseline held at the panel mean
            x = (cons - g.cons.mean()) / 10.0
            c[names.index("cons10")] = x
            for nm, val in ((f"C(alpha)[T.{a}]", 1.0), (f"cons10:C(alpha)[T.{a}]", x)):
                if nm in names:
                    c[names.index(nm)] = val
            est = float(c @ beta)
            se = float(np.sqrt(c @ cov @ c))
            rows.append(
                {
                    "metric": metric,
                    "alpha": a,
                    "conservation": float(cons),
                    "U_held_at": round(float(g.U.mean()), 3),
                    "predicted": round(est, 4),
                    "ci_lo": round(est - 1.96 * se, 4),
                    "ci_hi": round(est + 1.96 * se, 4),
                }
            )
    return pd.DataFrame(rows)


# ── figures


def scatter_figure(frames: dict, out: Path, stem: str) -> None:
    """U vs raw change S-U for each alpha, coloured by stratum, with an OLS line per panel."""
    fig, axes = plt.subplots(
        2, len(ALPHAS), figsize=(2.55 * len(ALPHAS), 5.6), dpi=300, squeeze=False
    )
    for row, (metric, g) in enumerate(frames.items()):
        for col, a in enumerate(ALPHAS):
            ax = axes[row][col]
            sub = g[g.alpha == a]
            for st in sorted(sub.stratum.unique()):
                q = sub[sub.stratum == st]
                ax.scatter(
                    q.U,
                    q.delta,
                    s=7,
                    alpha=0.7,
                    lw=0,
                    color=CONS_CMAP(STRATUM_POS[int(st)]),
                    label=f"s{int(st)}" if (row == 0 and col == 0) else None,
                )
            # An OLS line is drawn because the relationship is monotone and roughly linear, but
            # U appears on the x axis AND inside y = S - U, so the slope is not a clean estimate
            # of baseline dependence. The model above avoids this by fitting S, not S - U.
            sl, ic, r, p, _ = stats.linregress(sub.U, sub.delta)
            xs = np.linspace(sub.U.min(), sub.U.max(), 50)
            ax.plot(xs, ic + sl * xs, color=acs.ANNOTATION, lw=1.2, ls="--")
            ax.axhline(0, color=GREY, ls=":", lw=0.8)
            ax.set_title(f"α={a:g}   slope {sl:+.2f}", fontsize=7.5, fontweight="bold")
            ax.set_xlabel("unsteered U (%)", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{metric}\nS − U (pp)", fontsize=7.5)
    axes[0][0].legend(fontsize=5.6, frameon=False, loc="upper right", ncol=2, title=None)
    fig.suptitle(
        "Unsteered baseline vs raw steering change, by dose", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.5,
        -0.02,
        "Dashed line = OLS fit. U is on the x axis and also inside y = S − U, so these slopes "
        "mix real baseline dependence with that shared\nterm; the fitted model addresses the "
        "question by predicting S with U as a covariate instead.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out}")


def predicted_figure(preds: dict, out: Path, stem: str) -> None:
    """Baseline-adjusted conservation effect: predicted outcome vs conservation, one line per α."""
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 3.8), dpi=300, squeeze=False)
    shades = [acs.line_safe(c) for c in acs.gradient_colors(acs.SEQUENTIAL, len(ALPHAS) + 1)]
    for ax, (metric, p) in zip(axes[0], preds.items(), strict=True):
        for a, color in zip(ALPHAS, shades, strict=False):
            q = p[p.alpha == a]
            ax.fill_between(q.conservation, q.ci_lo, q.ci_hi, color=color, alpha=0.14, lw=0)
            ax.plot(q.conservation, q.predicted, color=color, lw=1.8, label=f"α={a:g}")
        ax.set_xlabel("human–platypus protein identity (%)")
        ax.set_ylabel(f"predicted {metric} (%)")
        ax.set_title(metric.capitalize(), fontsize=9, fontweight="bold")
    axes[0][0].legend(fontsize=6.6, frameon=False, loc="best")
    fig.suptitle(
        "Baseline-adjusted conservation effect — unsteered value held at the panel mean",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Model-implied steered outcome from the random-intercept fit, with 95% CIs. Every point "
        "describes the same hypothetical gene\nexcept for its conservation, so a sloped line is a "
        "conservation effect that baseline position does not explain.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out}")


def diagnostics_figure(fits: dict, out: Path, stem: str) -> None:
    """Residuals vs fitted and a normal QQ plot per metric."""
    fig, axes = plt.subplots(len(fits), 2, figsize=(8.2, 3.6 * len(fits)), dpi=300, squeeze=False)
    for row, (metric, res) in enumerate(fits.items()):
        resid, fitted = res.resid.to_numpy(), res.fittedvalues.to_numpy()
        axes[row][0].scatter(fitted, resid, s=6, alpha=0.5, lw=0, color=acs.SERIES_PRIMARY)
        axes[row][0].axhline(0, color=GREY, ls=":", lw=1.0)
        axes[row][0].set_xlabel("fitted")
        axes[row][0].set_ylabel("residual")
        axes[row][0].set_title(f"{metric} — residuals vs fitted", fontsize=8.5, fontweight="bold")
        # Standardised, so the y = x reference line is the correct one: probplot returns the raw
        # sorted residuals against standard-normal quantiles, and an unscaled y = x would make any
        # residual SD != 1 look like non-normality.
        osm, osr = stats.probplot(resid / resid.std(ddof=1), dist="norm", fit=False)
        axes[row][1].scatter(osm, osr, s=6, alpha=0.5, lw=0, color=acs.SERIES_PRIMARY)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        axes[row][1].plot(lim, lim, color=GREY, ls="--", lw=1.0)
        axes[row][1].set_xlabel("theoretical quantiles")
        axes[row][1].set_ylabel("sample quantiles")
        axes[row][1].set_title(
            f"{metric} — normal QQ (standardised)", fontsize=8.5, fontweight="bold"
        )
    fig.suptitle("Model diagnostics (primary models)", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out}")


def aa_baseline_distribution(u: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Why the ratio normalisation was unstable: how small the AA denominator actually gets."""
    v = u[AA_COL].to_numpy(float)
    qs = {
        f"q{int(q * 100):02d}": float(np.percentile(v, q * 100))
        for q in (0.01, 0.05, 0.10, 0.25)
    }
    t = pd.DataFrame(
        [
            {
                "quantity": "unsteered amino-acid identity to platypus (%)",
                "n": len(v),
                "min": round(float(v.min()), 3),
                **{k: round(x, 3) for k, x in qs.items()},
                "median": round(float(np.median(v)), 3),
                "max": round(float(v.max()), 3),
            }
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    ax.hist(v, bins=50, color=acs.SERIES_PRIMARY, alpha=0.85)
    for q, lab in ((1, "1%"), (5, "5%"), (10, "10%")):
        x = float(np.percentile(v, q))
        ax.axvline(x, color=acs.HIGHLIGHT, ls="--", lw=1.0)
        ax.annotate(
            f"{lab}: {x:.1f}", (x, ax.get_ylim()[1] * 0.95), fontsize=6.4, rotation=90, va="top"
        )
    ax.set_xlabel("unsteered amino-acid identity to platypus (%)")
    ax.set_ylabel("genes")
    ax.set_title(
        "Denominator of the AA ratio normalisation", fontsize=9, fontweight="bold"
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"unsteered_aa_baseline_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  unsteered_aa_baseline_distribution  ->  {out}")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", type=Path, default=ROOT / "figure_data/exp3_steering_outcomes.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/2026-08-08_platypus-strat-400/baseline_adjustment",
    )
    ap.add_argument("--metric", choices=sorted(METRICS) + sorted(ALIASES), default=DEFAULT_METRIC)
    args = ap.parse_args()

    acs.setup()
    mspec = METRICS[ALIASES.get(args.metric, args.metric)]
    d = pd.read_csv(args.scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"private recovery": mspec["col"], "amino-acid identity": AA_COL}
    frames, fits, preds = {}, {}, {}
    coefs, slopes = [], []

    for metric, col in metrics.items():
        g = long_frame(d, col)
        frames[metric] = g
        g.to_csv(args.out_dir / f"model_frame_{col}.csv", index=False)

        res = fit(g, "y")
        fits[metric] = res
        (args.out_dir / f"model_summary_primary_{col}.txt").write_text(
            f"PRIMARY  {metric}\n{FORMULA}  + (1|gene),  alpha categorical (ref 0.5)\n"
            f"conservation is per +10 pp identity, centred; U centred at "
            f"{g.U.mean():.3f}\n\n{res.summary()}\n"
        )
        coefs.append(coef_table(res, "primary (S)", metric))
        slopes.append(simple_slopes(res, metric, "primary (S)"))
        preds[metric] = predicted(res, g, metric)

        # Secondary, as requested and labelled as such.
        res_d = fit(g, "delta")
        (args.out_dir / f"model_summary_secondary_delta_{col}.txt").write_text(
            f"SECONDARY  {metric}  (response = S - U; U is part of the response)\n"
            f"{FORMULA}  + (1|gene)\n\n{res_d.summary()}\n"
        )
        coefs.append(coef_table(res_d, "secondary (S-U)", metric))
        slopes.append(simple_slopes(res_d, metric, "secondary (S-U)"))
        # D = S - U with U centred, so the two fits are the SAME fit re-expressed: every
        # coefficient is identical except U_c (shifted by exactly -1) and the intercept (shifted
        # by exactly the mean of U). Asserted here so the secondary table cannot be mistaken for
        # independent evidence; if this ever fails, one of the two models has changed.
        shared = [n for n in res.params.index if n not in ("U_c", "Intercept")]
        assert np.allclose(res.params[shared], res_d.params[shared], atol=1e-6), metric
        assert np.isclose(res.params["U_c"] - 1.0, res_d.params["U_c"], atol=1e-6), metric
        assert np.isclose(
            res.params["Intercept"] - g.U.mean(), res_d.params["Intercept"], atol=1e-4
        ), metric

    pd.concat(coefs, ignore_index=True).to_csv(args.out_dir / "coefficients.csv", index=False)
    slope_t = pd.concat(slopes, ignore_index=True)
    slope_t.to_csv(args.out_dir / "conservation_slopes_by_alpha.csv", index=False)
    pred_t = pd.concat(preds.values(), ignore_index=True)
    pred_t.to_csv(args.out_dir / "predicted_by_conservation.csv", index=False)

    scatter_figure(frames, args.out_dir, "baseline_vs_change_scatter")
    predicted_figure(preds, args.out_dir, "predicted_outcome_by_conservation")
    diagnostics_figure(fits, args.out_dir, "model_diagnostics")

    aa = aa_baseline_distribution(d[d.condition == "unsteered"], args.out_dir)
    aa.to_csv(args.out_dir / "unsteered_aa_baseline_distribution.csv", index=False)

    print("\n=== conservation slope at each alpha (primary models) ===")
    print(slope_t[slope_t.model == "primary (S)"].to_string(index=False))
    print("\n=== unsteered AA identity distribution ===")
    print(aa.to_string(index=False))


if __name__ == "__main__":
    main()
