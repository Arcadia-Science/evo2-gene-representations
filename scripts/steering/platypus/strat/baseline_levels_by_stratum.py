"""Absolute unsteered vs steered levels by stratum, for the two figure-7 readouts. No GPU.

Figures 7 and 8 plot deltas. This plots the LEVELS those deltas are built from, so the headroom
question -- do conserved genes start closer to the ceiling, leaving less room to gain and more to
lose? -- can be read straight off the page instead of argued about.

The answer this panel shows: no for the median gene in any stratum (both readouts sit far below
the ceiling everywhere), but the amino-acid readout is bimodal in the conserved strata, where a
minority of genes are reproduced almost exactly by the unsteered model.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module
import arcadia_style as acs  # noqa: E402
from strat_metric import ALIASES, DEFAULT_METRIC, METRICS  # noqa: E402

# Same conservation ramp and band labels the delta violin uses, so a stratum keeps its colour
# across figure 7 and this one.
CONS_CMAP = LinearSegmentedColormap.from_list(
    "cons_seq", [acs.apc.sky, acs.apc.vital, acs.apc.aegean, acs.SERIES_PRIMARY]
)
STRATUM_POS = [0.12, 0.34, 0.55, 0.76, 0.96]
STRATUM_LAB = ["s0\nfastest", "s1", "s2", "s3", "s4\nconserved"]
GREY = acs.SERIES_MUTED

# The site-recovery column and its label come from the ONE selector every strat figure uses, so
# this panel cannot drift from figures 7 and 8 the way a hardcoded label would.
AA = ("aa_id_to_target", "Amino-acid identity to platypus (%)")


def readouts(mspec: dict) -> list[tuple[str, str]]:
    return [(mspec["col"], mspec["axis_level"]), AA]


def boot_ci(
    x: np.ndarray, stat=np.median, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for a summary statistic of one stratum."""
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def trend_figure(u, s, strata, readout, steered_label, out_dir, stem) -> pd.DataFrame:
    """Levels vs conservation as two lines, so a monotone trend (or its absence) is readable.

    x is the stratum's MEDIAN human-platypus identity, not the stratum index: the bins are not
    evenly spaced in conservation, and evenly spaced ticks would invent a linearity the panel is
    meant to test. The reported rho is per-GENE across all 400 genes, never across the five
    plotted points -- a correlation on five binned means is neither powered nor honest.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=300)
    xs = [float(u.loc[u.stratum == st, "perc_id_hp"].median()) for st in strata]
    rows = []
    for ax, (col, label) in zip(axes, readout, strict=True):
        for frame, name, color in (
            (u, "unsteered", GREY),
            (s, f"steered ({steered_label})", CONS_CMAP(0.62)),
        ):
            med = [float(frame.loc[frame.stratum == st, col].median()) for st in strata]
            mean = [float(frame.loc[frame.stratum == st, col].mean()) for st in strata]
            cis = [
                boot_ci(frame.loc[frame.stratum == st, col].to_numpy(float)) for st in strata
            ]
            ax.fill_between(
                xs, [c[0] for c in cis], [c[1] for c in cis], color=color, alpha=0.18, lw=0
            )
            ax.plot(xs, med, "-o", color=color, lw=1.8, ms=4.5, label=f"{name} — median", zorder=3)
            # The mean is drawn too because the two diverge sharply in the conserved strata of the
            # amino-acid panel: that gap IS the bimodality, and a median-only panel would hide it.
            ax.plot(
                xs, mean, "--^", color=color, lw=1.0, ms=3.5, alpha=0.75, label=f"{name} — mean"
            )
            rho, p = stats.spearmanr(frame.perc_id_hp, frame[col], nan_policy="omit")
            rows.append(
                {
                    "metric": col,
                    "condition": name,
                    "rho_level_vs_conservation": round(float(rho), 3),
                    "p": float(p),
                    "n_genes": int(frame[col].notna().sum()),
                }
            )
        txt = "\n".join(
            f"{r['condition'].split(' (')[0]}: ρ={r['rho_level_vs_conservation']:+.2f}, "
            f"p={r['p']:.1g}"
            for r in rows
            if r["metric"] == col
        )
        ax.text(
            0.02,
            0.97,
            f"per-gene Spearman vs conservation (n={rows[-1]['n_genes']})\n{txt}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.4,
            color=acs.ANNOTATION,
        )
        for x, st in zip(xs, strata, strict=True):
            ax.annotate(
                f"s{int(st)}",
                (x, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -15),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=GREY,
            )
        ax.set_xlabel("human–platypus protein identity (%, stratum median)", labelpad=13)
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0].capitalize(), fontweight="bold", fontsize=9)
    axes[0].legend(frameon=False, fontsize=6.4, loc="lower left", ncol=2)
    fig.suptitle(
        "Level vs conservation — is a stratum easier or harder to recover, steered and unsteered?",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Solid = median, dashed = mean, band = 95% bootstrap CI of the median (2,000 resamples, "
        "n=80 genes per stratum).\nThe quoted ρ is across all genes, not across the five plotted "
        "points.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out_dir}")
    return pd.DataFrame(rows)


def violin(ax, xs, data, offset, color, width=0.34):
    """One offset violin per stratum, medians as a white tick."""
    parts = ax.violinplot(
        data,
        positions=[x + offset for x in xs],
        widths=width,
        showextrema=False,
        showmedians=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.85)
        body.set_linewidth(0.6)
    for x, v in zip(xs, data, strict=True):
        ax.plot(
            [x + offset - width / 3, x + offset + width / 3],
            [np.median(v)] * 2,
            color="white",
            lw=1.4,
            solid_capstyle="butt",
            zorder=4,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "figure_data" / "exp3_steering_outcomes.csv",
        help="per-gene x condition outcomes (default: the tracked figure_data table)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "2026-08-08_platypus-strat-400" / "figures",
        help="where the figure and its numbers are written",
    )
    ap.add_argument(
        "--metric",
        choices=sorted(METRICS) + sorted(ALIASES),
        default=DEFAULT_METRIC,
        help="site set to plot (default private: the run's headline readout)",
    )
    ap.add_argument("--steered", default="add_a1.0", help="steered condition to compare (dose arm)")
    ap.add_argument("--stem", default="14_baseline_levels_by_stratum")
    args = ap.parse_args()

    acs.setup()
    mspec = METRICS[ALIASES.get(args.metric, args.metric)]
    readout = readouts(mspec)
    d = pd.read_csv(args.scores)
    if mspec["col"] not in d.columns:
        raise SystemExit(
            f"{mspec['col']} not in {args.scores} -- rescore with the `{args.metric}` site set"
        )
    for cond in ("unsteered", args.steered):
        if cond not in set(d.condition):
            raise SystemExit(
                f"condition {cond!r} not in {args.scores}; have {sorted(set(d.condition))}"
            )
    u = d[d.condition == "unsteered"].set_index("gene")
    s = d[d.condition == args.steered].set_index("gene")
    strata = sorted(u.stratum.unique())
    xs = list(range(len(strata)))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=300)
    for ax, (col, label) in zip(axes, readout, strict=True):
        un = [u.loc[u.stratum == st, col].to_numpy(float) for st in strata]
        stv = [s.loc[s.stratum == st, col].to_numpy(float) for st in strata]
        violin(ax, xs, un, -0.19, GREY)
        for x, st, v in zip(xs, strata, stv, strict=True):
            violin(ax, [x], [v], +0.19, CONS_CMAP(STRATUM_POS[int(st)]))

        # The annotated number is the PAIRED per-gene mean delta -- the statistic figures 7 and 8
        # plot -- NOT the gap between the two medians drawn above it. The two can differ in sign
        # (stratum 2 at alpha 1: medians -0.1, paired mean +2.0) whenever steering reshapes the
        # distribution instead of translating it, so annotating the visible gap would invite a
        # reader to "correct" the delta figures against a statistic they do not show.
        for x, st in zip(xs, strata, strict=True):
            g = s.loc[s.stratum == st, col] - u.loc[u.stratum == st, col]
            ax.annotate(
                f"{g.mean():+.1f}",
                (x, -7.5),
                ha="center",
                fontsize=6.5,
                color=acs.ANNOTATION,
            )
        ax.axhline(100, color=GREY, ls=":", lw=0.8)
        ax.text(len(strata) - 0.6, 100.5, "ceiling", fontsize=6, color=GREY, va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels(STRATUM_LAB, fontsize=7)
        ax.set_xlim(-0.55, len(strata) - 0.45)
        ax.set_ylim(-12, 108)
        ax.set_xlabel("conservation stratum (human–platypus protein identity)")
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0].capitalize(), fontweight="bold", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=GREY, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=CONS_CMAP(0.55), alpha=0.85),
    ]
    axes[0].legend(
        handles,
        ["unsteered", f"steered ({args.steered})"],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    fig.suptitle(
        "Absolute levels behind the delta violins — how close the unsteered model already gets, "
        "by stratum",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.06,
        "White tick = median of the levels drawn. The row underneath is the PAIRED per-gene mean "
        "delta — the statistic figures 7 and 8 plot — which is not\nthe gap between the two "
        "medians and can differ from it in sign. "
        f"n={len(u) // len(strata)} genes per stratum. Both readouts sit far below the ceiling in "
        "every stratum,\nso neither delta is compressed by headroom.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{args.stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {args.stem}  ->  {args.out_dir}")

    trend = trend_figure(
        u, s, strata, readout, args.steered, args.out_dir, f"{args.stem}_trend"
    )
    trend.to_csv(args.out_dir / f"{args.stem}_trend.csv", index=False)
    print(trend.to_string(index=False))

    # ---- the numbers behind the picture -------------------------------------------------------
    rows = []
    for col, _ in readout:
        for st in strata:
            a = u.loc[u.stratum == st, col]
            b = s.loc[s.stratum == st, col]
            rows.append(
                {
                    "metric": col,
                    "stratum": int(st),
                    "n": len(a),
                    "unsteered_median": round(float(a.median()), 2),
                    "unsteered_mean": round(float(a.mean()), 2),
                    "unsteered_p90": round(float(a.quantile(0.9)), 2),
                    "unsteered_max": round(float(a.max()), 2),
                    "steered_median": round(float(b.median()), 2),
                    # unpaired gap between the two medians -- what the picture shows
                    "median_gap": round(float(b.median() - a.median()), 2),
                    # paired per-gene delta -- what figures 7 and 8 report
                    "paired_mean_delta": round(float((b - a).mean()), 2),
                    "paired_median_delta": round(float((b - a).median()), 2),
                    "frac_genes_improved": round(float((b - a).gt(0).mean()), 3),
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(args.out_dir / f"{args.stem}.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
