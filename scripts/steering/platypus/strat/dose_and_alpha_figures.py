"""Dose and alpha-convention figures, built from existing stage-4 output. No GPU."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from strat_metric import add_metric_args, coverage_note, load_scores  # noqa: E402

INK, WARM, PLUM, GREY = (acs.SERIES_PRIMARY, acs.SERIES_NULL,
                         acs.SERIES_THIRD, acs.SERIES_MUTED)
COMPLETE_FRAC = 0.98   # a condition must cover this fraction of the panel to be plotted

# Height of one publication panel row, in points.
PUB_PANEL_H = 330.0


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_gene_delta(d: pd.DataFrame, cond: str, col: str,
                   genes: set[str] | None = None) -> pd.Series:
    """Per-gene mean of `col` for `cond`, minus the same gene's unsteered mean."""
    if genes is not None:
        d = d[d.gene.isin(genes)]
    a = d[d.condition == cond].groupby("gene")[col].mean()
    u = d[d.condition == "unsteered"].groupby("gene")[col].mean()
    return (a - u).dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--alphas", nargs="*", type=float, default=None,
                    help="default: every alpha present in the scores")
    ap.add_argument("--min-frac", type=float, default=COMPLETE_FRAC,
                    help="panel fraction a condition must cover to be plotted (default 0.98). "
                         "Lower it to preview a run in progress: every series is then restricted "
                         "to the genes they all share and the figure is stamped PROVISIONAL.")
    ap.add_argument("--pub", action="store_true",
                    help="render the PUBLICATION figure (6d only): an exact 1,000 pt panel, the "
                         "style guide's 15 pt type with monospaced numerals, the five strata "
                         "wrapped to three columns rather than run across the page, and the "
                         "series key below. Writes into <run>/figures/pub/.")
    add_metric_args(ap)
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    # `metric` is an alias of the chosen site set, so the plotting code below names one column
    # whichever readout is selected. Exits with the rescore command if the column is not there —
    # never silently substitutes the other site set, which differs by about a factor of two.
    d, spec = load_scores(args.run, args.dir, scores=args.scores, metric=args.metric,
                          min_voters=args.min_voters)
    out = args.run / "figures"
    out.mkdir(exist_ok=True)

    # Match the bare dose arms EXACTLY. `add_a1.0_L24` (the layer-24 arms, which carry a
    # --cond-suffix and land in this same directory by design) and `add_gc_removed_a1.0` are not
    # points on this curve, and a loose split("_a")[1] turns the first into float("1.0_L24").
    dose_re = re.compile(r"^add_a(\d+(?:\.\d+)?)$")
    present = sorted({float(m.group(1)) for c in d.condition.unique()
                      if (m := dose_re.match(c))})
    alphas = args.alphas or present
    print(f"alphas present in the scores: {present}")

    # ── 6b: dose-response, standalone, with the numbers beside it
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(8.8, 4.4), dpi=300,
                                   gridspec_kw={"width_ratios": [1, 0.46], "wspace": 0.05})
    # Compare each additive arm with the same gene's unsteered cell.
    series = [(spec["delta_label"], "metric", INK, "o"),
              ("amino-acid identity cost", "aa_id_to_target", PLUM, "^")]
    rows = []
    # A cell is only plotted once it is COMPLETE. stage4_steer appends each (gene, condition) as it
    # finishes, so a run in progress leaves partial conditions in the scores file — plotting those
    # would put a point built from whichever genes happened to finish first (2% of the panel, in
    # gene order) next to points built from all 398, with no visual cue. Require ~the full panel.
    n_ref = d[d.condition == "unsteered"].gene.nunique()
    counts = d.groupby("condition").gene.nunique()
    have = {c for c, n in counts.items() if n >= args.min_frac * n_ref}
    partial = {c: int(n) for c, n in counts.items() if c not in have}
    if partial:
        print(f"  SKIPPING incomplete conditions (< {args.min_frac:.0%} of {n_ref} genes): "
              + ", ".join(f"{c} {n}/{n_ref}" for c, n in sorted(partial.items())))
    alphas = [a for a in alphas if f"add_a{a}" in have]

    # For partial conditions, restrict every series to their shared genes.
    plotted = {"unsteered", *(f"add_a{a}" for a in alphas)} & have
    common: set[str] | None = None
    if any(counts[c] < n_ref for c in plotted):
        common = set.intersection(*(set(d[d.condition == c].gene) for c in plotted))
        incomplete = sorted(c for c in plotted if counts[c] < n_ref)
        print(f"  PROVISIONAL: {', '.join(incomplete)} still running; all series restricted to "
              f"the {len(common)}/{n_ref} genes every plotted condition has")
    # Report measured panel means without population-level confidence intervals.
    means: dict[str, list[float]] = {}
    for label, col, colr, mk in series:
        ys = []
        for a in alphas:
            g = per_gene_delta(d, f"add_a{a}", col, genes=common)
            ys.append(g.mean())
            rows.append({"series": label, "alpha": a, "mean_delta_pp": g.mean(),
                         "n_genes": len(g), "provisional": common is not None})
        means[col] = ys
        ax.plot(alphas, ys, color=colr, lw=2.0, marker=mk, ms=6, label=label)
    ax.axhline(0, color=GREY, ls=":", lw=1.0)
    ax.set_xticks(alphas)
    ax.set_xlabel("α")
    ax.set_ylabel("Δ vs the same gene's unsteered cell (pp)")
    ax.set_title(f"Dose–response and its protein cost — {args.metric} sites",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    # ── the same numbers as a table
    # Report cumulative recovery gained per amino-acid identity point lost; leave zero-cost
    # ratios blank.
    ax_t.axis("off")
    priv, aa = means["metric"], means["aa_id_to_target"]
    cells, rate_col = [], []
    for a, p, c in zip(alphas, priv, aa):
        rate = f"{p / -c:.2f}" if c < 0 else "–"
        rate_col.append(rate)
        cells.append([f"{a:.1f}", f"{p:+.2f}", f"{c:+.2f}", rate])   # match the x-axis labels
    tbl = ax_t.table(cellText=cells, cellLoc="center", loc="center",
                     colLabels=["α", f"Δ {args.metric[:4]}\n(pp)", "Δ aa id\n(pp)",
                                "bp per\naa pp"],
                     bbox=[0.02, 0.14, 0.96, 0.72])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_height(0.1)
        if r == 0:
            cell.set_text_props(fontweight="bold", fontsize=7.5, color=acs.apc.black)
            cell.visible_edges = "B"
            cell.set_edgecolor(GREY)
        else:
            cell.set_facecolor(acs.apc.gray if r % 2 else acs.apc.white)
            if c == 1:
                cell.set_text_props(color=INK, fontweight="bold")
            elif c == 2:
                cell.set_text_props(color=PLUM, fontweight="bold")
    ax_t.text(0.5, 0.98, "Δ vs the same gene's unsteered cell", transform=ax_t.transAxes,
              ha="center", va="bottom", fontsize=8, style="italic", color=GREY)
    if any(r == "–" for r in rate_col):
        ax_t.text(0.5, 0.02, "– no identity cost at this dose", transform=ax_t.transAxes,
                  ha="center", va="top", fontsize=6.5, style="italic", color=GREY)
    if common is not None:
        ax_t.text(0.5, -0.06, f"PROVISIONAL · run in progress\nsame {len(common)}/{n_ref} genes "
                              f"throughout", transform=ax_t.transAxes, ha="center", va="top",
                  fontsize=7, fontweight="bold", color=WARM,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=WARM, lw=0.8))
    # No tight_layout here — a table Axes is not compatible with it, and the gridspec already
    # places the two panels; bbox_inches="tight" trims the margins on save.
    for ext in ("png", "pdf"):
        fig.savefig(out / f"6b_dose_response.{ext}", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(out / "6b_dose_response.csv", index=False)
    print(f"[wrote] {out}/6b_dose_response.png")


    # ── 6d: the same curve, one panel per conservation stratum
    # Plot the dose response within each conservation stratum on a shared y-axis.
    strata = (d[d.condition == "unsteered"].drop_duplicates("gene")
              .set_index("gene")[["stratum", "perc_id_hp"]])
    if common is not None:
        strata = strata.loc[strata.index.isin(common)]
    keys = sorted(strata.stratum.unique())
    if pub.is_on():
        # Five panels across a 1,000 pt page leave 180 pt each, which is narrower than one
        # panel title at 17 pt. Wrapped to three columns they get ~310 pt and the strata still
        # read in order, left to right and then down.
        ncol = 3
        nrow = int(np.ceil(len(keys) / ncol))
        # sharey only. Sharing x would strip the tick numbers from every panel that is not in
        # the bottom ROW — but with 5 panels in a 3-wide grid the third column has nothing
        # below it, so that panel would print "alpha" over a bare axis.
        fig, axgrid = plt.subplots(nrow, ncol, dpi=300, sharey=True,
                                   figsize=pub.size(pub.FULL, PUB_PANEL_H * nrow))
        axes = axgrid.ravel()
        for extra_ax in axes[len(keys):]:   # the 6th cell of a 5-stratum panel
            extra_ax.axis("off")
    else:
        fig, axes = plt.subplots(1, len(keys), figsize=(3.05 * len(keys), 3.7), dpi=300,
                                 sharey=True)
    srows = []
    for ax_s, s in zip(np.atleast_1d(axes), keys):
        sub = strata[strata.stratum == s]
        genes = set(sub.index)
        # Genes with no evidence for this site set are NaN, and per_gene_delta drops them silently.
        # Label the panel with what was actually averaged, and say so when the two differ.
        n_ok = d[d.gene.isin(genes) & d["metric"].notna()].gene.nunique()
        peak_a, peak_y = None, -np.inf
        for label, col, colr, mk in series:
            ys = []
            for a in alphas:
                g = per_gene_delta(d, f"add_a{a}", col, genes=genes)
                ys.append(g.mean())
                srows.append({"stratum": s, "alpha": a, "series": label,
                              "mean_delta_pp": g.mean(), "n_genes": len(g)})
            if col == "metric":
                peak_a, peak_y = alphas[int(np.argmax(ys))], max(ys)
            ax_s.plot(alphas, ys, color=colr, lw=1.8, marker=mk, ms=5, label=label)
        ax_s.axhline(0, color=GREY, ls=":", lw=1.0)
        ax_s.axvline(peak_a, color=INK, ls="--", lw=1.0, alpha=0.35)
        # Nudge the label inward when the peak sits on an edge tick, or it runs off the frame —
        # which is exactly where the interesting strata peak (several are still rising at α=4).
        ha = "right" if peak_a == alphas[-1] else "left" if peak_a == alphas[0] else "center"
        dx = -5 if ha == "right" else 5 if ha == "left" else 0
        ax_s.annotate(f"peak α={peak_a:g}\n{peak_y:+.2f} pp", xy=(peak_a, peak_y),
                      xytext=(dx, 8), textcoords="offset points", ha=ha, va="bottom",
                      fontsize=7.5, fontweight="bold", color=INK)
        ax_s.set_xticks(alphas)
        # In the wrapped publication grid the axis is titled only where nothing sits below it.
        if not pub.is_on() or list(keys).index(s) >= len(keys) - ncol:
            ax_s.set_xlabel("α")
        # Warn on incomplete stratum coverage without adding counts to the panel.
        if n_ok != len(genes):
            print(f"  WARNING stratum {s}: only {n_ok} of {len(genes)} genes scorable on this "
                  f"metric — the panel no longer prints this")
        ax_s.set_title(
            (f"s{s} · {sub.perc_id_hp.min():.0f}–{sub.perc_id_hp.max():.0f}% identity"
             if pub.is_on() else
             f"stratum {s} · {sub.perc_id_hp.min():.0f}–{sub.perc_id_hp.max():.0f}% id"),
            fontsize=9, fontweight="bold")
    # Headroom for the peak annotation: the axes are shared, the peak sits at the top of the data,
    # and an 8-point upward offset otherwise lands on top of the panel title.
    ax0 = np.atleast_1d(axes)[0]
    lo_y, hi_y = ax0.get_ylim()
    ax0.set_ylim(lo_y, hi_y + 0.16 * (hi_y - lo_y))
    if pub.is_on():
        # With a wrapped grid the y title belongs on every left-hand panel, not just the first.
        for i, ax_s in enumerate(axes[:len(keys)]):
            if i % ncol == 0:
                ax_s.set_ylabel("Δ vs unsteered (pp)")
    else:
        ax0.set_ylabel("Δ vs unsteered (pp)")
        ax0.legend(fontsize=7.5, frameon=False, loc="lower left")
    # Metric provenance and coverage are recorded in the sibling CSV.
    if pub.is_on():
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        _, key_h = pub.key_below(fig, handles, labels, title="Readout", width=pub.FULL)
        total_h = PUB_PANEL_H * nrow + key_h
        fig.set_size_inches(pub.FULL / 72.0, total_h / 72.0)
        pub.tight(fig, bottom=key_h)
        pub.finish(fig, "6d_dose_by_stratum", directory=out)
    else:
        fig.suptitle("Dose–response by conservation stratum", fontsize=11, fontweight="bold")
        if common is not None:
            fig.text(0.5, -0.02, f"PROVISIONAL · run in progress · same {len(common)}/{n_ref} "
                                 f"genes throughout", ha="center", fontsize=7.5,
                     fontweight="bold", color=WARM,
                     bbox=dict(boxstyle="round,pad=0.3", fc=acs.apc.white, ec=WARM, lw=0.8))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"6d_dose_by_stratum.{ext}", bbox_inches="tight")
        print(f"[wrote] {out}/6d_dose_by_stratum.png")
    pd.DataFrame(srows).to_csv(out / "6d_dose_by_stratum.csv", index=False)

    # ── 6e: Absolute levels by dose and stratum
    # Include alpha=0 so each stratum's unsteered baseline remains visible.
    def level(cond: str, col: str, genes: set[str]) -> float:
        sub = d[d.gene.isin(genes) & (d.condition == cond)]
        return sub.groupby("gene")[col].mean().mean()

    xs0 = [0.0] + alphas                      # 0 = unsteered
    conds0 = ["unsteered"] + [f"add_a{a}" for a in alphas]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.05 * len(keys), 3.7), dpi=300, sharey=True)
    lrows = []
    for ax_s, s in zip(np.atleast_1d(axes), keys):
        sub = strata[strata.stratum == s]
        genes = set(sub.index)
        for label, col, colr, mk in series:
            ys = [level(c, col, genes) for c in conds0]
            for a, c, y in zip(xs0, conds0, ys):
                lrows.append({"stratum": s, "alpha": a, "condition": c, "series": label,
                              "mean_pct": y, "n_genes": len(genes)})
            ax_s.plot(xs0, ys, color=colr, lw=1.8, marker=mk, ms=5,
                      label=label.replace("-bp gain", " bp correct").replace(" cost", ""))
            # Mark the unsteered level so the reader can see how far the line has travelled.
            ax_s.axhline(ys[0], color=colr, ls=":", lw=0.9, alpha=0.5)
        ax_s.set_xticks(xs0)
        ax_s.set_xticklabels([f"{a:g}" for a in xs0])
        ax_s.set_xlabel("α   (0 = unsteered)")
        n_ok = d[d.gene.isin(genes) & d["metric"].notna()].gene.nunique()
        n_lab = f"n = {n_ok} genes" if n_ok == len(genes) else \
                f"n = {n_ok} of {len(genes)} genes scorable"
        ax_s.set_title(f"stratum {s} · {sub.perc_id_hp.min():.0f}–{sub.perc_id_hp.max():.0f}% id\n"
                       f"{n_lab}", fontsize=9, fontweight="bold")
    np.atleast_1d(axes)[0].set_ylabel("% (dotted = that stratum's unsteered level)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, frameon=False, loc="center left")
    fig.suptitle("Absolute levels by conservation stratum "
                 "(left = most diverged from platypus, right = most conserved)\n"
                 + coverage_note(spec), fontsize=11, fontweight="bold")
    if common is not None:
        fig.text(0.5, -0.02, f"PROVISIONAL · run in progress · same {len(common)}/{n_ref} genes "
                             f"throughout", ha="center", fontsize=7.5, fontweight="bold",
                 color=WARM, bbox=dict(boxstyle="round,pad=0.3", fc=acs.apc.white, ec=WARM, lw=0.8))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"6e_levels_by_stratum.{ext}", bbox_inches="tight")
    pd.DataFrame(lrows).to_csv(out / "6e_levels_by_stratum.csv", index=False)
    print(f"[wrote] {out}/6e_levels_by_stratum.png")
    # ── 6c: fixed alpha vs own-scaled alpha
    if "add_own" not in set(d.condition):
        print("no add_own arm — skipping 6c")
        return
    fix = per_gene_delta(d, "add_a1.0", "metric").rename("fixed")
    own = per_gene_delta(d, "add_own", "metric").rename("own")
    # alpha_own is stored as a per-LAYER json dict, e.g. '{"27": 1.5768}', and is recorded on
    # every row (it describes the gene, not the arm). Single-layer runs have exactly one key.
    def _alpha(v):
        try:
            vals = list(json.loads(v).values())
            return float(vals[0]) if len(vals) == 1 else float(np.mean(vals))
        except Exception:
            return np.nan
    aow = (d[d.condition == "add_own"].assign(a=lambda x: x.alpha_own.map(_alpha))
           .groupby("gene")["a"].mean().rename("alpha_own"))
    m = pd.concat([fix, own, aow], axis=1).dropna()
    m["diff"] = m["own"] - m["fixed"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=300)
    ax = axes[0]
    ax.hist(m.alpha_own, bins=40, color=INK, alpha=0.8)
    ax.axvline(1.0, color=WARM, lw=2.0, label="α = 1 (fixed-dose arm)")
    ax.set_xlabel(r"$\alpha_i = q_{loo,i}\ /\ \|v_{-i}\|$")
    ax.set_ylabel("genes")
    ax.set_title(f"Own-scaled dose per gene\nmedian {m.alpha_own.median():.2f}, "
                 f"{100*(m.alpha_own < 1).mean():.0f}% below α=1", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.scatter(m.fixed, m.own, s=9, color=INK, alpha=0.45)
    lim = [min(m.fixed.min(), m.own.min()) - 1, max(m.fixed.max(), m.own.max()) + 1]
    ax.plot(lim, lim, color=GREY, ls=":", lw=1.0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{spec['axis_delta'][:-5]}, fixed α=1 (pp)")
    ax.set_ylabel(f"{spec['axis_delta'][:-5]}, own-scaled α (pp)")
    ax.set_title(f"Per gene: own-scaled vs fixed\nmean Δ {m['diff'].mean():+.2f} pp, "
                 f"own wins in {100*(m['diff'] > 0).mean():.0f}% of genes", fontsize=9)

    ax = axes[2]
    ax.scatter(m.alpha_own, m["diff"], s=9, color=PLUM, alpha=0.5)
    ax.axhline(0, color=GREY, ls=":", lw=1.0)
    ax.axvline(1.0, color=WARM, lw=1.5)
    ax.set_xlabel(r"$\alpha_i$")
    ax.set_ylabel("Δ(own) − Δ(fixed)  (pp)")
    r = m.alpha_own.corr(m["diff"], method="spearman")
    ax.set_title(f"Does the dose gap explain the difference?\nSpearman ρ = {r:+.3f}", fontsize=9)

    fig.suptitle("Two dose conventions: identical intervention (α=1) vs identical effect size "
                 "(α_i matched to the gene's own shift)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"6c_alpha_convention.{ext}", bbox_inches="tight")
    m.reset_index().to_csv(out / "6c_alpha_convention.csv", index=False)
    print(f"[wrote] {out}/6c_alpha_convention.png")

    print("\n=== fixed vs own-scaled alpha ===")
    print(f"  genes                      {len(m)}")
    print(f"  alpha_own  median {m.alpha_own.median():.3f}  IQR "
          f"{m.alpha_own.quantile(.25):.3f}-{m.alpha_own.quantile(.75):.3f}  "
          f"range {m.alpha_own.min():.3f}-{m.alpha_own.max():.3f}")
    print(f"  Δ fixed α=1                {m.fixed.mean():+.3f} pp")
    print(f"  Δ own-scaled               {m.own.mean():+.3f} pp")
    print(f"  paired difference          {m['diff'].mean():+.3f} pp "
          f"(own wins {100*(m['diff'] > 0).mean():.1f}% of genes)")
    lo, hi = boot_ci(m["diff"].values)
    print(f"  bootstrap 95% CI on diff   [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
