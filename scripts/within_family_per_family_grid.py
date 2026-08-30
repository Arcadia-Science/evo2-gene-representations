"""Per-family "all metrics" grid — is a within-family baseline confounded by a composition control?"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from layer_sweep_summary import _pub_family_label  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

COLORS = acs.categorical(7)

# Height of one publication panel row, in points.
PUB_PANEL_H = 360.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True,
                    help="tidy within_family_vs_layer*.csv (layer, metric, family, rho)")
    ap.add_argument("--out-dir", default=None, help="default: alongside --csv")
    ap.add_argument("--title", default="")
    ap.add_argument("--exclude-metric", nargs="*", default=[])
    ap.add_argument("--ncols", type=int, default=8)
    ap.add_argument("--families", nargs="*", default=None,
                    help="plot only these families, in the order given (default: all, sorted). "
                         "Use with --out-suffix so the subset cannot overwrite the full grid.")
    ap.add_argument("--out-suffix", default="per_family",
                    help="output stem suffix. A families subset REQUIRES a non-default value: the "
                         "full grid and its metric-pairs CSV are the reference artefacts and a "
                         "3-family file must not land on their names.")
    ap.add_argument("--panel-w", type=float, default=2.05, help="panel width, inches")
    ap.add_argument("--panel-h", type=float, default=1.75, help="panel height, inches")
    ap.add_argument("--pub", action="store_true",
                    help="render at PUBLICATION geometry: an exact 1,000 pt panel, the style "
                         "guide's 15 pt type with monospaced numerals, and the metric key below "
                         "the panels. --panel-w/--panel-h are ignored. Writes into <out-dir>/pub/.")
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    df = pd.read_csv(args.csv)
    df = df[~df["metric"].isin(args.exclude_metric)]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv).parent
    stem = Path(args.csv).stem

    metrics = list(dict.fromkeys(df["metric"]))
    if args.families:
        if args.out_suffix == "per_family":
            raise SystemExit("--families needs --out-suffix (e.g. 'zoom3'); the default name is "
                             "the full grid's and would overwrite it and its metric_pairs.csv")
        missing = [f for f in args.families if f not in set(df["family"])]
        if missing:
            raise SystemExit(f"not in {args.csv}: {missing}\n"
                             f"available: {sorted(df['family'].dropna().unique())}")
        fams = list(args.families)                     # keep the caller's order, not alphabetical
        df = df[df.family.isin(fams)]
    else:
        fams = sorted(df["family"].dropna().unique())
    print(f"{len(fams)} families x {len(metrics)} metrics x {df.layer.nunique()} layers")

    # ── per-family metric-pair agreement across layers
    rows = []
    for fam in fams:
        sub = df[df.family == fam]
        w = sub.pivot_table(index="layer", columns="metric", values="rho")
        for a, b in itertools.combinations(metrics, 2):
            if a not in w or b not in w:
                continue
            ok = w[a].notna() & w[b].notna()
            rows.append({"family": fam, "metric_a": a, "metric_b": b,
                         "n_layers": int(ok.sum()),
                         "spearman_across_layers": (spearmanr(w[a][ok], w[b][ok]).statistic
                                                    if ok.sum() >= 4 else np.nan)})
    pairs = pd.DataFrame(rows)
    # Give subset runs a distinct filename.
    pair_csv = out_dir / (f"{stem}_metric_pairs.csv" if args.out_suffix == "per_family"
                          else f"{stem}_{args.out_suffix}_metric_pairs.csv")
    pairs.to_csv(pair_csv, index=False)
    print(f"wrote {pair_csv}")

    print("\nmedian across-layer Spearman between metric curves, over families:")
    med = (pairs.groupby(["metric_a", "metric_b"])["spearman_across_layers"]
           .agg(["median", "min", "max", "count"]).sort_values("median", ascending=False))
    print(med.round(3).to_string())

    # ── the grid
    # Type scales with the panel, so a 3-family zoom is not drawn with 48-panel small-multiple type.
    # Capped at 2x: past that the labels dominate a panel whose job is to show a curve shape.
    sc = min(args.panel_w / 2.05, 2.0)
    set_pub_style(title_size=round(7 * sc), tick_size=round(5 * sc))
    ncols = min(args.ncols, len(fams))
    nrows = int(np.ceil(len(fams) / ncols))
    if pub.is_on():
        # The publication figure is the 3-family zoom, so the row fits the full 1,000 pt at
        # ~310 pt a panel — wide enough for 15 pt type without stacking. `sc` is left at 1.0
        # so nothing below re-derives a size from the inches-based panel arguments.
        sc = 1.0
        fig, axes = plt.subplots(nrows, ncols, dpi=300, squeeze=False, sharex=True, sharey=True,
                                 figsize=pub.size(pub.FULL, PUB_PANEL_H * nrows))
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(args.panel_w * ncols, args.panel_h * nrows),
                                 dpi=300, squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()
    for i, fam in enumerate(fams):
        ax = axes[i]
        sub = df[df.family == fam]
        for j, m in enumerate(metrics):
            s = sub[sub.metric == m].sort_values("layer")
            if s["rho"].notna().sum() == 0:
                continue
            ax.plot(s["layer"], s["rho"], "-", lw=1.0 * sc, color=COLORS[j % len(COLORS)],
                    marker="o" if sc > 1.4 else None, ms=2.4 * sc, label=m)
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.5 * sc, ls="--")
        ax.set_title(_pub_family_label(fam) if pub.is_on() else fam, fontsize=round(6 * sc))
        ax.tick_params(labelsize=round(5 * sc))
    for j in range(len(fams), len(axes)):
        axes[j].axis("off")
    # FIGURE-level type does not scale with the panel: suptitle, shared axis labels and the legend
    # are sized to the figure, and at sc=1.8 they swallowed a 3-panel zoom. Panel-level type (titles,
    # ticks) still scales, which is the part that was unreadable when panels grew.
    fsc = min(sc, 1.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if pub.is_on():
        # Real per-axes labels, not supxlabel/supylabel. The figure-level versions are placed
        # against the FIGURE edge rather than the axes, so tight_layout does not reserve room
        # for them: the y one lands on top of the tick labels and the x one inside the key.
        # With sharey the y label is needed once, on the leftmost panel.
        for i, ax in enumerate(axes[:len(fams)]):
            ax.set_xlabel("layer")
            if i % ncols == 0:
                ax.set_ylabel("within-family Spearman ρ")
        panels_h = PUB_PANEL_H * nrows
        _, key_h = pub.key_below(fig, handles, labels, title="Baseline", width=pub.FULL)
        total_h = panels_h + key_h
        fig.set_size_inches(pub.FULL / 72.0, total_h / 72.0)
        pub.tight(fig, bottom=key_h)
        pub.drop_titles(fig)
        pub.finish(fig, f"{stem}_{args.out_suffix}", directory=out_dir)
        return
    # Clear the shared x-label by a fixed physical distance rather than a fixed axes fraction: the
    # same fraction is ~0.5 in on a 3 in figure and ~3 in on a 10 in one, so the legend either
    # overprints "layer" or drifts far below it as the grid changes shape.
    fig.legend(handles, labels, loc="lower center", ncol=min(len(metrics), 5),
               frameon=False, fontsize=round(7 * fsc),
               bbox_to_anchor=(0.5, -0.34 / (args.panel_h * nrows)))
    fig.suptitle(f"{args.title} — within-family ρ across layers, per family",
                 fontsize=round(11 * fsc), fontweight="bold", y=1.0)
    fig.supxlabel("layer", fontsize=round(8 * fsc))
    fig.supylabel("within-family Spearman ρ", fontsize=round(8 * fsc))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}_{args.out_suffix}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    print(f"wrote {out_dir}/{stem}_{args.out_suffix}.{{png,pdf}}")


if __name__ == "__main__":
    main()
