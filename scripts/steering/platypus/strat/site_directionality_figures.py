"""Figures for the site-directionality decomposition. Reads `site_directionality/`, no recompute."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gcf = _load("gc_codon_figures")
figures_dir, layer_suffix, dose_ladder = _gcf.figures_dir, _gcf.layer_suffix, _gcf.dose_ladder

GC_CLASSES = ["gc_up", "gc_down", "gc_neutral"]
GC_LABELS = {"gc_up": "GC-increasing\nA/T → G/C", "gc_down": "GC-decreasing\nG/C → A/T",
             "gc_neutral": "GC-neutral\nA↔T, G↔C"}
# GC up/down is a signed contrast, so it takes a warm/cool pair around a neutral.
GC_COLOURS = {"gc_up": acs.apc.dragon, "gc_down": acs.apc.aegean,
              "gc_neutral": acs.SERIES_MUTED}
SET_LABEL = {"private": "private bp (platypus base carried by no other sampled mammal)",
             "platy_not_human": "platy bp, not human (pairwise difference)"}
EXT = ("png", "pdf")

# Figures 9a and 9b use matching half-page panels.
PUB_PANEL_H = 500.0

# Short site-set names for the publication panel titles: SET_LABEL's parenthetical gloss is a
# sentence long and overruns a 500 pt panel at 17 pt. The gloss belongs in the caption.
PUB_SET_LABEL = {"private": "Private bp", "platy_not_human": "Platypus bp, not human"}


def dose_labels(suffix: str) -> dict[str, str]:
    return dict(zip(dose_ladder(suffix), ["0", "0.5", "1", "2", "3", "4"]))


def arms_for_layer(summ: pd.DataFrame, layer: int) -> tuple[list[str], list[str]]:
    """(dose-ladder conditions, other arms) present for one steering layer."""
    suffix = layer_suffix(layer)
    have = set(summ.condition)
    ladder = [c for c in dose_ladder(suffix) if c in have and c != "unsteered"]
    other = sorted(c for c in have if c not in set(dose_ladder(suffix)) and (
        c.endswith("_L24") if suffix else not c.endswith("_L24")))
    return ladder, other


def _thin(summ: pd.DataFrame, min_frac: float = 0.95) -> set[str]:
    """Conditions covering well under the full panel -- flagged in every label, never silently mixed."""
    full = int(summ.n_genes_total.max())
    return set(summ.loc[summ.n_genes_total < min_frac * full, "condition"])


def label_for(cond: str, dl: dict[str, str], thin: set[str], n: int | None = None,
              strip_alpha: bool = True) -> str:
    """Display label for a condition."""
    base = f"α={dl[cond]}" if cond in dl else cond.replace("_L24", "")
    if strip_alpha:
        base = base.replace("_a1.0", "")
    if cond in thin:
        base += f"\n(n={n} genes)" if n else "\n(partial)"
    return base


def fig18(summ: pd.DataFrame, out: Path, layer: int, site_set: str, sfx: str = "",
          variant_arms: bool = False) -> pd.DataFrame:
    """Delta leave-human rate against delta platypus-choice rate: the mechanism quadrants."""
    s = summ[summ.site_set == site_set].set_index("condition")
    ladder, other = arms_for_layer(summ, layer)
    dl, thin = dose_labels(layer_suffix(layer)), _thin(summ)

    set_pub_style(title_size=9, tick_size=7)
    fig, ax = plt.subplots(figsize=(pub.size(pub.HALF, PUB_PANEL_H) if pub.is_on()
                                    else (6.1, 5.2)))
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.7, zorder=1)
    ax.axvline(0, color=acs.ZERO_LINE, lw=0.7, zorder=1)

    def area(cond):
        return 28 + 340 * min(abs(float(s.loc[cond, "d_K"])), 1.5) / 1.5

    lad = [c for c in ladder if c in s.index]
    if len(lad) > 1:
        x, y = s.loc[lad, "d_L"].to_numpy(), s.loc[lad, "d_C"].to_numpy()
        for i in range(len(lad) - 1):
            ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->", color=acs.apc.dragon, lw=1.1,
                                        alpha=0.75, shrinkA=6, shrinkB=6), zorder=2)
    for c in lad:
        ax.scatter(s.loc[c, "d_L"], s.loc[c, "d_C"], s=area(c), color=acs.apc.dragon,
                   edgecolor="white", lw=0.6, zorder=3)
        # At 15 pt on a 500 pt panel "alpha=2" and "alpha=3" overlap, so the publication label is
        # the bare dose and the key says what the number is.
        lab = dl.get(c, label_for(c, dl, thin, int(s.loc[c, "n_genes_total"]))) if pub.is_on() \
            else label_for(c, dl, thin, int(s.loc[c, "n_genes_total"]))
        ax.annotate(lab, (s.loc[c, "d_L"], s.loc[c, "d_C"]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=6.5, color=acs.apc.cinnabar, zorder=5)
    # THE UNSTEERED BASELINE IS THE ORIGIN. Both axes are changes measured against the same gene's
    # unsteered cell, so unsteered sits at (0, 0) by construction -- it has no coordinates of its
    # own and is absent from summary.csv, which only holds the non-baseline arms. Drawing it makes
    # the reference explicit instead of leaving the reader to infer that the crosshair is an arm.
    ax.scatter(0, 0, s=42, color=acs.SERIES_MUTED, marker="o", edgecolor=acs.apc.white, lw=0.8,
               zorder=4)
    # Labelled to the lower RIGHT: the random arms all sit just left of the origin at C ~ 0, so a
    # left-hand label overprints theirs, while the fourth quadrant is empty in every run of this
    # figure (a steered arm with L up and C down would be a result, not a layout problem).
    # Labelled to the lower RIGHT, no leader line: the random arms are all to the LEFT of the origin
    # and now label leftward too, so this side is clear.
    ax.annotate("unsteered", (0, 0), textcoords="offset points",
                xytext=(8, -7), ha="left", va="top", fontsize=5.8, color=acs.ANNOTATION, zorder=5)

    others = [c for c in other if c in s.index]
    if not variant_arms:
        skipped = [c for c in others if "random" not in c]
        others = [c for c in others if "random" in c]
        if skipped:
            print(f"  fig18: not plotting {len(skipped)} variant arm(s) "
                  f"(in the CSV, plotted=False): {', '.join(skipped)}")
    # The non-ladder arms land almost on top of each other -- the three random rungs sit within
    # 0.3 pp of each other and of the origin -- so ANY fixed offset ring puts one arm's label on
    # top of another's, or nearer a neighbour's marker than its own. Instead the labels are STACKED
    # in a column to the right of the cluster, ordered by the arm's own C so a leader line never
    # crosses its neighbour's, and each is tied back to its marker. Guaranteed collision-free for
    # any number of arms, which a hand-tuned offset list is not.
    for c in others:
        col = acs.STEER_COLORS["random"] if "random" in c else acs.STEER_COLORS["target"]
        x, y = float(s.loc[c, "d_L"]), float(s.loc[c, "d_C"])
        ax.scatter(x, y, s=area(c), color=col, marker="s",
                   edgecolor="white", lw=0.6, zorder=3)
        # Place labels left of the marker to avoid the origin and unsteered label.
        # strip_alpha=False: the random arms differ ONLY in alpha, so a bare "random" beside
        # "random_a0.5" would read as a dose-free arm rather than the alpha=1 rung.
        if not pub.is_on():
            ax.annotate(label_for(c, dl, thin, int(s.loc[c, "n_genes_total"]), strip_alpha=False),
                        (x, y), textcoords="offset points", xytext=(-9, 0),
                        ha="right", va="center", fontsize=5.8,
                        color=acs.apc.charcoal if "random" in c else acs.apc.depths, zorder=5)

    # Asymmetric padding protects labels near the lower-left data boundary.
    # Left/bottom get more room than right/top: the random cluster sits at the bottom-left corner of
    # the data and its leader-line labels point outward, so a symmetric pad clips them.
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(xl[0] - 0.16 * (xl[1] - xl[0]), xl[1] + 0.08 * (xl[1] - xl[0]))
    ax.set_ylim(yl[0] - 0.14 * (yl[1] - yl[0]), yl[1] + 0.10 * (yl[1] - yl[0]))
    if pub.is_on():
        # "L" and "C" are the summary CSV's column names, not something a reader can decode off
        # the page. The two panels are otherwise identical, so the site set has to stay on the
        # artwork: it is what tells 9a from 9b.
        ax.set_xlabel("Δ leave-human rate (pp)")
        ax.set_ylabel("Δ platypus-choice rate (pp)")
        ax.set_title(PUB_SET_LABEL[site_set])
        ax.legend(handles=[
            Line2D([], [], marker="o", ls="none", color=acs.apc.dragon, ms=8,
                   label="Dose ladder (label = alpha)"),
            Line2D([], [], marker="s", ls="none", color=acs.STEER_COLORS["random"], ms=8,
                   label="Norm-matched random"),
        ], frameon=False, loc="lower right", handletextpad=0.5)
        pub.tight(fig)
        pub.finish(fig, f"18_leave_human_vs_platypus_choice{sfx}", directory=out)
    else:
        ax.set_xlabel("Δ L  (pp)")
        ax.set_ylabel("Δ C  (pp)")
        ax.set_title(f"Mechanism of the recovery gain — blocks.{layer}\n{SET_LABEL[site_set]}")
        fig.tight_layout()
        for e in EXT:
            fig.savefig(out / f"18_leave_human_vs_platypus_choice{sfx}.{e}", dpi=300,
                        bbox_inches="tight")
    plt.close(fig)

    # The CSV keeps EVERY arm even when the figure hides some -- the numbers are valid and
    # `add_gc_removed` in particular is the run's main mechanistic result. `plotted` records what the
    # figure actually showed, so the table can never be mistaken for the panel's contents.
    keep = [c for c in lad + other if c in s.index]
    tab = s.loc[keep, ["n_genes_total", "d_A_cov", "d_A_all", "d_L", "d_C", "d_K",
                       "d_excess"]].copy()
    shown = set(lad) | set(others)
    tab.insert(0, "plotted", [c in shown for c in tab.index])
    tab.to_csv(out / f"18_leave_human_vs_platypus_choice{sfx}.csv")
    return tab


def fig19(gc: pd.DataFrame, summ: pd.DataFrame, out: Path, layer: int,
          site_set: str, sfx: str = "") -> pd.DataFrame:
    """Recovery change split by the GC direction of the human -> platypus substitution."""
    g = gc[gc.site_set == site_set]
    ladder, other = arms_for_layer(summ, layer)
    conds = [c for c in ladder + other if c in set(g.condition)]
    dl, thin = dose_labels(layer_suffix(layer)), _thin(summ)

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    lad = [c for c in ladder if c in set(g.condition)]
    for cls in GC_CLASSES:
        d = g[(g.gc_class == cls)].set_index("condition").reindex(lad)
        ax.plot(range(len(lad)), d.d_A_mean, "o-", color=GC_COLOURS[cls], lw=1.3, ms=4.5,
                label=f"{GC_LABELS[cls]}  ({d.n_sites_per_gene.mean():.0f} sites/gene)")
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.7)
    ax.set_xticks(range(len(lad)), [dl.get(c, c) for c in lad], fontsize=7)
    ax.set_xlabel("α")
    ax.set_ylabel("Δ recovery (pp)")
    ax.set_title(f"Recovery change by GC direction of the site — blocks.{layer}\ndose ladder")
    ax.legend(fontsize=6, frameon=False, loc="best")

    ax = axes[1]
    w = 0.26
    for i, cls in enumerate(GC_CLASSES):
        d = g[g.gc_class == cls].set_index("condition").reindex(conds)
        ax.bar(np.arange(len(conds)) + (i - 1) * w, d.d_A_mean, width=w,
               color=GC_COLOURS[cls], label=GC_LABELS[cls].split("\n")[0])
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.7)
    ngenes = summ[summ.site_set == site_set].set_index("condition")["n_genes_total"]
    ax.set_xticks(range(len(conds)),
                  [label_for(c, dl, thin, int(ngenes[c])) for c in conds],
                  rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Δ recovery (pp)")
    ax.set_title("all arms")
    ax.legend(fontsize=6, frameon=False)
    fig.suptitle(SET_LABEL[site_set], fontsize=7.5, y=1.02, color=acs.ANNOTATION)
    fig.tight_layout()
    for e in EXT:
        fig.savefig(out / f"19_gc_class_recovery{sfx}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tab = g[g.condition.isin(conds)].pivot_table(
        index="condition", columns="gc_class",
        values=["d_A_mean", "d_A_frac_improved", "n_sites_per_gene"]).reindex(conds)
    tab.to_csv(out / f"19_gc_class_recovery{sfx}.csv")
    return tab


def fig20(summ: pd.DataFrame, per: pd.DataFrame, out: Path, layer: int,
          site_set: str, sfx: str = "") -> pd.DataFrame:
    """Raw recovery change against excess over the composition-matched shuffle."""
    s = summ[summ.site_set == site_set].set_index("condition")
    ladder, other = arms_for_layer(summ, layer)
    conds = [c for c in ladder + other if c in s.index]
    dl, thin = dose_labels(layer_suffix(layer)), _thin(summ)
    labs = [label_for(c, dl, thin, int(s.loc[c, "n_genes_total"])) for c in conds]

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    x = np.arange(len(conds))
    ax.bar(x - 0.2, s.loc[conds, "d_A_cov"], width=0.4, color=acs.apc.cloud,
           label="raw Δ recovery")
    ax.bar(x + 0.2, s.loc[conds, "d_excess"], width=0.4, color=acs.apc.aster,
           label="Δ excess (over the shuffle null)")
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.7)
    for i, c in enumerate(conds):
        raw, exc = float(s.loc[c, "d_A_cov"]), float(s.loc[c, "d_excess"])
        if raw > 0.4:
            ax.annotate(f"{100 * exc / raw:.0f}%", (i, max(raw, exc)),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5.5,
                        color=acs.apc.tanzanite)
    ax.set_xticks(x, labs, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Δ vs unsteered (pp)")
    ax.set_title(f"How much of the gain survives the composition null — blocks.{layer}\n"
                 "% = the share of the raw gain that is excess")
    ax.legend(fontsize=6.5, frameon=False)

    # observed per-gene spread of the excess change: the only uncertainty statement available
    ax = axes[1]
    p = per[per.site_set == site_set]
    base = p[p.condition == "unsteered"].set_index("gene")["excess"]
    show = [c for c in conds if c not in thin][:9]
    data, keep = [], []
    for c in show:
        d = p[p.condition == c].set_index("gene")["excess"]
        common = base.index.intersection(d.index)
        v = (d.loc[common] - base.loc[common]).dropna()
        if len(v) > 10:
            data.append(v.to_numpy())
            keep.append(c)
    if data:
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for b in parts["bodies"]:
            b.set_facecolor(acs.apc.aster)
            b.set_alpha(0.35)
            b.set_edgecolor(acs.apc.tanzanite)
        ax.scatter(range(1, len(data) + 1), [np.mean(v) for v in data], s=14, color=acs.apc.tanzanite,
                   zorder=4, label="mean")
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.7)
        ax.set_xticks(range(1, len(keep) + 1),
                      [label_for(c, dl, thin) for c in keep], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Δ excess recovery per gene (pp)")
    ax.set_title("observed per-gene spread\n(no resampling)")
    fig.suptitle(SET_LABEL[site_set], fontsize=7.5, y=1.02, color=acs.ANNOTATION)
    fig.tight_layout()
    for e in EXT:
        fig.savefig(out / f"20_excess_vs_raw_recovery{sfx}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tab = s.loc[conds, ["n_genes_total", "d_A_cov", "d_A_null", "d_excess",
                        "d_excess_median", "d_excess_sd", "d_excess_frac_improved"]].copy()
    tab["excess_share_of_gain"] = tab.d_excess / tab.d_A_cov
    tab.to_csv(out / f"20_excess_vs_raw_recovery{sfx}.csv")
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", default="site_directionality")
    ap.add_argument("--layers", type=int, nargs="*", default=[27, 24])
    ap.add_argument("--site-set", default="private",
                    choices=["private", "platy_not_human", "both"])
    ap.add_argument("--variant-arms", action="store_true",
                    help="fig18: also draw the non-ladder, non-random arms (cross_gene, panel_*, "
                         "add_own, add_cone_removed, add_gc_removed). Off by default -- most are "
                         "not readable off that panel (cross_gene is vacuous, the panel_* pair is "
                         "an unlicensed tier-C contrast), and six squares crowded out the dose "
                         "ladder. They are always in the CSV with plotted=False.")
    ap.add_argument("--pub", action="store_true",
                    help="render the PUBLICATION figure (fig18 only, draft 9a/9b): exact 500 pt "
                         "half panels so the two site sets sit side by side, the style guide's "
                         "15 pt type with monospaced numerals, and axis titles spelled out rather "
                         "than the CSV's L/C column names. Writes into <figures>/pub/.")
    args = ap.parse_args()

    if args.pub:
        pub.enable()
    d = args.run / args.data
    summ = pd.read_csv(d / "summary.csv")
    gc = pd.read_csv(d / "gc_class.csv")
    per = pd.read_csv(d / "per_gene.csv")

    sets = ["private", "platy_not_human"] if args.site_set == "both" else [args.site_set]
    for layer in args.layers:
        out = figures_dir(args.run, layer)
        out.mkdir(parents=True, exist_ok=True)
        for site_set in sets:
            # The headline site set keeps the bare figure name; the secondary one is suffixed. The
            # suffix goes into the FILENAME rather than being renamed afterwards -- renaming after
            # the fact overwrites the headline figure before moving it.
            sfx = "" if site_set == "private" else f"_{site_set}"
            print(f"\n=== blocks.{layer}  |  {site_set} -> {out} ===")
            t18 = fig18(summ, out, layer, site_set, sfx, variant_arms=args.variant_arms)
            t19 = fig19(gc, summ, out, layer, site_set, sfx)
            t20 = fig20(summ, per, out, layer, site_set, sfx)
            print(t18.to_string(float_format=lambda z: f"{z:.2f}"))
            print("\nexcess share of the raw gain:")
            print(t20[["d_A_cov", "d_excess", "excess_share_of_gain"]]
                  .to_string(float_format=lambda z: f"{z:.3f}"))
            _ = t19


if __name__ == "__main__":
    main()
