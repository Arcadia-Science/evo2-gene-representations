"""Top-15 genes by each steering criterion, with every readout shown for each of them."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402
from steering_delta_strip import CONS_CMAP, SITE_KEY, load_deltas, readouts  # noqa: E402
from strat_metric import add_metric_args, coverage_note, load_scores  # noqa: E402

INK = acs.SERIES_PRIMARY
WARM = acs.SERIES_NULL
GREY = acs.SERIES_MUTED
# Diverging: warm/cool poles through a NEUTRAL grey midpoint -- zero must read as "nothing happened"
DIVERGE = LinearSegmentedColormap.from_list(
    "warm_ink", [WARM, acs.apc.shell, acs.apc.gray, acs.apc.denim, INK])

# Derive ranking labels and CSV columns from the selected site-set specification.
def spec_of(mspec: dict) -> dict:
    return {k: (blk, lab, arrow) for k, blk, lab, arrow, _ in readouts(mspec)}


def short_of(mspec: dict) -> dict:
    return {SITE_KEY: mspec["short"], "aa_id_to_target": "aa identity\n(pp)",
            "indel_frac": "indel burden\n(pp)", "stop_density": "stops/100\ncodons (pp)"}


def criteria_of(mspec: dict) -> list:
    """criterion label, ranking column, basis ("delta" = biggest change, "level" = best absolute
    value after steering), gloss."""
    return [
        (mspec["report"]["metric"] + " bp", SITE_KEY, "delta",
         f"largest gain in {mspec['level_label']} — {mspec['gloss']}"),
        ("amino acid", "aa_id_to_target", "delta",
         "largest gain in amino-acid identity to platypus"),
        ("coherence", "coherence_level", "level",
         "most coherent sequences AFTER steering — fewest indels and premature stops in absolute "
         "terms, however they started"),
    ]


def num(v: float) -> str:
    """Two decimals for the small readouts, one for the large -- same width either way."""
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def per_gene_se(run: Path, arm_dir: str, cond: str, mspec: dict, *,
                scores: str | None = None, min_voters: int = 1) -> pd.DataFrame:
    """SE of each gene's delta from the sample spread within its two cells."""
    s, _ = load_scores(run, arm_dir, scores=scores, metric=mspec["report"]["metric"],
                       min_voters=min_voters, quiet=True)
    for key, _, _, _, derive in readouts(mspec):
        if derive is not None:
            s[key] = derive(s)
    keys = list(spec_of(mspec))
    out = []
    for key in keys:
        g = s.groupby(["gene", "condition"])[key].agg(["var", "count"])
        v = g["var"].unstack()
        n = g["count"].unstack()
        se = np.sqrt(v[cond] / n[cond] + v["unsteered"] / n["unsteered"])
        out.append(pd.DataFrame({"gene": se.index, "metric": key, "se": se.values}))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path,
                    default=ROOT / "results" / "2026-08-08_platypus-strat-400")
    ap.add_argument("--arm-dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--condition", default="add_a1.0")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--stem", default="11_top_genes")
    ap.add_argument("--out", type=Path, default=None)
    add_metric_args(ap)
    args = ap.parse_args()

    run = args.run
    out = args.out or (run / "figures")
    out.mkdir(parents=True, exist_ok=True)
    set_pub_style(title_size=9, tick_size=7)

    long, mspec = load_deltas(run, args.arm_dir, [args.condition], "pp",
                              scores=args.scores, metric=args.metric,
                              min_voters=args.min_voters)
    SPEC, SHORT, CRITERIA = spec_of(mspec), short_of(mspec), criteria_of(mspec)
    wide = long.pivot(index="gene", columns="metric", values="delta")
    base = long.pivot(index="gene", columns="metric", values="baseline")
    meta = long.groupby("gene")[["perc_id_hp", "stratum"]].first()
    # HGNC symbols, extracted once from the local Ensembl 111 GTF (all 400 panel genes resolve);
    # fall back to the ENSG when the file is absent so the figure never depends on it
    sym_path = run / "stage1" / "gene_symbols.tsv"
    if sym_path.exists():
        sym = pd.read_csv(sym_path, sep="\t").set_index("gene").symbol
        meta["symbol"] = pd.Series(meta.index.map(sym), index=meta.index).fillna(
            pd.Series(meta.index, index=meta.index))
    else:
        meta["symbol"] = meta.index
    # NB `metric` here is the long frame's readout-NAME column, unrelated to the site-set alias.
    se = per_gene_se(run, args.arm_dir, args.condition, mspec, scores=args.scores,
                     min_voters=args.min_voters).pivot(index="gene", columns="metric", values="se")

    keys = list(SPEC)

    def cname(k: str) -> str:
        """CSV column name: spell out the site set rather than saving a generic `metric`."""
        return mspec["col"] if k == SITE_KEY else k

    # the STEERED level of each readout -- delta is measured against the gene's own unsteered cell,
    # so adding it back recovers the absolute value the steered sequences actually achieved
    steer = wide[keys] + base[keys]

    def z(v):
        return (v - v.mean()) / v.std(ddof=0)

    # Coherence ranks on the LEVEL, not the change: the question is which steered sequences came out
    # most coherent overall, not which improved most from a bad start. Both readouts are "lower is
    # better" so the sign flips, and each is z-scored across all 398 genes so the noisier one cannot
    # dominate the composite.
    wide["coherence_level"] = -0.5 * (z(steer.indel_frac) + z(steer.stop_density))
    wide["coherence_gain"] = 0.5 * (z(-wide.indel_frac) + z(-wide.stop_density))
    # per-readout colour scale: the 95th percentile of |delta| over ALL genes, so a saturated cell
    # means "extreme for this readout", not "large in pp"
    scale = {k: max(float(np.nanpercentile(np.abs(wide[k]), 95)), 1e-6) for k in keys}

    fig, axes = plt.subplots(1, len(CRITERIA), figsize=(17.4, 6.6))
    fig.subplots_adjust(wspace=0.42)
    cnorm = Normalize(vmin=float(meta.perc_id_hp.min()), vmax=100.0)
    dnorm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    rows = []

    for ax, (title, rank_key, basis, gloss) in zip(axes, CRITERIA):
        top = wide.sort_values(rank_key, ascending=False).head(args.top)
        genes = list(top.index)
        n = len(genes)

        M = np.array([[wide.loc[g, k] / scale[k] * SPEC[k][2] for k in keys] for g in genes])
        ax.imshow(np.clip(M, -1, 1), cmap=DIVERGE, norm=dnorm, aspect="auto",
                  extent=(-0.5, len(keys) - 0.5, n - 0.5, -0.5))
        # Each cell states where the gene started and where it ended: "unsteered → steered". The
        # fill still encodes the change, so level and change are both on the page and neither has to
        # be inferred from the other.
        for i, g in enumerate(genes):
            for j, k in enumerate(keys):
                shade = abs(np.clip(M[i, j], -1, 1))
                col = acs.apc.white if shade > 0.62 else acs.apc.black
                ax.text(j, i - 0.14, f"{num(base.loc[g, k])} → {num(steer.loc[g, k])}",
                        ha="center", va="center", fontsize=6.3, color=col)
                ax.text(j, i + 0.24, f"({wide.loc[g, k]:+.2f})", ha="center", va="center",
                        fontsize=5.4, color=col, alpha=0.78)
        # thin white gridlines so each cell reads as its own mark
        for j in range(len(keys) + 2):
            ax.axvline(j - 0.5, color="white", lw=1.4)
        for i in range(n + 1):
            ax.axhline(i - 0.5, color="white", lw=1.4)
        # a wider gutter before the conservation column: it is a sequential magnitude sitting next
        # to diverging Δ cells, and without the break a dark swatch there reads as "better"
        ax.axvline(len(keys) - 0.5, color="white", lw=5.0, zorder=3)

        ax.set_xticks(range(len(keys) + 1))
        ax.set_xticklabels([*(SHORT[k] for k in keys), "conservation\n(%)"], fontsize=6.8)
        ax.set_yticks(range(n))
        ax.set_yticklabels([meta.loc[g, "symbol"] for g in genes], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # wrapped: the coherence gloss is long enough to run into the next panel's title
        ax.set_title(f"top {n} by {title}\n" + textwrap.fill(gloss, 52), fontsize=8.5)

        # conservation as a fifth column, drawn on the SEQUENTIAL ramp: it is a magnitude, a
        # different job from the diverging Δ cells, so it must not borrow their colours. Kept inside
        # the axes rather than hung off the right edge, where it would overrun the next panel.
        cx = len(keys)
        for i, g in enumerate(genes):
            c = float(meta.loc[g, "perc_id_hp"])
            ax.add_patch(plt.Rectangle((cx - 0.5, i - 0.5), 1.0, 1.0,
                                       facecolor=CONS_CMAP(cnorm(c)), edgecolor=acs.apc.white, lw=1.4))
            ax.text(cx, i, f"{c:.0f}", ha="center", va="center", fontsize=6.6,
                    color=acs.apc.white if cnorm(c) > 0.62 else acs.apc.black)
        ax.set_xlim(-0.5, cx + 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        for g in genes:
            rows.append({"criterion": title, "gene": g,
                         "symbol": meta.loc[g, "symbol"],
                         "perc_id_hp": round(float(meta.loc[g, "perc_id_hp"]), 2),
                         "stratum": int(meta.loc[g, "stratum"]),
                         "ranked_on": basis,
                         **{f"unsteered_{cname(k)}": round(float(base.loc[g, k]), 3)
                            for k in keys},
                         **{f"steered_{cname(k)}": round(float(steer.loc[g, k]), 3)
                            for k in keys},
                         **{f"delta_{cname(k)}": round(float(wide.loc[g, k]), 3) for k in keys},
                         **{f"se_delta_{cname(k)}": round(float(se.loc[g, k]), 3) for k in keys},
                         "coherence_level": round(float(wide.loc[g, "coherence_level"]), 3),
                         "coherence_gain": round(float(wide.loc[g, "coherence_gain"]), 3)})

    sm = plt.cm.ScalarMappable(norm=dnorm, cmap=DIVERGE)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.10,
                      aspect=55, ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["worse", "unsteered", "better"], fontsize=7)
    cb.set_label("cell fill = per-readout Δ, sign-corrected so ink = better, normalised by that "
                 "readout's 95th percentile |Δ|.   Cell text = unsteered → steered, with the Δ "
                 "beneath.", fontsize=7)
    cb.outline.set_visible(False)

    fig.suptitle(f"Which genes steer best, and what it costs them elsewhere  "
                 f"({args.condition}, {args.arm_dir.replace('stage4_', '')})\n"
                 + coverage_note(mspec), y=0.99, fontsize=10)
    fig.text(0.5, 0.005,
             "Winner's curse: these are the top tail of 398 genes measured from 5 samples each, so "
             "the selected Δ overstate the truth and will regress on a re-run — the per-gene se is "
             "in the CSV.\nPopulation estimates belong to figure 10, not here.",
             ha="center", va="top", fontsize=6.8, color=GREY)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"{args.stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"{args.stem}.csv", index=False)
    print(f"  {args.stem}  ->  {out}")
    show = ["criterion", "symbol", "perc_id_hp", f"steered_{mspec['col']}",
            "steered_aa_id_to_target", "steered_indel_frac", "steered_stop_density",
            "unsteered_indel_frac", "unsteered_stop_density"]
    print(tab[show].to_string(index=False))


if __name__ == "__main__":
    main()