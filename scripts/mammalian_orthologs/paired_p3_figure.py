"""The `paired_p3` matched pair: does the latent geometry track the protein or the nucleotides?"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

RUN = ROOT / "results" / "2026-07-16_mammalian-orthologs-transcript_cdsmask"
SYN, MIS, FLOOR = "paired_p3_syn", "paired_p3_missense", "kmer6_shuffle"
# Evo2 taps the last block twice and blocks 28+ blow up in norm on this arm; see the module
# docstring.
DEGENERATE_BLOCKS = (28, 29, 30, 31)
# Height of one stacked publication panel, in points (matches plot_control_wasserstein, the
# sibling controls figure this one is read next to).
PUB_PANEL_H = 330.0
LAYERS = [L for L in range(32) if L not in DEGENERATE_BLOCKS]


def _style():
    """Repo figure style, guarded — a mid-refactor style module must not cost us the figure."""
    for mod, fn in (("arcadia_style", "setup"), ("plot_utils", "set_pub_style")):
        try:
            getattr(__import__(mod), fn)()
            return
        except Exception as e:  # noqa: BLE001
            print(f"  [style] {mod}.{fn}() unavailable ({type(e).__name__}: {e})")


def load_within() -> pd.DataFrame:
    d = pd.read_csv(RUN / "control_rho_by_layer.csv")
    return d.pivot_table(index="layer", columns="condition", values="rho_preservation").loc[LAYERS]


def load_between() -> pd.DataFrame:
    rows = []
    for L in LAYERS:
        s = pd.read_csv(RUN / f"blocks{L}/controls/control_between_scores.csv").set_index(
            "condition"
        )
        rows.append(
            {
                "layer": L,
                **{
                    c: s.loc[c, "rho_vs_natural_between"] for c in (SYN, MIS, FLOOR) if c in s.index
                },
            }
        )
    return pd.DataFrame(rows).set_index("layer")


def load_per_family(layer: int) -> pd.DataFrame:
    d = pd.read_csv(RUN / f"blocks{layer}/controls/control_within_scores.csv")
    d = d[d.condition.isin([SYN, MIS])]
    return d.pivot_table(
        index="family", columns="condition", values="rho_geodesic_vs_natural"
    ).dropna()


def paired_stats(layers: list[int]) -> pd.DataFrame:
    rows = []
    for L in layers:
        p = load_per_family(L)
        diff = p[MIS] - p[SYN]
        rows.append(
            {
                "layer": L,
                "n": len(diff),
                "mean_gap": diff.mean(),
                "frac_lower": (diff < 0).mean(),
                "wilcoxon_p": stats.wilcoxon(diff).pvalue,
            }
        )
    return pd.DataFrame(rows)


# ── figure


def figure(within: pd.DataFrame, between: pd.DataFrame, out: Path) -> None:
    """The two preservation panels, arms only."""
    import matplotlib

    matplotlib.use("Agg")
    import arcadia_pub as pub
    import matplotlib.pyplot as plt

    _style()
    import arcadia_style as acs

    C = acs.CONTROL_COLORS
    c_syn, c_mis = C[SYN], C[MIS]
    lab_syn, lab_mis = "Synonymous recoding", "Missense recoding"

    if pub.is_on():
        # Side by side, two panels across a 1,000 pt page leave ~470 pt each. Stacked, each gets
        # the full 940 pt of chart, and the shared layer axis means stacking removes a duplicate
        # x-axis rather than costing anything. Same call the sibling controls figure makes.
        fig, (axC, axB) = plt.subplots(
            2, 1, dpi=300, sharex=True, figsize=pub.size(pub.FULL, PUB_PANEL_H * 2)
        )
    else:
        fig = plt.figure(figsize=(11.0, 4.3))
        gs = fig.add_gridspec(1, 2, wspace=0.20, left=0.06, right=0.985, top=0.90, bottom=0.13)
        axC, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    for ax, tbl, ttl in (
        (axC, between, "Between-family (Wasserstein)"),
        (axB, within, "Within-family (angular distance)"),
    ):
        ax.plot(tbl.index, tbl[SYN], "-o", ms=3, lw=1.5, color=c_syn, label=lab_syn)
        ax.plot(tbl.index, tbl[MIS], "-o", ms=3, lw=1.5, color=c_mis, label=lab_mis)
        # Stacked pub panels share the layer axis, so only the bottom one carries its label.
        if not pub.is_on() or ax is axB:
            ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman \u03c1 vs. Natural")
        ax.set_title(ttl, **({"fontweight": "bold"} if pub.is_on() else {"fontsize": 7.5}))
        if not pub.is_on():
            ax.legend(
                title="Control", fontsize=6, loc="lower left", frameon=False, alignment="left"
            )
        # Y-limits still span the 6-mer floor even though the floor is no longer drawn, so the
        # arms keep the scale they had when it was -- the gap must not look bigger here than in
        # the ladder figures simply because a series was removed. Also leaves the key clear space.
        lo = min(tbl[SYN].min(), tbl[MIS].min(), tbl[FLOOR].min())
        hi = max(tbl[SYN].max(), tbl[MIS].max(), tbl[FLOOR].max())
        ax.set_ylim(lo - 0.30 * (hi - lo), hi + 0.03 * (hi - lo))

    if pub.is_on():
        # One shared key under both panels, in place of the two corner keys -- at 15 pt they
        # would each eat a corner of a 940 pt chart.
        handles, labels = axC.get_legend_handles_labels()
        _, key_h = pub.key_below(fig, handles, labels, title="Control", width=pub.FULL)
        fig.set_size_inches(pub.FULL / 72.0, (PUB_PANEL_H * 2 + key_h) / 72.0)
        pub.tight(fig, bottom=key_h)
        pub.finish(fig, out.name, directory=out.parent)
        print(f"    pub: {out.parent}/pub/{out.name}.png")
        return

    _key_rule(fig, (axB, axC), acs.apc.chateau)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"[wrote] {out}.png + .pdf")


def _key_rule(fig, axes, colour) -> None:
    """Draw the guide's rule under each key title."""
    from matplotlib.lines import Line2D

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        leg = ax.get_legend()
        if leg is None or leg.get_title() is None:
            continue
        bb, tb = leg.get_window_extent(r), leg.get_title().get_window_extent(r)
        inv = ax.transAxes.inverted()
        (x0, y), (x1, _) = inv.transform((bb.x0 + 2, tb.y0 - 3)), inv.transform((bb.x1 - 2, tb.y0))
        ax.add_line(
            Line2D(
                [x0, x1],
                [y, y],
                transform=ax.transAxes,
                color=colour,
                lw=0.8,
                clip_on=False,
                zorder=5,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--peak-layer",
        type=int,
        default=18,
        help="layer for the per-family panel (default 18, the deepest gap)",
    )
    ap.add_argument("--out", default=str(RUN / "paired_p3_protein_vs_nucleotide"))
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at publication geometry/type into <out-dir>/pub/ "
        "(1,000 pt wide, panels stacked, shared key below)",
    )
    a = ap.parse_args()
    if a.pub:
        import arcadia_pub as pub

        pub.enable()

    within, between = load_within(), load_between()
    stats_tbl = paired_stats(LAYERS)
    w, b = within[MIS] - within[SYN], between[MIS] - between[SYN]
    print(
        f"within  gap: mean {w.mean():+.4f}, lower in {(w < 0).sum()}/{len(w)} layers, "
        f"deepest {w.min():+.4f} @ block {w.idxmin()}"
    )
    print(
        f"between gap: mean {b.mean():+.4f}, lower in {(b < 0).sum()}/{len(b)} layers, "
        f"deepest {b.min():+.4f} @ block {b.idxmin()}"
    )

    row = stats_tbl[stats_tbl.layer == a.peak_layer].iloc[0]
    print(
        f"block {a.peak_layer}: missense lower in "
        f"{int(row.frac_lower * row.n)}/{int(row.n)} families, "
        f"mean gap {row.mean_gap:+.3f}, Wilcoxon p = {row.wilcoxon_p:.1e}"
    )

    out = Path(a.out)
    stats_tbl.to_csv(f"{out}_paired_stats.csv", index=False)
    figure(within, between, out)


if __name__ == "__main__":
    main()
