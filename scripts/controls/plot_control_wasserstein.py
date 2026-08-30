"""Between-family control preservation: centroid geodesic vs centroid-free Wasserstein."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

# Rung names as a reader sees them. The keys are the condition slugs used throughout the
# control pipeline; the ladder reads as "progressively more sequence composition destroyed",
# so the labels name the unit that was preserved rather than the function that shuffled it.
PUB_RUNG_LABELS = {
    "gc_match": "GC-matched",
    "dinuc_shuffle": "Dinucleotide shuffle",
    "kmer4_shuffle": "4-mer shuffle",
    "kmer6_shuffle": "6-mer shuffle",
    "synonymous_recode": "Synonymous recode (protein kept)",
    "natural": "Natural (self = 1.0)",
}

# Height of one stacked publication panel, in points.
PUB_PANEL_H = 330.0

# same rung colours as plot_control_layer_summaries so the two figures read as one set
COLORS = {
    "synonymous_recode": acs.CONTROL_COLORS["synonymous_recode"],
    "gc_match": acs.CONTROL_COLORS["gc_match"],
    "dinuc_shuffle": acs.CONTROL_COLORS["dinuc_shuffle"],
    "kmer4_shuffle": acs.CONTROL_COLORS["kmer4_shuffle"],
    "kmer6_shuffle": acs.CONTROL_COLORS["kmer6_shuffle"],
    "natural": acs.CONTROL_COLORS["natural"],
}
ORDER = ["gc_match", "dinuc_shuffle", "kmer4_shuffle", "kmer6_shuffle", "synonymous_recode"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="transcript_cdsmask")
    ap.add_argument("--slug", default="mammalian-orthologs-cdsmask-48fam")
    ap.add_argument(
        "--title",
        default="Evo2 mammalian orthologs (transcript, CDS-masked, 24 mammals, 48 families)",
    )
    ap.add_argument("--out-dir", default="results/layer_sweep_summaries")
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at PUBLICATION geometry: an exact 1,000 pt panel, the style "
        "guide's 15 pt type with monospaced numerals, the two panels STACKED "
        "rather than side by side, and the key below them. Writes into "
        "<out-dir>/pub/.",
    )
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    src = Path(f"results/_ot_control_preservation_{args.arm}.csv")
    if not src.exists():
        sys.exit(f"missing {src} — run controls_score_graphfree.py --axis between first")
    d = pd.read_csv(src).dropna(subset=["rho_wasserstein"])
    # Use graph-free angular preservation for within-family pairs.
    wsrc = Path(f"results/_angular_control_preservation_{args.arm}.csv")
    if not wsrc.exists():
        sys.exit(f"missing {wsrc} — run controls_score_graphfree.py --axis within")
    wdf = pd.read_csv(wsrc).rename(
        columns={"condition": "rung", "rho_within_angular": "rho_within"}
    )
    d["rung"] = d["condition"].str.replace(f"{args.arm}_", "", regex=False)
    rungs = [r for r in ORDER if r in set(d.rung)]
    missing = [r for r in ORDER if r not in rungs]
    print(f"rungs present: {rungs}")
    if missing:
        print(f"NOT YET SCORED (figure will say so): {missing}")

    set_pub_style(title_size=10, tick_size=8)
    # Stack publication panels to give each plot the full chart width.
    if pub.is_on():
        fig, axes = plt.subplots(
            2, 1, dpi=300, sharex=True, figsize=pub.size(pub.FULL, PUB_PANEL_H * 2)
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)
    label = (lambda r: PUB_RUNG_LABELS.get(r, r)) if pub.is_on() else (lambda r: r)
    ax = axes[0]
    for r in rungs:
        s_ = d[d.rung == r].sort_values("layer")
        ax.plot(
            s_.layer, s_.rho_wasserstein, "-o", ms=3, lw=1.6, color=COLORS.get(r), label=label(r)
        )
    ax.axhline(1.0, color=acs.REFERENCE_LINE, lw=1.2, label=label("natural"))
    ax.set_title("Between-family preservation (Wasserstein)", fontweight="bold")
    ax.set_ylabel("ρ (control vs natural)")
    ax.set_ylim(0.0, 1.05)

    ax = axes[1]
    for r in rungs:
        s_ = wdf[wdf.rung == r].sort_values("layer")
        if s_.empty:
            continue
        ax.plot(s_.layer, s_.rho_within, "-o", ms=3, lw=1.6, color=COLORS.get(r), label=label(r))
    ax.axhline(1.0, color=acs.REFERENCE_LINE, lw=1.2)
    ax.set_title("Within-family preservation (angular)", fontweight="bold")
    ax.set_ylabel("ρ (control vs natural)")
    ax.set_ylim(0.0, 1.05)
    for i, a in enumerate(axes):
        # Stacked panels share the layer axis, so only the bottom one carries its label —
        # the top would otherwise print "Layer" under an axis with no tick numbers on it.
        if not pub.is_on() or i == len(axes) - 1:
            a.set_xlabel("layer")
        if not pub.is_on():
            # The guide asks for no gridlines unless a reader has to read values off the chart;
            # these curves are read as shape and separation, not as numbers.
            a.grid(alpha=0.25, lw=0.5)

    out = Path(args.out_dir) / f"controls_layer_summary_{args.slug}-wasserstein"
    if pub.is_on():
        if missing:
            print(f"    NOTE: not yet scored, absent from the figure: {', '.join(missing)}")
        handles, labels = axes[0].get_legend_handles_labels()
        _, key_h = pub.key_below(fig, handles, labels, title="Control", width=pub.FULL)
        total_h = PUB_PANEL_H * 2 + key_h
        fig.set_size_inches(pub.FULL / 72.0, total_h / 72.0)
        pub.tight(fig, bottom=key_h)
        pub.finish(fig, out.name, directory=out.parent)
        return

    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    # Panel titles already say which metric each side uses, so the suptitle is just the panel
    # identity. The missing-rung warning still appends when something has not been scored — that
    # one changes how the figure should be read, so it is not decoration.
    sub = args.title
    if missing:
        sub += f"   [{', '.join(missing)} not yet scored]"
    fig.suptitle(sub, fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"[wrote] {out}.png")


if __name__ == "__main__":
    main()
