"""Figure 3: do composition controls reproduce Evo2's within-family signal?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import arcadia_style as acs  # noqa: E402
from plot_utils import control_preservation_figure, grouped_rho_bars, set_pub_style  # noqa: E402

CONTROL_ROOT = Path("data/evo2_gene_families/controls")
# (key, label, control_dir or None for natural, colour). Ordered as a composition
# gradient: natural → 6-mer → 4-mer → codon(3) → dinucleotide(2) → GC(1) preserved, plus
# synonymous-recode (preserves PROTEIN, scrambles nucleotides) as the orthogonal rung.
_C = acs.CONTROL_COLORS
CONDITIONS = [
    ("natural", "Natural", None, _C["natural"]),
    ("kmer6_shuffle", "6-mer shuffle", "kmer6_shuffle", _C["kmer6_shuffle"]),
    ("kmer4_shuffle", "4-mer shuffle", "kmer4_shuffle", _C["kmer4_shuffle"]),
    ("codon_shuffle", "Codon shuffle", "codon_shuffle", _C["codon_shuffle"]),
    ("dinuc_shuffle", "Dinucleotide shuffle", "dinuc_shuffle", _C["dinuc_shuffle"]),
    ("gc_match", "GC-matched random", "gc_match", _C["gc_match"]),
    # The nested pair: missense_subset edits a strict SUBSET of synonymous_recode's changed bases,
    # so it perturbs the nucleotides less while damaging the protein the recode kept.
    ("synonymous_recode", "Synonymous recode (protein kept)", "synonymous_recode",
     _C["synonymous_recode"]),
    ("missense_subset", "Missense at recode sites (protein damaged)", "missense_subset",
     _C["missense_subset"]),
]
TAX_COL = "spearman_geodesic_taxonomy"


def preservation_figure(out_dir: Path, family_order: list[str]) -> None:
    """Companion 'control vs natural GEODESIC' figure, styled identically to the Evo2-human and GPN-Star-human panels via the shared plot_utils.control_preservation_figure()."""
    win_p = out_dir / "control_within_scores.csv"
    if not win_p.exists():
        print(f"  [skip] control_preservation: no {win_p.name} (run the control scoring first)")
        return
    win = pd.read_csv(win_p)
    if "rho_geodesic_vs_natural" not in win.columns:
        print("  [skip] control_preservation: no rho_geodesic_vs_natural column")
        return
    within_by = {cond: dict(zip(g["family"], g["rho_geodesic_vs_natural"]))
                 for cond, g in win.groupby("condition")}

    between_by = None
    btw_p = out_dir / "control_between_scores.csv"
    if btw_p.exists():
        btw = pd.read_csv(btw_p)
        if "rho_vs_natural_centroid" in btw.columns:
            between_by = dict(zip(btw["condition"], btw["rho_vs_natural_centroid"]))

    # Display = the named controls (Natural is the self=1.0 reference line drawn by the
    # helper), in the composition-gradient order, reusing the CONDITIONS labels/colours.
    display = [(k, lbl, c) for k, lbl, _, c in CONDITIONS if k != "natural"]
    ok = control_preservation_figure(
        str(out_dir / "control_preservation"), family_order, display,
        within_by, between_by_condition=between_by,
        title_left="Composition controls: is the within-family geometry preserved? "
                   "(Evo2 gene families)",
    )
    print(f"Saved {out_dir}/control_preservation.{{pdf,png}}" if ok
          else "  [skip] control_preservation: no control had within-family data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--natural-run", required=True, help="Natural Evo2 gene-family results dir")
    ap.add_argument("--layer", type=int, default=None,
                    help="Read controls embedded at this block (axisB_control_scores_blocks<N>.csv). "
                         "Omit for the default blocks.24 controls.")
    ap.add_argument("--out", default=None, help="Output basename (default: <natural-run>/controls/control_comparison)")
    args = ap.parse_args()
    nat_dir = Path(args.natural_run)
    suffix = f"_blocks{args.layer}" if args.layer is not None else ""

    # Self-contained controls/ folder in the run dir (mirrors the GPN msa_controls/ layout):
    # per-condition CSVs + a combined summary + the figure.
    out_dir = nat_dir / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    natural = pd.read_csv(nat_dir / "axisB_within_family_correlations.csv").set_index("family")
    family_order = natural.index.tolist()
    natural.to_csv(out_dir / "natural.csv")

    def df_for(control_dir):
        if control_dir is None:
            return natural
        p = CONTROL_ROOT / control_dir / f"axisB_control_scores{suffix}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p).set_index("family")
        df.to_csv(out_dir / f"{control_dir}.csv")  # copy into the controls folder
        return df

    series, summary = [], []
    for key, label, cdir, color in CONDITIONS:
        df = df_for(cdir)
        if df is None:
            print(f"  [skip] {key}: no scores at {suffix or 'blocks.24'}")
            continue
        r = [df.loc[f, TAX_COL] if f in df.index else np.nan for f in family_order]
        series.append((label, r, None, color))
        summary.append((label, np.nanmean(r), np.nanstd(r), color))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 6.5), dpi=200, gridspec_kw={"width_ratios": [3, 1]})
    grouped_rho_bars(
        axL, family_order, series,
        "Within-family geodesic vs taxonomy: natural vs composition controls",
        "Within-family Spearman ρ (geodesic vs host taxonomy)",
        ylim=(-0.3, 1.0), bar_width=0.16, tick_fontsize=8,
    )
    # Right: across-family mean ρ per condition (the summary readout).
    labels = [s[0] for s in summary]
    means = [s[1] for s in summary]
    stds = [s[2] for s in summary]
    colors = [s[3] for s in summary]
    xs = np.arange(len(summary))
    axR.bar(xs, means, yerr=stds, color=colors, edgecolor=acs.apc.white, linewidth=0.5,
            capsize=4, zorder=3)
    for x, m in zip(xs, means):
        axR.text(x, m + 0.02, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axR.axhline(0, color=acs.ZERO_LINE, linewidth=0.8)
    axR.set_ylim(-0.3, 1.0)
    axR.set_ylabel("Mean within-family ρ across families")
    axR.set_title("Mean across families", fontweight="bold")
    axR.spines[["top", "right"]].set_visible(False)
    axR.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    out = Path(args.out) if args.out else out_dir / "control_comparison"
    fig.savefig(f"{out}.pdf", dpi=200)
    fig.savefig(f"{out}.png", dpi=200)
    plt.close(fig)

    # Combined summary CSV (the headline table) alongside the per-condition CSVs.
    nat_mean = summary[0][1] if summary else float("nan")
    summ_df = pd.DataFrame(
        [{"condition": lbl, "mean_rho_taxonomy": m, "std_rho_taxonomy": sd,
          "pct_of_natural": (m / nat_mean if nat_mean else np.nan)} for lbl, m, sd, _ in summary]
    )
    summ_df.to_csv(out_dir / "control_summary.csv", index=False)

    # Companion control-vs-natural geodesic PRESERVATION figure (human-panel style).
    preservation_figure(out_dir, family_order)

    print(f"Saved {out}.{{pdf,png}} and {out_dir}/control_summary.csv")
    print(f"Controls folder: {out_dir}/  ({len(summary)} conditions + per-condition CSVs)")
    print("\nMean within-family geodesic-vs-taxonomy ρ:")
    for label, m, sd, _ in summary:
        print(f"  {label:28} {m:+.3f}  (±{sd:.3f})  {m / nat_mean * 100:5.0f}% of natural")


if __name__ == "__main__":
    main()
