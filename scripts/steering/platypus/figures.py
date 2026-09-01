"""Stage 2 figures — eight statistic panels plus a structure supplement.

uv run python scripts/steering/platypus/figures.py
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

import figure_data  # noqa: E402

REAL, SF, MM = acs.apc.dusk, acs.apc.amber, acs.apc.aster

# Publication mode renders only the two page-sized panels; other outputs remain diagnostics.
PUB_STEMS = {"1_loo_median_by_layer", "8b_magnitude_spread"}

# 6a and 6b are half panels so they sit side by side on the page.
PUB_PANEL_H = 360.0


def save(fig, out: Path, stem: str) -> None:
    if pub.is_on():
        if stem in PUB_STEMS:
            pub.drop_titles(
                fig, keep_axes=False
            )  # 6a/6b are single charts: the caption titles them
            pub.tight(fig)
            pub.finish(fig, stem, directory=out)
        plt.close(fig)
        return
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}")


def _floor(ax, y, label):
    ax.axhline(y, color=acs.GRID, ls=":", lw=1.0, zorder=0)
    ax.text(
        0.99,
        y,
        label,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6,
        color=acs.ANNOTATION,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "results" / "2026-07-28_evo2-platypus-paired"
    ap.add_argument("--stage2-dir", type=Path, default=base / "stage2")
    ap.add_argument(
        "--from-figure-data",
        action="store_true",
        help="read the tracked tables in figure_data/ instead of a stage-2 run directory",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="default: <stage2-dir>/figures")
    ap.add_argument(
        "--structure-suffix",
        default="",
        help="suffix on delta_spectrum/delta_by_group/family_direction_agreement, e.g. "
        "_cds_mean when delta_structure.py was run with --pooled",
    )
    ap.add_argument(
        "--label", default="last prompt position", help="representation name for the figure titles"
    )
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at PUBLICATION geometry: exact 500 pt half panels (draft figures "
        "6a and 6b sit side by side), the style guide's 15 pt type with "
        f"monospaced numerals, and no in-artwork title. Only {sorted(PUB_STEMS)} "
        "are written, into <stage2-dir>/figures/pub/.",
    )
    args = ap.parse_args()

    if args.pub:
        pub.enable()
    d2 = args.stage2_dir
    out = args.out_dir or (d2 / "figures")
    out.mkdir(parents=True, exist_ok=True)
    set_pub_style(title_size=9, tick_size=7)

    if args.from_figure_data:
        st = figure_data.table("exp3_direction_layer_stats").query("panel == 'paired103'")
        pg = figure_data.table("exp3_direction_per_gene_by_layer").query("panel == 'paired103'")
        nd = np.load(figure_data.path("exp3_direction_nulls.npz"))
    else:
        st = pd.read_csv(d2 / "layer_stats.csv")
        pg = pd.read_csv(d2 / "per_gene_by_layer.csv")
        nd = np.load(d2 / "null_distributions.npz")
    x = st.layer.values
    fl = st.coherence_isotropic_floor.iloc[0]

    def band(ax, key, color, label):
        """Null distribution as a 2.5-97.5 band plus median line."""
        lo, mid, hi = [], [], []
        for li in x:
            v = nd[f"{key}_{li}"]
            lo.append(np.nanpercentile(v, 2.5))
            mid.append(np.nanmedian(v))
            hi.append(np.nanpercentile(v, 97.5))
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(x, mid, color=color, lw=1.2, ls="--", label=label)

    # Each reference line marks the statistic's null value.
    specs = [
        ("loo_median", "Leave-one-out cosine (median)", "1_loo_median_by_layer", 0.0),
        ("loo_frac_pos", "Fraction of genes with LOO cosine > 0", "2_loo_frac_positive", 0.5),
        ("coherence", "Directional coherence  C = ||mean u_i||", "3_coherence_by_layer", fl),
        ("split_median", "Split-half stability  cos(v_A, v_B)", "4_splithalf_stability", 0.0),
    ]
    for col, ylab, stem, ref in specs:
        fig, ax = plt.subplots(
            figsize=(pub.size(pub.HALF, PUB_PANEL_H) if pub.is_on() else (6.4, 3.0))
        )
        if col == "split_median":
            ax.fill_between(x, st["split_p2.5"], st["split_p97.5"], color=REAL, alpha=0.18, lw=0)
        ax.plot(x, st[col], color=REAL, lw=1.8, marker="o", ms=2.6)
        if ref is not None:
            _floor(ax, ref, "isotropic floor 1/sqrt(N)" if col == "coherence" else "")
        ax.axvspan(27.5, 31.5, color=acs.apc.gray, alpha=0.5, lw=0, zorder=0)
        if pub.is_on():
            # At 15 pt the label is wider than the four blocks it marks, so it cannot sit inside
            # the shaded band. Above the axes, right-aligned to the same edge the band ends on,
            # it reads as a caption for the shading without landing on the curve.
            ax.text(
                1.0,
                1.01,
                "blocks 28–31 saturated",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=acs.ANNOTATION,
            )
        else:
            ax.text(
                29.5,
                ax.get_ylim()[1],
                "saturated\n(28-31)",
                ha="center",
                va="top",
                fontsize=6,
                color=acs.ANNOTATION,
            )
        ax.set_xlabel("Evo2 block")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} — {len(pg.gene.unique())} human/platypus pairs, {args.label}")
        save(fig, out, stem)

    # 5-6 : real vs nulls ------------------------------------------------------
    for nullkey, nullname, stem in (
        ("sf", "sign-flip null", "5_real_vs_signflip_null"),
        ("mm", "mismatched-pair null (within family)", "6_real_vs_mismatch_null"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1))
        for ax, (col, key, ylab) in zip(
            axes,
            [
                ("loo_mean", f"{nullkey}_loo", "mean leave-one-out cosine"),
                ("coherence", f"{nullkey}_coh", "directional coherence C"),
            ],
            strict=False,
        ):
            band(ax, key, SF if nullkey == "sf" else MM, nullname)
            ax.plot(x, st[col], color=REAL, lw=1.8, marker="o", ms=2.6, label="observed")
            if col == "coherence":
                _floor(ax, fl, "1/sqrt(N)")
            ax.set_xlabel("Evo2 block")
            ax.set_ylabel(ylab)
            ax.legend(fontsize=6, loc="upper left")
        fig.suptitle(f"Observed statistics vs the {nullname}", y=1.02)
        save(fig, out, stem)

    # 7 : per-gene LOO heatmap -------------------------------------------------
    piv = pg.pivot_table(index="gene", columns="layer", values="loo_cos")
    fam = pg.groupby("gene")["family"].first()
    order = fam.sort_values().index
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(8.2, 11.0))
    m = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, aspect="auto", cmap=acs.DIVERGING, vmin=-m, vmax=m)
    ax.set_xticks(range(0, 32, 2))
    ax.set_xticklabels(range(0, 32, 2), fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{g}  [{fam[g][:12]}]" for g in order], fontsize=4.2)
    ax.set_xlabel("Evo2 block")
    ax.set_title("Per-gene leave-one-out cosine a_i by layer (genes grouped by family)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="a_i")
    save(fig, out, "7_per_gene_loo_heatmap")

    # 8 : delta norm by gene and layer ----------------------------------------
    pivn = pg.pivot_table(index="gene", columns="layer", values="delta_norm").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for g in pivn.index:
        axes[0].plot(x, pivn.loc[g], color=acs.GRID, lw=0.4, alpha=0.5)
    axes[0].plot(x, pivn.median(), color=REAL, lw=2.0, label="median")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Evo2 block")
    axes[0].set_ylabel("||delta_i||  (log)")
    axes[0].set_title("Paired delta norm, per gene")
    axes[0].legend(fontsize=6)
    cv = st.delta_norm_cv.values
    axes[1].plot(x, cv, color=REAL, lw=1.8, marker="o", ms=2.6)
    axes[1].set_xlabel("Evo2 block")
    axes[1].set_ylabel("CV of ||delta_i|| across genes")
    axes[1].set_title("Magnitude spread across genes")
    save(fig, out, "8_delta_norm_by_gene_layer")

    # 8b : the magnitude-spread panel on its own ------------------------------
    # Same data as the right panel above. The two-panel version stays the diagnostic view;
    # this one is what the publication uses, where the per-gene trajectories on the left are
    # detail the figure is not making a claim about.
    fig, ax = plt.subplots(figsize=(pub.size(pub.HALF, PUB_PANEL_H) if pub.is_on() else (4.7, 3.2)))
    ax.plot(x, cv, color=REAL, lw=1.8, marker="o", ms=2.6)
    ax.set_xlabel("Evo2 block")
    # Atkinson has Δ (checked against the font's cmap), so the publication label can use the
    # symbol the text uses instead of spelling the variable out.
    ax.set_ylabel("CV of ||Δ|| across genes" if pub.is_on() else "CV of ||delta_i|| across genes")
    ax.set_title("Magnitude spread across genes")
    acs.style_axes(ax, monospaced_axes="both")
    save(fig, out, "8b_magnitude_spread")

    # 9 : structure supplement ------------------------------------------------
    sfx = args.structure_suffix
    if not args.from_figure_data and (d2 / f"delta_spectrum{sfx}.csv").exists():
        sp = pd.read_csv(d2 / f"delta_spectrum{sfx}.csv")
        gp = pd.read_csv(d2 / f"delta_by_group{sfx}.csv")
        cr = pd.read_csv(d2 / f"family_direction_agreement{sfx}.csv")
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2))
        axes[0].plot(sp.layer, sp.participation_ratio, color=REAL, lw=1.8, marker="o", ms=2.6)
        axes[0].axhline(1, color=acs.GRID, ls=":", lw=1.0)
        axes[0].set_ylabel("participation ratio of the delta spectrum")
        axes[0].set_title("Effective dimensionality of the deltas\n(1 = a single shared direction)")
        g = gp[gp.grouping == "family"]
        for name, sub in g.groupby("group"):
            axes[1].plot(
                sub.layer,
                sub.coherence - sub.isotropic_floor,
                lw=1.3,
                label=f"{name} (n={int(sub.n.iloc[0])})",
            )
        axes[1].axhline(0, color=acs.ZERO_LINE, ls=":", lw=1.0)
        axes[1].set_ylabel("within-family C  -  its isotropic floor")
        axes[1].set_title("Per-family coherence, floor-corrected")
        axes[1].legend(fontsize=5.2)
        agg = cr.groupby("layer")["cos"].agg(["median", "min", "max"])
        axes[2].fill_between(agg.index, agg["min"], agg["max"], color=MM, alpha=0.18, lw=0)
        axes[2].plot(agg.index, agg["median"], color=MM, lw=1.8, marker="o", ms=2.6)
        axes[2].axhline(0, color=acs.ZERO_LINE, ls=":", lw=1.0)
        axes[2].set_ylabel("cos between family mean directions")
        axes[2].set_title("Do families agree on a direction?")
        for ax in axes:
            ax.set_xlabel("Evo2 block")
        save(fig, out, "9_structure_supplement")

    print(f"-> {out}")


if __name__ == "__main__":
    main()
