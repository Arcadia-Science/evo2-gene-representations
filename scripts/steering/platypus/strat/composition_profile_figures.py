"""Figures for the composition profile. Reads `composition_profile/`, no recompute."""
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

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load("composition_stats")
_gcf = _load("gc_codon_figures")
figures_dir = _gcf.figures_dir

HUMAN_REF, PLAT_REF = "__human_reference__", "__platypus_reference__"
H_COL, P_COL, GEN_COL, STEER_COL = (acs.STEER_COLORS["source"], acs.STEER_COLORS["target"],
                                    acs.STEER_COLORS["unsteered"], acs.STEER_COLORS["steered"])
EXT = ("png", "pdf")

# The ladder the brief asks for: human reference -> unsteered -> steered -> platypus reference.
LADDER = [HUMAN_REF, "unsteered", "add_a1.0_L24", "add_a0.5", "add_a1.0", "add_a2.0", "add_a3.0",
          "add_a4.0", PLAT_REF]
LADDER_LABELS = {HUMAN_REF: "human\nref", PLAT_REF: "platypus\nref", "unsteered": "unsteered",
                 "add_a1.0_L24": "L24\nα=1", "add_a0.5": "L27\nα=0.5", "add_a1.0": "L27\nα=1",
                 "add_a2.0": "L27\nα=2", "add_a3.0": "L27\nα=3", "add_a4.0": "L27\nα=4"}
# The compact set used where nine columns will not fit.
FOUR = [HUMAN_REF, "unsteered", "add_a1.0", "add_a4.0", PLAT_REF]

SCALAR_PANELS = [
    ("gc", "overall GC", 100),
    ("gc1", "GC at codon position 1", 100),
    ("gc2", "GC at codon position 2", 100),
    ("gc3", "GC3 (wobble position)", 100),
    ("cpg_oe", "CpG observed / expected", 1),
    ("ts_tv", "transition / transversion\n(vs the human window)", 1),
    ("f_A", "A frequency", 100),
    ("f_C", "C frequency", 100),
    ("f_G", "G frequency", 100),
    ("f_T", "T frequency", 100),
    ("subst_rate", "substitution rate vs human", 100),
    # TA, not CG: rho_CG is cpg_oe by definition (same quantity, panel 5), whereas TA is the other
    # classically depleted dinucleotide and carries information the GC panels do not.
    ("rho_TA", "TA relative abundance ρ(TA)", 1),
]


# Distinct shades along the dose ladder: on a grouped bar chart two rungs sharing one red are
# indistinguishable in the legend, which is worse than no legend.
# The α ladder is an ordered series, so it walks blue_shades light → dark (the steered
# hue getting more saturated as the dose rises). The L24 rung is a different LAYER, not a
# further dose, so it takes a separate hue instead of a slot on the ramp.
DOSE_SHADE = {"add_a0.5": acs.apc.sky, "add_a1.0": acs.apc.vital, "add_a2.0": acs.apc.aegean,
              "add_a3.0": acs.apc.lapis, "add_a4.0": acs.apc.dusk,
              "add_a1.0_L24": acs.SERIES_THIRD}


def colour_for(cond: str) -> str:
    if cond == HUMAN_REF:
        return H_COL
    if cond == PLAT_REF:
        return P_COL
    if cond == "unsteered":
        return GEN_COL
    return DOSE_SHADE.get(cond, STEER_COL)


def present(per: pd.DataFrame, wanted: list[str], min_genes: int = 350) -> list[str]:
    """Levels with near-full gene coverage, in the given order."""
    n = per.groupby("condition").gene.nunique()
    return [c for c in wanted if int(n.get(c, 0)) >= min_genes]


def fig21(per: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Each scalar's per-gene distribution at every level, with both species' means as rules."""
    conds = present(per, LADDER)
    set_pub_style(title_size=8, tick_size=6)
    ncol = 4
    nrow = int(np.ceil(len(SCALAR_PANELS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, title, scale) in zip(axes, SCALAR_PANELS):
        data, keep = [], []
        for c in conds:
            v = per.loc[per.condition == c, col].dropna() * scale
            if len(v) > 10:
                data.append(v.to_numpy())
                keep.append(c)
        if not data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(data, showextrema=False, widths=0.86)
        for b, c in zip(parts["bodies"], keep):
            b.set_facecolor(colour_for(c))
            b.set_alpha(0.45)
            b.set_edgecolor(colour_for(c))
            b.set_linewidth(0.6)
        ax.scatter(range(1, len(data) + 1), [np.mean(v) for v in data], s=9, color="black",
                   zorder=5)
        for ref, col_ref, lab in ((HUMAN_REF, H_COL, "human"), (PLAT_REF, P_COL, "platypus")):
            v = per.loc[per.condition == ref, col].dropna() * scale
            if len(v):
                ax.axhline(float(v.mean()), color=col_ref, lw=0.8, ls="--", alpha=0.85,
                           zorder=1, label=f"{lab} mean")
        ax.set_xticks(range(1, len(keep) + 1),
                      [LADDER_LABELS.get(c, c).replace("\n", " ") for c in keep],
                      fontsize=5.2, rotation=38, ha="right")
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel("%" if scale == 100 else "ratio", fontsize=6.5)
        if col == "gc":
            ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    for ax in axes[len(SCALAR_PANELS):]:
        ax.set_visible(False)
    fig.suptitle("Composition of the generations against the real human and platypus CDS windows\n"
                 "per-gene distributions, 398 genes; dashed rules = the two species' means",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for e in EXT:
        fig.savefig(out / f"21_composition_vs_reference_distributions.{e}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    tab = per[per.condition.isin(conds)].groupby("condition")[
        [c for c, _t, _s in SCALAR_PANELS]].mean().reindex(conds)
    tab.to_csv(out / "21_composition_vs_reference_distributions.csv")
    return tab


def fig22(dn: pd.DataFrame, out: Path) -> pd.DataFrame:
    """All 16 dinucleotide relative abundances, human / unsteered / steered / platypus."""
    conds = [c for c in FOUR if c in dn.columns]
    set_pub_style(title_size=9, tick_size=7)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(dn.index))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, dn[c], width=w, label=LADDER_LABELS.get(c, c),
               color=colour_for(c), alpha=0.9 if c in (HUMAN_REF, PLAT_REF) else 0.75,
               edgecolor="white", lw=0.3)
    ax.axhline(1.0, color=acs.ANNOTATION, lw=0.8, ls=":")
    ax.annotate("ρ = 1: exactly as often as base composition predicts", (0.005, 1.0),
                xycoords=("axes fraction", "data"), ha="left", va="bottom", fontsize=6,
                color=acs.ANNOTATION)
    cg = list(dn.index).index("CG") if "CG" in list(dn.index) else None
    if cg is not None:
        ax.axvspan(cg - 0.5, cg + 0.5, color=acs.apc.oat, alpha=0.45, zorder=0)
    ax.set_xticks(x, dn.index, fontsize=7)
    ax.set_xlabel("dinucleotide")
    ax.set_ylabel("relative abundance  ρ = f(XY) / f(X)f(Y)")
    ax.set_title("Dinucleotide relative abundance — GC-shift-corrected dinucleotide structure\n"
                 "CpG shaded; ρ divides out the base composition the steering moves hardest")
    ax.legend(fontsize=6.5, frameon=False, ncol=len(conds))
    fig.tight_layout()
    for e in EXT:
        fig.savefig(out / f"22_dinucleotide_relative_abundance.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    dn[conds].to_csv(out / "22_dinucleotide_relative_abundance.csv")
    return dn[conds]


def fig23(js: pd.DataFrame, per: pd.DataFrame, out: Path) -> pd.DataFrame:
    """d_H against d_P per vector family: did it move toward platypus, or only away from human?"""
    conds = present(per, LADDER)
    fams = [f for f in cs.VECTOR_FAMILIES]
    names = {"dinuc": "dinucleotides (16)", "kmer4": "4-mers (256)", "codon": "codon usage (61)",
             "syn_usage": "synonymous codon usage"}
    set_pub_style(title_size=8.5, tick_size=6.5)
    fig, axes = plt.subplots(1, len(fams), figsize=(3.05 * len(fams), 3.35))
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams):
        s = js[(js.family == fam)].set_index("condition")
        keep = [c for c in conds if c in s.index and c not in (HUMAN_REF, PLAT_REF)]
        lim = float(max(s.loc[keep, "dH_mean"].max(), s.loc[keep, "dP_mean"].max())) * 1.18
        ax.plot([0, lim], [0, lim], color=acs.GRID, lw=0.8, ls="--", zorder=1)
        ax.annotate("equidistant", (lim * 0.97, lim * 0.97), fontsize=5.5, color=acs.GRID,
                    ha="right", va="bottom", rotation=45)
        hp = float(s["HP_mean"].dropna().mean())
        ax.axvline(hp, color=acs.SERIES_MUTED, lw=0.8, ls=":", zorder=1)
        ax.annotate(f"JSD(H,P) = {hp:.3f}", (hp, lim * 0.02), fontsize=5.5, color=acs.ANNOTATION,
                    rotation=90, va="bottom", ha="right")
        xs, ys = s.loc[keep, "dH_mean"].to_numpy(), s.loc[keep, "dP_mean"].to_numpy()
        ladder = [c for c in keep if c.startswith("add_a") and not c.endswith("_L24")]
        if len(ladder) > 1:
            lx = s.loc[ladder, "dH_mean"].to_numpy()
            ly = s.loc[ladder, "dP_mean"].to_numpy()
            ax.plot(lx, ly, "-", color=STEER_COL, lw=0.9, alpha=0.7, zorder=2)
        # The low-dose rungs sit almost on top of each other, so labels are staggered around the
        # marker instead of all going to its right, where they overprint into illegibility.
        offs = [(6, -9), (-6, -10), (6, 4), (-6, 5), (7, -2), (-7, -2), (6, 9)]
        for i, (c, xx, yy) in enumerate(zip(keep, xs, ys)):
            ax.scatter(xx, yy, s=26, color=colour_for(c), edgecolor="white", lw=0.5, zorder=4)
            dx, dy = offs[i % len(offs)]
            ax.annotate(LADDER_LABELS.get(c, c).replace("\n", " "), (xx, yy),
                        textcoords="offset points", xytext=(dx, dy), fontsize=5.3,
                        ha="left" if dx > 0 else "right", color=colour_for(c), zorder=6)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("d$_H$ = JSD to human (bits)")
        ax.set_ylabel("d$_P$ = JSD to platypus (bits)")
        ax.set_title(names.get(fam, fam))
        # The point the panel is really making: every generation sits far outside the separation
        # between the two species, so "which species does it resemble" is barely a question yet.
        far = float(s.loc[keep, "dH_mean"].min() / hp) if hp > 0 else np.nan
        ax.annotate(f"nearest generation is {far:.1f}× JSD(H,P) from human",
                    (0.03, 0.97), xycoords="axes fraction", fontsize=5.4, color=acs.ANNOTATION,
                    va="top")
    fig.suptitle("Distance to each species, per gene's own windows — below the diagonal = closer to "
                 "platypus", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for e in EXT:
        fig.savefig(out / f"23_jsd_to_each_species.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    t = js[js.condition.isin(conds)][["family", "condition", "dH_mean", "dP_mean",
                                      "dP_minus_dH", "HP_mean", "n_genes"]]
    t.to_csv(out / "23_jsd_to_each_species.csv", index=False)
    return t


def fig24(sub: pd.DataFrame, per: pd.DataFrame, out: Path) -> pd.DataFrame:
    """The 12 substitution classes against human, with the real platypus-vs-human spectrum beside."""
    conds = [c for c in FOUR if c in sub.columns and c != HUMAN_REF]
    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    ax = axes[0]
    x = np.arange(len(sub.index))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        ax.bar(x + (i - (len(conds) - 1) / 2) * w, sub[c], width=w,
               label=LADDER_LABELS.get(c, c), color=colour_for(c),
               alpha=0.9 if c == PLAT_REF else 0.75, edgecolor="white", lw=0.3)
    for i, k in enumerate(sub.index):
        if k in cs.TRANSITIONS:
            ax.axvspan(i - 0.5, i + 0.5, color=acs.apc.zephyr, zorder=0)
    ax.set_xticks(x, sub.index, fontsize=7, rotation=45)
    ax.set_xlabel("substitution class (human base → observed base)")
    ax.set_ylabel("share of substitutions")
    ax.set_title("Substitution spectrum against the human window\n"
                 "shaded = transitions; the platypus reference IS the real 166-My spectrum")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)

    ax = axes[1]
    lad = present(per, LADDER)
    data, keep = [], []
    for c in lad:
        v = per.loc[per.condition == c, "ts_tv"].dropna()
        if len(v) > 10:
            data.append(v.to_numpy())
            keep.append(c)
    parts = ax.violinplot(data, showextrema=False, widths=0.86)
    for b, c in zip(parts["bodies"], keep):
        b.set_facecolor(colour_for(c))
        b.set_alpha(0.45)
        b.set_edgecolor(colour_for(c))
    ax.scatter(range(1, len(data) + 1), [np.mean(v) for v in data], s=10, color="black", zorder=5)
    pv = per.loc[per.condition == PLAT_REF, "ts_tv"].dropna()
    if len(pv):
        ax.axhline(float(pv.mean()), color=P_COL, lw=0.9, ls="--",
                   label=f"real platypus vs human = {float(pv.mean()):.2f}")
        ax.legend(fontsize=6, frameon=False)
    ax.set_xticks(range(1, len(keep) + 1), [LADDER_LABELS.get(c, c) for c in keep], fontsize=5.5)
    ax.set_ylabel("transitions / transversions")
    ax.set_title("Ts/Tv against the human window")
    fig.tight_layout()
    for e in EXT:
        fig.savefig(out / f"24_substitution_spectrum.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    t = sub[[c for c in FOUR if c in sub.columns]].copy()
    t.to_csv(out / "24_substitution_spectrum.csv")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", default="composition_profile")
    ap.add_argument("--layer", type=int, default=27)
    args = ap.parse_args()
    d = args.run / args.data
    per = pd.read_csv(d / "features_per_gene.csv.gz")
    js = pd.read_csv(d / "jsd_by_condition.csv")
    dn = pd.read_csv(d / "dinucleotide_relative_abundance.csv", index_col=0)
    sub = pd.read_csv(d / "substitution_classes.csv", index_col=0)
    out = figures_dir(args.run, args.layer)
    out.mkdir(parents=True, exist_ok=True)

    print("building 21_composition_vs_reference_distributions ...")
    t21 = fig21(per, out)
    print(t21.to_string(float_format=lambda z: f"{z:.4f}"))
    print("\nbuilding 22_dinucleotide_relative_abundance ...")
    fig22(dn, out)
    print("\nbuilding 23_jsd_to_each_species ...")
    t23 = fig23(js, per, out)
    print(t23.pivot(index="condition", columns="family", values=["dH_mean", "dP_mean"])
          .to_string(float_format=lambda z: f"{z:.4f}"))
    print("\nbuilding 24_substitution_spectrum ...")
    t24 = fig24(sub, per, out)
    print(t24.to_string(float_format=lambda z: f"{z:.4f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
