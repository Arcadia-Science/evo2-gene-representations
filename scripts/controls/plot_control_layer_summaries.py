"""Per-model control summary figures across all layers (x = layer)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arcadia_style as acs  # noqa: E402

SUMMARY = Path("results/layer_sweep_summaries")
LAYER_RE = re.compile(r"(?:blocks|L)(\d+)\b")

MODELS = [
    {
        "name": "Evo2 cross-species (15 gene families)",
        "slug": "evo2-gene-families-15fam",
        "glob": "results/2026-06-30_evo2-gene-families/blocks*",
        "recov_col": "spearman_geodesic_taxonomy",
        "recov_label": "within recovery vs taxonomy",
        "natural_metric": "taxonomy",
    },
    {
        "name": "Evo2 human (9 paralogs)",
        "slug": "evo2-human-9fam",
        "glob": "results/2026-07-01_evo2-human-panel/blocks*",
        "recov_col": "rho_geodesic_patristic",  # gene-tree recovery (no cross-organism taxonomy)
        "recov_label": "within recovery vs patristic",
        "natural_metric": "patristic tree",
    },
    {
        "name": "Evo2 human CDS (48 families, spliced-CDS input)",
        "slug": "evo2-human-cds-48fam",
        "glob": "results/2026-07-22_evo2-human-panel-cds/blocks*",
        "recov_col": "rho_geodesic_patristic",
        "recov_label": "within recovery vs patristic",
        "natural_metric": "patristic tree",
    },
    {
        "name": "Evo2 human (transcript, CDS-masked pool)",
        "slug": "evo2-human-cdspool-transcript-48fam",
        "glob": "results/2026-07-20_evo2-human-cdspool-transcript/blocks*",
        "recov_col": "rho_geodesic_patristic",
        "recov_label": "within recovery vs patristic",
        "natural_metric": "patristic tree",
    },
    {
        "name": "Evo2 mammalian orthologs (transcript, 24 mammals, 48 families)",
        "slug": "mammalian-orthologs-48fam",
        "glob": "results/2026-07-16_mammalian-orthologs-transcript/blocks*",
        "recov_col": "rho_geodesic_speciestree",  # within-group recovery vs the species tree
        "recov_label": "within recovery vs species tree",
        "natural_metric": "species tree (mammal)",
    },
    {
        "name": "Evo2 mammalian orthologs (CDS, 24 mammals)",
        "slug": "mammalian-orthologs-cds-48fam",
        "glob": "results/2026-07-16_mammalian-orthologs-cds/blocks*",
        "recov_col": "rho_geodesic_speciestree",  # within-group recovery vs the species tree
        "recov_label": "within recovery vs species tree",
        "natural_metric": "species tree (mammal)",
    },
    {
        # CDS masks permit protein-preserving recoding of the transcript input.
        "name": "Evo2 mammalian orthologs (transcript, CDS-masked, 24 mammals, 48 families)",
        "slug": "mammalian-orthologs-cdsmask-48fam",
        "glob": "results/2026-07-16_mammalian-orthologs-transcript_cdsmask/blocks*",
        "recov_col": "rho_geodesic_speciestree",  # within-group recovery vs the species tree
        "recov_label": "within recovery vs species tree",
        "natural_metric": "species tree (mammal)",
    },
]


def _layer(p: Path) -> int:
    m = LAYER_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _collect(glob: str):
    """Return (between_df, within_df) concatenated across layers with a `layer` column."""
    b_rows, w_rows = [], []
    for d in sorted((p for p in Path().glob(glob) if p.is_dir()), key=_layer):
        cdir = d / "controls"
        L = _layer(d)
        bp, wp = cdir / "control_between_scores.csv", cdir / "control_within_scores.csv"
        if bp.exists():
            b = pd.read_csv(bp)
            b["layer"] = L
            b_rows.append(b)
        if wp.exists():
            w = pd.read_csv(wp)
            w["layer"] = L
            w_rows.append(w)
    return (
        pd.concat(b_rows, ignore_index=True) if b_rows else pd.DataFrame(),
        pd.concat(w_rows, ignore_index=True) if w_rows else pd.DataFrame(),
    )


def _natural_recovery(slug: str, metric: str) -> pd.Series | None:
    """Mean-over-families natural recovery ρ per layer, from the layer-sweep within CSV."""
    f = SUMMARY / slug / "within_family_vs_layer.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    return sub.groupby("layer")["rho"].mean()


# Fixed color per condition so a colour means the same thing in every subplot / figure.
COND_COLOR = {
    "natural": acs.CONTROL_COLORS["natural"],
    "synonymous_recode": acs.CONTROL_COLORS["synonymous_recode"],
    "kmer6_shuffle": acs.CONTROL_COLORS["kmer6_shuffle"],
    "kmer4_shuffle": acs.CONTROL_COLORS["kmer4_shuffle"],
    "codon_shuffle": acs.CONTROL_COLORS["codon_shuffle"],
    "dinuc_shuffle": acs.CONTROL_COLORS["dinuc_shuffle"],
    "gc_match": acs.CONTROL_COLORS["gc_match"],
}


def _line_by_condition(ax, df, value_col, title, ylabel):
    for c in sorted(df["condition"].unique()):
        s = df[df["condition"] == c].groupby("layer")[value_col].mean().sort_index()
        lw = 2.2 if c == "natural" else 1.3
        ax.plot(s.index, s.values, "-o", ms=3, lw=lw, color=COND_COLOR.get(c, acs.CONTROL_FALLBACK), label=c)
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.6)
    ax.axhline(1, color=acs.GRID, lw=0.4, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")


def main(with_recovery: bool = False) -> None:
    # These are 2–3-panel figures, so they carry a slightly larger type scale than the
    # dense per-family grids the default in arcadia_style is tuned for.
    acs.setup(font_size=9, title_size=10, tick_size=8, legend_size=8)
    for m in MODELS:
        between, within = _collect(m["glob"])
        if between.empty and within.empty:
            print(f"[skip] no controls for {m['slug']}")
            continue
        has_recov = with_recovery and m.get("recov_col") and m["recov_col"] in within.columns
        ncols = 2 + (1 if has_recov else 0)
        fig, axs = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.0), squeeze=False)
        axs = axs[0]

        _line_by_condition(
            axs[0], between, "rho_vs_natural_centroid",
            "Between-family preservation", "ρ (control vs natural centroid geo)",
        )
        _line_by_condition(
            axs[1], within, "rho_geodesic_vs_natural",
            "Within-family preservation (mean over families)", "ρ (control vs natural geo)",
        )
        if has_recov:
            _line_by_condition(
                axs[2], within, m["recov_col"],
                m["recov_label"] + " (mean over families)", "ρ (control geo vs ground truth)",
            )
            nat = _natural_recovery(m["slug"], m["natural_metric"])
            if nat is not None:
                axs[2].plot(nat.index, nat.values, "-", color=acs.CONTROL_COLORS["natural"], lw=2.0,
                            label="natural (uncontrolled)")

        axs[0].legend(fontsize=7, loc="best")
        if has_recov:
            axs[2].legend(fontsize=7, loc="best")
        fig.suptitle(f"Composition controls across layers — {m['name']}", fontsize=11)
        fig.tight_layout()
        out = SUMMARY / f"controls_layer_summary_{m['slug']}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-recovery", action="store_true",
                    help="add the third panel (within-family recovery vs biological ground truth); "
                         "off by default since 2026-08-14")
    main(with_recovery=ap.parse_args().with_recovery)
