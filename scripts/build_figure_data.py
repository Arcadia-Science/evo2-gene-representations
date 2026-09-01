"""Build `figure_data/` — the tidy tables every publication figure reads.

Each figure generator reads exactly one or two files from `figure_data/`. This script is the only
thing that touches the dated run directories under `results/`, so those stay local build artifacts
and `figure_data/` is the tracked, citable form of the same numbers.

Run it after a full `--run`, before `--figures`:

    uv run python scripts/build_figure_data.py
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_data"
RESULTS = ROOT / "results"

MAMMAL = RESULTS / "2026-07-16_mammalian-orthologs-transcript_cdsmask"
PAIRED = RESULTS / "2026-07-28_evo2-platypus-paired" / "stage2_cds_mean"
STRAT = RESULTS / "2026-08-08_platypus-strat-400"
ARM = "stage4_cds_mean_blocks27"

LAYERS = list(range(32))

# Within-family baseline file -> the reader-facing metric name used on the figures.
WITHIN_BASELINES = {
    "kmer_within_family_correlations_angular.csv": (
        "k-mer composition (CDS, k=6)",
        "spearman_geodesic_kmer",
    ),
    "within_family_patristic_angular.csv": ("patristic tree", "spearman_geodesic_patristic"),
    "within_family_speciestree_angular.csv": (
        "species tree (mammal)",
        "spearman_geodesic_speciestree",
    ),
    "within_family_gc_angular.csv": ("GC content (control)", "spearman_geodesic_gc"),
}


def _blocks(root: Path, rel: str) -> list[tuple[int, Path]]:
    """(layer, path) for every block dir that has `rel`, in layer order."""
    return [(L, root / f"blocks{L}" / rel) for L in LAYERS if (root / f"blocks{L}" / rel).exists()]


# ── Experiment 1


def between_family_by_layer() -> pd.DataFrame:
    """Figure 1: W2 family geometry vs each between-family baseline, per block."""
    rows = []
    for layer, path in _blocks(MAMMAL, "between_family_ot_scores.csv"):
        d = pd.read_csv(path)
        d = d[d["approach"] == "wasserstein"]
        for r in d.itertuples():
            rows.append(
                {
                    "layer": layer,
                    "baseline": r.baseline,
                    "axis": r.axis,
                    "rho": r.spearman_rho,
                    "p_mantel": r.p_mantel,
                }
            )
    return pd.DataFrame(rows)


def within_family_by_layer() -> pd.DataFrame:
    """Figures 2 and 3: per-family mean within-ortholog-group rho against each baseline."""
    rows = []
    for fname, (metric, col) in WITHIN_BASELINES.items():
        for layer, path in _blocks(MAMMAL, fname):
            d = pd.read_csv(path)
            for r in d.itertuples():
                rows.append(
                    {
                        "layer": layer,
                        "metric": metric,
                        "family": r.family,
                        "rho": getattr(r, col),
                    }
                )
    return pd.DataFrame(rows)


# ── Experiment 2


def control_preservation() -> pd.DataFrame:
    """Figures 4 and 12: how well each control reproduces the natural geometry, per block.

    One long table over both axes. `family` is set for within-family rows and empty for
    between-family rows, whose geometry is a single family x family matrix per block.
    """
    rows = []
    for layer, path in _blocks(MAMMAL, "controls/control_between_scores.csv"):
        for r in pd.read_csv(path).itertuples():
            rows.append(
                {
                    "layer": layer,
                    "condition": r.condition,
                    "axis": "between_family",
                    "family": "",
                    "rho": r.rho_vs_natural_between,
                }
            )
    for layer, path in _blocks(MAMMAL, "controls/control_within_scores.csv"):
        for r in pd.read_csv(path).itertuples():
            rows.append(
                {
                    "layer": layer,
                    "condition": r.condition,
                    "axis": "within_family",
                    "family": r.family,
                    "rho": r.rho_angular_vs_natural,
                }
            )
    return pd.DataFrame(rows)


# ── Experiment 3


def direction_layer_stats() -> pd.DataFrame:
    """Figure 6a: per-block leave-one-out and magnitude statistics for the paired panel."""
    d = pd.read_csv(PAIRED / "layer_stats.csv")
    d.insert(0, "panel", "paired103")
    return d


def direction_per_gene_by_layer() -> pd.DataFrame:
    """Figure 6b (paired panel) and Figure 11 (stratified panel): per-gene direction geometry."""
    frames = []
    for panel, path in (
        ("paired103", PAIRED / "per_gene_by_layer.csv"),
        ("strat400", STRAT / "geom_cds_mean" / "per_gene_by_layer.csv"),
    ):
        d = pd.read_csv(path)
        d.insert(0, "panel", panel)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def steering_outcomes() -> tuple[pd.DataFrame, list[str]]:
    """Figures 7 and 8: one row per (gene, condition).

    The saved table has one row per generated sample. Both figures average over samples within a
    gene before plotting anything, so this aggregation is lossless for them. Columns that are
    constant inside every (gene, condition) cell are carried through unchanged; the rest are
    averaged, and the averaged ones are named in the return value so the README can list them.
    """
    d = pd.read_csv(STRAT / ARM / "stage4_scores_nt.csv")
    keys = ["gene", "condition"]
    g = d.groupby(keys, sort=False)
    numeric = [c for c in d.columns if c not in keys and pd.api.types.is_numeric_dtype(d[c])]
    other = [c for c in d.columns if c not in keys and c not in numeric]
    constant = [c for c in numeric if (g[c].nunique(dropna=False) <= 1).all()]
    averaged = sorted(set(numeric) - set(constant) - {"sample"})
    agg = {c: "first" for c in constant + other}
    agg.update({c: "mean" for c in averaged})
    out = g.agg(agg).reset_index()
    out = out.drop(columns=[c for c in ("sample",) if c in out.columns])
    return out[[c for c in d.columns if c in out.columns]], averaged


def _copy_table(src: Path) -> pd.DataFrame:
    return pd.read_csv(src)


TABLES: dict[str, tuple[str, object]] = {
    # name: (one-line description, builder)
    "exp1_between_family_by_layer": (
        "Figure 1 — W2 family geometry vs baselines",
        between_family_by_layer,
    ),
    "exp1_within_family_by_layer": (
        "Figures 2, 3 — within-group rho vs baselines",
        within_family_by_layer,
    ),
    "exp2_control_preservation": (
        "Figures 4, 12 — control vs natural geometry",
        control_preservation,
    ),
    "exp3_direction_layer_stats": (
        "Figure 6a — paired-panel direction statistics",
        direction_layer_stats,
    ),
    "exp3_direction_per_gene_by_layer": (
        "Figures 6b, 11 — per-gene direction geometry",
        direction_per_gene_by_layer,
    ),
    "exp3_panel": (
        "Figure 5 — the 400 human/platypus pairs",
        lambda: _copy_table(STRAT / "stage1" / "pairs.csv"),
    ),
    "exp3_panel_attrition": (
        "Figure 5 — candidates dropped by QC",
        lambda: _copy_table(STRAT / "stage1" / "attrition.csv"),
    ),
    "exp3_panel_coverage": (
        "Figure 5 — aligned CDS coverage per pair",
        lambda: _copy_table(STRAT / "stage2" / "aligned_coverage.csv"),
    ),
    "exp3_steering_outcomes": ("Figures 7, 8 — steering outcomes per gene and condition", None),
    "exp3_site_directionality_summary": (
        "Figures 9a, 9b — leave-human and platypus-choice rates",
        lambda: _copy_table(STRAT / "site_directionality" / "summary.csv"),
    ),
    "exp3_site_directionality_gc_class": (
        "Figures 9a, 9b — the same, split by GC class",
        lambda: _copy_table(STRAT / "site_directionality" / "gc_class.csv"),
    ),
    "exp3_site_directionality_per_gene": (
        "Figures 9a, 9b — per-gene spread",
        lambda: _copy_table(STRAT / "site_directionality" / "per_gene.csv"),
    ),
    "exp3_generation_composition": (
        "Figure 10 — GC of the generations",
        lambda: _copy_table(STRAT / "figures" / "12b_composition_gene_means.csv"),
    ),
    "exp3_generation_reference_windows": (
        "Figure 10 — human and platypus reference GC",
        lambda: _copy_table(STRAT / "figures" / "12c_reference_windows.csv"),
    ),
    "exp3_codon_substitutions": (
        "Figure 10 — GC by codon position",
        lambda: _copy_table(STRAT / "figures" / "13b_codon_substitution_stats.csv"),
    ),
    "exp3_rates_tree_stats": (
        "Figure 11 — fixed-topology branch lengths",
        lambda: _copy_table(STRAT / "stage5" / "tree_stats.csv"),
    ),
    "exp3_rates_dnds": (
        "Figure 11 — dN, dS and omega",
        lambda: _copy_table(STRAT / "stage5" / "dnds.csv"),
    ),
    "exp3_rates_vs_direction": (
        "Figure 11 — rate vs direction geometry",
        lambda: _copy_table(STRAT / "stage5" / "rate_vs_direction.csv"),
    ),
    "exp3_rates_vs_direction_shape": (
        "Figure 11 — the same, shape tests",
        lambda: _copy_table(STRAT / "stage5" / "rate_vs_direction_shape.csv"),
    ),
    "exp3_rates_vs_gain": (
        "Figure 11 — rate vs steering gain",
        lambda: _copy_table(STRAT / "stage5" / "rate_vs_gain.csv"),
    ),
}

COPIES = {"exp3_direction_nulls.npz": PAIRED / "null_distributions.npz"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    averaged: list[str] = []
    rows = []
    for name, (desc, fn) in TABLES.items():
        if name == "exp3_steering_outcomes":
            table, averaged = steering_outcomes()
        else:
            table = fn()
        path = args.out / f"{name}.csv"
        table.to_csv(path, index=False)
        rows.append((name + ".csv", len(table), len(table.columns), path.stat().st_size, desc))
        print(f"  {len(table):>7,} x {len(table.columns):<3} {path.name}")

    for name, src in COPIES.items():
        if not src.exists():
            sys.exit(f"missing {src}")
        shutil.copy2(src, args.out / name)
        rows.append(
            (name, 0, 0, (args.out / name).stat().st_size, "Figures 6a, 6b — permutation nulls")
        )
        print(f"  {'(binary)':>11} {name}")

    total = sum(r[3] for r in rows)
    print(f"\n{len(rows)} files, {total / 1048576:.1f} MiB -> {args.out}")
    if averaged:
        print(f"steering outcomes: averaged over samples for {len(averaged)} columns")
    pd.DataFrame(rows, columns=["file", "rows", "columns", "bytes", "description"]).to_csv(
        args.out / "MANIFEST.csv", index=False
    )


if __name__ == "__main__":
    main()
