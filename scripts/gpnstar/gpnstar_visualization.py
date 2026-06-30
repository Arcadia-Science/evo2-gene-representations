"""
Visualizations for the GPN-Star gene-family geodesic analysis.

Figure sets, selectable via --figures (default: all available):

  between            : family-level (F×F) geodesic centroid / Pfam JSD / CDS seq-identity /
                       k-mer divergence heatmaps + a between-family ρ bar chart (centroid
                       geodesic vs each baseline, bootstrap CIs).
                       -> between_comparison.{pdf,png}

  within-heatmaps    : rows of per-family within-family heatmaps (geodesic, k-mer
                       divergence, Ensembl Compara), no bar charts. -> within_heatmaps.{pdf,png}

  within-correlations: per-family within-family Spearman ρ bars (geodesic vs k-mer
                       divergence & Ensembl Compara). -> within_correlations.{pdf,png}

  (Seq identity is still loaded and its per-family ρ printed; it is just not plotted —
  flip its WITHIN_BASELINES "plot" flag to re-enable.)

Baselines whose CSV is absent are skipped (a sparse paralog matrix still renders, with
NaN pairs greyed out / dropped from the ρ).

Usage:
    uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir RESULTS_DIR
    uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir RESULTS_DIR --figures between
    uv run python scripts/gpnstar/gpnstar_visualization.py \
        --run-dir RESULTS_DIR --figures within-heatmaps within-correlations
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gene_families import family_colors  # noqa: E402

FAMILY_COLORS = family_colors("human")
from plot_utils import (  # noqa: E402
    WITHIN_PALETTE,
    between_rho_bars,
    draw_family_heatmap,
    grouped_rho_bars,
    make_cmap_with_nan,
    per_family_rho,
    pstar,
    set_pub_style,
    within_csv_series,
)

# ════════════════════════════════════════════════════════════════════════════════
# Within-family figures: heatmaps (per family) and the per-family Spearman ρ bars
# ════════════════════════════════════════════════════════════════════════════════

# Within-family baselines the geodesic is compared against, in display order. Each is a
# gene×gene matrix; `to_distance` maps the stored values to a distance in [0, 1]
# (NaN preserved). All are reindexed to the geodesic's gene order before use. `plot`
# baselines are still loaded and their per-family ρ printed, but kept out of the
# within figures (seq identity is computed but not plotted for now).
WITHIN_BASELINES = [
    {
        "csv": "sequence_identity_genes.csv",
        "to_distance": lambda m: np.where(np.isfinite(m), 1.0 - m, np.nan),
        "cmap": "Greens",
        "row_label": "CDS seq identity\n(1 − identity)",
        "cbar_label": "1 − seq identity",
        "color": "#2D6A4F",
        "short": "vs seq identity",
        "nan_legend": False,
        "plot": False,
    },
    {
        "csv": "kmer_distance_genes.csv",
        "to_distance": lambda m: m,
        "cmap": "Purples",
        "row_label": "k-mer divergence\n(1 − cosine)",
        "cbar_label": "k-mer distance",
        "color": "#6A4C93",
        "short": "vs k-mer composition",
        "nan_legend": False,
        "plot": True,
    },
    {
        "csv": "ensembl_paralog_identity_genes.csv",
        "to_distance": lambda m: np.where(np.isfinite(m), 1.0 - m / 100.0, np.nan),
        "cmap": "Blues",
        "row_label": "Ensembl Compara\n(1 − prot. ID)",
        "cbar_label": "1 − prot. identity",
        "color": "#2C6E8A",
        "short": "vs Ensembl Compara",
        "nan_legend": True,
        "plot": False,  # supplementary: human-only, no Evo2 counterpart (still in heatmaps/logs)
    },
]


def load_within_baseline(run_dir: Path, cfg: dict, gene_names: list) -> np.ndarray | None:
    """Load a within-family baseline as a gene×gene distance matrix, gene-order aligned."""
    p = run_dir / cfg["csv"]
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0).reindex(index=gene_names, columns=gene_names)
    dist = cfg["to_distance"](df.values)
    np.fill_diagonal(dist, 0.0)
    return dist


def fam_sub(mat, fam, families_arr, gene_names):
    """Return (indices, gene names, sub-matrix) for one family's genes."""
    idx = np.where(families_arr == fam)[0]
    genes = [gene_names[i] for i in idx]
    return idx, genes, mat[np.ix_(idx, idx)]


def shared_vmax(per_fam, family_order, pct=95):
    """Robust shared colour-scale max across all families' off-diagonal values."""
    vals = []
    for fam in family_order:
        _, genes, sub = per_fam[fam]
        if len(genes) >= 2:
            t = sub[np.triu_indices(len(genes), k=1)]
            vals.extend(t[np.isfinite(t)])
    return float(np.percentile(vals, pct)) if vals else 1.0


def draw_within_hm(ax, sub, genes, cmap, vmax, title, color_strip=None, fontsize=4.5):
    """Draw one NaN-aware per-family heatmap with gene tick labels."""
    norm = Normalize(vmin=0, vmax=vmax)
    masked = np.ma.array(sub, mask=~np.isfinite(sub))
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(genes)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(genes, fontsize=fontsize)
    ax.tick_params(length=1.5, pad=0.5)
    ax.set_title(title, fontsize=6.5, pad=5)
    if color_strip is not None:
        strip = ax.inset_axes([0, 1.04, 1, 0.07], transform=ax.transAxes)
        strip.set_xlim(0, 1)
        strip.set_ylim(0, 1)
        strip.axis("off")
        strip.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color_strip, linewidth=0))


def make_within_heatmaps(run_dir: Path) -> None:
    """One figure: rows = geodesic + each within-family baseline, columns = families.

    Per-family gene×gene heatmaps with gene tick labels; no bar charts — the per-family
    ρ summary lives in the within_correlations figure.
    """
    set_pub_style()
    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    gene_names = df_geo.index.tolist()

    meta_df = pd.read_csv(run_dir / "metadata.csv")
    families_arr = meta_df["family"].values
    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()
    N_fam = len(family_order)

    # Rows: geodesic first, then whichever baselines are present on disk.
    rows = [
        {
            "label": "Geodesic\n(GPN-Star)", "mat": geodesic, "cmap": "YlOrRd",
            "color": "#A8330E", "cbar": "Geodesic dist.", "nan_legend": False,
        }
    ]
    for cfg in WITHIN_BASELINES:
        if not cfg["plot"]:
            continue
        dist = load_within_baseline(run_dir, cfg, gene_names)
        if dist is not None:
            rows.append(
                {
                    "label": cfg["row_label"],
                    "mat": dist,
                    "cmap": cfg["cmap"],
                    "color": cfg["color"],
                    "cbar": cfg["cbar_label"],
                    "nan_legend": cfg["nan_legend"],
                }
            )
    n_rows = len(rows)

    per = {ri: {f: fam_sub(r["mat"], f, families_arr, gene_names) for f in family_order}
           for ri, r in enumerate(rows)}
    vmaxes = [shared_vmax(per[ri], family_order) for ri in range(n_rows)]
    cmaps = [make_cmap_with_nan(r["cmap"]) for r in rows]

    fig = plt.figure(figsize=(4 * N_fam, 3.8 * n_rows), dpi=300)
    gs = fig.add_gridspec(n_rows, N_fam, hspace=0.7, wspace=0.5)
    for ri, r in enumerate(rows):
        row_axes = []
        for fi, fam in enumerate(family_order):
            ax = fig.add_subplot(gs[ri, fi])
            row_axes.append(ax)
            _, genes, sub = per[ri][fam]
            title = fam.replace("_", " ").title() if ri == 0 else ""
            strip = FAMILY_COLORS[fam] if ri == 0 else None
            draw_within_hm(ax, sub, genes, cmaps[ri], vmaxes[ri], title, color_strip=strip)
            if fi == 0:
                ax.set_ylabel(
                    r["label"], fontsize=9, fontweight="bold", color=r["color"], labelpad=26
                )
        sm = plt.cm.ScalarMappable(cmap=cmaps[ri], norm=Normalize(0, vmaxes[ri]))
        sm.set_array([])
        cb = fig.colorbar(
            sm, ax=row_axes, orientation="vertical", fraction=0.007, pad=0.02, shrink=0.7
        )
        cb.set_label(r["cbar"], fontsize=6)
        cb.ax.tick_params(labelsize=5)

    if any(r["nan_legend"] for r in rows):
        nan_patch = mpatches.Patch(color="#DDDDDD", label="No Ensembl\nparalog data")
        fig.legend(handles=[nan_patch], loc="lower right", fontsize=6, framealpha=0.9)
    fig.suptitle(
        "Within-family geodesic vs baseline distances (per family)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(run_dir / "within_heatmaps.pdf", dpi=300)
    fig.savefig(run_dir / "within_heatmaps.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/within_heatmaps.{{pdf,png}}")


def make_within_correlations(run_dir: Path) -> None:
    """One figure: the three standardized within-family baselines (identical across the Evo2
    and GPN-Star pipelines) — k-mer composition, sequence identity, patristic tree — as
    per-family Spearman ρ. (CDS seq-identity and Ensembl paralog stay in the logs/heatmaps as
    supplementary, plot=False.)"""
    set_pub_style(title_size=9, tick_size=7)
    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    gene_names = df_geo.index.tolist()

    meta_df = pd.read_csv(run_dir / "metadata.csv")
    families_arr = meta_df["family"].values
    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()

    # Compute ρ for every gene-matrix baseline (so the supplementary ones stay in the logs)
    # but only draw bars for the `plot` ones — k-mer (the alignment-free composition control).
    series, printable = [], []
    for cfg in WITHIN_BASELINES:
        dist = load_within_baseline(run_dir, cfg, gene_names)
        if dist is None:
            continue
        # min_pairs=6 ⇔ ≥4 members, matching protein_alignment_patristic_seqid.py's MIN_MEMBERS so the
        # k-mer bar covers exactly the families the alignment baselines do (no spurious n=3
        # ρ=±1 bars for nitric_oxide_synthase / heme_oxygenase).
        rhos, pvals = per_family_rho(geodesic, dist, families_arr, family_order, min_pairs=6)
        printable.append((cfg["short"], rhos, pvals))
        if cfg["plot"]:
            series.append((cfg["short"], rhos, pvals, WITHIN_PALETTE["kmer"]))

    # The two alignment-based baselines (seq identity, patristic) from the shared CSVs that
    # scripts/baselines/protein_alignment_patristic_seqid.py writes — identical definition to the Evo2 figure.
    series += within_csv_series(run_dir, family_order)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    grouped_rho_bars(
        ax,
        family_order,
        series,
        "Within-family: geodesic vs k-mer composition, sequence identity & patristic tree (per family)",
        "Within-family Spearman ρ",
        ylim=(-0.4, 1.0),
    )
    fig.savefig(run_dir / "within_correlations.pdf", dpi=300)
    fig.savefig(run_dir / "within_correlations.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/within_correlations.{{pdf,png}}")

    print("\nPer-family within-family Spearman ρ (seq identity computed but not plotted):")
    for label, rhos, pvals in printable:
        vals = "  ".join(
            f"{f}={r:+.2f}{pstar(p)}" if np.isfinite(r) else f"{f}=n/a"
            for f, r, p in zip(family_order, rhos, pvals, strict=False)
        )
        print(f"  {label}: {vals}")


# ════════════════════════════════════════════════════════════════════════════════
# Between-family comparison: family-level geodesic vs. ground-truth baselines
# ════════════════════════════════════════════════════════════════════════════════


def make_between_figure(run_dir: Path) -> None:
    set_pub_style(title_size=9, tick_size=7)

    # ── Load core data ─────────────────────────────────────────────────────────
    centroid_candidates = list(run_dir.glob("*_centroid_distances.csv"))
    if not centroid_candidates:
        sys.exit(f"No *_centroid_distances.csv found in {run_dir}")
    model_prefix = centroid_candidates[0].stem.replace("_centroid_distances", "")

    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()
    N_fam = len(family_order)

    dist_centroid = pd.read_csv(run_dir / f"{model_prefix}_centroid_distances.csv", index_col=0)
    dist_jsd = pd.read_csv(run_dir / "pfam_jsd_distances.csv", index_col=0)

    # ── Family-level baselines ───────────────────────────────────────────────────
    seqid_fam_path = run_dir / "sequence_identity_family.csv"
    dist_seqid_fam = pd.read_csv(seqid_fam_path, index_col=0) if seqid_fam_path.exists() else None
    kmer_fam_path = run_dir / "kmer_distance_family.csv"
    dist_kmer_fam = pd.read_csv(kmer_fam_path, index_col=0) if kmer_fam_path.exists() else None

    idx_upper = np.triu_indices(N_fam, k=1)
    x_geo = dist_centroid.values[idx_upper]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Single row: 4 family-level (F×F) heatmaps [Geodesic][Pfam JSD][Seq Identity][k-mer]
    # + a between-family ρ bar chart (geodesic centroid vs each baseline, bootstrap CIs).
    fig = plt.figure(figsize=(26, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 5, wspace=0.5)
    ax_geo_fam = fig.add_subplot(gs[0])
    ax_jsd = fig.add_subplot(gs[1])
    ax_seqid = fig.add_subplot(gs[2])
    ax_kmer = fig.add_subplot(gs[3])
    ax_between = fig.add_subplot(gs[4])

    # ── Panels A–D: family-level heatmaps ─────────────────────────────────────
    draw_family_heatmap(
        fig, ax_geo_fam, dist_centroid.values, "A  Geodesic centroid\n(GPN-Star)",
        "YlOrRd", family_order, FAMILY_COLORS, cbar_label="Mean geodesic dist.",
    )
    draw_family_heatmap(
        fig, ax_jsd, dist_jsd.values, "B  Pfam HMM JSD\n(baseline 1)",
        "Blues", family_order, FAMILY_COLORS, cbar_label="JSD", show_yticks=False,
    )
    draw_family_heatmap(
        fig, ax_seqid,
        1.0 - dist_seqid_fam.values if dist_seqid_fam is not None else None,
        "C  Seq identity distance\n(baseline 2)",
        "Greens", family_order, FAMILY_COLORS, cbar_label="1 − mean identity", show_yticks=False,
    )
    draw_family_heatmap(
        fig, ax_kmer,
        dist_kmer_fam.values if dist_kmer_fam is not None else None,
        "D  k-mer divergence\n(baseline 3)",
        "Purples", family_order, FAMILY_COLORS, cbar_label="mean k-mer dist.", show_yticks=False,
    )

    # ── Panel E: between-family ρ (centroid geodesic vs each baseline) + CI ─────
    between_bars = [
        ("Pfam JSD\n(baseline 1)", dist_jsd.values, "#2C6E8A"),
        (
            "Seq identity\n(baseline 2)",
            1.0 - dist_seqid_fam.values if dist_seqid_fam is not None else None,
            "#2D6A4F",
        ),
        (
            "k-mer divergence\n(baseline 3)",
            dist_kmer_fam.values if dist_kmer_fam is not None else None,
            "#6A4C93",
        ),
    ]
    results = between_rho_bars(
        ax_between, between_bars, x_geo, idx_upper,
        "E  Between-family:\ngeodesic vs baselines",
        "Spearman ρ  (centroid geodesic vs baseline)",
        ylim=(-1, 1),
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    fig.savefig(run_dir / "between_comparison.pdf", dpi=300)
    fig.savefig(run_dir / "between_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {run_dir}/between_comparison.{{pdf,png}}")

    print("\nBetween-family Spearman ρ vs centroid geodesic:")
    for label, rho, ci_lo, ci_hi in results:
        tag = label.replace(chr(10), " ")
        if rho is not None:
            print(f"  {tag}: ρ={rho:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f"  {tag}: TBD")


# ── CLI ─────────────────────────────────────────────────────────────────────────


FIGURES = {
    "between": lambda run_dir: make_between_figure(run_dir),
    "within-heatmaps": lambda run_dir: make_within_heatmaps(run_dir),
    "within-correlations": lambda run_dir: make_within_correlations(run_dir),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, help="Dated results folder")
    p.add_argument(
        "--figures",
        nargs="+",
        choices=list(FIGURES),
        default=list(FIGURES),
        help="Which figures to generate (default: all)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    for name in args.figures:
        print(f"\n=== {name} ===")
        FIGURES[name](run_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
