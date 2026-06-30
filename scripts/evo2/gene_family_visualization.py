"""Visualizations for the Evo2 cross-kingdom gene-family geodesic analysis.

The direct analog of scripts/gpnstar/gpnstar_visualization.py, adapted for the Evo2
pipeline's data: ~400 CDS/family (vs ~31 human paralogs), so the within-family
heatmaps are ordered by host taxonomy (domain → group) with a domain colour strip
instead of unreadable per-gene labels. The ground-truth baselines map across cleanly:

  GPN-Star CDS seq-identity   →  Evo2 k-mer sequence divergence   (within-kmer figure)
  GPN-Star Pfam/PANTHER phylo →  Evo2 taxonomic-rank distance     (within-taxonomy figure)

Figure sets, selectable via --figures (default: all):

  family          : Figure 2 — per-family within-family geodesic heatmaps (Axis B) +
                    family-level (F×F) geodesic-centroid / k-mer / taxonomy heatmaps
                    (Axis A) + per-family Spearman ρ bars.   -> figure2.{pdf,png}
  within-kmer     : within-family geodesic vs k-mer sequence divergence, per family.
                    -> figure_within_comparison_kmer.{pdf,png}
  within-taxonomy : within-family geodesic vs taxonomic-rank distance, per family.
                    -> figure_within_comparison_taxonomy.{pdf,png}

Usage:
    uv run python scripts/evo2/gene_family_visualization.py \
        --run-dir results/YYYY-MM-DD_evo2-gene-families
    uv run python scripts/evo2/gene_family_visualization.py --run-dir RESULTS_DIR --figures family
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_utils import pstar, set_pub_style  # noqa: E402

# Embedding metadata (family, domain, group per org_gene) lives next to the cache,
# not in the dated results dir; fall back to it when run_dir has no copy.
EMBED_META_FALLBACK = Path("data/evo2_gene_families/embeddings/metadata.csv")

FAMILY_COLORS = {
    "globins": "#D1495B",
    "heme_copper_oxidase": "#E76F51",
    "cytochrome_p450": "#EDAE49",
    "olfactory_receptors": "#66A182",
    "opsins": "#2E86AB",
    "ras_gtpases": "#8E7DBE",
}
# Domain colour strip — the bacterial → archaeal → eukaryotic gradient that Axis B
# expects the within-family geodesic to recapitulate.
DOMAIN_COLORS = {
    "Bacteria": "#56B4E9",
    "Archaea": "#E69F00",
    "Eukaryota": "#009E73",
    "Unknown": "#BBBBBB",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def load_run(run_dir: Path):
    """Load geodesic (org_gene-ordered) + aligned per-CDS metadata for a run dir."""
    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    org_genes = df_geo.index.tolist()

    meta_path = run_dir / "metadata.csv"
    if not meta_path.exists():
        meta_path = EMBED_META_FALLBACK
    if not meta_path.exists():
        sys.exit(f"No metadata.csv in {run_dir} or {EMBED_META_FALLBACK}")
    meta = pd.read_csv(meta_path)
    # Align metadata to the geodesic row order via org_gene (the matrices are stored
    # in manifest order; positional mismatch would silently mislabel everything).
    meta = meta.set_index("org_gene").reindex(org_genes).reset_index()
    if meta["family"].isna().any():
        sys.exit("metadata.csv does not cover every org_gene in the geodesic matrix")

    families = meta["family"].to_numpy()
    family_order = sorted(pd.unique(families).tolist())
    return geodesic, org_genes, meta, families, family_order


def family_order_idx(meta_fam: pd.DataFrame) -> np.ndarray:
    """Local row order within a family: group taxonomically (domain → group → org)."""
    return np.argsort(
        [
            f"{d}|{g}|{o}"
            for d, g, o in zip(
                meta_fam["domain"], meta_fam["group"], meta_fam["org_gene"], strict=False
            )
        ],
        kind="stable",
    )


def aggregate_family_matrix(mat, families, family_order):
    """F×F: within-family mean (upper-tri) on the diagonal, cross-block mean off it."""
    F = len(family_order)
    out = np.zeros((F, F), dtype=float)
    for fi, fa in enumerate(family_order):
        ia = np.where(families == fa)[0]
        for fj, fb in enumerate(family_order):
            ib = np.where(families == fb)[0]
            if fi == fj:
                tri = mat[np.ix_(ia, ia)][np.triu_indices(len(ia), k=1)]
                out[fi, fj] = tri.mean() if len(tri) else 0.0
            else:
                out[fi, fj] = mat[np.ix_(ia, ib)].mean()
    return out


def domain_strip(ax, domains, where="top"):
    """Draw a per-row/col domain colour strip alongside an ordered heatmap."""
    n = len(domains)
    if where == "top":
        strip = ax.inset_axes([0, 1.01, 1, 0.035], transform=ax.transAxes)
        for i, d in enumerate(domains):
            strip.add_patch(
                mpatches.Rectangle(
                    (i / n, 0), 1 / n, 1, color=DOMAIN_COLORS.get(d, "#BBBBBB"), linewidth=0
                )
            )
    else:  # left
        strip = ax.inset_axes([-0.05, 0, 0.035, 1], transform=ax.transAxes)
        for i, d in enumerate(domains):
            strip.add_patch(
                mpatches.Rectangle(
                    (0, 1 - (i + 1) / n),
                    1,
                    1 / n,
                    color=DOMAIN_COLORS.get(d, "#BBBBBB"),
                    linewidth=0,
                )
            )
    strip.set_xlim(0, 1)
    strip.set_ylim(0, 1)
    strip.axis("off")


# ── Per-family within-family ρ (read from the analysis CSV when present) ───────


def load_axisB(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "axisB_within_family_correlations.csv"
    return pd.read_csv(p) if p.exists() else None


# ════════════════════════════════════════════════════════════════════════════════
# Within-family comparison figure (geodesic vs one baseline), per family
# ════════════════════════════════════════════════════════════════════════════════

WITHIN_CFG = {
    "kmer": {
        "npy": "kmer_distance.npy",
        "cmap": "Greens",
        "row_label": "k-mer divergence\n(1 − cosine)",
        "row_color": "#2D6A4F",
        "cbar_label": "k-mer distance",
        "rho_col": "spearman_geodesic_kmer",
        "p_col": "p_kmer",
        "title": "Within-family: Evo2 geodesic vs k-mer sequence divergence",
        "stem": "figure_within_comparison_kmer",
    },
    "taxonomy": {
        "npy": "taxonomic_distance.npy",
        "cmap": "Purples",
        "row_label": "Taxonomic-rank\ndistance",
        "row_color": "#6A4C93",
        "cbar_label": "taxonomic distance",
        "rho_col": "spearman_geodesic_taxonomy",
        "p_col": "p_taxonomy",
        "title": "Within-family: Evo2 geodesic vs host taxonomic distance",
        "stem": "figure_within_comparison_taxonomy",
    },
}


def make_within_comparison_figure(run_dir: Path, baseline: str) -> None:
    cfg = WITHIN_CFG[baseline]
    set_pub_style()
    geodesic, org_genes, meta, families, family_order = load_run(run_dir)
    base_path = run_dir / cfg["npy"]
    if not base_path.exists():
        print(f"  SKIP within-{baseline}: {cfg['npy']} not found in {run_dir}")
        return
    base = np.load(base_path)
    axisB = load_axisB(run_dir)
    N_fam = len(family_order)

    # Shared robust colour scales across families (95th pct of within-family pairs).
    def shared_vmax(mat):
        vals = []
        for fam in family_order:
            idx = np.where(families == fam)[0]
            vals.extend(mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
        return float(np.percentile(vals, 95)) if vals else 1.0

    vmax_geo, vmax_base = shared_vmax(geodesic), shared_vmax(base)

    fig = plt.figure(figsize=(26, 13), dpi=300)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.5)
    gs_heat = gs_outer[0].subgridspec(2, N_fam, hspace=0.3, wspace=0.35)
    ax_bar = fig.add_subplot(gs_outer[1])

    geo_axes, base_axes = [], []
    for fi, fam in enumerate(family_order):
        idx = np.where(families == fam)[0]
        order = family_order_idx(meta.iloc[idx])
        oidx = idx[order]
        domains = meta.iloc[oidx]["domain"].tolist()

        ax_g = fig.add_subplot(gs_heat[0, fi])
        geo_axes.append(ax_g)
        ax_b = fig.add_subplot(gs_heat[1, fi])
        base_axes.append(ax_b)
        ax_g.imshow(
            geodesic[np.ix_(oidx, oidx)],
            cmap="YlOrRd",
            norm=Normalize(0, vmax_geo),
            aspect="equal",
            interpolation="nearest",
        )
        ax_b.imshow(
            base[np.ix_(oidx, oidx)],
            cmap=cfg["cmap"],
            norm=Normalize(0, vmax_base),
            aspect="equal",
            interpolation="nearest",
        )
        for ax in (ax_g, ax_b):
            ax.set_xticks([])
            ax.set_yticks([])
        domain_strip(ax_g, domains, "top")
        domain_strip(ax_b, domains, "top")
        ax_g.set_title(
            fam.replace("_", " ").title(),
            fontsize=8,
            color=FAMILY_COLORS[fam],
            fontweight="bold",
            pad=12,
        )

        if axisB is not None and fam in set(axisB["family"]):
            r = axisB[axisB["family"] == fam].iloc[0]
            rho, pval, n = r[cfg["rho_col"]], r[cfg["p_col"]], int(r["n_members"])
            ax_b.text(
                0.5,
                -0.13,
                f"ρ={rho:+.2f}{pstar(pval)}  (n={n})",
                transform=ax_b.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
            )

    fig.text(
        0.006,
        0.78,
        "Geodesic\n(Evo2)",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=90,
        color="#A8330E",
    )
    fig.text(
        0.006,
        0.50,
        cfg["row_label"],
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=90,
        color=cfg["row_color"],
    )
    for axes, cmap, vmax, label in [
        (geo_axes, "YlOrRd", vmax_geo, "Geodesic dist."),
        (base_axes, cfg["cmap"], vmax_base, cfg["cbar_label"]),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.006, pad=0.02, shrink=0.7)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    # Domain legend
    handles = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        ncol=4,
        title="Host domain (heatmap order)",
        title_fontsize=7,
        framealpha=0.9,
    )

    # Bottom: per-family ρ bar chart
    _draw_rho_bars(
        ax_bar,
        run_dir,
        family_order,
        cfg["rho_col"],
        cfg["p_col"],
        cfg["title"],
        f"Spearman ρ (geodesic vs {cfg['cbar_label']})",
    )

    fig.savefig(run_dir / f"{cfg['stem']}.pdf", dpi=300)
    fig.savefig(run_dir / f"{cfg['stem']}.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/{cfg['stem']}.{{pdf,png}}")


def _draw_rho_bars(ax, run_dir, family_order, rho_col, p_col, title, ylabel):
    axisB = load_axisB(run_dir)
    x = np.arange(len(family_order))
    rhos, pvals = [], []
    for fam in family_order:
        if axisB is not None and fam in set(axisB["family"]):
            r = axisB[axisB["family"] == fam].iloc[0]
            rhos.append(float(r[rho_col]))
            pvals.append(float(r[p_col]))
        else:
            rhos.append(np.nan)
            pvals.append(np.nan)
    ax.bar(
        x,
        [r if np.isfinite(r) else 0 for r in rhos],
        color=[FAMILY_COLORS[f] for f in family_order],
        width=0.65,
        zorder=3,
        edgecolor="white",
        linewidth=0.5,
    )
    for xi, (rho, pval) in enumerate(zip(rhos, pvals, strict=False)):
        if np.isfinite(rho):
            ax.text(
                xi,
                rho + (0.03 if rho >= 0 else -0.07),
                f"ρ={rho:.2f}{pstar(pval)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace("_", " ").title() for f in family_order], rotation=25, ha="right", fontsize=8
    )
    ax.set_ylim(-1, 1.1)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)


# ════════════════════════════════════════════════════════════════════════════════
# Figure 2: Axis A (between-family) + Axis B (within-family) overview
# ════════════════════════════════════════════════════════════════════════════════


def make_family_figure(run_dir: Path) -> None:
    set_pub_style(title_size=9, tick_size=7)
    geodesic, org_genes, meta, families, family_order = load_run(run_dir)
    axisB = load_axisB(run_dir)
    N_fam = len(family_order)

    kmer = (
        np.load(run_dir / "kmer_distance.npy") if (run_dir / "kmer_distance.npy").exists() else None
    )
    tax = (
        np.load(run_dir / "taxonomic_distance.npy")
        if (run_dir / "taxonomic_distance.npy").exists()
        else None
    )

    cen_path = list(run_dir.glob("*_centroid_distances.csv"))
    centroid = (
        pd.read_csv(cen_path[0], index_col=0).values
        if cen_path
        else aggregate_family_matrix(geodesic, families, family_order)
    )
    kmer_fam = aggregate_family_matrix(kmer, families, family_order) if kmer is not None else None
    tax_fam = aggregate_family_matrix(tax, families, family_order) if tax is not None else None

    n_rows_top = 2
    n_cols_top = math.ceil(N_fam / n_rows_top)
    fig = plt.figure(figsize=(26, 16), dpi=300)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.4)
    gs_top = gs_outer[0].subgridspec(n_rows_top, n_cols_top, hspace=0.45, wspace=0.3)
    gs_bot = gs_outer[1].subgridspec(1, 4, wspace=0.4)
    ax_cen = fig.add_subplot(gs_bot[0])
    ax_kmer = fig.add_subplot(gs_bot[1])
    ax_tax = fig.add_subplot(gs_bot[2])
    ax_bar = fig.add_subplot(gs_bot[3])

    # ── Panel A: per-family within-family geodesic heatmaps (taxonomy-ordered) ──
    within_vals = []
    for fam in family_order:
        idx = np.where(families == fam)[0]
        within_vals.extend(geodesic[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
    vmax_within = float(np.percentile(within_vals, 95))

    for fi, fam in enumerate(family_order):
        ax = fig.add_subplot(gs_top[fi // n_cols_top, fi % n_cols_top])
        idx = np.where(families == fam)[0]
        order = family_order_idx(meta.iloc[idx])
        oidx = idx[order]
        ax.imshow(
            geodesic[np.ix_(oidx, oidx)],
            cmap="YlOrRd",
            norm=Normalize(0, vmax_within),
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        domain_strip(ax, meta.iloc[oidx]["domain"].tolist(), "top")
        ax.set_title(
            fam.replace("_", " ").title(),
            fontsize=8.5,
            color=FAMILY_COLORS[fam],
            fontweight="bold",
            pad=12,
        )
        if axisB is not None and fam in set(axisB["family"]):
            r = axisB[axisB["family"] == fam].iloc[0]
            ax.text(
                0.97,
                0.03,
                f"ρ_kmer={r['spearman_geodesic_kmer']:.2f}\nρ_tax={r['spearman_geodesic_taxonomy']:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
            )

    cb_ax = fig.add_axes([0.92, 0.57, 0.006, 0.33])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_within))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Geodesic distance", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.text(
        0.02,
        0.97,
        "A  Within-family geodesic distances (Axis B — rows ordered by host taxonomy)",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    handles = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        ncol=4,
        title="Host domain",
        title_fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.91, 0.99),
    )

    # ── Panels B–D: family-level (Axis A) heatmaps ──────────────────────────────
    def draw_fam_hm(ax, mat, title, cmap, cbar_label, show_y=True):
        labels = [f.replace("_", " ").title() for f in family_order]
        im = ax.imshow(mat, cmap=cmap, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(N_fam))
        ax.set_yticks(range(N_fam) if show_y else [])
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        if show_y:
            ax.set_yticklabels(labels, fontsize=6.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(cbar_label, fontsize=6.5)
        cb.ax.tick_params(labelsize=5.5)
        for xi, fam in enumerate(family_order):
            ax.add_patch(
                mpatches.Rectangle(
                    (xi - 0.5, N_fam - 0.5),
                    1,
                    0.18,
                    color=FAMILY_COLORS[fam],
                    clip_on=False,
                    linewidth=0,
                )
            )
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=10)

    draw_fam_hm(ax_cen, centroid, "B  Geodesic centroid (Evo2)", "YlOrRd", "mean geodesic")
    if kmer_fam is not None:
        draw_fam_hm(
            ax_kmer, kmer_fam, "C  k-mer divergence", "Greens", "mean k-mer dist.", show_y=False
        )
    if tax_fam is not None:
        draw_fam_hm(
            ax_tax, tax_fam, "D  Taxonomic distance", "Purples", "mean tax. dist.", show_y=False
        )

    # ── Panel E: per-family Axis-B ρ bars (k-mer vs taxonomy, grouped) ──────────
    x = np.arange(N_fam)
    w = 0.38
    if axisB is not None:
        ab = axisB.set_index("family")
        rk = [
            ab.loc[f, "spearman_geodesic_kmer"] if f in ab.index else np.nan for f in family_order
        ]
        rt = [
            ab.loc[f, "spearman_geodesic_taxonomy"] if f in ab.index else np.nan
            for f in family_order
        ]
        ax_bar.bar(x - w / 2, rk, w, label="vs k-mer divergence", color="#2D6A4F", zorder=3)
        ax_bar.bar(x + w / 2, rt, w, label="vs taxonomy", color="#6A4C93", zorder=3)
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f.replace("_", " ").title() for f in family_order], rotation=30, ha="right", fontsize=7
    )
    ax_bar.set_ylim(-0.2, 1.0)
    ax_bar.set_ylabel("Within-family Spearman ρ", fontsize=8)
    ax_bar.set_title(
        "E  Axis B: geodesic vs divergence & taxonomy", fontsize=8.5, fontweight="bold", pad=6
    )
    ax_bar.legend(fontsize=6.5, loc="upper right")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    fig.savefig(run_dir / "figure2.pdf", dpi=300)
    fig.savefig(run_dir / "figure2.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/figure2.{{pdf,png}}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

FIGURES = {
    "family": lambda run_dir: make_family_figure(run_dir),
    "within-kmer": lambda run_dir: make_within_comparison_figure(run_dir, "kmer"),
    "within-taxonomy": lambda run_dir: make_within_comparison_figure(run_dir, "taxonomy"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, help="Dated results folder")
    p.add_argument("--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES))
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
