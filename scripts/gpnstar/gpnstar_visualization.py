"""
Visualizations for the GPN-Star gene-family geodesic analysis.

Figure sets, selectable via --figures (default: all available):

  family        : Figure 2 — per-family within-family geodesic heatmaps (2×5) plus
                  family-level (10×10) geodesic centroid / Pfam JSD / CDS seq-identity
                  / PANTHER heatmaps and a Spearman ρ bar chart with bootstrap CIs.
                  -> figure2.{pdf,png}

  within-paralog: within-family geodesic vs. Ensembl Compara paralog protein identity
                  (sparse; gray = pair not in the paralog DB).
                  -> figure_within_comparison.{pdf,png}

  within-seqid  : within-family geodesic vs. CDS nucleotide sequence identity (dense).
                  -> figure_within_comparison_seqid.{pdf,png}

A requested within-* figure whose baseline CSV is absent is skipped with a warning.

Usage:
    uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir RESULTS_DIR
    uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir RESULTS_DIR --figures family
    uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir RESULTS_DIR --figures within-paralog within-seqid
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_utils import make_cmap_with_nan, pstar, set_pub_style  # noqa: E402

from families import FAMILY_COLORS  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════
# Within-family comparison (geodesic vs. an evolutionary baseline)
# ════════════════════════════════════════════════════════════════════════════════

# Per-baseline config. `to_distance` maps the stored identity matrix to a distance
# in [0, 1] (NaN preserved); `reindex` aligns the matrix to gene order when it isn't
# already (the seq-identity CSV needs it, the paralog CSV doesn't).
BASELINES = {
    "paralog": {
        "csv": "ensembl_paralog_identity_genes.csv",
        "reindex": False,
        "to_distance": lambda m: np.where(np.isfinite(m), 1.0 - m / 100.0, np.nan),
        "cmap": "Blues",
        "row_label": "Ensembl Compara\n(1 − prot. ID)",
        "row_color": "#2C6E8A",
        "cbar_label": "1 − prot. identity",
        "bar_ylabel": "Spearman ρ  (geodesic vs. 1 − protein identity)",
        "title": "Within-family: GPN-Star geodesic vs. Ensembl Compara protein identity",
        "print_label": "Ensembl Compara protein identity distance",
        "show_nan_legend": True,
        "out_stem": "figure_within_comparison",
    },
    "seqid": {
        "csv": "sequence_identity_genes.csv",
        "reindex": True,
        "to_distance": lambda m: np.where(np.isfinite(m), 1.0 - m, np.nan),
        "cmap": "Greens",
        "row_label": "CDS seq identity\n(1 − identity)",
        "row_color": "#2D6A4F",
        "cbar_label": "1 − seq identity",
        "bar_ylabel": "Spearman ρ  (geodesic vs. 1 − seq identity)",
        "title": "Within-family: GPN-Star geodesic vs. CDS sequence identity",
        "print_label": "CDS sequence-identity distance",
        "show_nan_legend": False,
        "out_stem": "figure_within_comparison_seqid",
    },
}


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
        strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
        strip.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color_strip, linewidth=0))


def make_within_comparison_figure(run_dir: Path, baseline: str) -> None:
    cfg = BASELINES[baseline]
    set_pub_style()

    # ── Load geodesic + baseline matrices, both aligned to gene order ──────────
    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    gene_names = df_geo.index.tolist()

    baseline_path = run_dir / cfg["csv"]
    if not baseline_path.exists():
        print(f"  SKIP within-{baseline}: {cfg['csv']} not found in {run_dir}")
        return
    base_df = pd.read_csv(baseline_path, index_col=0)
    if cfg["reindex"]:
        base_df = base_df.reindex(index=gene_names, columns=gene_names)
    base_dist = cfg["to_distance"](base_df.values)
    np.fill_diagonal(base_dist, 0.0)

    meta_df = pd.read_csv(run_dir / "metadata.csv")
    families_arr = meta_df["family"].values
    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()
    N_fam = len(family_order)

    per_geo = {f: fam_sub(geodesic, f, families_arr, gene_names) for f in family_order}
    per_base = {f: fam_sub(base_dist, f, families_arr, gene_names) for f in family_order}

    # ── Per-family Spearman ρ (geodesic vs. baseline distance) ─────────────────
    per_rho: dict[str, tuple[float, float, int]] = {}
    for fam in family_order:
        _, genes, sub_geo = per_geo[fam]
        _, _, sub_base = per_base[fam]
        tri = np.triu_indices(len(genes), k=1)
        g_pairs, b_pairs = sub_geo[tri], sub_base[tri]
        valid = np.isfinite(b_pairs) & np.isfinite(g_pairs)
        n_valid = int(valid.sum())
        if n_valid < 3:
            per_rho[fam] = (np.nan, np.nan, n_valid)
        else:
            rho, pval = spearmanr(g_pairs[valid], b_pairs[valid])
            per_rho[fam] = (float(rho), float(pval), n_valid)

    vmax_geo = shared_vmax(per_geo, family_order)
    vmax_base = shared_vmax(per_base, family_order)
    cmap_geo = make_cmap_with_nan("YlOrRd")
    cmap_base = make_cmap_with_nan(cfg["cmap"])

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 13), dpi=300)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.48)
    gs_heat = gs_outer[0].subgridspec(2, N_fam, hspace=0.65, wspace=0.52)
    ax_bar_row = fig.add_subplot(gs_outer[1])

    geo_axes, base_axes = [], []
    for fi, fam in enumerate(family_order):
        ax_geo = fig.add_subplot(gs_heat[0, fi])
        ax_base = fig.add_subplot(gs_heat[1, fi])
        geo_axes.append(ax_geo)
        base_axes.append(ax_base)

        _, genes, sub_geo = per_geo[fam]
        _, _, sub_base = per_base[fam]
        draw_within_hm(ax_geo, sub_geo, genes, cmap_geo, vmax_geo,
                       fam.replace("_", " ").title(), color_strip=FAMILY_COLORS[fam])
        draw_within_hm(ax_base, sub_base, genes, cmap_base, vmax_base, "")

        rho, pval, n_valid = per_rho[fam]
        n_pairs = len(genes) * (len(genes) - 1) // 2
        ann = f"ρ={rho:.2f}{pstar(pval)}\n({n_valid}/{n_pairs})" if np.isfinite(rho) else f"n={n_valid}/{n_pairs}"
        ax_base.text(
            0.5, -0.32, ann, transform=ax_base.transAxes, ha="center", va="top", fontsize=5.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
        )

    fig.text(0.005, 0.80, "Geodesic\n(model)", fontsize=8, fontweight="bold",
             ha="left", va="center", rotation=90, color="#333333")
    fig.text(0.005, 0.57, cfg["row_label"], fontsize=8, fontweight="bold",
             ha="left", va="center", rotation=90, color=cfg["row_color"])

    for axes, cmap, vmax, label in [
        (geo_axes, cmap_geo, vmax_geo, "Geodesic dist."),
        (base_axes, cmap_base, vmax_base, cfg["cbar_label"]),
    ]:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.007, pad=0.02, shrink=0.7)
        cb.set_label(label, fontsize=6)
        cb.ax.tick_params(labelsize=5)

    if cfg["show_nan_legend"]:
        nan_patch = mpatches.Patch(color="#DDDDDD", label="No Ensembl\nparalog data")
        fig.legend(handles=[nan_patch], loc="lower right", fontsize=6, framealpha=0.9)

    # ── Bottom bar chart ───────────────────────────────────────────────────────
    x = np.arange(N_fam)
    rhos = [per_rho[f][0] for f in family_order]
    pvals = [per_rho[f][1] for f in family_order]
    nvs = [per_rho[f][2] for f in family_order]
    ax_bar_row.bar(
        x, [r if np.isfinite(r) else 0 for r in rhos],
        color=[FAMILY_COLORS[f] for f in family_order],
        width=0.65, zorder=3, edgecolor="white", linewidth=0.5,
    )
    for xi, (rho, pval, nv) in enumerate(zip(rhos, pvals, nvs)):
        if np.isfinite(rho):
            ypos = rho + (0.04 if rho >= 0 else -0.09)
            ax_bar_row.text(xi, ypos, f"ρ={rho:.2f}{pstar(pval)}", ha="center", va="bottom", fontsize=6.5)
        else:
            ax_bar_row.text(xi, 0.04, "n<3" if nv < 3 else "no pairs",
                            ha="center", va="bottom", fontsize=6.5, color="#999999")

    ax_bar_row.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax_bar_row.set_xticks(x)
    ax_bar_row.set_xticklabels([f.replace("_", " ").title() for f in family_order],
                               rotation=30, ha="right", fontsize=7)
    ax_bar_row.set_ylim(-1, 1.15)
    ax_bar_row.set_ylabel(cfg["bar_ylabel"], fontsize=7.5)
    ax_bar_row.set_title(cfg["title"], fontsize=9, fontweight="bold", pad=6)
    ax_bar_row.spines[["top", "right"]].set_visible(False)
    ax_bar_row.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    stem = cfg["out_stem"]
    fig.savefig(run_dir / f"{stem}.pdf", dpi=300)
    fig.savefig(run_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"Saved to {run_dir}/{stem}.{{pdf,png}}")

    print(f"\nPer-family Spearman ρ (geodesic vs. {cfg['print_label']}):")
    for fam in family_order:
        rho, pval, nv = per_rho[fam]
        _, genes, _ = per_geo[fam]
        n_total = len(genes) * (len(genes) - 1) // 2
        if np.isfinite(rho):
            print(f"  {fam:<22}: ρ={rho:+.3f}  p={pval:.3f} {pstar(pval, ns='(n.s.)')}  ({nv}/{n_total} pairs)")
        else:
            print(f"  {fam:<22}: insufficient pairs ({nv}/{n_total})")


# ════════════════════════════════════════════════════════════════════════════════
# Figure 2: family-level geodesic vs. ground-truth baselines
# ════════════════════════════════════════════════════════════════════════════════


def bootstrap_spearman(x, y, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    rho_obs, _ = spearmanr(x, y)
    boot = [spearmanr(x[rng.integers(0, n, n)], y[rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


def make_family_figure(run_dir: Path) -> None:
    set_pub_style(title_size=9, tick_size=7)

    # ── Load core data ─────────────────────────────────────────────────────────
    centroid_candidates = list(run_dir.glob("*_centroid_distances.csv"))
    if not centroid_candidates:
        sys.exit(f"No *_centroid_distances.csv found in {run_dir}")
    model_prefix = centroid_candidates[0].stem.replace("_centroid_distances", "")

    geo_labeled = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_labeled:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir} — re-run embed_and_geodesic_genes.py")
    df_geo = pd.read_csv(geo_labeled[0], index_col=0)
    geodesic = df_geo.values
    gene_names = df_geo.index.tolist()

    meta_df = pd.read_csv(run_dir / "metadata.csv")
    families_arr = meta_df["family"].values
    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()
    N_fam = len(family_order)

    dist_centroid = pd.read_csv(run_dir / f"{model_prefix}_centroid_distances.csv", index_col=0)
    dist_jsd = pd.read_csv(run_dir / "pfam_jsd_distances.csv", index_col=0)

    # ── Optional baselines ───────────────────────────────────────────────────────
    seqid_fam_path = run_dir / "sequence_identity_family.csv"
    dist_seqid_fam = pd.read_csv(seqid_fam_path, index_col=0) if seqid_fam_path.exists() else None
    seqid_gene_path = run_dir / "sequence_identity_genes.csv"
    seqid_genes = pd.read_csv(seqid_gene_path, index_col=0) if seqid_gene_path.exists() else None
    panther_path = run_dir / "panther_distances.csv"
    dist_panther = pd.read_csv(panther_path, index_col=0) if panther_path.exists() else None

    idx_upper = np.triu_indices(N_fam, k=1)
    x_geo = dist_centroid.values[idx_upper]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Top: 10 per-family geodesic heatmaps (2×5). Bottom: 4 family-level heatmaps
    # [Geodesic][Pfam JSD][Seq Identity][PANTHER] + Spearman ρ bar chart.
    fig = plt.figure(figsize=(28, 16), dpi=300)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.44)
    gs_top = gs_outer[0].subgridspec(2, 5, hspace=0.72, wspace=0.55)
    gs_bot = gs_outer[1].subgridspec(1, 5, wspace=0.42)
    ax_geo_fam = fig.add_subplot(gs_bot[0])
    ax_jsd = fig.add_subplot(gs_bot[1])
    ax_seqid = fig.add_subplot(gs_bot[2])
    ax_panther = fig.add_subplot(gs_bot[3])
    ax_bar = fig.add_subplot(gs_bot[4])

    def draw_family_heatmap(ax, matrix, title, cmap, vmin=None, vmax=None,
                            cbar_label="", show_yticks=True, tbd=False):
        """Draw one 10×10 family-level heatmap (or a 'TBD' placeholder)."""
        fam_labels = [f.replace("_", " ").title() for f in family_order]
        if tbd:
            ax.set_facecolor("#F5F5F5")
            ax.text(0.5, 0.5, "TBD", transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#AAAAAA", fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            norm = Normalize(vmin=vmin if vmin is not None else matrix.min(),
                             vmax=vmax if vmax is not None else matrix.max())
            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            ax.set_xticks(range(N_fam))
            ax.set_yticks(range(N_fam) if show_yticks else [])
            ax.set_xticklabels(fam_labels, rotation=40, ha="right", fontsize=5.5)
            if show_yticks:
                ax.set_yticklabels(fam_labels, fontsize=5.5)
            cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cb.set_label(cbar_label, fontsize=6)
            cb.ax.tick_params(labelsize=5)
            for xi, fam in enumerate(family_order):  # family color ticks on x-axis
                ax.add_patch(mpatches.Rectangle(
                    (xi - 0.5, N_fam - 0.5), 1, 0.22, color=FAMILY_COLORS[fam],
                    transform=ax.transData, clip_on=False, linewidth=0,
                ))
        ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
        ax.spines[["top", "right", "bottom", "left"]].set_linewidth(0.5)

    # ── Panel A: per-family geodesic heatmaps ─────────────────────────────────
    within_vals = []
    for fam in family_order:
        idx = np.where(families_arr == fam)[0]
        sub = geodesic[np.ix_(idx, idx)]
        within_vals.extend(sub[np.triu_indices(len(idx), k=1)])
    vmax_within = float(np.percentile(within_vals, 95))

    for fi, fam in enumerate(family_order):
        ax = fig.add_subplot(gs_top[fi // 5, fi % 5])
        fam_idx = np.where(families_arr == fam)[0]
        fam_genes = [gene_names[i] for i in fam_idx]
        sub_geo = geodesic[np.ix_(fam_idx, fam_idx)]

        ax.imshow(sub_geo, cmap="YlOrRd", norm=Normalize(0, vmax_within),
                  aspect="equal", interpolation="nearest")
        ax.set_xticks(range(len(fam_genes)))
        ax.set_yticks(range(len(fam_genes)))
        ax.set_xticklabels(fam_genes, rotation=90, fontsize=4.5)
        ax.set_yticklabels(fam_genes, fontsize=4.5)
        ax.tick_params(length=2, pad=1)

        color = FAMILY_COLORS[fam]
        ax.set_title(fam.replace("_", " ").title(), fontsize=7, color=color, fontweight="bold", pad=8)
        strip = ax.inset_axes([0, 1.04, 1, 0.07], transform=ax.transAxes)
        strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
        strip.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color, linewidth=0))

        if seqid_genes is not None and len(fam_idx) >= 3:  # within-family ρ vs seq identity
            tri = np.triu_indices(len(fam_idx), k=1)
            geo_pairs = sub_geo[tri]
            si_pairs = (1.0 - seqid_genes.values[np.ix_(fam_idx, fam_idx)])[tri]
            rho_w, _ = spearmanr(geo_pairs, si_pairs)
            ax.text(0.97, 0.03, f"ρ={rho_w:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))

    cb_ax = fig.add_axes([0.922, 0.57, 0.006, 0.33])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=0, vmax=vmax_within))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("Geodesic distance", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.text(0.02, 0.97, "A  Within-family geodesic distances", fontsize=10, fontweight="bold", va="top")

    # ── Panels B–E: family-level heatmaps ─────────────────────────────────────
    draw_family_heatmap(ax_geo_fam, dist_centroid.values, "B  Geodesic centroid\n(model)",
                        "YlOrRd", cbar_label="Mean geodesic dist.")
    draw_family_heatmap(ax_jsd, dist_jsd.values, "C  Pfam HMM JSD\n(baseline 1)",
                        "Blues", cbar_label="JSD", show_yticks=False)
    if dist_seqid_fam is not None:
        draw_family_heatmap(ax_seqid, 1.0 - dist_seqid_fam.values, "D  Seq identity distance\n(baseline 2)",
                            "Greens", cbar_label="1 − mean identity", show_yticks=False)
    else:
        draw_family_heatmap(ax_seqid, None, "D  Seq identity\n(baseline 2)", None, tbd=True)
    if dist_panther is not None:
        draw_family_heatmap(ax_panther, dist_panther.values, "E  PANTHER / TimeTree\n(baseline 3)",
                            "Purples", cbar_label="Branch length / Mya", show_yticks=False)
    else:
        draw_family_heatmap(ax_panther, None, "E  PANTHER / TimeTree\n(baseline 3)", None, tbd=True)

    # ── Panel F: Spearman ρ bar chart ─────────────────────────────────────────
    baselines_bar = [("Pfam JSD\n(baseline 1)", dist_jsd.values, "#2C6E8A")]
    if dist_seqid_fam is not None:
        baselines_bar.append(("Seq identity\n(baseline 2)", 1.0 - dist_seqid_fam.values, "#E76F51"))
    else:
        baselines_bar.append(("Seq identity\n(baseline 2)", None, "#BDBDBD"))
    if dist_panther is not None:
        baselines_bar.append(("PANTHER\n(baseline 3)", dist_panther.values, "#9C4DC4"))
    else:
        baselines_bar.append(("PANTHER / TimeTree\n(baseline 3)", None, "#BDBDBD"))

    x = np.arange(len(baselines_bar))
    for xi, (label, mat, color) in enumerate(baselines_bar):
        if mat is not None:
            rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, mat[idx_upper])
            ax_bar.bar(xi, rho, color=color, width=0.52, zorder=3, edgecolor="white", linewidth=0.5)
            ax_bar.errorbar(xi, rho, yerr=[[rho - ci_lo], [ci_hi - rho]],
                            fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
            ax_bar.text(xi, ci_hi + 0.04, f"ρ={rho:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            ax_bar.bar(xi, 0, color=color, width=0.52, zorder=3)
            ax_bar.text(xi, 0.04, "TBD", ha="center", va="bottom", fontsize=8, color="#888888")

    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([b[0] for b in baselines_bar], fontsize=7)
    ax_bar.set_ylim(-1, 1)
    ax_bar.set_ylabel("Spearman ρ  (centroid geodesic vs. baseline)", fontsize=7.5)
    ax_bar.set_title("F  Baseline correlation\n(GPN-Star Vertebrate)", fontsize=8, fontweight="bold", pad=6)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

    # ── Save ───────────────────────────────────────────────────────────────────
    fig.savefig(run_dir / "figure2.pdf", dpi=300)
    fig.savefig(run_dir / "figure2.png", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {run_dir}/figure2.{{pdf,png}}")

    print("\nBaseline Spearman ρ vs centroid geodesic:")
    for label, mat, _ in baselines_bar:
        if mat is not None:
            rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, mat[idx_upper])
            print(f"  {label.replace(chr(10), ' ')}: ρ={rho:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            print(f"  {label.replace(chr(10), ' ')}: TBD")


# ── CLI ─────────────────────────────────────────────────────────────────────────


FIGURES = {
    "family": lambda run_dir: make_family_figure(run_dir),
    "within-paralog": lambda run_dir: make_within_comparison_figure(run_dir, "paralog"),
    "within-seqid": lambda run_dir: make_within_comparison_figure(run_dir, "seqid"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Dated results folder")
    p.add_argument("--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES),
                   help="Which figures to generate (default: all)")
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
