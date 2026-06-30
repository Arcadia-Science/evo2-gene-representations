"""Shared matplotlib helpers for the figure scripts (gpnstar and evo2).

Holds the publication rcParams block, a NaN-aware colormap helper, the
significance-star formatter, and the family-level heatmap / Spearman-ρ bar-chart
panels shared by the between-family comparison figures. Scripts run as
``uv run python scripts/<dir>/<figure>.py`` add ``scripts/`` to sys.path before
importing this module.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

# Shared colour for the three standardized within-family baselines, used identically by the
# Evo2 and GPN-Star within_correlations figures so the two read as one comparison.
WITHIN_PALETTE = {"kmer": "#6A4C93", "seqid": "#2C6E8A", "patristic": "#B5651D"}


def within_csv_series(run_dir: Path, family_order: list) -> list:
    """The two alignment-based within-family series (seq identity, patristic tree) from the
    shared within_family_*.csv that scripts/baselines/protein_alignment_patristic_seqid.py writes. Each is a
    (label, rhos, pvals, color) tuple aligned to family_order, ready for grouped_rho_bars.
    k-mer is added by each pipeline separately (it needs no alignment)."""
    specs = [
        ("within_family_seqid.csv", "spearman_geodesic_seqid", "p_seqid", "vs seq identity", "seqid"),
        ("within_family_patristic.csv", "spearman_geodesic_patristic", "p_patristic", "vs patristic tree", "patristic"),
    ]
    series = []
    for csv_name, rho_col, p_col, label, key in specs:
        path = Path(run_dir) / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path).set_index("family")
        series.append((
            label,
            [df.loc[f, rho_col] if f in df.index else np.nan for f in family_order],
            [df.loc[f, p_col] if f in df.index else np.nan for f in family_order],
            WITHIN_PALETTE[key],
        ))
    return series


def set_pub_style(
    font_size: int = 8,
    title_size: int = 8,
    tick_size: int = 6,
    legend_size: int = 7,
) -> None:
    """Apply the shared publication rcParams (Helvetica, 300 dpi, type-42 fonts)."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": title_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def make_cmap_with_nan(base_cmap_name: str, nan_color: str = "#DDDDDD"):
    """Return a copy of a named colormap that renders NaN/masked cells in nan_color."""
    cmap = plt.get_cmap(base_cmap_name).copy()
    cmap.set_bad(color=nan_color)
    return cmap


def pstar(pval: float, ns: str = "") -> str:
    """Significance stars for a p-value: ***<0.001, **<0.01, *<0.05, else `ns`."""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ns


# ── Statistics ────────────────────────────────────────────────────────────────


def bootstrap_spearman(x, y, n_boot: int = 1000, seed: int = 42):
    """Spearman ρ of (x, y) with a paired-resample bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rho_obs, _ = spearmanr(x, y)
    boot = [spearmanr(x[rng.integers(0, n, n)], y[rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


def per_family_rho(geodesic, baseline, families_arr, family_order, min_pairs: int = 3):
    """Per-family within-family Spearman ρ of the geodesic vs a baseline distance matrix.

    Both matrices are gene×gene in the same row order. Returns (rhos, pvals) lists
    aligned to family_order, with NaN where a family has < min_pairs finite pairs.
    """
    rhos, pvals = [], []
    for fam in family_order:
        idx = np.where(families_arr == fam)[0]
        tri = np.triu_indices(len(idx), k=1)
        g = geodesic[np.ix_(idx, idx)][tri]
        b = baseline[np.ix_(idx, idx)][tri]
        valid = np.isfinite(g) & np.isfinite(b)
        if valid.sum() < min_pairs:
            rhos.append(np.nan)
            pvals.append(np.nan)
        else:
            r, p = spearmanr(g[valid], b[valid])
            rhos.append(float(r))
            pvals.append(float(p))
    return rhos, pvals


# ── Shared panels for the between-family comparison figures ─────────────────────


def draw_family_heatmap(
    fig,
    ax,
    matrix,
    title,
    cmap,
    family_order,
    family_colors,
    vmin=None,
    vmax=None,
    cbar_label="",
    show_yticks=True,
    label_fontsize=7,
    title_fontsize=9,
):
    """One F×F family-level heatmap with a per-column family colour strip.

    Pass matrix=None to render a grey 'TBD' placeholder for a missing baseline.
    """
    fam_labels = [f.replace("_", " ").title() for f in family_order]
    n = len(family_order)
    if matrix is None:
        ax.set_facecolor("#F5F5F5")
        ax.text(
            0.5, 0.5, "TBD", transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="#AAAAAA", fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        norm = Normalize(
            vmin=vmin if vmin is not None else matrix.min(),
            vmax=vmax if vmax is not None else matrix.max(),
        )
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n) if show_yticks else [])
        ax.set_xticklabels(fam_labels, rotation=40, ha="right", fontsize=label_fontsize)
        if show_yticks:
            ax.set_yticklabels(fam_labels, fontsize=label_fontsize)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
        for xi, fam in enumerate(family_order):  # family colour ticks on the x-axis
            ax.add_patch(
                mpatches.Rectangle(
                    (xi - 0.5, n - 0.5), 1, 0.22, color=family_colors[fam],
                    clip_on=False, linewidth=0,
                )
            )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_linewidth(0.5)


def grouped_rho_bars(
    ax, group_labels, series, title, ylabel, ylim=(-0.4, 1.0), bar_width=0.38, tick_fontsize=9
):
    """Per-group grouped Spearman-ρ bars with significance stars.

    series: list of (label, rhos, pvals, color); each rhos/pvals list is aligned to
    group_labels (NaN entries are skipped). pvals may be None for "no stars".
    """
    x = np.arange(len(group_labels))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_width
    for (label, rhos, pvals, color), off in zip(series, offsets, strict=False):
        ax.bar(
            x + off, [r if np.isfinite(r) else 0 for r in rhos], bar_width,
            label=label, color=color, zorder=3, edgecolor="white", linewidth=0.5,
        )
        pv = pvals if pvals is not None else [None] * len(rhos)
        for xi, (r, p) in enumerate(zip(rhos, pv, strict=False)):
            if np.isfinite(r):
                star = pstar(p) if (p is not None and np.isfinite(p)) else ""
                ax.text(
                    xi + off, r + (0.02 if r >= 0 else -0.08), f"{r:.2f}{star}",
                    ha="center", va="bottom", fontsize=7,
                )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("_", " ").title() for g in group_labels],
        rotation=30, ha="right", fontsize=tick_fontsize,
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)


def between_rho_bars(ax, bars, x_geo, idx_upper, title, ylabel, ylim=(-0.2, 1.0)):
    """Between-group Spearman-ρ bars (a reference vector vs each baseline) + bootstrap CIs.

    bars: list of (label, matrix_or_None, color); each matrix is F×F and correlated
    (upper triangle) against x_geo. A None matrix renders a grey 'TBD' bar.
    Returns [(label, rho, ci_lo, ci_hi), ...] (rho/ci None for TBD bars) for logging.
    """
    x = np.arange(len(bars))
    results = []
    for xi, (label, mat, color) in enumerate(bars):
        if mat is not None:
            rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, mat[idx_upper])
            ax.bar(xi, rho, color=color, width=0.5, zorder=3, edgecolor="white", linewidth=0.5)
            ax.errorbar(
                xi, rho, yerr=[[max(0.0, rho - ci_lo)], [max(0.0, ci_hi - rho)]],
                fmt="none", color="black", capsize=5, linewidth=1.2, zorder=4,
            )
            # Label just beyond the error bar, on the same side as the bar.
            if rho >= 0:
                ax.text(xi, ci_hi + 0.04, f"ρ={rho:.2f}", ha="center", va="bottom", fontsize=10)
            else:
                ax.text(xi, ci_lo - 0.04, f"ρ={rho:.2f}", ha="center", va="top", fontsize=10)
            results.append((label, rho, ci_lo, ci_hi))
        else:
            ax.bar(xi, 0, color=color, width=0.5, zorder=3)
            ax.text(xi, 0.04, "TBD", ha="center", va="bottom", fontsize=10, color="#888888")
            results.append((label, None, None, None))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    return results
