"""Shared Matplotlib helpers for analysis figures."""

from pathlib import Path

import arcadia_style as acs  # noqa: I001 - sibling module, resolved via the scripts/ sys.path entry
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

# Shared colors for standardized within-family baselines.
WITHIN_PALETTE = acs.WITHIN_PALETTE


def within_csv_series(run_dir: Path, family_order: list) -> list:
    """Load the patristic within-family series."""
    specs = [
        (
            "within_family_patristic.csv",
            "spearman_geodesic_patristic",
            "p_patristic",
            "vs patristic tree",
            "patristic",
        ),
    ]
    series = []
    for csv_name, rho_col, p_col, label, key in specs:
        path = Path(run_dir) / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path).set_index("family")
        series.append(
            (
                label,
                [df.loc[f, rho_col] if f in df.index else np.nan for f in family_order],
                [df.loc[f, p_col] if f in df.index else np.nan for f in family_order],
                WITHIN_PALETTE[key],
            )
        )
    return series


def set_pub_style(
    font_size: int = 8,
    title_size: int = 8,
    tick_size: int = 6,
    legend_size: int = 7,
) -> None:
    """Apply the Arcadia style guide plus this project's compact type scale."""
    acs.setup(
        font_size=font_size, title_size=title_size, tick_size=tick_size, legend_size=legend_size
    )


def make_cmap_with_nan(base_cmap_name: str, nan_color=acs.MISSING):
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


# ── Statistics


def bootstrap_spearman(x, y, n_boot: int = 1000, seed: int = 42):
    """Spearman ρ of (x, y) with a paired-resample bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rho_obs, _ = spearmanr(x, y)
    boot = [spearmanr(x[rng.integers(0, n, n)], y[rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


def per_family_rho(geodesic, baseline, families_arr, family_order, min_pairs: int = 3):
    """Per-family within-family Spearman ρ of the geodesic vs a baseline distance matrix."""
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


# ── Shared panels for the between-family comparison figures


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
        ax.set_facecolor(acs.PLACEHOLDER_FILL)
        ax.text(
            0.5,
            0.5,
            "TBD",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color=acs.PLACEHOLDER_TEXT,
            fontweight="bold",
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
                    (xi - 0.5, n - 0.5),
                    1,
                    0.22,
                    color=family_colors[fam],
                    clip_on=False,
                    linewidth=0,
                )
            )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_linewidth(0.5)


def grouped_rho_bars(
    ax,
    group_labels,
    series,
    title,
    ylabel,
    ylim=(-0.4, 1.0),
    bar_width=0.38,
    tick_fontsize=9,
    show_values=True,
):
    """Per-group grouped Spearman-ρ bars with significance stars."""
    n_series = max(1, len(series))
    bar_width = min(bar_width, 0.9 / n_series)  # never let a group's bars overlap its neighbours
    x = np.arange(len(group_labels))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * bar_width
    for (label, rhos, pvals, color), off in zip(series, offsets, strict=False):
        ax.bar(
            x + off,
            [r if np.isfinite(r) else 0 for r in rhos],
            bar_width,
            label=label,
            color=color,
            zorder=3,
            edgecolor=acs.apc.white,
            linewidth=0.5,
        )
        if not show_values:
            continue
        pv = pvals if pvals is not None else [None] * len(rhos)
        for xi, (r, p) in enumerate(zip(rhos, pv, strict=False)):
            if np.isfinite(r):
                star = pstar(p) if (p is not None and np.isfinite(p)) else ""
                ax.text(
                    xi + off,
                    r + (0.02 if r >= 0 else -0.08),
                    f"{r:.2f}{star}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.axhline(0, color=acs.ZERO_LINE, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace("_", " ").title() for g in group_labels],
        rotation=30,
        ha="right",
        fontsize=tick_fontsize,
    )
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=acs.GRID, linestyle=":", linewidth=0.5, zorder=0)
    # x is a categorical axis whose labels are already title-cased and rotated above, so only
    # the numeric y gets the guide's monospaced treatment.
    acs.style_axes(ax, monospaced_axes="y")


def control_preservation_figure(
    out_path,
    fam_order,
    display,
    within_by_condition,
    between_by_condition=None,
    title_left="Composition controls: is the within-family geometry preserved?",
    ylabel_left="Within-family Spearman ρ (control vs natural geodesic)",
):
    """Draw within- and between-family control-preservation panels."""
    present = [(k, lbl, c) for k, lbl, c in display if k in within_by_condition]
    if not present:
        return False
    # Drop families with no finite within-family value across any shown control (e.g. families
    # too small to score) so the left panel has no empty slots.
    fam_order = [
        f
        for f in fam_order
        if any(np.isfinite(within_by_condition[k].get(f, np.nan)) for k, _, _ in present)
    ]
    betweens = [(between_by_condition or {}).get(k, np.nan) for k, _, _ in present]
    has_between = between_by_condition is not None and any(np.isfinite(b) for b in betweens)

    set_pub_style(title_size=11, tick_size=8)
    # Left-panel width scales with family count so a 15-family panel doesn't crush its
    # bars (the ~2–8-family human panels keep their compact look). Per-bar value labels
    # are dropped once there are too many families to place them legibly — the right
    # panel carries the numeric summary.
    n_fam, n_series = len(fam_order), len(present)
    left_w = max(9.0, 0.95 * n_fam)
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(left_w + 5.0, 6), dpi=200, gridspec_kw={"width_ratios": [left_w / 5.0, 1.1]}
    )

    series = []
    for key, lbl, color in present:
        wd = within_by_condition[key]
        series.append((lbl, [wd.get(f, np.nan) for f in fam_order], None, color))
    grouped_rho_bars(
        axL,
        fam_order,
        series,
        title_left,
        ylabel_left,
        ylim=(-0.2, 1.05),
        bar_width=0.9 / max(1, n_series),
        tick_fontsize=8,
        show_values=(n_fam <= 8),
    )

    labels = [lbl for _, lbl, _ in present]
    colors = [c for _, _, c in present]
    mean_within = [
        float(np.nanmean([within_by_condition[k].get(f, np.nan) for f in fam_order]))
        for k, _, _ in present
    ]
    xs = np.arange(len(present))
    if has_between:
        w = 0.38
        axR.bar(
            xs - w / 2,
            mean_within,
            w,
            label="Within (mean)",
            color=colors,
            edgecolor=acs.apc.white,
            linewidth=0.5,
            zorder=3,
        )
        axR.bar(
            xs + w / 2,
            betweens,
            w,
            label="Between (centroid)",
            color=colors,
            edgecolor=acs.apc.white,
            linewidth=0.5,
            hatch="//",
            alpha=0.85,
            zorder=3,
        )
        label_pairs = list(zip(xs - w / 2, mean_within, strict=False)) + list(
            zip(xs + w / 2, betweens, strict=False)
        )
    else:
        axR.bar(
            xs,
            mean_within,
            0.5,
            label="Within (mean)",
            color=colors,
            edgecolor=acs.apc.white,
            linewidth=0.5,
            zorder=3,
        )
        label_pairs = list(zip(xs, mean_within, strict=False))
    for x, v in label_pairs:
        if np.isfinite(v):
            axR.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")
    axR.axhline(
        1.0,
        color=acs.REFERENCE_LINE,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="natural (self = 1.0)",
    )
    axR.axhline(0, color=acs.ZERO_LINE, linewidth=0.8)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axR.set_ylim(-0.2, 1.05)
    axR.set_ylabel("Preservation ρ (control vs natural)")
    axR.set_title("Mean preservation", fontweight="bold")
    axR.legend(fontsize=8, loc="lower left")
    axR.spines[["top", "right"]].set_visible(False)
    axR.yaxis.grid(True, color=acs.GRID, linestyle=":", linewidth=0.5, zorder=0)
    acs.style_axes(axR, monospaced_axes="y")

    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", dpi=200)
    fig.savefig(f"{out_path}.png", dpi=200)
    plt.close(fig)
    return True


def between_rho_bars(ax, bars, x_geo, idx_upper, title, ylabel, ylim=(-0.2, 1.0)):
    """Between-group Spearman-ρ bars (a reference vector vs each baseline) + bootstrap CIs."""
    x = np.arange(len(bars))
    results = []
    for xi, (label, mat, color) in enumerate(bars):
        if mat is not None:
            rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, mat[idx_upper])
            ax.bar(
                xi, rho, color=color, width=0.5, zorder=3, edgecolor=acs.apc.white, linewidth=0.5
            )
            ax.errorbar(
                xi,
                rho,
                yerr=[[max(0.0, rho - ci_lo)], [max(0.0, ci_hi - rho)]],
                fmt="none",
                color=acs.apc.black,
                capsize=5,
                linewidth=1.2,
                zorder=4,
            )
            # Label just beyond the error bar, on the same side as the bar.
            if rho >= 0:
                ax.text(xi, ci_hi + 0.04, f"ρ={rho:.2f}", ha="center", va="bottom", fontsize=10)
            else:
                ax.text(xi, ci_lo - 0.04, f"ρ={rho:.2f}", ha="center", va="top", fontsize=10)
            results.append((label, rho, ci_lo, ci_hi))
        else:
            ax.bar(xi, 0, color=color, width=0.5, zorder=3)
            ax.text(
                xi, 0.04, "TBD", ha="center", va="bottom", fontsize=10, color=acs.PLACEHOLDER_TEXT
            )
            results.append((label, None, None, None))
    ax.axhline(0, color=acs.ZERO_LINE, linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=acs.GRID, linestyle=":", linewidth=0.5, zorder=0)
    acs.style_axes(ax, monospaced_axes="y")
    return results
