"""Shared matplotlib helpers for the figure scripts (gpnstar and evo2).

Holds the publication rcParams block, a NaN-aware colormap helper, and the
significance-star formatter that were previously copy-pasted into every figure
script. Scripts run as ``uv run python scripts/<dir>/<figure>.py`` add ``scripts/``
to sys.path before importing this module.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


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
