"""Global matplotlib styling for the Arcadia 2026 style guide."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

from .colors import BACKGROUNDS, BLACK, palette, register_cmaps
from .fonts import MONO, NEXT, ensure_fonts

__all__ = [
    "apply",
    "RC",
    "SIZES",
    "MARGIN",
    "GAP",
    "FONT_SIZE",
    "WEIGHT",
    "style_axes",
    "mono_ticks",
    "set_cycle",
    "ArcadiaNumberFormatter",
]

# Panel widths in px from "Panel sizes". `data` is the width the whole chart
# (axis titles and tick labels included) is allowed to occupy: width - 2*30.
SIZES = {
    "full_wide": 1000,
    "float": 650,
    "half_square": 490,
}
MARGIN = 30.0  # px from panel edge to the outermost chart element
GAP = 20.0  # px between adjacent panels

FONT_SIZE = {
    "panel_letter": 38.0,
    "key_title": 17.0,
    "axis_title": 15.0,
    "body": 15.0,
    "number": 14.5,
    "annotation": 15.0,
}

WEIGHT = {
    "panel_letter": 400,  # Regular
    "key_title": 600,  # SemiBold
    "axis_title": 500,  # Medium
    "body": 400,  # Regular
    "annotation": 200,  # ExtraLight
}

LINE_WIDTH = {
    "axis": 0.75,
    "leader": 1.0,
    "arrow": 2.0,
    "key_underline": 1.5,
}

TICK_LEN = 5.0  # px
LABEL_PAD = 5.0  # px from axis to tick labels
TITLE_PAD = 10.0  # px from tick labels to axis title


class ArcadiaNumberFormatter(ScalarFormatter):
    """Tick numbers the way the guide writes them: 1,000 and −0.1."""

    def __init__(self, comma: bool = True, **kwargs):
        kwargs.setdefault("useOffset", False)
        kwargs.setdefault("useMathText", False)
        super().__init__(**kwargs)
        self.comma = comma

    def __call__(self, x, pos=None):
        text = super().__call__(x, pos)
        if not text:
            return text
        if self.comma:
            sign = ""
            body = text
            if body[0] in "-−+":
                sign, body = body[0], body[1:]
            if "e" not in body.lower():
                whole, _, frac = body.partition(".")
                if whole.isdigit() and len(whole) > 3:
                    whole = f"{int(whole):,}"
                body = f"{whole}.{frac}" if frac else whole
            text = sign + body
        return text.replace("-", "−")


def _rc(background: str = "white") -> dict:
    face = BACKGROUNDS.get(str(background).lower(), background)
    body, number = FONT_SIZE["body"], FONT_SIZE["number"]
    return {
        # ---- type -------------------------------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": [
            NEXT,
            "Atkinson Hyperlegible",
            "Helvetica Neue",
            "Helvetica",
            "Arial",
            "DejaVu Sans",
        ],
        "font.monospace": [MONO, "Atkinson Hyperlegible Mono", "Menlo", "DejaVu Sans Mono"],
        "font.size": body,
        "font.weight": WEIGHT["body"],
        "text.color": BLACK,
        # Units and exponents come up constantly in axis titles, and Atkinson
        # has no superscript-minus glyph, so route mathtext through the same
        # family: r"Uptake ($\mathrm{nmol\ h^{-1}}$)" stays on-brand.
        "mathtext.fontset": "custom",
        "mathtext.default": "regular",
        "mathtext.rm": NEXT,
        "mathtext.it": f"{NEXT}:italic",
        # SemiBold is the heaviest weight the guide uses, so bold math maps to
        # it rather than making matplotlib hunt for a 700 weight that we never
        # pin.
        "mathtext.bf": f"{NEXT}:semibold",
        "mathtext.sf": NEXT,
        "mathtext.tt": MONO,
        "mathtext.cal": f"{NEXT}:italic",
        "axes.labelsize": FONT_SIZE["axis_title"],
        "axes.labelweight": WEIGHT["axis_title"],
        "axes.labelcolor": BLACK,
        "axes.labelpad": TITLE_PAD,
        "axes.titlesize": FONT_SIZE["axis_title"],
        # Charts are titled in the figure caption, never in the artwork, so
        # nothing here encourages an in-axes title.
        "xtick.labelsize": number,
        "ytick.labelsize": number,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.labelcolor": BLACK,
        "ytick.labelcolor": BLACK,
        # ---- spines, ticks, grid ---------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": BLACK,
        "axes.linewidth": LINE_WIDTH["axis"],
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": BACKGROUNDS["gray"],
        "grid.linewidth": LINE_WIDTH["axis"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.minor.size": TICK_LEN * 0.6,
        "ytick.minor.size": TICK_LEN * 0.6,
        "xtick.major.width": LINE_WIDTH["axis"],
        "ytick.major.width": LINE_WIDTH["axis"],
        "xtick.minor.width": LINE_WIDTH["axis"],
        "ytick.minor.width": LINE_WIDTH["axis"],
        "xtick.major.pad": LABEL_PAD,
        "ytick.major.pad": LABEL_PAD,
        # ---- marks ------------------------------------------------------
        "axes.prop_cycle": mpl.cycler(color=palette(6)),
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "patch.edgecolor": "none",
        "scatter.edgecolors": "none",
        "boxplot.boxprops.linewidth": LINE_WIDTH["axis"],
        "boxplot.medianprops.color": BLACK,
        "hatch.linewidth": LINE_WIDTH["axis"],
        "errorbar.capsize": 3,
        # ---- key (legend) ----------------------------------------------
        # Geometry from "Chart layout": 15 px swatch, 5 px swatch-to-label,
        # no frame. legend() sizes are multiples of the legend font size.
        "legend.frameon": False,
        "legend.fontsize": body,
        "legend.title_fontsize": FONT_SIZE["key_title"],
        "legend.handlelength": 15.0 / body,
        "legend.handleheight": 15.0 / body,
        "legend.handletextpad": 5.0 / body,
        "legend.labelspacing": 10.0 / body,
        "legend.borderpad": 0.0,
        "legend.borderaxespad": 10.0 / body,
        "legend.columnspacing": 15.0 / body,
        "legend.markerscale": 1.0,
        # ---- canvas -----------------------------------------------------
        "figure.facecolor": face,
        "axes.facecolor": face,
        "savefig.facecolor": face,
        "figure.edgecolor": "none",
        "savefig.edgecolor": "none",
        # 144 = 2x the guide's 72, so on-screen and inline previews are crisp
        # without changing the layout: figsize is in inches derived from px/72,
        # so one point stays one guide pixel at any dpi.
        "figure.dpi": 144,
        "savefig.dpi": 300,
        "savefig.bbox": None,  # honor the exact panel size; do not crop
        "figure.constrained_layout.use": False,
        # Keep text as text so figures stay editable in Illustrator.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
    }


RC = _rc()


def apply(background: str = "white", grid: bool = False, cycle=None) -> None:
    """Install the Arcadia style globally."""
    ensure_fonts()
    register_cmaps()
    rc = _rc(background)
    if grid:
        rc["axes.grid"] = True
        rc["axes.grid.axis"] = "y"
    if cycle is not None:
        colors = palette(cycle) if isinstance(cycle, (str, int)) else list(cycle)
        rc["axes.prop_cycle"] = mpl.cycler(color=colors)
    mpl.rcParams.update(rc)
    _fix_inline_backend()


def _fix_inline_backend() -> bool:
    """Stop Jupyter's inline backend from cropping figures to their content."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    patched = False
    # The live singleton is what the inline backend actually reads...
    try:
        from matplotlib_inline.config import InlineBackend

        config = InlineBackend.instance()
        kwargs = dict(config.print_figure_kwargs or {})
        kwargs["bbox_inches"] = None
        config.print_figure_kwargs = kwargs
        patched = True
    except Exception:
        pass
    # ...and the shell config covers a singleton created later.
    try:
        section = shell.config.InlineBackend
        kwargs = dict(section.get("print_figure_kwargs", {}) or {})
        kwargs["bbox_inches"] = None
        section["print_figure_kwargs"] = kwargs
        patched = True
    except Exception:
        pass
    return patched


def set_cycle(cycle) -> None:
    """Set the default color cycle from a palette name, count, or list."""
    colors = palette(cycle) if isinstance(cycle, (str, int)) else list(cycle)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)


def mono_ticks(ax, axis: str = "both", comma: bool = True) -> None:
    """Render tick numbers in Atkinson Mono with Arcadia number formatting."""
    for which, mpl_axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        if axis not in ("both", which):
            continue
        numeric = not any(
            t.get_text() and not _looks_numeric(t.get_text()) for t in mpl_axis.get_ticklabels()
        )
        if numeric:
            ax.tick_params(axis=which, labelfontfamily="monospace")
            if mpl_axis.get_scale() == "linear":
                mpl_axis.set_major_formatter(ArcadiaNumberFormatter(comma=comma))


def _looks_numeric(text: str) -> bool:
    cleaned = text.replace("−", "-").replace(",", "").replace("%", "")
    cleaned = cleaned.replace("$", "").strip()
    try:
        float(cleaned)
    except ValueError:
        return cleaned.startswith("$") or cleaned in {"", "-"}
    return True


def style_axes(ax, mono: bool = True, ticks: str = "auto") -> None:
    """Bring an axes created elsewhere (seaborn, pandas.plot) into the style."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(LINE_WIDTH["axis"])
        ax.spines[side].set_color(BLACK)
    ax.tick_params(
        which="major",
        direction="out",
        length=TICK_LEN,
        width=LINE_WIDTH["axis"],
        color=BLACK,
        labelcolor=BLACK,
        pad=LABEL_PAD,
    )
    ax.tick_params(
        which="minor", direction="out", length=TICK_LEN * 0.6, width=LINE_WIDTH["axis"], color=BLACK
    )
    if ticks == "auto":
        ax.tick_params(top=False, right=False)
    ax.xaxis.label.set_size(FONT_SIZE["axis_title"])
    ax.yaxis.label.set_size(FONT_SIZE["axis_title"])
    ax.xaxis.label.set_weight(WEIGHT["axis_title"])
    ax.yaxis.label.set_weight(WEIGHT["axis_title"])
    ax.xaxis.labelpad = TITLE_PAD
    ax.yaxis.labelpad = TITLE_PAD
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontsize(FONT_SIZE["number"])
    if mono:
        mono_ticks(ax)
    if ax.get_title():
        ax.set_title("")
    return ax
