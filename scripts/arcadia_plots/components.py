"""Figure, panel, key, and annotation components from the Arcadia style guide."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .colors import BLACK, CHARCOAL, CHATEAU
from .style import (
    FONT_SIZE, GAP, LINE_WIDTH, MARGIN, SIZES, TICK_LEN, WEIGHT,
    mono_ticks, style_axes,
)

__all__ = [
    "figure", "panels", "panel_letter", "key", "colorbar_key",
    "annotate", "save", "size_px", "PT",
]

PT = 1.0 / 72.0  # one guide pixel, in inches

# Default panel heights, as a fraction of width, when the caller does not say.
# full_wide panels are usually a wide strip; the smaller sizes read best a
# little taller than square.
_DEFAULT_ASPECT = {"full_wide": 0.42, "float": 0.72, "half_square": 0.80}

_NAME_RE = re.compile(
    r"^Fig[\dSA-Z]+[a-z]?_[A-Za-z0-9][A-Za-z0-9-]*_(Full_wide|Float|Half_square)$"
)


def size_px(size) -> float:
    """Panel width in px from a guide name ('float') or a raw number."""
    if isinstance(size, str):
        key = size.lower().replace("-", "_")
        if key not in SIZES:
            raise KeyError(f"size must be one of {sorted(SIZES)} or a width in px")
        return float(SIZES[key])
    width = float(size)
    if not 490 <= width <= 1000:
        warnings.warn(
            f"Panel width {width:g} px is outside the guide's 490-1,000 px range; "
            "figures wider than 1,000 px will be downscaled in a pub.",
            stacklevel=3,
        )
    return width


def _apply_layout(fig, width_pt: float, height_pt: float, margin: float, gap: float):
    """Constrained layout tuned to the guide's spacing."""
    fig.set_layout_engine(
        "constrained",
        w_pad=margin * PT,
        h_pad=margin * PT,
        wspace=gap / width_pt,
        hspace=gap / height_pt,
    )
    return fig


def figure(size="float", height: float | None = None, margin: float = MARGIN,
           **kwargs):
    """One-panel figure at a guide panel size. Returns (fig, ax)."""
    width = size_px(size)
    if height is None:
        height = width * _DEFAULT_ASPECT.get(
            size if isinstance(size, str) else "float", 0.72
        )
    if height > 1200:
        warnings.warn("The guide asks for panels under 1,200 px tall.", stacklevel=2)
    fig = plt.figure(figsize=(width * PT, height * PT), **kwargs)
    _apply_layout(fig, width, height, margin, GAP)
    ax = fig.add_subplot()
    fig._arcadia = {"width": width, "height": height, "cells": {ax: (0.0, 0.0, width, height)},
                    "keys": [], "panels": {}}
    fig.arcadia_panels = fig._arcadia["panels"]
    return fig, ax


def panels(mosaic="AB", size="full_wide", height: float | None = None,
           letters: bool = True, margin: float = MARGIN, gap: float = GAP,
           width_ratios=None, height_ratios=None, **kwargs):
    """Multi-panel figure laid out by mosaic string. Returns (fig, dict_of_axes)."""
    width = size_px(size)
    grid = _parse_mosaic(mosaic)
    nrows, ncols = len(grid), len(grid[0])
    if height is None:
        aspect = _DEFAULT_ASPECT.get(size if isinstance(size, str) else "float", 0.72)
        height = width * aspect * nrows
    if height > 1200:
        warnings.warn("The guide asks for panels under 1,200 px tall.", stacklevel=2)

    fig = plt.figure(figsize=(width * PT, height * PT), **kwargs)
    fig._arcadia = {"width": width, "height": height, "cells": {}, "keys": [],
                    "panels": {}}
    fig.arcadia_panels = fig._arcadia["panels"]

    # Parent gridspec spans the whole artboard; wspace/hspace are fractions of
    # the average panel size, which is how gridspec measures them.
    wr = list(width_ratios or [1] * ncols)
    hr = list(height_ratios or [1] * nrows)
    panel_w = (width - gap * (ncols - 1)) / ncols
    panel_h = (height - gap * (nrows - 1)) / nrows
    gs = fig.add_gridspec(
        nrows, ncols, left=0, right=1, bottom=0, top=1,
        wspace=gap / panel_w if ncols > 1 else 0,
        hspace=gap / panel_h if nrows > 1 else 0,
        width_ratios=wr, height_ratios=hr,
    )

    # Subfigures inherit the parent's layout engine, so the margin is set once
    # on the figure and applied inside every panel: each chart lands `margin`
    # px inside its own panel, which puts `2 * margin + gap` between the charts
    # of neighboring panels -- the guide's 30/20/30.
    fig.set_layout_engine("constrained", w_pad=margin * PT, h_pad=margin * PT,
                          wspace=0, hspace=0)

    facecolor = plt.rcParams["figure.facecolor"]
    axd = {}
    for name, (rows, cols) in _mosaic_spans(grid).items():
        subfig = fig.add_subfigure(gs[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1])
        subfig.set_facecolor(facecolor)
        ax = subfig.add_subplot()
        axd[name] = ax
        fig._arcadia["panels"][name] = subfig
        fig._arcadia["cells"][ax] = _cell_rect(
            rows, cols, width, height, gap, wr, hr
        )
        if letters:
            panel_letter(ax, str(name))
    return fig, axd


def _parse_mosaic(mosaic) -> list[list[str]]:
    """Normalize a mosaic spec into a rectangular 2-D list of panel keys."""
    if isinstance(mosaic, str):
        text = mosaic.strip("\n")
        rows = text.split(";") if ";" in text else text.split("\n")
        grid = [list(row.strip()) for row in rows if row.strip()]
    else:
        grid = [list(row) for row in mosaic]
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        raise ValueError("every row of the mosaic must have the same length")
    return grid


def _mosaic_spans(grid) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Map each panel key to its ((row0, row1), (col0, col1)) span."""
    spans: dict[str, list[list[int]]] = {}
    for i, row in enumerate(grid):
        for j, key in enumerate(row):
            if key == ".":
                continue
            if key not in spans:
                spans[key] = [[i, i], [j, j]]
            else:
                spans[key][0][0] = min(spans[key][0][0], i)
                spans[key][0][1] = max(spans[key][0][1], i)
                spans[key][1][0] = min(spans[key][1][0], j)
                spans[key][1][1] = max(spans[key][1][1], j)
    for key, ((r0, r1), (c0, c1)) in list(spans.items()):
        cells = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)}
        actual = {(i, j) for i, row in enumerate(grid)
                  for j, k in enumerate(row) if k == key}
        if cells != actual:
            raise ValueError(f"panel {key!r} is not a rectangle in the mosaic")
    return {k: ((v[0][0], v[0][1]), (v[1][0], v[1][1])) for k, v in spans.items()}


def _cell_rect(rows, cols, width, height, gap, wr, hr):
    """Panel rectangle (x0, y0, x1, y1) in px, measured from the bottom-left."""
    ncols, nrows = len(wr), len(hr)
    avail_w = width - gap * (ncols - 1)
    avail_h = height - gap * (nrows - 1)
    widths = [avail_w * r / sum(wr) for r in wr]
    heights = [avail_h * r / sum(hr) for r in hr]
    x0 = sum(widths[:cols[0]]) + gap * cols[0]
    x1 = sum(widths[:cols[1] + 1]) + gap * cols[1]
    y1 = height - (sum(heights[:rows[0]]) + gap * rows[0])
    y0 = height - (sum(heights[:rows[1] + 1]) + gap * rows[1])
    return (x0, y0, x1, y1)


def _root_figure(obj):
    """The top-level Figure behind an artist, axes, or SubFigure."""
    fig = getattr(obj, "figure", obj) or obj
    seen = set()
    while getattr(fig, "figure", None) is not None and id(fig) not in seen:
        seen.add(id(fig))
        parent = fig.figure
        if parent is fig:
            break
        fig = parent
    return fig


def _meta(obj) -> dict:
    fig = _root_figure(obj)
    if not hasattr(fig, "_arcadia"):
        fig._arcadia = {
            "width": fig.get_figwidth() * 72, "height": fig.get_figheight() * 72,
            "cells": {}, "keys": [], "panels": {},
        }
    return fig._arcadia


def _all_axes(fig):
    """Every axes in a figure, including those inside panels (subfigures)."""
    out = list(fig.axes)
    for sub in getattr(fig, "subfigs", ()):
        out.extend(_all_axes(sub))
    return out


def panel_letter(ax, letter: str, dx: float = 10.0, dy: float = 10.0, **kwargs):
    """Draw a panel letter at the top-left of the panel `ax` sits in."""
    fig = _root_figure(ax)
    meta = _meta(ax)
    width, height = meta["width"], meta["height"]
    cell = meta["cells"].get(ax, (0.0, 0.0, width, height))
    x0, _, _, y1 = cell
    return fig.text(
        (x0 + dx) / width, (y1 - dy) / height, letter,
        ha="left", va="top",
        fontsize=kwargs.pop("fontsize", FONT_SIZE["panel_letter"]),
        fontweight=kwargs.pop("fontweight", WEIGHT["panel_letter"]),
        color=kwargs.pop("color", CHATEAU), **kwargs,
    )


# Key (legend)

def key(ax, title: str | None = None, entries=None, loc: str = "upper right",
        kind: str = "auto", underline: bool = True, ncol: int = 1, **kwargs):
    """Draw a key in the Arcadia "Basic key" style and return the Legend."""
    handles, labels = [], []
    if entries is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        items = entries.items() if isinstance(entries, dict) else entries
        for label, spec in items:
            color = spec if isinstance(spec, str) else spec.get("color", BLACK)
            style = kind if kind != "auto" else "patch"
            handles.append(_swatch(style, color, spec))
            labels.append(label)
    if not handles:
        raise ValueError(
            "Nothing to put in the key: pass entries=..., or label your artists "
            "with plot(..., label='...') before calling key()."
        )

    legend = ax.legend(handles, labels, title=title, loc=loc, ncols=ncol, **kwargs)
    legend.set_alignment("left")
    if title:
        legend.get_title().set_fontsize(FONT_SIZE["key_title"])
        legend.get_title().set_fontweight(WEIGHT["key_title"])
        legend.get_title().set_color(BLACK)
        if underline:
            _add_title_rule(legend)
    for text in legend.get_texts():
        text.set_color(BLACK)
        text.set_fontsize(FONT_SIZE["body"])
    return legend


def _swatch(style: str, color: str, spec):
    opts = spec if isinstance(spec, dict) else {}
    if style == "patch":
        return Patch(facecolor=color, edgecolor="none")
    if style == "line":
        return Line2D([], [], color=color, linewidth=opts.get("linewidth", 1.5),
                      linestyle=opts.get("linestyle", "-"))
    return Line2D([], [], color=color, linestyle="none",
                  marker=opts.get("marker", "o"),
                  markersize=opts.get("markersize", 6))


def _renderer(fig):
    for getter in ("get_renderer",):
        candidate = getattr(fig.canvas, getter, None)
        if candidate is not None:
            try:
                return candidate()
            except Exception:
                pass
    try:
        return fig._get_renderer()
    except Exception:
        fig.draw_without_rendering()
        return fig._get_renderer()


def _add_title_rule(legend, pad: float = 3.0):
    """Attach the 1.5 pt Chateau rule under a key title."""
    owner = legend.figure  # the panel (SubFigure) or the figure itself
    line = Line2D([0, 0], [0, 0], transform=owner.transFigure,
                  color=CHATEAU, linewidth=LINE_WIDTH["key_underline"],
                  solid_capstyle="butt", zorder=legend.get_zorder())
    owner.add_artist(line)
    _meta(legend)["keys"].append(
        {"legend": legend, "line": line, "pad": pad, "owner": owner}
    )
    sync_keys(legend)
    return line


def sync_keys(obj) -> None:
    """Re-place every key title rule against the current layout."""
    fig = _root_figure(obj)
    records = _meta(fig)["keys"]
    if not records:
        return
    fig.draw_without_rendering()
    renderer = _renderer(fig)
    for rec in records:
        legend, line, owner = rec["legend"], rec["line"], rec["owner"]
        try:
            box = legend.get_window_extent(renderer)
            title = legend.get_title().get_window_extent(renderer)
        except Exception:  # pragma: no cover - no renderer available
            continue
        pad = rec["pad"] * fig.dpi / 72.0
        inv = owner.transFigure.inverted()
        (x0, y), (x1, _) = inv.transform([(box.x0, title.y0 - pad),
                                          (box.x1, title.y0 - pad)])
        line.set_data([x0, x1], [y, y])


def colorbar_key(mappable, ax, title: str | None = None, loc: str = "upper right",
                 width: float = 150.0, height: float = 12.0, pad: float = 10.0,
                 **kwargs):
    """A gradient key for continuous color, sized and placed in px."""
    if loc in ("above", "above right", "above left", "top", "right"):
        return _colorbar_outside(mappable, ax, title, loc, width, height, pad,
                                 **kwargs)
    corners = {
        "upper right": (1.0, 1.0, 1, 1), "upper left": (0.0, 1.0, 0, 1),
        "lower right": (1.0, 0.0, 1, 0), "lower left": (0.0, 0.0, 0, 0),
    }
    if loc not in corners:
        raise KeyError(f"loc must be a panel corner, 'above*', or 'right', not {loc!r}")
    fx, fy, sx, sy = corners[loc]
    fig = _root_figure(ax)
    # The panel size is only final once the layout has run, and the corner
    # offsets below are in px, so lay out first and measure.
    fig.draw_without_rendering()
    box = ax.get_window_extent()
    ax_w = box.width / fig.dpi * 72
    ax_h = box.height / fig.dpi * 72
    w, h = width / ax_w, height / ax_h
    px, py = pad / ax_w, pad / ax_h
    x = fx - w - px if sx else px
    y = fy - h - py if sy else py

    cax = ax.inset_axes([x, y, w, h])
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal", **kwargs)
    cbar.outline.set_visible(False)
    cax.tick_params(length=TICK_LEN * 0.6, width=LINE_WIDTH["axis"], pad=3,
                    labelsize=FONT_SIZE["number"], labelfontfamily="monospace",
                    color=BLACK, labelcolor=BLACK)
    cax._arcadia_key_axes = True
    if title:
        cax.set_title(title, fontsize=FONT_SIZE["key_title"],
                      fontweight=WEIGHT["key_title"], color=BLACK, pad=6)
    return cbar


def _colorbar_outside(mappable, ax, title, loc, width, height, pad, **kwargs):
    """Gradient key outside the chart, using layout space instead of data space."""
    fig = _root_figure(ax)
    fig.draw_without_rendering()
    box = ax.get_window_extent()
    ax_w, ax_h = box.width / fig.dpi * 72, box.height / fig.dpi * 72

    vertical = loc == "right"
    span = min(height if vertical else width, ax_h if vertical else ax_w)
    thickness = width if vertical else height
    if vertical:
        span = min(150.0, ax_h)
        thickness = 12.0 if height == 12.0 else height
    shrink = span / (ax_h if vertical else ax_w)
    anchor = {"above left": (0.0, 0.0), "above right": (1.0, 0.0)}.get(loc, (0.5, 0.0))

    cbar = ax.figure.colorbar(
        mappable, ax=ax, location="right" if vertical else "top",
        shrink=shrink, aspect=max(span / thickness, 1.0),
        pad=pad / (ax_w if vertical else ax_h),
        anchor=anchor if not vertical else (0.0, 0.5),
        **kwargs,
    )
    cax = cbar.ax
    cax._arcadia_key_axes = True
    cbar.outline.set_visible(False)
    if not vertical:
        # Keep the guide's reading order: title, gradient, then numbers.
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("bottom")
    cax.tick_params(length=TICK_LEN * 0.6, width=LINE_WIDTH["axis"], pad=3,
                    labelsize=FONT_SIZE["number"], labelfontfamily="monospace",
                    color=BLACK, labelcolor=BLACK)
    if title:
        if vertical:
            cax.set_ylabel(title, fontsize=FONT_SIZE["key_title"],
                           fontweight=WEIGHT["key_title"], color=BLACK)
        else:
            cax.set_title(title, fontsize=FONT_SIZE["key_title"],
                          fontweight=WEIGHT["key_title"], color=BLACK, pad=6)
    return cbar


# Annotation

def annotate(ax, text: str, xy, xytext=None, arrow: bool = False,
             leader: bool = True, ha: str = "left", va: str = "center", **kwargs):
    """Callout label in the Annotation style, with an optional leader or arrow."""
    props = None
    if xytext is not None and (leader or arrow):
        props = {
            "arrowstyle": "-|>" if arrow else "-",
            "linewidth": LINE_WIDTH["arrow"] if arrow else LINE_WIDTH["leader"],
            "color": CHARCOAL,
            "shrinkA": 0,
            "shrinkB": 2,
        }
        if arrow:
            props["mutation_scale"] = 12
    return ax.annotate(
        text, xy=xy, xytext=xytext if xytext is not None else xy,
        textcoords=kwargs.pop("textcoords", "data"),
        fontsize=kwargs.pop("fontsize", FONT_SIZE["annotation"]),
        fontweight=kwargs.pop("fontweight", WEIGHT["annotation"]),
        color=kwargs.pop("color", CHARCOAL),
        ha=ha, va=va, arrowprops=kwargs.pop("arrowprops", props), **kwargs,
    )


# Saving

def save(fig, name, formats=("pdf", "png"), dpi: int = 300, directory=".",
         check_name: bool = True, **kwargs):
    """Save a figure at exactly its panel size, keeping text editable."""
    stem = Path(str(name)).stem
    if check_name and not _NAME_RE.match(stem):
        warnings.warn(
            f"{stem!r} does not follow the guide's figure naming convention "
            "FigX_Short-title_Size, where Size is Full_wide, Float, or "
            "Half_square (e.g. Fig2_Peak-shift_Half_square).",
            stacklevel=2,
        )
    fig = _root_figure(fig)
    titled = [ax.get_title() for ax in _all_axes(fig)
              if ax.get_title() and not getattr(ax, "_arcadia_key_axes", False)]
    if titled:
        warnings.warn(
            "Axes titles are set (" + "; ".join(titled[:3]) + "). The guide puts "
            "chart titles in the figure caption, not in the artwork -- consider "
            "moving this text to the caption.",
            stacklevel=2,
        )
    sync_keys(fig)

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    given_suffix = Path(str(name)).suffix.lstrip(".")
    for ext in ([given_suffix] if given_suffix else list(formats)):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches=None, **kwargs)
        written.append(path)
    return written
