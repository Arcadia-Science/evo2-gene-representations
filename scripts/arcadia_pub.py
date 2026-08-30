"""Publication geometry and type, on top of the Arcadia 2026 style guide."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib.ticker import FixedLocator

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # so `import arcadia_plots` works however we were invoked
    sys.path.insert(0, str(_HERE))

import arcadia_plots as arc  # noqa: E402

__all__ = [
    "arc",
    "is_on",
    "enable",
    "FULL",
    "HALF",
    "FLOAT",
    "size",
    "setup",
    "place",
    "enforce_type",
    "finish",
    "FIGURE_DIR",
]

# Publication panel widths in points. Half-width panels are stacked, not paired.
FULL = 1000.0
HALF = 500.0
FLOAT = arc.SIZES["float"]  # 650 — the guide's single-column size, unused here

FIGURE_DIR = _HERE.parent / "pub" / "figures"

_ON = os.environ.get("ARCADIA_PUB", "") == "1"


def is_on() -> bool:
    """True when this process is rendering publication figures rather than diagnostics."""
    return _ON


def enable(on: bool = True) -> None:
    """Turn pub mode on for this process. Call before the first `setup()`."""
    global _ON
    _ON = on
    os.environ["ARCADIA_PUB"] = "1" if on else ""


def size(width_pt: float, height_pt: float) -> tuple[float, float]:
    """A matplotlib ``figsize`` for an exact panel in points — ``size(FULL, 620)``."""
    return (width_pt / 72.0, height_pt / 72.0)


def setup() -> None:
    """Install the guide's style, unscaled, with tight-bbox saving off."""
    arc.apply(background="white")
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "savefig.bbox": None,  # never "tight" — see the docstring
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,  # live text in Illustrator
            "ps.fonttype": 42,
        }
    )


# ── Type

# Spell unsupported Greek glyphs and arrows instead of allowing font substitution.
GREEK_IN_LATIN = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "ν": "nu", "ξ": "xi", "ο": "omicron", "ρ": "rho",
    "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi",
    "ψ": "psi", "ω": "omega",
    "→": "to",
}


# Characters Atkinson also lacks that stand in for an ordinary one, so they are swapped
# straight across with no spacing: sub/superscript digits become plain digits, and "log₁₀"
# reads "log10". Handled separately from GREEK_IN_LATIN because that map inserts a space
# before the word it substitutes, which would turn "log₁₀" into "log 1 0".
DIRECT_SUBSTITUTES = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "‖": "||",
}


def spell_greek(text: str) -> str:
    """Rewrite Greek letters as their Latin names — "Spearman ρ" → "Spearman rho"."""
    for bad, good in DIRECT_SUBSTITUTES.items():
        text = text.replace(bad, good)
    out = []
    for ch in text:
        word = GREEK_IN_LATIN.get(ch)
        if word is None:
            out.append(ch)
            continue
        if out and out[-1] not in " ([{-–—/=<>":
            out.append(" ")
        out.append(word)
    return "".join(out)


def audit_glyphs(fig) -> list[str]:
    """Every character in `fig` that the Atkinson faces cannot draw."""
    import matplotlib.font_manager as fm
    from fontTools.ttLib import TTFont

    covered: set[int] = set()
    for family in (arc.NEXT, arc.MONO):
        path = fm.findfont(fm.FontProperties(family=family))
        for table in TTFont(path, fontNumber=0)["cmap"].tables:
            covered |= set(table.cmap.keys())

    bad: set[str] = set()
    for txt in _all_text(fig):
        for ch in txt.get_text():
            # Mathtext runs through matplotlib's own fonts, so skip anything in $...$.
            if ord(ch) > 127 and ord(ch) not in covered and "$" not in txt.get_text():
                bad.add(ch)
    return sorted(bad)


def _all_text(fig):
    """Every Text artist in a figure: axis labels, ticks, titles, keys, callouts."""
    for ax in fig.get_axes():
        yield from (ax.xaxis.label, ax.yaxis.label, ax.title)
        yield from ax.get_xticklabels()
        yield from ax.get_yticklabels()
        yield from ax.texts
        legend = ax.get_legend()
        if legend is not None:
            yield from legend.get_texts()
            if legend.get_title() is not None:
                yield legend.get_title()
    yield from fig.texts
    for legend in fig.legends:
        yield from legend.get_texts()
        if legend.get_title() is not None:
            yield legend.get_title()


def _sentence_case(text: str) -> str:
    """Capitalize the first word of a label, when that word is prose."""
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    if not body or not body[0].isascii() or not body[0].islower():
        return text
    first = body.split()[0]
    if "_" in first:  # a data identifier, not prose
        return text
    if len(first) > 1 and first[1] == "-":  # k-mer, p-value, n-gram
        return text
    if any(c.isdigit() for c in first):  # a label, not a word: s0, s4, p53, blocks27
        return text
    if any(c.isupper() for c in first):  # notation, and its case is meaningful: dN, dS, pH, mRNA
        return text
    return text[:lead] + body[0].upper() + body[1:]


def enforce_type(fig, body: float | None = None, number: float | None = None) -> None:
    """Rewrite every Text artist in `fig` to the guide's size, weight and family."""
    body = arc.FONT_SIZE["body"] if body is None else body
    number = arc.FONT_SIZE["number"] if number is None else number

    for txt in _all_text(fig):  # Greek → Latin before anything measures a string
        spelled = spell_greek(txt.get_text())
        if spelled != txt.get_text():
            txt.set_text(spelled)

    # Tick labels are REBUILT from the axis formatter on every draw, so rewriting the Text
    # artists above does not survive to the saved file. They have to be re-set through the
    # axis. Keyed on the LOCATOR being fixed, not on the formatter's type: `set_xticks(pos,
    # labels)` installs a FuncFormatter, not the FixedFormatter the name suggests, so a
    # formatter-type check silently matches nothing.
    fig.draw_without_rendering()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_locator(), FixedLocator):
                continue
            current = [t.get_text() for t in axis.get_ticklabels()]
            spelled = [spell_greek(t) for t in current]
            if spelled != current:
                axis.set_ticklabels(spelled)

    for ax in fig.get_axes():
        arc.mono_ticks(ax)  # mono numerals, comma thousands, real minus signs
        for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            lbl.set_fontsize(number)
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_fontsize(arc.FONT_SIZE["axis_title"])
            lbl.set_fontweight(arc.WEIGHT["axis_title"])
            lbl.set_color(arc.BLACK)
            lbl.set_text(_sentence_case(lbl.get_text()))
        title = ax.title
        if title.get_text():
            title.set_fontsize(arc.FONT_SIZE["key_title"])
            title.set_fontweight(arc.WEIGHT["key_title"])
            title.set_color(arc.BLACK)
            title.set_text(_sentence_case(title.get_text()))
        # Value labels, callouts and in-axes annotations.
        for txt in ax.texts:
            if txt.get_fontsize() < body:
                txt.set_fontsize(body)
        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontsize(body)
            if legend.get_title() is not None and legend.get_title().get_text():
                legend.get_title().set_fontsize(arc.FONT_SIZE["key_title"])
                legend.get_title().set_fontweight(arc.WEIGHT["key_title"])
        # Colorbars are Axes too; their tick labels are numbers and want Mono. `_colorbars` is
        # private and holds Colorbar objects on the mappable's parent Axes — but the colorbar's
        # OWN Axes appears in fig.get_axes() as well, where the entry is the Axes itself. Take
        # whichever it is rather than assuming.
        for artist in getattr(ax, "_colorbars", []):
            cax = getattr(artist, "ax", artist)
            for lbl in (*cax.get_xticklabels(), *cax.get_yticklabels()):
                lbl.set_fontfamily("monospace")
                lbl.set_fontsize(number)

    for txt in fig.texts:  # suptitles and figure-level annotations
        if txt.get_fontsize() < body:
            txt.set_fontsize(body)

    for legend in fig.legends:  # figure-level keys shared across panels
        for txt in legend.get_texts():
            txt.set_fontsize(body)
        if legend.get_title() is not None and legend.get_title().get_text():
            legend.get_title().set_fontsize(arc.FONT_SIZE["key_title"])
            legend.get_title().set_fontweight(arc.WEIGHT["key_title"])


def drop_titles(fig, keep_axes: bool = True) -> None:
    """Remove the figure-level title."""
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    if not keep_axes:
        for ax in fig.get_axes():
            ax.set_title("")


# ── Geometry


def key_below(fig, handles, labels, title: str, width: float = FULL,
              max_cols: int = 6, pad: float = 24.0, cols: int | None = None):
    """A shared key UNDER the panels, in as many columns as actually fit the panel width."""
    fig.canvas.draw()  # a renderer has to exist before anything can be measured
    renderer = fig.canvas.get_renderer()
    # Keys may use the full panel width.
    inner = width
    legend, box = None, None
    tries = [cols] if cols else list(range(max_cols, 0, -1))
    for ncol in tries:
        if legend is not None:
            legend.remove()
        legend = fig.legend(handles, labels, frameon=False, loc="lower left",
                            bbox_to_anchor=(0.0, 0.0), ncol=ncol, title=title,
                            handlelength=1.2, columnspacing=0.9, handletextpad=0.5,
                            borderaxespad=0.0)
        for text in legend.get_texts():
            text.set_fontsize(arc.FONT_SIZE["body"])
        legend.get_title().set_fontsize(arc.FONT_SIZE["key_title"])
        legend.get_title().set_fontweight(arc.WEIGHT["key_title"])
        legend._legend_box.align = "left"  # title over the first column, not centred
        fig.canvas.draw()
        box = legend.get_window_extent(renderer)
        if box.width * 72.0 / fig.dpi <= inner:
            break
    got = box.width * 72.0 / fig.dpi
    if got > inner:  # only reachable when `cols` was forced
        print(f"    WARNING: key at {ncol} columns is {got:.0f} pt wide, over the {inner:.0f} pt "
              f"panel — it will overhang. Drop a column or shorten the longest label.")
    # `tight` re-anchors these once the figure has its final height: the anchor is a FIGURE
    # FRACTION, so a key pinned at y=0 now would still be pinned to the very bottom edge after
    # the figure grows, with no margin under it.
    fig._arcadia_bottom_keys = [*getattr(fig, "_arcadia_bottom_keys", []), legend]
    return legend, box.height * 72.0 / fig.dpi + pad


def tight(fig, bottom: float = 0.0, margin: float = arc.MARGIN) -> None:
    """`tight_layout` inside the guide's 30 pt panel inset."""
    w, h = fig.get_size_inches() * 72.0
    for legend in getattr(fig, "_arcadia_bottom_keys", []):
        legend.set_bbox_to_anchor((margin / w, margin / h), transform=fig.transFigure)
    fig.tight_layout(rect=(margin / w, (bottom + margin) / h,
                           1.0 - margin / w, 1.0 - margin / h))


def place(fig, width: float, height: float, margin: float = arc.MARGIN,
          gap: float = arc.GAP, right: float | None = None, top: float | None = None) -> None:
    """Resize `fig` to an exact panel and lay its existing axes out on the guide's grid."""
    fig.set_size_inches(width / 72.0, height / 72.0)
    fig.subplots_adjust(
        left=margin / width,
        right=1.0 - (margin if right is None else right) / width,
        bottom=margin / height,
        top=1.0 - (margin if top is None else top) / height,
        wspace=gap / max(1.0, width / 4),
        hspace=gap / max(1.0, height / 8),
    )


def finish(fig, name: str, directory: Path | str | None = None,
           formats=("pdf", "png"), dpi: int = 300) -> None:
    """Type-check, then save at exactly the size the figure was given."""
    enforce_type(fig)
    # A key's Chateau rule is positioned in FIGURE coordinates against the rendered width of
    # the title, so it has to be re-placed after the final layout — `arcadia_plots.save` does
    # this and we do not go through it. Without this the rule floats away from its title.
    arc.sync_keys(fig)
    # Refresh formatted tick labels before auditing glyph coverage.
    fig.draw_without_rendering()
    missing = audit_glyphs(fig)
    if missing:
        raise SystemExit(
            f"{name}: Atkinson cannot draw {missing} — matplotlib would render a tofu box. "
            f"Add each to arcadia_pub.GREEK_IN_LATIN with a Latin spelling, or rewrite the "
            f"label that uses it."
        )
    out = (Path(directory) / "pub") if directory is not None else FIGURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches=None, pad_inches=0.0)
    w, h = fig.get_size_inches() * 72.0
    print(f"    pub: {name}  {w:.0f} x {h:.0f} pt")
