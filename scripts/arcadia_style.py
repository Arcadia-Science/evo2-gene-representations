"""Arcadia style-guide glue for every figure in this repo."""

from __future__ import annotations

import arcadia_pycolor as apc
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "apc",
    "setup",
    "style_axes",
    "categorical",
    "resolve",
    "SEQUENTIAL",
    "DIVERGING",
    "HEATMAP_GEODESIC",
    "HEATMAP_JSD",
    "HEATMAP_SEQID",
    "HEATMAP_KMER",
    "MISSING",
    "PLACEHOLDER_FILL",
    "PLACEHOLDER_TEXT",
    "REFERENCE_LINE",
    "ZERO_LINE",
    "GRID",
    "ANNOTATION",
    "NULL_BAND",
    "WITHIN_PALETTE",
    "DOMAIN_COLORS",
    "CONTROL_COLORS",
    "STEER_COLORS",
]


# ── Setup

# Register the "apc:*" colormaps at IMPORT, not only in setup(). The constants below are
# colormap NAME strings, so a module-level `cmap=acs.SEQUENTIAL` in a script that forgot to
# call setup() would otherwise raise "not a valid value for cmap" at draw time. Registering
# names has no visual side effect on its own — the theme still only applies via setup().
apc.mpl.load_colormaps()

_SETUP_DONE = False


def setup(
    font_size: int = 8,
    title_size: int = 8,
    tick_size: int = 6,
    legend_size: int = 7,
) -> None:
    """Apply the Arcadia style, then this project's compact type scale."""
    import arcadia_pub as pub  # local import: arcadia_pub imports nothing from this module

    if pub.is_on():
        pub.setup()
        return

    global _SETUP_DONE
    if not _SETUP_DONE:
        apc.mpl.setup()
        _SETUP_DONE = True
    mpl.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": title_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "legend.title_fontsize": legend_size,
            # Arcadia zeroes the legend padding, which assumes the legend sits OUTSIDE the
            # axes. Most legends here are placed inside with loc="best", and at zero padding
            # matplotlib parks them flush in a corner, on top of the tick labels. Small pads
            # keep the frameless look and put the legend back inside the plotting area.
            "legend.borderaxespad": 0.7,
            "legend.borderpad": 0.3,
            "figure.titlesize": title_size + 2,
            # Arcadia's padding is sized for one standalone panel; multi-panel grids here
            # are laid out with tight_layout/constrained_layout instead.
            "axes.titlepad": 6,
            "axes.labelpad": 4,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Legible PNGs for the markdown results docs (see the module docstring).
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _is_prose(text: str) -> bool:
    """True if `text` is ordinary lowercase prose, safe to sentence-case."""
    stripped = text.strip()
    return bool(stripped) and all(c.isascii() and (c.isalpha() or c in " -/()") for c in stripped)


def style_axes(ax, monospaced_axes=None) -> None:
    """The style guide's per-Axes pass: sentence-cased axis labels, monospaced numeric ticks."""
    try:
        for get, put in ((ax.get_xlabel, ax.set_xlabel), (ax.get_ylabel, ax.set_ylabel)):
            text = get()
            if text.islower() and _is_prose(text):
                put(text[0].upper() + text[1:])
        if monospaced_axes in ("x", "both", "all"):
            for label in ax.get_xticklabels():
                label.set_fontfamily(apc.mpl.MONOSPACE_FONT)
        if monospaced_axes in ("y", "both", "all"):
            for label in ax.get_yticklabels():
                label.set_fontfamily(apc.mpl.MONOSPACE_FONT)
    except Exception:  # noqa: BLE001 - styling must never break a figure
        pass


# ── Continuous colour: matplotlib colormap names
# Arcadia's monocolour gradients run dark→light, so the "high value = saturated"
# convention every heatmap here uses needs the reversed variant.

SEQUENTIAL = "apc:magma"  # perceptually uniform, the Arcadia default
SEQUENTIAL_ALT = "apc:viridis"
DIVERGING = "apc:red_blue_r"  # low = blue, high = red (matplotlib RdBu_r's direction)
DIVERGING_WARM_COOL = "apc:orange_sage_r"  # low = sage/green, high = orange (RdYlGn_r)

# Per-baseline heatmap gradients: one hue per baseline, held constant across the Evo2 and
# GPN-Star figures so the two panels read as one comparison.
HEATMAP_GEODESIC = "apc:oranges_r"  # model geodesic  (was YlOrRd)
HEATMAP_JSD = "apc:blues_r"  # Pfam-domain JSD  (was Blues)
HEATMAP_SEQID = "apc:greens_r"  # sequence identity (was Greens)
HEATMAP_KMER = "apc:purples_r"  # k-mer divergence  (was Purples)
HEATMAP_TAXONOMY = "apc:sages_r"  # taxonomic-rank distance


# ── Semantic single colours

MISSING = apc.gray  # NaN / masked heatmap cells
PLACEHOLDER_FILL = apc.gray  # "TBD" panel background
PLACEHOLDER_TEXT = apc.bark
REFERENCE_LINE = apc.seaweed  # "natural / self = 1.0"-style reference
ZERO_LINE = apc.charcoal
GRID = apc.chateau
ANNOTATION = apc.charcoal  # text callouts on a plot
NULL_BAND = apc.chateau  # shaded permutation-null / CI bands
HIGHLIGHT = apc.dragon  # the one element a panel is about


# ── Semantic categorical groups

# The three standardized within-family baselines, identical in the Evo2 and GPN-Star
# within_correlations figures. Each bar colour is the mid-tone of the gradient that
# baseline's heatmap uses, so a baseline keeps one hue across the whole figure set.
WITHIN_PALETTE = {
    "kmer": apc.aster,  # HEATMAP_KMER (purples)
    "seqid": apc.seaweed,  # HEATMAP_SEQID (greens)
    "patristic": apc.amber,
    "taxonomy": apc.asparagus,  # HEATMAP_TAXONOMY (sages)
    "geodesic": apc.terracotta,  # HEATMAP_GEODESIC (oranges)
    "jsd": apc.aegean,  # HEATMAP_JSD (blues)
}

# Domain strip on the cross-kingdom within-family heatmaps.
DOMAIN_COLORS = {
    "Bacteria": apc.vital,
    "Archaea": apc.canary,
    "Eukaryota": apc.seaweed,
    "Unknown": apc.chateau,
}

# Composition / ablation controls. The nucleotide rungs walk blue_shades dark → light in
# ladder order (6-mer → 4-mer → codon → dinucleotide preserved), so the ramp itself carries
# the "progressively more composition destroyed" axis; the composition floor is neutral,
# `natural` is the intact reference (the same green as REFERENCE_LINE), and the two rungs
# that ablate something ORTHOGONAL to nucleotide composition — synonymous recode (protein
# preserved) and conservation ablation — are amber so they never read as part of the ramp.
CONTROL_COLORS = {
    "natural": apc.seaweed,
    # sequence-composition ladder (Evo2 / GPN CDS controls)
    "kmer6_shuffle": apc.dusk,
    "kmer4_shuffle": apc.lapis,
    "codon_shuffle": apc.aegean,
    "dinuc_shuffle": apc.vital,
    "mononuc_shuffle": apc.sky,
    "gc_match": apc.chateau,
    "gc_matched": apc.chateau,
    "gc_removed": apc.chateau,
    # The nested recode/missense pair, read together: amber = protein kept, dragon = protein
    # damaged. (Nested, NOT identity-matched — missense_subset edits a strict subset of the
    # recode's sites, so it is the weaker perturbation on both axes.)
    "synonymous_recode": apc.amber,
    "missense_subset": apc.dragon,
    # The matched p3 pair — same hues as the rung pair above so the "protein kept / protein
    # damaged" reading is consistent across figures, but the lighter tints mark them as a
    # SELF-CONTAINED pair scored against each other, not rungs of the composition ladder.
    "paired_p3_syn": apc.canary,
    "paired_p3_missense": apc.rose,
    # MSA-anchored ladder (GPN-Star multiz controls)
    "column_shuffle": apc.dusk,
    "ref_kmer6_column": apc.lapis,
    "ref_kmer4_column": apc.aegean,
    "ref_dinuc_column": apc.vital,
    "conservation_ablation": apc.amber,
    # structural ablations
    "cds_masked": apc.aster,
    "shuffled": apc.rose,
}
CONTROL_FALLBACK = apc.chateau  # an unlisted condition, rather than silent matplotlib blue

# Steering conditions. `unsteered` is the baseline the deltas are measured against;
# `source`/`target` are the two species the vector runs between; the rest are controls.
STEER_COLORS = {
    "unsteered": apc.charcoal,
    "steered": apc.aegean,
    "source": apc.amber,  # human, the sequence being steered
    "target": apc.seaweed,  # platypus / the species being steered toward
    "control": apc.amber,
    "random": apc.chateau,
    "null": apc.chateau,
    "shuffled": apc.rose,
    "cross": apc.canary,
    "self": apc.aster,
}

# Generic ordered series slots, for the many panels that plot "observed vs its null vs a
# third arm". Named by ROLE, not colour, so a panel that adds a fourth series does not
# have to invent a hue. SERIES_NULL is deliberately the warm one: a null curve sitting
# next to the observed curve must never read as a second observation.
SERIES_PRIMARY = apc.dusk  # the observed quantity the panel is about
SERIES_NULL = apc.amber  # its null / permutation / control arm
SERIES_THIRD = apc.aster  # a third arm (never placed adjacent to PRIMARY in a legend)
SERIES_FOURTH = apc.aegean
SERIES_MUTED = apc.chateau  # per-item background traces, reference lines, annotation


# ── Helpers


def resolve(name: str):
    """An Arcadia colour by name — ``resolve("aegean")`` → ``apc.aegean``."""
    try:
        return getattr(apc, name)
    except AttributeError as exc:  # a typo'd name should fail loudly, not plot grey
        raise KeyError(f"{name!r} is not an arcadia_pycolor named colour") from exc


def categorical(n: int) -> list:
    """`n` Arcadia colours for `n` unordered categories."""
    pool = list(apc.palettes.all_ordered)
    for shades in (
        apc.palettes.blue_shades,
        apc.palettes.red_shades,
        apc.palettes.green_shades,
        apc.palettes.purple_shades,
        apc.palettes.yellow_shades,
        apc.palettes.teal_shades,
    ):
        pool.extend(list(shades)[:4])
    if n <= len(pool):
        return pool[:n]
    return [pool[i % len(pool)] for i in range(n)]


def cmap_with_missing(name: str = SEQUENTIAL, missing=MISSING):
    """A copy of an ``apc:`` colormap that renders NaN/masked cells in `missing`."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=missing)
    return cmap


def line_safe(color, min_contrast: float = 2.6):
    """The same hue, darkened just enough to read as a thin line on a white page."""
    import colorsys

    from matplotlib.colors import to_rgb

    def contrast(rgb):
        chan = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        lum = 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
        return 1.05 / (lum + 0.05)

    rgb = to_rgb(color)
    if contrast(rgb) >= min_contrast:
        return color
    h, light, sat = colorsys.rgb_to_hls(*rgb)
    for _ in range(24):  # walk lightness down in small steps until it clears the floor
        light = max(0.0, light - 0.02)
        rgb = colorsys.hls_to_rgb(h, light, sat)
        if contrast(rgb) >= min_contrast or light == 0.0:
            break
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


def gradient_colors(name: str, n: int) -> list:
    """`n` evenly spaced colours sampled from an Arcadia gradient — for ordered series
    (a layer sweep, a dose ladder) that want a ramp rather than distinct hues."""
    key = name.removeprefix("apc:")
    reverse = key.endswith("_r")
    gradient = getattr(apc.gradients, key.removesuffix("_r"))
    if reverse:
        gradient = gradient.reverse()
    return list(gradient.resample_as_palette(n))
