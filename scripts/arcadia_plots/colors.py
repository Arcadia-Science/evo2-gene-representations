"""Arcadia Science 2026 color system."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "COLORS", "PRIMARY", "SECONDARY", "NEUTRALS", "BACKGROUNDS", "SHADES",
    "PALETTES", "BICOLOR", "HIGHLIGHT", "GRADIENTS", "HEATMAP_GRADIENTS",
    "BLACK", "CHARCOAL", "CHATEAU", "WHITE",
    "color", "palette", "gradient", "gradient_colors", "cmap", "register_cmaps",
    "pycolor_available", "GRADIENT_STOPS", "DIVERGING",
]


# Named colors

PRIMARY = {
    "aegean": "#5088C5", "amber": "#F28360", "canary": "#F7B846",
    "aster": "#7A77AB", "seaweed": "#3B9886", "rose": "#F898AE",
    "vital": "#73B5E3", "tangerine": "#FFB883", "oat": "#F5E4BE",
    "wish": "#BABEE0", "lime": "#97CD78", "dragon": "#C85152",
}

SECONDARY = {
    "sky": "#C6E7F4", "dress": "#F8C5C1", "taupe": "#DBD1C3",
    "denim": "#B6C8D4", "sage": "#B5BEA4", "mars": "#DA9085",
    "marine": "#8A99AD", "shell": "#EDE0D6",
}

NEUTRALS = {
    "white": "#FFFFFF", "gray": "#EBEDE8", "chateau": "#B9AFA7",
    "bark": "#8F8885", "slate": "#43413F", "charcoal": "#484B50",
    "crow": "#292928", "black": "#09090A", "forest": "#596F74",
    # arcadia_pycolor calls the guide's Black "pitch" and reserves apc.black
    # for pure #000000. Both names resolve to the guide's value here.
    "pitch": "#09090A",
}

# Tints that are legal as figure/panel backgrounds.
BACKGROUNDS = {
    "parchment": "#FEF7F1", "zephyr": "#F4FBFE", "lichen": "#F7FBEF",
    "dawn": "#F8F4F1", "shell": "#EDE0D6", "marine": "#8A99AD",
    "white": "#FFFFFF", "gray": "#EBEDE8", "black": "#09090A",
    "forest": "#596F74",
}

# Monochromatic ramps, ordered light -> dark. Use these when a variable is
# ordinal (dose, timepoint, depth) but you still want one hue per group.
SHADES = {
    "blue": ["#C6E7F4", "#73B5E3", "#5088C5", "#2B66A2", "#094468"],
    "orange": ["#FFCFAF", "#FFB883", "#F28360", "#C85152", "#9E3F41"],
    "yellow": ["#F5E4BE", "#FFD364", "#F7B846", "#D68D22", "#A85E28"],
    "purple": ["#DCDFEF", "#BABEE0", "#7A77AB", "#54448C", "#341E60"],
    "teal": ["#C3E2DB", "#6FBCAD", "#3B9886", "#2A6B5E", "#09473E"],
    "pink": ["#FFE3D4", "#F8C5C1", "#F898AE", "#E2718F", "#C04C70"],
    "warm_gray": ["#EDE6DA", "#DBD1C3", "#B9AFA7", "#8F8885", "#635C5A"],
    "cool_gray": ["#E6EAED", "#CAD4DB", "#ABBAC4", "#8A99AD", "#687787"],
}

# Names for every shade above, plus the extra colors that only appear inside
# gradients. Lets you write color("tumbleweed") without hunting the guide.
_SHADE_NAMES = {
    "sky": "#C6E7F4", "vital": "#73B5E3", "aegean": "#5088C5",
    "lapis": "#2B66A2", "dusk": "#094468",
    "melon": "#FFCFAF", "tangerine": "#FFB883", "amber": "#F28360",
    "dragon": "#C85152", "cinnabar": "#9E3F41",
    "oat": "#F5E4BE", "sun": "#FFD364", "canary": "#F7B846",
    "mustard": "#D68D22", "umber": "#A85E28",
    "iris": "#DCDFEF", "wish": "#BABEE0", "aster": "#7A77AB",
    "tanzanite": "#54448C", "concord": "#341E60",
    "glass": "#C3E2DB", "teal": "#6FBCAD", "seaweed": "#3B9886",
    "asparagus": "#2A6B5E", "depths": "#09473E",
    "putty": "#FFE3D4", "dress": "#F8C5C1", "rose": "#F898AE",
    "candy": "#E2718F", "azalea": "#C04C70",
    "stone": "#EDE6DA", "taupe": "#DBD1C3", "chateau": "#B9AFA7",
    "mud": "#635C5A", "ice": "#E6EAED", "dove": "#CAD4DB",
    "cloud": "#ABBAC4", "steel": "#687787",
    # gradient-only colors
    "heather": "#A96789", "tumbleweed": "#E9A482", "wheat": "#F5DFB2",
    "shire": "#4E7F72", "topaz": "#FFCC7B", "space": "#282A49",
    "butter": "#FFFDBD", "redwood": "#52180A", "blossom": "#F4CAE3",
    "soil": "#4D2500", "terracotta": "#964222", "blush": "#FFF3F4",
    "ghost": "#FCF7FF", "fern": "#47784A", "lilac": "#6862AB",
}

COLORS: dict[str, str] = {}
for _group in (PRIMARY, SECONDARY, NEUTRALS, BACKGROUNDS, _SHADE_NAMES):
    COLORS.update(_group)

BLACK = COLORS["black"]
WHITE = COLORS["white"]
CHARCOAL = COLORS["charcoal"]
CHATEAU = COLORS["chateau"]
PARCHMENT = COLORS["parchment"]


def color(name: str) -> str:
    """Hex for an Arcadia color name ('aegean', 'chateau', ...)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in COLORS:
        raise KeyError(
            f"{name!r} is not an Arcadia color. Close matches: "
            + ", ".join(sorted(k for k in COLORS if k.startswith(key[:3])) or ["(none)"])
        )
    return COLORS[key]


def _hexes(*names: str) -> list[str]:
    return [COLORS[n] for n in names]


# Categorical palettes ("Suggested color palettes", p. 8)
# Keyed by number of series. The plain integer key is the guide's first-choice
# palette; "_alt" is the guide's second column (a legal alternative, useful
# when you need to stay distinct from a neighboring figure).

PALETTES: dict[str | int, list[str]] = {
    2: _hexes("aegean", "amber"),
    "2_alt": _hexes("canary", "aster"),
    "2_alt2": _hexes("seaweed", "rose"),
    3: _hexes("canary", "seaweed", "aster"),
    4: _hexes("aegean", "amber", "canary", "lime"),
    "4_alt": _hexes("aster", "amber", "canary", "vital"),
    5: _hexes("aegean", "amber", "canary", "lime", "rose"),
    "5_alt": _hexes("aster", "amber", "canary", "vital", "seaweed"),
    6: _hexes("dragon", "amber", "canary", "seaweed", "lime", "aegean"),
    "6_alt": _hexes("seaweed", "canary", "oat", "rose", "wish", "aster"),
    7: _hexes("dragon", "canary", "seaweed", "lime", "vital", "aster", "rose"),
    "7_alt": _hexes("mars", "oat", "sage", "shell", "sky", "wish", "dress"),
    8: _hexes("dragon", "canary", "seaweed", "lime", "vital", "aster", "rose", "marine"),
    "8_alt": _hexes("mars", "oat", "sage", "shell", "sky", "wish", "dress", "denim"),
}

# Two families that read as two groups at a glance -- for example treatment
# arms (blues) vs. controls (oranges).
BICOLOR = {
    "blue_orange": _hexes("aegean", "sky", "denim", "amber", "dress", "mars"),
    "yellow_purple": _hexes("canary", "oat", "taupe", "aster", "wish", "marine"),
}

# First entry is the emphasized series; the rest recede.
HIGHLIGHT = {
    "aegean_on_orange": _hexes("aegean", "melon", "tangerine", "amber", "dragon"),
    "amber_on_blue": _hexes("amber", "sky", "vital", "aegean", "lapis"),
    "amber_on_warm_gray": _hexes("amber", "stone", "taupe", "bark", "mud"),
    "canary_on_cool_gray": _hexes("canary", "ice", "dove", "marine", "steel"),
}


def palette(n: int | str = 6, variant: str = "") -> list[str]:
    """Suggested categorical palette for `n` series."""
    if isinstance(n, str):
        key = n.lower()
        if key in BICOLOR:
            return list(BICOLOR[key])
        if key in HIGHLIGHT:
            return list(HIGHLIGHT[key])
        if key == "bicolor":
            return list(BICOLOR["blue_orange"])
        if key == "highlight":
            return list(HIGHLIGHT["amber_on_blue"])
        if key in PALETTES:
            return list(PALETTES[key])
        raise KeyError(f"unknown palette {n!r}")
    if n == 1:
        return [COLORS["aegean"]]
    key = f"{n}_{variant}" if variant else n
    if key not in PALETTES:
        raise KeyError(
            f"No suggested palette for {n} series. Above 8 categories the guide's "
            "advice is to group or bin the data, or switch to a sequential gradient."
        )
    return list(PALETTES[key])


# Gradients ("Color gradients", p. 7)
# Preserve the guide's nonuniform anchor positions and dark-to-light direction.
# Reverse a gradient when low values should appear pale.

_LINE_GRADIENTS = {
    "magma": [(0, "concord"), (0.217, "tanzanite"), (0.498, "heather"),
              (0.799, "tumbleweed"), (1, "wheat")],
    "verde": [(0, "depths"), (0.357, "shire"), (0.909, "topaz"), (1, "putty")],
    "viridis": [(0, "space"), (0.468, "aegean"), (0.746, "lime"), (1, "butter")],
    "wine": [(0, "redwood"), (0.451, "dragon"), (0.828, "tangerine"), (1, "dawn")],
    "lisafrank": [(0, "depths"), (0.484, "aegean"), (0.862, "wish"), (1, "blossom")],
    "sunset": [(0, "soil"), (0.407, "umber"), (0.767, "tumbleweed"),
               (0.915, "topaz"), (1, "putty")],
}

_HEATMAP_SEQUENTIAL = {
    "oranges": [(0, "terracotta"), (0.761, "tangerine"), (1, "dawn")],
    "sages": [(0, "asparagus"), (0.641, "sage"), (1, "lichen")],
    "reds": [(0, "cinnabar"), (0.212, "dragon"), (1, "blush")],
    "blues": [(0, "lapis"), (0.254, "aegean"), (1, "zephyr")],
    "purples": [(0, "lilac"), (0.144, "aster"), (1, "ghost")],
    "greens": [(0, "fern"), (0.622, "lime"), (1, "lichen")],
}

# Diverging gradients -- only use these when zero (or some other value) is a
# meaningful midpoint, and center the norm on it. The doubled 0.5 anchor is a
# hard midpoint: the two near-white stops meet without blending through gray.
_HEATMAP_DIVERGING = {
    "orange_sage": [(0.0, "terracotta"), (0.3805, "tangerine"), (0.5, "dawn"),
                    (0.5, "lichen"), (0.6795, "sage"), (1.0, "asparagus")],
    "red_blue": [(0.0, "cinnabar"), (0.106, "dragon"), (0.5, "blush"),
                 (0.5, "zephyr"), (0.873, "aegean"), (1.0, "lapis")],
    "purple_green": [(0.0, "lilac"), (0.072, "aster"), (0.5, "ghost"),
                     (0.5, "lichen"), (0.689, "lime"), (1.0, "fern")],
}

GRADIENT_STOPS = {**_LINE_GRADIENTS, **_HEATMAP_SEQUENTIAL, **_HEATMAP_DIVERGING}
HEATMAP_GRADIENTS = {**_HEATMAP_SEQUENTIAL, **_HEATMAP_DIVERGING}
DIVERGING = set(_HEATMAP_DIVERGING)


def _build(name: str) -> LinearSegmentedColormap:
    stops = [(pos, COLORS[color]) for pos, color in GRADIENT_STOPS[name]]
    # A doubled position is a legal hard stop, but from_list needs strictly
    # increasing values, so split the pair by one part in a million.
    fixed, previous = [], -1.0
    for pos, hexcode in stops:
        if pos <= previous:
            pos = previous + 1e-6
        fixed.append((pos, hexcode))
        previous = pos
    return LinearSegmentedColormap.from_list(f"arcadia:{name}", fixed)


def pycolor_available() -> bool:
    """True if arcadia_pycolor, Arcadia's own color library, is importable."""
    try:
        import arcadia_pycolor  # noqa: F401
    except Exception:
        return False
    return True


_CACHE: dict[str, LinearSegmentedColormap] = {}


def _cmap_for(name: str) -> LinearSegmentedColormap:
    if name in _CACHE:
        return _CACHE[name]
    cm = None
    try:  # defer to arcadia_pycolor when the project has it installed
        import arcadia_pycolor as apc

        official = getattr(apc.gradients, name, None)
        if official is not None:
            cm = official.to_mpl_cmap()
    except Exception:
        cm = None
    _CACHE[name] = cm if cm is not None else _build(name)
    return _CACHE[name]


class _Gradients(dict):
    """Lazy mapping so gradients are only built when first asked for."""

    def __missing__(self, key):
        if key not in GRADIENT_STOPS:
            raise KeyError(key)
        value = _cmap_for(key)
        self[key] = value
        return value

    def keys(self):  # keep dict(**GRADIENTS) and iteration predictable
        return GRADIENT_STOPS.keys()

    def __iter__(self):
        return iter(GRADIENT_STOPS)

    def items(self):
        return ((k, self[k]) for k in GRADIENT_STOPS)

    def values(self):
        return (self[k] for k in GRADIENT_STOPS)

    def __len__(self):
        return len(GRADIENT_STOPS)

    def __contains__(self, key):
        return key in GRADIENT_STOPS


GRADIENTS: dict[str, LinearSegmentedColormap] = _Gradients()


def register_cmaps() -> None:
    """Register every gradient with matplotlib as 'arcadia:<name>'."""
    import matplotlib

    for name in GRADIENT_STOPS:
        full = f"arcadia:{name}"
        if full not in matplotlib.colormaps:
            cm = _cmap_for(name)
            matplotlib.colormaps.register(cm, name=full)
            matplotlib.colormaps.register(cm.reversed(), name=f"{full}_r")


def gradient(name: str) -> LinearSegmentedColormap:
    """Colormap for a gradient name; append '_r' to reverse."""
    key = name.lower().replace("arcadia:", "").replace("apc:", "").replace("-", "_")
    reverse = key.endswith("_r")
    if reverse:
        key = key[:-2]
    if key not in GRADIENT_STOPS:
        raise KeyError(
            f"{name!r} is not an Arcadia gradient. Options: {sorted(GRADIENT_STOPS)}"
        )
    cm = _cmap_for(key)
    return cm.reversed() if reverse else cm


cmap = gradient  # alias


def gradient_colors(name: str, n: int, lo: float = 0.0, hi: float = 1.0) -> list[str]:
    """`n` discrete hexes sampled from a gradient -- for ordered line series."""
    import numpy as np
    from matplotlib.colors import to_hex

    cm = gradient(name)
    if n == 1:
        return [to_hex(cm(0.5))]
    return [to_hex(cm(v)) for v in np.linspace(lo, hi, n)]
