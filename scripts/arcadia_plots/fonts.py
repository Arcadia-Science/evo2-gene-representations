"""Make the Atkinson Hyperlegible family usable from matplotlib."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

NEXT = "Atkinson Hyperlegible Next"
MONO = "Atkinson Hyperlegible Mono"

# (target wght, name-table subfamily). The subfamily strings are chosen from
# matplotlib.font_manager.weight_dict so that matplotlib infers the right
# numeric weight: light->200, regular->400, medium->500, semibold->600.
# ExtraLight in the guide == wght 200 == matplotlib "light".
_UPRIGHT = [(200, "Light"), (400, "Regular"), (500, "Medium"), (600, "SemiBold")]
_ITALIC = [(400, "Italic"), (500, "Medium Italic")]

_XDG_CACHE = os.environ.get("XDG_CACHE_HOME")
CACHE_DIR = Path(
    os.environ.get(
        "ARCADIA_FONT_CACHE",
        (Path(_XDG_CACHE) if _XDG_CACHE else Path.home() / ".cache") / "arcadia_style/fonts",
    )
)
# Downloaded variable fonts live beside the pinned static instances, so a
# headless machine needs nothing installed system-wide.
DOWNLOAD_DIR = CACHE_DIR / "variable"

_SEARCH_DIRS = [
    Path(os.environ.get("ARCADIA_FONT_DIR", "")),
    DOWNLOAD_DIR,
    Path.home() / "Library/Fonts",
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("C:/Windows/Fonts"),
]

_state: dict[str, object] = {"ready": False, "available": False}

INSTALL_HINT = (
    "Atkinson Hyperlegible Next + Mono were not found, so figures will render "
    "with substitute type. Fix it either way:\n"
    "  * this machine only:  install the fonts from "
    "https://www.brailleinstitute.org/freefont/ or Google Fonts (double-click "
    "on macOS, or drop them in ~/.fonts on Linux)\n"
    "  * anywhere, no install (servers, CI, containers):  run "
    "`python check_install.py --fetch`, or arcadia_style.fetch_fonts(), which "
    "downloads the two variable fonts from Google Fonts into this user's cache\n"
    "Set ARCADIA_FONT_DIR if you keep them somewhere else."
)

# Google Fonts is the upstream for both families; the OFL license travels with
# them. Italics are optional -- the guide never asks for them, but species names
# do, so grab them when upstream has them.
_GF = "https://github.com/google/fonts/raw/main/ofl"
_DOWNLOADS = {
    "next": (f"{_GF}/atkinsonhyperlegiblenext/AtkinsonHyperlegibleNext%5Bwght%5D.ttf",
             "AtkinsonHyperlegibleNext-VariableFont_wght.ttf", True),
    "next_italic": (f"{_GF}/atkinsonhyperlegiblenext/AtkinsonHyperlegibleNext-Italic%5Bwght%5D.ttf",
                    "AtkinsonHyperlegibleNext-Italic-VariableFont_wght.ttf", False),
    "mono": (f"{_GF}/atkinsonhyperlegiblemono/AtkinsonHyperlegibleMono%5Bwght%5D.ttf",
             "AtkinsonHyperlegibleMono-VariableFont_wght.ttf", True),
    "mono_italic": (f"{_GF}/atkinsonhyperlegiblemono/AtkinsonHyperlegibleMono-Italic%5Bwght%5D.ttf",
                    "AtkinsonHyperlegibleMono-Italic-VariableFont_wght.ttf", False),
}
_LICENSE_URL = f"{_GF}/atkinsonhyperlegiblenext/OFL.txt"


def fetch(force: bool = False, quiet: bool = False) -> Path:
    """Download the Atkinson variable fonts into the cache. Returns that directory."""
    import urllib.request

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    got = 0
    for key, (url, filename, required) in _DOWNLOADS.items():
        target = DOWNLOAD_DIR / filename
        if target.exists() and not force:
            got += 1
            continue
        try:
            if not quiet:
                print(f"downloading {filename} ...")
            urllib.request.urlretrieve(url, target)
            got += 1
        except Exception as exc:
            target.unlink(missing_ok=True)
            if required:
                raise RuntimeError(
                    f"Could not download {filename} from Google Fonts ({exc}). "
                    "On a machine without network access, copy the two variable "
                    f"TTFs into {DOWNLOAD_DIR} by hand, or point ARCADIA_FONT_DIR "
                    "at wherever they already live."
                ) from exc
            if not quiet:
                print(f"  skipped optional {filename}: {exc}")
    license_path = DOWNLOAD_DIR / "OFL.txt"
    if not license_path.exists():
        try:
            urllib.request.urlretrieve(_LICENSE_URL, license_path)
        except Exception:
            pass
    if not quiet:
        print(f"{got} variable fonts in {DOWNLOAD_DIR}")
    ensure_fonts(force=True)
    return DOWNLOAD_DIR


def _find_variable_fonts() -> dict[str, Path]:
    """Locate the four variable TTFs by filename."""
    wanted = {
        "next": "atkinsonhyperlegiblenext-variablefont",
        "next_italic": "atkinsonhyperlegiblenext-italic-variablefont",
        "mono": "atkinsonhyperlegiblemono-variablefont",
        "mono_italic": "atkinsonhyperlegiblemono-italic-variablefont",
    }
    found: dict[str, Path] = {}
    for directory in _SEARCH_DIRS:
        if not directory or not directory.is_dir():
            continue
        for path in directory.glob("Atkinson*"):
            stem = path.name.lower()
            for key, needle in wanted.items():
                if key in found:
                    continue
                # the italic keys are substrings of nothing else, but the
                # upright needle also matches the italic filename, so require
                # that "-italic" is absent for upright keys
                if needle in stem and (("italic" in key) == ("italic" in stem)):
                    found[key] = path
    return found


def _instantiate(src: Path, weight: int, subfamily: str, family: str, out: Path) -> None:
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(str(src))
    instancer.instantiateVariableFont(font, {"wght": weight}, inplace=True)

    full = f"{family} {subfamily}"
    postscript = full.replace(" ", "")
    name = font["name"]
    for nid, value in ((1, family), (2, subfamily), (4, full), (6, postscript)):
        name.setName(value, nid, 3, 1, 0x409)  # Windows/Unicode
        name.setName(value, nid, 1, 0, 0)      # Mac/Roman
    # Drop the typographic family/subfamily records: matplotlib reads ids 1/2,
    # and leftover id 16/17 records confuse other font tools about the weight.
    for nid in (16, 17, 21, 22):
        name.removeNames(nameID=nid)
    font["OS/2"].usWeightClass = weight

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    font.save(str(tmp))
    tmp.replace(out)
    font.close()


def ensure_fonts(force: bool = False) -> bool:
    """Build (if needed) and register the static instances. Returns availability."""
    if _state["ready"] and not force:
        return bool(_state["available"])
    _state["ready"] = True
    _state["available"] = False

    sources = _find_variable_fonts()
    if "next" not in sources or "mono" not in sources:
        warnings.warn(INSTALL_HINT, stacklevel=2)
        return False

    try:
        import fontTools  # noqa: F401
    except ImportError:
        warnings.warn(
            "fontTools is not installed, so only the default weight of Atkinson "
            "Hyperlegible can be used (Medium/SemiBold/ExtraLight will look like "
            "Regular). Fix with: pip install fonttools",
            stacklevel=2,
        )
        _register_raw(sources)
        _state["available"] = True
        return True

    jobs = []
    for key, family, variants in (
        ("next", NEXT, _UPRIGHT),
        ("next_italic", NEXT, _ITALIC),
        ("mono", MONO, [(400, "Regular"), (600, "SemiBold")]),
        ("mono_italic", MONO, [(400, "Italic")]),
    ):
        src = sources.get(key)
        if src is None:
            continue
        for weight, subfamily in variants:
            slug = f"{family.replace(' ', '')}-{subfamily.replace(' ', '')}-{weight}.ttf"
            jobs.append((src, weight, subfamily, family, CACHE_DIR / slug))

    from matplotlib import font_manager as fm

    for src, weight, subfamily, family, out in jobs:
        if force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            try:
                _instantiate(src, weight, subfamily, family, out)
            except Exception as exc:  # pragma: no cover - depends on font internals
                warnings.warn(f"Could not pin {family} {subfamily}: {exc}", stacklevel=2)
                continue
        try:
            fm.fontManager.addfont(str(out))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Could not register {out.name}: {exc}", stacklevel=2)

    _state["available"] = True
    return True


def _register_raw(sources: dict[str, Path]) -> None:
    from matplotlib import font_manager as fm

    for path in sources.values():
        try:
            fm.fontManager.addfont(str(path))
        except Exception:  # pragma: no cover
            pass


def available() -> bool:
    """True if the Atkinson family is registered with matplotlib."""
    ensure_fonts()
    return bool(_state["available"])


def report() -> str:
    """Human-readable summary of which weight resolves to which file."""
    from matplotlib.font_manager import FontProperties, findfont

    ensure_fonts()
    lines = []
    for family, weight, label in [
        (NEXT, 200, "ExtraLight (annotation)"),
        (NEXT, 400, "Regular (body, panel letter)"),
        (NEXT, 500, "Medium (axis titles)"),
        (NEXT, 600, "SemiBold (key titles)"),
        (MONO, 400, "Mono Regular (numbers)"),
    ]:
        path = findfont(FontProperties(family=family, weight=weight), fallback_to_default=True)
        lines.append(f"{family} {weight:>3} {label:<28} -> {Path(path).name}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if "--fetch" in sys.argv:
        fetch(force="--force" in sys.argv)
    ok = ensure_fonts(force="--force" in sys.argv)
    print("Atkinson fonts available:", ok)
    print(report())
