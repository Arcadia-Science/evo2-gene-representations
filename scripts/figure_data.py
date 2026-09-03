"""Locate and load the tidy tables the figures read.

Every figure generator reads through here, so the tracked tables are the single source the
published panels are drawn from. `scripts/build_figure_data.py` writes them from a completed run.

Two directories hold them, and a name is looked up in both: `figure_data/` for the publication
experiments, and `analyses/controls/figure_data/` for the composition-controls analysis, whose
tables live beside its own runner rather than in the publication set.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "figure_data"
CONTROLS_DIR = ROOT / "analyses" / "controls" / "figure_data"
DIRS = (DIR, CONTROLS_DIR)


def path(name: str) -> Path:
    """Absolute path of a table, with a pointed error if the build step has not been run."""
    filename = name if "." in name else f"{name}.csv"
    for directory in DIRS:
        if (p := directory / filename).exists():
            return p
    searched = "\n    ".join(str(d / filename) for d in DIRS)
    raise SystemExit(
        f"missing {filename}, looked in:\n    {searched}\n"
        f"  build it with: uv run python scripts/build_figure_data.py"
    )


def table(name: str, **kwargs) -> pd.DataFrame:
    """Read one tidy table by name (no extension needed)."""
    return pd.read_csv(path(name), **kwargs)
