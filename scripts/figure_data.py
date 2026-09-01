"""Locate and load the tidy publication tables in `figure_data/`.

Every figure generator reads through here, so the tracked tables are the single source the
published panels are drawn from. `scripts/build_figure_data.py` writes them from a completed run.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "figure_data"


def path(name: str) -> Path:
    """Absolute path of a table, with a pointed error if the build step has not been run."""
    p = DIR / (name if "." in name else f"{name}.csv")
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n  build it with: uv run python scripts/build_figure_data.py"
        )
    return p


def table(name: str, **kwargs) -> pd.DataFrame:
    """Read one tidy table by name (no extension needed)."""
    return pd.read_csv(path(name), **kwargs)
