"""Safe writes for the per-layer composition-control score tables."""

from __future__ import annotations
import datetime as _dt
import shutil
from pathlib import Path

import pandas as pd

KEY = "condition"
PROV_COLS = ("written_at", "generator")


def _today() -> str:
    # Date only: re-running the same stage twice in a day should not look like a provenance change.
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")


def write_control_table(
    path: str | Path,
    new: pd.DataFrame,
    *,
    generator: str,
    key: str = KEY,
    backup: bool = True,
    drop_conditions: list[str] | None = None,
) -> pd.DataFrame:
    """Upsert `new` into the control table at `path` and return the table as written."""
    path = Path(path)
    new = new.copy()
    if key not in new.columns:
        raise ValueError(f"{path.name}: new frame has no '{key}' column; refusing to write")
    new["written_at"] = _today()
    new["generator"] = generator

    if path.exists():
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        old = pd.read_csv(path)
        if key in old.columns:
            stale = set(new[key].unique()) | set(drop_conditions or [])
            kept = old[~old[key].isin(stale)]
            # Align schema variants before concatenation.
            for c in PROV_COLS:
                if c not in kept.columns:
                    kept = kept.assign(**{c: "unknown"})
            new = pd.concat([kept, new], ignore_index=True, sort=False)
        # else: unrecognised schema — fall through and overwrite (the .bak holds the original)

    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new
