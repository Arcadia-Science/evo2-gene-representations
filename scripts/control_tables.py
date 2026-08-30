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


def audit_control_table(path: str | Path) -> dict:
    """
    Provenance summary for one table: {n_rows, conditions, written_at:[...], generator:[...], mixed:
    bool}.
    """
    path = Path(path)
    df = pd.read_csv(path)
    dates = sorted(df["written_at"].dropna().unique()) if "written_at" in df else []
    gens = sorted(df["generator"].dropna().unique()) if "generator" in df else []
    return {
        "path": str(path),
        "n_rows": len(df),
        "conditions": sorted(df[KEY].dropna().unique()) if KEY in df else [],
        "written_at": dates,
        "generator": gens,
        "mixed": len(dates) > 1 or len(gens) > 1,
    }


def snapshot_control_tables(
    label: str, runs: list[str], *, reason: str, root: str | Path = "results"
) -> Path:
    """
    Copy every per-layer control table from `runs` into results/_control_scores_<label>/ and write a
    README recording `reason`. Returns the snapshot dir.
    """
    root = Path(root)
    snap = root / f"_control_scores_{label}"
    snap.mkdir(parents=True, exist_ok=True)
    n = 0
    for run in runs:
        run_path = Path(run)
        for f in sorted(run_path.glob("blocks*/controls/control_*_scores.csv")):
            layer = f.parent.parent.name
            dest = snap / run_path.name / f"{layer}_{f.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(f, dest)
                n += 1
    (snap / "README.md").write_text(
        f"# Control score snapshot: {label}\n\n"
        f"Taken {_today()} (UTC). {n} table(s) copied from {len(runs)} run dir(s).\n\n"
        f"## Reason\n\n{reason}\n\n"
        f"## Layout\n\n`<run>/blocks<L>_control_{{within,between}}_scores.csv` contains copied\n"
        f"tables as they stood at snapshot time.\n\n"
        f"Snapshots use labels so the reason stays attached to the data;\n"
        f"see scripts/control_tables.py for the rationale.\n"
    )
    return snap
