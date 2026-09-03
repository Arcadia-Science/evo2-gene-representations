"""Rename `spearman_geodesic_*` to `spearman_angular_*` inside *_angular.csv tables. CPU only.

The angular within-family tables were written with the geodesic column name, so anyone who opened
one concluded the wrong metric had been scored (the filename was the only honest marker). The
scorer now names the metric in both places; this brings existing run directories into line without
re-running the ~3 h scoring pass.

Header-only edit: the values are not touched, and the script verifies that by comparing the data
block before and after. Idempotent -- a table that is already migrated, or that legitimately holds
geodesic results (no `_angular` in its name), is skipped.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD, NEW = "spearman_geodesic_", "spearman_angular_"


def migrate(path: Path, apply: bool) -> str:
    """Return the outcome for one file: 'renamed', 'already', or 'no-op'."""
    df = pd.read_csv(path)
    hits = [c for c in df.columns if c.startswith(OLD)]
    if not hits:
        return "already" if any(c.startswith(NEW) for c in df.columns) else "no-op"
    renamed = df.rename(columns={c: c.replace(OLD, NEW) for c in hits})
    # The values must be identical -- this is a header edit, nothing else. Compare with .equals,
    # which treats NaN in the same position as equal; an elementwise == would fail on every table
    # holding a family with too few members to score, which is most of them.
    back = renamed.rename(columns={c.replace(OLD, NEW): c for c in hits})
    assert back.equals(df), f"{path}: values changed"
    if apply:
        renamed.to_csv(path, index=False)
    return "renamed"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT / "results")
    ap.add_argument(
        "--apply", action="store_true", help="without this, only report what would change"
    )
    args = ap.parse_args()

    files = sorted(args.root.rglob("*_angular.csv"))
    counts: dict[str, int] = {}
    changed: list[Path] = []
    for f in files:
        try:
            outcome = migrate(f, args.apply)
        except Exception as exc:  # a malformed table should name itself, not abort the sweep
            print(f"  SKIP {f}: {exc}")
            counts["error"] = counts.get("error", 0) + 1
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome == "renamed":
            changed.append(f)

    print(f"scanned {len(files)} *_angular.csv under {args.root}")
    for k in ("renamed", "already", "no-op", "error"):
        if k in counts:
            print(f"  {k:8s} {counts[k]}")
    if changed:
        by_name: dict[str, int] = {}
        for f in changed:
            by_name[f.name] = by_name.get(f.name, 0) + 1
        for name, n in sorted(by_name.items()):
            print(f"    {n:>4}  {name}")
    if not args.apply and changed:
        print("\nDRY RUN -- re-run with --apply to write.")
        sys.exit(0)


if __name__ == "__main__":
    main()
