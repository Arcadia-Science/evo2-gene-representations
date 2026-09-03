"""Rename the `autapomorphy` score columns to `private_bp` in stored tables. CPU only.

"Autapomorphy" is a cladistic term for a derived character unique to one terminal taxon on a
phylogeny. The metric only asks whether a base is absent from the mammals that happened to be
sampled, which is a weaker claim, so the name was dropped from the code and the figures. This
brings existing tables into line without re-running the ~40 h stage-4 rescore.

    pct_autapomorphy_correct -> pct_private_bp_correct
    n_autapomorphy_scorable  -> n_private_bp_scorable

`pct_private_correct` is deliberately NOT touched. Despite the name it is the deprecated
pre-2026-08-18 pairwise metric scored through the protein alignment, whose own control read 9.3%
in the most-diverged stratum where it must be 0. Renaming the strict metric onto that name would
make the two indistinguishable in every stored table, which is exactly the confusion
`strat_metric.py` warns about.

Header-only edit: the values are not touched, and the script verifies that by comparing the data
block before and after. Idempotent -- a table already migrated, or one that never carried the
columns, is skipped.

    uv run python scripts/steering/platypus/strat/migrate_private_bp_column_names.py
    uv run python scripts/steering/platypus/strat/migrate_private_bp_column_names.py --apply
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RENAMES = {
    "pct_autapomorphy_correct": "pct_private_bp_correct",
    "n_autapomorphy_scorable": "n_private_bp_scorable",
}
# Never rename this one; see the module docstring.
PROTECTED = "pct_private_correct"


def _rename_for(col: str) -> str | None:
    """The new name for one column, or None if it is not affected.

    Substring, not equality: the summary and top-gene tables compose derived columns off the base
    name (`unsteered_pct_autapomorphy_correct`, `pct_autapomorphy_correct_delta`), and those are
    the same rename. PROTECTED is untouched by construction -- it does not contain either old name.
    """
    for old, new in RENAMES.items():
        if old in col:
            return col.replace(old, new)
    return None


def migrate(path: Path, apply: bool) -> str:
    """Return the outcome for one file: 'renamed', 'already', or 'no-op'."""
    df = pd.read_csv(path)
    hits = {c: n for c in df.columns if (n := _rename_for(c)) is not None}
    if not hits:
        done = any(n in c for c in df.columns for n in RENAMES.values())
        return "already" if done else "no-op"
    # A table holding both the old and the new name is ambiguous -- refuse rather than pick one.
    collide = [new for new in hits.values() if new in df.columns]
    if collide:
        raise ValueError(f"already has {collide} alongside the old name")
    renamed = df.rename(columns=hits)
    # The values must be identical -- this is a header edit, nothing else. Compare with .equals,
    # which treats NaN in the same position as equal; an elementwise == would fail on every table
    # holding an unscorable gene, which is most of them.
    back = renamed.rename(columns={new: old for old, new in hits.items()})
    assert back.equals(df), f"{path}: values changed"
    assert (PROTECTED in df.columns) == (PROTECTED in renamed.columns), path
    if apply:
        renamed.to_csv(path, index=False)
    return "renamed"


def candidates(root: Path) -> list[Path]:
    """CSVs whose header names a column we rename. Reads one line per file, not the table."""
    out = []
    for f in sorted(root.rglob("*.csv")):
        try:
            with f.open() as fh:
                header = fh.readline()
        except OSError:
            continue
        if any(c in header for c in RENAMES):
            out.append(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        nargs="*",
        default=[ROOT / "results", ROOT / "figure_data"],
        help="directories to scan (default: results/ and figure_data/)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="without this, only report what would change"
    )
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.root:
        if not root.exists():
            print(f"  skip {root}: not present")
            continue
        files.extend(candidates(root))

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

    print(f"scanned {len(files)} CSVs carrying an old column name")
    for k in ("renamed", "already", "no-op", "error"):
        if k in counts:
            print(f"  {k:8s} {counts[k]}")
    for f in changed:
        print(f"    {f.relative_to(ROOT)}")
    if not args.apply and changed:
        print("\nDRY RUN -- re-run with --apply to write.")
        sys.exit(0)


if __name__ == "__main__":
    main()
