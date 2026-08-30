"""Copy the per-layer composition-control score tables into their layer-sweep summary folder."""

from __future__ import annotations
import argparse
import datetime as dt
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control_tables import audit_control_table  # noqa: E402

SUMMARY = Path("results/layer_sweep_summaries")
GLOB_RE = re.compile(r"^- glob: `([^`]+)`", re.M)
LAYER_RE = re.compile(r"(?:blocks|L)(\d+)\b")
TABLES = ("control_within_scores.csv", "control_between_scores.csv")
FIGURES = ("control_preservation.png", "control_preservation.pdf")
COPIED = TABLES + FIGURES

# `ot-*/` roll-ups are built from the SAME run dirs as a main folder, so copying controls into them
# would duplicate bytes and give the same numbers two homes. Value = the folder that does carry
# them.
OT_COVERED_BY = {
    "ot-evo2-human-cds": "evo2-human-cds-9fam",
    "ot-evo2-human-cds-48fam": "evo2-human-cds-48fam",
    "ot-evo2-human-cdsmask": "evo2-human-cdspool-transcript-48fam",
    "ot-evo2-human-transcript": "evo2-human-9fam",
    "ot-evo2-mammal-cds": "mammalian-orthologs-cds-48fam",
    "ot-evo2-mammal-cdsmask": "mammalian-orthologs-cdsmask-48fam",
    "ot-evo2-mammal-transcript": "mammalian-orthologs-9fam",
}


def _layer(p: Path) -> int:
    m = LAYER_RE.search(p.name)
    return int(m.group(1)) if m else -1


def _source_glob(folder: Path) -> str | None:
    src = folder / "SOURCE.md"
    if not src.exists():
        return None
    m = GLOB_RE.search(src.read_text())
    return m.group(1) if m else None


def _manifest(folder: Path, glob: str, copied: list[tuple[int, str, Path]]) -> str:
    """MANIFEST body: what was copied, from where, and the provenance audit of every copy."""
    by_file: dict[str, list[int]] = defaultdict(list)
    conds: dict[str, set[int]] = defaultdict(set)
    dates: set[str] = set()
    gens: set[str] = set()
    mixed: list[str] = []
    for layer, name, dest in copied:
        by_file[name].append(layer)
        if name not in TABLES:
            continue
        a = audit_control_table(dest)
        for c in a["conditions"]:
            conds[c].add(layer)
        dates |= set(a["written_at"]) or {"(no written_at column)"}
        gens |= set(a["generator"]) or {"(no generator column)"}
        if a["mixed"]:
            mixed.append(f"{dest.name}: written_at={a['written_at']} generator={a['generator']}")

    layers_all = sorted({layer for layer, _, _ in copied})
    n_layers = len(layers_all)
    lines = [
        "# Per-layer composition-control figures and the score tables they were drawn from",
        "# (verbatim copies).",
        "# Copied by scripts/controls/copy_controls_to_summaries.py from the run dir named in",
        "# SOURCE.md. The per-layer dirs are regenerated in place and results/ is gitignored, so",
        "# this is the only copy that stays put beside the roll-up it backs.",
        f"# Cross-layer view: ../controls_layer_summary_{folder.name}.png.",
        f"# Copied {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d')} (UTC).",
        "",
        f"source glob: {glob}",
        f"layers: {n_layers} ({layers_all[0]}..{layers_all[-1]})",
        f"files copied: {len(copied)}",
        "",
        "# file  layers_present",
    ]
    for name in COPIED:
        got = sorted(by_file.get(name, []))
        if not got:
            why = (
                " — this arm's scorer never wrote per-layer figures; regenerable from the tables,"
                " see the module docstring"
                if name in FIGURES
                else ""
            )
            lines.append(f"{name}\t0 — ABSENT in every source layer dir{why}")
        else:
            miss = sorted(set(layers_all) - set(got))
            lines.append(f"blocks<L>_{name}\t{len(got)}" + (f"  MISSING at {miss}" if miss else ""))

    lines += ["", "# condition  n_layers_present  (of the tables that exist)"]
    for c in sorted(conds):
        got = conds[c]
        miss = sorted(set(layers_all) - got)
        lines.append(f"{c}\t{len(got)}" + (f"  MISSING at {miss}" if miss else ""))

    lines += [
        "",
        "# provenance (audit_control_table over the copies; see scripts/control_tables.py)",
        f"written_at: {sorted(dates)}",
        f"generator:  {sorted(gens)}",
    ]
    if mixed:
        lines += [
            "",
            "# MIXED PROVENANCE WITHIN A TABLE — these tables each combine more than one methods",
            "# state, so a single table must not be read as one result:",
        ]
        lines += [f"#   {m}" for m in mixed]
    # audit_control_table's `mixed` is per-file, so it cannot see provenance that varies BETWEEN
    # layers: 32 tables each internally consistent but stamped on two dates still means the sweep
    # straddled a boundary. Report that spread explicitly, per layer, or it stays invisible.
    if len(dates) > 1 or len(gens) > 1:
        spread: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for layer, _, dest in copied:
            a = audit_control_table(dest)
            spread[(tuple(a["written_at"]), tuple(a["generator"]))].append(layer)
        lines += [
            "",
            "# PROVENANCE VARIES BY LAYER — each table is internally consistent, but they were not",
            "# all written in one state; check the boundary before comparing layers to each other:",
        ]
        for (d, g), ls in sorted(spread.items(), key=lambda kv: min(kv[1])):
            lines.append(f"#   written_at={list(d)} generator={list(g)}  layers={sorted(set(ls))}")
    return "\n".join(lines) + "\n"


def copy_folder(folder: Path, *, dry_run: bool = False) -> tuple[str, str]:
    """Copy one folder's control tables. Returns (status, detail) for the run report."""
    glob = _source_glob(folder)
    if glob is None:
        return "skip", "no `- glob:` in SOURCE.md (frozen or hand-written folder)"

    src_dirs = sorted((p for p in Path().glob(glob) if p.is_dir()), key=_layer)
    if not src_dirs:
        return "skip", f"no run dirs matched {glob!r}"

    copied: list[tuple[int, str, Path]] = []
    dest_dir = folder / "controls"
    for d in src_dirs:
        for name in COPIED:
            src = d / "controls" / name
            if not src.exists():
                continue
            dest = dest_dir / f"{d.name}_{name}"
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)  # copy2 keeps the source mtime, so provenance survives
            copied.append((_layer(d), name, dest if not dry_run else src))

    if not copied:
        return "skip", f"{len(src_dirs)} layer dirs, none with a controls/ file"

    n_layers = len({layer for layer, _, _ in copied})
    n_fig = sum(1 for _, name, _ in copied if name in FIGURES)
    n_tab = len(copied) - n_fig
    if not dry_run:
        (dest_dir / "MANIFEST.txt").write_text(_manifest(folder, glob, copied))
    figs = f"{n_fig} figures" if n_fig else "NO figures at source"
    return "ok", f"{figs} + {n_tab} tables, {n_layers} layers -> {dest_dir}/"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*", help="summary folder names (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be copied")
    args = ap.parse_args()

    if not SUMMARY.is_dir():
        sys.exit(f"{SUMMARY} not found — run from the repo root")
    wanted = args.folders or sorted(p.name for p in SUMMARY.iterdir() if p.is_dir())

    for name in wanted:
        folder = SUMMARY / name
        if not folder.is_dir():
            print(f"[skip] {name}: no such folder")
            continue
        if name in OT_COVERED_BY and not args.folders:
            print(
                f"[skip] {name}: same run dirs as {OT_COVERED_BY[name]}/, which carries the copies"
            )
            continue
        status, detail = copy_folder(folder, dry_run=args.dry_run)
        tag = ("[dry]" if args.dry_run else "[ok]") if status == "ok" else "[skip]"
        print(f"{tag} {name}: {detail}")


if __name__ == "__main__":
    main()
