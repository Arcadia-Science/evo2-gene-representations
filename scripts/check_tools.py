"""External (non-PyPI) tools a full `--run` needs, and whether this machine has them.

    uv run python scripts/check_tools.py             # every tool
    uv run python scripts/check_tools.py --exp exp2  # only what experiment 2 needs

`--figures` needs none of these; it reads `figure_data/` and renders. They are only reached by
`--run`, which is why a missing one used to surface hours in, as a bare FileNotFoundError from a
subprocess whose stderr was discarded, or as a remedy naming a build tree that is not in a clone
(`data/tools/` is git-ignored, so nothing under it survives cloning).

One registry so the runners can pre-flight, the stages can fail with the same wording, and
REPRODUCING.md has something to list.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "data" / "tools"


@dataclass(frozen=True)
class Tool:
    name: str
    experiments: tuple[str, ...]
    stage: str  # human-readable "where it is used"
    install: str
    used_version: str  # the build this publication's results were produced with
    on_path: bool = True  # False -> a binary under data/tools/
    version_cmd: tuple[str, ...] = ()
    version_line: int = 0  # which line of the output carries the version

    @property
    def path(self) -> Path | None:
        """Resolved executable, or None when it cannot be found."""
        if self.on_path:
            found = shutil.which(self.name)
            return Path(found) if found else None
        p = TOOLS_DIR / self.name
        return p if p.exists() else None


TOOLS: tuple[Tool, ...] = (
    Tool(
        "mafft",
        ("exp1", "exp2"),
        "protein alignment for the patristic baseline (exp1 B2) and stage-5 gene trees (exp2 E2)",
        "apt install mafft   |   conda install -c bioconda mafft",
        "v7.505",
        version_cmd=("mafft", "--version"),
    ),
    Tool(
        "FastTree",
        ("exp1",),
        "patristic distances from the protein alignment (exp1 B2)",
        "apt install fasttree   |   conda install -c bioconda fasttree",
        "2.1.11",
        version_cmd=("FastTree",),
    ),
    Tool(
        "mmseqs",
        ("exp2",),
        "homology-block clustering for the candidate pool (exp2 A1)",
        "apt install mmseqs2   |   conda install -c bioconda mmseqs2",
        "15-6f452",
        version_cmd=("mmseqs", "version"),
    ),
    Tool(
        "iqtree2",
        ("exp2",),
        "branch lengths on the fixed species topology (exp2 E2)",
        "download a release from https://github.com/iqtree/iqtree2/releases into data/tools/",
        "2.3.6",
        on_path=False,
        version_cmd=(str(TOOLS_DIR / "iqtree2"), "--version"),
    ),
    Tool(
        "trimal",
        ("exp2",),
        "alignment trimming before the trees and codeml (exp2 E2, E3)",
        "build from https://github.com/inab/trimal into data/tools/",
        "1.5.rev0",
        on_path=False,
        version_cmd=(str(TOOLS_DIR / "trimal"), "--version"),
    ),
    Tool(
        "codeml",
        ("exp2",),
        "dN, dS and omega under PAML's codon models (exp2 E3)",
        "build PAML 4.10.10 and copy src/codeml into data/tools/ "
        "(http://abacus.gene.ucl.ac.uk/software/paml.html)",
        "4.10.10",
        on_path=False,
    ),
    Tool(
        "yn00",
        ("exp2",),
        "pairwise dN/dS (Yang & Nielsen) alongside codeml (exp2 E3)",
        "build PAML 4.10.10 and copy src/yn00 into data/tools/ "
        "(http://abacus.gene.ucl.ac.uk/software/paml.html)",
        "4.10.10",
        on_path=False,
    ),
)

BY_NAME = {t.name: t for t in TOOLS}


def missing_message(name: str) -> str:
    """The one wording every stage uses when a tool is absent."""
    t = BY_NAME[name]
    where = "on PATH" if t.on_path else f"at {TOOLS_DIR / t.name}"
    return (
        f"missing external tool: {t.name} (expected {where})\n"
        f"  needed for: {t.stage}\n"
        f"  install:    {t.install}\n"
        f"  this work used {t.name} {t.used_version}\n"
        f"  check everything: uv run python scripts/check_tools.py"
    )


def require(name: str) -> Path:
    """Resolved path to `name`, or exit with the standard message."""
    p = BY_NAME[name].path
    if p is None:
        sys.exit(missing_message(name))
    return p


def installed_version(t: Tool) -> str:
    """Best-effort version string; '?' when the tool has no cheap version flag."""
    if not t.version_cmd:
        return "?"
    try:
        r = subprocess.run(t.version_cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "?"
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    return lines[t.version_line] if len(lines) > t.version_line else "?"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exp",
        choices=["exp1", "exp2", "controls"],
        help="only this experiment's tools ('controls' is the composition-controls analysis)",
    )
    ap.add_argument("--quiet", action="store_true", help="print only what is missing")
    args = ap.parse_args()

    wanted = [t for t in TOOLS if not args.exp or args.exp in t.experiments]
    if not wanted:
        print(f"{args.exp} needs no external tools")
        return

    missing = [t for t in wanted if t.path is None]
    if not args.quiet:
        print(f"{'tool':<10} {'used':<10} {'found':<8} reported version ('?' = no version flag)")
        print("-" * 78)
        for t in wanted:
            found = "yes" if t.path else "NO"
            detail = installed_version(t) if t.path else "not found"
            print(f"{t.name:<10} {t.used_version:<10} {found:<8} {detail[:52]}")
        print()
    for t in missing:
        print(missing_message(t.name), file=sys.stderr)
        print(file=sys.stderr)
    if missing:
        names = ", ".join(t.name for t in missing)
        sys.exit(f"{len(missing)} of {len(wanted)} external tools missing: {names}")
    if not args.quiet:
        print(f"all {len(wanted)} external tools present")


if __name__ == "__main__":
    main()
