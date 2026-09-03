"""Where the large, machine-local datasets live.

These are reference downloads -- Ensembl genomes, GTFs and CDS FASTAs -- that are far too big to
track and are not produced by any stage of the pipeline. Everything else the pipeline needs is
addressed relative to the repository root, so it follows a clone anywhere; these were the exception,
pinned by absolute path to one machine's NVMe instance store:

    DEST = Path("/opt/dlami/nvme/mammal_genomes")

That failed two ways. On any other machine the path does not exist, and `/opt` is usually not
writable, so a full `--run` died with a permissions error that read like a broken script rather
than a missing dataset. And on the authors' own instance the store is EPHEMERAL -- it is wiped when
the instance stops, which is how the platypus sequence set was lost while the mammal genomes
survived.

The default now sits inside the checkout, under `data/` (already git-ignored), so a fresh clone
resolves somewhere writable with no configuration. Point any of these elsewhere with an environment
variable -- for instance, to reuse a copy that already exists on a fast local disk:

    export GLM_SCRATCH=/opt/dlami/nvme          # moves all of them at once
    export GLM_MAMMAL_GENOMES=/mnt/big/genomes  # or just one

Nothing here creates directories or checks that they exist; `require()` does the checking, at the
point of use, so a missing dataset names itself and says how to obtain it.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(var: str, default: Path) -> Path:
    """`$var` as a path if it is set and non-empty, else `default`."""
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else default


# Root for every machine-local reference download. `data/` is git-ignored, so the default keeps
# these out of version control exactly as the previous absolute path did.
SCRATCH = _env_path("GLM_SCRATCH", ROOT / "data" / "external")

# Ensembl genome FASTA + GTF per species; written by mammalian_orthologs/download_bulk.py and read
# by extract_loci_bulk.py (experiment 1, stages A2-A3).
MAMMAL_GENOMES = _env_path("GLM_MAMMAL_GENOMES", SCRATCH / "mammal_genomes")

# Human and platypus CDS/peptide FASTAs and GTFs for the steering panel (experiment 3, stages
# A1-A2). No script in the repository fetches these; see REPRODUCING.md.
STRAT_SEQS = _env_path("GLM_STRAT_SEQS", SCRATCH / "strat_seqs")

# Per-species CDS FASTAs for the 24-mammal rate analysis (experiment 3, stage E1).
STRAT_MAMMAL_CDS = _env_path("GLM_STRAT_MAMMAL_CDS", STRAT_SEQS / "mammals")


def tmp_dir() -> str | None:
    """Parent for large scratch directories, or None for the system default.

    None lets `tempfile` honour `TMPDIR` as usual. Set `GLM_TMPDIR` only to override that -- for
    example when the system temp is a small tmpfs and an alignment pass needs real disk.
    """
    return os.environ.get("GLM_TMPDIR", "").strip() or None


def require(path: Path, what: str, how: str, var: str = "GLM_SCRATCH") -> Path:
    """Return `path`, or exit naming the dataset, where it was expected, and how to get it."""
    if path.exists():
        return path
    sys.exit(
        f"missing {what}\n"
        f"  expected at: {path}\n"
        f"  obtain it:   {how}\n"
        f"  or point elsewhere: export {var}=/path/to/it"
    )
