"""Codon-aligned diagnostic sites for platypus steering."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from steer_lib import codon_blocks  # noqa: E402

BASES = frozenset("ACGT")


def diagnostic_sites_in_continuation(
    human_cds: str, platypus_cds: str, prefix_bp: int
) -> dict[int, str]:
    """Map continuation offsets to differing platypus bases in aligned codons."""
    human_blocks, platypus_blocks, _human_protein, _platypus_protein = codon_blocks(
        human_cds, platypus_cds
    )
    sites: dict[int, str] = {}
    for (human_start, human_end), (platypus_start, _platypus_end) in zip(
        human_blocks, platypus_blocks, strict=False
    ):
        for offset in range(int(human_end - human_start)):
            human_index = int(human_start) + offset
            platypus_index = int(platypus_start) + offset
            human_codon = human_cds[3 * human_index : 3 * human_index + 3]
            platypus_codon = platypus_cds[3 * platypus_index : 3 * platypus_index + 3]
            if len(human_codon) < 3 or len(platypus_codon) < 3:
                continue
            for codon_position in range(3):
                human_base = human_codon[codon_position]
                platypus_base = platypus_codon[codon_position]
                if (
                    human_base not in BASES
                    or platypus_base not in BASES
                    or human_base == platypus_base
                ):
                    continue
                platypus_nt_index = 3 * platypus_index + codon_position
                if platypus_nt_index >= prefix_bp:
                    sites[platypus_nt_index - prefix_bp] = platypus_base
    return sites
