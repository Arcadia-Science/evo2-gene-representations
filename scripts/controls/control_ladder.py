"""Shared display order, labels and colours for the composition-control ladder."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import arcadia_style as acs  # noqa: E402

# (key, label, control_dir or None for natural, colour). Ordered as a composition
# gradient: natural → 6-mer → 4-mer → codon(3) → dinucleotide(2) → GC(1) preserved, plus
# synonymous-recode (preserves PROTEIN, scrambles nucleotides) as the orthogonal rung.
_C = acs.CONTROL_COLORS
CONDITIONS = [
    ("natural", "Natural", None, _C["natural"]),
    ("kmer6_shuffle", "6-mer shuffle", "kmer6_shuffle", _C["kmer6_shuffle"]),
    ("kmer4_shuffle", "4-mer shuffle", "kmer4_shuffle", _C["kmer4_shuffle"]),
    ("codon_shuffle", "Codon shuffle", "codon_shuffle", _C["codon_shuffle"]),
    ("dinuc_shuffle", "Dinucleotide shuffle", "dinuc_shuffle", _C["dinuc_shuffle"]),
    ("gc_match", "GC-matched random", "gc_match", _C["gc_match"]),
    # The nested pair: missense_subset edits a strict SUBSET of synonymous_recode's changed bases,
    # so it perturbs the nucleotides less while damaging the protein the recode kept.
    (
        "synonymous_recode",
        "Synonymous recode (protein kept)",
        "synonymous_recode",
        _C["synonymous_recode"],
    ),
    (
        "missense_subset",
        "Missense at recode sites (protein damaged)",
        "missense_subset",
        _C["missense_subset"],
    ),
]
