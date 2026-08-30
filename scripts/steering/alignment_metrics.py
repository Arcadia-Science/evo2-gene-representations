"""Alignment-based metrics used to score steered continuations."""

from __future__ import annotations

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq

BASES = frozenset("ACGT")


def nt_aligner() -> PairwiseAligner:
    """Return the nucleotide aligner used by the steering analyses."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 1.0, 0.0
    aligner.open_gap_score, aligner.extend_gap_score = -2.0, -0.5
    return aligner


def _protein(seq: str) -> str:
    clean = "".join(base if base in BASES else "N" for base in seq)
    return str(Seq(clean[: len(clean) // 3 * 3]).translate())


def score_generation(generated: str, target: str, private_sites: dict[int, str]) -> dict:
    """Score one generated continuation against its target and private sites."""
    alignment = nt_aligner().align(generated, target)[0] if generated and target else None
    if alignment is None:
        return {}
    generated_aligned, target_aligned = str(alignment[0]), str(alignment[1])

    target_index = private_hits = private_total = indel_bases = 0
    for generated_base, target_base in zip(generated_aligned, target_aligned, strict=False):
        if generated_base == "-" or target_base == "-":
            indel_bases += 1
        if target_base != "-":
            if target_index in private_sites:
                private_total += 1
                if generated_base == private_sites[target_index]:
                    private_hits += 1
            target_index += 1

    columns = max(len(generated_aligned), 1)
    matches = sum(
        1
        for generated_base, target_base in zip(generated_aligned, target_aligned, strict=False)
        if generated_base == target_base and generated_base != "-"
    )

    generated_protein, target_protein = _protein(generated), _protein(target)
    protein_alignment = (
        nt_aligner().align(generated_protein, target_protein)[0]
        if generated_protein and target_protein
        else None
    )
    aa_identity = (
        sum(
            1
            for generated_aa, target_aa in zip(
                str(protein_alignment[0]), str(protein_alignment[1]), strict=False
            )
            if generated_aa == target_aa and generated_aa != "-"
        )
        / max(len(str(protein_alignment[0])), 1)
        if protein_alignment is not None
        else np.nan
    )

    codons = [generated[i : i + 3] for i in range(0, len(generated) - 2, 3)]
    stop_codons = {"TAA", "TAG", "TGA"}
    n_stops = sum(codon in stop_codons for codon in codons)
    premature_stop = any(codon in stop_codons for codon in codons[:-1])

    indel_events = 0
    in_gap = False
    for generated_base, target_base in zip(generated_aligned, target_aligned, strict=False):
        gap = generated_base == "-" or target_base == "-"
        if gap and not in_gap:
            indel_events += 1
        in_gap = gap

    return {
        "pct_private_correct": 100 * private_hits / private_total if private_total else np.nan,
        "n_private_in_window": private_total,
        "aa_id_to_target": round(100 * aa_identity, 2),
        "nt_id_to_target": round(100 * matches / columns, 2),
        "indel_bp": indel_bases,
        "indel_events": indel_events,
        "n_stop_codons": n_stops,
        "premature_stop": bool(premature_stop),
    }
