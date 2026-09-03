"""Regression tests for the composition-control sequence generators.

Written after a silent correctness bug: `_euler_shuffle`/`dinuc_shuffle` sampled the
Altschul-Erikson arborescence by REJECTION (`max_tries=100`) and, on exhaustion, returned the
input sequence unchanged. Acceptance collapses with vertex count, so `kmer4_shuffle` (64
vertices) no-op'd 34% of human CDS and `kmer6_shuffle` (~1000 vertices) 68% — inflating every
k-mer control's preservation rho toward 1 and manufacturing an apparent 4-mer/6-mer gap.

The no-op path is the thing under test: `test_no_identity_output` and `test_positions_moved_flat_in_k`
fail on the old implementation and pass on the Wilson's-algorithm one.

Run: uv run pytest analyses/test_make_control_sequences.py -q
"""

from __future__ import annotations
import collections
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "controls"))

from make_control_sequences import (  # noqa: E402
    _CODON,
    build_family_codon_usage,
    codon_counts,
    codon_shuffle,
    dinuc_shuffle,
    gc_match,
    klet_shuffle,
    missense_subset,
    paired_p3,
    synonymous_recode,
)

RUNGS = [
    ("dinuc", 2, lambda s, r: dinuc_shuffle(s, r)),
    ("kmer4", 4, lambda s, r: klet_shuffle(s, 4, r)),
    ("kmer6", 6, lambda s, r: klet_shuffle(s, 6, r)),
]


def spectrum(s: str, k: int) -> collections.Counter:
    return collections.Counter(s[i : i + k] for i in range(len(s) - k + 1))


def synthetic_cds(n: int, seed: int) -> str:
    """A biased random CDS-like string (biased so k-mers repeat, as in real coding sequence)."""
    rng = random.Random(seed)
    return "ATG" + "".join(rng.choices("ACGT", weights=[3, 2, 2, 3], k=n - 3))


SEQS = [synthetic_cds(n, seed) for seed, n in enumerate([300, 900, 1500, 3000])]


@pytest.mark.parametrize("name,k,fn", RUNGS)
def test_spectrum_exactly_preserved(name, k, fn):
    """The defining contract: the k-mer multiset is identical, not merely similar."""
    for i, s in enumerate(SEQS):
        out = fn(s, random.Random(i))
        assert spectrum(out, k) == spectrum(s, k), f"{name}: {k}-mer spectrum changed"


@pytest.mark.parametrize("name,k,fn", RUNGS)
def test_length_and_endpoints_preserved(name, k, fn):
    for i, s in enumerate(SEQS):
        out = fn(s, random.Random(i))
        assert len(out) == len(s), f"{name}: length changed"
        assert out[0] == s[0] and out[-1] == s[-1], f"{name}: endpoints changed"


@pytest.mark.parametrize("name,k,fn", RUNGS)
def test_deterministic_given_seed(name, k, fn):
    s = SEQS[1]
    assert fn(s, random.Random(7)) == fn(s, random.Random(7)), f"{name}: not reproducible"


def test_codon_shuffle_preserves_codon_multiset_and_frame():
    s = synthetic_cds(900, 11)
    out = codon_shuffle(s, random.Random(0))
    cod = lambda x: collections.Counter(x[i : i + 3] for i in range(0, len(x) - 2, 3))  # noqa: E731
    assert len(out) == len(s)
    assert cod(out) == cod(s)


def test_gc_match_matches_gc_and_length():
    s = synthetic_cds(1500, 3)
    out = gc_match(s, random.Random(0))
    gc = lambda x: (x.count("G") + x.count("C")) / len(x)  # noqa: E731
    assert len(out) == len(s)
    assert abs(gc(out) - gc(s)) < 0.05


# ── missense_subset: the nonsynonymous rung nested inside synonymous_recode ──────────────────
#
# The contract that makes this rung interpretable is a SUBSET relation, not a matching one:
# every base it changes must also have been changed by the recode it was drawn from. That is what
# licenses the one-sided reading — it perturbs the nucleotide sequence strictly less than the recode
# does, so a rho below the recode's cannot be blamed on lost nucleotides. If the subset rule leaked,
# the rung would be editing sites the recode never touched and the argument would collapse.


def _recode_pair(seq: str, seed: int):
    """(recoded, missense_subset) for one sequence, using its own codon usage as the family."""
    usage = build_family_codon_usage({"fam": [seq]})["fam"]
    recoded = synonymous_recode(seq, usage, random.Random(seed))
    return recoded, missense_subset(seq, recoded, random.Random(seed + 1))


def _changed(a: str, b: str) -> set:
    return {i for i in range(min(len(a), len(b))) if a[i] != b[i]}


@pytest.mark.parametrize("seq", SEQS)
def test_missense_changes_only_bases_the_recode_changed(seq):
    """THE contract: a strict subset, never a base the recode left alone."""
    recoded, missense = _recode_pair(seq, 11)
    assert len(missense) == len(seq)
    assert _changed(seq, missense) <= _changed(seq, recoded)


@pytest.mark.parametrize("seq", SEQS)
def test_missense_introduces_no_new_internal_stop(seq):
    """A premature stop would truncate the protein rather than substitute it, which is a different
    perturbation from the one this rung is supposed to apply."""
    _, missense = _recode_pair(seq, 13)
    ncod = len(seq) // 3
    for i in range(ncod - 1):
        src, mis = seq[i * 3 : i * 3 + 3], missense[i * 3 : i * 3 + 3]
        if _CODON.get(src) != "*":
            assert _CODON.get(mis) != "*", f"new stop introduced at codon {i}"


# ── paired_p3: the matched synonymous / missense pair ─────────────────────────────────────────
# The entire value of this pair is that the two arms differ in ONE thing, so every clause of that
# claim is asserted rather than assumed. If any of these fail the pair is not a control any more —
# it is two unrelated perturbations.


def _pair(seq, seed=5):
    """(synonymous arm, missense arm) for one sequence, using its own codon usage as the family."""
    usage = build_family_codon_usage({"f": [seq]})["f"]
    return (
        paired_p3(seq, usage, random.Random(seed), "synonymous"),
        paired_p3(seq, usage, random.Random(seed + 1), "missense"),
    )


@pytest.mark.parametrize("seq", SEQS)
def test_paired_arms_edit_identical_positions(seq):
    """THE core contract. Eligibility depends only on the source codon, so the arms must touch the
    same positions for ANY seeds — hence the deliberately different seeds in _pair."""
    syn, mis = _pair(seq)
    assert [i for i in range(len(seq)) if seq[i] != syn[i]] == [
        i for i in range(len(seq)) if seq[i] != mis[i]
    ]


@pytest.mark.parametrize("seq", SEQS)
def test_paired_syn_keeps_the_protein_and_missense_does_not(seq):
    """The one intended difference, in both directions: without the second assertion a pair that
    silently stopped substituting would still pass everything else."""
    syn, mis = _pair(seq)
    ncod = len(seq) // 3
    aa = lambda s, i: _CODON.get(s[i * 3 : i * 3 + 3])
    assert all(aa(seq, i) == aa(syn, i) for i in range(ncod))
    changed = sum(aa(seq, i) != aa(mis, i) for i in range(ncod))
    assert changed / ncod > 0.10, f"only {changed / ncod:.1%} of residues changed"


@pytest.mark.parametrize("seq", SEQS)
def test_paired_arms_are_length_preserving(seq):
    syn, mis = _pair(seq)
    assert len(syn) == len(mis) == len(seq)


@pytest.mark.parametrize("seq", SEQS)
def test_paired_arms_draw_from_the_same_codon_distribution(seq):
    """Both arms weight their alternatives from one table, so they differ in protein outcome
    rather than in how the replacement was drawn. An arm sampling uniformly while the other
    sampled by usage would confound the contrast with a composition shift."""
    usage = build_family_codon_usage({"f": [seq]})["f"]
    counts = codon_counts(usage)
    for aa, (cods, weights) in usage.items():
        if aa == "*":
            continue
        for codon, weight in zip(cods, weights, strict=False):
            assert counts[codon] == weight


@pytest.mark.parametrize("seq", SEQS)
def test_paired_arms_keep_the_same_edit_rate(seq):
    """Eligibility is a property of the source codon, so matching the weighting must not change
    how many codons each arm rewrites."""
    syn, mis = _pair(seq)
    n = len(seq) // 3
    edited = lambda a: sum(seq[i * 3 : i * 3 + 3] != a[i * 3 : i * 3 + 3] for i in range(n))
    assert edited(syn) == edited(mis)
