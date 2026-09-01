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
def test_no_identity_output(name, k, fn):
    """No rung may silently hand back the input. This is the regression: the old rejection
    sampler returned the input verbatim once its 100 tries were exhausted, which happened for
    most sequences at k=6."""
    noop = sum(fn(s, random.Random(i)) == s for i, s in enumerate(SEQS))
    assert noop == 0, f"{name}: {noop}/{len(SEQS)} outputs identical to input (silent no-op)"


def test_positions_moved_flat_in_k():
    """Shuffle aggressiveness must not fall off with k, else any cross-rung comparison of
    control preservation is confounded by how much the sequence actually changed."""
    moved = {}
    for name, _k, fn in RUNGS:
        fracs = []
        for i, s in enumerate(SEQS):
            out = fn(s, random.Random(i))
            fracs.append(sum(a != b for a, b in zip(s, out)) / len(s))
        moved[name] = sum(fracs) / len(fracs)
    assert min(moved.values()) > 0.5, f"a rung barely moves anything: {moved}"
    spread = max(moved.values()) - min(moved.values())
    assert spread < 0.20, f"displacement is k-dependent, rungs not comparable: {moved}"


@pytest.mark.parametrize("name,k,fn", RUNGS)
def test_deterministic_given_seed(name, k, fn):
    s = SEQS[1]
    assert fn(s, random.Random(7)) == fn(s, random.Random(7)), f"{name}: not reproducible"


@pytest.mark.parametrize("name,k,fn", RUNGS)
def test_distinct_outputs_across_seeds(name, k, fn):
    """Different seeds must give different walks — a sampler stuck on one output would look
    like a working shuffle in every other test."""
    s = SEQS[2]
    outs = {fn(s, random.Random(seed)) for seed in range(12)}
    assert len(outs) > 6, f"{name}: only {len(outs)}/12 distinct outputs"


def test_degenerate_inputs_are_safe():
    """Too-short / single-symbol inputs return unchanged rather than raising."""
    for s in ["", "A", "AC", "ACG", "AAAAAAAA"]:
        for _name, _k, fn in RUNGS:
            out = fn(s, random.Random(0))
            assert len(out) == len(s)


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
def test_missense_perturbs_the_nucleotides_less_than_its_recode(seq):
    """The subset relation implies it, but state it as its own test: this is the property the
    interpretation rests on, and it must never invert."""
    recoded, missense = _recode_pair(seq, 15)
    assert len(_changed(seq, missense)) <= len(_changed(seq, recoded))


@pytest.mark.parametrize("seq", SEQS)
def test_missense_changes_the_amino_acid_of_every_codon_it_touches(seq):
    """A codon is either left exactly as the source, or comes back as a DIFFERENT amino acid. There
    is no third outcome: a silent edit would spend nucleotide change without buying protein change,
    which is the one thing this rung must not do."""
    recoded, missense = _recode_pair(seq, 12)
    for i in range(len(seq) // 3):
        src, mis = seq[i * 3 : i * 3 + 3], missense[i * 3 : i * 3 + 3]
        if _CODON.get(src) is None or mis == src:
            continue
        assert _CODON[mis] != _CODON[src], f"codon {i}: {src}->{mis} kept {_CODON[src]}"


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


@pytest.mark.parametrize("seq", SEQS)
def test_missense_still_damages_a_substantial_share_of_the_protein(seq):
    """Skipping unsatisfiable codons costs protein damage, so guard the floor: a rung that barely
    dents the protein cannot support any conclusion about protein. Measured ~28% of residues changed
    on the real panels; this asserts the mechanism has not silently stopped firing."""
    _, missense = _recode_pair(seq, 16)
    ncod = len(seq) // 3
    changed = sum(
        _CODON.get(seq[i * 3 : i * 3 + 3]) != _CODON.get(missense[i * 3 : i * 3 + 3])
        for i in range(ncod)
    )
    assert changed / ncod > 0.10, f"only {changed / ncod:.1%} of residues changed"


# ── paired_p3: the matched synonymous / missense pair ─────────────────────────────────────────
# The entire value of this pair is that the two arms differ in ONE thing, so every clause of that
# claim is asserted rather than assumed. If any of these fail the pair is not a control any more —
# it is two unrelated perturbations.


def _pair(seq, seed=5):
    """(synonymous arm, missense arm) for one sequence, using its own codon usage as the family."""
    usage = build_family_codon_usage({"f": [seq]})["f"]
    return (
        paired_p3(seq, usage, random.Random(seed), "synonymous"),
        paired_p3(seq, {}, random.Random(seed + 1), "missense"),
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
def test_paired_arms_only_ever_touch_codon_position_3(seq):
    """A change at position 1 or 2 would break the matched codon-position profile."""
    syn, mis = _pair(seq)
    for arm in (syn, mis):
        assert all(i % 3 == 2 for i in range(len(seq)) if seq[i] != arm[i])


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
def test_paired_missense_introduces_no_stop(seq):
    """Truncation is a different perturbation from substitution."""
    _, mis = _pair(seq)
    for i in range(len(seq) // 3):
        if _CODON.get(seq[i * 3 : i * 3 + 3]) != "*":
            assert _CODON.get(mis[i * 3 : i * 3 + 3]) != "*"


@pytest.mark.parametrize("seq", SEQS)
def test_paired_arms_are_length_preserving(seq):
    syn, mis = _pair(seq)
    assert len(syn) == len(mis) == len(seq)
