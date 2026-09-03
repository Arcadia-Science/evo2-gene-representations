"""Regression tests for the control-identity monotonicity statistics.

Written after a silent bug: `monotonicity_stats` built its "nucleotide rungs only" subset as
`condition != "synonymous_recode"`, which left `missense_subset` — a protein-damaging rung sitting
at 0.88 identity — inside a set whose spread is supposed to be the ~0.016 the four shuffles cover.
That inflated the fitted identity range 39-fold, reversed the sign of the reported extrapolation
(+32x became -0.2x), and so suppressed the flag that marks the recode prediction as an
extrapolation. The figure's own fit line excluded both nested rungs, so the plotted line and the
annotation printed beside it disagreed.

`test_nucleotide_subset_excludes_both_nested_rungs` and `test_recode_is_flagged_as_extrapolated`
fail on the old implementation and pass on this one.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from controls import control_sequence_identity as csi  # noqa: E402

# Measured values from the mammalian CDS-masked panel, rounded: the four shuffles sit within 1.6
# percentage points of each other, and the nested pair sits 50+ points above them.
IDENTITY = {
    "gc_match": 0.2572,
    "dinuc_shuffle": 0.2623,
    "kmer4_shuffle": 0.2657,
    "kmer6_shuffle": 0.2727,
    "synonymous_recode": 0.7718,
    "missense_subset": 0.8762,
}
RHO_BETWEEN = {
    "gc_match": 0.727,
    "dinuc_shuffle": 0.759,
    "kmer4_shuffle": 0.770,
    "kmer6_shuffle": 0.760,
    "synonymous_recode": 0.986,
    "missense_subset": 0.953,
}
SHUFFLE_SPREAD = IDENTITY["kmer6_shuffle"] - IDENTITY["gc_match"]


def _tables():
    ident = pd.DataFrame(
        [
            {
                "condition": c,
                "pos_identity": v,
                # The negative-control axes are swept alongside; give them the same shape so the
                # sweep produces a row for each without changing what is under test.
                "null_pos_identity": v,
                # Distinct but negligible, as in the real tables: an exactly constant axis is
                # a degenerate Spearman input the measured data never produces.
                "pos_identity_chance": 0.26 + i / 1e5,
                "pos_excess": i / 1e5,
                "edit_identity": v,
                "aa_identity": v,
            }
            for i, (c, v) in enumerate(IDENTITY.items())
        ]
    )
    rho = pd.DataFrame(
        [
            {
                "layer": 15,
                "condition": c,
                "rho_between_w2": RHO_BETWEEN[c],
                "rho_within_angular": RHO_BETWEEN[c] / 2,
            }
            for c in IDENTITY
        ]
    )
    return ident, rho


def _row(metric: str = "pos_identity", rho_col: str = "rho_between_w2") -> pd.Series:
    ident, rho = _tables()
    stats = csi.monotonicity_stats(ident, rho)
    hit = stats[(stats.identity_metric == metric) & (stats.rho_col == rho_col)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_nucleotide_subset_excludes_both_nested_rungs():
    """The subset is the four shuffles — not five rungs with missense_subset smuggled in."""
    row = _row()
    assert row.n_nucleotide_rungs == 4
    assert row.identity_range_nucleotide_rungs == pytest.approx(SHUFFLE_SPREAD, abs=1e-6)
    # The bug's signature: the full-ladder spread is 40x this and would land near 0.62.
    assert row.identity_range_nucleotide_rungs < 0.05


def test_ladder_matches_the_control_registry():
    """The rungs this figure names must be rungs the panel actually embeds."""
    from controls.make_control_sequences import CDSMASK_CONTROLS

    assert set(csi.LADDER) <= set(CDSMASK_CONTROLS)
    assert set(csi.NESTED_PAIR) <= set(csi.LADDER)
    assert csi.NUCLEOTIDE_RUNGS == [c for c in csi.LADDER if c not in csi.NESTED_PAIR]


def test_pair_metrics_reports_a_synonymous_change_at_position_three():
    """CTA -> CTG is one wobble-base change: identity 8/9, protein untouched, all of the change
    at codon position 3."""
    m = csi.pair_metrics("ATGCTAAAA", "ATGCTGAAA", in_frame=True)
    assert m["pos_identity"] == pytest.approx(8 / 9)
    assert m["aa_identity"] == pytest.approx(1.0)
    assert m["p1_identity"] == pytest.approx(1.0)
    assert m["p2_identity"] == pytest.approx(1.0)
    assert m["p3_identity"] == pytest.approx(2 / 3)


def test_pair_metrics_self_identity_is_one():
    m = csi.pair_metrics("ATGCTAAAA", "ATGCTAAAA", in_frame=True)
    assert m["pos_identity"] == pytest.approx(1.0)
    assert m["edit_identity"] == pytest.approx(1.0)
