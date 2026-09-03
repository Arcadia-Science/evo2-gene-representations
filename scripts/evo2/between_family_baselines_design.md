# Between-family baselines

This file formerly described exploratory mechanism, cofactor, functional-context, and convergent-pair
baselines for earlier human panels. Those analyses are not part of the active publication workflow;
their implementation and detailed rationale remain available from Git history.

Experiment 1 uses three between-family references:

| Baseline | Interpretation | Implementation |
|---|---|---|
| Pfam-HMM profile JSD | Domain-level homology | `scripts/baselines/pfam_hmm_jsd.py` |
| Mean cross-family 6-mer distance | Nucleotide composition | `scripts/mammalian_orthologs/mammal_between.py` |
| Difference in mean GC content | Coarse nucleotide composition | `scripts/mammalian_orthologs/mammal_between.py` |

These matrices are scored against exact Wasserstein representation distances at all 32 blocks.
The publication path is Wasserstein only; the centroid-geodesic scoring this document used to
describe remains in `mammal_between.py` for legacy analyses and produces no manuscript output. See [`ot_between_family_design.md`](../baselines/ot_between_family_design.md) and
[`REPRODUCING.md`](../../REPRODUCING.md).
