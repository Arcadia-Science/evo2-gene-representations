# Between-family Wasserstein design

**Status:** active in Experiment 1.

Centroid distances summarize each family with one vector. The Wasserstein analysis instead compares
the complete empirical distributions of locus embeddings and therefore retains family spread and
shape.

## Metric

For each block:

1. L2-normalize every locus embedding.
2. Compute cross-family angular costs in the shared representation space.
3. Assign uniform mass within each family, so every family has total mass one regardless of size.
4. Compute exact `emd2` on squared angular costs and report its square root as W2.

No per-family standardization or family-size subsampling is applied. The current publication sweep
uses exact W2 only; earlier FGW and multi-panel comparisons are not part of the active workflow.

## Scoring

The W2 family matrix is scored against:

- Pfam-HMM profile Jensen–Shannon divergence;
- mean cross-family 6-mer distance;
- absolute difference in mean family GC content.

`ot_between_family_sweep.py` writes one matrix and one score table per layer:

- `betweenfam_ot_wasserstein_distances.csv`;
- `betweenfam_ot_metadata.json`;
- `between_family_ot_scores.csv`.

The score table contains Spearman rho for each baseline, and a one-sided Mantel permutation
p-value when `--n-perms` is greater than zero. `ot_between_family_sweep.py` defaults to 9,999
permutations; the Experiment 1 runner passes `--n-perms 0`, so `p_mantel` is empty in the tracked
score tables.

## Implementation

- [`ot_between_family.py`](ot_between_family.py): exact transport computation and result writer;
- [`ot_between_family_sweep.py`](ot_between_family_sweep.py): mammalian all-block driver;
- [`mammal_between.py`](../mammalian_orthologs/mammal_between.py): active baseline matrices and
  centroid comparison.
