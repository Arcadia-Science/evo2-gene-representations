# Interpreting control preservation

Experiment 2 compares each control's representation geometry with the natural-sequence geometry.
This is a **preservation** measurement: Spearman rho near one means the control reproduces the
natural pairwise distances; lower values mean that the removed sequence constraint contributed to
the geometry.

## Two analysis levels

| Level | Geometry | Active output |
|---|---|---|
| Within family | Pairwise distances among ortholog groups inside each family | `blocks<L>/controls/control_within_scores.csv` |
| Between family | Pairwise distances among family distributions or centroids | `blocks<L>/controls/control_between_scores.csv` |

The consolidated layer table is
`results/2026-07-16_mammalian-orthologs-transcript_cdsmask/control_rho_by_layer.csv`.

Figure 4 uses graph-free preservation:

- direct angular distances within families;
- exact Wasserstein distances between families.

These summaries are written by
[`controls_score_graphfree.py`](../mammalian_orthologs/controls_score_graphfree.py) and rendered by
[`plot_control_wasserstein.py`](plot_control_wasserstein.py).

## Reading the ladder

- `gc_match`, `dinuc_shuffle`, `kmer4_shuffle`, and `kmer6_shuffle` preserve progressively
  richer nucleotide composition without preserving the encoded protein.
- `synonymous_recode` preserves the protein while resampling synonymous codons.
- `paired_p3_syn` and `paired_p3_missense` edit matched third-codon-position sites at the same
  rate and differ in whether the protein is preserved. Compare these two arms directly; Figure 12
  reports that contrast.

Do not compare a preservation rho with the baseline-recovery correlations from Experiment 1. The
former compares control geometry with natural geometry; the latter compares model geometry with an
external biological or compositional distance matrix.
