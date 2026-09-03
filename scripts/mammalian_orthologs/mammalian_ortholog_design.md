# Mammalian ortholog experiment design

This document describes the production design used by experiment 1 and by the composition-controls
analysis. Canonical commands and outputs are listed in [`REPRODUCING.md`](../../REPRODUCING.md), and
the control stages in [`analyses/controls/README.md`](../../analyses/controls/README.md).

## Panel

- 48 HGNC families defined by `scripts/families_data.json` and `scripts/gene_families.py`;
- each human paralog defines a separate ortholog group;
- one-to-one orthologs from Ensembl Compara release 116 across 24 mammals;
- transcript spans and CDS annotations extracted from each species' own assembly;
- groups require at least 10 embedded species for within-group scoring;
- a 400-locus-per-family capped manifest provides a family-size sensitivity analysis.

The complete assembled panel contains 11,288 loci.

## Representation

Each transcript span is divided into windows of at most 8,000 bases, with at most 24 windows for a
long locus. Evo2-7B hidden states are collected at annotated CDS positions from all 32 blocks.
Positions are kept in transcript order, the first half is discarded, and the remaining states are
mean-pooled into one vector per locus and block.

Composition controls replace only the coding positions; introns and UTRs remain natural. Control
operators are deterministic per locus and preserve sequence length.

## Within-family analysis

For each ortholog group with at least 10 species, the representation-distance matrix is compared
with four references:

1. an independent mammalian species-tree patristic matrix;
2. a MAFFT/FastTree protein patristic matrix;
3. pairwise 6-mer distance;
4. pairwise GC-fraction difference.

`mammal_score.py --distance both` writes graph-geodesic and direct-angular results. Publication
Figures 2–3 use the angular tables. Block 15 is the pre-specified inference layer:
`within_family_uncertainty.py` computes family-level bootstrap intervals and per-group species-tree
Mantel tests.

## Between-family analysis

Families with at least eight embedded loci are compared using:

- geodesic distances between normalized family centroids;
- exact Wasserstein distances between the full empirical family distributions.

Both are scored against Pfam-HMM Jensen–Shannon divergence, between-family 6-mer distance, and
between-family GC difference. `mammal_between.py` and `ot_between_family_sweep.py` use 9,999
Mantel permutations by default.

## Control analysis

The composition-controls analysis embeds GC-matched, dinucleotide-shuffled, 4-mer-shuffled,
6-mer-shuffled, and synonymously recoded sequences. A matched third-codon-position pair contrasts synonymous and
missense edits at the same eligible sites and rate.

Control preservation is measured against the natural geometry:

- direct angular distance within families;
- exact Wasserstein distance between families.

The paired-p3 comparison is reported separately in Figure 12.
