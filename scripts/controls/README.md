# Composition controls

The composition-controls analysis asks which sequence constraints are sufficient to preserve the
natural mammalian ortholog geometry. This directory holds its sequence operators and two of its
figure generators; the runner, the tracked tables and the full description are in
[`analyses/controls/`](../../analyses/controls/README.md).

| File | Purpose |
|---|---|
| [`make_control_sequences.py`](make_control_sequences.py) | GC, k-mer, synonymous, missense, and paired-p3 sequence operators |
| [`plot_control_wasserstein.py`](plot_control_wasserstein.py) | Panel 04: control preservation by block |
| [`control_sequence_identity.py`](control_sequence_identity.py) | Panel 13: each control's nucleotide identity to its source gene |
| [`control_metrics_guide.md`](control_metrics_guide.md) | what each metric means and which comparisons are valid |

`make_control_sequences.py` is imported by experiment 1's embedder as well, so it is a dependency of
the publication pipeline even when no control is being built.

Embedding and scoring remain with the mammalian dataset:

- [`embed_cds_masked_mammal.py`](../mammalian_orthologs/embed_cds_masked_mammal.py) replaces coding
  positions in place and embeds every control.
- [`mammal_controls_score.py`](../mammalian_orthologs/mammal_controls_score.py) writes the per-block
  angular and Wasserstein preservation tables that panels 04 and 12 are built from.
- [`controls_score_graphfree.py`](../mammalian_orthologs/controls_score_graphfree.py) recomputes the
  same quantities without the neighbour graph, as an optional cross-check that no panel reads.
- [`paired_p3_figure.py`](../mammalian_orthologs/paired_p3_figure.py) renders panel 12.

The former human-panel control pipeline is not part of any current analysis. It remains recoverable
from Git history.
