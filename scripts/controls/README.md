# Composition controls

Experiment 2 asks which sequence constraints are sufficient to preserve the natural mammalian
ortholog geometry. The canonical runner is
[`experiments/exp2_composition_controls.sh`](../../experiments/exp2_composition_controls.sh).

| File | Purpose |
|---|---|
| [`make_control_sequences.py`](make_control_sequences.py) | GC, k-mer, synonymous, missense, and paired-p3 sequence operators |
| [`plot_control_wasserstein.py`](plot_control_wasserstein.py) | Figure 4: graph-free control preservation by block |

Embedding and scoring remain with the mammalian dataset:

- [`embed_cds_masked_mammal.py`](../mammalian_orthologs/embed_cds_masked_mammal.py) replaces coding
  positions in place and embeds every control.
- [`mammal_controls_score.py`](../mammalian_orthologs/mammal_controls_score.py) computes graph-based
  preservation and writes `control_rho_by_layer.csv`.
- [`controls_score_graphfree.py`](../mammalian_orthologs/controls_score_graphfree.py) computes the
  angular and Wasserstein preservation tables used by Figure 4.
- [`paired_p3_figure.py`](../mammalian_orthologs/paired_p3_figure.py) renders Figure 12.

The former sequence-identity and human-panel control pipelines are not publication prerequisites.
They remain recoverable from Git history.
