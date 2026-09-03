# Evo2 shared embedding code

This directory contains the experiment-independent Evo2 wrapper:

| File | Purpose |
|---|---|
| [`evo2_embedding.py`](evo2_embedding.py) | Load Evo2-7B and return hidden states from all 32 blocks at selected sequence positions |

Dataset-specific embedding is kept with its experiment:

- Experiment 1 and the composition controls:
  [`mammalian_orthologs/embed_cds_masked_mammal.py`](../mammalian_orthologs/embed_cds_masked_mammal.py)
- Experiment 2:
  [`steering/platypus/strat/stage2_embed.py`](../steering/platypus/strat/stage2_embed.py)

Shared graph and distance functions are in [`geodesic_utils.py`](../geodesic_utils.py). See
[`REPRODUCING.md`](../../REPRODUCING.md) for the canonical commands and artifact paths.

Human-paralog, bacterial-phylogeny, ClinVar, and standalone layer-selection pipelines are not part
of the active publication workflow. Superseded versions remain available from Git history.
