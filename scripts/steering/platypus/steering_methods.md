# Methods — activation steering in Evo2, human → platypus

**Document status:** method reference for Experiment 2. The canonical command order is
`experiments/exp2_platypus_steering.sh`; production results are in `results_n400.md` and
`results_site_directionality.md`.

This document describes the experimental apparatus without assuming access to the source code.

**Model.** Evo2-7B, a DNA language model trained on genomes spanning bacteria, archaea and
eukaryotes. 32 transformer blocks; the internal state passed between blocks (the "residual stream")
is a 4096-dimensional vector per sequence position.

---

## 1. Question and design

The experiment tests whether adding a fixed vector to the residual stream during generation shifts
the generated sequence from human toward platypus sequence statistics. The contrast is human →
platypus; the two lineages diverged ~180 My ago, so orthologous genes remain alignable while carrying
many lineage-specific substitutions.

The apparatus has three components:

1. estimation of a human→platypus direction, excluding each gene from the direction applied to it;
2. injection of that direction during autoregressive generation, at a chosen layer and strength;
3. scoring of whether the output carries platypus-specific bases, against a set of control arms.

---

## 2. Gene panel

**Source.** 15,926 human genes with a platypus homolog (Ensembl release 116), filtered to 11,135
one-to-one high-confidence orthologs. Clustered all-against-all at 30 % identity / 50 % coverage into
**6,961 similarity blocks**; one gene drawn per block.

Blocks rather than genes are the sampling unit, so that excluding one gene from the direction
estimate (§3.2) also excludes its paralogues.

**Stratification.** Blocks were binned into five strata by human↔platypus protein identity (bin edges
64.7 / 72.9 / 80.5 / 88.1 %). A draw order was fixed before quality control began; the panel is the
first 80 passing genes per stratum in that order.

**Pairing and quality control.** A gene is kept if both species have an indel-free homologous window
of 30 codons beginning within the first 60 codons. **400 genes kept of 558 examined**, 80 per
stratum. QC pass rate by stratum, fastest-evolving to most conserved: 58.8 %, 63.0 %, 74.1 %, 87.0 %,
84.2 %. Human CDS length: 189 / 1,440 / 14,574 bp (min / median / max).

---

## 3. Estimating the steering direction

### 3.1 Representation

Each gene's coding sequence is run through the model once per species; residual-stream activations
are averaged over all coding-sequence positions to give one 4096-dimensional vector per (gene,
species, layer). Stored for all 32 layers.

The per-gene difference vector is

> **D_i = (platypus activation) − (human activation)** for gene *i*, at a given layer.

### 3.2 Leave-one-out construction

For gene *i*, the steering vector is the mean of every other gene's difference:

> **v₋ᵢ = mean over j ≠ i of D_j**

This gives 400 distinct directions per layer, each estimated from the other 399, so no gene
contributes to the direction later applied to it.

Recorded per gene and layer:

| quantity | definition |
|---|---|
| **loo_cos** | cosine between the consensus direction and this gene's own difference vector |
| **q_loo** | this gene's displacement along the consensus direction |
| **frac_on_cone** | squared cosine between the direction and the global mean-activation axis |
| **abs_cos_gc** | absolute cosine between the direction and the GC axis |
| **rel_norm** | magnitude of the steering vector divided by the mean magnitude of the residual stream at that layer |

Residual-stream magnitude and steering-vector magnitude both vary with depth, so `rel_norm` differs
between layers. Values are in `data/vector_diagnostics_by_layer.csv`.

### 3.3 Two nuisance axes

**Mean-activation axis ("cone").** The unit mean of all pooled activations across both species.
`frac_on_cone` measures each steering vector's alignment with it.

**GC axis.** Fitted, not specified: across genes, the activation difference is regressed on the
difference in GC content at the third codon position. The normalised coefficient vector is the GC
axis. The axis itself is stored, which is what the `gc_removed` arm (§4.4) projects out.

---

## 4. Injection

### 4.1 Operator

During generation a fixed vector is added to the chosen block's output at every sequence position,
and remains attached for the entire generation. Properties of this operator:

- the same vector is added to the prompt and to every generated base; there is no positional
  schedule or decay;
- the residual stream is added to, not replaced (replacement and interpolation operators exist in
  the codebase and were screened in an earlier pilot; all production arms are additive);
- the vector does not depend on what has been generated.

### 4.2 Dose conventions

| arm | strength | quantity held constant across genes |
|---|---|---|
| **add** | α = 1.0 for every gene | the injected vector |
| **add_own** | α set per gene to q_loo_i / ‖v₋ᵢ‖ | the displacement, matched to each gene's own observed shift |

Genes with larger difference vectors receive a different relative dose under a fixed α than under the
own-dose convention. Both arms were run. A dose ladder covers α ∈ {0.5, 1, 2, 3, 4}.

### 4.3 Generation settings

| parameter | value |
|---|---|
| prompt | first 90 bases (30 codons) of the human coding sequence |
| sampling | temperature 0.7, top-k 4 |
| length | 1000 bases, extended to 2500 for genes below the minimum-site threshold |
| samples per gene | 5–16, scaled inversely with the number of scorable sites |
| random seed | 42 |

### 4.4 Arms

| arm | construction |
|---|---|
| **unsteered** | no vector injected; involves no hook and is therefore identical across layers, generated once and shared |
| **add** | v₋ᵢ at α = 1 |
| **add_own** | v₋ᵢ at the per-gene dose of §4.2 |
| **random** | a random vector of the same magnitude as v₋ᵢ |
| **gc_removed** | v₋ᵢ with the GC axis projected out |
| **cone_removed** | v₋ᵢ with the mean-activation axis projected out |
| **cross_gene** | a different gene's vector, magnitude-matched |
| **cluster panels** | vectors drawn from the same or other directional sub-clusters of genes |

After any projection the vector's original magnitude is restored, so control arms differ from the
treatment arm in direction only.

### 4.5 Multiple layers

Arms for layers other than 27 are generated into the same output collection, tagged by layer.
`unsteered` involves no injection and is identical across layers, so both layers are scored against
the same baseline sequences gene by gene.

---

## 5. Scoring

### 5.1 Primary measurement

A **private bp** is a position where the platypus base differs from human and no other sampled
mammal carries that base. The primary metric is the percentage of a gene's private bp that the
generated sequence matches.

A secondary metric uses **platy bp (not human)** — positions where platypus differs from human,
without the uniqueness requirement. Median site counts per gene: 195 platy bp (not human), 57
private bp.

These two site sets were called "autapomorphic" and "diagnostic" until 2026-08-22. "Autapomorphy" is
a cladistic term for a derived character unique to one terminal taxon on a phylogeny; this metric
only asks whether a base is absent from the mammals that were sampled, so the name was dropped.

Each site call records the number of orthologous species that voted on it; the median across this
panel is 19.

### 5.2 Reference levels for the primary metric

Measured on 267 genes against the same site sets:

| sequence scored | private-bp score |
|---|---|
| the human sequence | 0.00 % |
| the platypus target | 100.00 % |
| a shuffled human sequence (composition preserved) | 42.64 % |
| the unsteered model generation | 36.22 % |

All results in `02_results.md` are reported as paired within-gene differences against the same
gene's unsteered generation.

### 5.3 Sequence-quality readouts

Reported alongside the primary metric: amino-acid identity to the platypus protein, nucleotide
identity to the platypus sequence, insertion/deletion burden as a percentage of generated bases, and
premature stop codons per 100 codons.

### 5.4 Composition readouts

Measured on the same window as the scored generation: overall GC, GC by codon position, codon-usage
Jensen–Shannon divergence over the 61 sense codons to each species' actual sequence, and pN/pS
against the platypus target (NG86 site counts; 1.0 indicates equal per-site rates of
protein-changing and silent differences).

The gene is the unit of analysis; multiple samples from one gene share a prompt and are averaged
before any across-gene statistic.

---

## 6. Scope of what was run

- The **random** arm was run at α ∈ {0.5, 1, 2}; α = 3 and α = 4 have no matched random arm at either
  layer.
- At layer 24, only α = 1 was run, for the **add**, **add_own** and **random** arms. The layer-24
  dose ladder and `gc_removed` arm were launched 2026-08-22 and are not included here.
- `gc_removed`, `cone_removed`, `cross_gene` and the cluster panels were run at layer 27 only.
- Candidate layers 24–27 were selected in an earlier pilot from a representation sweep and a
  mean-activation-axis check; the selection was not repeated at n = 400.
- Pooling is the mean over all coding-sequence positions; no positional structure is retained.

---

## Appendix — source files

Panel construction `strat/stage0_pool.py`, `strat/stage1_qc.py`; embedding and geometry
`strat/stage2_embed.py`; vectors, GC axis and cone `stage3_select.py`, `gc_control.py`,
`cone_check.py`; generation `strat/stage4_steer.py`, injection hook in
`scripts/steering/steer_lib.py`; scoring `strat/stage4_rescore_nt.py`, `strat/stage4_analysis.py`;
metric selector `strat/strat_metric.py`; figures `strat/hypothesis_figures.py`,
`strat/steering_delta_strip.py`, `strat/gc_codon_figures.py`. Driver `strat/run_400.sh`.
Pre-registered goals and decision gates: `conservation_stratified_design.md`.
