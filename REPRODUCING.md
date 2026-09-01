# Reproducing the publication

The publication is organized around three shell runners in [experiments/](experiments/). Run them
from the repository root. With no argument they print every stage and execute nothing:

```bash
bash experiments/exp1_gene_family_geometry.sh
bash experiments/exp2_composition_controls.sh
bash experiments/exp3_platypus_steering.sh
```

Use `--figures` to render from existing artifacts or `--run` for an end-to-end run. Full runs
are resumable but require the local datasets, external bioinformatics tools, Evo2-7B, and many
GPU-hours.

## Publication scope

| Experiment | Analysis | Figures |
|---|---|---|
| 1 | Between- and within-family geometry across 48 HGNC families and 24 mammals | 1–3 |
| 2 | Composition controls and the matched protein-versus-nucleotide comparison | 4, 12 |
| 3 | Conservation-stratified human-to-platypus steering at block 27 | 5–11 |

Experiments 1–2 embed all 32 residual-stream blocks. Transcript spans are split into windows;
hidden states are selected at CDS positions and mean-pooled over the second half of those positions.

## Required inputs

`--figures` reads only [figure_data/](figure_data/), which is tracked, so it runs on a fresh clone:

| Table prefix | Figures |
|---|---|
| `figure_data/exp1_*` | 1–3 |
| `figure_data/exp2_*` | 4, 12 |
| `figure_data/exp3_*` | 5–11 |

`figure_data/README.md` lists every table with its grain and size. `--run` additionally needs
`data/` and writes the dated run directories under `results/`; the last stage of each runner calls
`scripts/build_figure_data.py`, which collapses those into `figure_data/`.

Each runner copies its completed panels into `pub/figures/`.

## Experiment 1: gene-family geometry

```bash
bash experiments/exp1_gene_family_geometry.sh --run
bash experiments/exp1_gene_family_geometry.sh --figures
```

The runner performs the following stages:

1. Resolve one-to-one orthologs with Ensembl Compara release 116.
2. Download Ensembl annotations and genomes, extract transcript loci and CDS sequences, and assemble
   the complete and 400-cap manifests.
3. Build CDS-position masks and the independent mammalian species-tree distance matrix.
4. Embed the CDS positions of each transcript locus at all 32 blocks.
5. Score within-family geometry against the species tree, MAFFT/FastTree patristic distances,
   6-mer distances, and GC differences.
6. Score between-family centroid geodesics and exact Wasserstein distances against Pfam-HMM JSD,
   6-mer, and GC baselines.
7. Run the all-layer angular bootstrap and Wilcoxon uncertainty analyses.

Important settings:

- `mammal_score.py --distance angular` generates the direct-angular tables used by Figures 2–3.
- `ot_between_family_sweep.py --n-perms 9999` retains optional W2 Mantel significance tests; the
  publication driver passes `--n-perms 0` and skips them.
- `within_family_uncertainty.py` runs every available layer with 10,000 bootstrap replicates by
  default and Wilcoxon signed-rank inference. Geodesic-only within-family Mantel paths are archived.

Key active files:

| Responsibility | Code |
|---|---|
| Ortholog resolution and extraction | `scripts/mammalian_orthologs/{resolve_orthologs,download_bulk,extract_loci_bulk,assemble_datasets}.py` |
| Manifests, species tree, and CDS masks | `scripts/mammalian_orthologs/{build_capped_manifest,build_species_tree,build_cds_masks_mammal}.py` |
| Embedding | `scripts/mammalian_orthologs/embed_cds_masked_mammal.py` |
| Within-family scoring and inference | `scripts/mammalian_orthologs/{mammal_score,within_family_uncertainty}.py` |
| Between-family scoring | `scripts/mammalian_orthologs/mammal_between.py`, `scripts/baselines/{ot_between_family,ot_between_family_sweep}.py` |
| Figures 1–3 | `scripts/layer_sweep_summary.py`, `scripts/within_family_per_family_grid.py` |

## Experiment 2: composition controls

```bash
bash experiments/exp2_composition_controls.sh --run
bash experiments/exp2_composition_controls.sh --figures
```

The control ladder replaces coding positions in each natural transcript span while leaving
noncoding positions unchanged. The full runner embeds:

- synonymous recoding;
- GC matching;
- dinucleotide, 4-mer, and 6-mer shuffles;
- the paired third-codon-position synonymous and missense arms used by Figure 12.

`mammal_controls_score.py` writes graph-based preservation scores and the consolidated
`control_rho_by_layer.csv`. `controls_score_graphfree.py --axis both` writes the direct-angular
and Wasserstein preservation tables used by Figure 4.

| Figure | Generator |
|---|---|
| 4 | `scripts/controls/plot_control_wasserstein.py --pub` |
| 12 | `scripts/mammalian_orthologs/paired_p3_figure.py --pub` |

The removed sequence-identity analysis is not part of the publication pipeline.

## Experiment 3: platypus steering

```bash
bash experiments/exp3_platypus_steering.sh --run
bash experiments/exp3_platypus_steering.sh --figures
```

Experiment 3 has two connected panels:

- a 103-gene paired panel used for the all-block direction-geometry plots in Figures 6a–6b;
- a conservation-stratified panel of 400 human–platypus ortholog pairs used for generation,
  site-directionality, evolutionary-rate analyses, and Figures 5 and 7–11.

The production steering stages are:

1. Build a block-disjoint, five-stratum ortholog panel and apply shifted-window QC.
2. Embed human and platypus CDS representations across all 32 blocks.
3. Estimate leave-one-gene-out steering directions and run the pre-specified gates.
4. Generate at `blocks.27` for the primary, dose, and confound arms.
5. Score amino-acid and nucleotide outcomes, including private-site directionality.
6. Estimate fixed-topology branch lengths, dN, dS, and omega, then merge them with steering
   outcomes.

The clean runner generates at block 27. Its analysis commands read every saved condition in
`stage4_cds_mean_blocks27/`; therefore an existing directory may also contribute retained
`_L24` conditions to descriptive analyses.

| Figure | Generator |
|---|---|
| 5, 11 | `scripts/steering/platypus/strat/hypothesis_figures.py` |
| 6a, 6b | `scripts/steering/platypus/figures.py` |
| 7 | `scripts/steering/platypus/strat/steering_delta_strip.py` |
| 8 | `scripts/steering/platypus/strat/dose_and_alpha_figures.py` |
| 9a, 9b | `scripts/steering/platypus/strat/site_directionality_figures.py` |
| 10 | `scripts/steering/platypus/strat/gc_codon_figures.py` |

## Validate a checkout

```bash
uv run ruff format --check scripts
uv run ruff check scripts
uv run python -m compileall -q scripts
bash -n experiments/*.sh

bash experiments/exp1_gene_family_geometry.sh
bash experiments/exp2_composition_controls.sh
bash experiments/exp3_platypus_steering.sh
```

To validate the publication artifacts, run all three scripts with `--figures` and confirm that
each reports zero failed panels and that every panel measures exactly 1,000 or 500 points wide.
