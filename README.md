# Evo2 Gene Representations

This repository tests how Evo2-7B organizes gene-family sequence representations and whether that
structure can be used to steer what the model generates. The publication analysis has two
experiments:

1. **Gene-family geometry:** compare between- and within-family representation distances with
   homology, phylogeny, k-mer composition, and GC content across 48 HGNC families and 24 mammals.
2. **Platypus steering:** inject a human-to-platypus direction at block 27 and score whether
   generations move toward platypus-specific sequence features.

A third analysis, [composition controls](analyses/controls/), sits alongside them: it rewrites the
coding positions of experiment 1's panel under matched constraints and measures which of them are
sufficient to reproduce the natural geometry. It is not part of the publication pipeline and is run
on its own.

## Start here

| Resource | Purpose |
|---|---|
| [REPRODUCING.md](REPRODUCING.md) | Commands, inputs, outputs, and inference settings for both experiments |
| [experiments/](experiments/) | Canonical end-to-end runners; plan mode is the default |
| [figure_data/](figure_data/) | The tidy tables every figure reads; `--figures` needs nothing else |
| [analyses/controls/](analyses/controls/) | Composition controls: a matched sufficiency test over experiment 1's panel |

## Requirements

The reference system is an AWS g5.8xlarge with one NVIDIA A10G GPU, 32 vCPUs, and 124 GB RAM.
Evo2-7B requires approximately 14 GB for its model files. The complete data, embedding caches, and
result artifacts require substantially more disk space than the tracked repository.

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
export MAX_JOBS=4
uv sync
uv run pre-commit install
```

`flash-attn` is compiled during installation. `MAX_JOBS` limits parallel compilation and avoids
excessive memory use. A CUDA toolkit compatible with the installed PyTorch build must be available.

## Run the experiments

Each runner supports three modes:

- no argument: print the full plan without executing it;
- `--figures`: render publication figures from existing artifacts;
- `--run`: execute the complete precursor, analysis, and figure pipeline.

```bash
bash experiments/exp1_gene_family_geometry.sh
bash experiments/exp2_platypus_steering.sh

bash experiments/exp1_gene_family_geometry.sh --figures
bash experiments/exp2_platypus_steering.sh --figures
```

The composition-controls analysis has its own runner with the same three modes:

```bash
bash analyses/controls/run_composition_controls.sh --figures
```

Full runs require large local datasets and many GPU-hours. See
[REPRODUCING.md](REPRODUCING.md) before using `--run`.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/mammalian_orthologs/` | Experiment 1 dataset construction, embedding, scoring, and inference |
| `scripts/baselines/` | Pfam-HMM, patristic, k-mer, Wasserstein, and significance utilities |
| `scripts/steering/platypus/` | Experiment 2 paired-panel and steering analyses |
| `analyses/controls/`, `scripts/controls/` | Composition controls: runner, tables, sequence operators, and plotting |
| `scripts/evo2/evo2_embedding.py` | Shared Evo2 model loading and all-block position extraction |
| `scripts/geodesic_utils.py` | Graph, geodesic, centroid, and Mantel utilities (Mantel and upper-triangle helpers are metric-agnostic and used by the Wasserstein path; the geodesic and centroid functions serve legacy analyses only) |
| `scripts/arcadia_pub.py`, `scripts/arcadia_style.py` | Publication figure geometry and styling |

The canonical family definitions are in
[scripts/gene_families.py](scripts/gene_families.py) and
[scripts/families_data.json](scripts/families_data.json).

## Local data and results

[figure_data/](figure_data/) holds 20 tidy tables — one or two per figure, about 14 MB — and is the
only input `--figures` needs, so figure rendering works on a fresh clone with no downloads and no
GPU. `scripts/build_figure_data.py` writes it from a completed `--run`; the composition-control
tables it builds land in [analyses/controls/figure_data/](analyses/controls/figure_data/) instead.

Everything a full `--run` produces is local and ignored by Git: `data/` (sequences, manifests, Evo2
embedding caches) and the dated run directories under `results/`. It also reads
reference downloads that are neither tracked nor produced by the pipeline — Ensembl genomes, GTFs
and CDS FASTAs — which resolve under `data/external/` by default and are relocatable with the
environment variables in [REPRODUCING.md](REPRODUCING.md#reference-downloads-and-where-they-live).
The raw per-sample scores and generated sequences are archived separately. Earlier and superseded
analyses remain recoverable from Git history.

## Contributing

See Arcadia Science's
[software contribution guide](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide--credit-for-contributions.md).
