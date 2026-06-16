# glm-latent-mapping

Investigates whether genome language models encode biologically meaningful structure in their latent spaces, using geodesic distance analysis as the primary metric. Two parallel analyses:

1. **Gene family analysis** — do GPN-Star embeddings of human gene CDS sequences cluster by gene family (globins, HOX genes, kinases, etc.)? Ground truth: Pfam HMM-profile Jensen-Shannon divergence.
2. **Species phylogeny analysis** — do Evo 2 embeddings of random bacterial genomic windows recapitulate the GTDB phylogenetic tree? Ground truth: GTDB patristic distances from 120 marker genes. Replicates [Goodfire (2025)](https://www.goodfire.ai/research/phylogeny-manifold) at 500-species scale.

Both analyses share the same downstream pipeline: cosine → angular k-NN graph → Dijkstra geodesic distances → Spearman ρ + Mantel test against a biological ground truth.

---

## Requirements

| Resource | Recommended | Notes |
|----------|-------------|-------|
| GPU | 1× NVIDIA A10G (24 GB) | Reference config for both pipelines. Evo 2 7B (~14 GB) and GPN-Star 200M both fit comfortably. |
| CPU / RAM | ≥ 8 vCPU, ≥ 32 GB RAM | Driven almost entirely by the `flash-attn` build (see below), not by the analyses themselves. |
| Disk | ~80 GB (vertebrate-only) | Vertebrate MSA ~72 GB extracted; species data ~100 MB; models/results small. Both MSAs together push this to ~280+ GB. |

Reference instance: AWS **g5.8xlarge** (32 vCPU, 124 GB RAM, 1× A10G).

### Building flash-attn

`flash-attn` is a hard dependency (required by Evo 2) and is **compiled from source** during `uv sync` — there is no prebuilt wheel for this torch/CUDA combination. This compile is the single most resource-hungry step in setup and the most common reason a fresh environment fails to build.

**This was the lesson learned the hard way:** on a small instance (AWS **g5.xlarge** — 4 vCPU, 16 GB RAM) the build repeatedly **exhausted RAM and killed the SSH session** before it could finish. `ninja` spawns one nvcc compile job per CPU core by default, and each job consumes several GB of RAM, so a many-core / low-RAM box OOMs even though the GPU is fine.

Two things make the build reliable:

1. **Use a box with enough RAM** — the g5.8xlarge (124 GB) above completes cleanly. As a rule of thumb, budget roughly a few GB of RAM per parallel compile job.
2. **Cap the parallelism** so RAM stays bounded regardless of core count:

   ```bash
   export MAX_JOBS=4        # cap ninja to 4 parallel nvcc jobs
   uv sync
   ```

   With `MAX_JOBS=4` the build is slower (tens of minutes) but stays within a few GB of RAM and does not OOM.

Other notes:

- A **CUDA toolkit (`nvcc`)** matching the cu128 torch wheel must be available on the build host.
- `pyproject.toml` declares the build prerequisites under `[tool.uv.extra-build-dependencies]` (`torch`, `packaging`, `setuptools`, `wheel`, `ninja`) so `uv sync` can build the extension; **`ninja` in particular must be present**, or the compile falls back to extremely slow single-threaded mode.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Recommended on a fresh box, to keep the flash-attn build from OOMing:
export MAX_JOBS=4
uv sync
```

Pre-commit hooks (linting via `ruff`):

```bash
uv run pre-commit install
```

---

## Analysis 1 — Gene family geodesic (GPN-Star)

### Models

Two GPN-Star 200M checkpoints, downloaded automatically on first run into `models/`.

| Alignment | HuggingFace ID | Species |
|-----------|----------------|---------|
| Vertebrate | `songlab/gpn-star-hg38-v100-200m` | 100 |
| Mammalian | `songlab/gpn-star-hg38-m447-200m` | 447 |

**Compatibility note:** Transformers ≥4.44 initialises models inside an `accelerate` `init_empty_weights()` context, placing all tensors on the meta device. GPN-Star's `GPNStarPhyloInfo.__init__` loads numpy arrays and calls `.item()` on them, which crashes on meta tensors. `load_model_compat()` in [scripts/gpnstar/test_gpn_star.py](scripts/gpnstar/test_gpn_star.py) works around this by direct instantiation + manual safetensors loading.

### MSA alignment data

GPN-Star requires a zarr multiple-sequence alignment for each model variant.

```bash
# Vertebrate 100-way alignment (~42 GB download, ~72 GB extracted)
uv run python scripts/gpnstar/download_msa.py vertebrate
# → data/multiz100way.zarr

# Mammalian 447-way alignment (~121 GB download, ~200+ GB extracted)
uv run python scripts/gpnstar/download_msa.py mammalian
# → data/multiz447way.zarr

# Synthetic random alignment for smoke-testing (no download)
uv run python scripts/gpnstar/download_msa.py synthetic
```

**Disk requirements:** vertebrate only ~72 GB; both alignments ~280+ GB. The vertebrate alignment is already present at `data/multiz100way.zarr`.

### Running

```bash
# Verify model loads and runs a forward pass
uv run python scripts/gpnstar/test_gpn_star.py

# VEP benchmark on songlab/clinvar_vs_benign (requires zarr data)
uv run python scripts/gpnstar/test_gpn_star.py --alignments vertebrate

# Gene family embedding + geodesic analysis (main analysis)
uv run python scripts/gpnstar/embed_and_geodesic_genes.py
uv run python scripts/gpnstar/embed_and_geodesic_genes.py --model vertebrate --force-reembed
```

Results written to `results/YYYY-MM-DD_gpnstar-vertebrate/`.

---

## Analysis 2 — Species phylogeny geodesic (Evo 2)

Embeds 500 bacterial species sampled from [GTDB](https://gtdb.ecogenomic.org/) using Evo 2 7B (layer `blocks.24.mlp.l3`), then tests whether geodesic distances in that embedding space correlate with GTDB patristic distances.

### Model

Evo 2 7B is downloaded automatically on first run (~14 GB). No MSA data required — Evo 2 operates directly on raw DNA sequences.

### Data requirements

All data is fetched by the pipeline scripts; nothing needs to be pre-downloaded.

| Data | Source | Size | Path |
|------|--------|------|------|
| GTDB metadata | `data.gtdb.ecogenomic.org` | ~30 MB | `data/species/bac120_metadata.tsv.gz` |
| GTDB reference tree | `data.gtdb.ecogenomic.org` | ~5 MB | `data/species/bac120.tree` |
| 500-species manifest | generated by script | <1 MB | `data/species/gtdb_500_manifest.csv` |
| Genomic windows (~5% coverage/species, 4000 bp each) | NCBI Datasets genome packages | ~100 MB | `data/species/sequences_5pct/` |
| Evo 2 embeddings (500 × 4096) | computed | ~8 MB | `data/species/embeddings/` |

### Pipeline

The three scripts run sequentially. Each is fully resumable — it skips already-completed work.

**Step 1 — Build species manifest** (~5 min)

Downloads the latest GTDB release metadata, keeps GTDB species representatives only (the genomes present in `bac120.tree`), stratified-samples 500 of them proportionally across all phyla (seed 42), and writes `data/species/gtdb_500_manifest.csv`.

```bash
uv run python scripts/evo2/download_species_manifest.py
```

**Step 2 — Download genomic windows** (~30–60 min, rate-limited by NCBI)

For each species, samples ~5% of the genome as 4000 bp windows via the NCBI Datasets API + Entrez — `⌈genome_size × coverage / 4000⌉` windows (~50–60 for a typical bacterium, following Goodfire's "~5% of the genome"). The full 4000 bp window is stored; only the last 2000 bp are pooled at embed time (the first 2000 bp prime the autoregressive model). Window positions are deterministic from `md5(ncbi_accession + coverage + window_bp + idx)`.

```bash
export NCBI_EMAIL="you@example.com"        # required by NCBI as a contact
uv run python scripts/evo2/download_species_sequences.py --coverage 0.05 --sequences-dir data/species/sequences_5pct
# Set NCBI_API_KEY env var for 10 req/s instead of 3 req/s
```

**Step 3 — Embed + geodesic** (~2–8 hours depending on GPU)

Embeds each species as the mean of its window embeddings (last 2000 bp of each window) from Evo 2 layer 24, builds a cosine → angular k-NN graph, computes Dijkstra geodesic distances, downloads and parses the GTDB tree (via `dendropy`), computes patristic distances, and runs Spearman ρ + Mantel test.

```bash
uv run python scripts/evo2/embed_and_geodesic_species.py --sequences-dir data/species/sequences_5pct
uv run python scripts/evo2/embed_and_geodesic_species.py --sequences-dir data/species/sequences_5pct --force-reembed
```

Results written to `results/YYYY-MM-DD_evo2-species/`.

### Running the full pipeline in the background

```bash
# Run from the repo root:
screen -dmS species_pipeline bash scripts/evo2/run_species_pipeline.sh
# Detach from screen: Ctrl-A D
# Reattach: screen -r species_pipeline
# Watch log: tail -f logs/species_pipeline.log
```

---

## Scripts

Each analysis lives in its own folder, with a README describing every file:

- **[scripts/evo2/](scripts/evo2/README.md)** — species phylogeny pipeline (Evo 2): manifest → window sampling → embed + geodesic + patristic baseline → figures.
- **[scripts/gpnstar/](scripts/gpnstar/README.md)** — gene-family pipeline (GPN-Star): MSA download → embed + geodesic + Pfam/seq-id/paralog baselines → figures.

Shared helpers imported by both:

| Module | Purpose |
|--------|---------|
| [scripts/geodesic_utils.py](scripts/geodesic_utils.py) | k-NN graph (angular distance), Dijkstra geodesic, Spearman + Mantel, within-vs-between permutation test |
| [scripts/sequence_baselines.py](scripts/sequence_baselines.py) | Alignment-free k-mer sequence-divergence baseline used by both pipelines, so Evo 2 and GPN-Star are scored against an identical nucleotide-composition reference |
| [scripts/plot_utils.py](scripts/plot_utils.py) | Shared matplotlib publication style + NaN-aware colormap + significance-star helpers for figures |

---

## Data layout

```
data/
├── multiz100way.zarr/          # Vertebrate 100-way alignment (72 GB, hg38)
├── multiz447way.zarr/          # Mammalian 447-way alignment (not yet downloaded)
├── sequences/                  # CDS sequences for gene family analysis
│   ├── globins.fasta
│   ├── hox.fasta
│   └── ...
├── species/                    # Data for species phylogeny analysis
│   ├── bac120_metadata.tsv.gz  # GTDB bacterial metadata (latest release)
│   ├── bac120.tree             # GTDB reference tree (Newick)
│   ├── gtdb_500_manifest.csv   # 500-species sample manifest
│   ├── sequences_5pct/         # Per-species sampled FASTA files (~5% genome coverage)
│   └── embeddings/             # Evo 2 embeddings (.npy) + metadata.csv
models/                         # HuggingFace model cache
results/                        # Analysis outputs (one dir per run, dated)
logs/                           # Background run logs
```

---

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide--credit-for-contributions.md).
