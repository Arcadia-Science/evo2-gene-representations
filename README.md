# glm-latent-mapping

Investigates whether genome language models (GPN-Star, Evo2) encode gene family relationships in their latent spaces, using geodesic distance analysis as the primary metric.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Pre-commit hooks (linting via `ruff`):

```bash
uv run pre-commit install
```

## Models

Two GPN-Star 200M checkpoints are used. Both are downloaded automatically on first run and cached under `models/`.

| Alignment | HuggingFace ID | Species |
|-----------|----------------|---------|
| Vertebrate | `songlab/gpn-star-hg38-v100-200m` | 100 |
| Mammalian | `songlab/gpn-star-hg38-m447-200m` | 447 |

Each checkpoint includes:
- `model.safetensors` — model weights (~812 MB)
- `phylo_dist/` — pairwise and in-clade phylogenetic distance matrices
- `calibration_table/` — pre-computed VEP calibration tables

### Compatibility note

Transformers ≥4.44 initialises models inside an `accelerate` `init_empty_weights()` context, which places all tensors on the meta device. GPN-Star's `GPNStarPhyloInfo.__init__` loads numpy arrays and calls `.item()` on them, which crashes on meta tensors. `load_model_compat()` in [scripts/test_gpn_star.py](scripts/test_gpn_star.py) works around this by instantiating the model directly and loading the safetensors weights manually.

## MSA alignment data

Stage 2 of the GPN-Star verification (VEP benchmark) requires a zarr alignment file for each model variant. Both can be restricted to a single chromosome (~400 MB each) for quick testing.

### Vertebrate 100-way (`data/multiz100way.zarr`)

```bash
# Download (omit --include for full genome, ~42 GB)
huggingface-cli download songlab/multiz100way-pigz \
    --repo-type dataset \
    --include "chr22*" \
    --local-dir data/multiz100way-pigz

python -m gpn.data decompress data/multiz100way-pigz data/multiz100way.zarr
```

### Mammalian 447-way (`data/multiz447way.zarr`)

```bash
# Download (omit --include for full genome)
huggingface-cli download songlab/hg38_cactus447way \
    --repo-type dataset \
    --include "chr22*" \
    --local-dir data/multiz447way-pigz

python -m gpn.data decompress data/multiz447way-pigz data/multiz447way.zarr
```

Alternatively, `scripts/download_msa.py` provides a `--synthetic` flag that generates fast random alignment data for testing without the full download:

```bash
uv run python scripts/download_msa.py --synthetic
```

## Scripts

| Script | Purpose |
|--------|---------|
| [scripts/test_gpn_star.py](scripts/test_gpn_star.py) | Two-stage GPN-Star verification: (1) synthetic forward-pass sanity check, (2) VEP benchmark on `songlab/clinvar_vs_benign` |
| [scripts/test_evo2.py](scripts/test_evo2.py) | Evo2 verification: synthetic forward pass + optional VEP benchmark via log-likelihood ratio scoring |
| [scripts/download_msa.py](scripts/download_msa.py) | Downloads or synthesizes multiz100way alignment data |

### Running GPN-Star verification

```bash
# Stage 1 only — synthetic forward-pass check (no alignment data needed)
uv run python scripts/test_gpn_star.py

# Stage 1 + 2 — VEP benchmark (requires zarr alignment data)
uv run python scripts/test_gpn_star.py --vep
uv run python scripts/test_gpn_star.py --vep --alignments vertebrate
uv run python scripts/test_gpn_star.py --vep --alignments vertebrate --chrom chr22
```

### Tests

```bash
uv run python -m pytest tests/
```

The test suite (`tests/test_load_model_compat.py`) verifies the weight-loading compatibility workaround: correct weights loaded, tied weights restored, no meta-device tensors, and deterministic forward pass.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide--credit-for-contributions.md).
