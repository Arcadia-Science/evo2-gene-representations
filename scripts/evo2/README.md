# scripts/evo2 — Evo2 testing

Home to all Evo2 - related pipelines. 
Will soon include gene family pipelines, but currently contains: 

1. **`test_evo2.py`** — verify the `evo2_7b` checkpoint is used correctly by reproducing the zero-shot ClinVar variant-effect-prediction benchmark (expensive — see below).

2. **`run_species_pipeline.sh`** — test whether Evo 2 embeddings of bacterial genomic windows recapitulate the GTDB phylogenetic tree with knn geodesic distances, originated by Goodfire.

Shared math utilities (k-NN, geodesic, Spearman/Mantel) are in
[../geodesic_utils.py](../geodesic_utils.py); figure styling in
[../plot_utils.py](../plot_utils.py). Results are written to
`results/YYYY-MM-DD_*/`.

---

## 1. Model verification & ClinVar VEP — `test_evo2.py`

Reproduces Evo 2's zero-shot ClinVar VEP benchmark. WARNING: the full benchmark takes a lot of time to run, so
it is recommended to start with or a small `--n-variants` pilot (ex: 1000 variants) before the full set (~50000).

```bash
# Pilot, 1000 variants. Time/Hardware: Took ~85 min on 1x NVIDIA A10G (24 GB).
uv run python scripts/evo2/test_evo2.py --n-variants 1000

# Full benchmark, 50,164 variants. Time/Hardware: ~70 h (~3 days) on 1x NVIDIA A10G (24 GB).
uv run python scripts/evo2/test_evo2.py
```

**Requirements:** needs the GRCh38 reference genome at
`data/genome/GRCh38.primary_assembly.fa` (Ensembl release-111) and `pyfaidx`
(declared in `pyproject.toml`). The genome is **auto-downloaded on first run** if
it isn't already there (~880 MB download, ~3.1 GB on disk, one-time). Outputs
(`clinvar_vep_*.parquet`, `metrics_*.json`) at `results/YYYY-MM-DD_evo2-clinvar/`.

---

## 2. Species phylogeny pipeline

Tests whether **Evo 2 7B** embeddings of random bacterial genomic windows recapitulate the GTDB phylogenetic tree, via geodesic distances on a k-NN
manifold. A 500-species replication of
[Goodfire (2025)](https://www.goodfire.ai/research/phylogeny-manifold).

**Pipeline order:** `download_species_manifest.py` → `download_species_sequences.py`
→ `embed_and_geodesic_species.py` → `evo2_visualization.py`. All four are chained by `run_species_pipeline.sh`; each step is resumable (skips completed work).

| File | Purpose |
|------|---------|
| `download_species_manifest.py` | Download the latest GTDB metadata + tree; keep species representatives; stratified-sample 500 species by phylum (seed 42) → `gtdb_500_manifest.csv`. |
| `download_species_sequences.py` | Per species, sample ~5% of the genome as 4000 bp windows from NCBI (`⌈genome_size × coverage / 4000⌉` windows; deterministic md5 positions). |
| `embed_and_geodesic_species.py` | Embed each species as the mean of its windows (last 2000 bp for model confidence) from Evo 2 layer 24; calculate angular k-NN graph + Dijkstra geodesic; correlate vs. GTDB patristic distance (Spearman + Mantel) within- and between-phylums. |
| `calculate_species_baselines.py` | Ground-truth baseline: download `bac120.tree` and compute the pairwise **GTDB patristic distance** matrix (imported by the embed script). |
| `evo2_visualization.py` | Figures (`--figures`): cosine/geodesic-vs-patristic scatter plots, and UMAP + 3D PCA of the embeddings colored by GTDB rank. |
| `run_species_pipeline.sh` | Sequential background runner for all four steps (manifest → sequences → embed/geodesic → figures; tees to `logs/`). |

Results are written to `results/YYYY-MM-DD_evo2-species/`.

**Time/Hardware** (500 species, ~5% genome coverage, 1x NVIDIA A10G 24 GB):

| Step | Approx. time |
|------|--------------|
| 1. Build manifest | ~1 min |
| 2. Download sequences (NCBI) | ~7 min (network-dependent) |
| 3. Embed + geodesic | ~7–8 h (embedding ~1 min/species; geodesic analysis is minutes) |
| 4. Figures | ~1–2 min |

End to end ≈ **7–8 h**, dominated by Evo 2 embedding.
