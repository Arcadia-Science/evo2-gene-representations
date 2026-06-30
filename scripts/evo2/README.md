# scripts/evo2 — Evo2 pipelines

Home to all Evo2-related pipelines. Currently contains:

1. **`test_evo2.py`** — verify the `evo2_7b` checkpoint is used correctly by reproducing the zero-shot ClinVar variant-effect-prediction benchmark (expensive — see below).

2. **`run_species_pipeline.sh`** — test whether Evo2 embeddings of bacterial genomic windows recapitulate the GTDB phylogenetic tree with k-NN geodesic distances (originated by Goodfire).

3. **`run_ortholog_gene_pipeline_evo2.sh`** — the cross-kingdom **ortholog** gene-family experiment: one KEGG CDS per row spanning the tree of life, asking whether families separate (Axis A) and whether within-family geodesics track sequence divergence + host taxonomy (Axis B).

4. **`run_paralog_human_gene_pipeline_evo2.sh`** — the Evo2 side of the matched human **paralog** gene-panel experiment, on the same genes/loci the GPN-Star runner uses (`scripts/gpnstar/run_paralog_human_gene_pipeline_gpnstar.sh`).

The Evo2 embedding mechanics (model load, second-half pooling, the block taxonomy, single-/all-block
embed) are shared across the gene-panel scripts by [`evo2_embedding.py`](evo2_embedding.py) — the Evo2
analog of `../geodesic_utils.py`. Shared math utilities (k-NN, geodesic, Spearman/Mantel) are in
[../geodesic_utils.py](../geodesic_utils.py); ground-truth baselines in [../baselines/](../baselines/);
family definitions in [../gene_families.py](../gene_families.py); figure styling in
[../plot_utils.py](../plot_utils.py). Results are written to `results/YYYY-MM-DD_*/`.

---

## 1. Model verification & ClinVar VEP — `test_evo2.py`

Reproduces Evo2's zero-shot ClinVar VEP benchmark. WARNING: the full benchmark takes a lot of time to run, so
it is recommended to start with a small `--n-variants` pilot (e.g. 1000 variants) before the full set (~50000).

```bash
# Pilot, 1000 variants. Time/Hardware: ~85 min on 1x NVIDIA A10G (24 GB).
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

Tests whether **Evo2 7B** embeddings of random bacterial genomic windows recapitulate the GTDB phylogenetic tree, via geodesic distances on a k-NN
manifold. A 500-species replication of
[Goodfire (2025)](https://www.goodfire.ai/research/phylogeny-manifold).

**Pipeline order:** `download_species_manifest.py` → `download_species_sequences.py`
→ `embed_and_geodesic_species.py` → `evo2_visualization.py`. All four are chained by `run_species_pipeline.sh`; each step is resumable (skips completed work).

| File | Purpose |
|------|---------|
| `download_species_manifest.py` | Download the latest GTDB metadata + tree; keep species representatives; stratified-sample 500 species by phylum (seed 42) → `gtdb_500_manifest.csv`. |
| `download_species_sequences.py` | Per species, sample ~5% of the genome as 4000 bp windows from NCBI (`⌈genome_size × coverage / 4000⌉` windows; deterministic md5 positions). |
| `embed_and_geodesic_species.py` | Embed each species as the mean of its windows (last 2000 bp for model confidence) from Evo2 layer 24; calculate angular k-NN graph + Dijkstra geodesic; correlate vs. GTDB patristic distance (Spearman + Mantel) within- and between-phyla. Uses the **evo2_7b_262k** checkpoint + tail-pooling to match Goodfire's weights, so it keeps its own embedding code rather than `evo2_embedding.py`. |
| `evo2_visualization.py` | Figures (`--figures`): cosine/geodesic-vs-patristic scatter plots, and UMAP + 3D PCA of the embeddings colored by GTDB rank. |
| `run_species_pipeline.sh` | Sequential background runner for all four steps (manifest → sequences → embed/geodesic → figures; tees to `logs/`). |

The species ground-truth baseline (download `bac120.tree`, compute the pairwise **GTDB patristic
distance** matrix) is [../baselines/calculate_species_baselines.py](../baselines/calculate_species_baselines.py),
imported by the embed script. Results are written to `results/YYYY-MM-DD_evo2-species/`.

**Time/Hardware** (500 species, ~5% genome coverage, 1x NVIDIA A10G 24 GB):

| Step | Approx. time |
|------|--------------|
| 1. Build manifest | ~1 min |
| 2. Download sequences (NCBI) | ~7 min (network-dependent) |
| 3. Embed + geodesic | ~7–8 h (embedding ~1 min/species; geodesic analysis is minutes) |
| 4. Figures | ~1–2 min |

End to end ≈ **7–8 h**, dominated by Evo2 embedding.

---

## 3. Cross-kingdom ortholog gene-family pipeline

Tests whether **Evo2 7B** embeddings of cross-kingdom CDS (one KEGG ortholog per row, spanning all
sequenced life) (a) separate by gene family — *Axis A* — and (b) recapitulate within-family sequence
divergence + host taxonomy — *Axis B*. This is the panel no GPN-Star configuration can pose (GPN is
human-genome-anchored). Layer-selection-driven and resumable.

**Pipeline order:** `gene_families.py build-ortholog` → `layer_sweep.py` →
`layer_selection.py` → `embed_and_geodesic_ortholog.py --from-sweep-layer` → shared baselines →
`gene_family_visualization.py`. Chained by `run_ortholog_gene_pipeline_evo2.sh`.

| File | Purpose |
|------|---------|
| `../gene_families.py` | `build-ortholog`: resolve family membership from curated KEGG KO groups, stratified-sample across taxa, fetch CDS → per-family FASTA + `manifest.csv`. |
| `layer_sweep.py` | Dense single-forward-pass sweep — embed all 32 Evo2 blocks at once (second-half pooled) → a layer stack for layer selection. Uses `evo2_embedding.embed_all_blocks`. |
| `embed_and_geodesic_ortholog.py` | Pull the selected block from the sweep cache (no re-embed), build the angular k-NN geodesic + family-centroid geodesic, run the Axis-A separability test + Axis-B within-family ρ, write the run dir. |
| *(ground-truth baselines)* | The shared [../baselines/](../baselines/) scripts (`pfam_hmm_jsd`, `between_family_baselines`, `protein_alignment_patristic_seqid`, `kmer_sequence_divergence`, `taxonomic_distance`), run with `--seq-source evo2`. |
| `gene_family_visualization.py` | Between-family (Axis A) + within-family (Axis B) heatmaps + Spearman-ρ bars; the model-agnostic between-axes / matrix-gallery / convergent-ranks figures. |
| `run_ortholog_gene_pipeline_evo2.sh` | Sequential background runner: build → sweep → select → embed @ layer → baselines → figures (tees to `logs/`). |

Results are written to `results/YYYY-MM-DD_evo2-gene-families-blocks<layer>/`.

---

## 4. Matched human paralog gene-family pipeline

The Evo2 side of the apples-to-apples **paralog** comparison: embeds the GRCh38 genomic string over
each gene's transcript span — the *same* locus GPN-Star tiles with multiz windows — for the 580 genes
both models can embed, and scores it against the *same* human baselines as the GPN-Star run.

**Pipeline order:** `embed_and_geodesic_paralog.py` (dense sweep) → `layer_selection.py --panel human`
→ `embed_and_geodesic_paralog.py --from-layer` → `test_sample_human_genes.py prefetch-cds` → shared
baselines (+ the composition control) → `gene_family_visualization.py`. Chained by
`run_paralog_human_gene_pipeline_evo2.sh`.

| File | Purpose |
|------|---------|
| `embed_and_geodesic_paralog.py` | Fetch each gene's GRCh38 transcript-span string (Ensembl, cached); embed all blocks per window, mean-pool across windows; sweep cache → run dir at the selected block. The matched gene/locus set comes from `test_sample_human_genes.load_matched_panel()`. |
| *(ground-truth baselines)* | The shared [../baselines/](../baselines/) scripts (sampled CDS seq-identity, protein-alignment patristic/seq-id, k-mer, Pfam-JSD, between-family), run with `--seq-source gpn` so they score the Evo2-human run exactly as they score GPN-human. |
| `../controls/transcript_composition_control.py` | The transcript-span composition null (k-mer + GC on the actual genomic input). |
| `run_paralog_human_gene_pipeline_evo2.sh` | Sequential background runner for the Evo2 matched-human job. |

Results are written to `results/YYYY-MM-DD_evo2-human-panel-blocks<layer>/`.

---

**Troubleshooting / scaling experiments** (one-off, not part of the live pipelines) live in
[`troubleshooting/`](troubleshooting/) — layer/tensor/k sweeps, the 2400-species scale-up, and the
r220 membership checks.
