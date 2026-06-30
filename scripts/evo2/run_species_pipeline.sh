#!/usr/bin/env bash
# Sequential runner for the species phylogeny pipeline.
# The manifest is restricted to GTDB species representatives — the only genomes
# present in bac120.tree — and the stale embedding cache is cleared before
# re-embedding (re-sampled accessions differ from any earlier run).
# Designed to run in a detached screen session; all output is tee'd to logs/.
# Usage: screen -dmS species_pipeline bash scripts/evo2/run_species_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG=logs/species_pipeline.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "Species pipeline started: $(date)"
echo "========================================"

echo ""
echo "--- [$(date)] uv sync (install dependencies) ---"
uv sync

echo ""
echo "--- [$(date)] Step 1: Build species manifest (representatives only) ---"
uv run python scripts/evo2/download_species_manifest.py

echo ""
echo "--- [$(date)] Step 2: Download genome-wide 5% sampled windows ---"
uv run python scripts/evo2/download_species_sequences.py \
  --coverage 0.05 \
  --sequences-dir data/species/sequences_5pct

echo ""
echo "--- [$(date)] Clearing stale embedding cache (accessions changed) ---"
rm -f data/species/embeddings/evo2_species_embeddings.npy data/species/embeddings/metadata.csv

echo ""
echo "--- [$(date)] Step 3: Embed with Evo 2 + geodesic analysis ---"
uv run python scripts/evo2/embed_and_geodesic_species.py \
  --sequences-dir data/species/sequences_5pct

echo ""
echo "--- [$(date)] Step 4: Generate figures (correlation scatters + UMAP/PCA) ---"
# Use the results dir the embed step just wrote (newest match), so the date can't
# drift if the run crosses midnight.
RUN_DIR=$(ls -dt results/*_evo2-species 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
  echo "ERROR: no results/*_evo2-species directory found; skipping figures" >&2
else
  echo "Using run dir: $RUN_DIR"
  uv run python scripts/evo2/evo2_visualization.py --run-dir "$RUN_DIR"
fi

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "========================================"
