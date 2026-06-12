#!/usr/bin/env bash
# Sequential runner for the species phylogeny pipeline (v2).
# Same as run_species_pipeline.sh, but the manifest is now restricted to GTDB
# species representatives — the only genomes present in bac120.tree — so the
# re-sampled accessions differ and the stale embedding cache is cleared before
# re-embedding.
# Designed to run in a detached screen session; all output is tee'd to logs/.
# Usage: screen -dmS species_pipeline_2 bash scripts/run_species_pipeline_2.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/species_pipeline_2.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "Species pipeline v2 started: $(date)"
echo "========================================"

echo ""
echo "--- [$(date)] uv sync (install dependencies) ---"
uv sync

echo ""
echo "--- [$(date)] Step 1: Build species manifest (representatives only) ---"
uv run python scripts/download_species_manifest.py

echo ""
echo "--- [$(date)] Step 2: Download genomic windows ---"
uv run python scripts/download_species_sequences.py

echo ""
echo "--- [$(date)] Clearing stale embedding cache (accessions changed) ---"
rm -f data/species/embeddings/evo2_species_embeddings.npy data/species/embeddings/metadata.csv

echo ""
echo "--- [$(date)] Step 3: Embed with Evo 2 + geodesic analysis ---"
uv run python scripts/embed_and_geodesic_species.py

echo ""
echo "========================================"
echo "Pipeline v2 complete: $(date)"
echo "========================================"
