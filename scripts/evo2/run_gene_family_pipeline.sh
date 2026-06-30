#!/usr/bin/env bash
# Sequential runner for the Evo2 cross-kingdom gene-family pipeline.
# Chains: build family CDS sets (KEGG) -> embed + geodesic (Axis A + Axis B).
# Each step is resumable (skips completed work); all output is tee'd to logs/.
# Usage: screen -dmS gene_family_pipeline bash scripts/evo2/run_gene_family_pipeline.sh
#
# Optional env vars:
#   TARGET=400   CDS per family after dedup (passed to the build step)

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG=logs/evo2_gene_family_pipeline.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

TARGET="${TARGET:-400}"
DATA_DIR=data/evo2_gene_families

echo "========================================"
echo "Evo2 gene-family pipeline started: $(date)"
echo "========================================"

echo ""
echo "--- [$(date)] uv sync ---"
uv sync

echo ""
echo "--- [$(date)] Step 1: Build gene-family CDS sets from KEGG (target=$TARGET) ---"
# Resumable: skip the (slow, network-bound) build if a manifest already exists.
if [ -s "$DATA_DIR/manifest.csv" ]; then
  echo "Manifest present at $DATA_DIR/manifest.csv — skipping build (delete to rebuild)."
else
  uv run python scripts/evo2/build_gene_families_evo2.py --target "$TARGET"
fi

echo ""
echo "--- [$(date)] Step 2: Evo2 embed + geodesic (Axis A between-family, Axis B within-family) ---"
uv run python scripts/evo2/embed_and_geodesic_gene_families.py

echo ""
echo "--- [$(date)] Step 3: Figures ---"
RUN_DIR=$(ls -dt results/*_evo2-gene-families 2>/dev/null | head -1)
VIZ=scripts/evo2/gene_family_visualization.py
if [ -n "$RUN_DIR" ] && [ -f "$VIZ" ]; then
  echo "Using run dir: $RUN_DIR"
  uv run python "$VIZ" --run-dir "$RUN_DIR"
else
  echo "Skipping figures (visualization script not yet present or no results dir)."
fi

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "Results in: ${RUN_DIR:-results/<date>_evo2-gene-families}"
echo "========================================"
