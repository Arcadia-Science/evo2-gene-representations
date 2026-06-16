#!/usr/bin/env bash
# Sequential runner for the gene-family geodesic pipeline.
# Downloads the multiz100way vertebrate alignment, then embeds human gene CDS with
# GPN-Star, computes geodesic distances, and compares them to the Pfam JSD /
# CDS sequence-identity / Ensembl Compara paralog baselines. CDS coordinates and
# baseline sequences are fetched from Ensembl inside the embed step (cached).
# Designed to run in a detached screen session; all output is tee'd to logs/.
# Usage: screen -dmS gene_pipeline bash scripts/gpnstar/run_gene_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG=logs/gene_pipeline.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "Gene pipeline started: $(date)"
echo "========================================"

echo ""
echo "--- [$(date)] uv sync (install dependencies) ---"
uv sync

echo ""
echo "--- [$(date)] Step 1: Download multiz100way vertebrate alignment (~42 GB) ---"
# Swap 'vertebrate' for 'synthetic' to smoke-test the pipeline without the download.
uv run python scripts/gpnstar/download_msa.py vertebrate

echo ""
echo "--- [$(date)] Step 2: Embed genes with GPN-Star + geodesic analysis + baselines ---"
uv run python scripts/gpnstar/embed_and_geodesic_genes.py

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "========================================"
