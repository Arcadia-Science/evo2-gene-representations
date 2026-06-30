#!/usr/bin/env bash
# Full from-scratch Evo2 ortholog gene-family pipeline (layer-selection-driven).
#
# Chain:
#   1. Build family CDS sets from KEGG          (reused if a manifest already exists)
#   2. Dense layer sweep — RE-EMBED all 32 blocks, second-half pooling (the one GPU-heavy step)
#   3. Unsupervised layer selection             (z-scored within/between clustering ratio
#                                                + per-family centroid PCA)
#   4. Pick the recommended layer               (argmin within/between ratio)
#   5. Embed at that layer (free, from the sweep cache) + Axis-A/B baselines + figures
#
# Every step is resumable (re-embeds only when caches are stale) and tee'd to logs/.
# Usage: screen -dmS evo2_ortholog_genes bash scripts/evo2/run_ortholog_gene_pipeline_evo2.sh
#
# Optional env vars:
#   TARGET=400   CDS per family after dedup (build step; only used if no manifest yet)
#   PERMS=999    Axis-A separability permutations in the embed step (bump for a final run)

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG=logs/evo2_gene_family_pipeline.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

TARGET="${TARGET:-400}"
PERMS="${PERMS:-999}"
DATA_DIR=data/evo2_gene_families

echo "========================================"
echo "Evo2 ortholog gene-family pipeline (from scratch): $(date)"
echo "========================================"

echo ""
echo "--- [$(date)] uv sync ---"
uv sync

echo ""
echo "--- [$(date)] Step 1: Build gene-family CDS sets from KEGG (target=$TARGET) ---"
# The Evo2 panel is fixed (15 families / 5,276 CDS); reuse the manifest if present.
if [ -s "$DATA_DIR/manifest.csv" ]; then
  echo "Manifest present at $DATA_DIR/manifest.csv — reusing (delete to rebuild)."
else
  uv run python scripts/gene_families.py build-ortholog --target "$TARGET"
fi

echo ""
echo "--- [$(date)] Step 2: Dense layer sweep — re-embed all 32 blocks (second-half pooling) ---"
# --force-reembed regenerates layer_stack.npy with the current pooling. (The cached config
# would auto-invalidate anyway once the pooling key changed, but we make it explicit.)
uv run python scripts/evo2/layer_sweep.py --force-reembed

echo ""
echo "--- [$(date)] Step 3: Unsupervised layer selection (clustering ratio + per-family PCA, z-scored) ---"
uv run python scripts/layer_selection/layer_selection.py --model evo2 --standardize zscore
# Combined table/figure/recommendation (tolerates the GPN side being absent).
uv run python scripts/layer_selection/layer_selection_report.py --standardize zscore || true

echo ""
echo "--- [$(date)] Step 4: Select the recommended layer (min within/between ratio) ---"
SEL_DIR=$(ls -dt results/*_layer-selection | head -1)
SEL_CSV="$SEL_DIR/layer_scores_evo2_zscore.csv"
LAYER=$(uv run python - "$SEL_CSV" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print(int(df.loc[df["within_between_ratio"].idxmin(), "layer_idx"]))
PY
)
echo "Selected Evo2 layer: blocks.$LAYER  (from $SEL_CSV)"

echo ""
echo "--- [$(date)] Step 5: Embed @ blocks.$LAYER (from sweep cache) + Axis-A/B baselines + figures ---"
uv run python scripts/evo2/embed_and_geodesic_ortholog.py --from-sweep-layer "$LAYER" --n-perms "$PERMS"
RUN_DIR=$(ls -dt results/*_evo2-gene-families-blocks"${LAYER}" | head -1)
echo "Run dir: $RUN_DIR"
uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR"
uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source evo2 --n-perms 9999
uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source evo2
uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR"

echo ""
echo "--- [$(date)] Step 6: composition controls — within + between reconstruction (preservation) @ blocks.$LAYER ---"
# Build the control CDS sets if absent (idempotent), then re-embed @ blocks.$LAYER and score
# control-vs-natural at both axes:
#   within : $RUN_DIR/controls/control_within_scores.csv   (recovery vs taxonomy + rho_geodesic_vs_natural)
#   between: $RUN_DIR/controls/control_between_scores.csv   (centroid rho_vs_natural_centroid + vs Pfam-JSD)
if [ ! -d "$DATA_DIR/controls/dinuc_shuffle" ]; then
  uv run python analyses/make_control_sequences.py || echo "WARN: control-set build failed"
fi
uv run python analyses/embed_and_score_controls.py --control all --axis within --layer "$LAYER" --natural-run "$RUN_DIR" \
  || echo "WARN: ortholog within-control scoring failed"
uv run python analyses/embed_and_score_controls.py --control all --axis between --layer "$LAYER" --natural-run "$RUN_DIR" \
  || echo "WARN: ortholog between-control scoring failed"
uv run python analyses/control_comparison_figure.py --natural-run "$RUN_DIR" --layer "$LAYER" \
  || echo "WARN: control comparison figure failed"

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "Layer selection: $SEL_DIR   |   Results @ blocks.$LAYER: $RUN_DIR"
echo "========================================"
