#!/usr/bin/env bash
# Re-run the full GPN-Star pipeline analysis at EVERY hidden state, reusing the
# dense-sweep embedding cache (NO GPU / NO re-embedding). One run dir per layer.
#
# WHY THIS IS CHEAP — same logic as the Evo2 sweep:
#   1. GPN-Star returns all (num_hidden_layers + 1) states in one forward pass, so the
#      dense human-panel sweep already cached every layer. Each layer is sliced from
#      that cache (verified byte-identical to a fresh embed).
#   2. The ground-truth baseline DISTANCE matrices are sequence/annotation-derived and
#      IDENTICAL at every layer: computed ONCE on a donor layer, reused everywhere else
#      (--distances-from). Only the geodesic, the Spearman/Mantel scores, and their
#      permutation significance are recomputed per layer.
#
# Usage:
#   bash scripts/gpnstar/sweep_all_layers_gpnstar.sh
#   screen -dmS gpn_sweep bash scripts/gpnstar/sweep_all_layers_gpnstar.sh
#
# Env vars:
#   GPN_MODEL=vertebrate   GPN-Star model variant (matches the cached stack).
#   LAYERS="0 1 2 ..."     Hidden states to run (default: all in the cache). FIRST = donor.
#   PERMS=999              Mantel / within-between permutation count (sweep-grade).
#   SKIP_EXISTING=1        Skip a layer whose run dir already has its scores.
#
# Prereq (on disk from the production run):
#   data/cache/gpnstar_human_layer_sweep/layer_stack_<model>.npy   (17, 580, 1024)
# If missing, build once: scripts/gpnstar/embed_and_geodesic.py --model <m> --force-reembed (GPU).
# The Evo2-human transcript-composition control needs data/cache/evo2_human_genomic.json
# (produced by the Evo2-human embedder) — run the Evo2 human sweep/pipeline first.

set -uo pipefail
cd "$(dirname "$0")/../.."

GPN_MODEL="${GPN_MODEL:-vertebrate}"
PERMS="${PERMS:-999}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DATE=$(date +%F)

mkdir -p logs
LOG="logs/gpnstar_layer_sweep_${GPN_MODEL}.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo ">>> [$(date)] $*"; }

CACHE="data/cache/gpnstar_human_layer_sweep/layer_stack_${GPN_MODEL}.npy"
[ -f "$CACHE" ] || { say "ABORT: missing embedding cache $CACHE (build once with --force-reembed on GPU)"; exit 1; }

N_LAYERS=$(uv run python -c "import numpy as np,sys; print(np.load(sys.argv[1],mmap_mode='r').shape[0])" "$CACHE")
LAYERS="${LAYERS:-$(seq 0 $((N_LAYERS - 1)))}"

run_dir_for() { echo "results/${DATE}_gpnstar-human-panel-L${1}"; }

echo "############################################################"
echo "## GPN-Star all-layer sweep  model=$GPN_MODEL  layers=[$LAYERS]  perms=$PERMS"
echo "## START $(date)  host=$(hostname)  pid=$$   cache=$CACHE"
echo "############################################################"

say "uv sync"; uv sync || { say "ABORT: uv sync failed"; exit 1; }

say "Prefetch matched-panel CDS for shared human baselines (once)"
uv run python scripts/test_sample_human_genes.py prefetch-cds \
  || say "WARN: CDS prefetch incomplete; baselines cover cached genes only"

# ── per-layer worker ────────────────────────────────────────────────────────
# $1 = hidden-state index   $2 = donor run dir ("" if THIS layer is the donor)
build_layer() {
  local L="$1" DONOR="$2"
  local RUN_DIR; RUN_DIR=$(run_dir_for "$L")
  local DFROM=(); [ -n "$DONOR" ] && DFROM=(--distances-from "$DONOR")
  local mode; [ -n "$DONOR" ] && mode="reuse <- $(basename "$DONOR")" || mode="DONOR (compute distances once)"

  echo; echo "============================================================"
  say "LAYER hidden_state.$L  (model=$GPN_MODEL)  [$mode]"
  echo "============================================================"

  if [ "$SKIP_EXISTING" = 1 ] && [ -f "$RUN_DIR/between_family_baseline_scores.csv" ]; then
    say "skip (exists): $RUN_DIR"; return 0
  fi

  # geodesic + centroid from the cached stack (no GPU; production adaptive-k graph)
  uv run python scripts/gpnstar/embed_and_geodesic.py --model "$GPN_MODEL" --from-layer "$L" \
    || { say "WARN: run-dir build failed at hidden_state.$L"; return 1; }

  uv run python scripts/baselines/sampled_cds_sequence_identity.py --run-dirs "$RUN_DIR" "${DFROM[@]}" \
    || say "WARN: sampled CDS seq-id failed (L$L)"
  # patristic self-caches its MAFFT/FastTree matrices (data/cache/gpnstar_patristic) — reuses across layers
  uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source gpn \
    || say "WARN: protein-alignment patristic failed (L$L)"
  uv run python scripts/baselines/add_kmer_baseline.py --run-dir "$RUN_DIR" "${DFROM[@]}" \
    || say "WARN: k-mer baseline failed (L$L)"
  uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR" "${DFROM[@]}" \
    || say "WARN: Pfam-JSD failed (L$L)"
  uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source gpn --n-perms "$PERMS" "${DFROM[@]}" \
    || say "WARN: between-family scoring failed (L$L)"
  uv run python scripts/controls/transcript_composition_control.py --run-dir "$RUN_DIR" --n-perms "$PERMS" \
    || say "WARN: transcript-composition control failed (L$L; needs data/cache/evo2_human_genomic.json)"
  uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir "$RUN_DIR" \
    || say "WARN: GPN-Star figures failed (L$L)"
  uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR" \
    --figures between-axes between-matrices convergent-ranks \
    || say "WARN: model-agnostic between-family figures failed (L$L)"

  say "done hidden_state.$L -> $RUN_DIR"
}

# ── Phase 1: donor (first layer) computes every distance matrix once ──────────
DONOR_LAYER=$(echo "$LAYERS" | tr ' ' '\n' | head -1)
REST=$(echo "$LAYERS" | tr ' ' '\n' | tail -n +2)
say "Phase 1 — donor layer hidden_state.$DONOR_LAYER (computes layer-independent distances)"
build_layer "$DONOR_LAYER" "" || { say "ABORT: donor layer failed; cannot seed --distances-from"; exit 1; }
DONOR_DIR=$(run_dir_for "$DONOR_LAYER")

# ── Phase 2: remaining layers reuse the donor's distances, re-score only ──────
say "Phase 2 — remaining layers reuse $DONOR_DIR"
for L in $REST; do
  build_layer "$L" "$DONOR_DIR"
done

echo "############################################################"
echo "## GPN-Star all-layer sweep DONE $(date)  model=$GPN_MODEL   donor=$DONOR_DIR"
echo "############################################################"
