#!/usr/bin/env bash
# Re-run the full Evo2 pipeline analysis at EVERY block, reusing the dense-sweep
# embedding cache (NO GPU / NO re-embedding). One run dir per layer.
#
# WHY THIS IS CHEAP
#   1. The expensive step — pushing sequences through Evo2 — is already done: the
#      dense sweep captured all 32 blocks in one forward pass and cached the stack.
#      Each layer is sliced from that cache (verified byte-identical to a fresh embed).
#   2. The ground-truth baseline DISTANCE matrices (patristic, k-mer, Pfam-JSD,
#      taxonomy, seq-id, the between-family axes) are sequence/annotation-derived and
#      therefore IDENTICAL at every layer. They are computed ONCE on a donor layer;
#      every other layer reuses them (--distances-from) and only recomputes the
#      layer-dependent parts: the geodesic, the Spearman/Mantel scores, and their
#      permutation significance. After the donor, each layer is seconds of CPU.
#
# Usage:
#   PANEL=human         bash scripts/evo2/sweep_all_layers_evo2.sh   # 32 blocks, matched human panel
#   PANEL=gene-families bash scripts/evo2/sweep_all_layers_evo2.sh   # 32 blocks, cross-kingdom families
#   screen -dmS evo2_sweep env PANEL=gene-families bash scripts/evo2/sweep_all_layers_evo2.sh
#
# Env vars:
#   PANEL=human|gene-families   Which experiment (REQUIRED).
#   LAYERS="0 1 2 ..."          Blocks to run (default: all in the cache). FIRST = donor.
#   PERMS=999                   Mantel / within-between permutation count (sweep-grade).
#   SKIP_EXISTING=1             Skip a layer whose run dir already has its scores.
#
# Prereqs (already on disk from the production runs):
#   PANEL=human         -> data/cache/evo2_human_layer_sweep/layer_stack.npy  (32, 580, 4096)
#   PANEL=gene-families -> data/cache/evo2_layer_sweep/layer_stack.npy        (32, 5276, 4096)
# If a cache is missing, build it once with the matching embedder's --force-reembed (GPU).

set -uo pipefail
cd "$(dirname "$0")/../.."

PANEL="${PANEL:?set PANEL=human or PANEL=gene-families}"
PERMS="${PERMS:-999}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DATE=$(date +%F)

mkdir -p logs
LOG="logs/evo2_layer_sweep_${PANEL}.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo ">>> [$(date)] $*"; }

case "$PANEL" in
  human)         CACHE="data/cache/evo2_human_layer_sweep/layer_stack.npy" ;;
  gene-families) CACHE="data/cache/evo2_layer_sweep/layer_stack.npy" ;;
  *) echo "PANEL must be 'human' or 'gene-families'"; exit 2 ;;
esac
[ -f "$CACHE" ] || { say "ABORT: missing embedding cache $CACHE (build once with --force-reembed on GPU)"; exit 1; }

# Layer count = first axis of the cached stack (32 blocks for Evo2-7B).
N_LAYERS=$(uv run python -c "import numpy as np,sys; print(np.load(sys.argv[1],mmap_mode='r').shape[0])" "$CACHE")
LAYERS="${LAYERS:-$(seq 0 $((N_LAYERS - 1)))}"

run_dir_for() { echo "results/${DATE}_evo2-${1}-blocks${2}"; }   # $1=panel-tag $2=layer
[ "$PANEL" = human ] && TAG=human-panel || TAG=gene-families

echo "############################################################"
echo "## Evo2 all-layer sweep  panel=$PANEL  layers=[$LAYERS]  perms=$PERMS"
echo "## START $(date)  host=$(hostname)  pid=$$   cache=$CACHE"
echo "############################################################"

say "uv sync"; uv sync || { say "ABORT: uv sync failed"; exit 1; }

if [ "$PANEL" = human ]; then
  say "Prefetch matched-panel CDS for shared human baselines (once)"
  uv run python scripts/test_sample_human_genes.py prefetch-cds \
    || say "WARN: CDS prefetch incomplete; baselines cover cached genes only"
fi

# ── per-layer worker ────────────────────────────────────────────────────────
# $1 = layer index   $2 = donor run dir ("" if THIS layer is the donor → compute)
build_layer() {
  local L="$1" DONOR="$2"
  local RUN_DIR; RUN_DIR=$(run_dir_for "$TAG" "$L")
  local DFROM=(); [ -n "$DONOR" ] && DFROM=(--distances-from "$DONOR")
  local mode; [ -n "$DONOR" ] && mode="reuse <- $(basename "$DONOR")" || mode="DONOR (compute distances once)"

  echo; echo "============================================================"
  say "LAYER blocks.$L  ($PANEL)  [$mode]"
  echo "============================================================"

  if [ "$SKIP_EXISTING" = 1 ] && [ -f "$RUN_DIR/between_family_baseline_scores.csv" ]; then
    say "skip (exists): $RUN_DIR"; return 0
  fi

  if [ "$PANEL" = human ]; then
    # geodesic + centroid from the cached stack (no GPU; production adaptive-k graph)
    uv run python scripts/evo2/embed_and_geodesic_paralog.py --from-layer "$L" \
      || { say "WARN: run-dir build failed at blocks.$L"; return 1; }

    uv run python scripts/baselines/sampled_cds_sequence_identity.py --run-dirs "$RUN_DIR" "${DFROM[@]}" \
      || say "WARN: sampled CDS seq-id failed (blocks.$L)"
    # patristic self-caches its MAFFT/FastTree matrices (data/cache/evo2_patristic) — reuses across layers
    uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source gpn \
      || say "WARN: protein-alignment patristic failed (blocks.$L)"
    uv run python scripts/baselines/add_kmer_baseline.py --run-dir "$RUN_DIR" "${DFROM[@]}" \
      || say "WARN: k-mer baseline failed (blocks.$L)"
    uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR" "${DFROM[@]}" \
      || say "WARN: Pfam-JSD failed (blocks.$L)"
    uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source gpn --n-perms "$PERMS" "${DFROM[@]}" \
      || say "WARN: between-family scoring failed (blocks.$L)"
    # cheap 580-gene transcript k-mer/GC matrix (no GPU); recomputed per layer by design
    uv run python scripts/controls/transcript_composition_control.py --run-dir "$RUN_DIR" --n-perms "$PERMS" \
      || say "WARN: transcript-composition control failed (blocks.$L)"
    uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR" \
      || say "WARN: figures failed (blocks.$L)"

  else  # gene-families (Evo2-only; --seq-source evo2)
    # embedder builds geodesic + Axis-A/B; reuse the 107 MB k-mer/taxonomy matrices from the donor
    uv run python scripts/evo2/embed_and_geodesic_ortholog.py --from-sweep-layer "$L" --n-perms "$PERMS" "${DFROM[@]}" \
      || { say "WARN: run-dir build failed at blocks.$L"; return 1; }

    uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR" "${DFROM[@]}" \
      || say "WARN: Pfam-JSD failed (blocks.$L)"
    uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source evo2 --n-perms "$PERMS" "${DFROM[@]}" \
      || say "WARN: between-family scoring failed (blocks.$L)"
    uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source evo2 \
      || say "WARN: protein-alignment patristic failed (blocks.$L)"
    uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR" \
      || say "WARN: figures failed (blocks.$L)"

    # gzip the ~500 MB labeled geodesic CSV — LAST, after every consumer above read it.
    if [ -f "$RUN_DIR/evo2_gene_family_geodesic_labeled.csv" ]; then
      gzip -f "$RUN_DIR/evo2_gene_family_geodesic_labeled.csv" && say "gzipped labeled geodesic CSV (blocks.$L)"
    fi
  fi
  say "done blocks.$L -> $RUN_DIR"
}

# ── Phase 1: donor (first layer) computes every distance matrix once ──────────
DONOR_LAYER=$(echo "$LAYERS" | tr ' ' '\n' | head -1)
REST=$(echo "$LAYERS" | tr ' ' '\n' | tail -n +2)
say "Phase 1 — donor layer blocks.$DONOR_LAYER (computes layer-independent distances)"
build_layer "$DONOR_LAYER" "" || { say "ABORT: donor layer failed; cannot seed --distances-from"; exit 1; }
DONOR_DIR=$(run_dir_for "$TAG" "$DONOR_LAYER")

# ── Phase 2: remaining layers reuse the donor's distances, re-score only ──────
say "Phase 2 — remaining layers reuse $DONOR_DIR"
for L in $REST; do
  build_layer "$L" "$DONOR_DIR"
done

echo "############################################################"
echo "## Evo2 all-layer sweep DONE $(date)  panel=$PANEL   donor=$DONOR_DIR"
echo "############################################################"
