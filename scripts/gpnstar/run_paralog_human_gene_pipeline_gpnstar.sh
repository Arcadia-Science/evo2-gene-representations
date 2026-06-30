#!/usr/bin/env bash
# GPN-Star matched-human gene pipeline.
#
# Runs the GPN-Star side of the matched human Panel-1 experiment as an independent
# job. The gene/locus set is shared with the Evo2 human runner through
# test_sample_human_genes.load_matched_panel(): both models read the same
# data/sampling/master_gene_table.tsv rows and the same transcript-span loci.
#
# Usage:
#   screen -dmS gpn_human bash scripts/gpnstar/run_paralog_human_gene_pipeline_gpnstar.sh
#
# Optional env vars:
#   GPN_MODEL=vertebrate   GPN-Star model variant
#   STANDARDIZE=zscore     Layer-selection standardization mode
#   PERMS=9999             Between-family Mantel permutations

set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs
LOG=logs/gpnstar_human_gene_pipeline.log
exec > >(tee -a "$LOG") 2>&1

GPN_MODEL="${GPN_MODEL:-vertebrate}"
STANDARDIZE="${STANDARDIZE:-zscore}"
PERMS="${PERMS:-9999}"

say() { echo ">>> [$(date)] $*"; }

score_suffix() {
  if [ "$STANDARDIZE" = "none" ]; then
    echo ""
  else
    echo "_$STANDARDIZE"
  fi
}

parse_layer() {
  uv run python -c 'import sys, pandas as pd; df = pd.read_csv(sys.argv[1]); print(int(df.loc[df["within_between_ratio"].idxmin(), "layer_idx"]))' "$1"
}

echo "############################################################"
echo "## GPN-Star matched human gene pipeline"
echo "## START $(date)  host=$(hostname)  pid=$$"
echo "############################################################"

say "uv sync"
uv sync || { say "ABORT: uv sync failed"; exit 1; }

say "Step 1: ensure GPN-Star MSA is present ($GPN_MODEL)"
uv run python scripts/gpnstar/download_msa.py "$GPN_MODEL" \
  || { say "ABORT: MSA setup failed"; exit 1; }

say "Step 2: dense human-panel sweep (all hidden states)"
uv run python scripts/gpnstar/embed_and_geodesic.py --model "$GPN_MODEL" --force-reembed \
  || { say "ABORT: GPN-Star human sweep failed"; exit 1; }

say "Step 3: layer selection (human panel, $STANDARDIZE)"
uv run python scripts/layer_selection/layer_selection.py \
  --model gpnstar --gpn-model "$GPN_MODEL" --panel human --standardize "$STANDARDIZE" \
  || { say "ABORT: layer selection failed"; exit 1; }

SUFFIX=$(score_suffix)
SEL_CSV=$(ls -t results/*_layer-selection/layer_scores_gpnstar_human"${SUFFIX}".csv | head -1) \
  || { say "ABORT: no GPN-Star human layer-score CSV found"; exit 1; }
[ -n "$SEL_CSV" ] || { say "ABORT: empty GPN-Star human layer-score CSV path"; exit 1; }
GPN_L=$(parse_layer "$SEL_CSV") || { say "ABORT: failed to parse selected GPN-Star layer"; exit 1; }
say "selected layer: GPN-Star hidden_state.$GPN_L (from $SEL_CSV)"

say "Step 4: run dir at selected layer"
uv run python scripts/gpnstar/embed_and_geodesic.py --model "$GPN_MODEL" --from-layer "$GPN_L" \
  || { say "ABORT: run-dir build failed"; exit 1; }
RUN_DIR=$(ls -dt results/*_gpnstar-human-panel-L"$GPN_L" | head -1) \
  || { say "ABORT: no GPN-Star human run dir found for layer $GPN_L"; exit 1; }
[ -n "$RUN_DIR" ] || { say "ABORT: empty GPN-Star human run-dir path"; exit 1; }
say "run dir: $RUN_DIR"

say "Step 5: prefetch matched-panel CDS for shared human baselines"
uv run python scripts/test_sample_human_genes.py prefetch-cds \
  || say "WARN: CDS prefetch incomplete; baselines will cover cached genes only"

say "Step 6: shared human baselines + figures"
uv run python scripts/baselines/sampled_cds_sequence_identity.py --run-dirs "$RUN_DIR" \
  || say "WARN: sampled CDS seq-id baseline failed"
uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source gpn \
  || say "WARN: protein-alignment baseline failed"
uv run python scripts/baselines/add_kmer_baseline.py --run-dir "$RUN_DIR" \
  || say "WARN: k-mer baseline failed"
uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR" \
  || say "WARN: Pfam-JSD baseline failed"
uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source gpn --n-perms "$PERMS" \
  || say "WARN: between-family baseline scoring failed"
uv run python scripts/controls/transcript_composition_control.py --run-dir "$RUN_DIR" --n-perms "$PERMS" \
  || say "WARN: transcript-composition control failed (needs data/cache/evo2_human_genomic.json from the Evo2 embedder)"
uv run python scripts/gpnstar/gpnstar_visualization.py --run-dir "$RUN_DIR" \
  || say "WARN: GPN-Star figures failed"
uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR" \
  --figures between-axes between-matrices convergent-ranks \
  || say "WARN: model-agnostic between-family figures failed"

say "Step 7: MSA composition/ablation controls — reconstruction (preservation) at hidden_state.$GPN_L"
# between: control centroid geodesic vs natural -> $RUN_DIR/msa_controls/msa_control_between_scores.csv
# within : per-family recovery (vs patristic) + preservation (vs natural) -> $RUN_DIR/controls/control_within_scores.csv
uv run python analyses/embed_and_score_msa_controls.py \
  --natural-run "$RUN_DIR" --model "$GPN_MODEL" --axis between --layer "$GPN_L" \
  || say "WARN: GPN-Star between-family controls failed"
uv run python analyses/embed_and_score_msa_controls.py \
  --natural-run "$RUN_DIR" --model "$GPN_MODEL" --axis within --layer "$GPN_L" \
  || say "WARN: GPN-Star within-family controls failed"

echo "############################################################"
echo "## GPN-Star human pipeline DONE $(date)"
echo "## Results: $RUN_DIR  (hidden_state.$GPN_L)"
echo "############################################################"
