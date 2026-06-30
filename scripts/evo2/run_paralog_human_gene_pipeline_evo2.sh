#!/usr/bin/env bash
# Evo2 matched-human gene pipeline.
#
# Runs the Evo2 side of the matched human Panel-1 experiment as an independent
# job. The gene/locus set is shared with the GPN-Star human runner through
# test_sample_human_genes.load_matched_panel(): both models read the same
# data/sampling/master_gene_table.tsv rows and the same transcript-span loci.
#
# Usage:
#   screen -dmS evo2_human bash scripts/evo2/run_paralog_human_gene_pipeline_evo2.sh
#
# Optional env vars:
#   STANDARDIZE=zscore   Layer-selection standardization mode
#   PERMS=9999           Between-family Mantel permutations

set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs
LOG=logs/evo2_human_gene_pipeline.log
exec > >(tee -a "$LOG") 2>&1

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
echo "## Evo2 matched human gene pipeline"
echo "## START $(date)  host=$(hostname)  pid=$$"
echo "############################################################"

say "uv sync"
uv sync

say "Step 1: dense human-panel sweep (all Evo2 blocks, genomic transcript spans)"
uv run python scripts/evo2/embed_and_geodesic_paralog.py --force-reembed \
  || { say "ABORT: Evo2 human sweep failed"; exit 1; }

say "Step 2: layer selection (human panel, $STANDARDIZE)"
uv run python scripts/layer_selection/layer_selection.py \
  --model evo2 --panel human --standardize "$STANDARDIZE" \
  || { say "ABORT: layer selection failed"; exit 1; }

SUFFIX=$(score_suffix)
SEL_CSV=$(ls -t results/*_layer-selection/layer_scores_evo2_human"${SUFFIX}".csv | head -1) \
  || { say "ABORT: no Evo2 human layer-score CSV found"; exit 1; }
[ -n "$SEL_CSV" ] || { say "ABORT: empty Evo2 human layer-score CSV path"; exit 1; }
EVO_L=$(parse_layer "$SEL_CSV") || { say "ABORT: failed to parse selected Evo2 layer"; exit 1; }
say "selected layer: Evo2 blocks.$EVO_L (from $SEL_CSV)"

say "Step 3: run dir at selected layer"
uv run python scripts/evo2/embed_and_geodesic_paralog.py --from-layer "$EVO_L" \
  || { say "ABORT: run-dir build failed"; exit 1; }
RUN_DIR=$(ls -dt results/*_evo2-human-panel-blocks"$EVO_L" | head -1) \
  || { say "ABORT: no Evo2 human run dir found for layer $EVO_L"; exit 1; }
[ -n "$RUN_DIR" ] || { say "ABORT: empty Evo2 human run-dir path"; exit 1; }
say "run dir: $RUN_DIR"

say "Step 4: prefetch matched-panel CDS for shared human baselines"
uv run python scripts/test_sample_human_genes.py prefetch-cds \
  || say "WARN: CDS prefetch incomplete; baselines will cover cached genes only"

say "Step 5: shared human baselines + figures"
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
  || say "WARN: transcript-composition control failed"
uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR" \
  || say "WARN: Evo2 within/between figures failed"

say "Step 6: composition controls — re-embed genomic shuffles + reconstruction (preservation) @ blocks.$EVO_L"
# Each control is a full re-embed run dir; then score control-vs-natural at both axes ->
#   within : $RUN_DIR/controls/control_within_scores.csv   (per-family rho_geodesic_vs_natural)
#   between: $RUN_DIR/controls/control_between_scores.csv   (centroid rho_vs_natural_centroid)
for C in gc_match dinuc_shuffle kmer6_shuffle; do
  say "  control re-embed: $C"
  uv run python scripts/evo2/embed_and_geodesic_paralog.py --control "$C" --from-layer "$EVO_L" \
    || say "WARN: control embed $C failed"
done
uv run python analyses/embed_and_score_controls.py \
  --control all --panel human --natural-run "$RUN_DIR" --layer "$EVO_L" \
  || say "WARN: human control preservation scoring failed"

echo "############################################################"
echo "## Evo2 human pipeline DONE $(date)"
echo "## Results: $RUN_DIR  (blocks.$EVO_L)"
echo "############################################################"
