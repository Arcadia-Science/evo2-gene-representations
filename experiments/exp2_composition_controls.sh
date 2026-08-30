#!/usr/bin/env bash
# Experiment 2 — can composition-matched controls reproduce the gene-family geometry?
# Produces publication figures 4 and 12.
#
#   bash experiments/exp2_composition_controls.sh            # print the plan, run nothing
#   bash experiments/exp2_composition_controls.sh --figures  # re-render figures 4, 12  (~3 min, CPU)
#   bash experiments/exp2_composition_controls.sh --run      # full pipeline           (~135 h, GPU)
#
# Uses the experiment-1 mammal panel. Controls replace coding positions in place, preserving the
# transcript span and CDS mask; the ladder ranges from GC matching to protein-preserving recoding.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-plan}"
case "$MODE" in
  --run) MODE=run ;;
  --figures) MODE=figures ;;
  plan|--plan|"") MODE=plan ;;
  -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
  *) echo "unknown option: $MODE (use --run, --figures, or nothing)"; exit 2 ;;
esac

PY="uv run --no-sync python"
RUN=results/2026-07-16_mammalian-orthologs-transcript_cdsmask
OUT=pub/figures
source experiments/_helpers.sh
init_experiment exp2

echo "Experiment 2 — composition controls (figures 4, 12)"
[ "$MODE" = plan ]    && echo "DRY RUN — nothing will execute."
[ "$MODE" = figures ] && echo "FIGURES ONLY — re-rendering from artifacts on disk."

if [ "$MODE" != figures ]; then
say "Control embedding"

FAMS='<families from complete_manifest.csv>'
if [ "$MODE" = run ]; then
FAMS=$($PY -c "
import pandas as pd; print(' '.join(sorted(pd.read_csv('data/mammalian_orthologs/complete_manifest.csv').family.unique())))" 2>/dev/null) \
  || FAMS='<families from complete_manifest.csv>'
fi

# synonymous_recode runs first because it is the rung that separates "coding composition" from
# "protein" as the carrier of the geometry — the other four destroy the protein, so on their own
# they cannot. Each rung is resumable; scoring upserts, so it can be run against a partial chain.
for C in synonymous_recode gc_match dinuc_shuffle kmer4_shuffle kmer6_shuffle; do
  stage "~16-27 h, GPU" "C1. embed control rung: $C" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    $PY scripts/mammalian_orthologs/embed_cds_masked_mammal.py \
      --families $FAMS --mirror-arm cds --control "$C"
done

# The two arms of figure 12. Both edit codon position 3 at the same rate; only one keeps the
# protein, so the comparison is two-sided and needs no residual-mismatch caveat. Read them against
# each other, never against synonymous_recode.
for ARM in paired_p3_syn paired_p3_missense; do
  stage "~19 h, GPU"  "C2. embed matched-pair arm: $ARM" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    $PY scripts/mammalian_orthologs/embed_cds_masked_mammal.py \
      --families $FAMS --mirror-arm cds --control "$ARM"
done

say "Scoring"

# Preservation rho is measured against the natural geometry, so a rung is scored only once its
# embedding cache is complete; partial rungs are skipped rather than scored on a subset.
stage "~4 h, CPU"   "D1. control preservation, graph-based (within geodesic + between centroid)" \
  $PY scripts/mammalian_orthologs/mammal_controls_score.py --arm transcript_cdsmask
stage "~2 h, CPU"   "D2. control preservation, graph-free — the input to figure 4" \
  $PY scripts/mammalian_orthologs/controls_score_graphfree.py --axis both
stage "~30 min"     "D3. how much source sequence each control retains (identity confound check)" \
  $PY scripts/controls/control_sequence_identity.py --panel mammal_cdsmask
fi

say "Figures 4 and 12"

if need "fig 4: controls vs natural by layer" \
        results/_ot_control_preservation_transcript_cdsmask.csv \
        results/_angular_control_preservation_transcript_cdsmask.csv; then
  fig "fig 4: controls vs natural by layer" $PY scripts/controls/plot_control_wasserstein.py --pub
  collect "results/layer_sweep_summaries/pub/controls_layer_summary_mammalian-orthologs-cdsmask-48fam-wasserstein" \
          fig04_controls_vs_natural_by_layer
fi

if need "fig 12: paired p3, protein vs nucleotide" \
        "$RUN/blocks*/controls/control_between_scores.csv" \
        "$RUN/blocks*/controls/control_within_scores.csv" \
        "$RUN/control_rho_by_layer.csv"; then
  fig "fig 12: paired p3, protein vs nucleotide" \
    $PY scripts/mammalian_orthologs/paired_p3_figure.py --skip-identity --pub
  collect "$RUN/pub/paired_p3_protein_vs_nucleotide" fig12_paired_p3_protein_vs_nucleotide
fi

if [ "$MODE" = plan ]; then
  echo; echo "══ dry run complete. --figures to re-render, --run for the whole experiment."
  exit 0
fi

finish_figures \
  fig04_controls_vs_natural_by_layer \
  fig12_paired_p3_protein_vs_nucleotide
