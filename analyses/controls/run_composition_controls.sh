#!/usr/bin/env bash
# Composition controls — can composition-matched controls reproduce the gene-family geometry?
# Produces the three control panels (04, 12 and 13) into analyses/controls/figures/.
#
#   bash analyses/controls/run_composition_controls.sh            # print the plan, run nothing
#   bash analyses/controls/run_composition_controls.sh --figures  # re-render the panels (~3 min, CPU)
#   bash analyses/controls/run_composition_controls.sh --run      # full pipeline        (~160 h, GPU)
#
# Controls replace coding positions in place, preserving the transcript span and CDS mask; the
# ladder ranges from GC matching to protein-preserving recoding.
#
# This is an additional analysis over experiment 1's mammal panel, not one of the two publication
# experiments. See analyses/controls/README.md for what it measures and when to reach for it.
#
# REQUIRES EXPERIMENT 1 for --run: this reuses exp1's mammal panel. Stage C1 reads
# data/cache/mammal_cds_positions.json (exp1 A7, build_cds_masks_mammal.py) and stage D2 reads
# blocks15/betweenfam_ot_metadata.json (exp1 B3). Run exp1 --run first. --figures needs nothing
# but the tracked analyses/controls/figure_data/ tables.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-plan}"
case "$MODE" in
  --run) MODE=run ;;
  --figures) MODE=figures ;;
  plan|--plan|"") MODE=plan ;;
  -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
  *) echo "unknown option: $MODE (use --run, --figures, or nothing)"; exit 2 ;;
esac

PY="uv run --no-sync python"
RUN=results/2026-07-16_mammalian-orthologs-transcript_cdsmask
FD=analyses/controls/figure_data
OUT=analyses/controls/figures
source experiments/_helpers.sh
init_experiment controls
preflight_tools controls

echo "Composition controls (panels 04, 12, 13)"
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
# they cannot. missense_subset follows it as its nested partner: it edits ONLY bases the recode
# edited, so the two differ in whether the protein survives and not in how much sequence moved.
# It needs no ordering (the recode it nests inside is recreated deterministically per locus, not
# read from disk), but figure 13 cannot be built without it — the identity ladder has six rungs.
# Each rung is resumable; scoring upserts, so it can be run against a partial chain.
for C in synonymous_recode missense_subset gc_match dinuc_shuffle kmer4_shuffle kmer6_shuffle; do
  stage "~16-27 h, GPU" "C1. embed control rung: $C" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    $PY scripts/mammalian_orthologs/embed_cds_masked_mammal.py \
      --families $FAMS --control "$C"
done

# The two arms of figure 12. Both edit codon position 3 at the same rate; only one keeps the
# protein, so the comparison is two-sided and needs no residual-mismatch caveat. Read them against
# each other, never against synonymous_recode.
for ARM in paired_p3_syn paired_p3_missense; do
  stage "~19 h, GPU"  "C2. embed matched-pair arm: $ARM" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
    $PY scripts/mammalian_orthologs/embed_cds_masked_mammal.py \
      --families $FAMS --control "$ARM"
done

say "Scoring"

# Preservation rho is measured against the natural geometry, so a rung is scored only once its
# embedding cache is complete; partial rungs are skipped rather than scored on a subset.
stage "~2 h, CPU"   "D1. W2/angular control preservation and panel 12 source tables" \
  $PY scripts/mammalian_orthologs/mammal_controls_score.py --arm transcript_cdsmask
# OPTIONAL: an independent graph-free recomputation kept as a cross-check. Nothing reads its
# output; figure 4 comes from D1. Skipping D2 changes no figure.
stage "~2 h, CPU"   "D2. control preservation, graph-free (optional cross-check)" \
  $PY scripts/mammalian_orthologs/controls_score_graphfree.py --axis both

# Figure 13's question is whether the rho above is just retained source nucleotides, so it reads
# the same tables and is measured on the same loci. No model, no GPU.
stage "~15 min, CPU" "D3. nucleotide identity of each control to its source gene" \
  $PY scripts/controls/control_sequence_identity.py --stage identity

stage "~1 min, CPU"  "D4. refresh this analysis's figure_data/ tables" \
  $PY scripts/build_figure_data.py --experiments controls
fi

say "Panels 04, 12 and 13"

if need "panel 04: controls vs natural by layer" $FD/control_preservation.csv; then
  fig "panel 04: controls vs natural by layer" $PY scripts/controls/plot_control_wasserstein.py --pub
  collect "results/layer_sweep_summaries/pub/controls_layer_summary_mammalian-orthologs-cdsmask-48fam-wasserstein" \
          fig04_controls_vs_natural_by_layer
fi

if need "panel 12: paired p3, protein vs nucleotide" $FD/control_preservation.csv; then
  fig "panel 12: paired p3, protein vs nucleotide" \
    $PY scripts/mammalian_orthologs/paired_p3_figure.py --pub
  collect "$RUN/pub/paired_p3_protein_vs_nucleotide" fig12_paired_p3_protein_vs_nucleotide
fi

if need "panel 13: control identity vs rho" $FD/control_identity.csv \
                                          $FD/control_identity_by_family.csv; then
  fig "panel 13: control identity vs rho" \
    $PY scripts/controls/control_sequence_identity.py --stage figure --pub
  collect "$RUN/pub/control_identity_vs_rho" fig13_control_identity_vs_rho
fi

if [ "$MODE" = plan ]; then
  echo; echo "══ dry run complete. --figures to re-render, --run for the whole analysis."
  exit 0
fi

finish_figures \
  fig04_controls_vs_natural_by_layer \
  fig12_paired_p3_protein_vs_nucleotide \
  fig13_control_identity_vs_rho
