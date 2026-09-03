#!/usr/bin/env bash
# Experiment 2 — does steering toward platypus causally change what Evo2 generates?
# Produces publication figures 5 through 11.
#
#   bash experiments/exp2_platypus_steering.sh            # print the plan, run nothing
#   bash experiments/exp2_platypus_steering.sh --figures  # re-render figures 5-11  (~10 min, CPU)
#   bash experiments/exp2_platypus_steering.sh --run      # full pipeline           (~60 h, GPU)
#
# Uses 400 human/platypus ortholog pairs stratified by protein identity. Held-out mean directions
# are injected at block 27 across the configured dose ladder.
#
# REQUIRES EXPERIMENT 1 for --run: stages E1 and E2 read
# data/mammalian_orthologs/tree/species_tree.nwk, written by exp1 stage A6
# (build_species_tree.py). Run exp1 --run first, or at least that stage. --figures needs nothing
# but the tracked figure_data/ tables.
set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-plan}"
case "$MODE" in
  --run) MODE=run ;;
  --figures) MODE=figures ;;
  plan|--plan|"") MODE=plan ;;
  -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
  *) echo "unknown option: $MODE (use --run, --figures, or nothing)"; exit 2 ;;
esac

PY="uv run --no-sync python"
R=results/2026-08-08_platypus-strat-400
S=scripts/steering/platypus
ARM=stage4_cds_mean_blocks27
OUT=pub/figures
source experiments/_helpers.sh
init_experiment exp2
preflight_tools exp2

echo "Experiment 2 — platypus steering (figures 5-11)"
[ "$MODE" = plan ]    && echo "DRY RUN — nothing will execute."
[ "$MODE" = figures ] && echo "FIGURES ONLY — re-rendering from artifacts on disk."

if [ "$MODE" != figures ]; then
say "Panel and reference data"

stage "~1 h, CPU"   "A1. block-disjoint, conservation-stratified candidate pool" \
  $PY $S/strat/stage0_pool.py --out $R/stage0
stage "~30 min"     "A2. QC, shifted-window prompt rule, 80 pairs per stratum" \
  $PY $S/strat/stage1_qc.py --stage0 $R/stage0 --out $R/stage1 --per-stratum 80 --max-start-codon 30

# The 24-mammal ortholog CDS that defines which platypus bases count as "private". Figures 7-10 are
# scored against it, so it must exist before stage 4 is rescored (C6). It reads the panel and the
# platypus CDS that A2 writes, so it cannot run before A2 -- it used to be ordered first, which
# made a full --run fail on its very first data stage.
stage "~2 h, API"   "A3. 24-mammal ortholog CDS for the private-base reference" \
  $PY $S/strat/private_site_orthologs.py --run $R

say "Direction geometry"

stage "~2 h, GPU"   "B1. embed 400 x 2 species x 32 blocks" \
  $PY $S/strat/stage2_embed.py --stage1 $R/stage1 --out $R/stage2
stage "~20 min"     "B2. per-gene direction statistics (leave-one-out cosine, magnitude spread)" \
  $PY $S/delta_stats.py --stage1-dir $R/stage1 --pooled \
     --pooled-npz $R/stage2/pooled_representations.npz --representation cds_mean \
     --out-dir $R/geom_cds_mean

# All 32 layers so the layer profile is complete; injection is still block 27 only.
stage "~30 min"     "B3. leave-one-gene-out steering vectors, all layers" \
  $PY $S/stage3_select.py --stage1-dir $R/stage1 --sweep-dir $R/stage2 \
     --out-dir $R/stage3_cds_mean --representation cds_mean \
     --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
stage "~10 min"     "B4. direction-residual clusters" \
  $PY $S/strat/h1c_clusters.py --run $R --layer 27 --mode cds_mean
stage "~1 min"      "B5. pre-registered gate checks" \
  $PY $S/strat/stage3_gates.py --run $R --layer 27 --mode cds_mean --band 24 27

say "Generation and scoring"

# Every arm accumulates into one directory so --resume dedupes on (gene, condition) and the
# unsteered cells are generated exactly once. No arm selects genes on an outcome, so sharing the
# baseline is safe: comparisons are always within gene, at the same sites.
BASE=($PY $S/strat/stage4_steer.py --run $R --representation cds_mean --layers blocks.27 --out $R/$ARM --resume)
stage "~13 h, GPU"  "C1. primary arms at alpha 1" \
  "${BASE[@]}" --arms unsteered add add_own random --alphas 1.0
stage "~7 h, GPU"   "C2. dose ladder, alpha 0.5 and 2" \
  "${BASE[@]}" --arms add random --alphas 0.5 2.0
stage "~7 h, GPU"   "C3. dose extension, alpha 3 and 4, with their matched random nulls" \
  "${BASE[@]}" --arms add random --alphas 3.0 4.0
stage "~10 h, GPU"  "C4. confound arms: cone-removed, GC-removed, cross-gene" \
  "${BASE[@]}" --arms cone_removed gc_removed cross_gene --alphas 1.0

# The rescore comes first: it writes the private-base and platy-bp columns, and every analysis
# below defaults to the private-base metric. Ordered the other way, C6 exits on the missing column
# and _helpers.sh stops the chain -- and no --metric setting avoids it, since the platy-bp column
# is written by this same rescore. Only the deprecated `legacy` metric exists beforehand.
stage "~1 h, CPU"   "C5. rescore on the nucleotide alignment (private-base metric)" \
  $PY $S/strat/stage4_rescore_nt.py --run $R --dir $ARM
stage "~20 min"     "C6. stage-4 analysis" \
  $PY $S/strat/stage4_analysis.py --run $R --dir $ARM
stage "~2 h, CPU"   "C7. leave-human vs platypus-choice decomposition at private sites" \
  $PY $S/strat/site_directionality.py --run $R --dir $ARM

say "Evolutionary rates (CPU-only; independent of stages 2-4)"

stage "~4 h, CPU"   "E1. 1:1 orthologs across the 24-mammal topology" \
  $PY $S/strat/stage5_orthologs.py --run $R
stage "~6 h, CPU"   "E2. branch lengths on a fixed species topology" \
  $PY $S/strat/stage5_trees.py --run $R --jobs 12
stage "~6 h, CPU"   "E3. dN, dS and omega (PAML codeml)" \
  $PY $S/strat/stage5_dnds.py --run $R --jobs 10 --models yn m0 m2
stage "~10 min"     "E4. merge rates with the direction statistics" \
  $PY $S/strat/stage5_merge.py --run $R --layers 27 --modes cds_mean
stage "~10 min"     "E5. does evolutionary rate predict steering gain?" \
  $PY $S/strat/stage5_gain.py --run $R --arms $ARM

say "Tracked figure inputs"

# Figure 10's two source tables are built here, not as a side effect of rendering. They used to
# appear only when the now-archived cross-layer figures were run by hand, so a clean --run reached
# F2 with them missing.
stage "~5 min, CPU"  "F1. composition and codon tables for figure 10" \
  $PY $S/strat/gc_codon_figures.py --run $R --dir $ARM --emit-tables

# Collapses the run dirs above into figure_data/, which is what every figure below reads.
stage "~1 min, CPU"  "F2. build this experiment's figure_data/ tables" \
  $PY scripts/build_figure_data.py --experiments exp2
fi

say "Figures 5-11"

FD=figure_data
if need "figs 5 + 11: strata composition, rate matrix" \
        $FD/exp2_panel.csv $FD/exp2_panel_coverage.csv $FD/exp2_rates_tree_stats.csv \
        $FD/exp2_rates_dnds.csv $FD/exp2_rates_vs_direction.csv $FD/exp2_rates_vs_gain.csv \
        $FD/exp2_direction_per_gene_by_layer.csv $FD/exp2_steering_outcomes.csv; then
  fig "figs 5 + 11: strata composition, rate matrix" \
    $PY $S/strat/hypothesis_figures.py --run "$R" --from-figure-data --only 5 8 --pub
  collect "$R/figures/pub/8b_strata_composition_frame" fig05_stratum_composition
  collect "$R/figures/pub/5_rate_outcome_matrix"       fig11_rate_predictor_outcome_matrix
fi

# Figures 6a/6b came from the ~100-gene paired panel until 2026-09-01; they now use the same
# n=400 stratified panel as every other figure here. The retired renders are kept as "... (OLD)".
if need "figs 6a/6b: LOO cosine, magnitude spread" \
        $FD/exp2_direction_layer_stats.csv $FD/exp2_direction_per_gene_by_layer.csv \
        $FD/exp2_direction_nulls.npz; then
  fig "figs 6a/6b: LOO cosine, magnitude spread" $PY $S/figures.py \
    --from-figure-data --out-dir "$R/geom_cds_mean/figures" --structure-suffix _cds_mean \
    --label 'CDS mean' --pub
  collect "$R/geom_cds_mean/figures/pub/1_loo_median_by_layer" \
          fig06a_leave_one_out_cosine_by_block
  collect "$R/geom_cds_mean/figures/pub/8b_magnitude_spread" \
          fig06b_delta_magnitude_spread_by_block
fi

if need "fig 7: steering delta by stratum" $FD/exp2_steering_outcomes.csv; then
  fig "fig 7: steering delta by stratum" $PY $S/strat/steering_delta_strip.py \
    --run "$R" --arm-dir $ARM --from-figure-data --style violin-strata \
    --stem 10_steering_delta_violin_strata --pub
  collect "$R/figures/pub/10_steering_delta_violin_strata" fig07_steering_delta_by_stratum
fi

if need "fig 8: dose response by stratum" $FD/exp2_steering_outcomes.csv; then
  fig "fig 8: dose response by stratum" $PY $S/strat/dose_and_alpha_figures.py \
    --run "$R" --dir $ARM --from-figure-data --pub
  collect "$R/figures/pub/6d_dose_by_stratum" fig08_dose_response_by_stratum
fi

if need "figs 9a/9b: leave-human vs platypus choice" \
        $FD/exp2_site_directionality_summary.csv $FD/exp2_site_directionality_gc_class.csv \
        $FD/exp2_site_directionality_per_gene.csv; then
  fig "figs 9a/9b: leave-human vs platypus choice" $PY $S/strat/site_directionality_figures.py \
    --run "$R" --from-figure-data --layers 27 --site-set both --pub
  collect "$R/figures_27/pub/18_leave_human_vs_platypus_choice" \
          fig09a_leave_human_vs_platypus_choice_private
  collect "$R/figures_27/pub/18_leave_human_vs_platypus_choice_platy_not_human" \
          fig09b_leave_human_vs_platypus_choice_loose
fi

if need "fig 10: GC by codon position" \
        $FD/exp2_generation_composition.csv $FD/exp2_generation_reference_windows.csv \
        $FD/exp2_codon_substitutions.csv; then
  fig "fig 10: GC by codon position" $PY $S/strat/gc_codon_figures.py \
    --run "$R" --from-figure-data --layer 27 --pub
  collect "$R/figures_27/pub/14_gc_codon_position" fig10_gc_by_codon_position
fi

if [ "$MODE" = plan ]; then
  echo; echo "══ dry run complete. --figures to re-render, --run for the whole experiment."
  exit 0
fi

finish_figures \
  fig05_stratum_composition \
  fig06a_leave_one_out_cosine_by_block \
  fig06b_delta_magnitude_spread_by_block \
  fig07_steering_delta_by_stratum \
  fig08_dose_response_by_stratum \
  fig09a_leave_human_vs_platypus_choice_private \
  fig09b_leave_human_vs_platypus_choice_loose \
  fig10_gc_by_codon_position \
  fig11_rate_predictor_outcome_matrix
