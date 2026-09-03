#!/usr/bin/env bash
# Experiment 1 — between- and within-family geometry vs homology and composition baselines.
# Produces publication figures 1, 2 and 3.
#
#   bash experiments/exp1_gene_family_geometry.sh            # print the plan, run nothing
#   bash experiments/exp1_gene_family_geometry.sh --figures  # re-render figures 1-3  (~2 min, CPU)
#   bash experiments/exp1_gene_family_geometry.sh --run      # full pipeline          (~40 h, GPU)
#
# Panel: 48 HGNC families / 1,144 human paralogs, resolved to 1:1 orthologs across 24 mammals
# (Ensembl Compara release 116). Readout is the residual stream with non-coding positions masked,
# mean-pooled over the second half of token positions, L2-normalised, at all 32 blocks.

set -uo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-plan}"
case "$MODE" in
  --run) MODE=run ;;
  --figures) MODE=figures ;;
  plan|--plan|"") MODE=plan ;;
  -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  *) echo "unknown option: $MODE (use --run, --figures, or nothing)"; exit 2 ;;
esac

PY="uv run --no-sync python"
RUN=results/2026-07-16_mammalian-orthologs-transcript_cdsmask
GF=results/layer_sweep_summaries/mammalian-orthologs-cdsmask-48fam
OUT=pub/figures
source experiments/_helpers.sh
init_experiment exp1
preflight_tools exp1

echo "Experiment 1 — gene-family geometry (figures 1-3)"
[ "$MODE" = plan ]    && echo "DRY RUN — nothing will execute."
[ "$MODE" = figures ] && echo "FIGURES ONLY — re-rendering from artifacts on disk."

if [ "$MODE" != figures ]; then
say "Dataset"

# Check the per-family size report after A1 for incomplete Compara responses.
stage "~6 h, API"   "A1. Compara 1:1 ortholog resolution, 24 mammals" \
  $PY scripts/mammalian_orthologs/resolve_orthologs.py
stage "~3 h, net"   "A2. download Ensembl GTF + genome FASTA per species" \
  $PY scripts/mammalian_orthologs/download_bulk.py
stage "~2 h, CPU"   "A3. extract transcript-span locus + CDS per ortholog, with QC" \
  $PY scripts/mammalian_orthologs/extract_loci_bulk.py
stage "~20 min"     "A4. assemble manifests and per-locus FASTAs" \
  $PY scripts/mammalian_orthologs/assemble_datasets.py
stage "~5 min"      "A5. stratified 400-loci-per-family cap" \
  $PY scripts/mammalian_orthologs/build_capped_manifest.py
stage "~5 min"      "A6. VertLife/MamPhy species tree -> patristic matrix" \
  $PY scripts/mammalian_orthologs/build_species_tree.py
stage "~15 min"     "A7. per-locus CDS-position masks" \
  $PY scripts/mammalian_orthologs/build_cds_masks_mammal.py

say "Embedding and scoring"

FAMS='<families from complete_manifest.csv>'
if [ "$MODE" = run ]; then
FAMS=$($PY -c "
import pandas as pd; print(' '.join(sorted(pd.read_csv('data/mammalian_orthologs/complete_manifest.csv').family.unique())))" 2>/dev/null) \
  || FAMS='<families from complete_manifest.csv>'
fi

# Safe to interrupt: each locus is written to a .tmp.npy and atomically renamed, so a restart skips
# completed loci and loses at most the in-flight locus.
stage "~34 h, GPU"  "B1. embed 11,288 loci x 32 blocks, CDS-masked" \
  $PY scripts/mammalian_orthologs/embed_cds_masked_mammal.py --families $FAMS

# Score direct angular distance within families. The existing geodesic implementation remains
# available for optional legacy analyses, but is not part of the publication pipeline.
stage "~3 h, CPU"   "B2. angular within-family distances vs patristic / species tree / k-mer / GC" \
  $PY scripts/mammalian_orthologs/mammal_score.py --arm transcript_cdsmask --distance angular

# The composition-controls analysis (analyses/controls/) uses these natural W2 matrices
# as its reference for control preservation.
stage "~20 min, CPU" "B3. Wasserstein sweep, all layers (Mantel optional, skipped)" \
  $PY scripts/baselines/ot_between_family_sweep.py --n-perms 0

# Bootstrap and Wilcoxon inference use the existing angular per-group results at every layer.
stage "~20 min, CPU" "B4. angular within-family bootstrap/Wilcoxon inference, all layers" \
  $PY scripts/mammalian_orthologs/within_family_uncertainty.py --arm transcript_cdsmask

# Everything above writes into the dated run dirs, which stay local. This collapses them into the
# tracked tidy tables the figures actually read.
stage "~1 min, CPU" "B5. build this experiment's figure_data/ tables" \
  $PY scripts/build_figure_data.py --experiments exp1
fi

say "Figures 1-3"

# Publication panels use angular within-family scoring.
if need "figs 1-2: between/within rho by layer" \
        figure_data/exp1_between_family_by_layer.csv \
        figure_data/exp1_within_family_by_layer.csv; then
  fig "figs 1-2: between/within rho by layer" $PY scripts/layer_sweep_summary.py \
    --from-figure-data \
    --out-dir "$GF" \
    --title 'Evo2 mammalian orthologs (transcript, CDS-masked) — 48 families, W2 / angular' \
    --stem-suffix _wasserstein_angular --kmer-k 6 \
    --lead-per-baseline --no-baselines --pub
  collect "$GF/pub/between_axis_vs_layer_wasserstein_angular"  fig01_between_family_rho_by_layer
  collect "$GF/pub/within_family_vs_layer_wasserstein_angular" fig02_within_family_rho_by_layer
fi

if need "fig 3: three-family zoom" figure_data/exp1_within_family_by_layer.csv; then
  fig "fig 3: three-family zoom" $PY scripts/within_family_per_family_grid.py \
    --csv figure_data/exp1_within_family_by_layer.csv --out-dir "$GF" \
    --stem within_family_vs_layer_wasserstein_angular \
    --families adrenoceptor glutathione_peroxidase peroxidase --out-suffix zoom3 --pub
  collect "$GF/pub/within_family_vs_layer_wasserstein_angular_zoom3" \
          fig03_within_family_three_families
fi

if [ "$MODE" = plan ]; then
  echo; echo "══ dry run complete. --figures to re-render, --run for the whole experiment."
  exit 0
fi

finish_figures \
  fig01_between_family_rho_by_layer \
  fig02_within_family_rho_by_layer \
  fig03_within_family_three_families
