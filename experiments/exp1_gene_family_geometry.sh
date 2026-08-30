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

# B2 computes the MAFFT/FastTree patristic and k-mer baselines; B3 computes Pfam-HMM JSD.
stage "~3 h, CPU"   "B2. within-family distances vs patristic / species tree / k-mer / GC" \
  $PY scripts/mammalian_orthologs/mammal_score.py --arm transcript_cdsmask --distance both
stage "~1 h, CPU"   "B3. between-family distances vs Pfam JSD / k-mer / GC" \
  $PY scripts/mammalian_orthologs/mammal_between.py --arm transcript_cdsmask
stage "~1 h, CPU"   "B4. between-family on the 400-cap panel (family-size robustness)" \
  $PY scripts/mammalian_orthologs/mammal_between.py --arm transcript_cdsmask \
     --manifest complete_manifest_cap400.csv --tag _400

# Experiment 2 uses these natural W2 matrices as the reference for control preservation.
stage "~1.1 h, CPU" "B5. Wasserstein sweep, all layers, with 9,999 Mantel permutations" \
  $PY scripts/baselines/ot_between_family_sweep.py --n-perms 9999

# Block 15 is the pre-specified inference layer for the published within-family claims.
stage "~2 h, CPU"   "B6. within-family CIs and per-group Mantel inference at block 15" \
  $PY scripts/mammalian_orthologs/within_family_uncertainty.py \
    --layers 15 --report-layer 15 --mantel-layer 15 --n-perms 999
fi

say "Figures 1-3"

# Publication panels use angular within-family scoring.
if need "figs 1-2: between/within rho by layer" \
        "$RUN/blocks*/between_family_baseline_scores.csv" \
        "$RUN/blocks*/within_family_patristic_angular.csv"; then
  fig "figs 1-2: between/within rho by layer" $PY scripts/layer_sweep_summary.py \
    --glob "$RUN/blocks*" \
    --out-dir "$GF" \
    --title 'Evo2 mammalian orthologs (transcript, CDS-masked) — 48 families, graph-free metrics' \
    --stem-suffix _graphfree --within-file-suffix _angular --kmer-k 6 \
    --lead-per-baseline --no-baselines --pub
  collect "$GF/pub/between_axis_vs_layer_graphfree"  fig01_between_family_rho_by_layer
  collect "$GF/pub/within_family_vs_layer_graphfree" fig02_within_family_rho_by_layer
fi

if need "fig 3: three-family zoom" "$GF/within_family_vs_layer_graphfree.csv"; then
  fig "fig 3: three-family zoom" $PY scripts/within_family_per_family_grid.py \
    --csv "$GF/within_family_vs_layer_graphfree.csv" \
    --families adrenoceptor glutathione_peroxidase peroxidase --out-suffix zoom3 --pub
  collect "$GF/pub/within_family_vs_layer_graphfree_zoom3" fig03_within_family_three_families
fi

if [ "$MODE" = plan ]; then
  echo; echo "══ dry run complete. --figures to re-render, --run for the whole experiment."
  exit 0
fi

finish_figures \
  fig01_between_family_rho_by_layer \
  fig02_within_family_rho_by_layer \
  fig03_within_family_three_families
