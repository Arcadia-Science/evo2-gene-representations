#!/usr/bin/env bash
# Regenerate the within/between gene-family figures at a chosen layer-selection layer,
# pulling that layer's embeddings from the dense-sweep cache (NO re-embedding / GPU).
#
#   bash scripts/rerun_layer_figs.sh evo2 15        # Evo2 blocks.15
#   (GPN-Star gene-family layer figs were retired with the legacy paralog pipeline.)
#
# Output dir is suffixed with the layer (…-blocks<N> / …-L<N>) so it never clobbers the
# canonical production run. Axis-A separability uses a light permutation count (the effect
# is large; p-resolution 1e-2 suffices) to skip the slow 9999-perm pole.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; LAYER="$2"
DATE=$(date +%F)
PERMS="${PERMS:-99}"

if [ "$MODEL" = evo2 ]; then
  RUN_DIR="results/${DATE}_evo2-gene-families-blocks${LAYER}"
  echo "=== Evo2 gene figures @ blocks.${LAYER} -> $RUN_DIR ==="
  uv run python scripts/evo2/embed_and_geodesic_ortholog.py --from-sweep-layer "$LAYER" --n-perms "$PERMS"
  uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$RUN_DIR"
  uv run python scripts/baselines/between_family_baselines.py --run-dir "$RUN_DIR" --seq-source evo2 --n-perms 9999
  uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$RUN_DIR" --seq-source evo2
  uv run python scripts/evo2/gene_family_visualization.py --run-dir "$RUN_DIR"

else
  echo "usage: $0 evo2 <layer-index>"
  echo "(The GPN-Star gene-family layer-fig path was retired with the legacy paralog pipeline."
  echo " For GPN-Star, use scripts/gpnstar/run_paralog_human_gene_pipeline_gpnstar.sh / embed_and_geodesic.py.)"
  exit 1
fi
echo "=== done: $RUN_DIR ==="
