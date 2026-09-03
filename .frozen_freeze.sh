#!/usr/bin/env bash
# Freeze current publication outputs so the refactor-validation re-run can be diffed against them.
# Copies analysis artifacts only: .npy/.npz embedding caches are excluded (they are inputs, and no
# stage in the validation chain rewrites them).
set -euo pipefail
cd "$(dirname "$0")"

DEST=".frozen/$(date +%Y-%m-%d)"
mkdir -p "$DEST"

PATHS=(
  results/2026-07-16_mammalian-orthologs-transcript_cdsmask
  results/layer_sweep_summaries
  results/2026-08-08_platypus-strat-400
  results/2026-07-28_evo2-platypus-paired
  results/_ot_control_preservation_transcript_cdsmask.csv
  results/_angular_control_preservation_transcript_cdsmask.csv
  results/_centroid_k_sweep_transcript_cdsmask.csv
  pub/figures
  data/platypus_strat_orthologs
)

for p in "${PATHS[@]}"; do
  [ -e "$p" ] || { echo "MISSING SOURCE: $p"; exit 1; }
  mkdir -p "$DEST/$(dirname "$p")"
  rsync -a --links --exclude='*.npy' --exclude='*.npz' --exclude='*.fa.gz' \
        --exclude='*.fa' --exclude='__pycache__' "$p" "$DEST/$(dirname "$p")/"
done

# Small metadata from the mammal dataset dir (manifests/tree), not the per-locus FASTAs.
mkdir -p "$DEST/data/mammalian_orthologs"
rsync -a --include='*/' --include='*.csv' --include='*.json' --include='*.nwk' \
      --include='*.tsv' --include='*.md' --exclude='*' \
      data/mammalian_orthologs/ "$DEST/data/mammalian_orthologs/"

# Manifest of sha256 + size for every frozen file, for the post-run diff.
( cd "$DEST" && find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z | xargs -0 -P 8 -n 64 sha256sum > MANIFEST.sha256 )

echo
echo "frozen -> $DEST"
du -sh "$DEST"
wc -l < "$DEST/MANIFEST.sha256" | xargs echo "files:"
