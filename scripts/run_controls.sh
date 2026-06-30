#!/usr/bin/env bash
# Composition / ablation controls for the gene-family panels — run AFTER the GPN-human MSA
# controls (which run separately) free the GPU. Two model-appropriate sets:
#   1. Cross-kingdom Evo2 (CDS): full suite (dinuc/codon/synonymous_recode/gc/k-mer) re-embedded
#      with the current second-half pooling @ blocks.15 → within + between scores + figure.
#   2. Evo2-human (genomic): composition shuffles applicable to a genomic span (gc_match,
#      dinuc_shuffle, kmer6_shuffle) @ blocks.13 → per-control within (patristic) + between scores.
#      (codon/synonymous N/A to genomic — protein-level is covered by the cross-kingdom suite.)
# GPN-human MSA controls are launched separately (analyses/embed_and_score_msa_controls.py).
#
# Detached launch:
#   setsid nohup bash scripts/run_controls.sh </dev/null >>logs/controls.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
say() { echo ">>> [$(date)] $*"; }

XK_RUN="${XK_RUN:-results/2026-06-25_evo2-gene-families-blocks15}"
EVO_LAYER="${EVO_LAYER:-15}"      # cross-kingdom chosen block
HUMAN_LAYER="${HUMAN_LAYER:-13}"  # Evo2-human chosen block

echo "############################################################"
echo "## Gene-family composition controls  START $(date)  pid=$$"
echo "############################################################"

# Wait for the GPN-human MSA controls to finish (shared GPU).
say "waiting for GPN-human MSA controls to finish before using the GPU..."
while pgrep -f embed_and_score_msa_controls >/dev/null 2>&1; do sleep 60; done
say "GPU free — starting Evo2 controls"

# ── 1. Cross-kingdom Evo2 controls (CDS suite, second-half @ blocks.$EVO_LAYER) ──────
say "cross-kingdom: clearing stale blocks${EVO_LAYER} control embeddings (old [-2000:] pooling)"
rm -rf data/evo2_gene_families/controls/*/embeddings_blocks"${EVO_LAYER}"
say "cross-kingdom Evo2 controls — WITHIN axis (re-embed + recovery + vs-natural preservation)"
uv run python analyses/embed_and_score_controls.py --control all --axis within --layer "$EVO_LAYER" --natural-run "$XK_RUN" \
  || say "WARN: cross-kingdom within failed"
say "cross-kingdom Evo2 controls — BETWEEN axis (reuses cached embeddings)"
uv run python analyses/embed_and_score_controls.py --control all --axis between --natural-run "$XK_RUN" --layer "$EVO_LAYER" \
  || say "WARN: cross-kingdom between failed"
say "cross-kingdom control comparison figure"
uv run python analyses/control_comparison_figure.py --natural-run "$XK_RUN" --layer "$EVO_LAYER" \
  || say "WARN: cross-kingdom figure failed"

# ── 2. Evo2-human composition controls (genomic shuffles @ blocks.$HUMAN_LAYER) ──────
for C in gc_match dinuc_shuffle kmer6_shuffle; do
  say "Evo2-human control: $C — embed @ blocks.$HUMAN_LAYER"
  uv run python scripts/evo2/embed_and_geodesic_paralog.py --control "$C" --from-layer "$HUMAN_LAYER" --force-reembed \
    || { say "WARN: $C embed failed"; continue; }
  CRUN=$(ls -dt results/*_evo2-human-panel-blocks"${HUMAN_LAYER}"-"${C}" | head -1)
  say "  scoring control run $CRUN"
  uv run python scripts/baselines/protein_alignment_patristic_seqid.py --run-dir "$CRUN" --seq-source gpn || say "WARN: $C patristic"
  uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir "$CRUN" || say "WARN: $C pfam-jsd"
  uv run python scripts/baselines/between_family_baselines.py --run-dir "$CRUN" --seq-source gpn --n-perms 9999 || say "WARN: $C between"
done
# Reconstruction (preservation) of the natural human geometry by each control, within + between.
HUMAN_RUN=$(ls -dt results/*_evo2-human-panel-blocks"${HUMAN_LAYER}" 2>/dev/null | grep -v -- '-blocks'"${HUMAN_LAYER}"'-' | head -1)
if [ -n "$HUMAN_RUN" ]; then
  say "Evo2-human control preservation scoring vs natural run $HUMAN_RUN"
  uv run python analyses/embed_and_score_controls.py --control all --panel human \
    --natural-run "$HUMAN_RUN" --layer "$HUMAN_LAYER" || say "WARN: human control preservation failed"
fi

echo "############################################################"
echo "## Controls DONE $(date)"
echo "## cross-kingdom: $XK_RUN/controls/  | Evo2-human: results/*-evo2-human-panel-blocks${HUMAN_LAYER}-<control>/"
echo "############################################################"
