#!/usr/bin/env bash
# Orchestrator: run all three all-layer sweeps strictly in sequence (so each has the
# whole machine), CPU-only (embedding stacks are cached). Detached via setsid so it
# survives SSH/session close. Per-sweep logs live in logs/<sweep>.log; this driver's
# own progress is in logs/all_layer_sweeps.log.
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "######## [$(date)] $*"; }

say "ALL-LAYER SWEEPS START (pid=$$)"

say "[1/3] Evo2 human panel (32 blocks)"
PANEL=human bash scripts/evo2/sweep_all_layers_evo2.sh
say "[1/3] Evo2 human panel DONE (exit=$?)"

say "[2/3] Evo2 gene families (32 blocks)"
PANEL=gene-families bash scripts/evo2/sweep_all_layers_evo2.sh
say "[2/3] Evo2 gene families DONE (exit=$?)"

say "[3/3] GPN-Star human panel (17 hidden states)"
bash scripts/gpnstar/sweep_all_layers_gpnstar.sh
say "[3/3] GPN-Star human panel DONE (exit=$?)"

say "ALL-LAYER SWEEPS COMPLETE"
