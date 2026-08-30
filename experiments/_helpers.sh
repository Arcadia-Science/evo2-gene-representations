#!/usr/bin/env bash

init_experiment() {
  local id="$1"
  OK=0
  FAILED=0
  SKIPPED=0
  FAIL_LIST=()
  FIG_LOG="/tmp/${id}_figure.log"
  FIG_MARKER="/tmp/${id}_figure_started"
  FIG_SUCCEEDED=0
  CURRENT_FIG=
  [ "$MODE" = plan ] || mkdir -p "$OUT"
}

say() { echo; echo "══ $*"; }

stage() {
  local cost="$1" desc="$2"; shift 2
  echo; echo "── $desc   [$cost]"; echo "   $*"
  [ "$MODE" = run ] || return 0
  "$@" && echo "   ok" || { echo "   FAILED — chain stopped"; exit 1; }
}

need() {
  local label="$1" p; shift
  for p in "$@"; do
    compgen -G "$p" > /dev/null || {
      SKIPPED=$((SKIPPED+1)); echo "── $label"; echo "   SKIP — missing $p"; return 1; }
  done
}

fig() {
  local label="$1"; shift
  CURRENT_FIG="$label"
  FIG_SUCCEEDED=0
  echo "── $label"
  [ "$MODE" = plan ] && { echo "   $*"; return 0; }
  : > "$FIG_MARKER"
  if "$@" > "$FIG_LOG" 2>&1; then
    FIG_SUCCEEDED=1
    grep -E '^\s+pub: ' "$FIG_LOG" | sed 's/^/   /'
    return 0
  fi
  FAILED=$((FAILED+1)); FAIL_LIST+=("$label")
  echo "   FAILED:"; tail -6 "$FIG_LOG" | sed 's/^/   | /'
  return 1
}

collect() {
  local src="$1" dest="$2" e
  [ "$MODE" = plan ] && return 0
  [ "$FIG_SUCCEEDED" -eq 1 ] || return 1
  for e in png pdf; do
    if [ ! -f "$src.$e" ] || [ ! "$src.$e" -nt "$FIG_MARKER" ]; then
      FAILED=$((FAILED+1)); FAIL_LIST+=("$CURRENT_FIG ($e output missing or stale)")
      echo "   FAILED — expected fresh $src.$e"
      return 1
    fi
  done
  for e in png pdf; do
    cp -p "$src.$e" "$OUT/$dest.$e" || {
      FAILED=$((FAILED+1)); FAIL_LIST+=("$CURRENT_FIG (collect $e)"); return 1; }
  done
  OK=$((OK+1))
}

finish_figures() {
  echo; echo "══ $OK ok, $FAILED failed, $SKIPPED skipped"
  [ ${#FAIL_LIST[@]} -gt 0 ] && printf '   FAILED: %s\n' "${FAIL_LIST[@]}"
  echo; echo "══ panel geometry (published panels are exactly 1000 or 500 pt wide)"
  $PY - "$OUT" "$@" <<'PY'
from pathlib import Path
import sys

from PIL import Image

out = Path(sys.argv[1])
bad = 0
for stem in sys.argv[2:]:
    for ext in ("png", "pdf"):
        path = out / f"{stem}.{ext}"
        if not path.is_file():
            print(f"   MISSING: {path}")
            bad += 1
    path = out / f"{stem}.png"
    if not path.is_file():
        continue
    image = Image.open(path)
    dpi = image.info.get("dpi", (300, 300))[0]
    width, height = (dimension / (dpi / 72) for dimension in image.size)
    flag = "" if round(width) in (500, 1000) else "   <-- OFF-SPEC"
    bad += bool(flag)
    print(f"   {round(width):>5} x {round(height):<5} pt   {path.name}{flag}")
sys.exit(1 if bad else 0)
PY
  local geometry_failed=$?
  [ "$FAILED" -eq 0 ] && [ "$SKIPPED" -eq 0 ] && [ "$geometry_failed" -eq 0 ]
}
