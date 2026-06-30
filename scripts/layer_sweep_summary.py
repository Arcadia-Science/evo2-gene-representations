"""Summary figures tracking how each baseline correlation moves across layers.

The all-layer sweeps (scripts/evo2/sweep_all_layers_evo2.sh,
scripts/gpnstar/sweep_all_layers_gpnstar.sh) drop one run dir per layer, each with
its own `between_family_baseline_scores.csv` and per-family within-family
correlation CSVs. Per-layer figures answer "which axis does the geometry track at
*this* layer"; this script collapses the whole sweep into two figures answering
"how does each baseline's correlation move *across* layers":

  between_axis_vs_layer.{png,pdf}
      Small multiples, one panel per between-family baseline (7), x = layer,
      y = Spearman rho (centroid geodesic vs baseline). Filled marker = Mantel
      p < 0.05, hollow = not significant. A lead panel overlays the three
      axis-mean curves (homology / mechanism / control). Companion to the
      per-layer `between_axis_scores.png` bar chart.

  within_family_vs_layer.{png,pdf}
      Small multiples, one panel per within-family baseline present on disk
      (k-mer / sequence identity / patristic / taxonomy / k-mer-transcript),
      x = layer, y = per-family Spearman rho, one coloured line per family.
      Within-family ground-truth distances are layer-independent, so each curve
      isolates which layer's geometry best recovers within-family divergence.

Both figures also dump the tidy long-form CSV they were built from.

Usage
-----
  uv run python scripts/layer_sweep_summary.py \
      --glob 'results/2026-06-30_evo2-human-panel-blocks*' \
      --out-dir results/2026-06-30_layer-sweep-summaries/evo2-human \
      --title 'Evo2 human panel'

The script is glob-driven and layer-token agnostic ('blocks{N}' or 'L{N}'), so it
works on any sweep once its run dirs exist (e.g. the pending GPN-Star sweep).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import set_pub_style  # noqa: E402

# ── shared style constants (kept in sync with gene_family_visualization.py) ──────
AXIS_COLORS = {"1_homology": "#C1666B", "2_mechanism": "#48A9A6", "control": "#9AA0A6"}
AXIS_LABELS = {
    "1_homology": "Homology",
    "2_mechanism": "Mechanism / chemistry",
    "control": "Composition (control)",
}
# Display order of the seven between-family baselines, grouped by axis.
BETWEEN_ORDER = [
    ("homology_tier", "1_homology"),
    ("pfam_jsd", "1_homology"),
    ("cofactor", "2_mechanism"),
    ("ec_number", "2_mechanism"),
    ("go_mf", "2_mechanism"),
    ("gc_content", "control"),
    ("kmer", "control"),
]

# Within-family baselines: (label, csv filename, rho column). Whichever files/columns
# are present on disk get a panel; the rest are silently skipped. The k-mer baseline
# lives in axisB_within_family_correlations.csv (Evo2 cross-kingdom, carries taxonomy)
# or kmer_within_family_correlations.csv (human / GPN panels) — both share the column
# name, so the first file that has it wins.
WITHIN_SPECS = [
    ("k-mer composition", ["axisB_within_family_correlations.csv",
                           "kmer_within_family_correlations.csv"], "spearman_geodesic_kmer"),
    ("sequence identity", ["within_family_seqid.csv"], "spearman_geodesic_seqid"),
    ("patristic tree", ["within_family_patristic.csv"], "spearman_geodesic_patristic"),
    ("taxonomy", ["axisB_within_family_correlations.csv"], "spearman_geodesic_taxonomy"),
    ("k-mer (transcript null)", ["within_family_kmer_transcript.csv"],
     "spearman_geodesic_kmer_transcript"),
]

LAYER_RE = re.compile(r"(?:blocks|L|hidden_state\.?|layer)(\d+)$")


def parse_layer(run_dir: Path) -> int | None:
    """Pull the integer layer index out of a sweep run-dir name."""
    m = LAYER_RE.search(run_dir.name)
    return int(m.group(1)) if m else None


def discover_layers(pattern: str) -> list[tuple[int, Path]]:
    """Glob → [(layer_index, run_dir), ...] sorted by layer, dirs only."""
    layers = []
    for p in sorted(Path().glob(pattern)):
        if not p.is_dir():
            continue
        layer = parse_layer(p)
        if layer is None:
            print(f"  skip (no layer token in name): {p.name}")
            continue
        layers.append((layer, p))
    return sorted(layers, key=lambda t: t[0])


# ════════════════════════════════════════════════════════════════════════════════
# Data loading → tidy long-form frames
# ════════════════════════════════════════════════════════════════════════════════
def load_between(layers: list[tuple[int, Path]]) -> pd.DataFrame:
    """Long frame: one row per (layer, baseline) with rho + Mantel p."""
    rows = []
    for layer, run_dir in layers:
        f = run_dir / "between_family_baseline_scores.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({
                "layer": layer,
                "baseline": r["baseline"],
                "axis": r.get("axis"),
                "rho": r.get("spearman_rho"),
                "p_mantel": r.get("p_mantel"),
            })
    return pd.DataFrame(rows)


def load_within(layers: list[tuple[int, Path]]) -> pd.DataFrame:
    """Long frame: one row per (layer, metric, family) with rho."""
    rows = []
    for layer, run_dir in layers:
        for label, filenames, col in WITHIN_SPECS:
            for fn in filenames:
                f = run_dir / fn
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                if col not in df.columns or "family" not in df.columns:
                    continue
                for _, r in df.iterrows():
                    rows.append({
                        "layer": layer,
                        "metric": label,
                        "family": r["family"],
                        "rho": pd.to_numeric(r[col], errors="coerce"),
                    })
                break  # first file that supplies this metric wins
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════════════
def _grid(n: int, ncols: int) -> tuple[int, int]:
    return (int(np.ceil(n / ncols)), ncols)


def plot_between(between: pd.DataFrame, out_dir: Path, title: str) -> None:
    if between.empty:
        print("  SKIP between figure: no between_family_baseline_scores.csv found")
        return
    set_pub_style(title_size=9, tick_size=7)
    present = [(b, ax) for b, ax in BETWEEN_ORDER if b in set(between["baseline"])]
    # Panel 0 = axis-mean overview, then one panel per baseline.
    n_panels = 1 + len(present)
    nrows, ncols = _grid(n_panels, 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             dpi=300, squeeze=False)
    axes = axes.ravel()
    layers_sorted = sorted(between["layer"].unique())

    # ── lead panel: mean rho per axis across layers ──────────────────────────────
    ax0 = axes[0]
    for axis in ["1_homology", "2_mechanism", "control"]:
        sub = between[between["axis"] == axis]
        if sub.empty:
            continue
        m = sub.groupby("layer")["rho"].mean().reindex(layers_sorted)
        ax0.plot(m.index, m.values, "-o", ms=3, lw=1.6, color=AXIS_COLORS[axis],
                 label=AXIS_LABELS[axis])
    ax0.axhline(0, color="#444", lw=0.6, ls="--")
    ax0.set_title("Axis means", fontweight="bold")
    ax0.set_ylabel("mean Spearman ρ")
    ax0.set_xlabel("layer")
    ax0.legend(frameon=False, fontsize=6, loc="best")

    # shared y-limits across the per-baseline panels for honest comparison
    finite = between["rho"].replace([np.inf, -np.inf], np.nan).dropna()
    ymax = max(0.6, float(np.nanmax(np.abs(finite))) * 1.15) if len(finite) else 1.0
    ylim = (-ymax, ymax)

    for i, (baseline, axis) in enumerate(present, start=1):
        ax = axes[i]
        sub = between[between["baseline"] == baseline].sort_values("layer")
        color = AXIS_COLORS[axis]
        ax.plot(sub["layer"], sub["rho"], "-", lw=1.2, color=color, zorder=1)
        sig = sub["p_mantel"] < 0.05
        ax.scatter(sub["layer"][sig], sub["rho"][sig], s=22, color=color,
                   zorder=3, label="Mantel p<0.05")
        ax.scatter(sub["layer"][~sig], sub["rho"][~sig], s=22, facecolors="white",
                   edgecolors=color, linewidths=0.9, zorder=3)
        ax.axhline(0, color="#444", lw=0.6, ls="--")
        ax.set_ylim(*ylim)
        ax.set_title(f"{baseline}\n({AXIS_LABELS[axis]})", color=color)
        if i % ncols == 0:
            ax.set_ylabel("Spearman ρ")
        ax.set_xlabel("layer")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title} — between-family ρ across layers", fontsize=11,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "between_axis_vs_layer")


def plot_within(within: pd.DataFrame, out_dir: Path, title: str) -> None:
    if within.empty:
        print("  SKIP within figure: no within-family correlation CSVs found")
        return
    set_pub_style(title_size=9, tick_size=7)
    # only metrics that have at least one finite rho somewhere in the sweep
    metrics = [m for m in within["metric"].unique()
               if within[(within["metric"] == m)]["rho"].notna().any()]
    metrics = [m for m in [s[0] for s in WITHIN_SPECS] if m in metrics]
    if not metrics:
        print("  SKIP within figure: all within-family rho are NaN")
        return

    # stable colour per family across panels
    families = sorted(within["family"].dropna().unique())
    cmap = plt.get_cmap("tab20" if len(families) > 10 else "tab10")
    fam_color = {f: cmap(i % cmap.N) for i, f in enumerate(families)}
    layers_sorted = sorted(within["layer"].unique())

    nrows, ncols = _grid(len(metrics), min(3, len(metrics)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             dpi=300, squeeze=False)
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = within[within["metric"] == metric]
        n_fam = 0
        for fam in families:
            fs = sub[sub["family"] == fam].sort_values("layer")
            if fs["rho"].notna().sum() == 0:
                continue
            ax.plot(fs["layer"], fs["rho"], "-o", ms=2.5, lw=1.0,
                    color=fam_color[fam], label=fam)
            n_fam += 1
        ax.axhline(0, color="#444", lw=0.6, ls="--")
        ax.set_title(f"{metric}  (n={n_fam} families)")
        ax.set_xlabel("layer")
        if i % ncols == 0:
            ax.set_ylabel("within-family Spearman ρ")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    # one shared legend (families) to the right
    handles = [plt.Line2D([0], [0], color=fam_color[f], lw=2) for f in families]
    fig.legend(handles, families, frameon=False, fontsize=6,
               loc="center left", bbox_to_anchor=(1.0, 0.5), title="family")
    fig.suptitle(f"{title} — within-family ρ across layers (per family)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    _save(fig, out_dir, "within_family_vs_layer")


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/{stem}.{{png,pdf}}")


# ════════════════════════════════════════════════════════════════════════════════
# Layer-independent ground-truth baselines (stored once, not per layer)
# ════════════════════════════════════════════════════════════════════════════════
# Every layer dir re-emits these, but they are sequence/annotation-derived ground
# truth that does NOT depend on the embedding layer (verified: identical across the
# sweep up to float-repr noise on the donor layer). The per-layer SCORE files
# (between_family_baseline_scores.csv, *_within_family_correlations.csv,
# within_family_{seqid,patristic}.csv) carry the geodesic ρ and are intentionally
# excluded — they are results, not baselines.
BASELINE_FILES = [
    "between_family_baseline_distances.csv",  # long-form: every baseline × every pair
    "pfam_jsd_distances.csv",                 # F×F Pfam HMM JSD (homology)
    "kmer_distance.npy",                      # gene-level k-mer distance (within k-mer)
    "taxonomic_distance.npy",                 # gene-level taxonomy distance (within tax.)
    "sequence_identity_family.csv",
    "kmer_distance_family.csv",
    "kmer_distance_genes.csv",
    "family_order.txt",                       # row/col order needed to read the matrices
    "metadata.csv",                           # per-member family/group/organism labels
]
# F×F per-baseline matrices share the betweenfam_<name>_distances.csv pattern.
BASELINE_GLOB = "betweenfam_*_distances.csv"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_baselines(layers: list[tuple[int, Path]], out_dir: Path) -> None:
    """Copy the layer-independent ground-truth baselines once into out_dir/baselines/.

    Sources each file from the HIGHEST layer index that has it — the donor (lowest
    layer) is the lone float-repr outlier, so the highest layer is always in the
    consistent majority group. Warns if a file has >2 distinct hashes across layers
    (2 = benign donor formatting; >2 would mean the baseline is not layer-independent
    and should not be deduplicated)."""
    bdir = out_dir / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)

    # union of explicit names + glob matches across the sweep
    names = list(BASELINE_FILES)
    for _, run_dir in layers:
        names += [p.name for p in run_dir.glob(BASELINE_GLOB)]
    names = sorted(set(names))

    manifest = ["# Layer-independent ground-truth baselines (stored once).",
                "# Copied by scripts/layer_sweep_summary.py from the all-layer sweep;",
                "# identical across every layer (only the per-layer geodesic differs).",
                "# columns: file  n_layers_present  n_distinct_hashes  source_layer", ""]
    copied = 0
    for name in names:
        present = [(layer, run_dir / name) for layer, run_dir in layers
                   if (run_dir / name).exists()]
        if not present:
            continue
        hashes = {_md5(p) for _, p in present}
        if len(hashes) > 2:
            print(f"  WARN {name}: {len(hashes)} distinct hashes across "
                  f"{len(present)} layers — NOT layer-independent, skipping")
            manifest.append(f"{name}\t{len(present)}\t{len(hashes)}\tSKIPPED (varies)")
            continue
        src_layer, src = max(present, key=lambda t: t[0])  # highest layer = majority group
        shutil.copy2(src, bdir / name)
        copied += 1
        manifest.append(f"{name}\t{len(present)}\t{len(hashes)}\tblocks{src_layer}")
    (bdir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"  collected {copied} baseline files → {bdir}/ (+ MANIFEST.txt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", required=True,
                    help="glob for the sweep run dirs, e.g. "
                         "'results/2026-06-30_evo2-human-panel-blocks*'")
    ap.add_argument("--out-dir", required=True, help="where to write the summary figures")
    ap.add_argument("--title", default="", help="figure title prefix")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip collecting the layer-independent baselines into baselines/")
    args = ap.parse_args()

    layers = discover_layers(args.glob)
    if not layers:
        sys.exit(f"No layer run dirs matched {args.glob!r}")
    title = args.title or args.glob
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{title}: {len(layers)} layers "
          f"[{layers[0][0]}..{layers[-1][0]}]")

    between = load_between(layers)
    within = load_within(layers)
    if not between.empty:
        between.to_csv(out_dir / "between_axis_vs_layer.csv", index=False)
    if not within.empty:
        within.to_csv(out_dir / "within_family_vs_layer.csv", index=False)
    plot_between(between, out_dir, title)
    plot_within(within, out_dir, title)
    if not args.no_baselines:
        collect_baselines(layers, out_dir)


if __name__ == "__main__":
    main()
