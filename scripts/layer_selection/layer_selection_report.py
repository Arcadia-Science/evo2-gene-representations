"""Deliverable for the layer-selection benchmark: combined table + figure + recommendation.

Reads the per-model layer_scores_{evo2,gpnstar}.csv written by scripts/layer_selection/layer_selection.py
and emits the two-component deliverable:

  1. layer_selection_table.csv — one row per (model, layer): model, layer,
     within_dispersion, between_dispersion, within_between_ratio, the per-family PCA
     top-3 variance mean (and mean PC1/PC2/PC3), recommended_use.
  2. layer_selection.{png,pdf} — per-model layer curves: (a) the within/between Euclidean
     clustering ratio (with the raw within & between dispersions on a twin axis), best
     layer annotated; (b) the per-family PCA mean variance in PC1/PC2/PC3 (stacked) with
     the mean top-3 fraction.
  3. RECOMMENDATION.md — best clustering layer (min within/between ratio) per model and the
     per-family PCA intrinsic-dimensionality readout, plus the cross-model comparison.

The two components are reported SEPARATELY and are not combined into one score:
  * within/between ratio  — how tight families are relative to their separation (lower is
    better); within is member→member pairwise Euclidean averaged with EQUAL weight per
    family, between is pairwise distance between family centroids.
  * per-family PCA top-3   — for each family, the fraction of its member-gene variance in its
    top 3 PCs (equal-weight mean over families); how low-dimensional each family's cloud is.

Usage:
    uv run python scripts/layer_selection/layer_selection_report.py
    uv run python scripts/layer_selection/layer_selection_report.py --run-dir results/2026-06-22_layer-selection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plot_utils import set_pub_style  # noqa: E402

MODELS = ["evo2", "gpnstar"]
MODEL_TITLE = {"evo2": "Evo2 7B (cross-kingdom CDS, 32 blocks)",
               "gpnstar": "GPN-Star (human vertebrate MSA, 17 states)"}
MODEL_CAVEAT = {
    "evo2": "⚠ Embeddings pool only the last 2000 nt of each CDS, so genes >2 kb lose their "
            "N-terminus (4.3% of CDS overall; nitric_oxide_synthase 42.8%, heme_copper_oxidase "
            "8.2%). Layer-depth conclusions are robust, but NOS/HCO numbers will shift once the "
            "whole-CDS pooling fix is applied. See quarterly_progress_update.md §2.",
    "gpnstar": "Whole-gene exon-aware multi-window embedding (no truncation issue).",
}


def newest_run() -> Path:
    cands = sorted(Path("results").glob("*_layer-selection"))
    if not cands:
        sys.exit("No results/*_layer-selection run found. Run scripts/layer_selection/layer_selection.py first.")
    return cands[-1]


def load(run_dir: Path, standardize: str = "none") -> dict[str, pd.DataFrame]:
    suffix = "" if standardize == "none" else f"_{standardize}"
    out = {}
    for m in MODELS:
        f = run_dir / f"layer_scores_{m}{suffix}.csv"
        if f.exists():
            out[m] = pd.read_csv(f)
    if not out:
        sys.exit(f"No layer_scores_*{suffix}.csv in {run_dir}")
    return out


def build_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m, df in data.items():
        for _, r in df.iterrows():
            rows.append({
                "model": m,
                "layer": r["layer"],
                "within_dispersion": r["within_dispersion"],
                "between_dispersion": r["between_dispersion"],
                "within_between_ratio": r["within_between_ratio"],
                "pca_family_pc1_mean": r["pca_family_pc1_mean"],
                "pca_family_pc2_mean": r["pca_family_pc2_mean"],
                "pca_family_pc3_mean": r["pca_family_pc3_mean"],
                "pca_family_top3_mean": r["pca_family_top3_mean"],
                "recommended_use": r["recommended_use"],
            })
    return pd.DataFrame(rows)


def plot(data: dict[str, pd.DataFrame], run_dir: Path, suffix: str = "") -> None:
    set_pub_style()
    n = len(data)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.4 * n), squeeze=False)
    for row, (m, df) in enumerate(data.items()):
        x = df["layer_idx"].to_numpy()

        # (a) within/between clustering ratio, with raw dispersions on a twin axis.
        ax = axes[row][0]
        ax.plot(x, df["within_between_ratio"], "-o", color="#B5651D", lw=2, ms=4,
                label="within/between ratio (lower = tighter)")
        best = df["within_between_ratio"].idxmin()
        ax.axvline(df["layer_idx"][best], color="#B5651D", alpha=0.25, lw=6,
                   label=f"min ratio (L{int(df['layer_idx'][best])})")
        ax.set_title(MODEL_TITLE[m], fontsize=10)
        ax.set_xlabel("layer index (0 = input/embeddings → last = output)")
        ax.set_ylabel("within / between Euclidean ratio", color="#B5651D")
        ax.tick_params(axis="y", labelcolor="#B5651D")
        axb = ax.twinx()
        axb.plot(x, df["within_dispersion"], "--", color="#2C6E8A", alpha=0.6,
                 label="within dispersion (Euclidean)")
        axb.plot(x, df["between_dispersion"], ":", color="#4C9A2A", alpha=0.8,
                 label="between dispersion (centroids)")
        axb.set_ylabel("raw Euclidean dispersion")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axb.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

        # (b) per-family PCA: mean (over families) variance in each family's top 3 PCs.
        ax2 = axes[row][1]
        ax2.bar(x, df["pca_family_pc1_mean"], color="#2C6E8A", label="PC1 (mean)")
        ax2.bar(x, df["pca_family_pc2_mean"], bottom=df["pca_family_pc1_mean"],
                color="#6FB3C9", label="PC2 (mean)")
        ax2.bar(x, df["pca_family_pc3_mean"],
                bottom=df["pca_family_pc1_mean"] + df["pca_family_pc2_mean"],
                color="#B9DCE6", label="PC3 (mean)")
        ax2.plot(x, df["pca_family_top3_mean"], "-o", color="#B5651D", lw=1.5, ms=3,
                 label="top-3 (mean over families)")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("layer index")
        ax2.set_ylabel("fraction of within-family variance")
        ax2.set_title(f"{m}: per-family PCA (mean top-3 over families)", fontsize=10)
        ax2.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"layer_selection{suffix}.{ext}")
    plt.close(fig)
    print(f"  Saved {run_dir}/layer_selection{suffix}.png|pdf")


def recommendation(data: dict[str, pd.DataFrame]) -> str:
    lines = ["# Layer-selection recommendation\n",
             "The benchmark reports two components SEPARATELY (they are not combined):\n"
             "- **within/between clustering ratio** (Euclidean, lower is better): within is the "
             "per-family mean pairwise member→member distance, averaged with EQUAL weight across "
             "families so a large family cannot dominate; between is the mean pairwise distance "
             "between family centroids. The recommended layer minimizes this ratio.\n"
             "- **per-family PCA top-3 variance**: for each family, the fraction of that family's "
             "own member-gene variance captured by its first 3 principal components, reported as "
             "the EQUAL-weight mean across families — a readout of how low-dimensional each "
             "family's gene cloud is (per-family breakdown in family_pca_<model>.csv).\n"]
    per_model = {}
    for m, df in data.items():
        best = df.loc[df["within_between_ratio"].idxmin()]
        per_model[m] = best
        lines += [
            f"## {MODEL_TITLE[m]}\n",
            f"> {MODEL_CAVEAT[m]}\n",
            f"- **Best clustering layer (min within/between ratio):** `{best['layer']}` "
            f"(idx {int(best['layer_idx'])}): ratio={best['within_between_ratio']:.4f} "
            f"(within={best['within_dispersion']:.3f}, between={best['between_dispersion']:.3f}).",
            f"- **Per-family PCA at that layer:** mean top-3 variance={best['pca_family_top3_mean']:.3f} "
            f"(mean PC1={best['pca_family_pc1_mean']:.3f}, PC2={best['pca_family_pc2_mean']:.3f}, "
            f"PC3={best['pca_family_pc3_mean']:.3f}).",
            "(Full per-layer numbers in layer_selection_table.csv and the figure.)",
            "",
        ]
    ew, gw = per_model.get("evo2"), per_model.get("gpnstar")
    if ew is not None and gw is not None:
        lines.append("## Cross-model\n")
        lines.append(
            f"- Best clustering depth: Evo2 idx {int(ew['layer_idx'])} "
            f"(of {len(data['evo2'])-1}, ratio {ew['within_between_ratio']:.4f}) vs "
            f"GPN-Star idx {int(gw['layer_idx'])} "
            f"(of {len(data['gpnstar'])-1}, ratio {gw['within_between_ratio']:.4f}).\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--standardize", default="none", choices=["none", "zscore", "l2"],
                    help="Which layer_scores_<model>[_<mode>].csv variant to report on.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else newest_run()
    suffix = "" if args.standardize == "none" else f"_{args.standardize}"
    print(f"run-dir: {run_dir} (standardize={args.standardize})")
    data = load(run_dir, args.standardize)

    table = build_table(data)
    table.to_csv(run_dir / f"layer_selection_table{suffix}.csv", index=False)
    print(f"  Saved {run_dir}/layer_selection_table{suffix}.csv ({len(table)} rows)")

    plot(data, run_dir, suffix)

    rec = recommendation(data)
    (run_dir / f"RECOMMENDATION{suffix}.md").write_text(rec)
    print(f"  Saved {run_dir}/RECOMMENDATION{suffix}.md\n")
    print(rec)


if __name__ == "__main__":
    main()
