import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parent.parent

FAMILY_COLORS = {
    "globins":             "#E63946",
    "hox":                 "#F4A261",
    "ras_gtpases":         "#2A9D8F",
    "cytochrome_p450":     "#457B9D",
    "c2h2_zinc_fingers":   "#A8DADC",
    "aquaporins":          "#6A4C93",
    "sirtuins":            "#52B788",
    "toll_like_receptors": "#8B8B00",
    "wnt_ligands":         "#E9C46A",
    "kinesins":            "#264653",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Figure 2 from embed_and_geodesic.py outputs.")
    p.add_argument(
        "--run-dir",
        required=True,
        help="Path to a dated results folder (e.g. results/2026-06-10_gpnstar-vertebrate/)",
    )
    return p.parse_args()


args = parse_args()
RUN_DIR = Path(args.run_dir)

# ── Load core data ─────────────────────────────────────────────────────────────

_centroid_candidates = list(RUN_DIR.glob("*_centroid_distances.csv"))
if not _centroid_candidates:
    print(f"No *_centroid_distances.csv found in {RUN_DIR}", file=sys.stderr)
    sys.exit(1)
_model_prefix = _centroid_candidates[0].stem.replace("_centroid_distances", "")

_geo_labeled = list(RUN_DIR.glob("*_geodesic_labeled.csv"))
if not _geo_labeled:
    print("No *_geodesic_labeled.csv found — re-run embed_and_geodesic.py", file=sys.stderr)
    sys.exit(1)
df_geo = pd.read_csv(_geo_labeled[0], index_col=0)
geodesic = df_geo.values
gene_names = df_geo.index.tolist()

meta_df = pd.read_csv(RUN_DIR / "metadata.csv")
families_arr = meta_df["family"].values
family_order = (RUN_DIR / "family_order.txt").read_text().strip().splitlines()
N_fam = len(family_order)
N_genes = len(gene_names)

dist_centroid = pd.read_csv(RUN_DIR / f"{_model_prefix}_centroid_distances.csv", index_col=0)
dist_jsd = pd.read_csv(RUN_DIR / "pfam_jsd_distances.csv", index_col=0)

# ── Optional baselines ─────────────────────────────────────────────────────────

_seqid_fam_path = RUN_DIR / "sequence_identity_family.csv"
dist_seqid_fam = pd.read_csv(_seqid_fam_path, index_col=0) if _seqid_fam_path.exists() else None

_seqid_gene_path = RUN_DIR / "sequence_identity_genes.csv"
seqid_genes = pd.read_csv(_seqid_gene_path, index_col=0) if _seqid_gene_path.exists() else None

_panther_path = RUN_DIR / "panther_distances.csv"
dist_panther = pd.read_csv(_panther_path, index_col=0) if _panther_path.exists() else None

# ── Spearman ρ with bootstrap CI ──────────────────────────────────────────────

idx_upper = np.triu_indices(N_fam, k=1)
x_geo = dist_centroid.values[idx_upper]


def bootstrap_spearman(x, y, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    rho_obs, _ = spearmanr(x, y)
    boot = [spearmanr(x[rng.integers(0, n, n)], y[rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(rho_obs), float(ci_lo), float(ci_hi)


# ── Figure layout ──────────────────────────────────────────────────────────────
# Top 55%:  10 per-family geodesic heatmaps in a 2×5 grid
# Bottom 45%: side-by-side 10×10 baseline heatmaps + Spearman ρ bar chart
#   columns: [Geodesic] [Pfam JSD] [Seq Identity] [PANTHER] [Bar chart]

fig = plt.figure(figsize=(28, 16), dpi=300)
gs_outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.44)

gs_top = gs_outer[0].subgridspec(2, 5, hspace=0.72, wspace=0.55)
gs_bot = gs_outer[1].subgridspec(1, 5, wspace=0.42)

ax_geo_fam   = fig.add_subplot(gs_bot[0])
ax_jsd       = fig.add_subplot(gs_bot[1])
ax_seqid     = fig.add_subplot(gs_bot[2])
ax_panther   = fig.add_subplot(gs_bot[3])
ax_bar       = fig.add_subplot(gs_bot[4])

# ── Panel A: per-family geodesic heatmaps ─────────────────────────────────────

_within_vals = []
for fam in family_order:
    idx = np.where(families_arr == fam)[0]
    sub = geodesic[np.ix_(idx, idx)]
    _within_vals.extend(sub[np.triu_indices(len(idx), k=1)])
vmax_within = float(np.percentile(_within_vals, 95))

for fi, fam in enumerate(family_order):
    row, col = fi // 5, fi % 5
    ax = fig.add_subplot(gs_top[row, col])

    fam_idx = np.where(families_arr == fam)[0]
    fam_genes = [gene_names[i] for i in fam_idx]
    sub_geo = geodesic[np.ix_(fam_idx, fam_idx)]

    norm = Normalize(vmin=0, vmax=vmax_within)
    ax.imshow(sub_geo, cmap="YlOrRd", norm=norm, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(len(fam_genes)))
    ax.set_yticks(range(len(fam_genes)))
    ax.set_xticklabels(fam_genes, rotation=90, fontsize=4.5)
    ax.set_yticklabels(fam_genes, fontsize=4.5)
    ax.tick_params(length=2, pad=1)

    color = FAMILY_COLORS[fam]
    ax.set_title(fam.replace("_", " ").title(), fontsize=7, color=color, fontweight="bold", pad=8)

    strip = ax.inset_axes([0, 1.04, 1, 0.07], transform=ax.transAxes)
    strip.set_xlim(0, 1)
    strip.set_ylim(0, 1)
    strip.axis("off")
    strip.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color, linewidth=0))

    # Annotate within-family Spearman ρ vs seq identity if available
    if seqid_genes is not None and len(fam_idx) >= 3:
        tri = np.triu_indices(len(fam_idx), k=1)
        geo_pairs = sub_geo[tri]
        si_sub = seqid_genes.values[np.ix_(fam_idx, fam_idx)]
        si_pairs = (1.0 - si_sub)[tri]
        rho_w, _ = spearmanr(geo_pairs, si_pairs)
        ax.text(
            0.97, 0.03, f"ρ={rho_w:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
        )

cb_ax = fig.add_axes([0.922, 0.57, 0.006, 0.33])
sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=0, vmax=vmax_within))
sm.set_array([])
cb = fig.colorbar(sm, cax=cb_ax)
cb.set_label("Geodesic distance", fontsize=7)
cb.ax.tick_params(labelsize=6)

fig.text(0.02, 0.97, "A  Within-family geodesic distances", fontsize=10, fontweight="bold", va="top")


# ── Helper: draw a 10×10 family-level heatmap ─────────────────────────────────

def draw_family_heatmap(ax, matrix, title, cmap, vmin=None, vmax=None,
                        cbar_label="", show_yticks=True, tbd=False):
    fam_labels = [f.replace("_", " ").title() for f in family_order]
    if tbd:
        ax.set_facecolor("#F5F5F5")
        ax.text(0.5, 0.5, "TBD", transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#AAAAAA", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        norm = Normalize(vmin=vmin if vmin is not None else matrix.min(),
                         vmax=vmax if vmax is not None else matrix.max())
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(N_fam))
        ax.set_yticks(range(N_fam) if show_yticks else [])
        ax.set_xticklabels(fam_labels, rotation=40, ha="right", fontsize=5.5)
        if show_yticks:
            ax.set_yticklabels(fam_labels, fontsize=5.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label(cbar_label, fontsize=6)
        cb.ax.tick_params(labelsize=5)

        # Family color ticks on x-axis
        for xi, fam in enumerate(family_order):
            ax.add_patch(mpatches.Rectangle(
                (xi - 0.5, N_fam - 0.5), 1, 0.22,
                color=FAMILY_COLORS[fam], transform=ax.transData,
                clip_on=False, linewidth=0,
            ))

    ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
    ax.spines[["top", "right", "bottom", "left"]].set_linewidth(0.5)


# ── Panel B: geodesic centroid ─────────────────────────────────────────────────
draw_family_heatmap(
    ax_geo_fam, dist_centroid.values,
    "B  Geodesic centroid\n(model)", "YlOrRd",
    cbar_label="Mean geodesic dist.",
)

# ── Panel C: Pfam JSD ─────────────────────────────────────────────────────────
draw_family_heatmap(
    ax_jsd, dist_jsd.values,
    "C  Pfam HMM JSD\n(baseline 1)", "Blues",
    cbar_label="JSD", show_yticks=False,
)

# ── Panel D: Sequence identity ─────────────────────────────────────────────────
if dist_seqid_fam is not None:
    draw_family_heatmap(
        ax_seqid, 1.0 - dist_seqid_fam.values,
        "D  Seq identity distance\n(baseline 2)", "Greens",
        cbar_label="1 − mean identity", show_yticks=False,
    )
else:
    draw_family_heatmap(ax_seqid, None, "D  Seq identity\n(baseline 2)", None, tbd=True)

# ── Panel E: PANTHER / TimeTree ────────────────────────────────────────────────
if dist_panther is not None:
    draw_family_heatmap(
        ax_panther, dist_panther.values,
        "E  PANTHER / TimeTree\n(baseline 3)", "Purples",
        cbar_label="Branch length / Mya", show_yticks=False,
    )
else:
    draw_family_heatmap(ax_panther, None, "E  PANTHER / TimeTree\n(baseline 3)", None, tbd=True)

# ── Panel F: Spearman ρ bar chart ─────────────────────────────────────────────

baselines_bar = [("Pfam JSD\n(baseline 1)", dist_jsd.values, "#2C6E8A")]
if dist_seqid_fam is not None:
    baselines_bar.append(("Seq identity\n(baseline 2)", 1.0 - dist_seqid_fam.values, "#E76F51"))
else:
    baselines_bar.append(("Seq identity\n(baseline 2)", None, "#BDBDBD"))
if dist_panther is not None:
    baselines_bar.append(("PANTHER\n(baseline 3)", dist_panther.values, "#9C4DC4"))
else:
    baselines_bar.append(("PANTHER / TimeTree\n(baseline 3)", None, "#BDBDBD"))

x = np.arange(len(baselines_bar))
for xi, (label, mat, color) in enumerate(baselines_bar):
    if mat is not None:
        y = mat[idx_upper]
        rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, y)
        ax_bar.bar(xi, rho, color=color, width=0.52, zorder=3, edgecolor="white", linewidth=0.5)
        ax_bar.errorbar(xi, rho, yerr=[[rho - ci_lo], [ci_hi - rho]],
                        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        ax_bar.text(xi, ci_hi + 0.04, f"ρ={rho:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax_bar.bar(xi, 0, color=color, width=0.52, zorder=3)
        ax_bar.text(xi, 0.04, "TBD", ha="center", va="bottom", fontsize=8, color="#888888")

ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([b[0] for b in baselines_bar], fontsize=7)
ax_bar.set_ylim(-1, 1)
ax_bar.set_ylabel("Spearman ρ  (centroid geodesic vs. baseline)", fontsize=7.5)
ax_bar.set_title("F  Baseline correlation\n(GPN-Star Vertebrate)", fontsize=8, fontweight="bold", pad=6)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

# ── Save ───────────────────────────────────────────────────────────────────────

fig.savefig(RUN_DIR / "figure2.pdf", dpi=300)
fig.savefig(RUN_DIR / "figure2.png", dpi=300)
print(f"Saved figures to {RUN_DIR}/figure2.{{pdf,png}}")

print("\nBaseline Spearman ρ vs centroid geodesic:")
for label, mat, _ in baselines_bar:
    if mat is not None:
        rho, ci_lo, ci_hi = bootstrap_spearman(x_geo, mat[idx_upper])
        print(f"  {label.replace(chr(10), ' ')}: ρ={rho:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    else:
        print(f"  {label.replace(chr(10), ' ')}: TBD")
