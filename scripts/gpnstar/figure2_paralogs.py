"""
Within-family comparison: geodesic distances vs. Ensembl Compara paralog protein identity.

Layout (2 rows × 10 columns):
  Row 1: within-family geodesic heatmaps  (model output)
  Row 2: within-family Ensembl Compara protein identity heatmaps  (evolutionary baseline)
         Gray cells = gene pair not in Ensembl paralog database
  Bottom bar chart: per-family Spearman ρ (geodesic vs. paralog distance)

Note: PANTHER family tree API is not publicly accessible in PANTHER v19.
Ensembl Compara paralog perc_id is derived from the same protein MSA pipeline that
underlies PANTHER trees and is the closest available programmatic proxy.

Requires (from embed_and_geodesic.py step 7c):
  *_geodesic_labeled.csv
  ensembl_paralog_identity_genes.csv
  metadata.csv  /  family_order.txt
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, ListedColormap
from scipy.stats import spearmanr

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


args = parse_args()
RUN_DIR = Path(args.run_dir)

# ── Load data ──────────────────────────────────────────────────────────────────

_geo_files = list(RUN_DIR.glob("*_geodesic_labeled.csv"))
if not _geo_files:
    print("No *_geodesic_labeled.csv found", file=sys.stderr); sys.exit(1)
df_geo = pd.read_csv(_geo_files[0], index_col=0)
geodesic = df_geo.values
gene_names = df_geo.index.tolist()

_paralog_path = RUN_DIR / "ensembl_paralog_identity_genes.csv"
if not _paralog_path.exists():
    print("ensembl_paralog_identity_genes.csv not found — run embed_and_geodesic.py first", file=sys.stderr)
    sys.exit(1)
paralog_mat = pd.read_csv(_paralog_path, index_col=0).values
# Convert identity (%) to distance: 1 - perc_id/100; NaN preserved
paralog_dist = np.where(np.isfinite(paralog_mat), 1.0 - paralog_mat / 100.0, np.nan)
np.fill_diagonal(paralog_dist, 0.0)

meta_df = pd.read_csv(RUN_DIR / "metadata.csv")
families_arr = meta_df["family"].values
family_order = (RUN_DIR / "family_order.txt").read_text().strip().splitlines()
N_fam = len(family_order)

# ── Per-family sub-matrices ────────────────────────────────────────────────────

def fam_sub(mat, fam):
    idx = np.where(families_arr == fam)[0]
    genes = [gene_names[i] for i in idx]
    return idx, genes, mat[np.ix_(idx, idx)]


per_geo  = {f: fam_sub(geodesic,    f) for f in family_order}
per_para = {f: fam_sub(paralog_dist, f) for f in family_order}

# ── Per-family Spearman ρ ─────────────────────────────────────────────────────

per_rho: dict[str, tuple[float, float, int]] = {}
for fam in family_order:
    idx, genes, sub_geo  = per_geo[fam]
    _,   _,    sub_para  = per_para[fam]
    n = len(genes)
    tri = np.triu_indices(n, k=1)
    g_pairs = sub_geo[tri]
    p_pairs = sub_para[tri]
    valid = np.isfinite(p_pairs) & np.isfinite(g_pairs)
    n_valid = int(valid.sum())
    if n_valid < 3:
        per_rho[fam] = (np.nan, np.nan, n_valid)
    else:
        rho, pval = spearmanr(g_pairs[valid], p_pairs[valid])
        per_rho[fam] = (float(rho), float(pval), n_valid)

# ── Shared colour scales ───────────────────────────────────────────────────────

def shared_vmax(per_fam, pct=95):
    vals = []
    for fam in family_order:
        _, genes, sub = per_fam[fam]
        n = len(genes)
        if n >= 2:
            t = sub[np.triu_indices(n, k=1)]
            vals.extend(t[np.isfinite(t)])
    return float(np.percentile(vals, pct)) if vals else 1.0

vmax_geo  = shared_vmax(per_geo)
vmax_para = shared_vmax(per_para)

# ── Figure layout ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(26, 13), dpi=300)
gs_outer = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.48)
gs_heat = gs_outer[0].subgridspec(2, N_fam, hspace=0.65, wspace=0.52)
ax_bar_row = fig.add_subplot(gs_outer[1])


# ── NaN-aware colormap: gray for missing data ──────────────────────────────────

def make_cmap_with_nan(base_cmap_name, nan_color="#DDDDDD"):
    base = plt.get_cmap(base_cmap_name)
    cmap = base.copy()
    cmap.set_bad(color=nan_color)
    return cmap

cmap_geo  = make_cmap_with_nan("YlOrRd")
cmap_para = make_cmap_with_nan("Blues")

# ── Draw per-family heatmaps ───────────────────────────────────────────────────

def draw_hm(ax, sub, genes, cmap, vmax, title, color_strip=None, fontsize=4.5):
    n = len(genes)
    norm = Normalize(vmin=0, vmax=vmax)
    masked = np.ma.array(sub, mask=~np.isfinite(sub))
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(genes, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(genes, fontsize=fontsize)
    ax.tick_params(length=1.5, pad=0.5)
    ax.set_title(title, fontsize=6.5, pad=5)
    if color_strip is not None:
        strip = ax.inset_axes([0, 1.04, 1, 0.07], transform=ax.transAxes)
        strip.set_xlim(0, 1); strip.set_ylim(0, 1); strip.axis("off")
        strip.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=color_strip, linewidth=0))


for fi, fam in enumerate(family_order):
    ax_geo  = fig.add_subplot(gs_heat[0, fi])
    ax_para = fig.add_subplot(gs_heat[1, fi])

    idx, genes, sub_geo  = per_geo[fam]
    _,   _,    sub_para  = per_para[fam]
    color = FAMILY_COLORS[fam]
    label = fam.replace("_", " ").title()

    draw_hm(ax_geo,  sub_geo,  genes, cmap_geo,  vmax_geo,  label, color_strip=color)
    draw_hm(ax_para, sub_para, genes, cmap_para, vmax_para, "")

    # Spearman ρ annotation
    rho, pval, n_valid = per_rho[fam]
    n_pairs = len(genes) * (len(genes) - 1) // 2
    if np.isfinite(rho):
        pstar = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        ann = f"ρ={rho:.2f}{pstar}\n({n_valid}/{n_pairs})"
    else:
        ann = f"n={n_valid}/{n_pairs}"
    ax_para.text(
        0.5, -0.32, ann,
        transform=ax_para.transAxes, ha="center", va="top", fontsize=5.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )

# Row labels
fig.text(0.005, 0.80, "Geodesic\n(model)", fontsize=8, fontweight="bold",
         ha="left", va="center", rotation=90, color="#333333")
fig.text(0.005, 0.57, "Ensembl Compara\n(1 − prot. ID)", fontsize=8, fontweight="bold",
         ha="left", va="center", rotation=90, color="#2C6E8A")

# Shared colorbars
_dummy_geo = [fig.add_subplot(gs_heat[0, fi]) for fi in range(N_fam)]
sm_geo = plt.cm.ScalarMappable(cmap=cmap_geo, norm=Normalize(0, vmax_geo))
sm_geo.set_array([])
cb_geo = fig.colorbar(sm_geo, ax=_dummy_geo, orientation="vertical", fraction=0.007, pad=0.02, shrink=0.7)
cb_geo.set_label("Geodesic dist.", fontsize=6)
cb_geo.ax.tick_params(labelsize=5)

_dummy_para = [fig.add_subplot(gs_heat[1, fi]) for fi in range(N_fam)]
sm_para = plt.cm.ScalarMappable(cmap=cmap_para, norm=Normalize(0, vmax_para))
sm_para.set_array([])
cb_para = fig.colorbar(sm_para, ax=_dummy_para, orientation="vertical", fraction=0.007, pad=0.02, shrink=0.7)
cb_para.set_label("1 − prot. identity", fontsize=6)
cb_para.ax.tick_params(labelsize=5)

# NaN legend patch
nan_patch = mpatches.Patch(color="#DDDDDD", label="No Ensembl\nparalog data")
fig.legend(handles=[nan_patch], loc="lower right", fontsize=6, framealpha=0.9)

# ── Bottom bar chart ──────────────────────────────────────────────────────────

x = np.arange(N_fam)
rhos  = [per_rho[f][0] for f in family_order]
pvals = [per_rho[f][1] for f in family_order]
nvs   = [per_rho[f][2] for f in family_order]
colors_bar = [FAMILY_COLORS[f] for f in family_order]

ax_bar_row.bar(
    x,
    [r if np.isfinite(r) else 0 for r in rhos],
    color=colors_bar, width=0.65, zorder=3, edgecolor="white", linewidth=0.5,
)

for xi, (rho, pval, nv) in enumerate(zip(rhos, pvals, nvs)):
    if np.isfinite(rho):
        ypos = rho + (0.04 if rho >= 0 else -0.09)
        pstar = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        ax_bar_row.text(xi, ypos, f"ρ={rho:.2f}{pstar}", ha="center", va="bottom", fontsize=6.5)
    else:
        label = "n<3" if nv < 3 else "no pairs"
        ax_bar_row.text(xi, 0.04, label, ha="center", va="bottom", fontsize=6.5, color="#999999")

ax_bar_row.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
ax_bar_row.set_xticks(x)
ax_bar_row.set_xticklabels(
    [f.replace("_", " ").title() for f in family_order],
    rotation=30, ha="right", fontsize=7,
)
ax_bar_row.set_ylim(-1, 1.15)
ax_bar_row.set_ylabel("Spearman ρ  (geodesic vs. 1 − protein identity)", fontsize=7.5)
ax_bar_row.set_title(
    "Within-family: GPN-Star geodesic vs. Ensembl Compara protein identity",
    fontsize=9, fontweight="bold", pad=6,
)
ax_bar_row.spines[["top", "right"]].set_visible(False)
ax_bar_row.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────

out_pdf = RUN_DIR / "figure_within_comparison.pdf"
out_png = RUN_DIR / "figure_within_comparison.png"
fig.savefig(out_pdf, dpi=300)
fig.savefig(out_png, dpi=300)
print(f"Saved to {RUN_DIR}/figure_within_comparison.{{pdf,png}}")

print("\nPer-family Spearman ρ (geodesic vs. Ensembl Compara protein identity distance):")
for fam in family_order:
    rho, pval, nv = per_rho[fam]
    _, genes, _ = per_geo[fam]
    n_total = len(genes) * (len(genes) - 1) // 2
    if np.isfinite(rho):
        pstar = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "(n.s.)"
        print(f"  {fam:<22}: ρ={rho:+.3f}  p={pval:.3f} {pstar}  ({nv}/{n_total} pairs)")
    else:
        print(f"  {fam:<22}: insufficient pairs ({nv}/{n_total})")
