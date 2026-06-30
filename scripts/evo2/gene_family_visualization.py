"""Visualizations for the Evo2 cross-kingdom gene-family geodesic analysis.

The direct analog of scripts/gpnstar/gpnstar_visualization.py, adapted for the Evo2
pipeline's data: ~400 CDS/family (vs ~31 human paralogs), so the within-family
heatmaps are ordered by host taxonomy (domain → group) with a domain colour strip
instead of unreadable per-gene labels. The ground-truth baselines map across cleanly:

  GPN-Star CDS seq-identity   →  Evo2 k-mer sequence divergence   (within-kmer figure)
  GPN-Star Pfam/PANTHER phylo →  Evo2 taxonomic-rank distance     (within-taxonomy figure)

Figure sets, selectable via --figures (default: all):

  between            : family-level (F×F) geodesic-centroid / Pfam JSD / k-mer heatmaps
                       (Axis A) + a between-family ρ bar chart (geodesic centroid vs each
                       baseline with bootstrap CIs). Pfam JSD is the family-intrinsic
                       baseline shared with GPN-Star (run scripts/baselines/pfam_hmm_jsd.py first).
                       Taxonomic distance is computed but not plotted here (see the within
                       figures).   -> between_comparison.{pdf,png}
  within-heatmaps    : rows of per-family within-family heatmaps (geodesic, k-mer,
                       taxonomy), taxonomy-ordered, no bar charts.
                       -> within_heatmaps.{pdf,png}
  within-correlations: per-family within-family Spearman ρ bars (geodesic vs k-mer &
                       taxonomy).   -> within_correlations.{pdf,png}

Usage:
    uv run python scripts/evo2/gene_family_visualization.py \
        --run-dir results/YYYY-MM-DD_evo2-gene-families
    uv run python scripts/evo2/gene_family_visualization.py --run-dir RESULTS_DIR --figures between
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gene_families import family_colors  # noqa: E402
from plot_utils import (  # noqa: E402
    WITHIN_PALETTE,
    between_rho_bars,
    draw_family_heatmap,
    grouped_rho_bars,
    set_pub_style,
    within_csv_series,
)

# Embedding metadata (family, domain, group per org_gene) lives next to the cache,
# not in the dated results dir; fall back to it when run_dir has no copy.
EMBED_META_FALLBACK = Path("data/evo2_gene_families/embeddings/metadata.csv")

FAMILY_COLORS = family_colors("evo2")
# Domain colour strip — the bacterial → archaeal → eukaryotic gradient that Axis B
# expects the within-family geodesic to recapitulate.
DOMAIN_COLORS = {
    "Bacteria": "#56B4E9",
    "Archaea": "#E69F00",
    "Eukaryota": "#009E73",
    "Unknown": "#BBBBBB",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def _meta_path_for(run_dir: Path) -> Path:
    """Resolve the metadata CSV: run-dir copy (human panel) else the embeddings
    fallback (cross-kingdom families write metadata only beside the cache)."""
    p = run_dir / "metadata.csv"
    return p if p.exists() else EMBED_META_FALLBACK


def detect_panel(run_dir: Path) -> str:
    """Infer which gene panel a run belongs to from its metadata schema.

    The cross-kingdom families panel carries a per-CDS ``domain`` column (organisms
    span Bacteria/Archaea/Eukaryota); the matched human paralog panel (Evo2 *and*
    GPN-Star) is all-human and writes only ``gene,family``. Falls back to "evo2".
    """
    mp = _meta_path_for(run_dir)
    if not mp.exists():
        return "evo2"
    cols = pd.read_csv(mp, nrows=0).columns
    return "evo2" if "domain" in cols else "human"


def load_run(run_dir: Path):
    """Load geodesic (org_gene-ordered) + aligned per-CDS metadata for a run dir."""
    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv found in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    org_genes = df_geo.index.tolist()

    meta_path = _meta_path_for(run_dir)
    if not meta_path.exists():
        sys.exit(f"No metadata.csv in {run_dir} or {EMBED_META_FALLBACK}")
    meta = pd.read_csv(meta_path)
    # Align metadata to the geodesic row order via the gene-label column (the matrices
    # are stored in manifest order; positional mismatch would silently mislabel
    # everything). The cross-kingdom families pipeline names this column "org_gene";
    # the matched human-panel / GPN-Star pipeline names it "gene". Accept either and
    # normalise to "org_gene" so the downstream figures are schema-agnostic.
    label_col = "org_gene" if "org_gene" in meta.columns else "gene"
    meta = (
        meta.set_index(label_col)
        .reindex(org_genes)
        .reset_index()
        .rename(columns={label_col: "org_gene"})
    )
    if meta["family"].isna().any():
        sys.exit("metadata.csv does not cover every org_gene in the geodesic matrix")

    families = meta["family"].to_numpy()
    family_order = sorted(pd.unique(families).tolist())
    return geodesic, org_genes, meta, families, family_order


def family_order_idx(meta_fam: pd.DataFrame) -> np.ndarray:
    """Local row order within a family: group taxonomically (domain → group → org).

    The matched human panel carries no taxonomy columns (every gene is human), so
    there we fall back to ordering by gene symbol — still a stable, readable order.
    """
    if {"domain", "group"}.issubset(meta_fam.columns):
        keys = [
            f"{d}|{g}|{o}"
            for d, g, o in zip(
                meta_fam["domain"], meta_fam["group"], meta_fam["org_gene"], strict=False
            )
        ]
    else:
        keys = list(meta_fam["org_gene"])
    return np.argsort(keys, kind="stable")


def aggregate_family_matrix(mat, families, family_order):
    """F×F: within-family mean (upper-tri) on the diagonal, cross-block mean off it."""
    F = len(family_order)
    out = np.zeros((F, F), dtype=float)
    for fi, fa in enumerate(family_order):
        ia = np.where(families == fa)[0]
        for fj, fb in enumerate(family_order):
            ib = np.where(families == fb)[0]
            if fi == fj:
                tri = mat[np.ix_(ia, ia)][np.triu_indices(len(ia), k=1)]
                out[fi, fj] = tri.mean() if len(tri) else 0.0
            else:
                out[fi, fj] = mat[np.ix_(ia, ib)].mean()
    return out


def domain_strip(ax, domains, where="top"):
    """Draw a per-row/col domain colour strip alongside an ordered heatmap."""
    n = len(domains)
    if where == "top":
        strip = ax.inset_axes([0, 1.01, 1, 0.035], transform=ax.transAxes)
        for i, d in enumerate(domains):
            strip.add_patch(
                mpatches.Rectangle(
                    (i / n, 0), 1 / n, 1, color=DOMAIN_COLORS.get(d, "#BBBBBB"), linewidth=0
                )
            )
    else:  # left
        strip = ax.inset_axes([-0.05, 0, 0.035, 1], transform=ax.transAxes)
        for i, d in enumerate(domains):
            strip.add_patch(
                mpatches.Rectangle(
                    (0, 1 - (i + 1) / n),
                    1,
                    1 / n,
                    color=DOMAIN_COLORS.get(d, "#BBBBBB"),
                    linewidth=0,
                )
            )
    strip.set_xlim(0, 1)
    strip.set_ylim(0, 1)
    strip.axis("off")


# ── Per-family within-family ρ (read from the analysis CSV when present) ───────


def load_axisB(run_dir: Path) -> pd.DataFrame | None:
    """Per-family within-family k-mer ρ. The cross-kingdom families pipeline writes it
    as axisB_within_family_correlations.csv; the matched human paralog pipeline writes
    the same family/spearman_geodesic_kmer/p_kmer columns as kmer_within_family_correlations.csv."""
    for name in ("axisB_within_family_correlations.csv", "kmer_within_family_correlations.csv"):
        p = run_dir / name
        if p.exists():
            return pd.read_csv(p)
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Within-family figures: heatmaps (per family) and the per-family Spearman ρ bars
# ════════════════════════════════════════════════════════════════════════════════

# Within-family baselines the geodesic is compared against. Each baseline is a
# per-CDS distance matrix (.npy) with a matching column in axisB_*_correlations.csv.
WITHIN_BASELINES = [
    {
        "npy": "kmer_distance.npy",
        "cmap": "Greens",
        "row_label": "k-mer divergence\n(1 − cosine)",
        "cbar_label": "k-mer distance",
        "color": "#2D6A4F",
        "short": "vs k-mer divergence",
        "rho_col": "spearman_geodesic_kmer",
        "p_col": "p_kmer",
    },
    {
        "npy": "taxonomic_distance.npy",
        "cmap": "Purples",
        "row_label": "Taxonomic-rank\ndistance",
        "cbar_label": "taxonomic distance",
        "color": "#6A4C93",
        "short": "vs taxonomy",
        "rho_col": "spearman_geodesic_taxonomy",
        "p_col": "p_taxonomy",
    },
]


def make_within_heatmaps(run_dir: Path) -> None:
    """One figure: rows = geodesic + each within-family baseline, columns = families.

    Per-family heatmaps are ordered by host taxonomy (domain → group), with a domain
    colour strip on top. No bar charts — the per-family ρ summary lives in the
    within_correlations figure.
    """
    set_pub_style()
    geodesic, org_genes, meta, families, family_order = load_run(run_dir)
    N_fam = len(family_order)

    # Rows: geodesic first, then whichever baselines are present on disk.
    rows = [("Geodesic\n(Evo2)", geodesic, "YlOrRd", "#A8330E")]
    for cfg in WITHIN_BASELINES:
        p = run_dir / cfg["npy"]
        if p.exists():
            rows.append((cfg["row_label"], np.load(p), cfg["cmap"], cfg["color"]))
    n_rows = len(rows)

    # Shared robust colour scale per row (95th pct of within-family pairs).
    def shared_vmax(mat):
        vals = []
        for fam in family_order:
            idx = np.where(families == fam)[0]
            vals.extend(mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])
        return float(np.percentile(vals, 95)) if vals else 1.0

    vmaxes = [shared_vmax(mat) for _, mat, _, _ in rows]
    # Per-family taxonomy ordering (shared across rows).
    orders = {
        fam: np.where(families == fam)[0][family_order_idx(meta.iloc[np.where(families == fam)[0]])]
        for fam in family_order
    }

    fig = plt.figure(figsize=(4 * N_fam, 3.6 * n_rows), dpi=300)
    gs = fig.add_gridspec(n_rows, N_fam, hspace=0.3, wspace=0.35)
    for ri, (label, mat, cmap, color) in enumerate(rows):
        row_axes = []
        for fi, fam in enumerate(family_order):
            ax = fig.add_subplot(gs[ri, fi])
            row_axes.append(ax)
            oidx = orders[fam]
            ax.imshow(
                mat[np.ix_(oidx, oidx)],
                cmap=cmap,
                norm=Normalize(0, vmaxes[ri]),
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if "domain" in meta.columns:  # no taxonomy strip for the all-human panel
                domain_strip(ax, meta.iloc[oidx]["domain"].tolist(), "top")
            if ri == 0:
                ax.set_title(
                    fam.replace("_", " ").title(),
                    fontsize=8,
                    color=FAMILY_COLORS[fam],
                    fontweight="bold",
                    pad=12,
                )
            if fi == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold", color=color, labelpad=8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vmaxes[ri]))
        sm.set_array([])
        cb = fig.colorbar(
            sm, ax=row_axes, orientation="vertical", fraction=0.006, pad=0.02, shrink=0.7
        )
        cb.ax.tick_params(labelsize=6)

    handles = [mpatches.Patch(color=c, label=d) for d, c in DOMAIN_COLORS.items()]
    fig.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        ncol=4,
        title="Host domain (heatmap order)",
        title_fontsize=7,
        framealpha=0.9,
    )
    fig.suptitle(
        "Within-family geodesic vs baseline distances (per family, taxonomy-ordered)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(run_dir / "within_heatmaps.pdf", dpi=300)
    fig.savefig(run_dir / "within_heatmaps.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/within_heatmaps.{{pdf,png}}")


def make_within_correlations(run_dir: Path) -> None:
    """One figure: the three standardized within-family baselines (identical across the Evo2
    and GPN-Star pipelines) — k-mer composition, sequence identity, patristic tree — as
    per-family Spearman ρ. (Taxonomy is demoted to the supplementary within_heatmaps figure.)"""
    set_pub_style(title_size=9, tick_size=7)
    axisB = load_axisB(run_dir)
    if axisB is None:
        print(f"  SKIP within_correlations: no axisB CSV in {run_dir}")
        return
    family_order = sorted(pd.unique(axisB["family"]).tolist())
    ab = axisB.set_index("family")

    def col(name):
        return [ab.loc[f, name] if f in ab.index else np.nan for f in family_order]

    # k-mer composition (alignment-free) from the embedding pipeline's axisB CSV, then the
    # two alignment-based baselines (seq identity, patristic) from the shared CSVs.
    series = [("vs k-mer composition", col("spearman_geodesic_kmer"), col("p_kmer"),
               WITHIN_PALETTE["kmer"])]
    series += within_csv_series(run_dir, family_order)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    grouped_rho_bars(
        ax,
        family_order,
        series,
        "Within-family: geodesic vs k-mer composition, sequence identity & patristic tree (per family)",
        "Within-family Spearman ρ",
        ylim=(-0.2, 1.0),
    )
    fig.savefig(run_dir / "within_correlations.pdf", dpi=300)
    fig.savefig(run_dir / "within_correlations.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/within_correlations.{{pdf,png}}")


# ════════════════════════════════════════════════════════════════════════════════
# Between-family comparison: Axis A (family-level) heatmaps + Axis B ρ bar chart
# ════════════════════════════════════════════════════════════════════════════════


def make_between_figure(run_dir: Path) -> None:
    set_pub_style(title_size=9, tick_size=7)
    geodesic, org_genes, meta, families, family_order = load_run(run_dir)
    N_fam = len(family_order)

    kmer = (
        np.load(run_dir / "kmer_distance.npy") if (run_dir / "kmer_distance.npy").exists() else None
    )
    # Taxonomic distance is still computed by the pipeline and shown in the within
    # figures; it is intentionally not plotted in this between-family figure.

    cen_path = list(run_dir.glob("*_centroid_distances.csv"))
    centroid = (
        pd.read_csv(cen_path[0], index_col=0).values
        if cen_path
        else aggregate_family_matrix(geodesic, families, family_order)
    )
    kmer_fam = aggregate_family_matrix(kmer, families, family_order) if kmer is not None else None

    # Pfam HMM JSD — family-intrinsic between-family baseline shared with GPN-Star
    # (scripts/baselines/pfam_hmm_jsd.py). Reindexed to the geodesic's family order.
    jsd_path = run_dir / "pfam_jsd_distances.csv"
    pfam_fam = (
        pd.read_csv(jsd_path, index_col=0).reindex(index=family_order, columns=family_order).values
        if jsd_path.exists()
        else None
    )

    # Single row: family-level (Axis A) heatmaps + a between-family ρ bar chart
    # (geodesic centroid vs each baseline with bootstrap CIs).
    fig = plt.figure(figsize=(22, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 4, wspace=0.45)
    ax_cen = fig.add_subplot(gs[0])
    ax_pfam = fig.add_subplot(gs[1])
    ax_kmer = fig.add_subplot(gs[2])
    ax_between = fig.add_subplot(gs[3])

    # ── Panels A–C: family-level (Axis A) heatmaps ──────────────────────────────
    draw_family_heatmap(
        fig, ax_cen, centroid, "A  Geodesic centroid (Evo2)",
        "YlOrRd", family_order, FAMILY_COLORS, cbar_label="mean geodesic",
    )
    draw_family_heatmap(
        fig, ax_pfam, pfam_fam, "B  Pfam HMM JSD\n(shared baseline)",
        "Blues", family_order, FAMILY_COLORS, cbar_label="JSD", show_yticks=False,
    )
    draw_family_heatmap(
        fig, ax_kmer, kmer_fam, "C  k-mer divergence",
        "Greens", family_order, FAMILY_COLORS, cbar_label="mean k-mer dist.", show_yticks=False,
    )

    # ── Panel D: between-family ρ (centroid geodesic vs each baseline) + CI ──────
    idx_upper = np.triu_indices(N_fam, k=1)
    x_geo = centroid[idx_upper]
    between_bars = [
        ("Pfam JSD\n(shared)", pfam_fam, "#2C6E8A"),
        ("k-mer\ndivergence", kmer_fam, "#2D6A4F"),
    ]
    between_rho_bars(
        ax_between, between_bars, x_geo, idx_upper,
        "D  Between-family:\ngeodesic vs baselines",
        "Spearman ρ  (centroid geodesic vs baseline)",
        ylim=(-1, 1),
    )

    fig.savefig(run_dir / "between_comparison.pdf", dpi=300)
    fig.savefig(run_dir / "between_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/between_comparison.{{pdf,png}}")


# ════════════════════════════════════════════════════════════════════════════════
# Multi-axis between-family baselines: which axis does the geometry track?
# ════════════════════════════════════════════════════════════════════════════════

# Axis groupings + display colours for the between_family_baseline_scores.csv bars.
AXIS_COLORS = {
    "1_homology": "#C1666B",
    "2_mechanism": "#48A9A6",
    "control": "#9AA0A6",
}
AXIS_LABELS = {
    "1_homology": "Homology",
    "2_mechanism": "Mechanism / chemistry",
    "control": "Composition (control)",
}


def make_axis_scores_figure(run_dir: Path) -> None:
    """Bar chart of between-family ρ per baseline, grouped by axis (homology /
    mechanism / control). The core 'which kind of relatedness does the
    latent geometry encode' readout. Reads between_family_baseline_scores.csv
    (scripts/baselines/between_family_baselines.py); * marks Mantel p < 0.05."""
    set_pub_style(title_size=10, tick_size=8)
    sp = run_dir / "between_family_baseline_scores.csv"
    if not sp.exists():
        print(f"  SKIP between-axes: no between_family_baseline_scores.csv in {run_dir}")
        return
    scores = pd.read_csv(sp)
    # Order: by axis, then descending |ρ| within axis; drop uninformative (NaN ρ) bars
    # but note them in the axis-tick labels so the reader sees they were attempted.
    axis_order = ["1_homology", "2_mechanism", "control"]
    scores["axis_rank"] = scores["axis"].map({a: i for i, a in enumerate(axis_order)})
    scores = scores.sort_values(["axis_rank", "baseline"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    xs, labels, colors, dropped = [], [], [], []
    x = 0
    for axis in axis_order:
        block = scores[scores["axis"] == axis]
        for _, r in block.iterrows():
            if not np.isfinite(r["spearman_rho"]):
                dropped.append(r["baseline"])
                continue
            ax.bar(x, r["spearman_rho"], color=AXIS_COLORS[axis], width=0.8,
                   edgecolor="black", linewidth=0.5)
            if np.isfinite(r["p_mantel"]) and r["p_mantel"] < 0.05:
                y = r["spearman_rho"]
                ax.text(x, y + (0.03 if y >= 0 else -0.06), "*", ha="center",
                        fontsize=14, fontweight="bold")
            labels.append(r["baseline"].replace("_", "\n"))
            colors.append(AXIS_COLORS[axis])
            xs.append(x)
            x += 1
        x += 0.6  # gap between axis groups

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Between-family Spearman ρ\n(family-centroid geodesic vs baseline)")
    ax.set_ylim(-1, 1)
    model = "GPN-Star" if "gpnstar" in run_dir.name.lower() else "Evo2"
    title = f"Which axis does the {model} between-family geometry track?"
    if dropped:
        title += f"\n(uninformative on this panel, omitted: {', '.join(dropped)})"
    ax.set_title(title, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=AXIS_LABELS[a]) for a, c in AXIS_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9,
              title="Baseline axis", title_fontsize=8)
    ax.text(0.01, 0.02, "* Mantel p < 0.05", transform=ax.transAxes, fontsize=7,
            style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(run_dir / "between_axis_scores.pdf", dpi=300)
    fig.savefig(run_dir / "between_axis_scores.png", dpi=300)
    plt.close(fig)
    print(f"Saved {run_dir}/between_axis_scores.{{pdf,png}}")


AXIS_ORDER = ["1_homology", "2_mechanism", "control"]


def _abbrev(name: str) -> str:
    return name.replace("carbonic_anhydrase", "CA").replace("_", " ")


def make_between_matrix_gallery(run_dir: Path) -> None:
    """Gallery of F×F between-family distance heatmaps — the latent geodesic centroid
    plus every ground-truth baseline, grouped by axis. Each matrix is rank-normalised
    (off-diagonal) so block structure is visually comparable across panels regardless of
    scale; the diagonal is masked. Reads family_centroid_distances.csv + betweenfam_*_
    distances.csv + between_family_baseline_scores.csv. The visual answer to 'which
    baseline's structure matches the geodesic'."""
    from scipy.stats import rankdata
    set_pub_style(title_size=9, tick_size=6)

    cen = list(run_dir.glob("*_centroid_distances.csv"))
    if not cen:
        print(f"  SKIP gallery: no *_centroid_distances.csv in {run_dir}")
        return
    geo = pd.read_csv(cen[0], index_col=0)
    fams = geo.index.tolist()
    labels = [_abbrev(f) for f in fams]

    scores = {}
    sp = run_dir / "between_family_baseline_scores.csv"
    if sp.exists():
        sdf = pd.read_csv(sp)
        scores = {r.baseline: (r.axis, r.spearman_rho) for _, r in sdf.iterrows()}

    panels = [("geodesic centroid", geo.values, "(latent)")]
    for axis in AXIS_ORDER:
        for f in sorted(run_dir.glob("betweenfam_*_distances.csv")):
            name = f.stem[len("betweenfam_"):-len("_distances")]
            ax_tag, rho = scores.get(name, (None, np.nan))
            if ax_tag != axis:
                continue
            M = pd.read_csv(f, index_col=0).reindex(index=fams, columns=fams).values
            sub = f"ρ={rho:+.2f}" if np.isfinite(rho) else "(uninformative)"
            panels.append((name.replace("_", " "), M, sub))

    def rank_norm(M):
        D = M.astype(float).copy()
        np.fill_diagonal(D, np.nan)
        iu = np.triu_indices(len(D), k=1)
        u = D[iu]
        ok = ~np.isnan(u)
        if ok.sum() == 0 or np.ptp(u[ok]) == 0:
            return D  # constant/empty → leave as-is (will render flat)
        r = np.full(u.shape, np.nan)
        r[ok] = (rankdata(u[ok]) - 1) / (ok.sum() - 1)
        out = np.full_like(D, np.nan)
        out[iu] = r
        out.T[iu] = r
        return out

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.0 * nrows), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for k, (title, M, sub) in enumerate(panels):
        ax = axes[k]
        im = ax.imshow(rank_norm(M), cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{title}\n{sub}", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(fams)))
        ax.set_yticks(range(len(fams)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
    for k in range(len(panels), len(axes)):
        axes[k].axis("off")
    model = "GPN-Star" if "gpnstar" in run_dir.name.lower() else "Evo2"
    fig.suptitle(f"{model}: between-family distance matrices (rank-normalised, diagonal masked)",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.012, pad=0.02,
                 label="within-matrix rank  (0 = closest pair, 1 = farthest)")
    fig.savefig(run_dir / "between_matrix_gallery.pdf", dpi=200)
    fig.savefig(run_dir / "between_matrix_gallery.png", dpi=200)
    plt.close(fig)
    print(f"Saved {run_dir}/between_matrix_gallery.{{pdf,png}}")


def make_convergent_rank_figure(run_dir: Path) -> None:
    """Heatmap of the convergent-pair rank test: rows = curated family pairs, columns =
    geodesic + each biological axis, colour = percentile closeness (0=closest green,
    100=farthest red). A pair that is green under geodesic but red under homology is the
    'analogy beyond homology' signal. Reads convergent_pair_ranks.csv."""
    set_pub_style(title_size=9, tick_size=7)
    cp = run_dir / "convergent_pair_ranks.csv"
    if not cp.exists():
        print(f"  SKIP convergent-ranks: no convergent_pair_ranks.csv in {run_dir}")
        return
    df = pd.read_csv(cp)
    cols = [c for c in ["geodesic_pctile", "homology_tier_pctile", "cofactor_pctile",
                        "ec_number_pctile", "kmer_pctile"]
            if c in df.columns and df[c].notna().any()]
    M = df[cols].values.astype(float)
    rowlabels = [f"{_abbrev(a)} ↔ {_abbrev(b)}  [{r.replace('_', ' ')}]"
                 for a, b, r in zip(df.family_a, df.family_b, df.relationship)]
    collabels = [c.replace("_pctile", "").replace("_", " ") for c in cols]

    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 4, 0.5 * len(df) + 2), dpi=200)
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(collabels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(rowlabels, fontsize=7)
    for i in range(len(df)):
        for j in range(len(cols)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if (v < 25 or v > 75) else "black")
    # Separate geodesic from the baselines with a vertical rule.
    ax.axvline(0.5, color="black", linewidth=1.5)
    model = "GPN-Star" if "gpnstar" in run_dir.name.lower() else "Evo2"
    ax.set_title(f"{model}: convergent-pair rank test\n"
                 "percentile closeness among all family pairs (0=closest, 100=farthest)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="percentile (closeness)")
    fig.tight_layout()
    fig.savefig(run_dir / "convergent_pair_ranks.pdf", dpi=200)
    fig.savefig(run_dir / "convergent_pair_ranks.png", dpi=200)
    plt.close(fig)
    print(f"Saved {run_dir}/convergent_pair_ranks.{{pdf,png}}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

FIGURES = {
    "between": lambda run_dir: make_between_figure(run_dir),
    "between-axes": lambda run_dir: make_axis_scores_figure(run_dir),
    "between-matrices": lambda run_dir: make_between_matrix_gallery(run_dir),
    "convergent-ranks": lambda run_dir: make_convergent_rank_figure(run_dir),
    "within-heatmaps": lambda run_dir: make_within_heatmaps(run_dir),
    "within-correlations": lambda run_dir: make_within_correlations(run_dir),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run-dir", required=True, help="Dated results folder")
    p.add_argument("--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    p.add_argument(
        "--panel",
        choices=["human", "evo2", "auto"],
        default="auto",
        help="Which gene panel's colour palette to use. 'auto' (default) infers it "
        "from the run's metadata schema (human paralog vs cross-kingdom families).",
    )
    return p.parse_args()


def main() -> None:
    global FAMILY_COLORS
    args = parse_args()
    run_dir = Path(args.run_dir)
    panel = detect_panel(run_dir) if args.panel == "auto" else args.panel
    FAMILY_COLORS = family_colors(panel)
    print(f"panel={panel}  ({len(FAMILY_COLORS)} family colours)")
    for name in args.figures:
        print(f"\n=== {name} ===")
        FIGURES[name](run_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
