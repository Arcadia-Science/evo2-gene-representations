"""
Visualizations for the Evo2 species phylogeny analysis.

Two figure sets, selectable via --figures (default: both):

  correlation : two scatter plots vs. GTDB patristic (phylogenetic) distance,
                following https://www.goodfire.ai/research/phylogeny-manifold —
                  (1) Evo2 embedding cosine similarity vs. phylogenetic distance
                  (2) Evo2 embedding geodesic distance  vs. phylogenetic distance
                each annotated with the Spearman ρ over all species pairs.

  dimreduction: UMAP (cosine metric) and 3D PCA (L2-normalized embeddings) of the
                Evo2 embeddings, one panel per GTDB rank (class/order/family),
                coloring the most abundant groups and lumping the tail into "Other".

Usage:
    uv run python scripts/evo2/evo2_visualization.py
    uv run python scripts/evo2/evo2_visualization.py --figures correlation
    uv run python scripts/evo2/evo2_visualization.py \
        --figures dimreduction --method umap --ranks class order
    uv run python scripts/evo2/evo2_visualization.py --run-dir results/2026-06-12_evo2-species
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geodesic_utils import cosine_similarity_matrix  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

# Embeddings live at a fixed cache path (shared across runs), written by
# embed_and_geodesic_species.py; the per-run matrices live under --run-dir.
EMBED_NPY = Path("data/species/embeddings/evo2_species_embeddings.npy")
EMBED_META = Path("data/species/embeddings/metadata.csv")
MANIFEST = Path("data/species/gtdb_500_manifest.csv")
GTDB_META = Path("data/species/bac120_metadata.tsv.gz")
TAX_CACHE = Path("data/species/taxonomy_446.csv")

RANK_PREFIX = {"phylum": "p__", "class": "c__", "order": "o__", "family": "f__"}


# ── Distance-correlation scatter figures ───────────────────────────────────────


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def scatter_plot(x, y, xlabel, ylabel, title, color, out_stem: Path):
    rho, p = spearmanr(x, y)
    p_str = "p < 1e-300" if p == 0 else f"p = {p:.2e}"

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.scatter(x, y, s=2, alpha=0.04, color=color, linewidths=0, rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=22)
    ax.text(
        0.5,
        1.02,
        f"Spearman ρ = {rho:.3f}   ({p_str},  n = {len(x):,} pairs)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Spearman ρ = {rho:.4f} ({p_str})")
    print(f"  Saved {out_stem.with_suffix('.png')} and .pdf")
    return rho


def make_correlation_figures(run_dir: Path) -> None:
    # ── Load aligned matrices (all in metadata / manifest order) ──────────────
    embeddings = np.load(EMBED_NPY)
    meta = pd.read_csv(EMBED_META)
    geodesic = np.load(run_dir / "evo2_species_geodesic.npy")
    patristic = pd.read_csv(run_dir / "gtdb_patristic_distances.csv", index_col=0).values

    N = len(embeddings)
    assert geodesic.shape == (N, N), f"geodesic {geodesic.shape} != ({N},{N})"
    assert patristic.shape == (N, N), f"patristic {patristic.shape} != ({N},{N})"
    assert len(meta) == N
    print(f"Loaded {N} species (embeddings, geodesic, patristic all aligned)")

    cossim = cosine_similarity_matrix(embeddings)

    # All species are leaves in the tree for the current run, but guard anyway:
    # drop pairs where patristic is exactly 0 off-diagonal (absent taxa).
    pat_tri = upper_triangle(patristic)
    cos_tri = upper_triangle(cossim)
    geo_tri = upper_triangle(geodesic)
    keep = pat_tri > 0
    if not keep.all():
        print(f"  Dropping {(~keep).sum()} pairs with zero patristic distance")
    pat_tri, cos_tri, geo_tri = pat_tri[keep], cos_tri[keep], geo_tri[keep]

    print("\n[correlation: Plot 1] cosine similarity vs phylogenetic distance")
    scatter_plot(
        pat_tri,
        cos_tri,
        xlabel="Phylogenetic distance (GTDB patristic, branch length)",
        ylabel="Evo2 embedding cosine similarity",
        title="Evo2 cosine similarity vs. phylogeny",
        color="#1f77b4",
        out_stem=run_dir / "figure_cossim_vs_phylo",
    )
    print("\n[correlation: Plot 2] geodesic distance vs phylogenetic distance")
    scatter_plot(
        pat_tri,
        geo_tri,
        xlabel="Phylogenetic distance (GTDB patristic, branch length)",
        ylabel="Evo2 embedding geodesic distance (angular)",
        title="Evo2 geodesic distance vs. phylogeny",
        color="#d62728",
        out_stem=run_dir / "figure_geodesic_vs_phylo",
    )


# ── Dimensionality-reduction figures ───────────────────────────────────────────


def load_taxonomy(meta: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (in `meta` order) with phylum/class/order/family columns."""
    if TAX_CACHE.exists():
        cached = pd.read_csv(TAX_CACHE)
        if list(cached["ncbi_accession"]) == list(meta["ncbi_accession"]):
            print(f"  Using cached taxonomy: {TAX_CACHE}")
            return cached

    # Reading + parsing the ~122 MB gzipped GTDB metadata is the slow part, so only
    # do it on a cache miss.
    print(f"  Deriving taxonomy from {GTDB_META} (one-time, ~20s)...")
    man = pd.read_csv(MANIFEST)
    n2g = dict(zip(man["ncbi_accession"], man["gtdb_accession"], strict=False))
    gtdb_df = pd.read_csv(
        GTDB_META,
        sep="\t",
        comment="#",
        low_memory=False,
        usecols=["accession", "gtdb_taxonomy"],
    )
    tax_map = dict(zip(gtdb_df["accession"], gtdb_df["gtdb_taxonomy"], strict=False))

    def rank(tax_str: str, prefix: str) -> str:
        return (
            next((p[3:] for p in tax_str.split(";") if p.startswith(prefix)), "Unassigned")
            or "Unassigned"
        )

    rows = [
        {rk: rank(tax_map[n2g[ncbi]], pre) for rk, pre in RANK_PREFIX.items()}
        for ncbi in meta["ncbi_accession"]
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "ncbi_accession", meta["ncbi_accession"].values)
    out.to_csv(TAX_CACHE, index=False)
    print(f"  Cached taxonomy to {TAX_CACHE}")
    return out


def color_by_top(labels: np.ndarray, top: int):
    """Map labels -> colors: `top` most-common get distinct colors, rest -> grey 'Other'."""
    counts = pd.Series(labels).value_counts()
    top_labels = list(counts.head(top).index)
    cmap = plt.get_cmap("tab20")
    palette = {lab: cmap(i % 20) for i, lab in enumerate(top_labels)}
    point_colors = [palette.get(lab, (0.82, 0.82, 0.82, 1.0)) for lab in labels]
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            color=palette[lab],
            label=f"{lab} ({counts[lab]})",
        )
        for lab in top_labels
    ]
    n_other = len(labels) - counts.head(top).sum()
    if n_other > 0:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=6,
                color=(0.82, 0.82, 0.82, 1.0),
                label=f"Other ({n_other})",
            )
        )
    return point_colors, handles


def make_umap_figure(embeddings, tax, ranks, top, seed, n_neighbors, min_dist, out_stem):
    import umap

    print(f"  Running UMAP (cosine metric, n_neighbors={n_neighbors}, min_dist={min_dist})...")
    coords = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(embeddings)

    n = len(ranks)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.0))
    if n == 1:
        axes = [axes]
    for ax, rk in zip(axes, ranks, strict=False):
        labels = tax[rk].values
        point_colors, handles = color_by_top(labels, top)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=point_colors,
            s=18,
            alpha=0.85,
            linewidths=0.3,
            edgecolors="white",
        )
        ax.set_title(
            f"UMAP of Evo2 embeddings — by {rk}\n({pd.Series(labels).nunique()} {rk}-level groups)",
            fontsize=10,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=7,
            frameon=False,
            title=f"Top {top} {rk}",
        )
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved {out_stem.with_suffix('.png')} and .pdf")


def make_pca3d_figure(embeddings, tax, ranks, top, seed, out_stem):
    # L2-normalize rows so PCA is consistent with the cosine/angular metric.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norms, 1e-12, None)

    pca = PCA(n_components=3, random_state=seed)
    coords = pca.fit_transform(unit)
    evr = pca.explained_variance_ratio_ * 100
    print(
        f"  PCA variance explained: PC1={evr[0]:.1f}%  PC2={evr[1]:.1f}%  PC3={evr[2]:.1f}%  "
        f"(total {evr.sum():.1f}%)"
    )

    n = len(ranks)
    fig = plt.figure(figsize=(6.0 * n, 5.4))
    for i, rk in enumerate(ranks):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        labels = tax[rk].values
        point_colors, handles = color_by_top(labels, top)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=point_colors,
            s=16,
            alpha=0.85,
            linewidths=0.2,
            edgecolors="white",
        )
        ax.set_title(
            f"3D PCA of Evo2 embeddings — by {rk}\n"
            f"({pd.Series(labels).nunique()} {rk}-level groups)",
            fontsize=10,
        )
        ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)", fontsize=8)
        ax.set_zlabel(f"PC3 ({evr[2]:.1f}%)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=6.5,
            frameon=False,
            title=f"Top {top} {rk}",
        )
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved {out_stem.with_suffix('.png')} and .pdf")


def make_dimreduction_figures(run_dir, method, ranks, top, seed, n_neighbors, min_dist) -> None:
    embeddings = np.load(EMBED_NPY)
    meta = pd.read_csv(EMBED_META)
    print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
    tax = load_taxonomy(meta)

    if method in ("both", "umap"):
        print("\n[dimreduction: UMAP]")
        make_umap_figure(
            embeddings,
            tax,
            ranks,
            top,
            seed,
            n_neighbors,
            min_dist,
            run_dir / "figure_umap_taxonomy",
        )
    if method in ("both", "pca"):
        print("\n[dimreduction: 3D PCA]")
        make_pca3d_figure(embeddings, tax, ranks, top, seed, run_dir / "figure_pca3d_taxonomy")


# ── CLI ─────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Results folder containing the geodesic + patristic matrices "
        "(default: newest results/*_evo2-species)",
    )
    p.add_argument(
        "--figures",
        nargs="+",
        choices=["correlation", "dimreduction"],
        default=["correlation", "dimreduction"],
        help="Which figure sets to generate",
    )
    # dimreduction options
    p.add_argument("--method", choices=["both", "umap", "pca"], default="both")
    p.add_argument(
        "--ranks", nargs="+", default=["class", "order", "family"], choices=list(RANK_PREFIX)
    )
    p.add_argument("--top", type=int, default=12, help="Distinct-colored groups per panel")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def latest_run_dir(suffix: str) -> Path:
    """Newest results/*_<suffix> directory, or exit with a clear message if none."""
    matches = sorted(Path("results").glob(f"*_{suffix}"))
    if not matches:
        sys.exit(
            f"ERROR: no results/*_{suffix} run found. Run the pipeline first, "
            f"or pass --run-dir explicitly."
        )
    return matches[-1]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir("evo2-species")
    print(f"run-dir: {run_dir}")
    set_pub_style(font_size=9, title_size=10)

    if "correlation" in args.figures:
        print("\n=== Distance-correlation figures ===")
        make_correlation_figures(run_dir)
    if "dimreduction" in args.figures:
        print("\n=== Dimensionality-reduction figures ===")
        make_dimreduction_figures(
            run_dir, args.method, args.ranks, args.top, args.seed, args.n_neighbors, args.min_dist
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
