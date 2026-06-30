"""Embed cross-kingdom gene-family CDS with Evo2 and test two structures at once.

Companion to embed_and_geodesic_species.py (which embeds whole-genome windows). Here
each manifest row is ONE coding sequence pulled from KEGG across the tree of life
(scripts/evo2/build_gene_families_evo2.py), and we ask the two questions the design
doc (gene_family_design.md) frames:

  Axis A — between-family.  Do globins, P450s, opsins, … form separable clusters on
    the Evo2 manifold? Family-centroid geodesic matrix + within/between-family
    distance ratio with a label-permutation p-value. (The GPN-Star Figure-2 analog,
    but cross-kingdom.)

  Axis B — within-family phylogeny.  *Inside* one family, does the embedding geodesic
    recapitulate sequence divergence and host taxonomy (bacterial → plant →
    vertebrate globin)? Per-family Spearman ρ of the within-family geodesic submatrix
    vs k-mer sequence divergence and vs taxonomic-rank distance. This is the new
    contribution — no GPN-Star configuration can pose it.

Embedding strategy is identical to the species pipeline: Evo2 7B, residual-stream tap
blocks.24, autoregressive burn-in (pool only the final EMBED_BP positions). A standalone
CDS (median ~0.5–2 kb) is one window, so we pool the whole coding sequence.

Pipeline:
  1. Load manifest + per-family FASTAs.
  2. Embed each CDS (Evo2 blocks.24) -> one 4096-d vector.
  3. k-NN graph (angular distance), lowest connected k, all-pairs geodesic.
  4. Axis A: family-centroid distances + within/between-family ratio + Mantel.
  5. Axis B: per-family geodesic vs k-mer divergence and vs taxonomic distance.
  6. Save labeled CSVs + summaries.

Usage:
    uv run python scripts/evo2/embed_and_geodesic_gene_families.py
    uv run python scripts/evo2/embed_and_geodesic_gene_families.py --force-reembed
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calculate_gene_family_baselines import (  # noqa: E402
    kmer_distance_matrix,
    taxonomic_distance_matrix,
)
from geodesic_utils import (  # noqa: E402
    compute_geodesic,
    find_min_connected_k,
    upper_triangle,
    within_between_analysis,
)

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME = "evo2_7b"
# Residual-stream tap leaving block 24 — same choice validated for the species
# pipeline (lifts geodesic-vs-phylo Pearson ~0.36 → ~0.72 vs the MLP delta).
EMBED_LAYER = "blocks.24"
# Pool the final EMBED_BP positions; leading positions are autoregressive burn-in.
# CDS are short (one window), so [-EMBED_BP:] keeps the whole coding sequence.
EMBED_BP = 2000

DATA_DIR = Path("data/evo2_gene_families")
MANIFEST_PATH = DATA_DIR / "manifest.csv"
EMBED_DIR = DATA_DIR / "embeddings"
EMBED_NPY = EMBED_DIR / "evo2_gene_family_embeddings.npy"
EMBED_META = EMBED_DIR / "metadata.csv"
EMBED_NPY_PARTIAL = EMBED_DIR / "embeddings_partial.npy"
EMBED_META_PARTIAL = EMBED_DIR / "metadata_partial.csv"
CHECKPOINT_EVERY = 100

KMER_K = 6

# ── FASTA loading ────────────────────────────────────────────────────────────


def load_family_sequences(families: list[str]) -> dict[str, str]:
    """Map org_gene -> CDS from each family's FASTA (header: org_gene|family|ko|label)."""
    seqs: dict[str, str] = {}
    for family in sorted(set(families)):
        fasta = DATA_DIR / f"{family}.fasta"
        if not fasta.exists():
            raise FileNotFoundError(f"Family FASTA not found: {fasta}")
        cur_id, cur_seq = None, []
        with open(fasta) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_id:
                        seqs[cur_id] = "".join(cur_seq)
                    cur_id = line[1:].split("|")[0]  # org_gene
                    cur_seq = []
                else:
                    cur_seq.append(line.upper())
        if cur_id:
            seqs[cur_id] = "".join(cur_seq)
    return seqs


# ── Evo2 embedding ─────────────────────────────────────────────────────────────


def embed_cds(seq: str, model, device: str) -> np.ndarray:
    """Embed one CDS: forward pass, mean-pool the final EMBED_BP token positions."""
    input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
    with torch.no_grad():
        _, emb_dict = model(input_ids, return_embeddings=True, layer_names=[EMBED_LAYER])
    win_emb = emb_dict[EMBED_LAYER][0].float()  # (L, 4096)
    return win_emb[-EMBED_BP:].mean(dim=0).cpu().numpy().astype(np.float32)


def compute_embeddings(
    manifest: pd.DataFrame, seqs: dict[str, str], device: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed every manifest CDS in order; checkpoint cumulatively every CHECKPOINT_EVERY."""
    from evo2 import Evo2

    print(f"  Loading {MODEL_NAME} (downloads on first run ~14 GB)...")
    model = Evo2(MODEL_NAME)

    embeddings: list[np.ndarray] = []
    meta_rows: list[dict] = []

    def flush() -> None:
        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = EMBED_NPY_PARTIAL.with_suffix(".tmp.npy")
        np.save(tmp, np.stack(embeddings, axis=0))
        tmp.replace(EMBED_NPY_PARTIAL)
        mtmp = EMBED_META_PARTIAL.with_suffix(".tmp")
        pd.DataFrame(meta_rows).to_csv(mtmp, index=False)
        mtmp.replace(EMBED_META_PARTIAL)
        tqdm.write(f"    [checkpoint] flushed {len(embeddings)} embeddings")

    for i, (_, row) in enumerate(
        tqdm(manifest.iterrows(), total=len(manifest), desc="Embedding CDS")
    ):
        emb = embed_cds(seqs[row["org_gene"]], model, device)
        embeddings.append(emb)
        meta_rows.append(row.to_dict())
        if (i + 1) % CHECKPOINT_EVERY == 0:
            flush()

    return np.stack(embeddings, axis=0), pd.DataFrame(meta_rows)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--kmer-k", type=int, default=KMER_K)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    date_str = datetime.date.today().isoformat()
    OUT_DIR = Path("results") / f"{date_str}_evo2-gene-families"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Manifest + sequences ────────────────────────────────────────────────
    print("\n[1] Loading manifest and sequences")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  {len(manifest)} CDS across {manifest['family'].nunique()} families")
    print(f"  Per family: {manifest['family'].value_counts().to_dict()}")
    seqs = load_family_sequences(manifest["family"].tolist())
    missing = [g for g in manifest["org_gene"] if g not in seqs]
    if missing:
        raise ValueError(f"{len(missing)} manifest org_genes missing from FASTA: {missing[:5]}")

    org_genes = manifest["org_gene"].tolist()
    families = np.array(manifest["family"].tolist())
    family_order = sorted(manifest["family"].unique())
    N = len(manifest)

    # ── 2. Evo2 embedding ──────────────────────────────────────────────────────
    print("\n[2] Evo2 embedding (blocks.24)")
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    if EMBED_NPY.exists() and EMBED_META.exists() and not args.force_reembed:
        print(f"  Loading cached embeddings from {EMBED_NPY}")
        embeddings = np.load(EMBED_NPY)
        cached_meta = pd.read_csv(EMBED_META)
        if cached_meta["org_gene"].tolist() != org_genes:
            raise ValueError(
                f"Cached embeddings in {EMBED_NPY} do not match the current manifest "
                "(different/reordered org_genes). Re-run with --force-reembed."
            )
    else:
        embeddings, meta_df = compute_embeddings(manifest, seqs, device)
        np.save(EMBED_NPY, embeddings)
        meta_df.to_csv(EMBED_META, index=False)
        print(f"  Saved embeddings : {EMBED_NPY}  shape={embeddings.shape}")

    assert embeddings.shape == (N, 4096), f"Expected ({N}, 4096), got {embeddings.shape}"

    # ── 3. k-NN graph + geodesic ───────────────────────────────────────────────
    print("\n[3] k-NN graph (lowest connected k) + all-pairs geodesic")
    k_opt, W = find_min_connected_k(embeddings, k_min=3)
    geodesic = compute_geodesic(W)
    np.save(OUT_DIR / "evo2_gene_family_geodesic.npy", geodesic)
    pd.DataFrame(geodesic, index=org_genes, columns=org_genes).to_csv(
        OUT_DIR / "evo2_gene_family_geodesic_labeled.csv"
    )
    print(f"  k={k_opt}, geodesic range [{geodesic.min():.4f}, {geodesic.max():.4f}]")

    # ── 4. Axis A: between-family separability ─────────────────────────────────
    print("\n[4] Axis A — between-family separability")
    F = len(family_order)
    centroid = np.zeros((F, F), dtype=np.float32)
    for fi, fa in enumerate(family_order):
        ia = np.where(families == fa)[0]
        for fj, fb in enumerate(family_order):
            ib = np.where(families == fb)[0]
            if fi == fj:
                sub = geodesic[np.ix_(ia, ia)]
                tri = sub[np.triu_indices(len(ia), k=1)]
                centroid[fi, fj] = tri.mean() if len(tri) else 0.0
            else:
                centroid[fi, fj] = geodesic[np.ix_(ia, ib)].mean()
    pd.DataFrame(centroid, index=family_order, columns=family_order).to_csv(
        OUT_DIR / "family_centroid_distances.csv"
    )
    print("  Family mean geodesic (within diagonal / between off-diagonal):")
    print(pd.DataFrame(centroid, index=family_order, columns=family_order).round(3).to_string())

    within, between, ratio, wb_p = within_between_analysis(geodesic, families, n_perms=9999)
    print(f"\n  Mean within-family  : {within.mean():.4f}  (n={len(within)} pairs)")
    print(f"  Mean between-family : {between.mean():.4f}  (n={len(between)} pairs)")
    print(f"  Between/within ratio: {ratio:.4f}  (permutation p = {wb_p:.4e})")
    pd.DataFrame(
        [
            {
                "mean_within": float(within.mean()),
                "mean_between": float(between.mean()),
                "ratio": ratio,
                "p_value": wb_p,
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
            }
        ]
    ).to_csv(OUT_DIR / "axisA_within_between_summary.csv", index=False)

    # ── 5. Axis B: within-family phylogeny ─────────────────────────────────────
    print("\n[5] Axis B — within-family geodesic vs sequence divergence & taxonomy")
    seq_list = [seqs[g] for g in org_genes]
    kmer_dist = kmer_distance_matrix(seq_list, k=args.kmer_k)
    tax_dist = taxonomic_distance_matrix(manifest)
    np.save(OUT_DIR / "kmer_distance.npy", kmer_dist)
    np.save(OUT_DIR / "taxonomic_distance.npy", tax_dist)

    rows = []
    print(f"  {'family':<22} {'n':>4} {'pairs':>6} {'ρ_kmer':>8} {'p':>9}   {'ρ_tax':>7} {'p':>9}")
    for fam in family_order:
        idx = np.where(families == fam)[0]
        n = len(idx)
        if n < 4:
            print(f"  {fam:<22} {n:>4}  (too few for within-family correlation)")
            continue
        geo_f = upper_triangle(geodesic[np.ix_(idx, idx)])
        kmer_f = upper_triangle(kmer_dist[np.ix_(idx, idx)])
        tax_f = upper_triangle(tax_dist[np.ix_(idx, idx)])
        rho_k, p_k = spearmanr(geo_f, kmer_f)
        # taxonomic distance is highly tied (4 levels); guard the degenerate case.
        if np.ptp(tax_f) == 0:
            rho_t, p_t = float("nan"), float("nan")
        else:
            rho_t, p_t = spearmanr(geo_f, tax_f)
        rows.append(
            {
                "family": fam,
                "n_members": n,
                "n_pairs": len(geo_f),
                "spearman_geodesic_kmer": rho_k,
                "p_kmer": p_k,
                "spearman_geodesic_taxonomy": rho_t,
                "p_taxonomy": p_t,
            }
        )
        print(
            f"  {fam:<22} {n:>4} {len(geo_f):>6} {rho_k:>+8.3f} {p_k:>9.2e}   "
            f"{rho_t:>+7.3f} {p_t:>9.2e}"
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "axisB_within_family_correlations.csv", index=False)

    # Pooled within-family Mantel (block-diagonal): does divergence track geodesic
    # across all families at once, ignoring between-family pairs?
    within_mask = families[:, None] == families[None, :]
    iu = np.triu_indices(N, k=1)
    wpairs = within_mask[iu]
    geo_all, kmer_all = upper_triangle(geodesic), upper_triangle(kmer_dist)
    rho_pool, p_pool = spearmanr(geo_all[wpairs], kmer_all[wpairs])
    print(
        f"\n  Pooled within-family ρ (geodesic vs k-mer): {rho_pool:+.4f} "
        f"(p = {p_pool:.4e}, n = {int(wpairs.sum())} pairs)"
    )

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print(f"\n[6] Outputs written to {OUT_DIR}/")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
