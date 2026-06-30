"""
Embed human gene CDS sequences with GPN-Star and compute geodesic distances.

Pipeline:
  1. Fetch hg38 CDS coordinates from Ensembl REST API (cached).
  2. Extract MSA windows from multiz100way.zarr via GenomeMSA.
  3. Embed each gene with GPNStarModel (mean-pool last hidden state).
  4. Build k-NN graph with angular-distance edge weights.
  5. Compute all-pairs geodesic distances via Dijkstra.
  6. Compute 10x10 family centroid distance matrix.
  7. Compute Pfam JSD ground-truth matrix.
  8. Spearman ρ + Mantel test against JSD reference.

Usage:
    uv run python scripts/gpnstar/embed_and_geodesic_genes.py
    uv run python scripts/gpnstar/embed_and_geodesic_genes.py --model vertebrate --batch-size 4
"""

import argparse
import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

import gpn.star.model  # noqa: F401 — registers GPNStar with AutoModel/AutoConfig
import numpy as np
import pandas as pd
import torch
from gpn.star.data import GenomeMSA
from gpn.star.model import GPNStarModel
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoConfig

# Shared geodesic helpers live in scripts/geodesic_utils.py (one level up).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calculate_gene_baselines import (  # noqa: E402
    ENSEMBL_BASE,
    compute_ensembl_paralog_matrix,
    compute_pfam_jsd,
    compute_sequence_identity_matrix,
)

# Gene-family definitions and ground-truth baselines live in sibling modules.
from families import FAMILY_ORDER, GENE_FAMILIES  # noqa: E402
from geodesic_utils import (  # noqa: E402
    compute_geodesic,
    find_min_connected_k,
    mantel_test,
    upper_triangle,
    within_between_analysis,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = {
    "vertebrate": {
        "hf_id": "songlab/gpn-star-hg38-v100-200m",
        "n_species": 100,
        "msa_path": "data/multiz100way.zarr",
    },
}

# All output/cache paths are derived per-run inside main(); see that function.

# ── Step 1: CDS coordinates ───────────────────────────────────────────────────


def fetch_cds_coords(gene_symbol: str) -> dict:
    url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}"
        f"?content-type=application/json&expand=1"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 10 * 2**attempt
            print(f"  Ensembl request failed ({e}), retrying in {wait}s...")
            import time

            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {gene_symbol} after 5 attempts")

    chrom = data["seq_region_name"]
    strand = "+" if data["strand"] == 1 else "-"
    canonical_id = data["canonical_transcript"].split(".")[0]

    for tx in data.get("Transcript", []):
        if tx["id"] == canonical_id and "Translation" in tx:
            t = tx["Translation"]
            return {
                "gene": gene_symbol,
                "chrom": chrom,
                "cds_start": t["start"],
                "cds_end": t["end"],
                "strand": strand,
                "cds_len": t["end"] - t["start"],
            }
    raise ValueError(f"No canonical CDS found for {gene_symbol}")


def load_or_fetch_coords(all_genes: list[str], cache_path: Path) -> dict[str, dict]:
    """Return {gene: coord} for genes with a canonical CDS in Ensembl.

    Genes without one (no canonical translation, not found) are cached as a
    ``None`` sentinel and dropped from the result, so a single unmappable HGNC
    member can't abort the run. Transient network errors still raise.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [g for g in all_genes if g not in cache]
    if missing:
        print(f"Fetching CDS coordinates for {len(missing)} genes from Ensembl...")
        for gene in tqdm(missing, desc="Ensembl lookup"):
            try:
                cache[gene] = fetch_cds_coords(gene)
            except ValueError as e:  # definitive: no canonical CDS / not found
                print(f"  Skipping {gene}: {e}")
                cache[gene] = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    dropped = [g for g in all_genes if cache.get(g) is None]
    if dropped:
        print(f"  {len(dropped)} gene(s) without a canonical CDS were dropped: {dropped}")
    return {g: cache[g] for g in all_genes if cache.get(g) is not None}


# ── Step 2 + 3: MSA extraction and embedding ──────────────────────────────────


def load_base_model(model_path: Path, device: str) -> tuple[GPNStarModel, int]:
    from safetensors.torch import load_file

    config = AutoConfig.from_pretrained(str(model_path))
    if not os.path.exists(config.phylo_dist_path):
        fallback = os.path.join(str(model_path), "phylo_dist")
        if os.path.exists(fallback):
            config.phylo_dist_path = fallback
        else:
            raise FileNotFoundError(
                f"phylo_dist not found at '{config.phylo_dist_path}' or '{fallback}'"
            )

    model = GPNStarModel(config)
    state_dict = load_file(os.path.join(str(model_path), "model.safetensors"))
    base_state = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(base_state, strict=False)
    model.eval()
    model.to(device)
    return model, config.max_position_embeddings


def embed_gene(
    genome_msa: GenomeMSA,
    model: GPNStarModel,
    coord: dict,
    max_window: int,
    device: str,
) -> np.ndarray:
    chrom = coord["chrom"]
    cds_start_0 = coord["cds_start"] - 1  # convert 1-based inclusive -> 0-based
    cds_end_0 = coord["cds_end"]  # Ensembl end is inclusive; half-open = same value
    strand = coord["strand"]

    center = (cds_start_0 + cds_end_0) // 2
    half = max_window // 2
    win_start = center - half
    win_end = win_start + max_window

    msa = genome_msa.get_msa(chrom, win_start, win_end, strand=strand, tokenize=True)
    msa_t = torch.from_numpy(msa.astype(np.int64)).to(device)  # (L, N)

    input_ids = msa_t[:, :1].unsqueeze(0)  # (1, L, 1)
    source_ids = msa_t.unsqueeze(0)  # (1, L, N)
    target_species = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        )

    h = out.last_hidden_state  # (1, L, 1, hidden_size)
    emb = h[:, :, 0, :].mean(dim=1).squeeze(0)  # (hidden_size,)
    return emb.cpu().float().numpy()


def compute_embeddings(
    model_name: str,
    model_path: Path,
    coords: dict[str, dict],
    gene_list: list[str],
    family_labels: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    cfg = MODELS[model_name]

    print("Loading MSA (zarr)...")
    genome_msa = GenomeMSA(str(cfg["msa_path"]), n_species=cfg["n_species"])

    print("Loading GPNStarModel (base)...")
    model, max_window = load_base_model(model_path, device)

    embeddings = []
    rows = []

    for gene, family in tqdm(
        zip(gene_list, family_labels, strict=False), total=len(gene_list), desc="Embedding genes"
    ):
        coord = coords[gene]
        emb = embed_gene(genome_msa, model, coord, max_window, device)
        embeddings.append(emb)
        rows.append(
            {
                "gene": gene,
                "family": family,
                "chrom": coord["chrom"],
                "cds_start": coord["cds_start"],
                "cds_end": coord["cds_end"],
                "strand": coord["strand"],
                "cds_len": coord["cds_len"],
            }
        )

    emb_array = np.stack(embeddings, axis=0)  # (N_genes, hidden_size)
    meta_df = pd.DataFrame(rows)
    return emb_array, meta_df


# ── Step 6: Family centroid distances ─────────────────────────────────────────


def compute_family_centroid_distances(
    geodesic: np.ndarray,
    families: np.ndarray,
    family_order: list[str],
) -> np.ndarray:
    F = len(family_order)
    D = np.zeros((F, F))
    for fi, fam_i in enumerate(family_order):
        idx_i = np.where(families == fam_i)[0]
        for fj, fam_j in enumerate(family_order):
            idx_j = np.where(families == fam_j)[0]
            if fi == fj:
                sub = geodesic[np.ix_(idx_i, idx_i)]
                tri = sub[np.triu_indices(len(idx_i), k=1)]
                D[fi, fj] = np.mean(tri) if len(tri) > 0 else 0.0
            else:
                sub = geodesic[np.ix_(idx_i, idx_j)]
                D[fi, fj] = np.mean(sub)
    return D


# ── Model path resolution ──────────────────────────────────────────────────────


def resolve_model_path(model_name: str, models_dir: Path) -> Path:
    hf_id = MODELS[model_name]["hf_id"]
    # Check if already downloaded into models_dir as huggingface cache layout
    owner, repo = hf_id.split("/")
    cache_name = f"models--{owner}--{repo}"
    cache_dir = models_dir / cache_name / "snapshots"
    if cache_dir.exists():
        snapshots = sorted(cache_dir.iterdir())
        if snapshots:
            return snapshots[-1]

    # Fall back to snapshot_download
    from huggingface_hub import snapshot_download

    print(f"Downloading {hf_id}...")
    local = snapshot_download(
        repo_id=hf_id,
        cache_dir=str(models_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    return Path(local)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        default="vertebrate",
        choices=list(MODELS),
        help="Model to use (default: vertebrate)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding (currently one gene at a time; "
        "reserved for future batching)",
    )
    p.add_argument(
        "--models-dir",
        default="models",
        help="Directory for HuggingFace model cache (default: models/)",
    )
    p.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-run embedding even if cached embeddings exist",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # All outputs go under results/YYYY-MM-DD_gpnstar-<model>/
    date_str = datetime.date.today().isoformat()
    OUT_DIR = Path("results") / f"{date_str}_gpnstar-{args.model}"
    COORD_CACHE = Path("data/cache/gene_coords.json")  # shared across runs/models
    EMBED_PATH = OUT_DIR / f"gpnstar_{args.model}_embeddings.npy"
    META_PATH = OUT_DIR / "metadata.csv"
    GEODESIC_PATH = OUT_DIR / f"gpnstar_{args.model}_geodesic.npy"
    CENTROID_PATH = OUT_DIR / f"gpnstar_{args.model}_centroid_distances.csv"
    FAMILY_ORDER_PATH = OUT_DIR / "family_order.txt"
    JSD_PATH = OUT_DIR / "pfam_jsd_distances.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    gene_list: list[str] = []
    family_labels: list[str] = []
    for fam in FAMILY_ORDER:
        for gene in GENE_FAMILIES[fam]:
            gene_list.append(gene)
            family_labels.append(fam)

    families_arr = np.array(family_labels)
    N = len(gene_list)
    print(f"Total genes: {N} across {len(FAMILY_ORDER)} families")

    # ── 1. CDS coordinates ────────────────────────────────────────────────────
    print("\n[1] CDS coordinates")
    coords = load_or_fetch_coords(gene_list, COORD_CACHE)

    # Drop genes Ensembl couldn't map, keeping gene_list/family_labels aligned.
    kept = [(g, f) for g, f in zip(gene_list, family_labels, strict=False) if g in coords]
    gene_list = [g for g, _ in kept]
    family_labels = [f for _, f in kept]
    families_arr = np.array(family_labels)
    N = len(gene_list)
    print(f"  Genes with CDS: {N} across {len(set(family_labels))} families")

    # ── 2+3. Embeddings ───────────────────────────────────────────────────────
    print("\n[2+3] MSA extraction and embedding")
    if EMBED_PATH.exists() and META_PATH.exists() and not args.force_reembed:
        print(f"  Loading cached embeddings from {EMBED_PATH}")
        embeddings = np.load(EMBED_PATH)
        meta_df = pd.read_csv(META_PATH)
    else:
        models_dir = Path(args.models_dir)
        model_path = resolve_model_path(args.model, models_dir)
        print(f"  Model path: {model_path}")

        embeddings, meta_df = compute_embeddings(
            model_name=args.model,
            model_path=model_path,
            coords=coords,
            gene_list=gene_list,
            family_labels=family_labels,
            batch_size=args.batch_size,
            device=device,
        )
        np.save(EMBED_PATH, embeddings)
        meta_df.to_csv(META_PATH, index=False)
        print(f"  Saved embeddings: {EMBED_PATH}  shape={embeddings.shape}")
        print(f"  Saved metadata:   {META_PATH}")

    print(f"  Embeddings shape: {embeddings.shape}")

    # ── 4. k-NN graph ─────────────────────────────────────────────────────────
    print("\n[4] Building k-NN graph with angular distances")
    k_opt, W = find_min_connected_k(embeddings, k_min=3)
    print(f"  Optimal k: {k_opt}")

    # ── 5. Geodesic distances ─────────────────────────────────────────────────
    print("\n[5] Computing all-pairs geodesic distances")
    geodesic = compute_geodesic(W)
    np.save(GEODESIC_PATH, geodesic)
    # Save labeled CSV so distances are human-readable
    df_geo = pd.DataFrame(
        geodesic, index=meta_df["gene"].tolist(), columns=meta_df["gene"].tolist()
    )
    df_geo.to_csv(OUT_DIR / f"gpnstar_{args.model}_geodesic_labeled.csv")
    print(f"  Saved geodesic matrix: {GEODESIC_PATH}  shape={geodesic.shape}")
    print(f"  Geodesic range: [{geodesic.min():.4f}, {geodesic.max():.4f}]")

    # ── 6. Family centroid distances ──────────────────────────────────────────
    print("\n[6] Computing family centroid distance matrix")
    centroid_dist = compute_family_centroid_distances(geodesic, families_arr, FAMILY_ORDER)
    df_centroid = pd.DataFrame(centroid_dist, index=FAMILY_ORDER, columns=FAMILY_ORDER)
    df_centroid.to_csv(CENTROID_PATH)
    FAMILY_ORDER_PATH.write_text("\n".join(FAMILY_ORDER) + "\n")
    print(f"  Saved centroid distances: {CENTROID_PATH}")
    print(f"  Saved family order: {FAMILY_ORDER_PATH}")
    print("\n  Family centroid geodesic distances:")
    print(df_centroid.round(4).to_string())

    # ── 6b. Within vs. between family analysis ────────────────────────────────
    print("\n[6b] Within vs. between family distances")
    within, between, ratio, wb_p = within_between_analysis(geodesic, families_arr, n_perms=9999)
    print(f"  Mean within-family  : {within.mean():.4f}  (n={len(within)} pairs)")
    print(f"  Mean between-family : {between.mean():.4f}  (n={len(between)} pairs)")
    print(f"  Between/within ratio: {ratio:.4f}")
    print(f"  Permutation p-value : {wb_p:.4e}")

    pd.DataFrame(
        [
            {
                "mean_within": within.mean(),
                "mean_between": between.mean(),
                "ratio": ratio,
                "p_value": wb_p,
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
            }
        ]
    ).to_csv(OUT_DIR / "within_between_summary.csv", index=False)

    print("\n  Per-family pairwise geodesic distances:")
    for fam in FAMILY_ORDER:
        idx = np.where(families_arr == fam)[0]
        genes_in_fam = [gene_list[i] for i in idx]
        sub = geodesic[np.ix_(idx, idx)]
        df_fam = pd.DataFrame(sub, index=genes_in_fam, columns=genes_in_fam)
        print(f"\n    {fam}:")
        print(df_fam.round(4).to_string())

    # ── 7. Pfam JSD ground truth ──────────────────────────────────────────────
    print("\n[7] Computing Pfam JSD ground-truth matrix")
    if JSD_PATH.exists():
        print(f"  Loading cached JSD matrix from {JSD_PATH}")
        jsd_dist = pd.read_csv(JSD_PATH, index_col=0).values
    else:
        jsd_dist = compute_pfam_jsd(FAMILY_ORDER)
        df_jsd = pd.DataFrame(jsd_dist, index=FAMILY_ORDER, columns=FAMILY_ORDER)
        df_jsd.to_csv(JSD_PATH)
        print(f"  Saved JSD matrix: {JSD_PATH}")

    df_jsd = pd.DataFrame(jsd_dist, index=FAMILY_ORDER, columns=FAMILY_ORDER)
    print("  Pfam JSD distances:")
    print(df_jsd.round(4).to_string())

    # ── 7b. CDS sequence identity baseline ───────────────────────────────────
    print("\n[7b] Computing CDS sequence identity baseline")
    SEQ_CACHE = Path("data/cache/cds_sequences.json")
    SEQID_GENE_PATH = OUT_DIR / "sequence_identity_genes.csv"
    SEQID_FAM_PATH = OUT_DIR / "sequence_identity_family.csv"

    if SEQID_GENE_PATH.exists() and SEQID_FAM_PATH.exists():
        print("  Loading cached sequence identity matrices")
        gene_identity = pd.read_csv(SEQID_GENE_PATH, index_col=0).values
        family_identity = pd.read_csv(SEQID_FAM_PATH, index_col=0).values
    else:
        gene_identity, family_identity = compute_sequence_identity_matrix(
            gene_list, families_arr, FAMILY_ORDER, SEQ_CACHE
        )
        df_seqid_gene = pd.DataFrame(gene_identity, index=gene_list, columns=gene_list)
        df_seqid_gene.to_csv(SEQID_GENE_PATH)
        df_seqid_fam = pd.DataFrame(family_identity, index=FAMILY_ORDER, columns=FAMILY_ORDER)
        df_seqid_fam.to_csv(SEQID_FAM_PATH)
        print(f"  Saved gene-level identity: {SEQID_GENE_PATH}")
        print(f"  Saved family-level identity: {SEQID_FAM_PATH}")

    df_seqid_fam = pd.DataFrame(family_identity, index=FAMILY_ORDER, columns=FAMILY_ORDER)
    print("  Family mean CDS identity:")
    print(df_seqid_fam.round(4).to_string())

    # ── 7c. Ensembl Compara paralog identity baseline ────────────────────────
    # Note: PANTHER tree API is not publicly accessible (v19 removed that endpoint).
    # Ensembl Compara paralog perc_id is the closest available analog — it is
    # computed from the same protein MSA pipeline that underlies PANTHER trees.
    print("\n[7c] Computing Ensembl Compara paralog identity baseline")
    ENSG_CACHE = Path("data/cache/ensg_ids.json")
    PARALOG_CACHE = Path("data/cache/ensembl_paralogs.json")
    PARALOG_GENE_PATH = OUT_DIR / "ensembl_paralog_identity_genes.csv"
    PARALOG_FAM_PATH = OUT_DIR / "ensembl_paralog_identity_family.csv"

    if PARALOG_GENE_PATH.exists() and PARALOG_FAM_PATH.exists():
        print("  Loading cached Ensembl paralog matrices")
        paralog_gene_mat = pd.read_csv(PARALOG_GENE_PATH, index_col=0).values
        paralog_fam_mat = pd.read_csv(PARALOG_FAM_PATH, index_col=0).values
    else:
        paralog_gene_mat, paralog_fam_mat = compute_ensembl_paralog_matrix(
            gene_list, families_arr, FAMILY_ORDER, ENSG_CACHE, PARALOG_CACHE
        )
        pd.DataFrame(paralog_gene_mat, index=gene_list, columns=gene_list).to_csv(PARALOG_GENE_PATH)
        pd.DataFrame(paralog_fam_mat, index=FAMILY_ORDER, columns=FAMILY_ORDER).to_csv(
            PARALOG_FAM_PATH
        )
        print(f"  Saved gene-level paralog matrix:   {PARALOG_GENE_PATH}")
        print(f"  Saved family-level paralog matrix: {PARALOG_FAM_PATH}")

    df_paralog_fam = pd.DataFrame(paralog_fam_mat, index=FAMILY_ORDER, columns=FAMILY_ORDER)
    print("  Ensembl Compara mean protein identity (within/between families):")
    print(df_paralog_fam.round(3).to_string())

    # ── 8. Spearman ρ + Mantel test ───────────────────────────────────────────
    print("\n[8] Spearman ρ and Mantel test")
    geo_flat = upper_triangle(centroid_dist)
    jsd_flat = upper_triangle(jsd_dist)

    rho_sp, pval_sp = spearmanr(geo_flat, jsd_flat)
    print(f"  Spearman ρ (geodesic vs JSD)      = {rho_sp:.4f}  (p = {pval_sp:.4e})")

    seqid_dist_flat = upper_triangle(1.0 - family_identity)
    rho_seqid, pval_seqid = spearmanr(geo_flat, seqid_dist_flat)
    print(f"  Spearman ρ (geodesic vs seq id)   = {rho_seqid:.4f}  (p = {pval_seqid:.4e})")

    print("  Running Mantel test vs JSD (9999 permutations)...")
    rho_mantel, p_mantel = mantel_test(centroid_dist, jsd_dist, n_perms=9999)
    print(f"  Mantel ρ (vs JSD)  = {rho_mantel:.4f}  (p = {p_mantel:.4e})")

    print("\nDone.")


if __name__ == "__main__":
    main()
