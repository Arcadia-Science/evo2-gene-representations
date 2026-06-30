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
    uv run python scripts/embed_and_geodesic.py
    uv run python scripts/embed_and_geodesic.py --model vertebrate --batch-size 4
"""

import argparse
import datetime
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import gpn.star.model  # noqa: F401 — registers GPNStar with AutoModel/AutoConfig
from gpn.star.data import GenomeMSA
from gpn.star.model import GPNStarModel
from transformers import AutoConfig

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = {
    "vertebrate": {
        "hf_id": "songlab/gpn-star-hg38-v100-200m",
        "n_species": 100,
        "msa_path": "data/multiz100way.zarr",
    },
}

GENE_FAMILIES: dict[str, list[str]] = {
    "globins": ["HBB", "HBA1", "MB", "NGB"],
    "hox": ["HOXA1", "HOXA2", "HOXB1", "HOXB2", "HOXC4", "HOXD4", "HOXD10"],
    "ras_gtpases": ["KRAS", "HRAS", "NRAS", "RRAS"],
    "cytochrome_p450": ["CYP1A1", "CYP1A2", "CYP2D6", "CYP3A4", "CYP3A5"],
    "c2h2_zinc_fingers": ["SP1", "SP3", "KLF4", "KLF2", "WT1"],
    "aquaporins": ["AQP1", "AQP2", "AQP3", "AQP4", "AQP5"],
    "sirtuins": ["SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7"],
    "toll_like_receptors": ["TLR1", "TLR2", "TLR3", "TLR4", "TLR5", "TLR7", "TLR9"],
    "wnt_ligands": ["WNT1", "WNT2", "WNT3", "WNT4", "WNT5A", "WNT7A", "WNT10B"],
    "kinesins": ["KIF1A", "KIF1B", "KIF2A", "KIF5B", "KIF5C", "KIF11"],
}

FAMILY_ORDER = [
    "globins",
    "hox",
    "ras_gtpases",
    "cytochrome_p450",
    "c2h2_zinc_fingers",
    "aquaporins",
    "sirtuins",
    "toll_like_receptors",
    "wnt_ligands",
    "kinesins",
]

PFAM_ACCESSIONS: dict[str, str] = {
    "globins": "PF00042",
    "hox": "PF00046",
    "ras_gtpases": "PF00071",
    "cytochrome_p450": "PF00067",
    "c2h2_zinc_fingers": "PF00096",
    "aquaporins": "PF00230",
    "sirtuins": "PF02146",
    "toll_like_receptors": "PF01582",
    "wnt_ligands": "PF00110",
    "kinesins": "PF00225",
}

OUT_DIR = Path("data/embeddings")
COORD_CACHE = OUT_DIR / "gene_coords.json"
EMBED_PATH = OUT_DIR / "gpnstar_vertebrate_embeddings.npy"
META_PATH = OUT_DIR / "metadata.csv"
GEODESIC_PATH = OUT_DIR / "gpnstar_vertebrate_geodesic.npy"
CENTROID_PATH = OUT_DIR / "gpnstar_vertebrate_centroid_distances.npy"
FAMILY_ORDER_PATH = OUT_DIR / "family_order.txt"
JSD_PATH = OUT_DIR / "pfam_jsd_distances.npy"

ENSEMBL_BASE = "https://rest.ensembl.org"

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
            wait = 10 * 2 ** attempt
            print(f"  Ensembl request failed ({e}), retrying in {wait}s...")
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


def load_or_fetch_coords(all_genes: list[str]) -> dict[str, dict]:
    if COORD_CACHE.exists():
        with open(COORD_CACHE) as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [g for g in all_genes if g not in cache]
    if missing:
        print(f"Fetching CDS coordinates for {len(missing)} genes from Ensembl...")
        for gene in tqdm(missing, desc="Ensembl lookup"):
            cache[gene] = fetch_cds_coords(gene)
        COORD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(COORD_CACHE, "w") as f:
            json.dump(cache, f, indent=2)

    return {g: cache[g] for g in all_genes}


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
    base_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
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
    cds_end_0 = coord["cds_end"]           # Ensembl end is inclusive; half-open = same value
    strand = coord["strand"]

    center = (cds_start_0 + cds_end_0) // 2
    half = max_window // 2
    win_start = center - half
    win_end = win_start + max_window

    msa = genome_msa.get_msa(chrom, win_start, win_end, strand=strand, tokenize=True)
    msa_t = torch.from_numpy(msa.astype(np.int64)).to(device)  # (L, N)

    input_ids = msa_t[:, :1].unsqueeze(0)       # (1, L, 1)
    source_ids = msa_t.unsqueeze(0)             # (1, L, N)
    target_species = torch.zeros(1, 1, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        )

    h = out.last_hidden_state          # (1, L, 1, hidden_size)
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
        zip(gene_list, family_labels), total=len(gene_list), desc="Embedding genes"
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


# ── Step 4: k-NN graph with angular distances ─────────────────────────────────


def cosine_to_angular(cosine_dist: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(1.0 - cosine_dist, -1.0, 1.0))


def build_knn_graph(embeddings: np.ndarray, k: int) -> np.ndarray:
    N = len(embeddings)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    cosine_dists, indices = nn.kneighbors(embeddings)
    angular_dists = cosine_to_angular(cosine_dists)

    W = np.zeros((N, N))
    for i in range(N):
        for j_pos in range(k):
            j = indices[i, j_pos]
            w = angular_dists[i, j_pos]
            if W[j, i] > 0:
                sym_w = min(w, W[j, i])
                W[i, j] = sym_w
                W[j, i] = sym_w
            else:
                W[i, j] = w
    return W


def find_min_connected_k(embeddings: np.ndarray, k_min: int = 3) -> tuple[int, np.ndarray]:
    N = len(embeddings)
    for k in range(k_min, N):
        W = build_knn_graph(embeddings, k)
        n_components, _ = connected_components(
            csgraph=csr_matrix(W), directed=False, return_labels=True
        )
        print(f"  k={k}: {n_components} component(s)")
        if n_components == 1:
            print(f"  => Fully connected at k={k}")
            return k, W
    raise ValueError(f"Graph not connected even at k={N - 1}")


# ── Step 5: All-pairs geodesic distances ──────────────────────────────────────


def compute_geodesic(W: np.ndarray) -> np.ndarray:
    geo = shortest_path(csr_matrix(W), method="auto", directed=False)
    assert not np.any(np.isinf(geo)), "Geodesic matrix has inf — graph is not fully connected"
    return geo


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


# ── Step 7: Pfam JSD ground truth ─────────────────────────────────────────────


def fetch_mean_emission(pfam_acc: str) -> np.ndarray:
    import pyhmmer

    url = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_acc}?annotation=hmm"
    with urllib.request.urlopen(url, timeout=60) as r:
        compressed = r.read()
    raw = gzip.decompress(compressed)
    with pyhmmer.plan7.HMMFile(io.BytesIO(raw)) as f:
        hmm = next(f)
    mat = np.array(hmm.match_emissions)[1:]  # (M, 20), skip row 0
    return mat.mean(axis=0)


def compute_pfam_jsd(family_order: list[str]) -> np.ndarray:
    vectors = {}
    print("Fetching Pfam HMM profiles from InterPro...")
    for fam in tqdm(family_order, desc="Pfam download"):
        acc = PFAM_ACCESSIONS[fam]
        vectors[fam] = fetch_mean_emission(acc)

    F = len(family_order)
    D = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            jsd = jensenshannon(vectors[family_order[i]], vectors[family_order[j]], base=2) ** 2
            D[i, j] = D[j, i] = jsd
    return D


# ── Step 8: Spearman ρ + Mantel test ──────────────────────────────────────────


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def mantel_test(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = mat_a.shape[0]
    b_flat = upper_triangle(mat_b)
    obs_rho, _ = spearmanr(upper_triangle(mat_a), b_flat)
    count_extreme = 0
    for _ in range(n_perms):
        perm = rng.permutation(N)
        perm_flat = upper_triangle(mat_a[np.ix_(perm, perm)])
        perm_rho, _ = spearmanr(perm_flat, b_flat)
        if perm_rho >= obs_rho:
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_perms + 1)
    return float(obs_rho), float(p_value)


# ── Step 6b: Within vs. between family analysis ───────────────────────────────


def within_between_analysis(
    geodesic: np.ndarray,
    families: np.ndarray,
    n_perms: int = 9999,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compare within-family vs between-family geodesic distances via permutation test."""
    N = len(families)
    pairs_i, pairs_j = np.triu_indices(N, k=1)
    pair_dists = geodesic[pairs_i, pairs_j]
    same = families[pairs_i] == families[pairs_j]

    within = pair_dists[same]
    between = pair_dists[~same]
    obs_ratio = between.mean() / within.mean()

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        pf = rng.permutation(families)
        ps = pf[pairs_i] == pf[pairs_j]
        if ps.any() and (~ps).any():
            if pair_dists[~ps].mean() / pair_dists[ps].mean() >= obs_ratio:
                count += 1
    p_val = (count + 1) / (n_perms + 1)
    return within, between, float(obs_ratio), float(p_val)


# ── Step 7b: CDS sequence identity baseline ───────────────────────────────────


def fetch_cds_sequence(gene_symbol: str) -> str:
    """Fetch canonical CDS nucleotide sequence from Ensembl REST API."""
    # Step 1: look up canonical transcript ID
    lookup_url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}"
        f"?content-type=application/json"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(lookup_url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 10 * 2 ** attempt
            print(f"  Lookup failed for {gene_symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {gene_symbol}")

    transcript_id = data["canonical_transcript"].split(".")[0]

    # Step 2: fetch CDS sequence for that transcript
    seq_url = (
        f"{ENSEMBL_BASE}/sequence/id/{transcript_id}"
        f"?type=cds&content-type=text/plain"
    )
    for attempt in range(5):
        try:
            req = urllib.request.Request(seq_url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8").strip().upper()
        except Exception as e:
            wait = 10 * 2 ** attempt
            print(f"  Seq fetch failed for {gene_symbol}/{transcript_id} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch CDS sequence for {gene_symbol}")


def compute_sequence_identity_matrix(
    gene_list: list[str],
    families: np.ndarray,
    family_order: list[str],
    seq_cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch CDS sequences and compute pairwise sequence identity.

    Returns:
        gene_identity : (N, N) identity matrix in [0, 1], diagonal = 1
        family_identity: (F, F) mean family-level identity matrix
    """
    # Load or build sequence cache
    if seq_cache_path.exists():
        print(f"  Loading cached sequences from {seq_cache_path}")
        with open(seq_cache_path) as f:
            sequences = json.load(f)
    else:
        sequences = {}

    missing = [g for g in gene_list if g not in sequences]
    if missing:
        print(f"  Fetching {len(missing)} CDS sequences from Ensembl...")
        for gene in tqdm(missing, desc="Fetching CDS seqs"):
            sequences[gene] = fetch_cds_sequence(gene)
            time.sleep(0.35)  # polite rate limiting
        seq_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seq_cache_path, "w") as f:
            json.dump(sequences, f, indent=2)

    seqs = [sequences[g] for g in gene_list]
    N = len(seqs)

    # Pairwise identity via difflib SequenceMatcher.ratio()
    # ratio = 2*M/T where M=matches, T=total chars; preserves rank order for Spearman ρ
    print(f"  Computing {N * (N - 1) // 2} pairwise sequence identities...")
    gene_identity = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            ratio = SequenceMatcher(None, seqs[i], seqs[j], autojunk=False).ratio()
            gene_identity[i, j] = gene_identity[j, i] = ratio

    # Aggregate to family level
    F = len(family_order)
    family_identity = np.zeros((F, F))
    for fi, fam_i in enumerate(family_order):
        idx_i = np.where(families == fam_i)[0]
        for fj, fam_j in enumerate(family_order):
            idx_j = np.where(families == fam_j)[0]
            sub = gene_identity[np.ix_(idx_i, idx_j)]
            if fi == fj:
                tri = sub[np.triu_indices(len(idx_i), k=1)]
                family_identity[fi, fj] = tri.mean() if len(tri) > 0 else 1.0
            else:
                family_identity[fi, fj] = sub.mean()

    return gene_identity, family_identity


# ── Step 7c: Ensembl Compara paralog identity ─────────────────────────────────


def _fetch_ensg_id(gene_symbol: str) -> str:
    """Fetch Ensembl gene ID (ENSG...) for a human gene symbol."""
    url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())["id"]
        except Exception as e:
            wait = 10 * 2 ** attempt
            print(f"  ENSG lookup failed for {gene_symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"ENSG lookup failed for {gene_symbol}")


def _fetch_paralogs(gene_symbol: str) -> dict[str, float]:
    """
    Fetch within-species paralog perc_id values from Ensembl Compara.

    Returns dict: target_ensg_id -> mean protein % identity (mean of source and target perc_id).
    NOTE: PANTHER tree endpoint is not publicly accessible in PANTHER v19. Ensembl Compara
    paralog perc_id is derived from the same protein MSA pipeline that underlies PANTHER trees
    and is the closest available programmatic proxy for PANTHER branch length distances.
    """
    url = (
        f"{ENSEMBL_BASE}/homology/symbol/homo_sapiens/{gene_symbol}"
        f"?content-type=application/json&type=paralogues"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 15 * 2 ** attempt
            print(f"  Paralog fetch failed for {gene_symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        return {}

    result: dict[str, float] = {}
    for hom in data.get("data", [{}])[0].get("homologies", []):
        src = hom.get("source", {})
        tgt = hom.get("target", {})
        if tgt.get("species") != "homo_sapiens":
            continue
        tgt_id = tgt.get("id", "")
        src_pct = src.get("perc_id")
        tgt_pct = tgt.get("perc_id")
        vals = [v for v in (src_pct, tgt_pct) if v is not None]
        if vals:
            result[tgt_id] = float(np.mean(vals))
    return result


def compute_ensembl_paralog_matrix(
    gene_list: list[str],
    families: np.ndarray,
    family_order: list[str],
    ensg_cache_path: Path,
    paralog_cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pairwise protein identity matrices from Ensembl Compara paralog data.

    Returns:
        gene_identity   : (N, N) float matrix, diagonal = 100.0, NaN where not in paralog DB
        family_identity : (F, F) mean identity, NaN where no gene pairs have data
    """
    # ── ENSG IDs ──────────────────────────────────────────────────────────────
    if ensg_cache_path.exists():
        with open(ensg_cache_path) as f:
            ensg_cache = json.load(f)
    else:
        ensg_cache = {}

    missing_ensg = [g for g in gene_list if g not in ensg_cache]
    if missing_ensg:
        print(f"  Fetching {len(missing_ensg)} ENSG IDs from Ensembl...")
        for gene in tqdm(missing_ensg, desc="ENSG lookup"):
            ensg_cache[gene] = _fetch_ensg_id(gene)
            time.sleep(0.3)
        ensg_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ensg_cache_path, "w") as f:
            json.dump(ensg_cache, f, indent=2)

    ensg_ids = [ensg_cache[g] for g in gene_list]
    ensg_to_idx = {eid: i for i, eid in enumerate(ensg_ids)}

    # ── Paralog data ──────────────────────────────────────────────────────────
    if paralog_cache_path.exists():
        with open(paralog_cache_path) as f:
            paralog_cache = json.load(f)
    else:
        paralog_cache = {}

    missing_paralog = [g for g in gene_list if g not in paralog_cache]
    if missing_paralog:
        print(f"  Fetching paralog data for {len(missing_paralog)} genes from Ensembl Compara...")
        for gene in tqdm(missing_paralog, desc="Compara paralogs"):
            paralog_cache[gene] = _fetch_paralogs(gene)
            time.sleep(0.5)
        paralog_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(paralog_cache_path, "w") as f:
            json.dump(paralog_cache, f, indent=2)

    # ── Build symmetric gene-level matrix ─────────────────────────────────────
    N = len(gene_list)
    gene_identity = np.full((N, N), np.nan)
    np.fill_diagonal(gene_identity, 100.0)

    for i, gene_a in enumerate(gene_list):
        paralogs_a = paralog_cache.get(gene_a, {})  # {ensg_id: perc_id}
        for ensg_b, pct in paralogs_a.items():
            j = ensg_to_idx.get(ensg_b)
            if j is None:
                continue
            # Fill both directions; if already filled, average
            if np.isnan(gene_identity[i, j]):
                gene_identity[i, j] = pct
                gene_identity[j, i] = pct
            else:
                gene_identity[i, j] = (gene_identity[i, j] + pct) / 2
                gene_identity[j, i] = gene_identity[i, j]

    n_found = int((~np.isnan(gene_identity) & (np.arange(N)[:, None] != np.arange(N))).sum()) // 2
    n_total = N * (N - 1) // 2
    print(f"  Gene-pair coverage: {n_found}/{n_total} pairs ({100*n_found/n_total:.1f}%)")

    # ── Aggregate to family level ──────────────────────────────────────────────
    F = len(family_order)
    family_identity = np.full((F, F), np.nan)
    for fi, fam_i in enumerate(family_order):
        idx_i = np.where(families == fam_i)[0]
        for fj, fam_j in enumerate(family_order):
            idx_j = np.where(families == fam_j)[0]
            sub = gene_identity[np.ix_(idx_i, idx_j)]
            if fi == fj:
                vals = sub[np.triu_indices(len(idx_i), k=1)]
            else:
                vals = sub.flatten()
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                family_identity[fi, fj] = float(valid.mean())

    return gene_identity, family_identity


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vertebrate", choices=list(MODELS), help="Model to use (default: vertebrate)")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding (currently one gene at a time; reserved for future batching)")
    p.add_argument("--models-dir", default="models", help="Directory for HuggingFace model cache (default: models/)")
    p.add_argument("--force-reembed", action="store_true", help="Re-run embedding even if cached embeddings exist")
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
    coords = load_or_fetch_coords(gene_list)

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
    df_geo = pd.DataFrame(geodesic, index=meta_df["gene"].tolist(), columns=meta_df["gene"].tolist())
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

    pd.DataFrame([{
        "mean_within": within.mean(), "mean_between": between.mean(),
        "ratio": ratio, "p_value": wb_p,
        "n_within_pairs": len(within), "n_between_pairs": len(between),
    }]).to_csv(OUT_DIR / "within_between_summary.csv", index=False)

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
        print(f"  Loading cached sequence identity matrices")
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
    PARALOG_FAM_PATH  = OUT_DIR / "ensembl_paralog_identity_family.csv"

    if PARALOG_GENE_PATH.exists() and PARALOG_FAM_PATH.exists():
        print(f"  Loading cached Ensembl paralog matrices")
        paralog_gene_mat = pd.read_csv(PARALOG_GENE_PATH, index_col=0).values
        paralog_fam_mat  = pd.read_csv(PARALOG_FAM_PATH,  index_col=0).values
    else:
        paralog_gene_mat, paralog_fam_mat = compute_ensembl_paralog_matrix(
            gene_list, families_arr, FAMILY_ORDER, ENSG_CACHE, PARALOG_CACHE
        )
        pd.DataFrame(paralog_gene_mat, index=gene_list, columns=gene_list).to_csv(PARALOG_GENE_PATH)
        pd.DataFrame(paralog_fam_mat, index=FAMILY_ORDER, columns=FAMILY_ORDER).to_csv(PARALOG_FAM_PATH)
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
