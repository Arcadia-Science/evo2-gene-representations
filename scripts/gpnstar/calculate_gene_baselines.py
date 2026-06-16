"""Ground-truth baseline matrices for the GPN-Star gene-family analysis.

The geodesic distances from embed_and_geodesic_genes.py are compared against three
independent evolutionary references, computed here and imported by that script:

  - Pfam HMM JSD            : Jensen-Shannon divergence between family HMM emission
                              profiles (family-level, 10×10).
  - CDS sequence identity   : pairwise difflib identity of canonical CDS sequences
                              (gene-level + family-level).
  - Ensembl Compara paralog : within-species paralog protein % identity (gene-level +
                              family-level); PANTHER's tree API is not public in v19,
                              and Compara perc_id from the same MSA pipeline is the
                              closest available proxy.

ENSEMBL_BASE is defined here and re-used by the embedding script's coordinate lookup.
"""

import gzip
import io
import json
import time
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from families import PFAM_ACCESSIONS  # sibling module: scripts/gpnstar/families.py
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

ENSEMBL_BASE = "https://rest.ensembl.org"


# ── Pfam HMM JSD ────────────────────────────────────────────────────────────────


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


# ── CDS sequence identity ───────────────────────────────────────────────────────


def fetch_cds_sequence(gene_symbol: str) -> str:
    """Fetch canonical CDS nucleotide sequence from Ensembl REST API."""
    # Step 1: look up canonical transcript ID
    lookup_url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(lookup_url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 10 * 2**attempt
            print(f"  Lookup failed for {gene_symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {gene_symbol}")

    transcript_id = data["canonical_transcript"].split(".")[0]

    # Step 2: fetch CDS sequence for that transcript
    seq_url = f"{ENSEMBL_BASE}/sequence/id/{transcript_id}?type=cds&content-type=text/plain"
    for attempt in range(5):
        try:
            req = urllib.request.Request(seq_url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8").strip().upper()
        except Exception as e:
            wait = 10 * 2**attempt
            print(
                f"  Seq fetch failed for {gene_symbol}/{transcript_id} ({e}), "
                f"retrying in {wait}s..."
            )
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


# ── Ensembl Compara paralog identity ────────────────────────────────────────────


def _fetch_ensg_id(gene_symbol: str) -> str:
    """Fetch Ensembl gene ID (ENSG...) for a human gene symbol."""
    url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())["id"]
        except Exception as e:
            wait = 10 * 2**attempt
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
            wait = 15 * 2**attempt
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
    print(f"  Gene-pair coverage: {n_found}/{n_total} pairs ({100 * n_found / n_total:.1f}%)")

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
