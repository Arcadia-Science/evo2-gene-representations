"""Shared sequence-based ground-truth baselines for the gene-family pipelines.

Alignment-free k-mer sequence divergence, used by BOTH the Evo2 cross-kingdom
pipeline (scripts/evo2/) and the GPN-Star human-paralog pipeline (scripts/gpnstar/)
so the two models are scored against an identical baseline — the apples-to-apples
nucleotide-composition reference that complements CDS alignment identity (GPN-Star)
and lets the two models be compared directly.

A k-mer spectrum is O(N·L) to build and O(N²·4^k) to compare, vectorised in numpy,
and is the standard alignment-free divergence proxy; it sidesteps the O(N²) global
alignment that cost ~23 min for ~18k pairs in the GPN-Star difflib baseline.
"""

from __future__ import annotations

import numpy as np

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _kmer_vector(seq: str, k: int, dim: int) -> np.ndarray:
    """L1-normalised count vector over the 4^k nucleotide k-mers.

    Non-ACGT characters (N, ambiguity codes) reset the rolling k-mer, so a masked
    base simply drops the k-mers spanning it rather than corrupting the spectrum.
    """
    vec = np.zeros(dim, dtype=np.float64)
    idx = 0
    valid = 0
    mask = dim - 1  # 4^k - 1; rolling base-4 index
    for ch in seq:
        b = _BASE_IDX.get(ch)
        if b is None:
            valid = 0
            idx = 0
            continue
        idx = ((idx << 2) | b) & mask
        valid += 1
        if valid >= k:
            vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def kmer_distance_matrix(seqs: list[str], k: int = 6) -> np.ndarray:
    """(N, N) cosine distance in [0, 1] between k-mer frequency vectors.

    distance = 1 - cosine_similarity; diagonal forced to 0. Identical spectra give
    0, orthogonal (no shared k-mer) give 1. Rank order — all Spearman/Mantel cares
    about — is preserved; absolute scale is not meaningful.
    """
    dim = 4**k
    V = np.stack([_kmer_vector(s, k, dim) for s in seqs], axis=0)  # (N, dim)
    norms = np.linalg.norm(V, axis=1)
    norms[norms == 0] = 1.0
    Vn = V / norms[:, None]
    cos = Vn @ Vn.T
    np.clip(cos, -1.0, 1.0, out=cos)
    dist = 1.0 - cos
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)
