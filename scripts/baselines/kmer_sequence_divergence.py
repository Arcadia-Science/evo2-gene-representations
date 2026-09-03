"""Shared sequence-based ground-truth baselines for the gene-family pipelines."""

from __future__ import annotations

import numpy as np

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_frequency_vector(seq: str, k: int = 6, *, normalize: bool = True) -> np.ndarray:
    """L1-normalised count vector over the 4**k nucleotide k-mers."""
    if normalize:
        seq = seq.upper().replace("U", "T")
    dim = 4**k
    vec = np.zeros(dim, dtype=np.float64)
    idx = 0
    valid = 0
    mask = dim - 1  # 4**k - 1; rolling base-4 index
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


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """(M, M) cosine distance (1 − cosine similarity) between the rows of ``vectors``."""
    V = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    cos = V @ V.T
    np.clip(cos, -1.0, 1.0, out=cos)
    return 1.0 - cos


def kmer_distance_matrix(seqs: list[str], k: int = 6, *, normalize: bool = True) -> np.ndarray:
    """(N, N) cosine distance in [0, 1] between k-mer frequency vectors."""
    V = np.stack([kmer_frequency_vector(s, k, normalize=normalize) for s in seqs], axis=0)
    dist = cosine_distance_matrix(V)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)
