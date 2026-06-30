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


def kmer_frequency_vector(seq: str, k: int = 6, *, normalize: bool = True) -> np.ndarray:
    """L1-normalised count vector over the 4**k nucleotide k-mers.

    Non-ACGT characters (N, ambiguity codes) reset the rolling k-mer, so a masked
    base simply drops the k-mers spanning it rather than corrupting the spectrum.
    With ``normalize`` (the default) the sequence is upper-cased and U→T first, so
    soft-masked (lowercase genomic) and RNA input are counted rather than silently
    dropped as non-ACGT; pass ``normalize=False`` only if the input is guaranteed
    upper-case DNA and the extra pass is unwanted.
    """
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
    """(M, M) cosine distance (1 − cosine similarity) between the rows of ``vectors``.

    Zero-norm rows are handled (clipped denominator) and the similarity is clamped to
    [-1, 1] before subtracting. The diagonal is left as-is (≈0 for unit rows); callers
    that need an exact zero diagonal should fill it themselves.
    """
    V = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    cos = V @ V.T
    np.clip(cos, -1.0, 1.0, out=cos)
    return 1.0 - cos


def kmer_distance_matrix(seqs: list[str], k: int = 6, *, normalize: bool = True) -> np.ndarray:
    """(N, N) cosine distance in [0, 1] between k-mer frequency vectors.

    distance = 1 - cosine_similarity; diagonal forced to 0. Identical spectra give
    0, orthogonal (no shared k-mer) give 1. Rank order — all Spearman/Mantel cares
    about — is preserved; absolute scale is not meaningful. ``normalize`` is forwarded
    to :func:`kmer_frequency_vector`.
    """
    V = np.stack([kmer_frequency_vector(s, k, normalize=normalize) for s in seqs], axis=0)
    dist = cosine_distance_matrix(V)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)
