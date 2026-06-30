"""Transcript-span composition control for the matched human Panel-1.

The within/between homology yardsticks (seq-id, patristic, Pfam-JSD) are computed on the CDS —
that's where homology lives, and introns can't be aligned across paralogs. But the *composition
null* should match what the model actually ingested: the **genomic transcript span** (5'UTR +
exons + introns + 3'UTR), which is what Evo2-human reads as a string and GPN-human tiles with
multiz windows. This script computes k-mer + GC composition on those transcript-span sequences
(the cache the Evo2-human embedder already materialised) and scores each model's geodesic against
it — the rigorous "is the geometry just the composition of the input?" null.

Complements (does not replace) the CDS k-mer rung: CDS k-mer = "coding composition"; transcript
k-mer = "actual-input composition" (captures intronic/UTR composition the model could exploit).

Per run dir it writes:
  within_family_kmer_transcript.csv          family, n, spearman_geodesic_kmer_transcript, p
  between_family_transcript_composition.csv  baseline(kmer_transcript|gc_transcript), rho, p_spearman, p_mantel

Usage:
    uv run python scripts/controls/transcript_composition_control.py --run-dir results/<...>-human-panel-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))                 # scripts/ (geodesic_utils)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))   # kmer_sequence_divergence
from geodesic_utils import mantel_test, upper_triangle  # noqa: E402
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402

GENOMIC_CACHE = Path("data/cache/evo2_human_genomic.json")  # gene -> transcript-span genomic string
MIN_MEMBERS = 4  # within-family rank corr needs >= 4 members (>= 6 pairs)
KMER_K = 6


def gc_fraction(seq: str) -> float:
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    at = s.count("A") + s.count("T")
    return gc / (gc + at) if (gc + at) else float("nan")


def family_aggregate(gene_mat: np.ndarray, families: np.ndarray, order: list[str]) -> np.ndarray:
    """(F, F): off-diag = mean cross-family member-pair distance; diag = mean within-family pair."""
    F = len(order)
    idx = {f: np.where(families == f)[0] for f in order}
    D = np.zeros((F, F))
    for i, a in enumerate(order):
        ia = idx[a]
        for j, b in enumerate(order):
            if i == j:
                sub = gene_mat[np.ix_(ia, ia)]
                tri = sub[np.triu_indices(len(ia), 1)]
                D[i, j] = tri.mean() if len(tri) else 0.0
            else:
                D[i, j] = gene_mat[np.ix_(ia, idx[b])].mean()
    return D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--genomic-cache", default=str(GENOMIC_CACHE))
    ap.add_argument("--n-perms", type=int, default=9999, help="Mantel permutations for the between-family ρ.")
    args = ap.parse_args()
    run = Path(args.run_dir)

    geo_files = list(run.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv in {run}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    genes = list(df_geo.index.astype(str))
    geo = df_geo.values.astype(float)

    meta = pd.read_csv(run / "metadata.csv")
    fam_of = dict(zip(meta["gene"].astype(str), meta["family"]))
    families = np.array([fam_of.get(g, "?") for g in genes])
    order = (run / "family_order.txt").read_text().split() if (run / "family_order.txt").exists() \
        else sorted(set(families))

    seqs = json.loads(Path(args.genomic_cache).read_text())
    missing = [g for g in genes if g not in seqs]
    if missing:
        print(f"  WARNING: {len(missing)} genes lack a cached transcript-span string; skipped: {missing[:5]}")
    keep = [i for i, g in enumerate(genes) if g in seqs]
    genes_k = [genes[i] for i in keep]
    fam_k = families[keep]
    geo_k = geo[np.ix_(keep, keep)]

    print(f"Transcript-span composition on {len(genes_k)} genes (k={KMER_K})...")
    kmer_gene = kmer_distance_matrix([seqs[g] for g in genes_k], k=KMER_K).astype(float)
    gc = np.array([gc_fraction(seqs[g]) for g in genes_k])

    # ── within-family: per-family geodesic vs transcript k-mer ──────────────────
    rows = []
    for fam in order:
        ix = np.where(fam_k == fam)[0]
        if len(ix) < MIN_MEMBERS:
            continue
        gu = upper_triangle(geo_k[np.ix_(ix, ix)])
        ku = upper_triangle(kmer_gene[np.ix_(ix, ix)])
        ok = np.isfinite(gu) & np.isfinite(ku)
        if ok.sum() < 6 or np.ptp(ku[ok]) == 0:
            continue
        rho, p = spearmanr(gu[ok], ku[ok])
        rows.append({"family": fam, "n": len(ix),
                     "spearman_geodesic_kmer_transcript": round(float(rho), 4), "p": round(float(p), 4)})
    within = pd.DataFrame(rows)
    within.to_csv(run / "within_family_kmer_transcript.csv", index=False)
    print(f"  within: {len(within)} families scored -> within_family_kmer_transcript.csv")
    if len(within):
        print(within.to_string(index=False))

    # ── between-family: centroid geodesic vs transcript k-mer / GC ──────────────
    cen_files = list(run.glob("*_centroid_distances.csv"))
    brows = []
    if cen_files:
        cen = pd.read_csv(cen_files[0], index_col=0).reindex(index=order, columns=order).values.astype(float)
        iu = np.triu_indices(len(order), 1)
        kmer_fam = family_aggregate(kmer_gene, fam_k, order)
        gmean = np.array([np.nanmean(gc[fam_k == f]) for f in order])
        gc_fam = np.abs(gmean[:, None] - gmean[None, :])
        for name, M in [("kmer_transcript", kmer_fam), ("gc_transcript", gc_fam)]:
            cu, mu = cen[iu], M[iu]
            ok = np.isfinite(cu) & np.isfinite(mu)
            if ok.sum() < 4 or np.ptp(mu[ok]) == 0:
                continue
            rho, p_sp = spearmanr(cu[ok], mu[ok])
            _, p_m = mantel_test(cen, M, n_perms=args.n_perms)
            brows.append({"baseline": name, "spearman_rho": round(float(rho), 4),
                          "p_spearman": float(f"{p_sp:.2e}"), "p_mantel": round(float(p_m), 4)})
    between = pd.DataFrame(brows)
    between.to_csv(run / "between_family_transcript_composition.csv", index=False)
    print(f"  between -> between_family_transcript_composition.csv")
    if len(between):
        print(between.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
