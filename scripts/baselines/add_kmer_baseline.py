"""Add a k-mer sequence-divergence baseline to an existing GPN-Star gene-family run.

The GPN-Star pipeline already scores its geodesic against CDS alignment identity,
Pfam JSD, and Compara paralog identity. This adds the SAME alignment-free k-mer
divergence baseline the Evo2 pipeline uses (shared scripts/baselines/kmer_sequence_divergence.py),
so the two models can be compared apples-to-apples on one common reference.

Operates on a finished run dir — no re-embedding. It reuses:
  - <model>_geodesic_labeled.csv   (geodesic, gene-ordered)
  - metadata.csv                   (family per gene)
  - family_order.txt
  - <model>_centroid_distances.csv (family-level geodesic, for the between-family ρ)
  - data/cache/cds_sequences.json  (the CDS already fetched during the run)

Writes kmer_distance_genes.csv + kmer_distance_family.csv and prints/saves the
within- and between-family Spearman ρ (alongside the existing Pfam JSD between-ρ
for direct comparison).

Usage:
    uv run python scripts/baselines/add_kmer_baseline.py \
        --run-dir results/YYYY-MM-DD_gpnstar-vertebrate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))  # kmer_sequence_divergence
from geodesic_utils import upper_triangle  # noqa: E402
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402

CDS_CACHE = Path("data/cache/cds_sequences.json")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kmer-k", type=int, default=6)
    ap.add_argument("--distances-from", default=None,
                    help="Reuse the gene- and family-level k-mer matrices "
                         "(kmer_distance_genes.csv, kmer_distance_family.csv) from this donor "
                         "run dir instead of recomputing the O(N²) k-mer distances. They are "
                         "sequence-derived and layer-independent — for an all-layer sweep.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geodesic = df_geo.values
    genes = df_geo.index.tolist()

    meta = pd.read_csv(run_dir / "metadata.csv").set_index("gene").reindex(genes).reset_index()
    families = meta["family"].to_numpy()
    family_order = (run_dir / "family_order.txt").read_text().strip().splitlines()

    # ── k-mer distance, gene + family level ─────────────────────────────────────
    if args.distances_from:
        donor = Path(args.distances_from)
        kmer = pd.read_csv(donor / "kmer_distance_genes.csv", index_col=0) \
            .reindex(index=genes, columns=genes).values
        kmer_fam = pd.read_csv(donor / "kmer_distance_family.csv", index_col=0) \
            .reindex(index=family_order, columns=family_order).values
        print(f"Reusing k-mer distance matrices from {donor}")
    else:
        if not CDS_CACHE.exists():
            sys.exit(f"CDS cache not found: {CDS_CACHE} (run the GPN-Star pipeline first)")
        cds = json.loads(CDS_CACHE.read_text())
        missing = [g for g in genes if g not in cds or not cds[g]]
        if missing:
            sys.exit(f"{len(missing)} genes missing from CDS cache: {missing[:5]}")
        seqs = [cds[g].upper() for g in genes]
        print(f"Computing k-mer (k={args.kmer_k}) divergence for {len(genes)} genes...")
        kmer = kmer_distance_matrix(seqs, k=args.kmer_k)
        kmer_fam = aggregate_family_matrix(kmer, families, family_order)
    pd.DataFrame(kmer, index=genes, columns=genes).to_csv(run_dir / "kmer_distance_genes.csv")
    pd.DataFrame(kmer_fam, index=family_order, columns=family_order).to_csv(
        run_dir / "kmer_distance_family.csv"
    )

    # ── Within-family: per-family Spearman ρ (geodesic vs k-mer) ────────────────
    print("\nWithin-family Spearman ρ (GPN-Star geodesic vs k-mer divergence):")
    rows = []
    for fam in family_order:
        idx = np.where(families == fam)[0]
        n = len(idx)
        if n < 3:
            print(f"  {fam:<22}: n={n} (too few)")
            continue
        geo_f = upper_triangle(geodesic[np.ix_(idx, idx)])
        kmer_f = upper_triangle(kmer[np.ix_(idx, idx)])
        rho, p = spearmanr(geo_f, kmer_f)
        rows.append(
            {
                "family": fam,
                "n_members": n,
                "n_pairs": len(geo_f),
                "spearman_geodesic_kmer": rho,
                "p_kmer": p,
            }
        )
        print(f"  {fam:<22}: ρ={rho:+.3f}  p={p:.2e}  ({n} genes)")
    pd.DataFrame(rows).to_csv(run_dir / "kmer_within_family_correlations.csv", index=False)

    # ── Between-family: centroid geodesic vs family-mean k-mer (vs Pfam for ref) ─
    cen_files = list(run_dir.glob("*_centroid_distances.csv"))
    centroid = pd.read_csv(cen_files[0], index_col=0).values
    geo_flat = upper_triangle(centroid)
    rho_kmer, p_kmer = spearmanr(geo_flat, upper_triangle(kmer_fam))
    print("\nBetween-family Spearman ρ (centroid geodesic vs baseline):")
    print(f"  k-mer divergence : ρ={rho_kmer:+.4f}  p={p_kmer:.4e}")
    jsd_path = run_dir / "pfam_jsd_distances.csv"
    rho_jsd = None
    if jsd_path.exists():
        jsd = pd.read_csv(jsd_path, index_col=0).values
        rho_jsd, p_jsd = spearmanr(geo_flat, upper_triangle(jsd))
        print(f"  Pfam JSD (for ref): ρ={rho_jsd:+.4f}  p={p_jsd:.4e}")

    pd.DataFrame(
        [
            {
                "between_rho_kmer": rho_kmer,
                "between_p_kmer": p_kmer,
                "between_rho_pfam_jsd": rho_jsd,
            }
        ]
    ).to_csv(run_dir / "kmer_between_family_summary.csv", index=False)

    print(f"\nSaved k-mer baseline CSVs to {run_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
