"""Pfam-HMM Jensen-Shannon-divergence between-family homology baseline: shared kernel + run-dir CLI.

KERNEL — fetch_mean_emission / compute_pfam_jsd
    Jensen-Shannon divergence between families' canonical Pfam HMM match-emission profiles, fetched
    from InterPro by Pfam accession. The profile is a *family-intrinsic* property — independent of
    which sequences (cross-kingdom CDS or human paralogs) populate a family — so families that share
    a Pfam accession get identical JSD across pipelines, which is exactly what makes this a directly
    comparable between-family baseline for both models. compute_pfam_jsd is the single implementation,
    imported by the legacy GPN-Star baseline path (deprecated/gpnstar_calculate_gene_baselines_legacy.py)
    and by the CLI wrapper below — which both pipelines now call directly — so they score against an
    IDENTICAL metric.

CLI — main()
    Writes pfam_jsd_distances.csv for any finished gene-family run dir. The accession map comes from
    the shared scripts/gene_families.py (PFAM_ACCESSIONS, covering both panels); the run dir's own
    family_order selects the relevant subset, so the same call works for a human-panel or a
    cross-kingdom run with no flag. Pfam JSD is
    between-family only (one value per family pair), so it joins the between_comparison figure, not the
    within-family figures. (Supersedes the old separate pfam_jsd.py kernel + add_pfam_jsd.py wrapper.)

Import cost: the wrapper's situational imports (the gene_families accession map and pandas) are LOCAL
to the functions that use them, so importing this module for the kernel (compute_pfam_jsd) stays cheap.

Usage:
    uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir results/<...>            # human or cross-kingdom
    # optional: cross-check shared families against another run's Pfam JSD (e.g. the other model)
    uv run python scripts/baselines/pfam_hmm_jsd.py --run-dir ... \
        --compare-csv results/<other-run>/pfam_jsd_distances.csv
"""

import gzip
import io
import urllib.request

import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# InterPro serves the HMM gzip-compressed without a Content-Encoding header, so it
# must be gunzipped explicitly before pyhmmer parses it.
INTERPRO_HMM = "https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{acc}?annotation=hmm"


# ── kernel (accession-agnostic; imported by both pipelines) ──────────────────────


def fetch_mean_emission(pfam_acc: str) -> np.ndarray:
    """Mean match-emission profile (20-vector) of a family's canonical Pfam HMM."""
    import pyhmmer  # local: pulled in only when actually fetching profiles

    with urllib.request.urlopen(INTERPRO_HMM.format(acc=pfam_acc), timeout=60) as r:
        compressed = r.read()
    raw = gzip.decompress(compressed)
    with pyhmmer.plan7.HMMFile(io.BytesIO(raw)) as f:
        hmm = next(f)
    mat = np.array(hmm.match_emissions)[1:]  # (M, 20), skip the BEGIN row
    return mat.mean(axis=0)


def compute_pfam_jsd(family_order: list[str], accessions: dict[str, str]) -> np.ndarray:
    """F×F squared Jensen-Shannon divergence between families' mean HMM profiles."""
    vectors = {}
    print("Fetching Pfam HMM profiles from InterPro...")
    for fam in tqdm(family_order, desc="Pfam download"):
        vectors[fam] = fetch_mean_emission(accessions[fam])

    F = len(family_order)
    D = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            jsd = jensenshannon(vectors[family_order[i]], vectors[family_order[j]], base=2) ** 2
            D[i, j] = D[j, i] = jsd
    return D


# ── run-dir wrapper (CLI) ────────────────────────────────────────────────────────
# Wrapper-only below. Situational/heavy imports are local to keep the kernel import cheap.

def accession_map() -> dict[str, str]:
    """Family -> Pfam accession, from the shared scripts/gene_families.py (covers both panels;
    the run dir's family_order selects the relevant subset). Imported lazily so kernel consumers
    of compute_pfam_jsd needn't load the gene_families module at all."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
    from gene_families import PFAM_ACCESSIONS  # noqa: E402

    return PFAM_ACCESSIONS


def _upper(mat: np.ndarray) -> np.ndarray:
    return mat[np.triu_indices(len(mat), k=1)]


def _family_order(run) -> list[str]:
    """Family order from family_order.txt, falling back to the centroid-distance CSV index."""
    import sys

    import pandas as pd

    fo = run / "family_order.txt"
    if fo.exists():
        return fo.read_text().split()
    cen = list(run.glob("*_centroid_distances.csv"))
    if cen:
        return pd.read_csv(cen[0], index_col=0).index.tolist()
    sys.exit(f"No family_order.txt or *_centroid_distances.csv in {run}")


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    import pandas as pd
    from scipy.stats import spearmanr

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--compare-csv", default=None,
                    help="Optional Pfam JSD CSV to cross-check shared families against.")
    ap.add_argument("--distances-from", default=None,
                    help="Reuse the Pfam-JSD matrix (pfam_jsd_distances.csv) from this donor "
                         "run dir instead of fetching HMM emissions. The JSD is sequence/"
                         "annotation-derived and layer-independent — for an all-layer sweep.")
    args = ap.parse_args()
    run = Path(args.run_dir)

    fam_order = _family_order(run)
    if args.distances_from:
        donor = Path(args.distances_from) / "pfam_jsd_distances.csv"
        jsd = pd.read_csv(donor, index_col=0).reindex(index=fam_order, columns=fam_order).values
        print(f"Reusing Pfam-JSD matrix from {donor}")
    else:
        accessions = accession_map()
        missing = [f for f in fam_order if f not in accessions]
        if missing:
            sys.exit(f"No Pfam accession defined for families: {missing}")
        jsd = compute_pfam_jsd(fam_order, accessions)
    out = pd.DataFrame(jsd, index=fam_order, columns=fam_order)
    out.to_csv(run / "pfam_jsd_distances.csv")
    print(f"Wrote {run}/pfam_jsd_distances.csv ({len(fam_order)} families)")

    # Cross-check shared families against another run's Pfam JSD (e.g. the other model).
    if args.compare_csv:
        ref = pd.read_csv(args.compare_csv, index_col=0)
        shared = [f for f in fam_order if f in ref.index]
        if len(shared) >= 2:
            diff = np.abs(out.loc[shared, shared].values - ref.loc[shared, shared].values).max()
            status = "✓ identical" if diff < 1e-6 else "⚠ DIFFERS"
            print(f"Cross-check vs {Path(args.compare_csv).parent.name}: "
                  f"{len(shared)} shared families, max|Δ JSD|={diff:.2e}  {status}")

    # If the run has a centroid-geodesic matrix, report the between-family ρ against it.
    cen = list(run.glob("*_centroid_distances.csv"))
    if cen:
        centroid = pd.read_csv(cen[0], index_col=0).reindex(index=fam_order, columns=fam_order)
        rho, p = spearmanr(_upper(centroid.values), _upper(jsd))
        print(f"Between-family Spearman ρ (centroid geodesic vs Pfam JSD): ρ={rho:+.4f}  p={p:.4e}")


if __name__ == "__main__":
    main()
