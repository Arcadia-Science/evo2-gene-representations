"""Build all-layer Wasserstein distances for the publication mammal panel."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))

import mammal_between as mammal  # noqa: E402
from geodesic_utils import mantel_test, upper_triangle  # noqa: E402
from ot_between_family import compute_ot_matrices  # noqa: E402

ARM = "transcript_cdsmask"
MANIFEST = "complete_manifest.csv"
RUN = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{ARM}"
CENTROID_FILE = "evo2_mammal_centroid_distances.csv"


def family_order() -> list[str]:
    """Read the canonical family order established by the centroid analysis."""
    for block_dir in sorted(RUN.glob("blocks*")):
        path = block_dir / CENTROID_FILE
        if path.exists():
            return pd.read_csv(path, index_col=0).index.tolist()
    raise SystemExit(f"no {CENTROID_FILE} under {RUN}/blocks*")


def baseline_matrices(metadata: pd.DataFrame, families: list[str]) -> dict[str, np.ndarray]:
    """Build the three sequence-derived family-distance matrices used by Experiment 1."""
    cds = mammal.load_cds()
    subset = metadata[metadata["family"].isin(families)].reset_index(drop=True)
    return {
        "pfam_jsd": mammal.compute_pfam_jsd(families, mammal.PFAM_ACCESSIONS),
        "kmer": mammal.kmer_between(subset, cds, families),
        "gc_content": mammal.gc_between(subset, cds, families),
    }


def score_wasserstein(
    matrix: np.ndarray,
    baselines: dict[str, np.ndarray],
    n_perms: int,
) -> pd.DataFrame:
    """Score W2 against each active baseline with Spearman rho and a Mantel test."""
    rows = []
    w2 = upper_triangle(matrix)
    for name, baseline in baselines.items():
        reference = upper_triangle(baseline)
        valid = np.isfinite(w2) & np.isfinite(reference)
        if valid.sum() < 3 or np.ptp(w2[valid]) == 0 or np.ptp(reference[valid]) == 0:
            rho = p_mantel = np.nan
        else:
            rho = spearmanr(w2[valid], reference[valid]).statistic
            p_mantel = mantel_test(matrix, baseline, n_perms=n_perms)[1]
        rows.append(
            {
                "approach": "wasserstein",
                "baseline": name,
                "axis": mammal.AXIS[name],
                "spearman_rho": rho,
                "p_mantel": p_mantel,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", nargs="*", type=int, default=list(range(32)))
    parser.add_argument("--n-perms", type=int, default=9999)
    args = parser.parse_args()

    stack, metadata = mammal.load_embedded(ARM, MANIFEST)
    families = family_order()
    family_labels = metadata["family"].to_numpy()
    baselines = baseline_matrices(metadata, families)
    print(f"{stack.shape[1]} loci, {len(families)} families, {len(args.layers)} layers")

    for layer in args.layers:
        block_dir = RUN / f"blocks{layer}"
        block_dir.mkdir(parents=True, exist_ok=True)
        result = compute_ot_matrices(
            stack[layer], family_labels, families, alphas=(), solver="exact"
        )
        result.to_run_dir(block_dir)
        scores = score_wasserstein(result.matrices["wasserstein"], baselines, args.n_perms)
        scores.to_csv(block_dir / "between_family_ot_scores.csv", index=False)
        print(
            f"  blocks{layer}: Wasserstein matrix and {len(scores)} Mantel tests "
            f"({result.meta['runtime_seconds']}s OT)"
        )


if __name__ == "__main__":
    main()
