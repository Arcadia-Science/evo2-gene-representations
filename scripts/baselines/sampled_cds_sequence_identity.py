"""Family-level CDS sequence-identity baseline for the human between_comparison figure.

Full all-pairs difflib identity (deprecated/gpnstar_calculate_gene_baselines_legacy.py) is impractical for 580
genes — the 409-member olfactory family alone is ~83k within-pairs of pure-Python O(n·m) ratio
(>1 h, unfinished). The between_comparison figure only needs the F×F *family* matrix (mean pairwise
identity per family-pair), so we ESTIMATE each cell from a random SAMPLE of up to `--pairs` gene
pairs — statistically fine for a family mean, ~100× faster. Writes sequence_identity_family.csv
(raw identity, diagonal = within-family mean), reading CDS from cds_sequences.json. When several run
dirs are given (the matched Evo2-human + GPN-human pair) they must share gene order; the matrix is
computed once and written to all.

Usage:
    uv run python scripts/baselines/sampled_cds_sequence_identity.py --run-dirs results/<a> results/<b> [--pairs 80]
"""

import argparse
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_CACHE = Path("data/cache/cds_sequences.json")


def sample_pairs(A, B, within, k, rng):
    if within:
        if len(A) < 2:
            return []
        seen = set()
        target = min(k, len(A) * (len(A) - 1) // 2)
        while len(seen) < target:
            seen.add(frozenset(rng.sample(A, 2)))
        return [tuple(p) for p in seen]
    if not A or not B:
        return []
    seen = set()
    target = min(k, len(A) * len(B))
    while len(seen) < target:
        seen.add((rng.choice(A), rng.choice(B)))
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--pairs", type=int, default=80, help="Max gene pairs sampled per family-cell.")
    ap.add_argument("--distances-from", default=None,
                    help="Reuse sequence_identity_family.csv from this donor run dir and write "
                         "it to the target run dir(s). This baseline has no embedding "
                         "dependence at all — for an all-layer sweep it is computed once.")
    args = ap.parse_args()
    runs = [Path(r) for r in args.run_dirs]

    if args.distances_from:
        df = pd.read_csv(Path(args.distances_from) / "sequence_identity_family.csv", index_col=0)
        for r in runs:
            df.to_csv(r / "sequence_identity_family.csv")
            print(f"  reused sequence_identity_family.csv -> {r}")
        return

    metas = [pd.read_csv(r / "metadata.csv") for r in runs]
    genes0 = metas[0]["gene"].tolist()
    for r, m in zip(runs[1:], metas[1:]):
        if m["gene"].tolist() != genes0:
            sys.exit(f"Gene order differs in {r}.")
    fam_order = (runs[0] / "family_order.txt").read_text().split()
    seqs = json.loads(SEQ_CACHE.read_text())
    fam_of = dict(zip(genes0, metas[0]["family"]))
    by_fam = {f: [g for g in genes0 if fam_of.get(g) == f and g in seqs] for f in fam_order}

    rng = random.Random(0)
    F = len(fam_order)
    M = np.eye(F)
    for i, fi in enumerate(fam_order):
        for j in range(i, F):
            pairs = sample_pairs(by_fam[fi], by_fam[fam_order[j]], i == j, args.pairs, rng)
            if not pairs:
                val = 1.0 if i == j else 0.0
            else:
                val = float(np.mean([
                    SequenceMatcher(None, seqs[x], seqs[y], autojunk=False).ratio() for x, y in pairs
                ]))
            M[i, j] = M[j, i] = val
        print(f"  {fi}: done")

    df = pd.DataFrame(M, index=fam_order, columns=fam_order)
    for r in runs:
        df.to_csv(r / "sequence_identity_family.csv")
        print(f"  wrote sequence_identity_family.csv -> {r}  (sampled ≤{args.pairs} pairs/cell)")


if __name__ == "__main__":
    main()
