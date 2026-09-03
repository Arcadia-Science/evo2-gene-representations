"""All-layer uncertainty for angular mammalian-ortholog within-family rho."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]

# Published angular file and column for each baseline. Both name the metric: the column used to
# say `spearman_geodesic_*` in these angular tables, which read as the wrong metric to anyone
# opening the CSV. Older run directories would carry the old header; every one here has been
# migrated.
BASELINE_FILES = {
    "speciestree": ("within_family_speciestree_angular.csv", "spearman_angular_speciestree"),
    "patristic": ("within_family_patristic_angular.csv", "spearman_angular_patristic"),
    "kmer": ("kmer_within_family_correlations_angular.csv", "spearman_angular_kmer"),
    "gc": ("within_family_gc_angular.csv", "spearman_angular_gc"),
}
MIN_GROUPS_CI = 3  # below this a percentile bootstrap over groups is not interpretable
MIN_GROUPS_TEST = 6  # Wilcoxon signed-rank cannot reach p < 0.05 below ~6 pairs


# ── level 2: across groups (the published number)


def family_uncertainty(rhos: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap CI + signed-rank test for one family × baseline × layer."""
    n = len(rhos)
    out = {
        "n_groups": n,
        "rho_mean": float(np.mean(rhos)),
        "rho_sd": float(np.std(rhos, ddof=1)) if n > 1 else np.nan,
        "ci_lo": np.nan,
        "ci_hi": np.nan,
        "p_wilcoxon": np.nan,
    }
    if n >= MIN_GROUPS_CI:
        rng = np.random.default_rng(seed)
        means = np.mean(rng.choice(rhos, size=(n_boot, n), replace=True), axis=1)
        out["ci_lo"] = float(np.percentile(means, 2.5))
        out["ci_hi"] = float(np.percentile(means, 97.5))
    if n >= MIN_GROUPS_TEST and np.ptp(rhos) > 0:
        try:
            out["p_wilcoxon"] = float(wilcoxon(rhos).pvalue)
        except ValueError:
            pass
    out["ci_excludes_zero"] = bool(np.isfinite(out["ci_lo"]) and out["ci_lo"] * out["ci_hi"] > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arm", default="transcript_cdsmask", choices=["transcript_cdsmask", "cds", "transcript"]
    )
    ap.add_argument("--baselines", nargs="*", default=list(BASELINE_FILES))
    ap.add_argument(
        "--layers", nargs="*", type=int, default=None, help="default: every layer present"
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}"
    per_group_p = run_root / f"per_group_scores_{args.arm}_angular.csv"
    if not per_group_p.exists():
        sys.exit(f"missing {per_group_p} — run mammal_score.py for this arm first")
    pg = pd.read_csv(per_group_p)
    layers = args.layers if args.layers is not None else sorted(pg.layer.unique())
    baselines = [b for b in args.baselines if b in set(pg.baseline)]
    print(
        f"{run_root.name}: {pg.group.nunique()} groups, {pg.family.nunique()} families, "
        f"{len(layers)} layers, baselines {baselines}\n"
    )

    rows = []
    for L in layers:
        sub_l = pg[pg.layer == L]
        for base in baselines:
            sub = sub_l[sub_l.baseline == base]
            for fam, g in sub.groupby("family"):
                rows.append(
                    {
                        "arm": args.arm,
                        "layer": L,
                        "baseline": base,
                        "family": fam,
                        **family_uncertainty(g["rho"].to_numpy(), args.n_boot, args.seed + L),
                    }
                )
    t = pd.DataFrame(rows)

    # ── the published value must be reproduced exactly
    checked = mismatch = 0
    for L in layers:
        for base in baselines:
            fname, col = BASELINE_FILES[base]
            p = run_root / f"blocks{L}" / fname
            if not p.exists():
                continue
            pub = pd.read_csv(p).set_index("family")[col]
            mine = t[(t.layer == L) & (t.baseline == base)].set_index("family")["rho_mean"]
            common = pub.index.intersection(mine.index)
            if not len(common):
                continue
            d = (pub.loc[common] - mine.loc[common]).abs().max()
            checked += len(common)
            if d > 1e-9:
                mismatch += 1
                print(f"  [MISMATCH] blocks{L}/{fname}: max |published − recomputed| = {d:.2e}")
    if mismatch:
        sys.exit(
            f"aborting: {mismatch} published table(s) do not match the per-group means — "
            f"this tool would be annotating a number the paper does not report"
        )
    print(
        f"[check] {checked:,} published family values reproduced exactly from the per-group "
        f"scores (max deviation < 1e-9)\n"
    )

    out = run_root / f"within_family_uncertainty_{args.arm}_angular.csv"
    t.to_csv(out, index=False)

    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
