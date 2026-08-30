"""Permutation inference and confidence intervals for the within-family headline rho."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

MIN_PAIRS = 6  # matches score_within: fewer finite pairs than this and the family is skipped
MIN_MEMBERS = 4  # matches the within-family floor used across the pipeline


# ── the statistic


def _rho_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    """Pearson on pre-computed ranks == Spearman, but without re-ranking `ry` every permutation."""
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def mantel_within(
    geo_sub: np.ndarray, D: np.ndarray, n_perms: int, seed: int, two_sided: bool = True
) -> dict | None:
    """Observed ρ + Mantel p for one family, masking exactly as `score_within` does."""
    n = len(geo_sub)
    iu = np.triu_indices(n, k=1)
    d_u = D[iu]
    ok = ~np.isnan(d_u)
    if ok.sum() < MIN_PAIRS or np.ptp(d_u[ok]) == 0:
        return None
    geo_u = geo_sub[iu]
    obs_rho, p_naive = spearmanr(geo_u[ok], d_u[ok])
    if not np.isfinite(obs_rho):
        return None

    d_rank = rankdata(d_u[ok])  # fixed across permutations
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        pg = geo_sub[np.ix_(perm, perm)][iu][ok]
        r = _rho_from_ranks(rankdata(pg), d_rank)
        if (abs(r) >= abs(obs_rho)) if two_sided else (r >= obs_rho):
            count += 1
    return {
        "rho": float(obs_rho),
        "p_naive": float(p_naive),
        "p_mantel": (count + 1) / (n_perms + 1),
        "n_pairs": int(ok.sum()),
        "p_floor": p_floor(n, n_perms, two_sided),
    }


def p_floor(n: int, n_perms: int, two_sided: bool = True) -> float:
    """The smallest p this family could return, whatever the data — its resolution limit."""
    import math

    group_floor = 1.0 / math.factorial(n) if n <= 12 else 0.0  # n>12: n! dwarfs any n_perms
    return max(1.0 / (n_perms + 1), group_floor)


def bootstrap_ci(
    geo_sub: np.ndarray, D: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI from resampling GENES with replacement."""
    n = len(geo_sub)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        g, d = geo_sub[np.ix_(idx, idx)], D[np.ix_(idx, idx)]
        iu = np.triu_indices(n, k=1)
        same = idx[iu[0]] == idx[iu[1]]  # pairs of a gene with its own duplicate
        gu, du = g[iu], d[iu]
        ok = (~np.isnan(du)) & (~same)
        if ok.sum() < MIN_PAIRS or np.ptp(du[ok]) == 0:
            continue
        r = spearmanr(gu[ok], du[ok]).statistic
        if np.isfinite(r):
            out.append(r)
    if len(out) < max(20, n_boot // 10):
        return np.nan, np.nan
    return (
        float(np.percentile(out, 100 * alpha / 2)),
        float(np.percentile(out, 100 * (1 - alpha / 2))),
    )


# ── panel loading


def load_panel(run_dir: Path):
    """(geodesic DataFrame, gene->family, {baseline: full NxN matrix in geodesic order})."""
    hits = sorted(run_dir.glob("*_geodesic_labeled.csv")) + sorted(
        run_dir.glob("*_geodesic_labeled.csv.gz")
    )
    if not hits:
        sys.exit(f"no *_geodesic_labeled.csv[.gz] in {run_dir}")
    geo = pd.read_csv(hits[0], index_col=0)
    ids = geo.index.tolist()

    meta = pd.read_csv(run_dir / "metadata.csv")
    id_col = next((c for c in ("org_gene", "gene") if c in meta.columns), None)
    if id_col is None:
        sys.exit(f"metadata has neither org_gene nor gene: {list(meta.columns)}")
    fam_of = dict(zip(meta[id_col], meta["family"], strict=False))

    panel_mats = {}
    for name, fname in (("taxonomy", "taxonomic_distance.npy"), ("kmer", "kmer_distance.npy")):
        p = run_dir / fname
        if p.exists():
            M = np.load(p)
            # Only usable if it is the whole panel in the geodesic's own row order.
            if M.shape == (len(ids), len(ids)):
                panel_mats[name] = M
            else:
                print(f"  [skip] {fname}: shape {M.shape} != geodesic {len(ids)} — order unknown")
    return geo, fam_of, panel_mats


def cached_family_mats(cache_dir: Path, fam: str, members: list[str]) -> dict:
    """{baseline: matrix reordered to `members`} from the layer-independent per-family cache."""
    npy, idsj = cache_dir / f"{fam}.npy", cache_dir / f"{fam}.ids.json"
    if not (npy.exists() and idsj.exists()):
        return {}
    cached = json.loads(idsj.read_text())
    if not set(members) <= set(cached):
        return {}
    pos = {g: i for i, g in enumerate(cached)}
    order = [pos[g] for g in members]
    return {"patristic": np.load(npy)[np.ix_(order, order)]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--cache",
        default=None,
        help="per-family patristic cache (default: human panel cache)",
    )
    ap.add_argument(
        "--n-perms",
        type=int,
        default=999,
        help="Mantel permutations; p floor is 1/(n+1), so 999 -> p >= 0.001",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--one-sided",
        action="store_true",
        help="count only perm_rho >= obs_rho (matches geodesic_utils.mantel_test). "
        "Default is two-sided, which a negative within-family ρ needs.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    geo_df, fam_of, panel_mats = load_panel(run_dir)
    ids = geo_df.index.tolist()
    pos = {g: i for i, g in enumerate(ids)}
    geo = geo_df.values
    caches = [Path(args.cache)] if args.cache else [ROOT / "data/cache/human_patristic"]

    families = sorted({fam_of[g] for g in ids if g in fam_of})
    print(
        f"{run_dir}\n{len(ids)} genes, {len(families)} families, "
        f"{args.n_perms} permutations, {args.n_boot} bootstrap replicates, "
        f"{'one' if args.one_sided else 'two'}-sided\n"
    )
    print(
        f"{'family':<28}{'baseline':<11}{'n':>5}{'rho':>8}{'p_naive':>11}{'p_mantel':>10}"
        f"{'95% CI':>18}"
    )

    rows = []
    for fam in families:
        members = [g for g in ids if fam_of.get(g) == fam]
        if len(members) < MIN_MEMBERS:
            continue
        gi = [pos[g] for g in members]
        geo_sub = geo[np.ix_(gi, gi)]
        mats = {}
        for c in caches:
            mats = cached_family_mats(c, fam, members)
            if mats:
                break
        for name, M in panel_mats.items():
            mats[name] = M[np.ix_(gi, gi)]
        for bname in sorted(mats):
            res = mantel_within(
                geo_sub, mats[bname], args.n_perms, args.seed, two_sided=not args.one_sided
            )
            if res is None:
                continue
            lo, hi = bootstrap_ci(geo_sub, mats[bname], args.n_boot, args.seed + 1)
            rows.append(
                {
                    "family": fam,
                    "baseline": bname,
                    "n_members": len(members),
                    **res,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_excludes_zero": bool(np.isfinite(lo) and lo * hi > 0),
                }
            )
            print(
                f"{fam:<28}{bname:<11}{len(members):>5}{res['rho']:>+8.3f}"
                f"{res['p_naive']:>11.2e}{res['p_mantel']:>10.4f}"
                f"   [{lo:+.3f}, {hi:+.3f}]"
            )

    if not rows:
        sys.exit("no family × baseline combination was scoreable")
    t = pd.DataFrame(rows)
    out = Path(args.out) if args.out else run_dir / "within_family_permutation_stats.csv"
    t.to_csv(out, index=False)

    t["testable"] = t.p_floor < 0.05  # can this family reach significance at all?
    print("\n── summary by baseline (families able to reach p < 0.05 at all) ──")
    for bname, g in t.groupby("baseline"):
        gt = g[g.testable]
        sig_n = int((gt.p_naive < 0.05).sum())
        sig_m = int((gt.p_mantel < 0.05).sum())
        print(
            f"{bname:<11} {len(gt):>3}/{len(g)} testable | mean ρ {g.rho.mean():+.3f} | "
            f"significant at 0.05: naive {sig_n}/{len(gt)}, Mantel {sig_m}/{len(gt)} | "
            f"CI excludes 0: {int(g.ci_excludes_zero.sum())}/{len(g)}"
        )

    lost = t[t.testable & (t.p_naive < 0.05) & (t.p_mantel >= 0.05)]
    if len(lost):
        print(
            f"\n{len(lost)} result(s) significant under the naive p but NOT under Mantel — "
            f"these are the ones the non-independence was inflating:"
        )
        for _, r in lost.iterrows():
            print(
                f"  {r.family:<28} {r.baseline:<10} ρ {r.rho:+.3f}   "
                f"naive {r.p_naive:.1e} -> Mantel {r.p_mantel:.3f}"
            )
    else:
        print("\nEvery testable result significant under the naive p survives the Mantel test.")

    under = t[~t.testable]
    if len(under):
        fams = sorted(under.family.unique())
        print(
            f"\n{len(under)} result(s) across {len(fams)} famil(ies) are UNDERPOWERED, not "
            f"disproven: with n <= 12 members the label group is small enough that the Mantel p "
            f"has a floor above 0.05 (see p_floor). Report their ρ and CI, not their p."
        )
        print(
            "  "
            + ", ".join(
                f"{f} (n={int(under[under.family == f].n_members.iloc[0])}, "
                f"floor {under[under.family == f].p_floor.iloc[0]:.3f})"
                for f in fams
            )
        )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
