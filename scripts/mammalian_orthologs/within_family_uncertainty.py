"""Confidence intervals and permutation inference for the mammalian-ortholog within-family rho."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))

DATA = ROOT / "data" / "mammalian_orthologs"
# Published file + column for each baseline, so the check below reads the real table.
BASELINE_FILES = {
    "speciestree": ("within_family_speciestree.csv", "spearman_geodesic_speciestree"),
    "patristic": ("within_family_patristic.csv", "spearman_geodesic_patristic"),
    "seqid": ("within_family_seqid.csv", "spearman_geodesic_seqid"),
    "kmer": ("kmer_within_family_correlations.csv", "spearman_geodesic_kmer"),
    "gc": ("within_family_gc.csv", "spearman_geodesic_gc"),
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


# ── level 1: inside a group (Mantel over species labels)


def _rho_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def mantel_group(G: np.ndarray, B: np.ndarray, n_perms: int, seed: int) -> dict | None:
    """Two-sided Mantel p for one ortholog group: permute the species labels of the geodesic."""
    n = len(G)
    iu = np.triu_indices(n, 1)
    g, b = G[iu], B[iu]
    ok = np.isfinite(g) & np.isfinite(b)
    if ok.sum() < 6 or np.ptp(g[ok]) == 0 or np.ptp(b[ok]) == 0:
        return None
    obs = spearmanr(g[ok], b[ok]).statistic
    if not np.isfinite(obs):
        return None
    b_rank = rankdata(b[ok])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perms):
        p = rng.permutation(n)
        pg = G[np.ix_(p, p)][iu][ok]
        if abs(_rho_from_ranks(rankdata(pg), b_rank)) >= abs(obs):
            count += 1
    import math

    return {
        "rho": float(obs),
        "p_mantel": (count + 1) / (n_perms + 1),
        "p_floor": max(1.0 / (n_perms + 1), 1.0 / math.factorial(n) if n <= 12 else 0.0),
        "n_species": n,
        "n_pairs": int(ok.sum()),
    }


def run_mantel_layer(arm: str, layer: int, baselines: list[str], n_perms: int, seed: int):
    """Per-group Mantel at one layer."""
    from geodesic_utils import compute_geodesic, find_min_connected_k
    from mammal_controls_score import load_stack, present_keys

    cheap = [b for b in baselines if b == "speciestree"]
    if not cheap:
        print("[mantel] nothing to do: only `speciestree` is supported without realignment")
        return pd.DataFrame()
    man = pd.read_csv(DATA / "complete_manifest.csv")
    man["key"] = man.group + "__" + man.species
    keys = present_keys(arm, man["key"].tolist())
    kmeta = man[man.key.isin(set(keys))].set_index("key").loc[keys]
    pat = pd.read_csv(DATA / "tree" / "species_patristic.csv", index_col=0)

    print(
        f"[mantel] loading {len(keys):,} loci at blocks.{layer} and rebuilding the geodesic "
        f"(this is the slow part) ...",
        flush=True,
    )
    stack, order = load_stack(arm, keys, [layer])
    _, W = find_min_connected_k(stack[0], k_min=3)
    geo = pd.DataFrame(compute_geodesic(W), index=order, columns=order)

    rows = []
    groups = kmeta.reset_index().groupby(["family", "group"])
    for (fam, grp), sub in groups:
        if len(sub) < 10:  # MIN_SP in mammal_controls_score
            continue
        mem, sp = sub["key"].tolist(), sub["species"].tolist()
        r = mantel_group(geo.loc[mem, mem].values, pat.loc[sp, sp].values, n_perms, seed)
        if r:
            rows.append(
                {"family": fam, "group": grp, "baseline": "speciestree", "layer": layer, **r}
            )
    return pd.DataFrame(rows)


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
    ap.add_argument(
        "--mantel-layer",
        type=int,
        default=None,
        help="also run the per-group Mantel test at this layer (rebuilds the geodesic)",
    )
    ap.add_argument("--n-perms", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-layer", type=int, default=15, help="layer printed to stdout")
    args = ap.parse_args()

    run_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}"
    per_group_p = run_root / f"per_group_scores_{args.arm}.csv"
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

    out = run_root / f"within_family_uncertainty_{args.arm}.csv"
    t.to_csv(out, index=False)

    rl = t[t.layer == args.report_layer]
    if len(rl):
        print(f"── blocks.{args.report_layer} ──")
        for base, g in rl.groupby("baseline"):
            ok = g[g.n_groups >= MIN_GROUPS_CI]
            tst = g[g.n_groups >= MIN_GROUPS_TEST]
            print(
                f"{base:<12} {len(g):>2} families | mean ρ {g.rho_mean.mean():+.3f} | "
                f"CI excludes 0: {int(g.ci_excludes_zero.sum())}/{len(ok)} testable | "
                f"Wilcoxon p<0.05: {int((tst.p_wilcoxon < 0.05).sum())}/{len(tst)}"
            )
        small = rl[rl.n_groups < MIN_GROUPS_CI].family.nunique()
        print(
            f"\n{small} famil(ies) have < {MIN_GROUPS_CI} ortholog groups — no interval is "
            f"reported for them (their ρ is a mean over 1-2 genes)."
        )

    if args.mantel_layer is not None:
        m = run_mantel_layer(args.arm, args.mantel_layer, baselines, args.n_perms, args.seed)
        if len(m):
            mp = run_root / f"within_group_mantel_{args.arm}_blocks{args.mantel_layer}.csv"
            m.to_csv(mp, index=False)
            testable = m[m.p_floor < 0.05]
            print(f"\n── per-group Mantel at blocks.{args.mantel_layer} (speciestree) ──")
            print(
                f"{len(m)} groups | mean ρ {m.rho.mean():+.3f} | "
                f"p_mantel < 0.05: {int((testable.p_mantel < 0.05).sum())}/{len(testable)}"
            )
            print(f"Saved {mp}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
