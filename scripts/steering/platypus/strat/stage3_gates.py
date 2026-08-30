"""Gate 4 — the two free pre-checks that license the expensive Stage-4 tiers. CPU, seconds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

K_GRID = [3, 4, 5, 6, 8, 12, 16]
N_DRAW = 2000
SEED = 20260807


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def panel_pair_cos(D: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """cos between two k-gene panel means, each L2-normalised (norm-matching is implicit in cosine)."""
    return float(unit(D[idx_a].mean(0)) @ unit(D[idx_b].mean(0)))


def tier_c_gate(D: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """For each k: cos(same-cluster panel, other-cluster panel) vs cos(random panel, random panel)."""
    n = len(D)
    rows = []
    for k in K_GRID:
        same, rand = [], []
        for _ in range(N_DRAW):
            i = int(rng.integers(n))
            own = np.where((clusters == clusters[i]) & (np.arange(n) != i))[0]
            oth = np.where(clusters != clusters[i])[0]
            if len(own) < k or len(oth) < k:
                continue
            same.append(panel_pair_cos(D, rng.choice(own, k, replace=False),
                                       rng.choice(oth, k, replace=False)))
            pool = np.delete(np.arange(n), i)
            pick = rng.choice(pool, 2 * k, replace=False)
            rand.append(panel_pair_cos(D, pick[:k], pick[k:]))
        if len(same) < 50:
            rows.append({"k": k, "n_draws": len(same), "cos_same_vs_other": np.nan,
                         "cos_random_vs_random": np.nan, "contrast": np.nan, "usable": False})
            continue
        s, r = float(np.mean(same)), float(np.mean(rand))
        rows.append({"k": k, "n_draws": len(same),
                     "cos_same_vs_other": round(s, 4), "cos_random_vs_random": round(r, 4),
                     # POSITIVE contrast = cluster-matched panels are MORE different from
                     # cross-cluster panels than two arbitrary panels are. That is the thing Tier C
                     # needs; zero or negative means the split carries no directional information.
                     "contrast": round(r - s, 4),
                     "usable": True})
    return pd.DataFrame(rows)


def tier_d_gate(per_gene: pd.DataFrame, frozen_layer: int, band: tuple[int, int]) -> dict:
    """Spread of the per-gene argmax layer of loo_cos, searched INSIDE the candidate band only."""
    lo, hi = band
    sub = per_gene[(per_gene.layer >= lo) & (per_gene.layer <= hi)]
    arg = sub.loc[sub.groupby("gene").loo_cos.idxmax(), ["gene", "layer"]]
    counts = arg.layer.value_counts().sort_index()
    at_frozen = float((arg.layer == frozen_layer).mean())

    full = per_gene[per_gene.layer <= hi]
    farg = full.loc[full.groupby("gene").loo_cos.idxmax(), ["gene", "layer"]]

    return {"band": [lo, hi],
            "n_genes": int(len(arg)),
            "frac_at_frozen_layer": round(at_frozen, 4),
            "n_distinct_layers": int(arg.layer.nunique()),
            "layer_histogram": {int(k): int(v) for k, v in counts.items()},
            "median_layer": int(arg.layer.median()),
            "unrestricted_histogram_DIAGNOSTIC_ONLY":
                {int(k): int(v) for k, v in farg.layer.value_counts().sort_index().items()},
            # Tier D needs genes that disagree with the frozen layer. If nearly all of them peak
            # there, the arm is a re-run of the frozen-layer condition at full price.
            "spread_ok": bool(at_frozen < 0.80 and arg.layer.nunique() >= 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=27, help="the frozen layer")
    ap.add_argument("--mode", default="cds_mean")
    ap.add_argument("--clusters", type=Path, default=None,
                    help="h1c_clusters/assignments.csv; defaults to <run>/h1c_clusters/")
    ap.add_argument("--band", nargs=2, type=int, default=[24, 27], metavar=("LOW", "HIGH"),
                    help="candidate-layer band for the Tier-D argmax; must match the stage-3 "
                         "candidate layers. Searching outside it is a known trap -- see tier_d_gate")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = args.run / "stage3_gates"
    out.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.run / "stage2" / "pooled_representations.npz", allow_pickle=True)
    genes = [str(g) for g in npz["genes"]]
    a = npz[args.mode]
    D = (a[:, 1, args.layer, :] - a[:, 0, args.layer, :]).astype(np.float64)

    # ---- Tier C ----
    cpath = args.clusters or (args.run / "h1c_clusters" / "assignments.csv")
    summary: dict = {"layer": args.layer, "mode": args.mode, "seed": SEED, "k_grid": K_GRID}
    if cpath.exists():
        asg = pd.read_csv(cpath)
        asg = asg.set_index("gene").reindex(genes)
        if asg.cluster.isna().any():
            raise AssertionError("cluster assignments do not cover every gene in the npz")
        clusters = asg.cluster.to_numpy(int)
        h1c = json.loads((cpath.parent / "h1c_summary.json").read_text())
        tc = tier_c_gate(D, clusters, rng)
        tc.to_csv(out / "tier_c_cosine_contrast.csv", index=False)
        print(f"TIER C -- cluster-matched vs cross-cluster panel directions "
              f"(k* = {h1c['k_selected']}, {len(np.unique(clusters))} clusters)")
        print("  contrast > 0 means the cluster split carries directional information at that k\n")
        print(tc.to_string(index=False))
        feasible = tc[tc.usable & tc.k.isin([3, 4, 5, 6, 8])]
        best = feasible.contrast.max() if len(feasible) else np.nan
        # H1c must ALSO have passed its own gates -- a contrast on an unstable partition is not a
        # licence, it is a coincidence.
        licensed = bool(np.isfinite(best) and best > 0 and h1c.get("tier_c_licensed", False))
        summary["tier_c"] = {"best_contrast_at_feasible_k": None if not np.isfinite(best) else round(float(best), 4),
                             "h1c_gates_passed": bool(h1c.get("tier_c_licensed", False)),
                             "licensed": licensed}
        print(f"\n  best contrast at feasible k (3-8): {best:+.4f}"
              f"   H1c gates: {h1c.get('tier_c_licensed')}"
              f"   ->  TIER C {'LICENSED' if licensed else 'NOT LICENSED'}")
    else:
        print(f"TIER C -- skipped, no cluster assignments at {cpath}")
        summary["tier_c"] = {"licensed": False, "reason": "no H1c partition"}

    # ---- Tier D ----
    pgp = args.run / f"geom_{args.mode}" / "per_gene_by_layer.csv"
    if pgp.exists():
        td = tier_d_gate(pd.read_csv(pgp), args.layer, tuple(args.band))
        summary["tier_d"] = td
        print(f"\nTIER D -- per-gene argmax layer of loo_cos, band L{args.band[0]}-L{args.band[1]}")
        print(f"  {td['n_genes']} genes, {td['n_distinct_layers']} distinct peak layers, "
              f"median L{td['median_layer']}, {td['frac_at_frozen_layer']:.1%} peak at the frozen "
              f"layer L{args.layer}")
        print(f"  histogram: {td['layer_histogram']}")
        print(f"  unrestricted (DIAGNOSTIC ONLY, do not gate on this): "
              f"{td['unrestricted_histogram_DIAGNOSTIC_ONLY']}")
        print(f"  ->  TIER D {'LICENSED' if td['spread_ok'] else 'NOT LICENSED'}")
    else:
        print(f"\nTIER D -- skipped, no per-gene layer table at {pgp}")
        summary["tier_d"] = {"spread_ok": False, "reason": "no per_gene_by_layer.csv"}

    (out / "gate4_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
