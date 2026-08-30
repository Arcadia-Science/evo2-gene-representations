"""Stage 4 analysis — did the held-out direction steer, and does it depend on conservation?"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module
from strat_metric import add_metric_args, load_scores  # noqa: E402

# Everything except the site-recovery metric, which is selected at runtime and prepended.
SECONDARY_READOUTS = ["aa_id_to_target", "nt_id_to_target", "indel_bp", "n_stop_codons"]
SEED = 42


def boot_ci(x: np.ndarray, n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    """Bootstrap CI over genes -- genes are homology blocks here, so this is a cluster bootstrap."""
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", required=True, help="stage4 output dir name under --run")
    add_metric_args(ap)
    args = ap.parse_args()

    d, spec = load_scores(args.run, args.dir, scores=args.scores, metric=args.metric,
                          min_voters=args.min_voters)
    # Write the (possibly voter-filtered) alias back under its real name, so every emitted column is
    # labelled with the site set it came from and no downstream reader has to guess.
    prim = spec["col"]
    d[prim] = d["metric"]
    READOUTS = [prim] + SECONDARY_READOUTS
    print(f"{len(d)} rows, {d.gene.nunique()} genes, conditions: "
          f"{sorted(d.condition.unique())}\n")

    # gene x condition means: samples within a cell are correlated (shared prompt), so the gene mean
    # is the unit, not the individual generation
    g = d.groupby(["gene", "condition"])[READOUTS].mean().reset_index()
    # Use the scorable-site count for the selected site set.
    meta = d.groupby("gene").agg(stratum=("stratum", "first"),
                                 perc_id_hp=("perc_id_hp", "first"),
                                 n_sites=(spec["n_col"], "first"),
                                 gen_bp=("gen_bp", "first")).reset_index()

    base = g[g.condition == "unsteered"].set_index("gene")
    conds = [c for c in g.condition.unique() if c != "unsteered"]

    # ---- 1. does it steer -----------------------------------------------------------------------
    print("=" * 96)
    print("1. PAIRED steered vs unsteered, per gene (mean over samples). n = genes = blocks")
    print("=" * 96)
    rows = []
    for c in sorted(conds):
        sub = g[g.condition == c].set_index("gene")
        common = sub.index.intersection(base.index)
        line = {"condition": c, "n": len(common)}
        for r in READOUTS:
            delta = (sub.loc[common, r] - base.loc[common, r]).to_numpy(float)
            # Per-readout n, because `n` above counts genes PRESENT in the cell while the mean and
            # the CI are taken over the finite values only. For the autapomorphic site set those two
            # differ by every gene with no ortholog evidence, and reporting the larger number beside
            # an estimate built from the smaller one overstates the evidence.
            line[f"{r}_n"] = int(np.isfinite(delta).sum())
            lo, hi = boot_ci(delta)
            try:
                _s, p = stats.wilcoxon(delta[np.isfinite(delta)])
            except ValueError:
                p = np.nan
            line[f"{r}_delta"] = round(float(np.nanmean(delta)), 3)
            line[f"{r}_ci"] = f"[{lo:+.2f},{hi:+.2f}]"
            line[f"{r}_p"] = float(p)
        rows.append(line)
    res = pd.DataFrame(rows)
    show = ["condition", "n", f"{prim}_n", f"{prim}_delta", f"{prim}_ci",
            f"{prim}_p", "aa_id_to_target_delta", "aa_id_to_target_p",
            "indel_bp_delta", "n_stop_codons_delta"]
    print(res[show].to_string(index=False))
    print("\nabsolute means by condition:")
    print(g.groupby("condition")[READOUTS].mean().round(2).to_string())

    # ---- 2. steered vs its own random control ---------------------------------------------------
    print("\n" + "=" * 96)
    print("2. steered vs NORM-MATCHED RANDOM (the direction test, not the perturbation test)")
    print("=" * 96)
    rnd = g[g.condition == "random_a1.0"].set_index("gene") if "random_a1.0" in conds else None
    if rnd is not None:
        for c in sorted(x for x in conds if x != "random_a1.0"):
            sub = g[g.condition == c].set_index("gene")
            common = sub.index.intersection(rnd.index)
            for r in (prim, "aa_id_to_target"):
                delta = (sub.loc[common, r] - rnd.loc[common, r]).to_numpy(float)
                lo, hi = boot_ci(delta)
                _s, p = stats.wilcoxon(delta[np.isfinite(delta)])
                print(f"  {c:14s} vs random  {r:20s} {np.nanmean(delta):+.2f} pp "
                      f"[{lo:+.2f},{hi:+.2f}]  p={p:.4g}")

    # ---- 3. conservation gradient (exploratory) -------------------------------------------------
    print("\n" + "=" * 96)
    print("3. does the GAIN depend on conservation?  EXPLORATORY: n=100 detects rho~0.28 at 80% power")
    print("   -- the same power as the 2026-08-04 null, so a null here is uninformative")
    print("=" * 96)
    grows = []
    for c in sorted(conds):
        sub = g[g.condition == c].set_index("gene")
        common = sub.index.intersection(base.index)
        for r in (prim, "aa_id_to_target"):
            gain = (sub.loc[common, r] - base.loc[common, r]).rename("gain").reset_index()
            m = gain.merge(meta, on="gene")
            rho, p = stats.spearmanr(m.perc_id_hp, m.gain, nan_policy="omit")
            grows.append({"condition": c, "readout": r, "rho_vs_perc_id": round(float(rho), 3),
                          "p": float(p), "n": len(m)})
            if r == prim:
                print(f"\n  {c} -- mean gain by stratum (0 = fastest, 4 = most conserved):")
                bs = m.groupby("stratum").gain.agg(["mean", "std", "size"]).round(2)
                print("    " + bs.to_string().replace("\n", "\n    "))
    gr = pd.DataFrame(grows)
    print("\n  gain vs perc_id_hp (Spearman):")
    print("    " + gr.to_string(index=False).replace("\n", "\n    "))

    outp = args.run / args.dir / "analysis_summary.csv"
    res.to_csv(outp, index=False)
    gr.to_csv(args.run / args.dir / "analysis_gradient.csv", index=False)
    print(f"\n-> {outp}")


if __name__ == "__main__":
    sys.exit(main())
