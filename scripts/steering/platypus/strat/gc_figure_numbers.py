"""Quotable GC numbers for figure 10, matching what each panel actually draws. CPU only.

Figure 10 (`gc_codon_figures.fig14`) has two panels that summarise the SAME generations with
DIFFERENT estimators, which is the easy thing to misquote:

  left  `panel_gc_by_position`  GC1 / GC2 / GC3 as lines -> the MEAN over genes
  right `panel_gc`              overall GC as a boxplot  -> the MEDIAN over genes (box = IQR),
                                against human and platypus reference rules drawn at their MEAN

This emits both estimators for every quantity, plus the reference levels and the per-gene overshoot
against each species, so a number can be quoted without re-deriving which panel it came from.

Reads only the tracked figure_data tables that figure 10 renders from. Nothing is recomputed from
sequence, and no figure is modified.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

# The dose ladder in the order figure 10 plots it; unsteered is alpha = 0.
DOSE = ["unsteered", "add_a0.5", "add_a1.0", "add_a2.0", "add_a3.0", "add_a4.0"]
# Arms that are not rungs of the ladder but belong in a numbers table.
EXTRA = ["add_own", "random_a1.0", "add_gc_removed_a1.0"]
ALPHA = {
    "unsteered": 0.0,
    "add_a0.5": 0.5,
    "add_a1.0": 1.0,
    "add_a2.0": 2.0,
    "add_a3.0": 3.0,
    "add_a4.0": 4.0,
}


def summarise(s: pd.Series) -> dict:
    return {
        "n_genes": int(s.notna().sum()),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "q25": round(float(s.quantile(0.25)), 2),
        "q75": round(float(s.quantile(0.75)), 2),
        "sd": round(float(s.std(ddof=1)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--figure-data", type=Path, default=ROOT / "figure_data")
    ap.add_argument(
        "--out", type=Path, default=ROOT / "results/2026-08-08_platypus-strat-400/gc_numbers"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(args.figure_data / "exp2_generation_composition.csv")
    codon = pd.read_csv(args.figure_data / "exp2_codon_substitutions.csv")
    ref = pd.read_csv(args.figure_data / "exp2_generation_reference_windows.csv").set_index("gene")

    conds = [c for c in DOSE + EXTRA if c in set(comp.condition)]
    rows = []
    for cond in conds:
        cg = comp[comp.condition == cond].set_index("gene")
        ck = codon[codon.condition == cond].set_index("gene")
        ix = cg.index.intersection(ref.index)
        for col, panel, source in (
            ("gc", "right (boxplot, median drawn)", "composition"),
            ("gc3", "left (line, mean drawn)", "composition"),
        ):
            st = summarise(cg[col])
            d_p = cg.loc[ix, col] - ref.loc[ix, f"{col}_platypus"]
            d_h = cg.loc[ix, col] - ref.loc[ix, f"{col}_human"]
            rows.append(
                {
                    "quantity": {"gc": "overall GC", "gc3": "GC3"}[col],
                    "condition": cond,
                    "alpha": ALPHA.get(cond, np.nan),
                    "panel": panel,
                    "source_table": source,
                    **st,
                    "vs_platypus_mean": round(float(d_p.mean()), 2),
                    "vs_human_mean": round(float(d_h.mean()), 2),
                    "pct_genes_above_platypus": round(100 * float((d_p > 0).mean()), 1),
                }
            )
        for col in ("gc1", "gc2", "gc3"):
            if col in ck.columns and len(ck):
                rows.append(
                    {
                        "quantity": col.upper(),
                        "condition": cond,
                        "alpha": ALPHA.get(cond, np.nan),
                        "panel": "left (line, mean drawn)",
                        "source_table": "codon",
                        **summarise(ck[col]),
                        "vs_platypus_mean": np.nan,
                        "vs_human_mean": np.nan,
                        "pct_genes_above_platypus": np.nan,
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(args.out / "gc_figure_numbers.csv", index=False)

    refs = pd.DataFrame(
        [
            {
                "quantity": q,
                "species": sp,
                "mean_drawn_on_figure": round(float(ref[f"{c}_{sp}"].mean()), 2),
                "median": round(float(ref[f"{c}_{sp}"].median()), 2),
                "n_genes": int(ref[f"{c}_{sp}"].notna().sum()),
            }
            for q, c in (("overall GC", "gc"), ("GC3", "gc3"))
            for sp in ("human", "platypus")
        ]
    )
    refs.to_csv(args.out / "gc_reference_levels.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== reference windows (the dashed rules; figure draws the MEAN) ===")
    print(refs.to_string(index=False))
    for q in ("overall GC", "GC3", "GC1", "GC2"):
        t = tab[(tab.quantity == q) & tab.alpha.notna()].sort_values("alpha")
        if t.empty:
            continue
        print(f"\n=== {q} — {t.panel.iloc[0]} ===")
        # GC3 is emitted from BOTH source tables on purpose: the two are computed on the same
        # window by different code paths, so their agreement is a check, not a duplicate row.
        cols = ["alpha", "source_table", "n_genes", "mean", "median", "q25", "q75", "sd"]
        if t.vs_platypus_mean.notna().any():
            cols += ["vs_platypus_mean", "vs_human_mean", "pct_genes_above_platypus"]
        print(t[cols].to_string(index=False))
    ex = tab[tab.alpha.isna() & tab.quantity.isin(["overall GC", "GC3"])]
    if len(ex):
        print("\n=== non-ladder arms ===")
        cols = ["quantity", "source_table", "condition", "mean", "median", "vs_platypus_mean"]
        print(ex[cols].to_string(index=False))
    print(f"\nwrote {args.out / 'gc_figure_numbers.csv'} and gc_reference_levels.csv")


if __name__ == "__main__":
    main()
