"""Is the rise in platypus-choice (C) explained by the GC shift steering induces? CPU only.

Two tests on the site-level private-site calls (`extract_private_site_calls.py`), no regeneration:

1. SUBSTITUTION CLASS. If a generic push toward G/C explains C, the rise must live in the sites
   whose human->platypus substitution ADDS G or C (AT->GC) and must be absent -- or reversed -- at
   sites where it removes them (GC->AT). C is computed separately for AT->GC, GC->AT, A<->T, G<->C.

2. COMPOSITION-MATCHED PERMUTATION NULL. Holding the generated base Y, the human base H, coverage
   and the departure indicator (Y != H) fixed, the platypus target base P is permuted among private
   sites within (human base x codon position x conservation stratum). Generated composition and
   target composition are both preserved; only the site-specific correspondence between Y and the
   correct P is destroyed. Matching on H is what makes the null valid: every site in a group shares
   H, and P != H at every private site, so a permuted P can never equal H and so can never
   manufacture a hit at a site that is not a departure.

C uses the PUBLISHED aggregation throughout -- C = 100 * (Y==P) / (Y!=H) over covered sites per
(gene, condition, sample), averaged over samples within a gene, then over genes -- and the script
checks the observed values against `site_directionality/levels_by_condition.csv`. Because that
statistic pools sites within a gene and then weights genes equally, uncertainty is also reported by
resampling GENES rather than sites.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_style as acs  # noqa: E402

GC = frozenset("GC")
ALPHAS = [0.5, 1.0, 2.0, 3.0, 4.0]
CONDITIONS = ["unsteered"] + [f"add_a{a}" for a in ALPHAS]
COND_ALPHA = {"unsteered": 0.0, **{f"add_a{a}": a for a in ALPHAS}}
CLASSES = ["AT->GC", "GC->AT", "A<->T", "G<->C"]
GREY = acs.SERIES_MUTED


def subst_class(h: str, p: str) -> str:
    hg, pg = h in GC, p in GC
    if not hg and pg:
        return "AT->GC"
    if hg and not pg:
        return "GC->AT"
    return "G<->C" if hg else "A<->T"


class CAggregator:
    """The published C aggregation, precompiled for one slice of rows.

    Only the numerator changes when P is permuted; which rows are covered, which are departures,
    and the record and gene each row belongs to are fixed, so they are computed once here and
    reused across all 1,000 permutations.
    """

    def __init__(self, rows: pd.DataFrame):
        rec_code, rec_index = pd.factorize(
            pd.Series(list(zip(rows.gene, rows["sample"], strict=True))), sort=True
        )
        self.rec_id = rec_code.astype(np.int64)
        self.n_rec = len(rec_index)
        gene_code, gene_index = pd.factorize(pd.Series([r[0] for r in rec_index]), sort=True)
        self.gene_id_of_rec = gene_code.astype(np.int64)
        self.genes = np.asarray(gene_index)
        self.n_genes = len(self.genes)
        self.covered = rows.covered.to_numpy(bool)
        self.y = rows.y.to_numpy(np.int16)
        self.den = np.bincount(self.rec_id, rows.departure.to_numpy(bool), minlength=self.n_rec)

    def gene_means_from_p(self, p_row: np.ndarray) -> np.ndarray:
        """Per-gene C (mean over that gene's samples) for one row-aligned assignment of P."""
        hit = self.covered & (self.y == p_row)
        num = np.bincount(self.rec_id, hit, minlength=self.n_rec)
        ok = self.den > 0
        c_rec = np.full(self.n_rec, np.nan)
        c_rec[ok] = 100.0 * num[ok] / self.den[ok]
        g_sum = np.bincount(self.gene_id_of_rec[ok], c_rec[ok], minlength=self.n_genes)
        g_n = np.bincount(self.gene_id_of_rec[ok], minlength=self.n_genes)
        return np.where(g_n > 0, g_sum / np.maximum(g_n, 1), np.nan)


def permute_within_groups(
    p: np.ndarray, starts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle P inside each contiguous group of the group-sorted site table."""
    out = p.copy()
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        if b - a > 1:
            out[a:b] = rng.permutation(out[a:b])
    return out


def gene_bootstrap(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI of the panel mean, resampling GENES as the independent unit."""
    v = x[np.isfinite(x)]
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def excess_increments(per_gene: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired change in C, in the null, and in the excess, relative to the SAME gene unsteered.

    The absolute excess answers "does the model target the right site at all"; this answers the
    question actually asked -- does the RISE in C with dose exceed what the composition shift alone
    produces. Genes are the resampling unit, matching the aggregation.
    """
    per_gene = per_gene.copy()
    per_gene["excess"] = per_gene.C_obs - per_gene.C_null_mean
    wide = {
        k: per_gene.pivot_table(index="gene", columns="condition", values=k)
        for k in ("C_obs", "C_null_mean", "excess")
    }
    rng = np.random.default_rng(seed)
    rows = []
    for cond in CONDITIONS:
        if cond == "unsteered":
            continue
        d_ex = (wide["excess"][cond] - wide["excess"]["unsteered"]).to_numpy()
        v = d_ex[np.isfinite(d_ex)]
        draws = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
        d_obs = float((wide["C_obs"][cond] - wide["C_obs"]["unsteered"]).mean())
        d_null = float((wide["C_null_mean"][cond] - wide["C_null_mean"]["unsteered"]).mean())
        rows.append(
            {
                "condition": cond,
                "alpha": COND_ALPHA[cond],
                "n_genes": int(len(v)),
                "dC_obs": round(d_obs, 4),
                "dC_null": round(d_null, 4),
                "d_excess": round(float(v.mean()), 4),
                "d_excess_ci_lo": round(float(np.percentile(draws, 2.5)), 4),
                "d_excess_ci_hi": round(float(np.percentile(draws, 97.5)), 4),
                "frac_of_rise_explained_by_composition": round(d_null / d_obs, 4)
                if d_obs
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def class_figure(cls_tab: pd.DataFrame, comp: pd.DataFrame, out: Path, stem: str) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=300, width_ratios=[1.5, 1])
    for cls, color in zip(CLASSES, acs.categorical(len(CLASSES)), strict=True):
        s = cls_tab[cls_tab.subst_class == cls].sort_values("alpha")
        if s.empty:
            continue
        ax0.errorbar(
            s.alpha,
            s.C_mean,
            yerr=[s.C_mean - s.ci_lo, s.ci_hi - s.C_mean],
            color=color,
            lw=1.8,
            marker="o",
            ms=4.5,
            capsize=2,
            elinewidth=0.9,
            label=f"{cls}  ({int(s.n_sites.iloc[0]):,} sites)",
        )
    ax0.set_xticks([0, *ALPHAS])
    ax0.set_xlabel("alpha   (0 = unsteered)")
    ax0.set_ylabel("platypus choice C (%)")
    ax0.set_title("C by human-to-platypus substitution class", fontsize=9, fontweight="bold")
    ax0.legend(fontsize=6.4, frameon=False, loc="best")

    x = np.arange(len(comp))
    ax1.bar(
        x - 0.19, comp.frac_P_is_GC, width=0.36, color=acs.SERIES_PRIMARY, label="platypus base"
    )
    ax1.bar(x + 0.19, comp.frac_H_is_GC, width=0.36, color=GREY, label="human base")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{r.site_set}\n({int(r.n_sites):,} sites)" for r in comp.itertuples()], fontsize=7
    )
    ax1.set_ylabel("fraction G or C")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color=GREY, ls=":", lw=0.8)
    ax1.set_title("Base composition at mismatch sites", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=6.6, frameon=False)

    fig.suptitle(
        "Does a generic GC push explain platypus choice?", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.5,
        -0.04,
        "Left: C computed within each substitution class, same aggregation as the published "
        "statistic; error bars are 95% CIs from resampling genes.\nA generic GC shift should lift "
        "AT-to-GC and not GC-to-AT. Right: G/C fraction of the target and human bases at private "
        "vs non-private mismatch sites.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out}")


def null_figure(tab: pd.DataFrame, inc: pd.DataFrame, out: Path, stem: str) -> None:
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12.6, 3.9), dpi=300)
    t = tab.sort_values("alpha")
    ax0.fill_between(
        t.alpha, t.null_lo, t.null_hi, color=GREY, alpha=0.30, lw=0, label="null 95% interval"
    )
    ax0.plot(t.alpha, t.null_mean, color=GREY, lw=1.6, ls="--", label="composition-matched null")
    ax0.errorbar(
        t.alpha,
        t.C_obs,
        yerr=[t.C_obs - t.obs_ci_lo, t.obs_ci_hi - t.C_obs],
        color=acs.SERIES_PRIMARY,
        lw=2.0,
        marker="o",
        ms=5,
        capsize=2,
        elinewidth=0.9,
        label="observed C (95% CI over genes)",
    )
    ax0.set_xticks([0, *ALPHAS])
    ax0.set_xlabel("alpha   (0 = unsteered)")
    ax0.set_ylabel("platypus choice C (%)")
    ax0.set_title("Observed C vs composition-matched null", fontsize=9, fontweight="bold")
    ax0.legend(fontsize=6.4, frameon=False, loc="best")

    ax1.axhline(0, color=GREY, ls=":", lw=1.0)
    ax1.errorbar(
        t.alpha,
        t.C_excess,
        yerr=[t.C_excess - t.excess_ci_lo, t.excess_ci_hi - t.C_excess],
        color=acs.HIGHLIGHT,
        lw=2.0,
        marker="D",
        ms=4.5,
        capsize=2,
        elinewidth=0.9,
    )
    ax1.set_xticks([0, *ALPHAS])
    ax1.set_xlabel("alpha   (0 = unsteered)")
    ax1.set_ylabel("C excess = observed - null (pp)")
    ax1.set_ylim(0, max(tab.excess_ci_hi) * 1.15)
    ax1.set_title("Excess over composition (absolute)", fontsize=9, fontweight="bold")

    # The question actually asked: does the RISE exceed the composition-driven rise?
    i = inc.sort_values("alpha")
    ax2.axhline(0, color=GREY, ls=":", lw=1.0)
    ax2.errorbar(
        i.alpha,
        i.d_excess,
        yerr=[i.d_excess - i.d_excess_ci_lo, i.d_excess_ci_hi - i.d_excess],
        color=acs.HIGHLIGHT,
        lw=2.0,
        marker="D",
        ms=4.5,
        capsize=2,
        elinewidth=0.9,
        label="excess gained over unsteered",
    )
    ax2.plot(
        i.alpha, i.dC_obs, color=acs.SERIES_PRIMARY, lw=1.6, marker="o", ms=4, label="rise in C"
    )
    ax2.plot(
        i.alpha, i.dC_null, color=GREY, lw=1.6, ls="--", marker="s", ms=3.5, label="rise in null"
    )
    ax2.set_xticks(ALPHAS)
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("change vs the same gene unsteered (pp)")
    ax2.set_title("Is the RISE more than composition?", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=6.4, frameon=False, loc="best")

    fig.suptitle(
        "Platypus choice against a composition-matched permutation null",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Null permutes the platypus base among private sites within (human base x codon position x "
        "conservation stratum), holding the generated\nbase, coverage and the departure indicator "
        "fixed. Error bars are 95% CIs from resampling genes. Right panel is paired within gene "
        "against that gene's unsteered generation.",
        ha="center",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}  ->  {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, default=ROOT / "results/2026-08-08_platypus-strat-400")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/gc_composition_null/")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    acs.setup()
    out = args.out or (args.run / "gc_composition_null")
    calls = pd.read_csv(out / "private_site_calls.csv.gz")
    defs = pd.read_csv(out / "site_definitions.csv.gz")
    pairs = pd.read_csv(args.run / "stage1" / "pairs.csv")
    strat = dict(zip(pairs.gene, pairs.stratum.astype(int), strict=True))

    # 1a. target-base composition, private vs non-private mismatch sites
    defs["P_is_GC"] = defs.p_base.isin(list(GC))
    defs["H_is_GC"] = defs.h_base.isin(list(GC))
    comp = (
        defs.groupby("site_set")
        .agg(
            n_sites=("idx", "size"),
            frac_P_is_GC=("P_is_GC", "mean"),
            frac_H_is_GC=("H_is_GC", "mean"),
        )
        .reset_index()
    )
    comp["site_set"] = pd.Categorical(
        comp.site_set, ["private", "shared_not_private"], ordered=True
    )
    comp = comp.sort_values("site_set")
    comp.to_csv(out / "target_base_composition.csv", index=False)

    # 1b. C within each substitution class
    calls = calls[calls.condition.isin(CONDITIONS)].copy()
    calls["subst_class"] = [
        subst_class(h, p) for h, p in zip(calls.h_base, calls.p_base, strict=True)
    ]
    calls["stratum"] = calls.gene.map(strat)
    calls["alpha"] = calls.condition.map(COND_ALPHA)
    site_counts = (
        calls.drop_duplicates(["gene", "idx"]).groupby("subst_class").size().rename("n_sites")
    )

    rows = []
    for cond in CONDITIONS:
        c_all = calls[calls.condition == cond]
        for cls in CLASSES:
            sub = c_all[c_all.subst_class == cls]
            if sub.empty:
                continue
            agg = CAggregator(sub)
            gm = agg.gene_means_from_p(sub.p.to_numpy(np.int16))
            lo, hi = gene_bootstrap(gm, args.n_boot, args.seed)
            rows.append(
                {
                    "subst_class": cls,
                    "condition": cond,
                    "alpha": COND_ALPHA[cond],
                    "n_sites": int(site_counts.get(cls, 0)),
                    "n_genes": int(np.isfinite(gm).sum()),
                    "C_mean": float(np.nanmean(gm)),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    cls_tab = pd.DataFrame(rows)
    cls_tab.to_csv(out / "C_by_substitution_class.csv", index=False)

    # 2. composition-matched permutation null
    sites = calls.drop_duplicates(["gene", "idx"])[
        ["gene", "idx", "h", "p", "codon_pos", "stratum"]
    ].reset_index(drop=True)
    sites["group"] = (
        sites.h.astype(int) * 1000 + sites.codon_pos.astype(int) * 100 + sites.stratum.astype(int)
    )
    sites = sites.sort_values("group").reset_index(drop=True)
    site_index = {k: i for i, k in enumerate(zip(sites.gene, sites.idx, strict=True))}
    starts = np.concatenate(
        [[0], np.flatnonzero(np.diff(sites.group.to_numpy())) + 1, [len(sites)]]
    )
    p_obs = sites.p.to_numpy(np.int16)
    print(
        f"permutation groups: {len(starts) - 1} (human base x codon position x stratum) over "
        f"{len(sites):,} private sites"
    )

    null_rows, per_gene = [], []
    rng = np.random.default_rng(args.seed)
    for cond in CONDITIONS:
        sub = calls[calls.condition == cond]
        key = np.array([site_index[k] for k in zip(sub.gene, sub.idx, strict=True)], dtype=np.int64)
        agg = CAggregator(sub)
        obs_gm = agg.gene_means_from_p(p_obs[key])
        draws = np.empty(args.n_perm)
        g_sum, g_n = np.zeros(agg.n_genes), np.zeros(agg.n_genes)
        for b in range(args.n_perm):
            gm = agg.gene_means_from_p(permute_within_groups(p_obs, starts, rng)[key])
            draws[b] = float(np.nanmean(gm))
            fin = np.isfinite(gm)
            g_sum[fin] += gm[fin]
            g_n[fin] += 1
        gene_null = np.where(g_n > 0, g_sum / np.maximum(g_n, 1), np.nan)
        per_gene.append(
            pd.DataFrame(
                {
                    "condition": cond,
                    "alpha": COND_ALPHA[cond],
                    "gene": agg.genes,
                    "C_obs": obs_gm,
                    "C_null_mean": gene_null,
                }
            )
        )
        obs = float(np.nanmean(obs_gm))
        o_lo, o_hi = gene_bootstrap(obs_gm, args.n_boot, args.seed)
        ex_lo, ex_hi = gene_bootstrap(obs_gm - gene_null, args.n_boot, args.seed)
        null_rows.append(
            {
                "condition": cond,
                "alpha": COND_ALPHA[cond],
                "n_genes": int(np.isfinite(obs_gm).sum()),
                "C_obs": obs,
                "obs_ci_lo": o_lo,
                "obs_ci_hi": o_hi,
                "null_mean": float(draws.mean()),
                "null_lo": float(np.percentile(draws, 2.5)),
                "null_hi": float(np.percentile(draws, 97.5)),
                "C_excess": obs - float(draws.mean()),
                "excess_ci_lo": ex_lo,
                "excess_ci_hi": ex_hi,
                "p_perm": float((np.sum(draws >= obs) + 1) / (args.n_perm + 1)),
            }
        )
        print(
            f"  {cond:10s} C_obs {obs:6.2f}   null {draws.mean():6.2f}   "
            f"excess {obs - draws.mean():+6.2f}"
        )

    null_tab = pd.DataFrame(null_rows)
    null_tab.to_csv(out / "C_vs_composition_null.csv", index=False)
    pg = pd.concat(per_gene, ignore_index=True)
    pg.to_csv(out / "C_vs_null_per_gene.csv", index=False)
    inc = excess_increments(pg, args.n_boot, args.seed)
    inc.to_csv(out / "C_excess_increment_vs_unsteered.csv", index=False)

    # validation against the published table
    lv = pd.read_csv(args.run / "site_directionality" / "levels_by_condition.csv")
    lv = lv[lv.site_set == "private"].set_index("condition")
    chk = pd.DataFrame(
        [
            {
                "condition": r.condition,
                "C_recomputed": round(r.C_obs, 4),
                "C_published": round(float(lv.loc[r.condition, "C_mean"]), 4),
                "abs_diff": round(abs(r.C_obs - float(lv.loc[r.condition, "C_mean"])), 4),
                "n_genes_recomputed": r.n_genes,
                "n_genes_published": int(lv.loc[r.condition, "n_genes"]),
            }
            for r in null_tab.itertuples()
            if r.condition in lv.index
        ]
    )
    chk.to_csv(out / "validation_vs_published_C.csv", index=False)

    class_figure(cls_tab, comp, out, "C_by_substitution_class")
    null_figure(null_tab, inc, out, "C_vs_composition_null")

    print("\n=== validation against site_directionality/levels_by_condition.csv ===")
    print(chk.to_string(index=False))
    print("\n=== target base composition ===")
    print(comp.to_string(index=False))
    print("\n=== C by substitution class (%) ===")
    print(cls_tab.pivot_table(index="alpha", columns="subst_class", values="C_mean").round(2))
    print("\n=== observed vs composition-matched null ===")
    print(null_tab.round(3).to_string(index=False))
    print("\n=== rise vs unsteered: observed, null, and the excess gained ===")
    print(inc.to_string(index=False))


if __name__ == "__main__":
    main()
