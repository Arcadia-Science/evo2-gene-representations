"""Headroom-normalised steering deltas: does available room explain the stratum pattern? No GPU.

Reviewer question. The unsteered baseline differs across conservation strata, so a raw delta in
percentage points may reward a gene simply for having more room to move. This rescales each
gene's own delta by its own available room and asks whether the stratum pattern survives:

    private recovery    (S - U) / (100 - U)   fraction of the room toward 100% that steering takes
    amino-acid identity (S - U) / U           fraction of the unsteered identity that is lost

Nothing about generation or scoring changes -- S and U are the same per-gene condition means the
publication figures use, read from the tracked figure_data table.

The `random_a*` arms are carried through the identical normalisation as a control. They are not
part of the requested alpha ladder; they are here because a normalisation that inflates noise
would show it there, where the mean effect is known to be ~0.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling module
import arcadia_style as acs  # noqa: E402
from strat_metric import ALIASES, DEFAULT_METRIC, METRICS  # noqa: E402

CONS_CMAP = LinearSegmentedColormap.from_list(
    "cons_seq", [acs.apc.sky, acs.apc.vital, acs.apc.aegean, acs.SERIES_PRIMARY]
)
STRATUM_POS = [0.12, 0.34, 0.55, 0.76, 0.96]
INK, PLUM, GREY = acs.SERIES_PRIMARY, acs.SERIES_THIRD, acs.SERIES_MUTED

AA_COL = "aa_id_to_target"
ALPHAS = [0.5, 1.0, 2.0, 3.0, 4.0]
CONTROL_ALPHAS = [0.5, 1.0, 2.0]  # the alphas that have a matched norm-matched random arm
N_BOOT = 2000
SEED = 0


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over genes."""
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def per_gene(d: pd.DataFrame, priv_col: str) -> pd.DataFrame:
    """One row per gene x condition: raw and headroom-normalised delta for both readouts.

    The normalisers are the gene's OWN unsteered value, so a gene with no room left is undefined
    rather than silently infinite -- guarded here even though the panel contains no such gene.
    """
    u = d[d.condition == "unsteered"].set_index("gene")
    rows = []
    for cond in sorted(set(d.condition) - {"unsteered"}):
        s = d[d.condition == cond].set_index("gene")
        genes = s.index.intersection(u.index)
        up, sp = u.loc[genes, priv_col], s.loc[genes, priv_col]
        ua, sa = u.loc[genes, AA_COL], s.loc[genes, AA_COL]
        # (100 - U) == 0 and U == 0 are the two undefined cases the reviewer asked us to isolate.
        priv_room = (100.0 - up).where(lambda v: v > 0)
        aa_base = ua.where(lambda v: v > 0)
        rows.append(
            pd.DataFrame(
                {
                    "gene": genes,
                    "condition": cond,
                    "stratum": u.loc[genes, "stratum"].astype(int).to_numpy(),
                    "perc_id_hp": u.loc[genes, "perc_id_hp"].to_numpy(),
                    "priv_U": up.to_numpy(),
                    "priv_S": sp.to_numpy(),
                    "priv_raw": (sp - up).to_numpy(),
                    "priv_norm": (100.0 * (sp - up) / priv_room).to_numpy(),
                    "aa_U": ua.to_numpy(),
                    "aa_S": sa.to_numpy(),
                    "aa_raw": (sa - ua).to_numpy(),
                    "aa_norm": (100.0 * (sa - ua) / aa_base).to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarise(g: pd.DataFrame) -> pd.DataFrame:
    """n / mean / median / bootstrap 95% CI per readout x scale x stratum x condition."""
    rows = []
    specs = [
        ("private recovery", "raw", "priv_raw", "pp"),
        ("private recovery", "headroom-normalised", "priv_norm", "% of room to 100%"),
        ("amino-acid identity", "raw", "aa_raw", "pp"),
        ("amino-acid identity", "headroom-normalised", "aa_norm", "% of unsteered identity"),
    ]
    for cond, sub_c in g.groupby("condition"):
        for st in [None, *sorted(sub_c.stratum.unique())]:
            q = sub_c if st is None else sub_c[sub_c.stratum == st]
            for readout, scale, col, unit in specs:
                v = q[col].to_numpy(float)
                v = v[np.isfinite(v)]
                lo, hi = boot_ci(v)
                rows.append(
                    {
                        "readout": readout,
                        "scale": scale,
                        "unit": unit,
                        "condition": cond,
                        "stratum": "all" if st is None else int(st),
                        "n": len(v),
                        "n_undefined": int(len(q) - len(v)),
                        "mean": round(float(np.mean(v)), 3) if len(v) else np.nan,
                        "median": round(float(np.median(v)), 3) if len(v) else np.nan,
                        "ci_lo": round(lo, 3),
                        "ci_hi": round(hi, 3),
                    }
                )
    return pd.DataFrame(rows)


def widen(tab: pd.DataFrame) -> pd.DataFrame:
    """One row per stratum x alpha, the four readout-x-scale cells spread across columns.

    The long table is the machine-readable one; this is the shape a reader expects to scan.
    """
    keys = {
        ("private recovery", "raw"): "priv_raw",
        ("private recovery", "headroom-normalised"): "priv_norm",
        ("amino-acid identity", "raw"): "aa_raw",
        ("amino-acid identity", "headroom-normalised"): "aa_norm",
    }
    out = None
    for (readout, scale), prefix in keys.items():
        sub = tab[(tab.readout == readout) & (tab.scale == scale)].set_index(
            ["stratum", "condition"]
        )
        block = sub[["mean", "median", "ci_lo", "ci_hi"]].add_prefix(f"{prefix}_")
        if out is None:
            out = sub[["n", "n_undefined"]].join(block)
        else:
            out = out.join(block)
    out = out.reset_index().rename(columns={"condition": "alpha"})
    out["alpha"] = out.alpha.str.replace("add_a", "", regex=False).astype(float)
    return out.sort_values(["stratum", "alpha"]).reset_index(drop=True)


def dose_figure(g: pd.DataFrame, alphas: list[float], out: Path, stem: str) -> None:
    """Normalised analogue of figure 8: one panel per stratum, x = alpha, both readouts."""
    strata = sorted(g.stratum.unique())
    fig, axes = plt.subplots(1, len(strata), figsize=(2.6 * len(strata), 3.4), dpi=300, sharey=True)
    for ax, st in zip(np.atleast_1d(axes), strata, strict=True):
        sub = g[g.stratum == st]
        for col, label, color, mk in (
            ("priv_norm", "private bp gained\n(% of room to 100%)", INK, "o"),
            ("aa_norm", "amino-acid identity lost\n(% of unsteered)", PLUM, "^"),
        ):
            ys, los, his = [], [], []
            for a in alphas:
                v = sub.loc[sub.condition == f"add_a{a}", col].to_numpy(float)
                v = v[np.isfinite(v)]
                ys.append(np.mean(v))
                lo, hi = boot_ci(v)
                los.append(lo)
                his.append(hi)
            ax.fill_between(alphas, los, his, color=color, alpha=0.16, lw=0)
            ax.plot(alphas, ys, "-", marker=mk, color=color, lw=1.8, ms=5, label=label)
        ax.axhline(0, color=GREY, ls=":", lw=1.0)
        ax.set_xticks(alphas)
        ax.set_xlabel("α")
        ax.set_title(
            f"s{int(st)} · {sub.perc_id_hp.min():.0f}–{sub.perc_id_hp.max():.0f}% identity",
            fontsize=8.5,
            fontweight="bold",
            color=CONS_CMAP(STRATUM_POS[int(st)]),
        )
    np.atleast_1d(axes)[0].set_ylabel("headroom-normalised change (%)")
    np.atleast_1d(axes)[0].legend(fontsize=6.4, frameon=False, loc="upper left")
    fig.suptitle(
        "Headroom-normalised dose–response by conservation stratum",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.04,
        "Each gene's delta is divided by its OWN available room: (S−U)/(100−U) for private "
        "recovery, (S−U)/U for amino-acid identity.\nBands are 95% bootstrap CIs of the mean over "
        f"genes ({N_BOOT:,} resamples, n=80 per stratum).",
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


def comparison_figure(g: pd.DataFrame, alphas: list[float], out: Path, stem: str) -> None:
    """Raw vs normalised, side by side: does the pattern across strata survive rescaling?"""
    strata = sorted(g.stratum.unique())
    # Curves rise to the right in the private panels and fall in the amino-acid ones, so the free
    # corner for the rho annotation differs by row.
    panels = [
        ("priv_raw", "Private bp — raw Δ (pp)", "lower right"),
        ("priv_norm", "Private bp — normalised (% of room)", "lower right"),
        ("aa_raw", "Amino-acid identity — raw Δ (pp)", "upper right"),
        ("aa_norm", "Amino-acid identity — normalised (% of unsteered)", "upper right"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), dpi=300)
    shades = [acs.line_safe(c) for c in acs.gradient_colors(acs.SEQUENTIAL, len(alphas) + 1)]
    for ax, (col, title, loc) in zip(axes.ravel(), panels, strict=True):
        for a, color in zip(alphas, shades, strict=False):
            ys = [
                np.nanmean(g.loc[(g.stratum == st) & (g.condition == f"add_a{a}"), col])
                for st in strata
            ]
            ax.plot(strata, ys, "-o", color=color, lw=1.5, ms=4, label=f"α={a:g}")
        ax.axhline(0, color=GREY, ls=":", lw=1.0)
        ax.set_xticks(strata)
        ax.set_xticklabels([f"s{int(s)}" for s in strata])
        ax.set_xlabel("conservation stratum (s0 fastest → s4 conserved)")
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        # Trend across genes, not across the five plotted means.
        rho, p = stats.spearmanr(
            g.loc[g.condition == "add_a1.0", "perc_id_hp"],
            g.loc[g.condition == "add_a1.0", col],
            nan_policy="omit",
        )
        ax.text(
            0.98,
            0.03 if loc == "lower right" else 0.97,
            f"per-gene ρ vs conservation, α=1: {rho:+.2f} (p={p:.2g})",
            transform=ax.transAxes,
            ha="right",
            va="bottom" if loc == "lower right" else "top",
            fontsize=6.3,
            color=acs.ANNOTATION,
        )
    axes[0, 0].legend(fontsize=6.4, frameon=False, ncol=2, loc="upper left")
    fig.suptitle(
        "Does the conservation pattern survive headroom normalisation?",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Left column: the raw percentage-point deltas the publication figures plot. Right column: "
        "the same genes after dividing by their own\navailable room. Points are means over the 80 "
        "genes in each stratum.",
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


def trend_table(g: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Per-gene Spearman of each delta against conservation -- the requested pattern test."""
    rows = []
    for cond in conditions:
        sub = g[g.condition == cond]
        if sub.empty:
            continue
        for readout, col in (
            ("private recovery", "priv_raw"),
            ("private recovery", "priv_norm"),
            ("amino-acid identity", "aa_raw"),
            ("amino-acid identity", "aa_norm"),
        ):
            rho, p = stats.spearmanr(sub.perc_id_hp, sub[col], nan_policy="omit")
            rows.append(
                {
                    "readout": readout,
                    "scale": "raw" if col.endswith("_raw") else "headroom-normalised",
                    "condition": cond,
                    "n": int(np.isfinite(sub[col]).sum()),
                    "rho_vs_conservation": round(float(rho), 3),
                    "p": float(p),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", type=Path, default=ROOT / "figure_data/exp2_steering_outcomes.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/2026-08-08_platypus-strat-400/headroom_normalization",
    )
    ap.add_argument("--metric", choices=sorted(METRICS) + sorted(ALIASES), default=DEFAULT_METRIC)
    args = ap.parse_args()

    acs.setup()
    mspec = METRICS[ALIASES.get(args.metric, args.metric)]
    d = pd.read_csv(args.scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    g = per_gene(d, mspec["col"])
    alphas = [a for a in ALPHAS if f"add_a{a}" in set(g.condition)]
    controls = [f"random_a{a}" for a in CONTROL_ALPHAS if f"random_a{a}" in set(g.condition)]
    arms = [f"add_a{a}" for a in alphas]
    g_arms = g[g.condition.isin(arms + controls)].copy()

    g_arms.to_csv(args.out_dir / "per_gene_normalized.csv", index=False)
    tab = summarise(g_arms[g_arms.condition.isin(arms)])
    tab.to_csv(args.out_dir / "normalized_by_stratum_alpha.csv", index=False)
    widen(tab).to_csv(args.out_dir / "normalized_by_stratum_alpha_wide.csv", index=False)
    ctl = summarise(g_arms[g_arms.condition.isin(controls)])
    ctl.to_csv(args.out_dir / "normalized_random_null_control.csv", index=False)
    trends = trend_table(g_arms, arms + controls)
    trends.to_csv(args.out_dir / "conservation_trends_raw_vs_normalized.csv", index=False)

    dose_figure(g_arms, alphas, args.out_dir, "normalized_dose_by_stratum")
    comparison_figure(g_arms, alphas, args.out_dir, "raw_vs_normalized_by_stratum")

    # ---- undefined-case accounting -------------------------------------------------------------
    u = d[d.condition == "unsteered"]
    undef = pd.DataFrame(
        [
            {
                "quantity": "private recovery",
                "undefined_when": "U == 100 (no room left toward 100%)",
                "n_genes": int((u[mspec["col"]] == 100).sum()),
                "n_of": len(u),
            },
            {
                "quantity": "amino-acid identity",
                "undefined_when": "U == 0 (no unsteered identity to lose)",
                "n_genes": int((u[AA_COL] == 0).sum()),
                "n_of": len(u),
            },
        ]
    )
    undef.to_csv(args.out_dir / "undefined_cases.csv", index=False)
    print(undef.to_string(index=False))
    print(tab[tab.stratum == "all"].to_string(index=False))
    print(trends.to_string(index=False))


if __name__ == "__main__":
    main()
