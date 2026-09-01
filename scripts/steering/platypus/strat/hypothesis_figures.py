"""Hypothesis figures for a conservation-stratified platypus run. Nine figures, each tied to a
pre-registered hypothesis rather than to a pipeline stage:
"""

from __future__ import annotations
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "steering" / "platypus" / "strat"))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402
from stage5_merge import CONFIRM_ROLE, CONFIRMATORY, loo_pc1_cos, resid  # noqa: E402
from strat_metric import add_metric_args, coverage_note, load_scores  # noqa: E402

import figure_data  # noqa: E402

# Identity is never carried by colour alone: every categorical series also has a marker, a
# position, or a direct label.
INK = acs.SERIES_PRIMARY  # observed values
WARM = acs.SERIES_NULL  # nulls, controls, the "other" arm
PLUM = acs.SERIES_THIRD  # a third series, never placed adjacent to INK in a legend
TEAL = acs.SERIES_FOURTH
GREY = acs.SERIES_MUTED

# Diverging: two hues through a neutral grey midpoint, never a rainbow, never a hue at zero.
DIVERGE = LinearSegmentedColormap.from_list("warm_ink", [WARM, acs.apc.dawn, acs.apc.gray, INK])
SEQ = LinearSegmentedColormap.from_list("ink_seq", [acs.apc.white, INK])

STRATA = ["s0\nfastest", "s1", "s2", "s3", "s4\nconserved"]
OUTCOME_LABEL = {
    "loo_cos": "orientation  cos(D_i, v_-i)",
    "loo_cos_pc1": "orientation, PC1 removed",
    "delta_norm": "magnitude  ||D_i||",
}


# The two panels the publication uses. In pub mode only these are written; the rest are diagnostic
# grids never sized for a fixed page width.
PUB_STEMS = {"5_rate_outcome_matrix", "8b_strata_composition_frame"}

PUB_MATRIX_H = 800.0  # figure 11: 15 predictor rows plus the column labels

# Row-group titles for the publication matrix — the diagnostic two-line titles are taller than the
# groups they label once rotated at 15 pt. The definitions belong in the caption.
PUB_GROUP_LABELS = {
    "protein change\n(dN)": "dN",
    "neutral change\n(dS) — control": "dS (control)",
    "selection\n(ω = dN/dS)": "selection",
    "total divergence": "divergence",
    # Kept short deliberately: a rotated title is bounded by the HEIGHT of its group, and at
    # 15 pt "lineage rate" overflows a two-row group and lands on the one-row groups below it.
    "lineage-specific\nrate": "residual",
    "shape": "shape",
    "frame": "frame",
}

# Outcome columns as a reader sees them; the defaults are pipeline column names.
PUB_OUTCOME_LABELS = {
    "||D_i||": "||Δ||",
    "loo_cos": "LOO cosine",
    "loo_cos PC1-rm": "LOO cosine, PC1 removed",
    "gain α=1": "Gain, alpha = 1",
    "gain α_i": "Gain, own alpha",
}

# Predictor rows as a reader sees them: the index values are column names whose suffixes encode the
# estimator and the comparison, neither decodable off the page.
PUB_PREDICTOR_LABELS = {
    "dN_hp_yn": "dN, human–platypus",
    "dN_background_yn": "dN, background",
    "dS_hp_yn": "dS, human–platypus",
    "dS_background_yn": "dS, background",
    "omega_hp_yn": "ω, human–platypus",
    "omega_background_yn": "ω, background",
    "tree_len": "Tree length",
    "diameter": "Tree diameter",
    "focal_target_dist": "Focal-to-target distance",
    "mean_target_dist": "Mean target distance",
    "background_rate": "Background rate",
    "focal_residual": "Focal residual",
    "human_residual": "Human residual",
    "treeness": "Treeness",
    "perc_id_hp": "Human–platypus protein identity",
}


def save(fig, out: Path, stem: str) -> None:
    if pub.is_on():
        if stem in PUB_STEMS:
            pub.finish(fig, stem, directory=out)
        plt.close(fig)
        return
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}")


def pstr(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return f"p={p:.2g}" if p >= 1e-4 else f"p={p:.0e}"


# Set from --from-figure-data in main(); the loaders below are called from several figures.
FROM_FIGURE_DATA = False


def _read(run_path: Path, table: str) -> pd.DataFrame:
    """One table, from figure_data/ when running the publication path, else the run directory."""
    return figure_data.table(table) if FROM_FIGURE_DATA else pd.read_csv(run_path)


# shared per-gene table: the same joins stage5_merge makes, so the figures cannot drift from it
def per_gene_table(run: Path, layer: int, mode: str, with_pc1: bool = True) -> pd.DataFrame:
    """`with_pc1=False` skips the 754 MB npz load and the 400 leave-one-out PCAs behind
    `loo_cos_pc1` -- worth ~12 min when the requested figures do not need that column."""
    pairs = _read(run / "stage1" / "pairs.csv", "exp3_panel")
    tre = _read(run / "stage5" / "tree_stats.csv", "exp3_rates_tree_stats")
    cov = _read(run / "stage2" / "aligned_coverage.csv", "exp3_panel_coverage")
    df = (
        pairs[["gene", "stratum", "perc_id_hp", "cds_len_human"]]
        .merge(tre[tre.status == "ok"], on="gene", how="inner")
        .merge(cov[["gene", "retained_frac"]], on="gene", how="left")
    )
    df["focal_residual"] = resid(
        df.platypus_branch.to_numpy(float), df.background_rate.to_numpy(float)
    )
    df["human_residual"] = resid(
        df.human_branch.to_numpy(float), df.background_rate.to_numpy(float)
    )
    dn = _read(run / "stage5" / "dnds.csv", "exp3_rates_dnds")
    dn = dn[dn.status == "ok"]
    keep = [c for c in dn.columns if c.startswith(("dN_", "dS_", "omega_", "tree_dS", "tree_dN"))]
    df = df.merge(dn[["gene", *keep]], on="gene", how="left")

    if FROM_FIGURE_DATA:
        pg = figure_data.table("exp3_direction_per_gene_by_layer").query("panel == 'strat400'")
    else:
        pg = pd.read_csv(run / f"geom_{mode}" / "per_gene_by_layer.csv")
    pg = pg[pg.layer == layer][["gene", "loo_cos", "delta_norm"]]
    df = df.merge(pg, on="gene", how="left")

    if with_pc1:
        # `loo_cos_pc1` costs ~12 min and is a pure function of the frozen stage-2 array, so it is
        # cached. Delete the file to force a recompute.
        cache = run / f"geom_{mode}" / f"loo_cos_pc1_L{layer}.csv"
        if cache.exists():
            pc1 = pd.read_csv(cache)
        else:
            npz = np.load(run / "stage2" / "pooled_representations.npz", allow_pickle=True)
            a = npz[mode]
            D = (a[:, 1, layer, :] - a[:, 0, layer, :]).astype(np.float64)
            pc1 = pd.DataFrame(
                {"gene": [str(g) for g in npz["genes"]], "loo_cos_pc1": loo_pc1_cos(D)}
            )
            pc1.to_csv(cache, index=False)
            print(f"  cached {cache.name}")
        df = df.merge(pc1, on="gene", how="left")
    return df


def gain_table(s: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Per-gene gain in pp and in logits, plus the meta-regression weight."""
    g = (
        s.groupby(["gene", "condition"])
        .agg(p=("metric", "mean"), n_sites=(spec["n_col"], "mean"), n_samples=("n_samples", "mean"))
        .reset_index()
    )
    w = g.pivot(index="gene", columns="condition", values="p") / 100.0
    trials = (g.groupby("gene").n_sites.first() * g.groupby("gene").n_samples.first()) / 2.16
    out = {}
    for cond in ("add_a1.0", "add_own"):
        pu, ps = w["unsteered"].clip(1e-4, 1 - 1e-4), w[cond].clip(1e-4, 1 - 1e-4)
        var = (1 / (trials * ps * (1 - ps))) + (1 / (trials * pu * (1 - pu)))
        out[cond] = pd.DataFrame(
            {
                "gene": w.index,
                "gain_pp": (w[cond] - w["unsteered"]).values * 100,
                "gain_logit": (np.log(ps / (1 - ps)) - np.log(pu / (1 - pu))).values,
                "weight": (1 / var).values,
                "unsteered_pp": w["unsteered"].values * 100,
            }
        )
    return out


def fig1_qc(run: Path, out: Path) -> None:
    summ = json.loads((run / "stage1" / "stage1_summary.json").read_text())
    att = pd.read_csv(run / "stage1" / "attrition.csv")
    ex = [summ["examined_by_stratum"][str(i)] for i in range(5)]
    kept = [summ["per_stratum"][str(i)] for i in range(5)]
    rate = [summ["pass_rate_by_stratum"][str(i)] * 100 for i in range(5)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    x = np.arange(5)
    axes[0].bar(x - 0.19, ex, 0.36, color=acs.apc.denim, label="blocks examined")
    axes[0].bar(x + 0.19, kept, 0.36, color=INK, label="kept")
    for i, r in enumerate(rate):
        axes[0].text(i, ex[i] + 5, f"{r:.0f}%", ha="center", fontsize=7, color=GREY)
    axes[0].set_ylim(0, max(ex) * 1.22)
    axes[0].set_xticks(x, STRATA)
    axes[0].set_ylabel("blocks")
    axes[0].set_title("QC attrition by stratum  (pass rate above bars)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")

    # `reason` carries per-gene detail in parentheses -- window_starts_too_late(h49>30) -- so the
    # bars are built on the reason CLASS, and the 400 `pass` rows are not a rejection.
    rej = att[att.reason != "pass"].copy()
    rej["cls"] = rej.reason.str.replace(r"\(.*\)", "", regex=True)
    tab = pd.crosstab(rej.stratum, rej.cls)
    tab = tab[tab.sum().sort_values(ascending=False).index].reindex(range(5), fill_value=0)
    bottom = np.zeros(len(tab))
    for j, c in enumerate(tab.columns):
        axes[1].bar(
            tab.index,
            tab[c],
            0.62,
            bottom=bottom,
            color=[INK, WARM, PLUM][j % 3],
            edgecolor=acs.apc.white,
            lw=1.0,
            label=f"{c}  (n={int(tab[c].sum())})",
        )
        bottom += tab[c].values
    axes[1].set_xticks(range(5), STRATA)
    axes[1].set_ylabel("rejected blocks")
    axes[1].set_title("Rejection reason by stratum")
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.suptitle(
        f"Stage 1 QC — {summ['n_kept']} kept of {summ['n_examined']} examined, "
        f"--max-start-codon {summ['max_start_codon']}",
        y=1.03,
    )
    fig.tight_layout()
    save(fig, out, "1_qc_attrition")


def _shape_panel(ax, x, y, strata, fit, xlab, ylab, title, rho=None):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, strata = x[ok], y[ok], strata[ok]
    z = (x - x.mean()) / x.std()
    ax.scatter(z, y, s=7, color=INK, alpha=0.42, lw=0)
    if fit is not None and np.isfinite(fit["b_linear"]):
        zz = np.linspace(z.min(), z.max(), 100)
        ax.plot(
            zz,
            y.mean() + fit["b_linear"] * zz + fit["b_quad"] * zz**2,
            color=WARM,
            lw=2.0,
            zorder=3,
            label="linear + quadratic",
        )
    mid, ext = np.isin(strata, [2, 3]), np.isin(strata, [0, 4])
    ax.axhline(y[mid].mean(), color=PLUM, ls="--", lw=1.2, zorder=2)
    ax.axhline(y[ext].mean(), color=GREY, ls=":", lw=1.2, zorder=2)
    if fit is not None:
        # `lin`/`quad` are SLOPES per SD of predictor, not correlations. The Spearman is printed
        # beside them so the panel is not misread against the rho matrix.
        rtxt = f"   ρ {rho:+.3f}" if rho is not None and np.isfinite(rho) else ""
        ax.set_title(
            f"{title}\nslope/SD {fit['b_linear']:+.3f} {pstr(fit['p_linear'])}{rtxt}\n"
            f"quad {fit['b_quad']:+.3f} {pstr(fit['p_quad'])}   "
            f"mid−ext {fit['mid_minus_ext']:+.3f} {pstr(fit['p_mid_ext'])}",
            fontsize=6.5,
        )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def fig2_h1ab(run: Path, out: Path, df: pd.DataFrame, layer: int, mode: str) -> None:
    sh = pd.read_csv(run / "stage5" / "rate_vs_direction_shape.csv")
    rhos = (
        pd.read_csv(run / "stage5" / "rate_vs_direction.csv")
        .set_index(["predictor", "outcome"])
        .rho
    )
    outcomes = ["loo_cos", "loo_cos_pc1", "delta_norm"]
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 9.6))
    for r, outc in enumerate(outcomes):
        for c, pred in enumerate(CONFIRMATORY):
            row = sh[(sh.predictor == pred) & (sh.outcome == outc)]
            fit = None
            if len(row):
                q = row.iloc[0]
                fit = {
                    "b_linear": q.b_linear,
                    "p_linear": q.p_linear,
                    "b_quad": q.b_quad,
                    "p_quad": q.p_quad,
                    "mid_minus_ext": q.mid_minus_ext,
                    "p_mid_ext": q.p_mid_ext,
                }
            neg = "NEGATIVE CONTROL" in CONFIRM_ROLE[pred]
            _shape_panel(
                axes[r, c],
                df[pred].to_numpy(float),
                df[outc].to_numpy(float),
                df.stratum.to_numpy(int),
                fit,
                f"z({pred})" + ("   [neg. control]" if neg else ""),
                OUTCOME_LABEL[outc] if c == 0 else "",
                pred,
                rho=rhos.get((pred, outc), np.nan),
            )
            if neg:
                for s in axes[r, c].spines.values():
                    s.set_color(WARM)
                    s.set_linewidth(1.4)
    handles = [
        Line2D(
            [],
            [],
            color=WARM,
            lw=2.0,
            label="pre-registered linear + quadratic fit (slope is in OUTCOME units per SD "
            "of predictor, not a correlation — ρ is printed beside it)",
        ),
        Line2D([], [], color=PLUM, ls="--", lw=1.2, label="mean, mid strata (2,3)"),
        Line2D([], [], color=GREY, ls=":", lw=1.2, label="mean, extreme strata (0,4)"),
        Line2D(
            [],
            [],
            ls="",
            label="mid−ext is a property of the OUTCOME, so it repeats "
            "across the four predictors in a row",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"H1a orientation / H1b magnitude vs the four confirmatory rate predictors — "
        f"{mode}, block {layer}, n = {len(df)} genes, one point per gene",
        y=1.005,
    )
    fig.tight_layout()
    save(fig, out, "2_h1ab_geometry_vs_rate")


def fig3_h2a(run: Path, out: Path, df: pd.DataFrame, gains: dict) -> None:
    conf = pd.read_csv(run / "stage5" / "rate_vs_gain_confirmatory.csv")
    b0 = {"add_a1.0": 0.1236, "add_own": 0.1465}  # weighted mean logit gain, logs/06b_gain.log
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.6))
    for r, cond in enumerate(("add_a1.0", "add_own")):
        m = df.merge(gains[cond], on="gene", how="inner")
        for c, pred in enumerate(CONFIRMATORY):
            ax = axes[r, c]
            row = conf[(conf.condition == cond) & (conf.predictor == pred)]
            x = m[pred].to_numpy(float)
            y = m.gain_logit.to_numpy(float)
            w = m.weight.to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
            x, y, w = x[ok], y[ok], w[ok]
            z = (x - x.mean()) / x.std()
            ax.axhline(0, color=GREY, ls=":", lw=1.0, zorder=0)
            ax.scatter(z, y, s=2 + 22 * w / w.max(), color=INK, alpha=0.35, lw=0)
            if len(row):
                q = row.iloc[0]
                zz = np.linspace(z.min(), z.max(), 100)
                ax.plot(
                    zz,
                    b0[cond] + q.wls_beta_per_sd * zz + q.wls_quad_per_sd2 * zz**2,
                    color=WARM,
                    lw=2.0,
                    zorder=3,
                )
                # beta is in LOGIT-gain units per SD of predictor, not a correlation; the
                # Spearman is printed beside it for comparability with figure 5.
                ax.set_title(
                    f"{pred}\nslope/SD {q.wls_beta_per_sd:+.4f} {pstr(q.wls_p)}   "
                    f"ρ {q.spearman_rho:+.3f}\n"
                    f"quad {q.wls_quad_per_sd2:+.4f} {pstr(q.wls_quad_p)}   "
                    f"Holm {q.p_holm:.3g}",
                    fontsize=6.5,
                )
            if "NEGATIVE CONTROL" in CONFIRM_ROLE[pred]:
                for s in ax.spines.values():
                    s.set_color(WARM)
                    s.set_linewidth(1.4)
            ax.set_xlabel(f"z({pred})")
            if c == 0:
                ax.set_ylabel(f"{cond}\nsteering gain (logit)")
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color=WARM,
                lw=2.0,
                label="inverse-variance weighted linear + quadratic (the reported model); "
                "slope is logit-gain per SD, not a correlation",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                color=INK,
                alpha=0.5,
                label="one gene; marker area ∝ meta-regression weight",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        "H2a — steering gain vs the same four confirmatory predictors "
        "(same axes as figure 2, so geometry and outcome can be read against each other)",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, out, "3_h2a_gain_vs_rate")


def fig4_h1c(run: Path, out: Path) -> None:
    ex = pd.read_csv(run / "h1c_clusters" / "existence.csv")
    summ = json.loads((run / "h1c_clusters" / "h1c_summary.json").read_text())
    asg = pd.read_csv(run / "h1c_clusters" / "assignments.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    a = axes[0]
    a.fill_between(
        ex.k,
        ex.sil_null_mean - 2 * ex.sil_null_sd,
        ex.sil_null_mean + 2 * ex.sil_null_sd,
        color=WARM,
        alpha=0.18,
        lw=0,
    )
    a.plot(ex.k, ex.sil_null_mean, color=WARM, ls="--", lw=1.4, label="spectrum-matched null ±2 sd")
    a.plot(ex.k, ex.sil_isotropic_mean, color=GREY, ls=":", lw=1.4, label="isotropic null")
    a.plot(ex.k, ex.silhouette, color=INK, lw=2.0, marker="o", ms=4, label="observed")
    ks = int(summ["k_selected"])
    a.scatter(
        [ks],
        [ex.loc[ex.k == ks, "silhouette"].item()],
        s=90,
        facecolor="none",
        edgecolor=INK,
        lw=1.6,
        zorder=4,
    )
    for _, q in ex.iterrows():
        if q.sil_p_at_k < 0.05:
            a.text(q.k, q.silhouette + 0.008, "*", ha="center", color=INK, fontsize=11)
    a.set_xlabel("k")
    a.set_ylabel("silhouette")
    a.set_title(
        f"H1c part 1 existence — PASS\nexcess {summ['k_selected'] and ex.sil_excess.max():+.3f} "
        f"at k*={ks}, global p={summ['p_global']:.4f}",
        fontsize=8,
    )
    a.legend(fontsize=6, frameon=False, loc="upper right")

    b = axes[1]
    st = summ["stability"]
    vals = [st["bootstrap_ari_median"], st["bootstrap_ari_q05"], st["half_ari_median"]]
    labs = ["block-bootstrap\nARI median", "bootstrap\nARI q05", "disjoint-half\nARI"]
    b.bar(labs, vals, 0.55, color=[INK, acs.apc.denim, INK])
    b.axhline(0.6, color=WARM, ls="--", lw=1.4)
    b.text(
        2.45,
        0.605,
        "a stable partition\nsits here",
        ha="right",
        va="bottom",
        fontsize=6,
        color=WARM,
    )
    for i, v in enumerate(vals):
        b.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7, color=GREY)
    b.set_ylim(0, 0.8)
    b.set_ylabel("adjusted Rand index")
    b.set_title(f"H1c part 2 stability — FAIL at k={st['k']}\n→ Tier C not licensed", fontsize=8)

    c = axes[2]
    tab = pd.crosstab(asg.cluster, asg.stratum)
    im = c.imshow(tab.values, cmap=SEQ, aspect="auto")
    c.set_xticks(range(5), [s.split("\n")[0] for s in STRATA])
    c.set_yticks(
        range(len(tab)),
        [f"c{i} (n={int(n)})" for i, n in zip(tab.index, tab.sum(axis=1), strict=False)],
        fontsize=7,
    )
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            c.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                fontsize=7,
                color=acs.apc.white if v > tab.values.max() * 0.55 else INK,
            )
    c.set_xlabel("stratum")
    c.set_title("H1c part 3 membership (exploratory)\nclusters are conservation bands", fontsize=8)
    fig.colorbar(im, ax=c, fraction=0.035, pad=0.02, label="genes")
    fig.tight_layout()
    save(fig, out, "4_h1c_clusters")


def fig5_matrix(run: Path, out: Path) -> None:
    d = _read(run / "stage5" / "rate_vs_direction.csv", "exp3_rates_vs_direction")
    g = _read(run / "stage5" / "rate_vs_gain.csv", "exp3_rates_vs_gain")
    geo = d.pivot_table(index="predictor", columns="outcome", values="rho")
    geo = geo[["delta_norm", "loo_cos", "loo_cos_pc1"]]
    gai = g.pivot_table(index="predictor", columns="condition", values="spearman_rho")
    M = geo.join(gai, how="outer")
    M.columns = ["||D_i||", "loo_cos", "loo_cos PC1-rm", "gain α=1", "gain α_i"]

    # The PAML codon-model flavours are dropped from the display: under the one-ratio model
    # tree_dN = omega x tree_dS identically, so they are views of one fit, while the yn00 pairwise
    # estimates can genuinely disagree — which is what the dS control needs. They stay in the CSVs,
    # and the Bonferroni thresholds are still computed over the full grid.
    DROP = {"omega_m0", "omega_plat_m2", "omega_bg_m2", "tree_dS_m0"}
    M = M.drop(index=[p for p in M.index if p in DROP])

    # Grouped by what each statistic measures, not by pipeline origin. dN sits above dS on purpose:
    # same divergence split by whether the protein changed, and their contrast is the control.
    BY_MEANING = [
        ("protein change\n(dN)", ["dN_hp_yn", "dN_background_yn"]),
        ("neutral change\n(dS) — control", ["dS_hp_yn", "dS_background_yn"]),
        ("selection\n(ω = dN/dS)", ["omega_hp_yn", "omega_background_yn"]),
        (
            "total divergence",
            ["tree_len", "diameter", "focal_target_dist", "mean_target_dist", "background_rate"],
        ),
        ("lineage-specific\nrate", ["focal_residual", "human_residual"]),
        ("shape", ["treeness"]),
        ("frame", ["perc_id_hp"]),
    ]
    order, labels, groups = [], [], []
    for title, preds in BY_MEANING:
        preds = [p for p in preds if p in M.index]
        if not preds:
            continue
        groups.append((len(order), len(preds), title))
        order += preds
        labels += preds
    missing = [p for p in M.index if p not in order]
    if missing:  # a new predictor appeared upstream; never drop it silently
        groups.append((len(order), len(missing), "unclassified"))
        order += missing
        labels += missing
    M = M.loc[order]

    # Bonferroni markers, on each grid's own test count
    thr_geo = 0.05 / len(d[d.role != "frame"])
    thr_gain = 0.05 / len(g[g.role != "frame"])
    sig = pd.DataFrame(False, index=M.index, columns=M.columns)
    for _, q in d.iterrows():
        col = {"delta_norm": "||D_i||", "loo_cos": "loo_cos", "loo_cos_pc1": "loo_cos PC1-rm"}[
            q.outcome
        ]
        if q.predictor in sig.index:
            sig.loc[q.predictor, col] = q.p < thr_geo
    for _, q in g.iterrows():
        col = {"add_a1.0": "gain α=1", "add_own": "gain α_i"}[q.condition]
        if q.predictor in sig.index:
            sig.loc[q.predictor, col] = q.spearman_p < thr_gain

    if pub.is_on():
        labels = [PUB_PREDICTOR_LABELS.get(p, p.replace("_", " ")) for p in order]
        M.columns = [PUB_OUTCOME_LABELS.get(c, c) for c in M.columns]
        sig.columns = M.columns
        groups = [(a, b, PUB_GROUP_LABELS.get(t, t.replace("\n", " "))) for a, b, t in groups]
    fig, ax = plt.subplots(
        figsize=(pub.size(pub.FULL, PUB_MATRIX_H) if pub.is_on() else (7.6, 8.0))
    )
    v = np.nanmax(np.abs(M.values))
    im = ax.imshow(M.values, cmap=DIVERGE, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=30, ha="right")
    ax.set_yticks(range(M.shape[0]), labels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M.values[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M.values[i, j]:+.2f}" + ("*" if sig.values[i, j] else ""),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=acs.apc.white if abs(M.values[i, j]) > 0.6 * v else acs.apc.black,
                )
    trans = ax.get_yaxis_transform()
    for start, n, title in groups:
        ax.axhline(start - 0.5, color=acs.apc.white, lw=3)
        # A rotated title is bounded by its group's height, so at 15 pt adjacent one-row groups
        # collide. Those rows are single predictors their own label already names.
        if not pub.is_on() or n > 1:
            ax.text(
                -0.42,
                start + n / 2 - 0.5,
                title,
                rotation=90,
                va="center",
                ha="center",
                fontsize=6.6,
                color=GREY,
                transform=trans,
                clip_on=False,
            )
        ax.plot(
            [-0.335, -0.335],
            [start - 0.35, start + n - 0.65],
            color=acs.GRID,
            lw=1.4,
            transform=trans,
            clip_on=False,
        )
    if pub.is_on():
        # No chart title — the caption carries it. The Bonferroni convention has to stay ON the
        # artwork though: the asterisks are unreadable without it, and they are the figure's point.
        ax.set_title("* survives Bonferroni within its grid")
    else:
        ax.set_title(
            "Rate predictor × outcome, Spearman ρ\n* = survives Bonferroni within its grid",
            fontsize=9,
        )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ρ")
    if pub.is_on():
        pub.tight(fig)
    else:
        fig.tight_layout()
    save(fig, out, "5_rate_outcome_matrix")


def fig6_arms(
    run: Path, out: Path, arm_dir: str, gains: dict, df: pd.DataFrame, spec: dict
) -> None:
    s = pd.read_csv(run / arm_dir / "analysis_summary.csv")
    prim = spec["col"]
    if f"{prim}_ci" not in s.columns:
        sys.exit(
            f"analysis_summary.csv has no `{prim}_ci` -- it was written for a different site "
            f"set. Rebuild it with:\n\n    uv run python "
            f"scripts/steering/platypus/strat/stage4_analysis.py \\\n"
            f"      --run {run} --dir {arm_dir} --metric {spec['report']['metric']}\n"
        )
    s["lo"] = s[f"{prim}_ci"].str.strip("[]").str.split(",").str[0].astype(float)
    s["hi"] = s[f"{prim}_ci"].str.strip("[]").str.split(",").str[1].astype(float)
    blocks = [
        ("primary", ["add_a1.0", "add_own", "random_a1.0"]),
        ("dose", ["add_a0.5", "add_a2.0", "random_a0.5", "random_a2.0"]),
        ("confounds", ["add_cone_removed_a1.0", "add_gc_removed_a1.0", "cross_gene_a1.0"]),
        ("H2c", ["panel_same_k5", "panel_other_k5"]),
    ]

    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, :2])
    ypos, ylabs, colors = [], [], []
    y = 0
    for title, conds in blocks:
        top = y
        for c in conds:
            q = s[s.condition == c]
            if not len(q):
                continue
            q = q.iloc[0]
            col = WARM if c.startswith("random") else (PLUM if "gc_removed" in c else INK)
            ax.plot([q.lo, q.hi], [y, y], color=col, lw=2.0, solid_capstyle="round")
            ax.plot([q[f"{prim}_delta"]], [y], "o", color=col, ms=6)
            ypos.append(y)
            ylabs.append(c)
            colors.append(col)
            y -= 1
        # group label in the left margin, in axes-fraction x so it cannot collide with the ticks
        trans = ax.get_yaxis_transform()
        ax.text(
            -0.30,
            (top + y + 1) / 2,
            title,
            fontsize=7,
            color=GREY,
            ha="center",
            va="center",
            rotation=90,
            transform=trans,
            clip_on=False,
        )
        ax.plot(
            [-0.255, -0.255],
            [top + 0.32, y + 0.68],
            color=acs.GRID,
            lw=1.4,
            transform=trans,
            clip_on=False,
        )
        y -= 1.0
    ax.axvline(0, color=GREY, ls=":", lw=1.0)
    ax.set_yticks(ypos, ylabs, fontsize=7.5)
    for t, c in zip(ax.get_yticklabels(), colors, strict=False):
        t.set_color(c)
    ax.set_xlabel(f"{spec['axis_delta'][:-5]} vs the same gene's unsteered cell (pp)")
    ax.set_title(
        "Every stage-4 condition, bootstrap CI over genes  (n = 398).  "
        "H2c is exploratory — its gate did not pass",
        fontsize=9,
    )

    ax = fig.add_subplot(gs[0, 2])
    alphas = [0.5, 1.0, 2.0]
    add = [s.loc[s.condition == f"add_a{a}", f"{prim}_delta"].item() for a in alphas]
    rnd = [s.loc[s.condition == f"random_a{a}", f"{prim}_delta"].item() for a in alphas]
    aa = [s.loc[s.condition == f"add_a{a}", "aa_id_to_target_delta"].item() for a in alphas]
    ax.plot(alphas, add, color=INK, lw=2.0, marker="o", ms=5, label=spec["delta_label"])
    ax.plot(alphas, rnd, color=WARM, lw=2.0, ls="--", marker="s", ms=5, label="norm-matched random")
    ax.plot(alphas, aa, color=PLUM, lw=2.0, marker="^", ms=5, label="amino-acid identity cost")
    ax.axhline(0, color=GREY, ls=":", lw=1.0)
    ax.set_xticks(alphas)
    ax.set_xlabel("α")
    ax.set_ylabel("Δ vs unsteered (pp)")
    ax.set_title("Dose–response and its protein cost", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    m = df.merge(gains["add_a1.0"], on="gene", how="inner")
    by = m.groupby("stratum")
    ax.bar(range(5), by.gain_pp.mean(), 0.6, color=INK, label="gain, add α=1")
    ax.errorbar(range(5), by.gain_pp.mean(), yerr=by.gain_pp.sem(), fmt="none", ecolor=GREY, lw=1)
    ax.set_xticks(range(5), STRATA, fontsize=7)
    ax.set_ylabel(spec["axis_delta"])
    ax.set_title("Gain by stratum", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(5), by.unsteered_pp.mean(), 0.6, color=acs.apc.denim)
    ax.errorbar(
        range(5), by.unsteered_pp.mean(), yerr=by.unsteered_pp.sem(), fmt="none", ecolor=GREY, lw=1
    )
    ax.set_xticks(range(5), STRATA, fontsize=7)
    # Let the y-axis accommodate each site set's baseline.
    ax.set_ylabel(f"unsteered {spec['axis_level']}")
    ax.set_title("…and the baseline it is measured against", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    # Mask non-finite pairs before fitting: a gene with no evidence has NaN baseline and gain, and
    # polyfit fails outright rather than skipping them.
    fit_ok = np.isfinite(m.unsteered_pp) & np.isfinite(m.gain_pp)
    ax.scatter(m.unsteered_pp[fit_ok], m.gain_pp[fit_ok], s=8, color=INK, alpha=0.4, lw=0)
    b, a = np.polyfit(m.unsteered_pp[fit_ok], m.gain_pp[fit_ok], 1)
    xx = np.linspace(m.unsteered_pp[fit_ok].min(), m.unsteered_pp[fit_ok].max(), 50)
    ax.plot(xx, a + b * xx, color=WARM, lw=2.0)
    ax.axhline(0, color=GREY, ls=":", lw=1.0)
    ax.set_xlabel(f"unsteered {spec['axis_level']}")
    ax.set_ylabel(spec["axis_delta"])
    ax.set_title("Headroom: genes with a lower baseline gain more", fontsize=9)

    fig.suptitle(
        "G2 — the causal arms, the dose–response, and the headroom that explains the "
        "conservation gradient",
        y=0.98,
    )
    save(fig, out, "6_stage4_arms")


def _scatter_rho(ax, x, y, xlab, ylab, title, resid_on=None, color=INK):
    """One gene per point, least-squares line, Spearman rho + p in the corner."""
    ok = np.isfinite(x) & np.isfinite(y)
    if resid_on is not None:
        ok &= np.isfinite(resid_on)
    x, y = x[ok], y[ok]
    if resid_on is not None:
        z = stats.rankdata(resid_on[ok])
        x, y = stats.rankdata(x), stats.rankdata(y)
        x = x - np.polyval(np.polyfit(z, x, 1), z)
        y = y - np.polyval(np.polyfit(z, y, 1), z)
        rho = np.corrcoef(x, y)[0, 1]
        n = len(x)
        t = rho * np.sqrt((n - 3) / (1 - rho**2))
        p = 2 * stats.t.sf(abs(t), n - 3)
    else:
        rho, p = stats.spearmanr(x, y)
    ax.scatter(x, y, s=8, color=color, alpha=0.38, lw=0)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color=WARM, lw=2.0, zorder=3)
    ax.text(
        0.035,
        0.955,
        f"ρ = {rho:+.3f}\n{pstr(p)}   n = {len(x)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.28", fc=acs.apc.white, ec=acs.GRID, lw=0.8),
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8.5)


def fig7_conservation(
    run: Path, out: Path, df: pd.DataFrame, gains: dict, scores: pd.DataFrame, spec: dict
) -> None:
    """Everything against the conservation axis itself, with a line and a rho on each panel."""
    m = df.merge(gains["add_a1.0"], on="gene", how="inner").merge(
        gains["add_own"][["gene", "gain_pp"]].rename(columns={"gain_pp": "gain_pp_own"}),
        on="gene",
        how="left",
    )
    aa = scores.groupby(["gene", "condition"]).aa_id_to_target.mean().unstack()
    m = m.merge(
        (aa["add_a1.0"] - aa["unsteered"]).rename("d_aa").reset_index(), on="gene", how="left"
    )
    c = m.perc_id_hp.to_numpy(float)
    XL = "human↔platypus protein identity (%)"

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.2))
    _scatter_rho(axes[0, 0], c, m.loo_cos.to_numpy(float), XL, "cos(D_i, v_-i)", "H1a orientation")
    _scatter_rho(
        axes[0, 1],
        c,
        m.loo_cos_pc1.to_numpy(float),
        XL,
        "cos(D_i, v_-i), PC1 removed",
        "H1a orientation, PC1 removed",
    )
    _scatter_rho(axes[0, 2], c, m.delta_norm.to_numpy(float), XL, "||D_i||", "H1b magnitude")
    _scatter_rho(
        axes[0, 3],
        c,
        m.unsteered_pp.to_numpy(float),
        XL,
        f"unsteered {spec['axis_level']}",
        "the READOUT's own baseline",
        color=PLUM,
    )
    _scatter_rho(
        axes[1, 0],
        c,
        m.gain_pp.to_numpy(float),
        XL,
        f"{spec['axis_delta'][:-5]}, add α=1 (pp)",
        "G2 steering gain, α = 1",
    )
    _scatter_rho(
        axes[1, 1],
        c,
        m.gain_pp_own.to_numpy(float),
        XL,
        f"{spec['axis_delta'][:-5]}, add α_i (pp)",
        "G2 steering gain, own-shift α",
    )
    _scatter_rho(
        axes[1, 2],
        c,
        m.d_aa.to_numpy(float),
        XL,
        "Δ amino-acid identity (pp)",
        "protein cost of steering",
    )
    _scatter_rho(
        axes[1, 3],
        c,
        m.gain_pp.to_numpy(float),
        "conservation (rank residual)",
        f"{spec['axis_delta'][:-5]}, α=1 (rank residual)",
        "gain vs conservation, AFTER the baseline",
        resid_on=m.unsteered_pp.to_numpy(float),
        color=WARM,
    )
    for ax in axes[1, :3]:
        ax.axhline(0, color=GREY, ls=":", lw=1.0, zorder=0)
    axes[1, 3].axhline(0, color=GREY, ls=":", lw=1.0, zorder=0)
    fig.suptitle(
        "Geometry and steering against the conservation axis — one point per gene, "
        "least-squares line, Spearman ρ.  `perc_id_hp` is the sampling frame, so these "
        "are descriptive; the confirmatory tests use the rate statistics of figures 2–3",
        y=1.005,
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, out, "7_conservation_scatter")


def autapomorphy_stats(scores: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Per-gene statistics of the READOUT ITSELF, for the composition figure."""
    u = scores[scores.condition == "unsteered"]
    agg = {"n_sites": (spec["n_col"], "mean"), "unsteered": ("metric", "mean")}
    if "n_diagnostic_scorable" in scores.columns:
        agg["n_diag"] = ("n_diagnostic_scorable", "mean")
    if "n_voting_species" in scores.columns:
        agg["voters"] = ("n_voting_species", "first")
    g = u.groupby("gene").agg(**agg).reset_index()
    # ONE definition of "no evidence", matching strat_metric: a NaN metric. Such a gene also carries
    # a meaningless 0 in the site-count column, which would otherwise plot as a real measurement of
    # zero sites and drag every box down; blank all four readouts together so the panels and the
    # per-panel miss counts agree with each other and with the caption.
    blank = g.unsteered.isna()
    g.loc[blank, "n_sites"] = np.nan
    if "voters" in g.columns:
        g.loc[blank, "voters"] = np.nan
    if "n_diag" in g.columns:
        # Guard the ratio: a gene with no diagnostic site has no strictness to report, and 0/0 would
        # otherwise land on the plot as a real 0%.
        g["strict_frac"] = 100.0 * g.n_sites / g.n_diag.where(g.n_diag > 0)
    return g


def fig8_strata(run: Path, out: Path, df: pd.DataFrame, scores: pd.DataFrame, spec: dict) -> None:
    """What is actually in each stratum: the defining axis, the covariates, the rate statistics, and
    the autapomorphy statistics of the readout.
    """
    pairs = _read(run / "stage1" / "pairs.csv", "exp3_panel")
    m = pairs.merge(
        df.drop(columns=[c for c in df.columns if c in pairs.columns and c != "gene"]),
        on="gene",
        how="left",
    )
    m["log_cds"] = np.log10(m.cds_len_human)
    m = m.merge(autapomorphy_stats(scores, spec), on="gene", how="left")
    # Omit the redundant fractional `aa_identity`; `perc_id_hp` is the percentage-scale measure.
    # Use parallel quantity names in panel titles and place symbols and units on axes.
    kind = spec["report"]["metric"]
    panels = [
        (
            "perc_id_hp",
            "Protein identity (%)",
            "Human–platypus protein identity\n(stratifying variable; Ensembl Compara)",
        ),
        ("log_cds", "log₁₀ CDS length (bp)", "Human coding-sequence length"),
        ("aln_indel_frac", "Indel fraction", "Indel content of the pairwise alignment"),
        ("prefix_offset_h_bp", "Offset (bp)", "Prompt-window offset, human CDS"),
        ("prefix_offset_p_bp", "Offset (bp)", "Prompt-window offset, platypus CDS"),
        ("dN_hp_yn", "dN (subs./site)", "Nonsynonymous divergence"),
        ("dS_hp_yn", "dS (subs./site)", "Synonymous divergence"),
        ("tree_len", "Tree length (subs./site)", "Total branch length of the gene tree"),
        # ---- Readout composition
        ("n_sites", "Sites per gene (n)", f"Scorable {kind} sites\n(in the scored window)"),
        (
            "strict_frac",
            "Private bp / platy bp (not human) (%)",
            "Strictness: diagnostic sites that survive\nthe uniqueness test",
        ),
        ("voters", "Ortholog species (n)", "Voting depth behind each call\n(mammals aligned)"),
        (
            "unsteered",
            f"{kind.capitalize()} recovery (%)",
            "Unsteered recovery floor\n(what every gain is measured against)",
        ),
    ]
    panels = [t for t in panels if t[0] in m.columns]
    # The last four panels describe the READOUT (how much evidence each gene carries), not the
    # Write a full diagnostic and a sampling-frame-only publication figure.
    frame_only = [t for t in panels if t[0] not in READOUT_PANELS]
    _draw_strata_panels(panels, m, spec, out, "8_strata_composition", ncols=4)
    if len(frame_only) < len(panels):
        # Portrait (2 x 4) for the publication version: at 8 panels it sets beside a column of
        # body text, and the eye reads down a stratum axis rather than across four wide rows.
        _draw_strata_panels(frame_only, m, spec, out, "8b_strata_composition_frame", ncols=2)


# Panels describing the READOUT (how much evidence each gene carries) rather than the sampling
# Keep readout panels separable from the sampling-frame publication figure.
READOUT_PANELS = ("n_sites", "strict_frac", "voters", "unsteered")


PANEL_W, PANEL_H = 3.6, 3.1  # inches per box-per-stratum panel, whatever the grid
PUB_STRATA_ROW_H = 250.0  # points per publication row of stratum panels


def _draw_strata_panels(panels, m, spec, out, stem: str, ncols: int = 4) -> None:
    """One box-per-stratum panel per entry in `panels`, `ncols` to a row."""
    cols = [SEQ(0.25 + 0.16 * i) for i in range(5)]
    nrow = -(-len(panels) // ncols)
    if pub.is_on():
        # Two columns across an exact 1,000 pt page; the grid still grows downward with the
        # panel count rather than widening, which is what the diagnostic version does too.
        fig, axes = plt.subplots(
            nrow, ncols, squeeze=False, figsize=pub.size(pub.FULL, PUB_STRATA_ROW_H * nrow)
        )
    else:
        fig, axes = plt.subplots(
            nrow, ncols, figsize=(PANEL_W * ncols, PANEL_H * nrow), squeeze=False
        )
    rs = np.random.default_rng(0)
    for ax, (col, ylab, title) in zip(axes.ravel(), panels, strict=False):
        data = [m.loc[m.stratum == s, col].dropna().to_numpy(float) for s in range(5)]
        bp = ax.boxplot(
            data,
            positions=range(5),
            widths=0.62,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color=acs.apc.white, lw=1.6),
        )
        for patch, c in zip(bp["boxes"], cols, strict=False):
            patch.set_facecolor(c)
            patch.set_edgecolor("none")
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color(acs.SERIES_MUTED)
        for s, d in enumerate(data):
            ax.scatter(
                s + rs.uniform(-0.16, 0.16, len(d)),
                d,
                s=3.2,
                color=acs.ANNOTATION,
                alpha=0.30,
                lw=0,
                zorder=3,
            )
        ax.set_xticks(range(5), ["s0", "s1", "s2", "s3", "s4"], fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=8.5)
        # A readout panel drawn over fewer genes than the stratum holds is saying something about
        # the evidence, not about biology -- so it says so, per panel.
        if col in READOUT_PANELS:
            miss = [int((m.stratum == st).sum() - len(d)) for st, d in enumerate(data)]
            if any(miss):
                ax.text(
                    0.02,
                    0.98,
                    "genes with no evidence, s0→s4:  " + "/".join(str(x) for x in miss),
                    transform=ax.transAxes,
                    fontsize=6.2,
                    color=WARM,
                    style="italic",
                    va="top",
                )
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    n = m.groupby("stratum").size().to_dict()
    # coverage_note defines the diagnostic-site set and how many genes are scorable — that is
    # about the READOUT, so it belongs only on the version that draws the readout panels.
    note = coverage_note(spec) if any(c in READOUT_PANELS for c, _, _ in panels) else ""
    title = (
        "Composition of each conservation stratum.  n = "
        + " / ".join(f"{n.get(s, 0)}" for s in range(5))
        + " for s0 (least conserved) through s4 (most conserved).  Boxes give the "
        "interquartile range and median; points are individual genes.  Strata are the "
        "sampling frame rather than an experimental factor, so these panels are "
        "descriptive and carry no significance tests." + ("  " + note if note else "")
    )
    # Wrap to the figure's own width rather than at hard-coded newlines: the same sentence has to
    # sit above a 4-column landscape grid and a 2-column portrait one. ~13 characters per inch at
    # this size, measured against the 4-column layout the wording was originally set for.
    if pub.is_on():
        # The suptitle is the caption: n per stratum, what the boxes are, and the fact that the
        # strata are the sampling frame rather than a factor. All of that travels in the pub
        # caption instead of on the artwork.
        pub.tight(fig)
    else:
        fig.suptitle(textwrap.fill(title, width=int(13 * PANEL_W * ncols)), y=1.005, fontsize=9.5)
        fig.tight_layout()
    save(fig, out, stem)


LEGACY5 = ["tree_len", "diameter", "focal_target_dist", "mean_target_dist", "treeness"]


def fig9_legacy(run: Path, out: Path, df: pd.DataFrame, gains: dict) -> None:
    """The five legacy tree statistics against every outcome (design § Stage 5, legacy arm)."""
    m = df.merge(gains["add_a1.0"], on="gene", how="inner")
    rows = [
        ("loo_cos", "orientation cos(D_i, v_-i)"),
        ("loo_cos_pc1", "orientation, PC1 removed"),
        ("delta_norm", "magnitude ||D_i||"),
        ("gain_pp", "steering gain, α=1 (pp)"),
    ]
    fig, axes = plt.subplots(4, 5, figsize=(17.0, 12.4))
    for r, (outc, ylab) in enumerate(rows):
        for c, pred in enumerate(LEGACY5):
            promoted = pred in CONFIRMATORY
            _scatter_rho(
                axes[r, c],
                m[pred].to_numpy(float),
                m[outc].to_numpy(float),
                pred + ("   [confirmatory]" if promoted else ""),
                ylab if c == 0 else "",
                pred if r == 0 else "",
                color=INK if not promoted else TEAL,
            )
            if promoted:
                for s in axes[r, c].spines.values():
                    s.set_color(TEAL)
                    s.set_linewidth(1.4)
            if outc == "gain_pp":
                axes[r, c].axhline(0, color=GREY, ls=":", lw=1.0, zorder=0)
    fig.suptitle(
        "The five legacy tree statistics vs every outcome — one point per gene, "
        "least-squares line, Spearman ρ.  EXPLORATORY: `tree_len`, `diameter` and the two "
        "target-distance statistics all load on PC1, so these are ~2–3 independent tests, "
        "not 5.\n`treeness` (teal) is the one orthogonal axis and is in the confirmatory "
        "four.  Fixed-topology recipe only — the 2026-08-04 free-topology recipe was not "
        "re-run, so the two-recipe comparison the design asks for does not exist.",
        y=1.005,
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, out, "9_legacy_tree_stats")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--mode", default="cds_mean")
    ap.add_argument("--arm-dir", default="stage4_cds_mean_blocks27")
    ap.add_argument(
        "--only",
        nargs="+",
        type=int,
        choices=range(1, 10),
        default=list(range(1, 10)),
        help="figure numbers to (re)build; 2 and 7 need loo_cos_pc1 (cached after the "
        "first run; the first one costs a 754 MB npz load)",
    )
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at PUBLICATION geometry: exact panel widths, the style guide's "
        "15 pt type with monospaced numerals, and no in-artwork title. Only "
        f"{sorted(PUB_STEMS)} are written, into <run>/figures/pub/ — the other "
        "panels are diagnostic grids that were never sized for a page.",
    )
    add_metric_args(ap)
    args = ap.parse_args()

    global FROM_FIGURE_DATA
    FROM_FIGURE_DATA = args.from_figure_data
    if args.pub:
        pub.enable()

    out = args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    set_pub_style(title_size=9, tick_size=7)
    want = set(args.only)

    df = gains = scores = spec = None
    if want & {2, 3, 6, 7, 8, 9}:
        print("building the per-gene table (same joins as stage5_merge)...")
        df = per_gene_table(args.run, args.layer, args.mode, with_pc1=bool(want & {2, 7, 9}))
        scores, spec = load_scores(
            args.run,
            args.arm_dir,
            scores=args.scores,
            metric=args.metric,
            min_voters=args.min_voters,
            from_figure_data=args.from_figure_data,
        )
        gains = gain_table(scores, spec)
        print(
            f"  {len(df)} genes with geometry + tree; "
            f"{len(gains['add_a1.0'])} with a steering outcome\n"
        )

    if 1 in want:
        fig1_qc(args.run, out)
    if 2 in want:
        fig2_h1ab(args.run, out, df, args.layer, args.mode)
    if 3 in want:
        fig3_h2a(args.run, out, df, gains)
    if 4 in want:
        fig4_h1c(args.run, out)
    if 5 in want:
        fig5_matrix(args.run, out)
    if 6 in want:
        fig6_arms(args.run, out, args.arm_dir, gains, df, spec)
    if 7 in want:
        fig7_conservation(args.run, out, df, gains, scores, spec)
    if 8 in want:
        fig8_strata(args.run, out, df, scores, spec)
    if 9 in want:
        fig9_legacy(args.run, out, df, gains)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
