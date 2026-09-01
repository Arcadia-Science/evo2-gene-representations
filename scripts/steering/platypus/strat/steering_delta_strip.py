"""Steering performance in one panel: what steering buys, what it costs, and whether it stays
coherent.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402
from strat_metric import (  # noqa: E402
    DEFAULT_METRIC,
    METRICS,
    add_metric_args,
    coverage_note,
    load_scores,
)

# Publication geometry. The height is the axes plus everything stacked under it at 15 pt:
# two-line readout labels, the block brackets, and the conservation-band key.
PUB_FIG_H = 640.0
PUB_BELOW_AXES = 175.0  # points under the axes for the readout labels and the block brackets
PUB_LEFT = 105.0  # left inset: the rotated y title plus the widest tick label ("+2.5")
PUB_TOP = 75.0  # top inset: the 30 pt margin plus the marks key above the axes

INK = acs.SERIES_PRIMARY
GREY = acs.SERIES_MUTED

# Sequential = ONE hue, light->dark. The light anchor is a visible tint rather than white so the
# least-conserved band stays legible against the panel; lightness is monotone across the ramp.
# Conservation ramp: blue_shades run pale → dark, i.e. the same ordered-series convention
# as everywhere else, ending on the primary series colour.
CONS_CMAP = LinearSegmentedColormap.from_list(
    "cons_seq", [acs.apc.sky, acs.apc.vital, acs.apc.aegean, INK]
)
STRATUM_POS = [0.12, 0.34, 0.55, 0.76, 0.96]  # where each band sits on the ramp
STRATUM_LAB = ["s0  fastest", "s1", "s2", "s3", "s4  conserved"]

# The site-recovery readout is whichever site set strat_metric loaded, always aliased to `metric`,
# so this module never names a specific column and cannot drift from the other figures.
SITE_KEY = "metric"


def readouts(mspec: dict) -> list:
    """key, block, label, arrow gloss (+1 = up is better), how to derive it from the per-sample
    frame.
    """
    return [
        (SITE_KEY, "on-target", mspec["strip"], +1, None),
        ("aa_id_to_target", "on-target", "amino-acid identity\nto platypus", +1, None),
        (
            "indel_frac",
            "coherence",
            "indel burden\n(% of generated bp)",
            -1,
            lambda s: 100.0 * s.indel_bp / s.gen_bp,
        ),
        (
            "stop_density",
            "coherence",
            "premature stops\n(per 100 codons)",
            -1,
            lambda s: 100.0 * s.n_stop_codons / (s.gen_bp / 3.0),
        ),
    ]


READOUTS = readouts(METRICS[DEFAULT_METRIC])  # module default, for importers and --help text


def beeswarm(y: np.ndarray, width: float, n_bins: int = 88) -> np.ndarray:
    """Deterministic symmetric offsets: bin on y, then fan each bin out around 0."""
    off = np.zeros_like(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return off
    edges = np.linspace(np.nanmin(y[ok]), np.nanmax(y[ok]) + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = np.where(ok & (idx == b))[0]
        if len(sel) == 0:
            continue
        sel = sel[np.argsort(y[sel])]  # stable fan order within the bin
        k = len(sel)
        seats = np.arange(k) - (k - 1) / 2.0
        span = max(abs(seats).max(), 1.0)
        off[sel] = seats / span * width * min(1.0, k / 12.0)
    return off


def paired_bootstrap(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Mean and a percentile CI resampling GENES (the unit of independence), not gene x sample."""
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_deltas(
    run: Path,
    arm_dir: str,
    conditions: list[str],
    scale: str,
    *,
    scores: str | None = None,
    metric: str = DEFAULT_METRIC,
    min_voters: int = 1,
    from_figure_data: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Long table: one row per gene x readout x condition, with conservation and stratum attached.
    """
    s, mspec = load_scores(
        run,
        arm_dir,
        scores=scores,
        metric=metric,
        min_voters=min_voters,
        from_figure_data=from_figure_data,
    )
    for key, _, _, _, derive in readouts(mspec):
        if derive is not None:
            s[key] = derive(s)

    keys = [k for k, _, _, _, _ in readouts(mspec)]
    g = s.groupby(["gene", "condition"])[keys].mean()
    cons = s.groupby("gene").perc_id_hp.first()
    strat = s.groupby("gene").stratum.first()

    rows = []
    for key in keys:
        w = g[key].unstack()
        base = w["unsteered"]
        for cond in conditions:
            if cond not in w.columns:
                raise SystemExit(f"condition {cond!r} not in {arm_dir}; have {sorted(w.columns)}")
            delta = w[cond] - base
            if scale == "relative":
                delta = 100.0 * delta / base.replace(0, np.nan)
            rows.append(
                pd.DataFrame(
                    {
                        "gene": w.index,
                        "metric": key,
                        "condition": cond,
                        "delta": delta.values,
                        "baseline": base.values,
                        "perc_id_hp": cons.reindex(w.index).values,
                        "stratum": strat.reindex(w.index).values,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True), mspec


def _kde_profile(y, disp, n=256):
    """Shared front half of every violin: KDE in DISPLAY space, returned on a display-space grid."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return None
    u = disp(y)
    if np.ptp(u) < 1e-9:
        return None
    try:
        kde = stats.gaussian_kde(u)
    except np.linalg.LinAlgError:
        return None
    pad = 0.12 * np.ptp(u)
    ug = np.linspace(u.min() - pad, u.max() + pad, n)
    return u, ug, kde(ug), kde.factor * np.std(u)


def violin(ax, x, y, half, face, disp, undisp, edge="white"):
    """Flat-filled violin -- one colour for the whole shape."""
    prof = _kde_profile(y, disp)
    if prof is None:
        yy = np.asarray(y, float)
        yy = yy[np.isfinite(yy)]
        if len(yy):
            ax.plot([x, x], [yy.min(), yy.max()], color=face, lw=3, zorder=2)
        return
    _, ug, d, _ = prof
    w = d / d.max() * half
    ax.fill_betweenx(
        undisp(ug), x - w, x + w, facecolor=face, edgecolor=edge, lw=0.7, zorder=2, alpha=0.95
    )


def gradient_profile(y, cvals, half, disp, undisp):
    """Geometry + local mean conservation for a gradient violin, computed but not drawn."""
    yy = np.asarray(y, float)
    cc = np.asarray(cvals, float)
    ok = np.isfinite(yy) & np.isfinite(cc)
    yy, cc = yy[ok], cc[ok]
    prof = _kde_profile(yy, disp)
    if prof is None:
        return None
    u, ug, d, bw = prof
    w = d / d.max() * half

    # local mean of the colour variable, Gaussian-weighted in display space
    z = (ug[:, None] - u[None, :]) / max(bw, 1e-9)
    kw = np.exp(-0.5 * z * z)
    denom = kw.sum(axis=1)
    cloc = np.where(denom > 1e-12, (kw @ cc) / np.maximum(denom, 1e-12), np.nan)
    # far tails can end up with no support; carry the nearest supported value outward
    cloc = pd.Series(cloc).ffill().bfill().to_numpy()
    return undisp(ug), w, cloc


def draw_gradient_violin(ax, x, geom, cmap, norm):
    """
    Render a measured gradient violin: each horizontal slice takes its local mean conservation.
    """
    from matplotlib.collections import PolyCollection

    yg, w, cloc = geom
    quads = [
        [(x - w[i], yg[i]), (x + w[i], yg[i]), (x + w[i + 1], yg[i + 1]), (x - w[i + 1], yg[i + 1])]
        for i in range(len(yg) - 1)
    ]
    face = cmap(norm(0.5 * (cloc[:-1] + cloc[1:])))
    # a hairline edge in the slice's own colour closes the seams between adjacent quads
    ax.add_collection(PolyCollection(quads, facecolors=face, edgecolors=face, lw=0.4, zorder=2))
    # one outline over the top, so the silhouette stays crisp against the gradient
    ax.plot(
        np.r_[x - w, (x + w)[::-1], x - w[0]],
        np.r_[yg, yg[::-1], yg[0]],
        color="white",
        lw=0.9,
        zorder=3,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, default=ROOT / "results" / "2026-08-08_platypus-strat-400")
    ap.add_argument("--arm-dir", default="stage4_cds_mean_blocks27")
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["add_a1.0"],
        help="steered conditions to draw; each adds a column per readout. Pass e.g. "
        "'add_a1.0 random_a1.0' to put the norm-matched control beside the arm",
    )
    ap.add_argument(
        "--style",
        choices=["violin-gradient", "violin-strata", "violin", "swarm"],
        default="violin-gradient",
        help="violin-gradient (default) is one violin per readout whose fill is a "
        "vertical conservation gradient; violin-strata splits each readout into "
        "one violin per conservation band; violin pools it into one flat shape; "
        "swarm draws every gene as a dot",
    )
    ap.add_argument(
        "--readouts",
        nargs="+",
        default=None,
        help=f"subset/order of readouts (default all): "
        f"{' '.join(k for k, _, _, _, _ in READOUTS)} "
        f"(the site-recovery readout is '{SITE_KEY}' whichever site set is "
        f"selected)",
    )
    ap.add_argument(
        "--scale",
        choices=["pp", "relative"],
        default="pp",
        help="pp = percentage points vs unsteered (default, comparable across "
        "readouts); relative = per cent of the gene's own baseline",
    )
    ap.add_argument(
        "--yscale",
        choices=["symlog", "linear"],
        default="symlog",
        help="symlog (default) is LINEAR through zero out to --linthresh and "
        "logarithmic beyond, which gives the dense middle most of the axis while "
        "keeping the broken-protein tail on the page. Distances are then not "
        "comparable across the break -- use linear when the reader must eyeball "
        "effect sizes",
    )
    ap.add_argument(
        "--linthresh",
        type=float,
        default=5.0,
        help="half-width of the linear region around zero, in the y unit",
    )
    ap.add_argument(
        "--clip-pct",
        type=float,
        default=None,
        help="clip the y axis to this percentile on each side; off-scale genes are "
        "drawn as carets at the edge and counted in the margin note. Defaults to "
        "1.0 under --yscale linear and 0 under symlog, which needs no clip",
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--notes",
        action="store_true",
        help="print the caveat block under the panel: the symlog break, any off-scale "
        "genes, and the H2a null behind the conservation colouring. Off by "
        "default so the figure drops straight into a deck — the caveats then "
        "travel in the caption instead, and they still have to travel",
    )
    ap.add_argument("--stem", default=None, help="output filename stem (default is derived)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <run>/figures)")
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at PUBLICATION geometry: an exact 1,000 pt panel, the style "
        "guide's 15 pt type with monospaced numerals, no in-artwork title, "
        "direction glosses in words (Atkinson has no arrow glyphs), and the "
        "conservation bands as a key rather than an inline table. Writes into "
        "<out>/pub/.",
    )
    add_metric_args(ap)
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    run = args.run
    if args.clip_pct is None:
        args.clip_pct = 0.0 if args.yscale == "symlog" else 1.0
    out = args.out or (run / "figures")
    out.mkdir(parents=True, exist_ok=True)
    set_pub_style(title_size=9, tick_size=7)

    df, mspec = load_deltas(
        run,
        args.arm_dir,
        args.conditions,
        args.scale,
        scores=args.scores,
        metric=args.metric,
        min_voters=args.min_voters,
        from_figure_data=args.from_figure_data,
    )
    spec = {k: (blk, lab, arrow) for k, blk, lab, arrow, _ in readouts(mspec)}
    keys = args.readouts or [k for k, _, _, _, _ in readouts(mspec)]
    bad = [k for k in keys if k not in spec]
    if bad:
        raise SystemExit(f"unknown readout(s) {bad}; choose from {list(spec)}")
    df = df[df.metric.isin(keys)]
    n_genes = df.gene.nunique()
    conds = args.conditions
    unit = "pp" if args.scale == "pp" else "% of baseline"
    strata = sorted(df.stratum.dropna().unique())

    # ---- column layout -----------------------------------------------------------------------
    # A "column" is one readout x one condition. Under violin-strata each column is further split
    # into one violin per conservation band; the column keeps a single tick label either way.
    per_col = len(strata) if args.style == "violin-strata" else 1
    col_w = 0.42 * per_col + 0.55
    cols, x = [], 0.0
    for key in keys:
        for ci, cond in enumerate(conds):
            cols.append({"metric": key, "cond": cond, "x": x, "first": ci == 0})
            x += col_w
        x += 0.42
    # a wider gap between the on-target block and the coherence block
    for c in cols:
        c["block"] = spec[c["metric"]][0]
    shift, prev = 0.0, cols[0]["block"]
    for c in cols:
        if c["block"] != prev:
            shift += 0.55
            prev = c["block"]
        c["x"] += shift

    if pub.is_on():
        # A fixed 1,000 pt page instead of a width derived from the column count. The extra
        # height is for what sits BELOW the axes at 15 pt — two-line readout labels, the
        # on-target / coherence brackets, and the conservation-band key.
        fig, ax = plt.subplots(figsize=pub.size(pub.FULL, PUB_FIG_H))
    else:
        fig_w = max(7.4, 2.9 + (cols[-1]["x"] + col_w))
        fig, ax = plt.subplots(figsize=(fig_w, 5.9))
    norm = Normalize(vmin=np.floor(df.perc_id_hp.min()), vmax=100.0)

    t = args.linthresh

    def disp(v):
        return (
            np.arcsinh(np.asarray(v, float) / t)
            if args.yscale == "symlog"
            else np.asarray(v, float)
        )

    def undisp(u):
        return (
            t * np.sinh(np.asarray(u, float)) if args.yscale == "symlog" else np.asarray(u, float)
        )

    if args.clip_pct > 0:
        ylo = float(np.nanpercentile(df.delta, args.clip_pct))
        yhi = float(np.nanpercentile(df.delta, 100 - args.clip_pct))
        pad = 0.10 * (yhi - ylo)
        ylo, yhi = ylo - pad, yhi + pad
    else:
        ylo, yhi = -np.inf, np.inf
    n_off = 0

    def summary_marker(xm, y, label=True, compact=False):
        """Mean with its bootstrap CI, plus the median -- computed on ALL genes in the cell."""
        mean, lo, hi = paired_bootstrap(y, args.n_boot, args.seed)
        med = float(np.nanmedian(y))
        w = 0.035 if compact else 0.055
        ax.plot([xm, xm], [lo, hi], color=acs.apc.black, lw=1.4, solid_capstyle="round", zorder=5)
        ax.plot([xm - w, xm + w], [mean, mean], color=acs.apc.black, lw=2.4, zorder=5)
        ax.plot(
            [xm - w * 0.8, xm + w * 0.8],
            [med, med],
            color=acs.apc.black,
            lw=1.1,
            ls=(0, (1.4, 1.0)),
            zorder=5,
        )
        if label:
            sig = "" if (lo <= 0 <= hi) else "*"
            ax.text(
                xm + w + 0.06,
                mean,
                f"{mean:+.2f}{sig}",
                fontsize=8,
                color=acs.apc.black,
                va="center",
                ha="left",
                zorder=6,
            )
        return mean, lo, hi, med

    for c in cols:
        sub = df[(df.metric == c["metric"]) & (df.condition == c["cond"])]
        is_control = c["cond"].startswith("random")

        if args.style == "swarm":
            y = sub.delta.to_numpy(float)
            xs = c["x"] + beeswarm(disp(y), 0.34)
            inside = (y >= ylo) & (y <= yhi)
            ax.scatter(
                xs[inside],
                y[inside],
                s=26,
                c=sub.perc_id_hp.to_numpy(float)[inside],
                cmap=CONS_CMAP,
                norm=norm,
                lw=0.45,
                edgecolor=acs.apc.white,
                alpha=0.95 if not is_control else 0.55,
                zorder=2,
            )
            for mask, edge, mark in ((y < ylo, ylo, "v"), (y > yhi, yhi, "^")):
                off = mask & np.isfinite(y)
                n_off += int(off.sum())
                if off.any():
                    ax.scatter(
                        xs[off],
                        np.full(off.sum(), edge),
                        s=22,
                        marker=mark,
                        c=sub.perc_id_hp.to_numpy(float)[off],
                        cmap=CONS_CMAP,
                        norm=norm,
                        lw=0.45,
                        edgecolor="white",
                        zorder=3,
                        clip_on=False,
                    )
            summary_marker(c["x"] + 0.54, y)

        elif args.style == "violin":
            y = sub.delta.to_numpy(float)
            violin(
                ax,
                c["x"],
                y,
                0.34,
                CONS_CMAP(0.45) if not is_control else acs.apc.denim,
                disp,
                undisp,
            )
            summary_marker(c["x"] + 0.46, y)

        elif args.style == "violin-gradient":
            pass  # measured and drawn in a second pass below, once the ramp range is known

        else:  # violin-strata
            step = 0.42
            x0 = c["x"] - step * (len(strata) - 1) / 2.0
            for si, st in enumerate(strata):
                ys = sub[sub.stratum == st].delta.to_numpy(float)
                face = CONS_CMAP(STRATUM_POS[int(st)] if int(st) < len(STRATUM_POS) else 0.5)
                xs = x0 + si * step
                violin(ax, xs, ys, step * 0.46, face, disp, undisp)
                summary_marker(xs, ys, label=False, compact=True)

    # ---- gradient violins: measure every column, then colour them on the range that occurs ----
    grad_norm = norm
    if args.style == "violin-gradient":
        geoms = {}
        for c in cols:
            sub = df[(df.metric == c["metric"]) & (df.condition == c["cond"])]
            g = gradient_profile(
                sub.delta.to_numpy(float), sub.perc_id_hp.to_numpy(float), 0.34, disp, undisp
            )
            if g is not None:
                geoms[(c["metric"], c["cond"])] = g
        if geoms:
            allc = np.concatenate([g[2] for g in geoms.values()])
            lo_c, hi_c = float(np.nanmin(allc)), float(np.nanmax(allc))
            span = max(hi_c - lo_c, 1e-6)
            grad_norm = Normalize(vmin=lo_c - 0.05 * span, vmax=hi_c + 0.05 * span)
        for c in cols:
            g = geoms.get((c["metric"], c["cond"]))
            if g is not None:
                draw_gradient_violin(ax, c["x"], g, CONS_CMAP, grad_norm)
            summary_marker(
                c["x"] + 0.46,
                df[(df.metric == c["metric"]) & (df.condition == c["cond"])].delta.to_numpy(float),
            )

    # ---- axes --------------------------------------------------------------------------------
    if args.yscale == "symlog":
        ax.set_yscale("symlog", linthresh=t, linscale=1.0)
        lim = float(np.nanmax(np.abs(df.delta))) * 1.35
        yt = [0.0]
        for v in (t / 2, t, 2 * t, 5 * t, 10 * t, 20 * t):
            if v <= lim:
                yt += [v, -v]
        yt = sorted(yt)
        ax.set_yticks(yt)
        ax.set_yticklabels(["0" if v == 0 else f"{v:+g}" for v in yt], fontsize=7)
        for s in (+1, -1):
            ax.axhline(s * t, color=acs.GRID, lw=0.9, ls=(0, (5, 3)), zorder=0)
        # annotations hug the LEFT spine: the right margin carries the per-column mean labels
        # At 15 pt these sit ON the leftmost violin, so the publication figure hangs them off
        # the right-hand end of the line instead, past the last column.
        _refx, _refha = (0.998, "right") if pub.is_on() else (0.006, "left")
        ax.text(
            _refx,
            t,
            f"log beyond ±{t:g}",
            transform=ax.get_yaxis_transform(),
            ha=_refha,
            va="bottom",
            fontsize=6.2,
            color=acs.ANNOTATION,
            zorder=6,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"),
        )

    ax.axhline(0, color=GREY, ls=":", lw=1.1, zorder=1)
    _ux, _uha = (0.998, "right") if pub.is_on() else (0.006, "left")
    ax.text(
        _ux,
        0.0,
        "unsteered",
        transform=ax.get_yaxis_transform(),
        ha=_uha,
        va="bottom",
        fontsize=6.5,
        color=GREY,
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"),
    )

    # Atkinson Hyperlegible has no arrow glyphs at all, so the publication figure says which
    # direction is better in words. (Checked against the font's cmap; arcadia_pub.audit_glyphs
    # fails the render rather than letting matplotlib draw a tofu box.)
    def better(key: str) -> str:
        up = spec[key][2] > 0
        if pub.is_on():
            return "higher is better" if up else "lower is better"
        return "↑ better" if up else "↓ better"

    ax.set_xticks([c["x"] for c in cols])
    if len(conds) > 1:
        ax.set_xticklabels([c["cond"] for c in cols], fontsize=7)
        for key in keys:
            xs_ = [c["x"] for c in cols if c["metric"] == key]
            ax.text(
                np.mean(xs_),
                -0.125,
                f"{spec[key][1]}\n{better(key)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=INK,
            )
    else:
        ax.set_xticklabels(
            [f"{spec[c['metric']][1]}\n{better(c['metric'])}" for c in cols], fontsize=8
        )

    # block brackets under the readout labels, so "on-target" and "coherence" read as two things
    ytxt = -0.275 if len(conds) > 1 else -0.245
    if pub.is_on():
        # The two-line readout labels are ~2x taller at 15 pt, so the bracket drops further to
        # clear them. Fraction of the AXES height, which pub mode also changes — hence a value
        # of its own rather than a tweak to the diagnostic one.
        ytxt = -0.34
    for blk in dict.fromkeys(c["block"] for c in cols):
        xs_ = [c["x"] for c in cols if c["block"] == blk]
        lo_, hi_ = min(xs_) - col_w * 0.42, max(xs_) + col_w * 0.42
        ax.plot(
            [lo_, hi_],
            [ytxt + 0.035] * 2,
            color=acs.GRID,
            lw=1.2,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        ax.text(
            np.mean([lo_, hi_]),
            ytxt,
            blk,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color=GREY,
            style="italic",
        )

    ax.set_xlim(cols[0]["x"] - col_w * 0.62, cols[-1]["x"] + col_w * 0.62 + 0.35)
    ax.set_ylabel(f"Δ vs the same gene's unsteered cell ({unit})")
    ax.tick_params(axis="x", length=0, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # ---- legends -----------------------------------------------------------------------------
    pub_extra_key = None  # (title, handles) for the publication key, filled by violin-strata
    if args.style == "violin-strata":
        # The table IS the legend: its stratum cell is filled with the band's own colour, so a
        # separate patch legend would repeat it and fight it for the same strip of space.
        # Edges are read off the DRAWN genes, not hard-coded -- the strata are pool quintiles of
        # perc_id_hp fixed at stage 0 (design § Sampling), so they land on odd numbers, and the
        # per-band n drifts off 80 wherever QC dropped a gene.
        edges = df.groupby("stratum").perc_id_hp.agg(["min", "max", "nunique"])
        body = [
            [
                STRATUM_LAB[int(st)],
                f"{edges.loc[st, 'min']:.0f}–{edges.loc[st, 'max']:.0f} %",
                f"{int(edges.loc[st, 'nunique'])}",
            ]
            for st in strata
        ]
        if pub.is_on():
            # Use color patches for the equal-sized conservation strata.
            band_key = [
                Patch(
                    facecolor=CONS_CMAP(STRATUM_POS[int(st)]),
                    edgecolor=acs.apc.white,
                    label=f"{STRATUM_LAB[int(st)]} "
                    f"({edges.loc[st, 'min']:.0f}–{edges.loc[st, 'max']:.0f}%)",
                )
                for st in strata
            ]
            pub_extra_key = ("Conservation band — human vs platypus protein identity", band_key)
        else:
            tab = ax.table(
                cellText=body,
                colLabels=["stratum", "protein identity", "n genes"],
                cellLoc="center",
                bbox=[0.425, -0.60, 0.15, 0.28],
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(6.6)
            for (r, c), cell in tab.get_celld().items():
                cell.set_edgecolor(acs.GRID)
                cell.set_linewidth(0.7)
                if r == 0:
                    cell.set_facecolor(acs.apc.gray)
                    cell.get_text().set_color(GREY)
                elif c == 0:
                    cell.set_facecolor(CONS_CMAP(STRATUM_POS[int(strata[r - 1])]))
                    cell.get_text().set_color(
                        "white" if STRATUM_POS[int(strata[r - 1])] > 0.5 else acs.apc.black
                    )
            ax.text(
                0.5,
                -0.305,
                "conservation band — human↔platypus protein identity",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=GREY,
            )
    elif args.style == "violin":
        pass  # flat fill carries no colour job, so a colourbar here would encode nothing
    else:
        # The gradient bar is deliberately labelled LOCAL MEAN and spans only the range the local
        # means reach (~74-80 %), not the 50-100 % the genes span. Stating that range on the bar is
        # what keeps the stretched ramp honest: it shows the relationship is weak, it doesn't hide
        # it.
        grad = args.style == "violin-gradient"
        cb = fig.colorbar(
            plt.cm.ScalarMappable(norm=grad_norm if grad else norm, cmap=CONS_CMAP),
            ax=ax,
            pad=0.045,
            fraction=0.036,
        )
        cb.set_label(
            (
                "local mean human↔platypus protein identity (%)\nof the genes at that height"
                if grad
                else "human↔platypus protein identity (%)  →  more conserved"
            ),
            fontsize=7.0 if grad else 7.5,
        )
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)

    marks = [
        Line2D([], [], color=acs.apc.black, lw=2.4, label="mean, 95 % bootstrap CI over genes"),
        Line2D([], [], color=acs.apc.black, lw=1.1, ls=(0, (1.4, 1.0)), label="median"),
    ]
    if args.style == "swarm":
        marks.append(
            Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc=acs.apc.vital,
                mec=acs.apc.white,
                mew=0.45,
                ms=5.5,
                label=f"one gene (n = {n_genes})",
            )
        )
    else:
        marks.append(
            Patch(
                facecolor=acs.apc.denim,
                edgecolor=acs.apc.white,
                label=f"KDE over genes (n = {n_genes})",
            )
        )
    style_tag = {
        "violin-gradient": "_violin_gradient",
        "violin-strata": "_violin_strata",
        "violin": "_violin",
        "swarm": "",
    }[args.style]
    stem = args.stem or (
        f"10_steering_delta{style_tag}" + ("" if args.scale == "pp" else "_relative")
    )
    if pub.is_on():
        # Two keys, as in the diagnostic: what the MARKS mean sits above the axes, and the
        # conservation bands sit below under their own title. Merging them into one block put
        # "mean, 95 % bootstrap CI" under a heading that reads "Conservation band — …", which
        # is simply the wrong label for it.
        ax.legend(
            handles=marks,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.005),
            ncol=len(marks),
            handletextpad=0.6,
            columnspacing=1.6,
        )
        title, extra = pub_extra_key if pub_extra_key else ("Key", [])
        band_legend, key_h = pub.key_below(
            fig, extra, [h.get_label() for h in extra], title=title, width=pub.FULL, cols=len(extra)
        )
        # key_below anchors at the figure's bottom-left corner and leaves it to the layout step
        # to lift it inside the margin. This figure places its axes by hand, so it has to do
        # that itself or the key sits half off the page.
        band_legend.set_bbox_to_anchor(
            (pub.arc.MARGIN / pub.FULL, pub.arc.MARGIN / PUB_FIG_H), transform=fig.transFigure
        )
        # Explicit placement, not tight_layout. Everything under the axes here — the three-line
        # readout labels and the block brackets — is drawn in AXES-fraction coordinates, which
        # tight_layout cannot see, so it reserves space for the (empty) tick labels instead and
        # leaves a third of the page blank between the brackets and the key.
        fig.subplots_adjust(
            left=PUB_LEFT / pub.FULL,
            right=1.0 - pub.arc.MARGIN / pub.FULL,
            bottom=(key_h + PUB_BELOW_AXES) / PUB_FIG_H,
            top=1.0 - PUB_TOP / PUB_FIG_H,
        )
        pub.finish(fig, stem, directory=out)
        plt.close(fig)
    else:
        ax.legend(
            handles=marks,
            fontsize=6.8,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.005),
            ncol=3,
            handletextpad=0.6,
            columnspacing=1.4,
        )

    arm = ", ".join(conds)
    # The suptitle, the caveat block and the tight-bbox save are the DIAGNOSTIC tail. In pub mode
    # the figure was already written above (the caption carries the title, and tight-bbox would
    # undo the exact panel width), but the tidy CSV below is written either way.
    if not pub.is_on():
        fig.suptitle(
            f"Steering buys on-target base pairs, costs little protein, and does not "
            f"decohere the sequence  ({arm}, {args.arm_dir.replace('stage4_', '')})\n"
            + coverage_note(mspec),
            y=0.995,
            fontsize=9.5,
        )
        notes = [
            "Every readout is a percentage of something in the generated sequence, so every "
            "column is percentage points on one axis."
        ]
        if args.yscale == "symlog":
            notes.append(
                f"y is symlog: linear within ±{t:g}, log beyond, so equal vertical "
                "distances are NOT equal deltas across the dashed break."
            )
        if n_off:
            notes.append(
                f"{n_off} gene(s) past the axis are drawn as carets and are still in "
                "every statistic."
            )
        notes.append(
            "\nConservation is the sampling frame, not a tested predictor: the "
            "pre-registered H2a test is null — the gradient in the gain column is "
            "unsteered-baseline headroom,\nnot conservation acting on steerability "
            "(results_n400.md § H2a)."
        )
        # the strata style parks its band legend under the block labels, so the note drops further
        if args.notes:
            fig.text(
                0.5,
                -0.13 if args.style == "violin-strata" else -0.075,
                " ".join(notes),
                ha="center",
                va="top",
                fontsize=6.8,
                color=GREY,
            )

        style_tag = {
            "violin-gradient": "_violin_gradient",
            "violin-strata": "_violin_strata",
            "violin": "_violin",
            "swarm": "",
        }[args.style]
        stem = args.stem or (
            f"10_steering_delta{style_tag}" + ("" if args.scale == "pp" else "_relative")
        )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  {stem}  ->  {out}")

    # ---- the numbers behind the picture ------------------------------------------------------
    rec = []
    for c in cols:
        sub = df[(df.metric == c["metric"]) & (df.condition == c["cond"])]
        for st in [None, *strata]:
            q = sub if st is None else sub[sub.stratum == st]
            y = q.delta.to_numpy(float)
            mean, lo, hi = paired_bootstrap(y, args.n_boot, args.seed)
            row = {
                "metric": c["metric"],
                "condition": c["cond"],
                "stratum": "all" if st is None else int(st),
                "n": int(np.isfinite(y).sum()),
                "mean": round(mean, 3),
                "ci_lo": round(lo, 3),
                "ci_hi": round(hi, 3),
                "median": round(float(np.nanmedian(y)), 3),
                "frac_positive": round(float(np.nanmean(y > 0)), 3),
                "baseline_mean": round(float(q.baseline.mean()), 3),
            }
            if st is None:
                rho, p = stats.spearmanr(q.perc_id_hp, q.delta, nan_policy="omit")
                rho_b, p_b = stats.spearmanr(q.baseline, q.delta, nan_policy="omit")
                row |= {
                    "rho_conservation": round(rho, 3),
                    "p_conservation": p,
                    "rho_baseline": round(rho_b, 3),
                    "p_baseline": p_b,
                }
            rec.append(row)
    tab = pd.DataFrame(rec)
    # Spell the site set out in the saved table: `metric` is an internal alias, and a CSV row
    # labelled "metric" does not say whether it is the autapomorphic or the diagnostic site set.
    tab["metric"] = tab["metric"].replace({SITE_KEY: mspec["col"]})
    tab.to_csv(out / f"{stem}.csv", index=False)
    print(tab[tab.stratum == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
