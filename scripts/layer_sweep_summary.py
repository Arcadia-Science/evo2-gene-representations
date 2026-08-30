"""Summary figures tracking how each baseline correlation moves across layers."""

from __future__ import annotations
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from gene_families import family_colors  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

# ── Shared style constants
AXIS_COLORS = {
    "1_homology": acs.apc.dragon,
    "control": acs.apc.chateau,
}
AXIS_LABELS = {
    "1_homology": "Homology",
    "control": "Composition (control)",
}
# Display order of the between-family baselines, grouped by axis.
BETWEEN_ORDER = [
    ("pfam_jsd", "1_homology"),
    ("gc_content", "control"),
    ("kmer", "control"),
]
ACTIVE_BETWEEN = {name for name, _axis in BETWEEN_ORDER}

# Reader-facing names for the publication figure; the keys above are CSV column names.
PUB_BASELINE_LABELS = {
    "pfam_jsd": "Pfam-domain JSD",
    "gc_content": "GC content",
    "kmer": "k-mer composition",
}

# Colour carries the axis, marker shape the individual baseline — so two baselines sharing an
# axis colour (gc_content and kmer are both "control" grey) stay distinguishable. Lines are solid
# throughout: dashes read as a data property rather than an identifier.
LEAD_SHAPES = [("o", "-"), ("s", "-"), ("^", "-"), ("D", "-")]

# Within-family baselines: (label, csv filename, rho column). Whatever is present on disk gets a
# panel. The k-mer baseline lives in one of two filenames that share a column name, first wins.
WITHIN_SPECS = [
    (
        "k-mer composition (CDS)",
        ["axisB_within_family_correlations.csv", "kmer_within_family_correlations.csv"],
        "spearman_geodesic_kmer",
    ),
    ("patristic tree", ["within_family_patristic.csv"], "spearman_geodesic_patristic"),
    ("taxonomy", ["axisB_within_family_correlations.csv"], "spearman_geodesic_taxonomy"),
    (
        "k-mer (transcript null)",
        ["within_family_kmer_transcript.csv"],
        "spearman_geodesic_kmer_transcript",
    ),
    # mammalian ortholog panel: within-ortholog-group geodesic vs the INDEPENDENT species tree
    ("species tree (mammal)", ["within_family_speciestree.csv"], "spearman_geodesic_speciestree"),
    # Mononucleotide composition control.
    ("GC content (control)", ["within_family_gc.csv"], "spearman_geodesic_gc"),
]

# Optional trailing "-<tag>" (e.g. "-cds") lets an input-variant sweep (…-blocks15-cds) parse the
# same way as the production sweep (…-blocks15).
LAYER_RE = re.compile(r"(?:blocks|L|hidden_state\.?|layer)(\d+)(?:-[\w-]+)?$")


def parse_layer(run_dir: Path) -> int | None:
    """Pull the integer layer index out of a sweep run-dir name."""
    m = LAYER_RE.search(run_dir.name)
    return int(m.group(1)) if m else None


def discover_layers(pattern: str) -> list[tuple[int, Path]]:
    """Glob → [(layer_index, run_dir), ...] sorted by layer, dirs only."""
    layers = []
    for p in sorted(Path().glob(pattern)):
        if not p.is_dir():
            continue
        layer = parse_layer(p)
        if layer is None:
            print(f"  skip (no layer token in name): {p.name}")
            continue
        layers.append((layer, p))
    return sorted(layers, key=lambda t: t[0])


# Data loading → tidy long-form frames
def load_between(
    layers: list[tuple[int, Path]],
    filename: str = "between_family_baseline_scores.csv",
    approach: str | None = None,
) -> pd.DataFrame:
    """Long frame: one row per (layer, baseline) with rho + Mantel p."""
    rows = []
    for layer, run_dir in layers:
        f = run_dir / filename
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if approach is not None:
            if "approach" not in df.columns:
                raise ValueError(f"{f} has no 'approach' column but --between-approach was given")
            df = df[df["approach"] == approach]
            if df.empty:
                continue
        for _, r in df.iterrows():
            if r["baseline"] not in ACTIVE_BETWEEN:
                continue
            rows.append(
                {
                    "layer": layer,
                    "baseline": r["baseline"],
                    "axis": r.get("axis"),
                    "rho": r.get("spearman_rho"),
                    "p_mantel": r.get("p_mantel"),
                }
            )
    return pd.DataFrame(rows)


def load_within(layers: list[tuple[int, Path]], file_suffix: str = "") -> pd.DataFrame:
    """Long frame: one row per (layer, metric, family) with rho."""
    rows = []
    for layer, run_dir in layers:
        for label, filenames, col in WITHIN_SPECS:
            for fn in filenames:
                f = run_dir / (fn.replace(".csv", f"{file_suffix}.csv") if file_suffix else fn)
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                if col not in df.columns or "family" not in df.columns:
                    continue
                for _, r in df.iterrows():
                    rows.append(
                        {
                            "layer": layer,
                            "metric": label,
                            "family": r["family"],
                            "rho": pd.to_numeric(r[col], errors="coerce"),
                        }
                    )
                break  # first file that supplies this metric wins
    return pd.DataFrame(rows)


# Figures
def _grid(n: int, ncols: int) -> tuple[int, int]:
    return (int(np.ceil(n / ncols)), ncols)


def _balanced_grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    """(rows, cols) preferring a full rectangle over a ragged last row: the widest column count that
    divides n exactly (4 -> 2x2, 6 -> 3x2), else max_cols.
    """
    exact = [c for c in range(2, min(max_cols, n) + 1) if n % c == 0]
    ncols = max(exact) if exact else min(max_cols, n)
    return (int(np.ceil(n / ncols)), ncols)


def plot_between(
    between: pd.DataFrame,
    out_dir: Path,
    title: str,
    exclude_between: list[str] | None = None,
    stem_suffix: str = "",
    lead_per_baseline: bool = False,
    *,
    kmer_k: int | None = None,
) -> None:
    if between.empty:
        print("  SKIP between figure: no between_family_baseline_scores.csv found")
        return
    set_pub_style(title_size=9, tick_size=7)
    # An axis mean over a subset of its baselines would be mislabelled, so an axis survives the
    # lead panel only if every one of its baselines is still plotted.
    drop = set(exclude_between or [])
    if drop:
        between = between[~between["baseline"].isin(drop)]
        if between.empty:
            print(
                f"  SKIP between figure: --exclude-between removed every baseline ({sorted(drop)})"
            )
            return
        print(f"  excluding between baselines: {', '.join(repr(b) for b in sorted(drop))}")
    present = [(b, ax) for b, ax in BETWEEN_ORDER if b in set(between["baseline"])]
    # Panel 0 = axis-mean overview, then one panel per baseline.
    n_panels = 1 + len(present)
    if pub.is_on():
        # The publication figure is the overview panel alone: the per-baseline panels re-plot the
        # same series one at a time, and every comparison is already legible in the overlay.
        n_panels = 1
        nrows, ncols = 1, 1
        fig, axes = plt.subplots(1, 1, dpi=300, squeeze=False, figsize=pub.size(pub.FULL, 560))
    else:
        nrows, ncols = _grid(n_panels, 4)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), dpi=300, squeeze=False
        )
    axes = axes.ravel()
    layers_sorted = sorted(between["layer"].unique())

    # Built once so the lead panel and the baseline's own panel draw the same marker.
    _seen: dict[str, int] = {}
    shape_of: dict[str, tuple] = {}
    for baseline, axis in present:
        shape_of[baseline] = LEAD_SHAPES[_seen.get(axis, 0) % len(LEAD_SHAPES)]
        _seen[axis] = _seen.get(axis, 0) + 1

    # ── lead panel: either one line per AXIS (mean) or one line per BASELINE
    ax0 = axes[0]
    plotted = {b for b, _ in present}
    if lead_per_baseline:
        # Every baseline on its own line — an axis mean hides disagreement within an axis, which
        # is the whole question when the two composition controls move differently.
        for baseline, axis in present:
            marker, ls = shape_of[baseline]
            sub = between[between["baseline"] == baseline].set_index("layer")["rho"]
            sub = sub.reindex(layers_sorted)
            if pub.is_on():
                disp = PUB_BASELINE_LABELS.get(baseline, baseline)
                label = f"{disp} (k={kmer_k})" if (baseline == "kmer" and kmer_k) else disp
            else:
                label = f"{baseline} ({AXIS_LABELS[axis]})"
            ax0.plot(
                sub.index,
                sub.values,
                ls=ls,
                marker=marker,
                ms=3.4,
                lw=1.3,
                color=AXIS_COLORS[axis],
                label=label,
            )
    else:
        for axis in ["1_homology", "2_mechanism", "control"]:
            # every baseline this axis contributes to the figure's fixed order
            axis_baselines = {b for b, a in BETWEEN_ORDER if a == axis}
            if axis_baselines - plotted:
                continue  # partially excluded axis — its mean would not be the axis mean
            sub = between[between["axis"] == axis]
            if sub.empty:
                continue
            m = sub.groupby("layer")["rho"].mean().reindex(layers_sorted)
            ax0.plot(
                m.index,
                m.values,
                "-o",
                ms=3,
                lw=1.6,
                color=AXIS_COLORS[axis],
                label=AXIS_LABELS[axis],
            )
    ax0.axhline(0, color=acs.ZERO_LINE, lw=0.6, ls="--")
    ax0.set_ylabel("Spearman ρ" if lead_per_baseline else "mean Spearman ρ")
    ax0.set_xlabel("layer")
    if pub.is_on():
        # No in-artwork title (the caption carries it). The curves cross the whole band, so there
        # is no free corner: the key sits above the axes in one row.
        pub.arc.key(
            ax0, title="Baseline", loc="lower left", ncol=len(present), bbox_to_anchor=(0.0, 1.02)
        )
    else:
        ax0.set_title("All baselines" if lead_per_baseline else "Axis means", fontweight="bold")
        ax0.legend(frameon=False, fontsize=6, loc="best")

    # shared y-limits across the per-baseline panels for honest comparison
    finite = between["rho"].replace([np.inf, -np.inf], np.nan).dropna()
    ymax = max(0.6, float(np.nanmax(np.abs(finite))) * 1.15) if len(finite) else 1.0
    ylim = (-ymax, ymax)

    # `present` still drives the lead panel's series above; this is only which baselines get
    # a panel of their OWN below it — none, in pub mode.
    panelled = [] if pub.is_on() else present
    for i, (baseline, axis) in enumerate(panelled, start=1):
        ax = axes[i]
        sub = between[between["baseline"] == baseline].sort_values("layer")
        color = AXIS_COLORS[axis]
        ax.plot(sub["layer"], sub["rho"], "-", lw=1.2, color=color, zorder=1)
        sig = sub["p_mantel"] < 0.05
        # Same marker this baseline wears in the lead panel, so the two panels read as one series.
        # Only in per-baseline lead mode: with an axis-mean lead panel there is no marker to match.
        mk = shape_of[baseline][0] if lead_per_baseline else "o"
        ax.scatter(
            sub["layer"][sig],
            sub["rho"][sig],
            s=22,
            color=color,
            marker=mk,
            zorder=3,
            label="Mantel p<0.05",
        )
        ax.scatter(
            sub["layer"][~sig],
            sub["rho"][~sig],
            s=22,
            facecolors=acs.apc.white,
            edgecolors=color,
            linewidths=0.9,
            marker=mk,
            zorder=3,
        )
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.6, ls="--")
        ax.set_ylim(*ylim)
        disp = f"kmer (k={kmer_k})" if (baseline == "kmer" and kmer_k) else baseline
        ax.set_title(f"{disp}\n({AXIS_LABELS[axis]})", color=color)
        if i % ncols == 0:
            ax.set_ylabel("Spearman ρ")
        ax.set_xlabel("layer")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title} — between-family ρ across layers", fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, f"between_axis_vs_layer{stem_suffix}")


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if pub.is_on():
        # Exact panel width, guide type, no tight-bbox crop — see arcadia_pub.finish.
        pub.drop_titles(fig)
        pub.finish(fig, stem, directory=out_dir)
    else:
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir}/{stem}.{{png,pdf}}")


def _within_metrics(
    within: pd.DataFrame, exclude: list[str] | None, quiet: bool = False
) -> list[str]:
    """Metric labels to draw: present on disk, not all-NaN, not excluded, in WITHIN_SPECS order."""
    metrics = [
        m for m in within["metric"].unique() if within[(within["metric"] == m)]["rho"].notna().any()
    ]
    metrics = [m for m in [s[0] for s in WITHIN_SPECS] if m in metrics]
    for m in exclude or []:
        if m in metrics:
            metrics.remove(m)
            if not quiet:
                print(f"  excluding within panel: {m!r} (still present in the CSV)")
        elif not quiet:
            print(
                f"  NOTE --exclude-within {m!r}: no such panel in this sweep "
                f"(labels present: {metrics})"
            )
    return metrics


def plot_within_band(
    within: pd.DataFrame,
    out_dir: Path,
    title: str,
    exclude: list[str] | None = None,
    stem_suffix: str = "",
    footnote: str | None = None,
) -> None:
    """Per-family spaghetti collapsed to mean +/- 1 SD across families, one panel per metric."""
    if within.empty:
        print("  SKIP within band figure: no within-family correlation CSVs found")
        return
    metrics = _within_metrics(within, exclude, quiet=True)
    if not metrics:
        print("  SKIP within band figure: all within-family rho are NaN")
        return
    set_pub_style(title_size=9, tick_size=7)
    # Categorical slots of the validated reference palette, fixed order, one per metric panel.
    colors = acs.categorical(7)

    # Panel 0 overlays every metric's mean with no ribbons: four translucent bands over a shared
    # y-range would obscure exactly the metric-to-metric comparison the panel exists for.
    n_panels = 1 + len(metrics)
    nrows, ncols = _grid(n_panels, min(3, n_panels))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), dpi=300, squeeze=False
    )
    axes = axes.ravel()
    # shared y-limits so panels are comparable at a glance
    lo, hi = [], []
    stats = {}
    for metric in metrics:
        sub = within[within["metric"] == metric].dropna(subset=["rho"])
        g = sub.groupby("layer")["rho"]
        m, sd, n = g.mean(), g.std(ddof=1), g.size()
        stats[metric] = (m, sd, n)
        lo.append(float((m - sd).min()))
        hi.append(float((m + sd).max()))
    ylim = (min(lo + [0.0]) - 0.05, max(hi) + 0.05)
    n_full = int(max(int(v[2].max()) for v in stats.values()))
    starred: list[str] = []  # panels whose family count falls short of n_full

    ax0 = axes[0]
    for i, metric in enumerate(metrics):
        m, _, _ = stats[metric]
        ax0.plot(m.index, m.values, "-o", ms=3, lw=1.8, color=colors[i % len(colors)], label=metric)
    ax0.axhline(0, color=acs.ZERO_LINE, lw=0.6, ls="--")
    ax0.set_ylim(*ylim)
    ax0.set_title("All metrics (means only)", fontweight="bold")
    ax0.set_xlabel("layer")
    ax0.set_ylabel("within-family Spearman ρ")
    ax0.legend(frameon=False, fontsize=6, loc="best")

    for i, metric in enumerate(metrics, start=1):
        ax, color = axes[i], colors[(i - 1) % len(colors)]
        m, sd, n = stats[metric]
        ax.fill_between(
            m.index, m - sd, m + sd, color=color, alpha=0.20, lw=0, label="±1 SD across families"
        )
        ax.plot(m.index, m.values, "-o", ms=3, lw=1.8, color=color, label="mean over families")
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.6, ls="--")
        ax.set_ylim(*ylim)
        # n is annotated only when it departs from the panel-wide family count in the suptitle.
        # On the mammal panel the protein-alignment baselines cover 47 of 48: histone_h4 is
        # identical in amino acids across all 14 species of its only scorable group, so every
        # pairwise distance is 0 and the zero-variance guard drops it. Its CDS still differ at the
        # nucleotide level, which is why the nucleotide baselines score it.
        n_lo, n_hi = int(n.min()), int(n.max())
        if n_lo == n_hi == n_full:
            ax.set_title(f"{metric}")
        else:
            # short panel -> "(n=47*)", with the * resolved in a footnote under the figure
            starred.append(metric)
            ax.set_title(
                f"{metric}  (n={n_lo}*)" if n_lo == n_hi else f"{metric}  (n={n_lo}–{n_hi}*)"
            )
        ax.set_xlabel("layer")
        if i % ncols == 0:
            ax.set_ylabel("within-family Spearman ρ")
        if i == 1:
            ax.legend(frameon=False, fontsize=6, loc="best")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"{title} — within-family ρ across layers (mean ± 1 SD over families)",
        fontsize=11,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    if starred:
        # Never leave the asterisk undefined: use the caller's note, else name the missing families.
        note = footnote or (
            "; ".join(
                f"{m}: missing "
                + ", ".join(
                    sorted(
                        set(within.family.dropna().unique())
                        - set(within[(within.metric == m) & within.rho.notna()].family.unique())
                    )
                )
                for m in starred
            )
        )
        fig.text(
            0.01, -0.01, f"* {note}", ha="left", va="top", fontsize=7, style="italic", wrap=True
        )
    _save(fig, out_dir, f"within_family_vs_layer{stem_suffix}_band")


def _family_colors(families: list[str]) -> dict[str, str]:
    """
    One colour per family from the panel's canonical chemistry-block palette, so a family keeps its
    colour across every figure.
    """
    panel_colors = family_colors("human")
    if set(families) <= set(panel_colors):
        # line_safe: these are 1 px curves, not filled patches — the pale end of the
        # panel palette needs darkening to be visible at all (hue is preserved).
        return {f: acs.line_safe(panel_colors[f]) for f in families}
    return dict(zip(families, acs.categorical(len(families)), strict=True))


# Height of one publication panel row, in points. Two rows plus the family key keep the
# figure under the guide's 1,200 pt ceiling.
PUB_PANEL_H = 330.0

# Four columns keep the 48-family key compact.
PUB_KEY_COLS = 4


# Family names as a reader should see them. No mechanical rule works ("p450" -> P450, "gtpase" ->
# GTPase, "s_transferase" -> S-transferase), so only the exceptions are listed; the rest fall
# through to underscores-to-spaces plus a leading capital.
PUB_FAMILY_LABELS = {
    "adam_metallopeptidase": "ADAM metallopeptidase",
    "adamts_metallopeptidase": "ADAMTS metallopeptidase",
    "arf_gtpase": "Arf GTPase",
    "cxc_chemokine_receptor": "CXC chemokine receptor",
    "cytochrome_p450": "Cytochrome P450",
    "glutathione_s_transferase": "Glutathione S-transferase",
    "histone_deacetylase_classI": "Histone deacetylase class I",
    "histone_h4": "Histone H4",
    "m14_carboxypeptidase": "M14 carboxypeptidase",
    "nadph_oxidase": "NADPH oxidase",
    "p2y_receptor": "P2Y receptor",
    "rab_gtpase": "Rab GTPase",
    "ras_gtpases": "Ras GTPases",
    "rho_gtpase": "Rho GTPase",
    "steap_metalloreductase": "STEAP metalloreductase",
    "taste2_receptor": "Taste 2 receptor",
    "udp_glucuronosyltransferase": "UDP-glucuronosyltransferase",
}


def _pub_family_label(family: str) -> str:
    """A family key entry as a reader sees it — sentence case, correct capitalisation."""
    if family in PUB_FAMILY_LABELS:
        return PUB_FAMILY_LABELS[family]
    words = family.replace("_", " ")
    return words[:1].upper() + words[1:]


def plot_within(
    within: pd.DataFrame,
    out_dir: Path,
    title: str,
    exclude: list[str] | None = None,
    stem_suffix: str = "",
) -> None:
    if within.empty:
        print("  SKIP within figure: no within-family correlation CSVs found")
        return
    set_pub_style(title_size=9, tick_size=7)
    metrics = _within_metrics(within, exclude)
    if not metrics:
        print("  SKIP within figure: all within-family rho are NaN")
        return

    # stable colour per family across panels
    families = sorted(within["family"].dropna().unique())
    fam_color = _family_colors(families)
    sorted(within["layer"].unique())

    nrows, ncols = _balanced_grid(len(metrics))
    if pub.is_on():
        # A 48-family key down the right side would be ~960 pt tall against panels half that, and
        # eat a third of the width, so it goes underneath. Its height is measured, not guessed.
        fig, axes = plt.subplots(
            nrows, ncols, dpi=300, squeeze=False, figsize=pub.size(pub.FULL, PUB_PANEL_H * nrows)
        )
    else:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), dpi=300, squeeze=False
        )
    axes = axes.ravel()
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = within[within["metric"] == metric]
        n_fam = 0
        for fam in families:
            fs = sub[sub["family"] == fam].sort_values("layer")
            if fs["rho"].notna().sum() == 0:
                continue
            ax.plot(fs["layer"], fs["rho"], "-o", ms=2.5, lw=1.0, color=fam_color[fam], label=fam)
            n_fam += 1
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.6, ls="--")
        ax.set_title(f"{metric}  (n={n_fam} families)")
        ax.set_xlabel("layer")
        if i % ncols == 0:
            ax.set_ylabel("within-family Spearman ρ")

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    # one shared legend (families) to the right
    handles = [plt.Line2D([0], [0], color=fam_color[f], lw=2) for f in families]
    if pub.is_on():
        panels_h = PUB_PANEL_H * nrows
        _, key_h = pub.key_below(
            fig,
            handles,
            [_pub_family_label(f) for f in families],
            title="Gene family",
            width=pub.FULL,
            cols=PUB_KEY_COLS,
        )
        total_h = panels_h + key_h
        fig.set_size_inches(pub.FULL / 72.0, total_h / 72.0)
        pub.tight(fig, bottom=key_h)
    else:
        fig.legend(
            handles,
            families,
            frameon=False,
            fontsize=6,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title="family",
        )
        fig.suptitle(
            f"{title} — within-family ρ across layers (per family)",
            fontsize=11,
            fontweight="bold",
            y=1.0,
        )
        fig.tight_layout(rect=(0, 0, 0.88, 1))
    _save(fig, out_dir, f"within_family_vs_layer{stem_suffix}")


# Layer-independent ground truth, stored once. Every layer dir re-emits these, but they are
# sequence/annotation-derived and do not depend on the embedding layer. The per-layer SCORE files
# are excluded — they are results, not baselines.
# BETWEEN-family ground-truth distances (from the run dirs):
BETWEEN_FILES = [
    "between_family_baseline_distances.csv",  # long-form: every baseline × every pair
    "pfam_jsd_distances.csv",  # F×F Pfam HMM JSD (homology)
]
BETWEEN_GLOB = "betweenfam_*_distances.csv"  # F×F per-baseline matrices
# WITHIN-family gene-level ground-truth distance matrices (from the run dirs):
WITHIN_GENELEVEL_FILES = [
    "kmer_distance.npy",  # gene×gene k-mer distance (within k-mer baseline)
    "taxonomic_distance.npy",  # gene×gene taxonomy distance (within taxonomy baseline)
]
# WITHIN-family alignment distances live in a project-wide cache, not the run dirs.
# <fam>.npy is renamed to <fam>.patristic.npy on copy.
PATRISTIC_CACHE_FILES = [
    (".npy", ".patristic.npy"),
    (".ids.json", ".ids.json"),
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _copy_layer_invariant(
    names: list[str],
    glob_pat: str | None,
    layers: list[tuple[int, Path]],
    dest: Path,
    manifest: list[str],
) -> int:
    """
    Copy each name into dest from the highest layer that has it (the donor lowest layer is the lone
    float-repr outlier).
    """
    all_names = list(names)
    if glob_pat:
        for _, run_dir in layers:
            all_names += [p.name for p in run_dir.glob(glob_pat)]
    copied = 0
    for name in sorted(set(all_names)):
        if name.startswith("betweenfam_") and name.endswith("_distances.csv"):
            baseline = name[len("betweenfam_") : -len("_distances.csv")]
            if baseline not in ACTIVE_BETWEEN:
                continue
        present = [
            (layer, run_dir / name) for layer, run_dir in layers if (run_dir / name).exists()
        ]
        if not present:
            continue
        hashes = {_md5(p) for _, p in present}
        if len(hashes) > 2:
            print(
                f"  WARN {name}: {len(hashes)} distinct hashes across "
                f"{len(present)} layers — NOT layer-independent, skipping"
            )
            manifest.append(f"{dest.name}/{name}\t{len(present)}\t{len(hashes)}\tSKIPPED (varies)")
            continue
        src_layer, src = max(present, key=lambda t: t[0])
        shutil.copy2(src, dest / name)
        copied += 1
        manifest.append(f"{dest.name}/{name}\t{len(present)}\t{len(hashes)}\tblocks{src_layer}")
    return copied


def _resolve_patristic_cache(glob_pattern: str, override: str | None) -> Path | None:
    """Which alignment-distance cache feeds the within-family baselines — EXPLICIT ONLY."""
    return Path(override) if override else None


def collect_baselines(
    layers: list[tuple[int, Path]], within: pd.DataFrame, out_dir: Path, patristic_cache: Path
) -> None:
    """Store the layer-independent ground truth once under out_dir/baselines/."""
    # Only read completed layers, so this is safe to run against a live sweep.
    layers = [(lay, d) for lay, d in layers if (d / "between_family_baseline_scores.csv").exists()]
    if not layers:
        print("  SKIP baselines: no completed layers yet")
        return

    bdir = out_dir / "baselines"
    between_dir = bdir / "between_family"
    within_dir = bdir / "within_family"
    between_dir.mkdir(parents=True, exist_ok=True)
    within_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        "# Layer-independent ground-truth baselines (stored once).",
        "# Copied by scripts/layer_sweep_summary.py from the all-layer sweep;",
        "# identical across every layer (only the per-layer geodesic differs).",
        "# columns: dir/file  n_layers_present  n_distinct_hashes  source",
        "",
    ]

    n_btw = _copy_layer_invariant(BETWEEN_FILES, BETWEEN_GLOB, layers, between_dir, manifest)
    n_win = _copy_layer_invariant(WITHIN_GENELEVEL_FILES, None, layers, within_dir, manifest)

    # per-family alignment distances from the project cache (not in run dirs)
    families = sorted(within["family"].dropna().unique()) if not within.empty else []
    n_cache = 0
    if patristic_cache is None:
        print(
            "  NOTE per-family patristic matrices not collected "
            "(pass --patristic-cache DIR only if that cache belongs to THIS panel)"
        )
    elif not patristic_cache.is_dir():
        print(
            f"  NOTE within per-family cache absent: {patristic_cache} "
            f"(skipping patristic matrices)"
        )
    else:
        for fam in families:
            for src_suffix, dst_suffix in PATRISTIC_CACHE_FILES:
                src = patristic_cache / f"{fam}{src_suffix}"
                if src.exists():
                    shutil.copy2(src, within_dir / f"{fam}{dst_suffix}")
                    n_cache += 1
        manifest.append(
            f"within_family/<fam>.patristic.npy + .ids.json"
            f"\t-\t-\t{patristic_cache} ({len(families)} families)"
        )

    (bdir / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(
        f"  collected baselines → {bdir}/ "
        f"(between_family: {n_btw}, within_family: {n_win} gene-level + "
        f"{n_cache} per-family cache files; + MANIFEST.txt)"
    )


def write_provenance(
    layers: list[tuple[int, Path]],
    between: pd.DataFrame,
    within: pd.DataFrame,
    out_dir: Path,
    glob_pat: str,
    title: str,
    exclude_within: list[str] | None = None,
    exclude_between: list[str] | None = None,
    stem_suffix: str = "",
    lead_per_baseline: bool = False,
) -> None:
    """Record where every number in this folder came from, as SOURCE.md."""
    root = {p.parent for _, p in layers}
    n_between = (
        between.groupby("layer")["baseline"].nunique()
        if not between.empty
        else pd.Series(dtype=int)
    )
    fam_per_layer, mtimes = {}, {}
    for layer, run_dir in layers:
        f = run_dir / "between_family_baseline_scores.csv"
        if f.exists():
            mtimes[layer] = f.stat().st_mtime
        cen = next(iter(run_dir.glob("*_centroid_distances.csv")), None)
        if cen is not None:
            fam_per_layer[layer] = len(pd.read_csv(cen, index_col=0))

    lines = [
        f"# {title}",
        "",
        "Cross-layer roll-up. Every value in the CSVs and figures here was read out of the",
        "per-layer run dirs below — this folder derives from them and holds no primary results.",
        "Written by `scripts/layer_sweep_summary.py`; re-running it regenerates this file.",
        "",
        "## Source",
        "",
        f"- glob: `{glob_pat}`",
        f"- per-layer results: {', '.join(f'`{r}/`' for r in sorted(map(str, root)))}",
        f"- layers: {len(layers)} ({layers[0][0]}..{layers[-1][0]}), dirs "
        f"`{layers[0][1].name}` .. `{layers[-1][1].name}`",
    ]
    if fam_per_layer:
        counts = sorted(set(fam_per_layer.values()))
        lines.append(
            f"- families (centroid matrix): {counts[0]}"
            if len(counts) == 1
            else f"- families (centroid matrix): **MIXED** {counts} across layers"
        )
    if not within.empty:
        lines.append(
            "- within-family metrics: "
            + ", ".join(
                f"{m} ({within[within.metric == m].family.nunique()} families)"
                for m in sorted(within["metric"].unique())
            )
        )
    if exclude_between:
        lines += [
            "",
            f"- **between-family baselines excluded from the figure**: "
            f"{', '.join(repr(b) for b in exclude_between)}. They are still in "
            f"`between_axis_vs_layer{stem_suffix}.csv` and in the per-layer sources — only "
            f"the panels are suppressed, and any axis left only partially represented is "
            f"dropped from the lead axis-means panel rather than averaged over a subset. "
            f"Re-run with the same `--exclude-between` to reproduce this figure.",
        ]
    if exclude_within:
        lines += [
            "",
            f"- **within panels excluded from the figure**: "
            f"{', '.join(repr(m) for m in exclude_within)}. The metric is still in "
            f"`within_family_vs_layer.csv` and in the per-layer sources — only the panel "
            f"is suppressed. Re-run with the same `--exclude-within` to reproduce this "
            f"figure; omit it to get every panel back.",
        ]
    if lead_per_baseline:
        lines += [
            "",
            "- **lead panel**: one line per between-family baseline (colour = axis, "
            "marker shape = baseline), NOT the per-axis mean. Reproduce with "
            "`--lead-per-baseline`; omit it for the axis-mean lead panel.",
        ]
    lines += ["", "## Per-layer files read", "", "| file | supplies |", "|---|---|"]
    if not between.empty:
        lines.append(
            "| `between_family_baseline_scores.csv` | between_axis_vs_layer.{csv,png,pdf} |"
        )
    present = set(within["metric"].unique()) if not within.empty else set()
    for label, filenames, _ in WITHIN_SPECS:  # only the specs that actually resolved on disk
        if label in present:
            lines.append(f"| {' or '.join(f'`{f}`' for f in filenames)} | within: {label} |")

    warns = []
    if len(set(fam_per_layer.values())) > 1:
        odd = {
            L: n
            for L, n in sorted(fam_per_layer.items())
            if n != max(set(fam_per_layer.values()), key=list(fam_per_layer.values()).count)
        }
        warns.append(
            f"Family count is not constant across layers — odd layers: {odd}. "
            "The curves mix panel sizes and are not comparable layer-to-layer."
        )
    if len(mtimes) > 1 and max(mtimes.values()) - min(mtimes.values()) > 86400:
        oldest, newest = min(mtimes, key=mtimes.get), max(mtimes, key=mtimes.get)
        span = (max(mtimes.values()) - min(mtimes.values())) / 86400
        warns.append(
            f"Between-family source files span {span:.1f} days "
            f"(oldest layer {oldest}, newest layer {newest}) — if the sweep ran in one "
            "batch, the old ones are stale leftovers from a step that failed."
        )
    if not n_between.empty and n_between.nunique() > 1:
        warns.append(f"Baseline count varies by layer: {n_between.value_counts().to_dict()}.")
    if warns:
        lines += ["", "## ⚠ Consistency warnings", ""] + [f"- {w}" for w in warns]

    name = f"SOURCE{stem_suffix}.md"
    (out_dir / name).write_text("\n".join(lines) + "\n")
    print(f"  saved {out_dir}/{name}" + ("  ⚠ with consistency warnings" if warns else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--glob",
        required=True,
        help="glob for the sweep run dirs, e.g. 'results/2026-07-01_evo2-human-panel/blocks*'",
    )
    ap.add_argument("--out-dir", required=True, help="where to write the summary figures")
    ap.add_argument("--title", default="", help="figure title prefix")
    ap.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip collecting the layer-independent baselines into baselines/",
    )
    ap.add_argument(
        "--footnote",
        default=None,
        metavar="TEXT",
        help="text for the '*' under the band figure, marking any within panel scored "
        "on fewer families than the rest. Without it the footnote just names the "
        "missing families — factual, but it cannot say WHY, which is usually the "
        "part worth recording.",
    )
    ap.add_argument(
        "--between-scores",
        default="between_family_baseline_scores.csv",
        metavar="FILE",
        help="per-layer between-family scores file. Use between_family_ot_scores.csv "
        "with --between-approach to draw the figure from an OT metric instead.",
    )
    ap.add_argument(
        "--between-approach",
        default=None,
        metavar="NAME",
        help="filter --between-scores to one approach (geodesic / wasserstein / "
        "fgw_alpha0.25). Only meaningful for a file that has an 'approach' column.",
    )
    ap.add_argument(
        "--within-file-suffix",
        default="",
        metavar="SUFFIX",
        help="read within_family_*<SUFFIX>.csv instead of the plain tables — "
        "'_angular' selects the graph-free within-family scoring.",
    )
    ap.add_argument(
        "--kmer-k",
        type=int,
        default=None,
        metavar="K",
        help="annotate the k-mer panels as k-mer(k=K). The k is NOT recorded in the "
        "score CSVs (the column is just spearman_geodesic_kmer), and it differs "
        "between panels, so it must be asserted rather than inferred. The mammal "
        "arms use k=6 (mammal_between.kmer_between / mammal_score, both k=6).",
    )
    ap.add_argument(
        "--exclude-within",
        nargs="*",
        default=[],
        metavar="LABEL",
        help="within-family baseline LABELs to drop from the figure (exact WITHIN_SPECS "
        "labels). The CSV keeps every metric — this only "
        "controls which panels are drawn. Recorded in SOURCE.md so the figure is "
        "reproducible. Use when two baselines are near-redundant and showing both "
        "overstates how many independent baselines agree.",
    )
    ap.add_argument(
        "--exclude-between",
        nargs="*",
        default=[],
        metavar="BASELINE",
        help="between-family BASELINE keys to drop from the between figure (exact "
        "BETWEEN_ORDER keys). The CSV keeps every "
        "baseline. An axis left only partially represented is dropped from the "
        "lead axis-means panel rather than averaged over a subset of itself.",
    )
    ap.add_argument(
        "--within-band",
        action="store_true",
        help="ALSO write within_family_vs_layer[SUFFIX]_band.{png,pdf}: the per-family "
        "lines collapsed to mean ± 1 SD across families, one panel per metric. "
        "The per-family figure is still written — the band is a companion, not a "
        "replacement, since ±1 SD hides families outside it.",
    )
    ap.add_argument(
        "--lead-per-baseline",
        action="store_true",
        help="draw the between figure's lead panel as one line per BASELINE (colour = "
        "axis, marker shape = baseline) instead of one mean line per axis. "
        "Use when baselines inside one axis disagree — e.g. the gc_content and kmer "
        "composition controls, whose mean hides which of the two the geometry "
        "actually tracks.",
    )
    ap.add_argument(
        "--stem-suffix",
        default="",
        metavar="SUFFIX",
        help="append SUFFIX to every output filename (e.g. '_v2'), so a restricted-"
        "baseline variant lands beside the full figure instead of overwriting it. "
        "Applies to the figures, the tidy CSVs and SOURCE.md.",
    )
    ap.add_argument(
        "--patristic-cache",
        default=None,
        help="per-family alignment-distance cache dir to archive into baselines/. NOT "
        "inferred — pass it ONLY when that cache belongs to this panel, since the "
        "cache is keyed by family name and names collide across panels "
        "(data/cache/human_patristic = the human panel). "
        "Omitted = nothing archived.",
    )
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render at PUBLICATION geometry instead of the compact diagnostic scale: "
        "an exact 1,000 pt panel, the style guide's 15 pt type with monospaced "
        "numerals, panels stacked rather than widened, and no in-artwork title "
        "(the caption carries it). Writes into <out-dir>/pub/ so the diagnostic "
        "figures the results docs link to are left alone.",
    )
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    layers = discover_layers(args.glob)
    if not layers:
        sys.exit(f"No layer run dirs matched {args.glob!r}")
    title = args.title or args.glob
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{title}: {len(layers)} layers [{layers[0][0]}..{layers[-1][0]}]")

    if args.kmer_k:
        # k is not in the score CSVs and differs between panels, so it is asserted on the command
        # line. Patch SPECS, not the loaded frame: panel order is matched against these labels.
        for i, (lab, files, col) in enumerate(WITHIN_SPECS):
            if lab.startswith("k-mer"):
                WITHIN_SPECS[i] = (
                    lab.replace(")", f", k={args.kmer_k})")
                    if lab.endswith(")")
                    else f"{lab} (k={args.kmer_k})",
                    files,
                    col,
                )

    between = load_between(layers, args.between_scores, args.between_approach)
    within = load_within(layers, args.within_file_suffix)
    sfx = args.stem_suffix
    if not between.empty:
        between.to_csv(out_dir / f"between_axis_vs_layer{sfx}.csv", index=False)
    if not within.empty:
        within.to_csv(out_dir / f"within_family_vs_layer{sfx}.csv", index=False)
    plot_between(
        between,
        out_dir,
        title,
        args.exclude_between,
        sfx,
        args.lead_per_baseline,
        kmer_k=args.kmer_k,
    )
    plot_within(within, out_dir, title, args.exclude_within, sfx)
    if args.within_band:
        plot_within_band(within, out_dir, title, args.exclude_within, sfx, args.footnote)
    write_provenance(
        layers,
        between,
        within,
        out_dir,
        args.glob,
        title,
        args.exclude_within,
        args.exclude_between,
        sfx,
        args.lead_per_baseline,
    )
    if not args.no_baselines:
        cache = _resolve_patristic_cache(args.glob, args.patristic_cache)
        collect_baselines(layers, within, out_dir, cache)


if __name__ == "__main__":
    main()
