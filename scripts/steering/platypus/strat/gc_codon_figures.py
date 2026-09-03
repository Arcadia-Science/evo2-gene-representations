"""Figure 10: GC by codon position, beside overall GC against both species' reference lines.

CPU only, built from artefacts the run already has. Two modes:

    --emit-tables   compute <run>/figures/{12b_composition_gene_means,12c_reference_windows,
                    13b_codon_substitution_stats}.csv and exit. build_figure_data.py reads the
                    first and third, so a full --run must do this BEFORE building figure_data/.
    (default)       render the figure, from the tracked tables with --from-figure-data or from
                    the cached tables above without it.

Five further diagnostic figures (this script's internal numbering 12, 13, 15, 16, 17) lived here
until 2026-09-03. None was in the pub, and figures 16/17 -- the block-24 vs block-27 comparison --
were the reason the shared cache spanned two intervention layers, which leaked block-24 rows into
the tracked Figure 10 tables. They are archived, not deleted, under
deprecated/gc-codon-diagnostics-2026-09-03/.
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "steering"))  # alignment_metrics
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from alignment_metrics import read_fasta  # noqa: E402

import figure_data  # noqa: E402

# Set from --from-figure-data in main(); the cached-table helpers are called from several figures.
FROM_FIGURE_DATA = False
from plot_utils import set_pub_style  # noqa: E402

INK, WARM, PLUM, GREY = (acs.SERIES_PRIMARY, acs.SERIES_NULL, acs.SERIES_THIRD, acs.SERIES_MUTED)
PLAT_C, HUM_C = acs.STEER_COLORS["target"], acs.STEER_COLORS["source"]
BASES = set("ACGT")
STOPS = {"TAA", "TAG", "TGA"}
# The dose ladder, in the order it is plotted. `unsteered` is alpha = 0.
# The reported intervention layer. Conditions from it carry no suffix; every other layer's carry
# `_L<n>` (see layer_suffix), which is how a row's provenance is recovered from its condition name.
STEER_LAYER = 27

DOSE = ["unsteered", "add_a0.5", "add_a1.0", "add_a2.0", "add_a3.0", "add_a4.0"]
EXTRA = ["random_a1.0", "add_gc_removed_a1.0"]
# Tick labels for the rungs of DOSE, in the same order.
DOSE_LABELS = ["0\n(unst)", "0.5", "1", "2", "3", "4"]
# Spelled out for the roomier two-panel figures, where the abbreviation is not worth its ambiguity.
DOSE_LABELS_LONG = ["0\n(unsteered)", "0.5", "1", "2", "3", "4"]

# Figure 14 uses two panels across a 1,000-point page.
PUB_FIG_H = 480.0


def layer_suffix(layer: int) -> str:
    """The stage-4 condition suffix for a steering layer."""
    return "" if layer == 27 else f"_L{layer}"


def figures_dir(run: Path, layer: int) -> Path:
    """Per-layer output directory: <run>/figures_24, <run>/figures_27, ..."""
    return run / f"figures_{layer}"


def cache_dir(run: Path) -> Path:
    d = run / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d




def complete_conditions(
    table: pd.DataFrame, wanted: list[str], what: str, min_frac: float = 0.95
) -> list[str]:
    """Keep only conditions the table covers for (nearly) every gene it covers at all."""
    n = table.groupby("condition").gene.nunique()
    full = int(n.max()) if len(n) else 0
    keep = [c for c in wanted if int(n.get(c, 0)) >= min_frac * full]
    thin = [(c, int(n.get(c, 0))) for c in wanted if c not in keep and int(n.get(c, 0)) > 0]
    absent = [c for c in wanted if int(n.get(c, 0)) == 0]
    if thin:
        print(
            f"  {what}: DROPPED still-generating "
            + ", ".join(f"{c} ({k}/{full} genes)" for c, k in thin)
        )
    if absent:
        print(f"  {what}: absent {', '.join(absent)}")
    return keep


def ladder_and_labels(
    table: pd.DataFrame, suffix: str, what: str, labels: list[str] = DOSE_LABELS
) -> tuple[list[str], list[str]]:
    """The complete rungs of a layer's dose ladder, with their matching tick labels."""
    full = dose_ladder(suffix)
    keep = complete_conditions(table, full, what)
    return keep, [labels[full.index(c)] for c in keep]


def dose_ladder(suffix: str = "") -> list[str]:
    """The dose ladder for one steering layer, named as stage 4 writes it."""
    return [c if c == "unsteered" else c + suffix for c in DOSE]


def extra_arms(suffix: str = "") -> list[str]:
    return [c + suffix for c in EXTRA]


# Every condition the cached tables should cover: the reported block-27 arm and nothing else.
# It used to include the `_L24` ladder as well, because the archived cross-layer figures 16/17
# compared the two layers -- but these tables are also build_figure_data's source for Figure 10,
# so the tracked publication artifact inherited block-24 rows that no figure plots. Conditions
# that do not exist yet simply do not appear; nothing downstream assumes they are there.
KEEP = list(dict.fromkeys(DOSE + EXTRA + ["add_own"]))


def _only_kept(df: pd.DataFrame, src: Path) -> pd.DataFrame:
    """Drop conditions outside KEEP from a table read back off disk.

    The build path already filters, but a cache written before KEEP narrowed to the reported layer
    still holds `_L24` rows and the read path handed them through unchanged -- which is how block-24
    conditions reached the tracked Figure 10 tables. Filtering here makes a stale cache self-correct
    instead of silently widening the artifact.
    """
    extra = sorted(set(df.condition) - set(KEEP))
    if extra:
        print(f"  ({src.name}: dropping {len(extra)} conditions outside this layer: {extra})")
        df = df[df.condition.isin(KEEP)].reset_index(drop=True)
    return df


def gc_stats(seq: str) -> tuple[float, float]:
    """(overall GC, GC3) as percentages in the sequence's own frame."""
    s = [c for c in seq.upper() if c in BASES]
    if not s:
        return np.nan, np.nan
    gc = 100.0 * sum(1 for c in s if c in "GC") / len(s)
    third = [seq[i + 2].upper() for i in range(0, len(seq) - 2, 3) if seq[i + 2].upper() in BASES]
    gc3 = 100.0 * sum(1 for c in third if c in "GC") / len(third) if third else np.nan
    return gc, gc3


def _sense_codons() -> list[str]:
    return [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT" if a + b + c not in STOPS]


SENSE = _sense_codons()


def codon_freq(seq: str) -> np.ndarray:
    """L1-normalised frequency over the 61 sense codons, in a fixed order."""
    codons = [seq[i : i + 3].upper() for i in range(0, len(seq) - 2, 3)]
    c = Counter(x for x in codons if len(x) == 3 and set(x) <= BASES and x not in STOPS)
    v = np.array([c.get(k, 0) for k in SENSE], dtype=float)
    t = v.sum()
    return v / t if t else v


def _codon_table() -> dict[str, str]:
    from Bio.Seq import Seq

    return {
        c: str(Seq(c).translate())
        for c in (a + b + d for a in "ACGT" for b in "ACGT" for d in "ACGT")
    }


AA = _codon_table()


def _syn_sites(codon: str) -> float:
    """NG86 synonymous-site count for one codon: per position, the fraction of the 3 possible
    substitutions that leave the amino acid unchanged, summed over the 3 positions."""
    aa = AA[codon]
    s = 0.0
    for i in range(3):
        same = sum(1 for b in "ACGT" if b != codon[i] and AA[codon[:i] + b + codon[i + 1 :]] == aa)
        s += same / 3.0
    return s


SYN_SITES = {c: _syn_sites(c) for c in AA if AA[c] != "*"}


def gc_by_position(seq: str) -> tuple[float, float, float]:
    """GC at codon positions 1, 2, 3 separately (percentages), read in the sequence's own frame."""
    out = []
    for off in (0, 1, 2):
        b = [
            seq[i].upper()
            for i in range(off, len(seq) - (len(seq) - off) % 3, 3)
            if seq[i].upper() in BASES
        ]
        out.append(100.0 * sum(1 for c in b if c in "GC") / len(b) if b else np.nan)
    return tuple(out)


# Fewest scorable codons a generation must contribute before its pN/pS is estimable at all. The
# ratio is a proportion over S and N sites, so a handful of codons gives a value that swings on a
# single substitution. Was 20 until 2026-09-02, which excluded COX17 entirely: its 54 bp of
# continuation is 18 codons. Lowered to keep the gene, a deliberate precision-for-coverage trade
# rather than a free win -- COX17's pN/pS is the least reliable point in the codon table, and
# `n_codons` is carried per cell so it can be weighted or filtered. Verified surgical: at 15 the
# only cells that appear are COX17's 17, and every pre-existing cell is bit-identical.
MIN_CODONS = 15


def codon_diff_stats(gen: str, target: str, min_codons: int = MIN_CODONS) -> dict:
    """Synonymous / nonsynonymous differences between a generation and the platypus target."""
    from alignment_metrics import nt_aligner

    if not gen or not target:
        return {}
    a = nt_aligner().align(gen, target)[0]
    g_aln, t_aln = str(a[0]), str(a[1])
    # walk in target coordinates, collecting (gen_codon, target_codon) for clean triplets
    tpos, gcod, tcod = 0, [], []
    buf_g, buf_t = [], []
    for gc, tc in zip(g_aln, t_aln, strict=False):
        if tc == "-":
            continue  # insertion in the generation: not a target codon column
        buf_t.append(tc)
        buf_g.append(gc)
        tpos += 1
        if len(buf_t) == 3:
            g3, t3 = "".join(buf_g), "".join(buf_t)
            if set(g3) <= BASES and set(t3) <= BASES:
                gcod.append(g3)
                tcod.append(t3)
            buf_g, buf_t = [], []
    S = N = Sd = Nd = 0.0
    n_cod = 0
    for g3, t3 in zip(gcod, tcod, strict=False):
        if t3 not in SYN_SITES or AA[g3] == "*":
            continue
        n_cod += 1
        s = SYN_SITES[t3]
        S += s
        N += 3.0 - s
        if g3 == t3:
            continue
        if sum(1 for x, y in zip(g3, t3, strict=False) if x != y) == 1:
            if AA[g3] == AA[t3]:
                Sd += 1
            else:
                Nd += 1
    if n_cod < min_codons:
        return {}
    pS = Sd / S if S > 0 else np.nan
    pN = Nd / N if N > 0 else np.nan
    # Numerator and denominator must cover the same codon pairs — both single-position differences.
    # `syn_site_frac` is S/(S+N), the value this takes when pN = pS, and is the right reference
    # line;
    # not 1/3, since only ~70% of wobble changes are silent.
    return {
        "n_codons": n_cod,
        "syn_per_100": 100.0 * Sd / n_cod,
        "nonsyn_per_100": 100.0 * Nd / n_cod,
        "frac_diff_syn": (Sd / (Sd + Nd)) if (Sd + Nd) else np.nan,
        "syn_site_frac": S / (S + N) if (S + N) else np.nan,
        "pS": pS,
        "pN": pN,
        "pN_over_pS": (pN / pS) if (pS and pS > 0) else np.nan,
    }


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two codon-frequency vectors."""
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = (a > 0) & (b > 0)
        return float(np.sum(a[ok] * np.log2(a[ok] / b[ok])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# composition of the generations (shared by figures 12, 14, 15)
def composition_table(
    run: Path, arm: str, out: Path, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-gene-per-condition generation composition, per-gene reference windows)."""
    if FROM_FIGURE_DATA:
        return (
            figure_data.table("exp3_generation_composition"),
            figure_data.table("exp3_generation_reference_windows").set_index("gene"),
        )
    gc_cache, ref_cache = out / "12b_composition_gene_means.csv", out / "12c_reference_windows.csv"
    if gc_cache.exists() and ref_cache.exists() and not refresh:
        print(f"  (reusing {gc_cache.name}; pass --refresh-composition to recompute)")
        return _only_kept(pd.read_csv(gc_cache), gc_cache), pd.read_csv(ref_cache).set_index("gene")

    ch = read_fasta(run / "stage1" / "cds_human.fasta", uppercase=False)
    cp = read_fasta(run / "stage1" / "cds_platypus.fasta", uppercase=False)
    plan = pd.read_csv(run / arm / "scoring_plan.csv")
    plan = plan[plan.usable].set_index("gene")

    # Reference composition is measured on the same window that was scored: a whole-CDS GC is a
    # different quantity from the generations' GC.
    ref = []
    for gene, r in plan.iterrows():
        if gene not in ch or gene not in cp:
            continue
        nt = int(r.n_tokens)
        h = ch[gene][int(r.off_h) + 90 :][:nt]
        p = cp[gene][int(r.off_p) + 90 :][:nt]
        gh, g3h = gc_stats(h)
        gp, g3p = gc_stats(p)
        ref.append(
            {
                "gene": gene,
                "gc_human": gh,
                "gc3_human": g3h,
                "gc_platypus": gp,
                "gc3_platypus": g3p,
                "cf_human": codon_freq(h),
                "cf_platypus": codon_freq(p),
            }
        )
    R = pd.DataFrame(ref).set_index("gene")
    print(f"  reference windows: {len(R)} genes")

    rows = []
    with gzip.open(run / arm / "generations.jsonl.gz", "rt") as fh:
        try:
            for line in fh:
                r = json.loads(line)
                if r["condition"] not in KEEP or r["gene"] not in R.index:
                    continue
                gc, gc3 = gc_stats(r["seq"])
                cf = codon_freq(r["seq"])
                rows.append(
                    {
                        "gene": r["gene"],
                        "condition": r["condition"],
                        "gc": gc,
                        "gc3": gc3,
                        "jsd_plat": jsd(cf, R.at[r["gene"], "cf_platypus"]),
                        "jsd_hum": jsd(cf, R.at[r["gene"], "cf_human"]),
                    }
                )
        except EOFError:
            print("  NOTE generations.jsonl.gz truncated; using what survived")
    G = pd.DataFrame(rows)
    # gene means first: samples in a cell share a prompt, so the gene is the unit
    g = G.groupby(["gene", "condition"]).mean(numeric_only=True).reset_index()
    print(
        f"  generations: {len(G)} samples over {g.gene.nunique()} genes, "
        f"{g.condition.nunique()} conditions"
    )

    g.to_csv(gc_cache, index=False)
    R.drop(columns=["cf_human", "cf_platypus"]).to_csv(ref_cache)
    print(f"  [wrote] {gc_cache}")
    return g, R


# panel painters -- one implementation per panel, shared by every figure that shows it


def _dose_axis(ax, n: int, labels: list[str] = DOSE_LABELS, xlabel: str = "α") -> None:
    ax.set_xticks(range(n), labels[:n], fontsize=7)
    ax.set_xlabel(xlabel)


# Each painter takes `conditions` (stage-4 names in x order) and `labels`. A dose ladder gives a
# dose-response panel; an arbitrary set of arms gives a categorical comparison. `connect` is
# meaningful along a dose axis and misleading across unordered categories.
def panel_gc(
    ax,
    g: pd.DataFrame,
    R: pd.DataFrame,
    col: str,
    labels: list[str] = DOSE_LABELS,
    conditions: list[str] | None = None,
    xlabel: str = "α",
) -> None:
    """Boxplots of overall GC (col='gc') or GC3 (col='gc3'), against the platypus and human
    reference lines measured on the same scored window."""
    conditions = DOSE if conditions is None else conditions
    ylab, title = {
        "gc": ("GC (%)", "Overall GC content"),
        "gc3": ("GC3 (%)", "GC at the third codon position\n(the axis `gc_removed` projects out)"),
    }[col]
    data = [g.loc[g.condition == c, col].dropna().to_numpy() for c in conditions]
    bp = ax.boxplot(
        data,
        positions=range(len(conditions)),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=acs.apc.white, lw=1.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(INK)
        patch.set_edgecolor("none")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(acs.SERIES_MUTED)
    for key, c, nm, side in (
        (f"{col}_platypus", PLAT_C, "platypus CDS", "bottom"),
        (f"{col}_human", HUM_C, "human CDS", "top"),
    ):
        m = float(R[key].mean())
        ax.axhline(m, color=c, ls="--", lw=1.6, zorder=0)
        if pub.is_on():
            # The references are ~4 pp apart, so at 15 pt a two-line label collides. One line each,
            # on the outer side of its own rule.
            ax.text(len(conditions) - 0.45, m, f" {nm} {m:.1f}%", color=c, va=side, ha="left")
        else:
            ax.text(
                len(conditions) - 0.45,
                m,
                f" {nm}\n {m:.1f}%",
                color=c,
                fontsize=6.6,
                va="center",
                ha="left",
            )
    _dose_axis(ax, len(conditions), labels, xlabel)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8.5)
    # Headroom for the reference labels, sized for the narrowest panel this is drawn in.
    ax.set_xlim(-0.6, len(conditions) + 1.5)




def panel_gc_by_position(
    ax,
    C: pd.DataFrame,
    present: list[str],
    labels: list[str] = DOSE_LABELS,
    annotate_wobble: bool = True,
    xlabel: str = "α",
    connect: bool = True,
) -> None:
    """GC at codon positions 1/2/3 separately over the dose ladder."""
    xs = range(len(present))
    # GC1/GC2/GC3 are an ordered series -> blue_shades light -> dark.
    for col, c, nm in (
        ("gc1", acs.apc.vital, "GC1"),
        ("gc2", acs.apc.aegean, "GC2"),
        ("gc3", INK, "GC3 (wobble)" if annotate_wobble else "GC3"),
    ):
        m = [C.loc[C.condition == cc, col].mean() for cc in present]
        ax.plot(xs, m, color=c, lw=2.0, ls="-" if connect else "none", marker="o", ms=4.5, label=nm)
    _dose_axis(ax, len(present), labels, xlabel)
    ax.set_ylabel("GC (%)")
    ax.set_title(
        "GC by codon position"
        + ("\n(wobble vs the two coding positions)" if annotate_wobble else ""),
        fontsize=8.5,
    )
    ax.legend(fontsize=7, frameon=False)








# codon-level substitution stats (row 2 of figure 13)
_CS: dict = {}


def _cs_init(payload: dict) -> None:
    _CS.update(payload)


def _cs_one_gene(gene: str) -> list[dict]:
    tgt = _CS["target"][gene]
    out = []
    for rec in _CS["gens"].get(gene, []):
        g1, g2, g3 = gc_by_position(rec["seq"])
        st = codon_diff_stats(rec["seq"], tgt, _CS.get("min_codons", MIN_CODONS))
        if not st:
            continue
        out.append(
            {
                "gene": gene,
                "condition": rec["condition"],
                "sample": rec["sample"],
                "gc1": g1,
                "gc2": g2,
                "gc3": g3,
                **st,
            }
        )
    return out


def codon_stats(
    run: Path, arm: str, out: Path, workers: int = 8, refresh: bool = False
) -> pd.DataFrame:
    """Per-(gene, condition) codon-level stats, cached — the alignments cost a few minutes."""
    if FROM_FIGURE_DATA:
        return figure_data.table("exp3_codon_substitutions")
    cache = out / "13b_codon_substitution_stats.csv"
    if cache.exists() and not refresh:
        print(f"  (reusing {cache.name}; pass --refresh-codons to recompute)")
        return _only_kept(pd.read_csv(cache), cache)

    cp = read_fasta(run / "stage1" / "cds_platypus.fasta", uppercase=False)
    plan = pd.read_csv(run / arm / "scoring_plan.csv")
    plan = plan[plan.usable].set_index("gene")
    target = {}
    for gene, r in plan.iterrows():
        if gene in cp:
            target[gene] = cp[gene][int(r.off_p) + 90 :][: int(r.n_tokens)]

    gens: dict[str, list[dict]] = {}
    with gzip.open(run / arm / "generations.jsonl.gz", "rt") as fh:
        try:
            for line in fh:
                r = json.loads(line)
                if r["condition"] in KEEP and r["gene"] in target:
                    gens.setdefault(r["gene"], []).append(r)
        except EOFError:
            print("  NOTE generations.jsonl.gz truncated; using what survived")
    genes = sorted(gens)
    n = sum(len(v) for v in gens.values())
    print(
        f"  codon stats over {n} generations, {len(genes)} genes "
        f"(one alignment each; {workers} workers)"
    )

    from multiprocessing import Pool

    rows = []
    with Pool(
        workers,
        initializer=_cs_init,
        initargs=({"target": target, "gens": gens, "min_codons": MIN_CODONS},),
    ) as pool:
        for i, res in enumerate(pool.imap_unordered(_cs_one_gene, genes, chunksize=1), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"    {i}/{len(genes)} genes", flush=True)
    per_sample = pd.DataFrame(rows)
    g = per_sample.groupby(["gene", "condition"]).mean(numeric_only=True).reset_index()
    g.to_csv(cache, index=False)
    print(f"  [wrote] {cache}")
    return g




# figures 14 and 15 -- two-panel regroupings of panels already defined above
def fig14(
    run: Path,
    arm: str,
    out: Path,
    workers: int = 8,
    refresh_codons: bool = False,
    refresh_composition: bool = False,
    layer: int = 27,
) -> None:
    suffix = layer_suffix(layer)
    """GC by codon position (from 13) beside GC3 vs both species' reference lines (from 12)."""
    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)
    g, R = composition_table(run, arm, cache_dir(run), refresh=refresh_composition)
    present, xlab = ladder_and_labels(C, suffix, "codon table", DOSE_LABELS_LONG)
    ladder, lab = ladder_and_labels(g, suffix, "composition table", DOSE_LABELS_LONG)

    set_pub_style(title_size=9, tick_size=7)
    # Two panels across 1,000 pt leave ~470 pt each, which carries 15 pt type without stacking.
    fig, axes = plt.subplots(
        1, 2, figsize=(pub.size(pub.FULL, PUB_FIG_H) if pub.is_on() else (8.4, 4.3))
    )
    panel_gc_by_position(axes[0], C, present, xlab, annotate_wobble=False)
    # Overall GC on the right, not GC3: the first question is whether GC rises with dose at all,
    # which is a whole-sequence quantity. The left panel already carries the per-position split.
    panel_gc(axes[1], g, R, "gc", lab, conditions=ladder)
    if pub.is_on():
        pub.tight(fig)
        pub.finish(fig, "14_gc_codon_position", directory=out)
    else:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"14_gc_codon_position.{ext}", dpi=300, bbox_inches="tight")
        print(f"[wrote] {out}/14_gc_codon_position.png")
    plt.close(fig)




# Figures 16 and 17 compare steering layers at their shared alpha=1 dose.
# Include random arms and pair both layers against the same unsteered cells.
CMP_CONDITIONS = ["unsteered", "random_a1.0_L24", "add_a1.0_L24", "random_a1.0", "add_a1.0"]
CMP_LABELS = ["unsteered", "random", "add", "random", "add"]







def main() -> None:
    global FROM_FIGURE_DATA, MIN_CODONS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--layer", type=int, default=STEER_LAYER)
    ap.add_argument(
        "--emit-tables",
        action="store_true",
        help="compute the composition and codon tables into <run>/figures/ and exit without "
        "rendering. build_figure_data.py reads those two tables, so a full --run must "
        "materialise them BEFORE it builds figure_data/ -- they used to appear only as a side "
        "effect of rendering the now-archived cross-layer figures.",
    )
    ap.add_argument(
        "--from-figure-data",
        action="store_true",
        help="read the tracked composition and codon tables in figure_data/ instead of "
        "recomputing them from the saved generations",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--refresh-codons",
        action="store_true",
        help="recompute the cached codon-substitution table (a few minutes of alignment)",
    )
    ap.add_argument(
        "--refresh-composition",
        action="store_true",
        help="recompute the cached generation-composition table (re-reads the generation dump)",
    )
    ap.add_argument(
        "--min-codons",
        type=int,
        default=MIN_CODONS,
        help=f"fewest scorable codons a generation must contribute to enter the codon table "
        f"(default {MIN_CODONS}). Lowering it trades pN/pS precision for gene coverage: at 15 "
        f"COX17's 18-codon window is admitted. Only takes effect with --refresh-codons.",
    )
    ap.add_argument(
        "--pub",
        action="store_true",
        help="render the PUBLICATION figure: an exact 1,000 pt panel, the style guide's 15 pt "
        "type with monospaced numerals, and no tight-bbox crop. Writes into <figures>/pub/.",
    )
    args = ap.parse_args()

    FROM_FIGURE_DATA = args.from_figure_data
    MIN_CODONS = args.min_codons
    if args.pub:
        pub.enable()

    if args.emit_tables:
        if args.from_figure_data:
            sys.exit("--emit-tables computes the tables; it cannot also read them back")
        shared = cache_dir(args.run)
        g, _ = composition_table(args.run, args.dir, shared, refresh=args.refresh_composition)
        c = codon_stats(
            args.run, args.dir, shared, workers=args.workers, refresh=args.refresh_codons
        )
        # Write back unconditionally. build_figure_data reads these CSVs directly rather than
        # through this module, so filtering a stale cache in memory would not reach it -- the file
        # on disk has to be the corrected one.
        g.to_csv(shared / "12b_composition_gene_means.csv", index=False)
        c.to_csv(shared / "13b_codon_substitution_stats.csv", index=False)
        print(
            f"[tables] composition {len(g):,} rows, codon {len(c):,} rows -> {shared}\n"
            f"[tables] conditions: {sorted(set(g.condition))}"
        )
        return

    # `--layer` picks the stage-4 condition suffix (blocks.27 has none, other layers carry _L<n>)
    # and the output directory <run>/figures_<layer>; the cached tables are shared, in
    # <run>/figures.
    out = figures_dir(args.run, args.layer)
    out.mkdir(parents=True, exist_ok=True)
    print(f"building 14_gc_codon_position (blocks.{args.layer}) ...")
    fig14(
        args.run,
        args.dir,
        out,
        workers=args.workers,
        refresh_codons=args.refresh_codons,
        refresh_composition=args.refresh_composition,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()
