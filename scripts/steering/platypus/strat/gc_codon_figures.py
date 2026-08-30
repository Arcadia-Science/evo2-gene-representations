"""Composition figures: where the generations sit relative to each species, and how much of the steering vector is GC. CPU only, built from artefacts the run already has."""
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
sys.path.insert(0, str(ROOT / "scripts" / "steering"))   # top_recon_steer_sweep.nt_aligner
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

INK, WARM, PLUM, GREY = (acs.SERIES_PRIMARY, acs.SERIES_NULL,
                        acs.SERIES_THIRD, acs.SERIES_MUTED)
PLAT_C, HUM_C = acs.STEER_COLORS["target"], acs.STEER_COLORS["source"]
BASES = set("ACGT")
STOPS = {"TAA", "TAG", "TGA"}
# The dose ladder, in the order it is plotted. `unsteered` is alpha = 0.
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


def _layer_note(layer: int) -> str:
    """Return a title suffix identifying the steering layer."""
    return f"  —  steering at blocks.{layer}"


def complete_conditions(table: pd.DataFrame, wanted: list[str], what: str,
                        min_frac: float = 0.95) -> list[str]:
    """Keep only conditions the table covers for (nearly) every gene it covers at all."""
    n = table.groupby("condition").gene.nunique()
    full = int(n.max()) if len(n) else 0
    keep = [c for c in wanted if int(n.get(c, 0)) >= min_frac * full]
    thin = [(c, int(n.get(c, 0))) for c in wanted if c not in keep and int(n.get(c, 0)) > 0]
    absent = [c for c in wanted if int(n.get(c, 0)) == 0]
    if thin:
        print(f"  {what}: DROPPED still-generating {', '.join(f'{c} ({k}/{full} genes)' for c, k in thin)}")
    if absent:
        print(f"  {what}: absent {', '.join(absent)}")
    return keep


def ladder_and_labels(table: pd.DataFrame, suffix: str, what: str,
                      labels: list[str] = DOSE_LABELS) -> tuple[list[str], list[str]]:
    """The complete rungs of a layer's dose ladder, with their matching tick labels."""
    full = dose_ladder(suffix)
    keep = complete_conditions(table, full, what)
    return keep, [labels[full.index(c)] for c in keep]


def dose_ladder(suffix: str = "") -> list[str]:
    """The dose ladder for one steering layer, named as stage 4 writes it."""
    return [c if c == "unsteered" else c + suffix for c in DOSE]


def extra_arms(suffix: str = "") -> list[str]:
    return [c + suffix for c in EXTRA]


# Every condition the cached tables should cover. Conditions that do not exist yet simply do not
# appear; nothing downstream assumes they are there.
KEEP = list(dict.fromkeys(DOSE + EXTRA + dose_ladder("_L24") + extra_arms("_L24")
                          + ["add_own", "add_own_L24"]))


def read_fasta(p: Path) -> dict[str, str]:
    out, k = {}, None
    for ln in p.read_text().splitlines():
        if ln.startswith(">"):
            k = ln[1:].split("|")[0]
            out[k] = ""
        elif k:
            out[k] += ln.strip()
    return out


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
    codons = [seq[i:i + 3].upper() for i in range(0, len(seq) - 2, 3)]
    c = Counter(x for x in codons if len(x) == 3 and set(x) <= BASES and x not in STOPS)
    v = np.array([c.get(k, 0) for k in SENSE], dtype=float)
    t = v.sum()
    return v / t if t else v


def _codon_table() -> dict[str, str]:
    from Bio.Seq import Seq
    return {c: str(Seq(c).translate()) for c in
            (a + b + d for a in "ACGT" for b in "ACGT" for d in "ACGT")}


AA = _codon_table()


def _syn_sites(codon: str) -> float:
    """NG86 synonymous-site count for one codon: per position, the fraction of the 3 possible
    substitutions that leave the amino acid unchanged, summed over the 3 positions."""
    aa = AA[codon]
    s = 0.0
    for i in range(3):
        same = sum(1 for b in "ACGT"
                   if b != codon[i] and AA[codon[:i] + b + codon[i + 1:]] == aa)
        s += same / 3.0
    return s


SYN_SITES = {c: _syn_sites(c) for c in AA if AA[c] != "*"}


def gc_by_position(seq: str) -> tuple[float, float, float]:
    """GC at codon positions 1, 2, 3 separately (percentages), read in the sequence's own frame."""
    out = []
    for off in (0, 1, 2):
        b = [seq[i].upper() for i in range(off, len(seq) - (len(seq) - off) % 3, 3)
             if seq[i].upper() in BASES]
        out.append(100.0 * sum(1 for c in b if c in "GC") / len(b) if b else np.nan)
    return tuple(out)


def codon_diff_stats(gen: str, target: str) -> dict:
    """Synonymous / nonsynonymous differences between a generation and the platypus target."""
    from top_recon_steer_sweep import nt_aligner
    if not gen or not target:
        return {}
    a = nt_aligner().align(gen, target)[0]
    g_aln, t_aln = str(a[0]), str(a[1])
    # walk in target coordinates, collecting (gen_codon, target_codon) for clean triplets
    tpos, gcod, tcod = 0, [], []
    buf_g, buf_t = [], []
    for gc, tc in zip(g_aln, t_aln):
        if tc == "-":
            continue                      # insertion in the generation: not a target codon column
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
    for g3, t3 in zip(gcod, tcod):
        if t3 not in SYN_SITES or AA[g3] == "*":
            continue
        n_cod += 1
        s = SYN_SITES[t3]
        S += s
        N += 3.0 - s
        if g3 == t3:
            continue
        if sum(1 for x, y in zip(g3, t3) if x != y) == 1:
            if AA[g3] == AA[t3]:
                Sd += 1
            else:
                Nd += 1
    if n_cod < 20:
        return {}
    pS = Sd / S if S > 0 else np.nan
    pN = Nd / N if N > 0 else np.nan
    # Numerator and denominator must cover the same codon pairs — both single-position differences.
    # `syn_site_frac` is S/(S+N), the value this takes when pN = pS, and is the right reference line;
    # not 1/3, since only ~70% of wobble changes are silent.
    return {"n_codons": n_cod,
            "syn_per_100": 100.0 * Sd / n_cod,
            "nonsyn_per_100": 100.0 * Nd / n_cod,
            "frac_diff_syn": (Sd / (Sd + Nd)) if (Sd + Nd) else np.nan,
            "syn_site_frac": S / (S + N) if (S + N) else np.nan,
            "pS": pS, "pN": pN,
            "pN_over_pS": (pN / pS) if (pS and pS > 0) else np.nan}


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two codon-frequency vectors."""
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = (a > 0) & (b > 0)
        return float(np.sum(a[ok] * np.log2(a[ok] / b[ok])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# composition of the generations (shared by figures 12, 14, 15)
def composition_table(run: Path, arm: str, out: Path,
                      refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-gene-per-condition generation composition, per-gene reference windows)."""
    gc_cache, ref_cache = out / "12b_composition_gene_means.csv", out / "12c_reference_windows.csv"
    if gc_cache.exists() and ref_cache.exists() and not refresh:
        print(f"  (reusing {gc_cache.name}; pass --refresh-composition to recompute)")
        return pd.read_csv(gc_cache), pd.read_csv(ref_cache).set_index("gene")

    ch = read_fasta(run / "stage1" / "cds_human.fasta")
    cp = read_fasta(run / "stage1" / "cds_platypus.fasta")
    plan = pd.read_csv(run / arm / "scoring_plan.csv")
    plan = plan[plan.usable].set_index("gene")

    # Reference composition is measured on the same window that was scored: a whole-CDS GC is a
    # different quantity from the generations' GC.
    ref = []
    for gene, r in plan.iterrows():
        if gene not in ch or gene not in cp:
            continue
        nt = int(r.n_tokens)
        h = ch[gene][int(r.off_h) + 90:][:nt]
        p = cp[gene][int(r.off_p) + 90:][:nt]
        gh, g3h = gc_stats(h)
        gp, g3p = gc_stats(p)
        ref.append({"gene": gene, "gc_human": gh, "gc3_human": g3h,
                    "gc_platypus": gp, "gc3_platypus": g3p,
                    "cf_human": codon_freq(h), "cf_platypus": codon_freq(p)})
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
                rows.append({"gene": r["gene"], "condition": r["condition"],
                             "gc": gc, "gc3": gc3,
                             "jsd_plat": jsd(cf, R.at[r["gene"], "cf_platypus"]),
                             "jsd_hum": jsd(cf, R.at[r["gene"], "cf_human"])})
        except EOFError:
            print("  NOTE generations.jsonl.gz truncated; using what survived")
    G = pd.DataFrame(rows)
    # gene means first: samples in a cell share a prompt, so the gene is the unit
    g = G.groupby(["gene", "condition"]).mean(numeric_only=True).reset_index()
    print(f"  generations: {len(G)} samples over {g.gene.nunique()} genes, "
          f"{g.condition.nunique()} conditions")

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
def panel_gc(ax, g: pd.DataFrame, R: pd.DataFrame, col: str,
             labels: list[str] = DOSE_LABELS, conditions: list[str] | None = None,
             xlabel: str = "α") -> None:
    """Boxplots of overall GC (col='gc') or GC3 (col='gc3'), against the platypus and human
    reference lines measured on the same scored window."""
    conditions = DOSE if conditions is None else conditions
    ylab, title = {
        "gc": ("GC (%)", "Overall GC content"),
        "gc3": ("GC3 (%)",
                "GC at the third codon position\n(the axis `gc_removed` projects out)"),
    }[col]
    data = [g.loc[g.condition == c, col].dropna().to_numpy() for c in conditions]
    bp = ax.boxplot(data, positions=range(len(conditions)), widths=0.6, showfliers=False,
                    patch_artist=True, medianprops=dict(color=acs.apc.white, lw=1.6))
    for patch in bp["boxes"]:
        patch.set_facecolor(INK)
        patch.set_edgecolor("none")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(acs.SERIES_MUTED)
    for key, c, nm, side in ((f"{col}_platypus", PLAT_C, "platypus CDS", "bottom"),
                             (f"{col}_human", HUM_C, "human CDS", "top")):
        m = float(R[key].mean())
        ax.axhline(m, color=c, ls="--", lw=1.6, zorder=0)
        if pub.is_on():
            # The references are ~4 pp apart, so at 15 pt a two-line label collides. One line each,
            # on the outer side of its own rule.
            ax.text(len(conditions) - 0.45, m, f" {nm} {m:.1f}%", color=c,
                    va=side, ha="left")
        else:
            ax.text(len(conditions) - 0.45, m, f" {nm}\n {m:.1f}%", color=c, fontsize=6.6,
                    va="center", ha="left")
    _dose_axis(ax, len(conditions), labels, xlabel)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8.5)
    # Headroom for the reference labels, sized for the narrowest panel this is drawn in.
    ax.set_xlim(-0.6, len(conditions) + 1.5)


def panel_codon_usage(ax, g: pd.DataFrame, labels: list[str] = DOSE_LABELS,
                      conditions: list[str] | None = None, xlabel: str = "α",
                      connect: bool = True) -> None:
    """Codon-usage JSD to each species."""
    conditions = DOSE if conditions is None else conditions
    for key, c, nm, mk in (("jsd_plat", PLAT_C, "to platypus", "o"),
                           ("jsd_hum", HUM_C, "to human", "s")):
        m = [g.loc[g.condition == cc, key].mean() for cc in conditions]
        e = [g.loc[g.condition == cc, key].sem() for cc in conditions]
        ax.errorbar(range(len(conditions)), m, yerr=e, color=c, lw=2.0, ls="-" if connect else "none", marker=mk, ms=5, capsize=2, label=nm)
    _dose_axis(ax, len(conditions), labels, xlabel)
    ax.set_ylabel("codon-usage JSD (bits)")
    ax.set_title("Codon usage: distance to each species\n(lower = more like that species)",
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)


def panel_gc_by_position(ax, C: pd.DataFrame, present: list[str],
                         labels: list[str] = DOSE_LABELS, annotate_wobble: bool = True,
                         xlabel: str = "α", connect: bool = True) -> None:
    """GC at codon positions 1/2/3 separately over the dose ladder."""
    xs = range(len(present))
    # GC1/GC2/GC3 are an ordered series -> blue_shades light -> dark.
    for col, c, nm in (("gc1", acs.apc.vital, "GC1"), ("gc2", acs.apc.aegean, "GC2"),
                       ("gc3", INK, "GC3 (wobble)" if annotate_wobble else "GC3")):
        m = [C.loc[C.condition == cc, col].mean() for cc in present]
        ax.plot(xs, m, color=c, lw=2.0, ls="-" if connect else "none",
                marker="o", ms=4.5, label=nm)
    _dose_axis(ax, len(present), labels, xlabel)
    ax.set_ylabel("GC (%)")
    ax.set_title("GC by codon position" + ("\n(wobble vs the two coding positions)"
                                           if annotate_wobble else ""), fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)


def panel_frac_synonymous(ax, C: pd.DataFrame, present: list[str],
                          labels: list[str] = DOSE_LABELS,
                          ylabel: str = "% of codon differences that are synonymous",
                          xlabel: str = "α", connect: bool = True) -> None:
    """Synonymous share of the single-position codon differences vs the platypus target."""
    xs = range(len(present))
    m = [100 * C.loc[C.condition == cc, "frac_diff_syn"].mean() for cc in present]
    e = [100 * C.loc[C.condition == cc, "frac_diff_syn"].sem() for cc in present]
    ax.errorbar(xs, m, yerr=e, color=PLUM, lw=2.0, ls="-" if connect else "none", marker="o", ms=4.5, capsize=2)
    _dose_axis(ax, len(present), labels, xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Is the change silent?\n(higher = codons preserved)", fontsize=8.5)


def panel_pn_ps(ax, C: pd.DataFrame, present: list[str], labels: list[str] = DOSE_LABELS,
                title: str = "Per-site rate ratio vs the target\n"
                             "(>1 = protein-changing in excess)",
                xlabel: str = "α", connect: bool = True) -> None:
    """pN/pS against the platypus target over the dose ladder."""
    xs = range(len(present))
    m = [C.loc[C.condition == cc, "pN_over_pS"].mean() for cc in present]
    e = [C.loc[C.condition == cc, "pN_over_pS"].sem() for cc in present]
    ax.errorbar(xs, m, yerr=e, color=INK, lw=2.0, ls="-" if connect else "none", marker="o", ms=4.5, capsize=2)
    ax.axhline(1.0, color=GREY, ls="--", lw=1.3)
    # Below the line, not above it: 1.0 is at the very top of the autoscaled range, so a label
    # sitting on top of it is clipped by the axes frame.
    ax.text(len(present) - 1, 1.0, " pN = pS", color=GREY, fontsize=6.5, va="top", ha="right")
    _dose_axis(ax, len(present), labels, xlabel)
    ax.set_ylabel("pN / pS")
    ax.set_title(title, fontsize=8.5)


# figure 12
def fig12(run: Path, arm: str, out: Path, refresh: bool = False,
          layer: int = 27) -> pd.DataFrame:
    suffix = layer_suffix(layer)
    g, R = composition_table(run, arm, cache_dir(run), refresh=refresh)
    ladder, lab = ladder_and_labels(g, suffix, "fig12")

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.3))

    panel_gc(axes[0], g, R, "gc", lab, conditions=ladder)
    panel_gc(axes[1], g, R, "gc3", lab, conditions=ladder)
    panel_codon_usage(axes[2], g, lab, conditions=ladder)

    # The directional test: toward platypus and away from human are different claims, so plot both
    # changes per gene. The diagonal is "no differential preference"; only points below it are
    # specifically platypus-ward.
    ax = axes[3]
    u = g[g.condition == "unsteered"].set_index("gene")
    s = g[g.condition == f"add_a1.0{suffix}"].set_index("gene")
    ix = s.index.intersection(u.index)
    dh = (s.loc[ix, "jsd_hum"] - u.loc[ix, "jsd_hum"]).to_numpy()
    dp = (s.loc[ix, "jsd_plat"] - u.loc[ix, "jsd_plat"]).to_numpy()
    ax.scatter(dh, dp, s=9, color=INK, alpha=0.45, lw=0)
    lim = float(np.nanmax(np.abs(np.concatenate([dh, dp])))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=GREY, ls=":", lw=1.2)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    n_spec = int(np.sum(dp < dh))
    ax.set_xlabel("Δ JSD to human  (α=1 − unsteered)")
    ax.set_ylabel("Δ JSD to platypus")
    ax.set_title(f"Toward platypus, or away from human?\n{n_spec}/{len(ix)} genes below the "
                 f"diagonal", fontsize=8.5)

    fig.suptitle("Composition of the generations against both species' actual CDS  "
                 f"(same scored window; gene means, not sample means){_layer_note(layer)}",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"12_composition_vs_platypus.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tab = g.groupby("condition").agg(
        gc=("gc", "mean"), gc3=("gc3", "mean"),
        jsd_to_platypus=("jsd_plat", "mean"), jsd_to_human=("jsd_hum", "mean"),
        n_genes=("gene", "nunique")).round(4)
    for key in ("gc", "gc3"):
        tab.loc["REFERENCE_platypus_CDS", key] = round(float(R[f"{key}_platypus"].mean()), 4)
        tab.loc["REFERENCE_human_CDS", key] = round(float(R[f"{key}_human"].mean()), 4)
    tab.to_csv(out / "12_composition_vs_platypus.csv")
    print(f"[wrote] {out}/12_composition_vs_platypus.png")
    return tab


# codon-level substitution stats (row 2 of figure 13)
_CS: dict = {}


def _cs_init(payload: dict) -> None:
    _CS.update(payload)


def _cs_one_gene(gene: str) -> list[dict]:
    tgt = _CS["target"][gene]
    out = []
    for rec in _CS["gens"].get(gene, []):
        g1, g2, g3 = gc_by_position(rec["seq"])
        st = codon_diff_stats(rec["seq"], tgt)
        if not st:
            continue
        out.append({"gene": gene, "condition": rec["condition"], "sample": rec["sample"],
                    "gc1": g1, "gc2": g2, "gc3": g3, **st})
    return out


def codon_stats(run: Path, arm: str, out: Path, workers: int = 8,
                refresh: bool = False) -> pd.DataFrame:
    """Per-(gene, condition) codon-level stats, cached — the alignments cost a few minutes."""
    cache = out / "13b_codon_substitution_stats.csv"
    if cache.exists() and not refresh:
        print(f"  (reusing {cache.name}; pass --refresh-codons to recompute)")
        return pd.read_csv(cache)

    cp = read_fasta(run / "stage1" / "cds_platypus.fasta")
    plan = pd.read_csv(run / arm / "scoring_plan.csv")
    plan = plan[plan.usable].set_index("gene")
    target = {}
    for gene, r in plan.iterrows():
        if gene in cp:
            target[gene] = cp[gene][int(r.off_p) + 90:][:int(r.n_tokens)]

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
    print(f"  codon stats over {n} generations, {len(genes)} genes "
          f"(one alignment each; {workers} workers)")

    from multiprocessing import Pool
    rows = []
    with Pool(workers, initializer=_cs_init,
              initargs=({"target": target, "gens": gens},)) as pool:
        for i, res in enumerate(pool.imap_unordered(_cs_one_gene, genes, chunksize=1), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"    {i}/{len(genes)} genes", flush=True)
    per_sample = pd.DataFrame(rows)
    g = per_sample.groupby(["gene", "condition"]).mean(numeric_only=True).reset_index()
    g.to_csv(cache, index=False)
    print(f"  [wrote] {cache}")
    return g


# figure 13
def fig13(run: Path, arm: str, out: Path, rep_dir: str, layer: int,
          workers: int = 8, refresh_codons: bool = False) -> pd.DataFrame:
    suffix = layer_suffix(layer)
    npz = run / rep_dir / "loo_vectors.npz"
    if not npz.exists():
        sys.exit(f"no {npz}; pass --rep-dir (stage3_cds_mean or stage3_aligned_mean)")
    d = np.load(npz, allow_pickle=True)
    if "gc_axis_L0" not in d:
        sys.exit("loo_vectors.npz has no gc_axis_L* -- re-run stage3_select.py (added 2026-08-07)")
    layers = [int(x) for x in d["layers"]]

    pooled = np.load(run / "stage2" / "pooled_representations.npz", allow_pickle=True)
    key = "cds_mean" if "cds_mean" in rep_dir else "aligned_mean"
    X = pooled[key]                      # (genes, 2, layers, dim); species order = pooled["species"]
    sp = [str(x) for x in pooled["species"]]
    ip = next(i for i, s in enumerate(sp)
              if s.lower().startswith("plat") or s.lower().startswith("orn"))
    ih = 1 - ip

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n else v

    rows = []
    for li in layers:
        v = d[f"v_pooled_L{li}"].astype(np.float64)
        gc = d[f"gc_axis_L{li}"].astype(np.float64)
        mu = d[f"muhat_L{li}"].astype(np.float64)
        D = (X[:, ip, li, :] - X[:, ih, li, :]).astype(np.float64)   # per-gene delta cloud
        Dc = D - D.mean(axis=0, keepdims=True)
        # PC1 of the MEAN-CENTRED cloud via SVD (no 4096x4096 covariance matrix needed)
        _U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
        pc1 = Vt[0]
        rows.append({
            "layer": li,
            "cos_v_gc": abs(float(unit(v) @ unit(gc))),
            "cos_v_mu": abs(float(unit(v) @ unit(mu))),
            "cos_pc1_gc": abs(float(pc1 @ unit(gc))),
            "cos_pc1_mu": abs(float(pc1 @ unit(mu))),
            "cos_pc1_v": abs(float(pc1 @ unit(v))),
            "pc1_var_frac": float(S[0] ** 2 / np.sum(S ** 2)),
        })
    T = pd.DataFrame(rows).set_index("layer")
    T["share_of_v2_along_gc"] = T.cos_v_gc ** 2
    T.to_csv(out / "13_gc_in_steering_vector.csv")
    at = T.loc[layer]

    # ---- the causal side: what the gc_removed arm did to the gain ----------------------------
    summ = run / arm / "analysis_summary.csv"
    causal = {}
    if summ.exists():
        s = pd.read_csv(summ)
        col = next((c for c in s.columns if c.endswith("_delta") and "correct" in c), None)
        if col:
            for cond in (f"add_a1.0{suffix}", f"add_gc_removed_a1.0{suffix}"):
                q = s[s.condition == cond]
                if len(q):
                    causal[cond] = float(q.iloc[0][col])
            causal["metric"] = col.replace("_delta", "")

    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)

    set_pub_style(title_size=9, tick_size=7)
    fig, axall = plt.subplots(2, 4, figsize=(16.2, 8.6))
    axes = axall[0]

    ax = axes[0]
    ax.plot(T.index, T.cos_v_gc, color=INK, lw=2.0, marker="o", ms=3.5, label="|cos(v, GC axis)|")
    ax.plot(T.index, T.cos_v_mu, color=GREY, lw=1.6, ls="--", marker="s", ms=3,
            label="|cos(v, μ̂)| — cone")
    ax.axvline(layer, color=WARM, ls=":", lw=1.4)
    ax.text(layer, ax.get_ylim()[1], f" L{layer}", color=WARM, fontsize=7, va="top")
    ax.set_xlabel("layer (block)")
    ax.set_ylabel("|cosine|")
    ax.set_title("How much of the steering direction\nis the GC axis, by layer", fontsize=8.5)
    ax.legend(fontsize=6.8, frameon=False)

    ax = axes[1]
    ax.plot(T.index, 100 * T.pc1_var_frac, color=PLUM, lw=2.0, marker="o", ms=3.5)
    ax.axvline(layer, color=WARM, ls=":", lw=1.4)
    ax.set_xlabel("layer (block)")
    ax.set_ylabel("PC1 variance explained (%)")
    ax.set_title("PC1 of the difference-vector cloud\n(fit across genes, per layer)", fontsize=8.5)

    ax = axes[2]
    bars = [("|cos(PC1, GC axis)|", at.cos_pc1_gc, PLAT_C),
            ("|cos(v, GC axis)|", at.cos_v_gc, INK),
            ("|cos(PC1, v)|", at.cos_pc1_v, PLUM),
            ("|cos(PC1, μ̂)|", at.cos_pc1_mu, GREY),
            ("|cos(v, μ̂)|", at.cos_v_mu, GREY)]
    ax.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1],
            color=[b[2] for b in bars][::-1], height=0.62)
    for i, b in enumerate(bars[::-1]):
        ax.text(b[1] + 0.015, i, f"{b[1]:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("|cosine|")
    ax.set_title(f"At L{layer}: PC1 IS the GC axis,\nbut v is only partly on it", fontsize=8.5)
    ax.tick_params(labelsize=7)

    # ---- geometry vs effect ------------------------------------------------------------------
    ax = axes[3]
    geo = 100 * float(at.share_of_v2_along_gc)
    if f"add_a1.0{suffix}" in causal and f"add_gc_removed_a1.0{suffix}" in causal:
        kill = 100 * (1 - causal[f"add_gc_removed_a1.0{suffix}"] / causal[f"add_a1.0{suffix}"])
        vals = [("share of ‖v‖²\nalong the GC axis", geo, INK),
                ("share of the GAIN\nlost when GC is\nprojected out", kill, WARM)]
        ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v[1] + 1.5, f"{v[1]:.0f}%", ha="center", fontsize=10, fontweight="bold",
                    color=v[2])
        ax.set_ylim(0, max(geo, kill) * 1.28)
        ax.set_ylabel("%")
        ax.set_title("Geometry vs effect — the dissociation\n"
                     f"({causal.get('metric', 'metric')}, α=1)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no analysis_summary.csv with a\n`gc_removed` arm in this dir",
                ha="center", va="center", fontsize=8, color=GREY, transform=ax.transAxes)

    # ---- row 2: does the perturbation respect the reading frame? -----------------------------
    present, xlab = ladder_and_labels(C, suffix, "codon table")
    xs = range(len(present))

    panel_gc_by_position(axall[1, 0], C, present, xlab)

    ax = axall[1, 1]
    for col, c, nm, mk in (("syn_per_100", PLAT_C, "synonymous", "o"),
                           ("nonsyn_per_100", WARM, "nonsynonymous", "s")):
        m = [C.loc[C.condition == cc, col].mean() for cc in present]
        e = [C.loc[C.condition == cc, col].sem() for cc in present]
        ax.errorbar(xs, m, yerr=e, color=c, lw=2.0, marker=mk, ms=4.5, capsize=2, label=nm)
    ax.set_xticks(list(xs), xlab, fontsize=7)
    ax.set_xlabel("α")
    ax.set_ylabel("differences per 100 codons")
    ax.set_title("Substitutions vs the platypus target\n(single-position codon differences)",
                 fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)

    panel_frac_synonymous(axall[1, 2], C, present, xlab)

    panel_pn_ps(axall[1, 3], C, present, xlab)

    fig.suptitle("Is the steering vector just GC?  Geometry says it is a minority of the direction, "
                 "the causal ablation says it carries most of the effect (top);\n"
                 "and the perturbation it applies is not confined to the wobble position (bottom)",
                 y=1.005, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"13_gc_in_steering_vector.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}/13_gc_in_steering_vector.png")

    print(f"\n  at L{layer}:")
    for k in ("cos_pc1_gc", "cos_v_gc", "cos_pc1_v", "cos_pc1_mu", "cos_v_mu",
              "pc1_var_frac", "share_of_v2_along_gc"):
        print(f"    {k:24s} {float(at[k]):.4f}")
    if causal:
        print(f"    causal: add {causal.get(f'add_a1.0{suffix}', float('nan')):+.3f} pp -> "
              f"gc_removed {causal.get(f'add_gc_removed_a1.0{suffix}', float('nan')):+.3f} pp")
    return T


# figures 14 and 15 -- two-panel regroupings of panels already defined above
def fig14(run: Path, arm: str, out: Path, workers: int = 8, refresh_codons: bool = False,
          refresh_composition: bool = False, layer: int = 27) -> None:
    suffix = layer_suffix(layer)
    """GC by codon position (from 13) beside GC3 vs both species' reference lines (from 12)."""
    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)
    g, R = composition_table(run, arm, cache_dir(run), refresh=refresh_composition)
    present, xlab = ladder_and_labels(C, suffix, "codon table", DOSE_LABELS_LONG)
    ladder, lab = ladder_and_labels(g, suffix, "composition table", DOSE_LABELS_LONG)

    set_pub_style(title_size=9, tick_size=7)
    # Two panels across 1,000 pt leave ~470 pt each, which carries 15 pt type without stacking.
    fig, axes = plt.subplots(1, 2, figsize=(pub.size(pub.FULL, PUB_FIG_H) if pub.is_on()
                                            else (8.4, 4.3)))
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


def fig15(run: Path, arm: str, out: Path, workers: int = 8, refresh_codons: bool = False,
          refresh_composition: bool = False, layer: int = 27) -> None:
    suffix = layer_suffix(layer)
    """Codon-usage divergence to each species (from 12) beside the synonymous share (from 13)."""
    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)
    g, _R = composition_table(run, arm, cache_dir(run), refresh=refresh_composition)
    present, xlab = ladder_and_labels(C, suffix, "codon table")
    ladder, lab = ladder_and_labels(g, suffix, "composition table")

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.3))
    panel_codon_usage(axes[0], g, lab, conditions=ladder)
    # pN/pS provides a site-normalized measure with a neutral reference of 1.
    panel_pn_ps(axes[1], C, present, xlab,
                title="Is the change silent?\n(pN / pS; >1 = protein-changing in excess)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"15_codon_usage_and_silence.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}/15_codon_usage_and_silence.png")


# Figures 16 and 17 compare steering layers at their shared alpha=1 dose.
# Include random arms and pair both layers against the same unsteered cells.
CMP_CONDITIONS = ["unsteered", "random_a1.0_L24", "add_a1.0_L24", "random_a1.0", "add_a1.0"]
CMP_LABELS = ["unsteered", "random", "add", "random", "add"]


def _layer_brackets(ax) -> None:
    """Write the layer under the two pairs of arms, so the tick labels can stay short."""
    for x, nm in ((1.5, "blocks.24"), (3.5, "blocks.27")):
        ax.annotate(nm, xy=(x, -0.115), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=7.5, color=INK, annotation_clip=False)


def fig16(run: Path, arm: str, out: Path, workers: int = 8,
          refresh_codons: bool = False, refresh_composition: bool = False) -> None:
    """GC by codon position and overall GC, at alpha = 1 in both steering layers."""
    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)
    g, R = composition_table(run, arm, cache_dir(run), refresh=refresh_composition)
    present = [c for c in CMP_CONDITIONS if c in set(C.condition)]
    missing = [c for c in CMP_CONDITIONS if c not in set(C.condition)]
    if missing:
        print(f"  NOTE conditions absent from the codon table, dropped: {missing}")

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5))
    panel_gc_by_position(axes[0], C, present, CMP_LABELS[:len(present)], annotate_wobble=False,
                         xlabel="", connect=False)
    panel_gc(axes[1], g, R, "gc", CMP_LABELS, conditions=CMP_CONDITIONS, xlabel="")
    for ax in axes:
        _layer_brackets(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"16_gc_codon_position_L24_vs_L27.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}/16_gc_codon_position_L24_vs_L27.png")


def fig17(run: Path, arm: str, out: Path, workers: int = 8,
          refresh_codons: bool = False, refresh_composition: bool = False) -> None:
    """Codon-usage divergence and pN/pS, at alpha = 1 in both steering layers."""
    C = codon_stats(run, arm, cache_dir(run), workers=workers, refresh=refresh_codons)
    g, _R = composition_table(run, arm, cache_dir(run), refresh=refresh_composition)
    present = [c for c in CMP_CONDITIONS if c in set(C.condition)]

    set_pub_style(title_size=9, tick_size=7)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.5))
    panel_codon_usage(axes[0], g, CMP_LABELS, conditions=CMP_CONDITIONS, xlabel="", connect=False)
    panel_pn_ps(axes[1], C, present, CMP_LABELS[:len(present)],
                title="Is the change silent?\n(pN / pS; >1 = protein-changing in excess)",
                xlabel="", connect=False)
    for ax in axes:
        _layer_brackets(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"17_codon_usage_and_silence_L24_vs_L27.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}/17_codon_usage_and_silence_L24_vs_L27.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--rep-dir", default="stage3_cds_mean")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--only", nargs="+", type=int, choices=[12, 13, 14, 15, 16, 17],
                    default=[12, 13, 14, 15, 16, 17])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh-codons", action="store_true",
                    help="recompute the cached codon-substitution table (a few minutes of alignment)")
    ap.add_argument("--pub", action="store_true",
                    help="render the PUBLICATION figure (14 only): an exact 1,000 pt panel, the "
                         "style guide's 15 pt type with monospaced numerals, and no tight-bbox "
                         "crop. Writes into <figures>/pub/.")
    ap.add_argument("--refresh-composition", action="store_true",
                    help="recompute the cached generation-composition table (re-reads the "
                         "generation dump)")
    args = ap.parse_args()

    if args.pub:
        pub.enable()
    # `--layer` is the ONLY layer switch: it picks the stage-4 condition suffix (blocks.27 has
    # none, later layers carry _L<n>) and the output directory. Figures 12-15 go to the per-layer
    # <run>/figures_<layer>; the shared caches and the cross-layer figures 16-17 go to
    # <run>/figures.
    out = figures_dir(args.run, args.layer)
    out.mkdir(parents=True, exist_ok=True)
    shared = cache_dir(args.run)
    # The --refresh-* flags are ONE-SHOT: the first figure that touches a cache rebuilds it, and the
    # figures after it read the freshly written table instead of recomputing the same thing again.
    rc, rk = args.refresh_codons, args.refresh_composition
    if 12 in args.only:
        print(f"building 12_composition_vs_platypus (blocks.{args.layer}) ...")
        t = fig12(args.run, args.dir, out, refresh=rk, layer=args.layer)
        rk = False
        print(t.to_string())
    if 13 in args.only:
        print(f"\nbuilding 13_gc_in_steering_vector (blocks.{args.layer}) ...")
        fig13(args.run, args.dir, out, args.rep_dir, args.layer,
              workers=args.workers, refresh_codons=rc)
        rc = False
    if 14 in args.only:
        print(f"\nbuilding 14_gc_codon_position (blocks.{args.layer}) ...")
        fig14(args.run, args.dir, out, workers=args.workers,
              refresh_codons=rc, refresh_composition=rk, layer=args.layer)
        rc = rk = False
    if 15 in args.only:
        print(f"\nbuilding 15_codon_usage_and_silence (blocks.{args.layer}) ...")
        fig15(args.run, args.dir, out, workers=args.workers,
              refresh_codons=rc, refresh_composition=rk, layer=args.layer)
        rc = rk = False
    if 16 in args.only:
        print("\nbuilding 16_gc_codon_position_L24_vs_L27 ...")
        fig16(args.run, args.dir, shared, workers=args.workers,
              refresh_codons=rc, refresh_composition=rk)
        rc = rk = False
    if 17 in args.only:
        print("\nbuilding 17_codon_usage_and_silence_L24_vs_L27 ...")
        fig17(args.run, args.dir, shared, workers=args.workers,
              refresh_codons=rc, refresh_composition=rk)


if __name__ == "__main__":
    main()
