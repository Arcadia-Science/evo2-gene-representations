"""Does the composition ladder's rho just track retained source nucleotides?

Sequence level only — no model, no GPU. Two stages:

    uv run python scripts/controls/control_sequence_identity.py --stage identity  # ~15 min, CPU
    uv run python scripts/controls/control_sequence_identity.py --stage figure --pub

`identity` measures every control rung against (a) its own source sequence and (b) a second
INDEPENDENT draw of the same rung, and writes the two summary tables `build_figure_data.py`
publishes. `figure` joins those to figure 4's rho and renders figure 13.

The rho here IS figure 4's rho, read from the same tidy table figure 4 reads: between-family
Wasserstein and within-family angular preservation. Nothing in this file recomputes geometry.

The question: the rungs rank GC-matched < dinucleotide < 4-mer < 6-mer < synonymous recode in rho,
and they rank the same way in nucleotide identity to the source gene, so the ordering could just be
"how much of the original sequence survived". It is not. Identity to source equals the identity
between two independent draws of the SAME rung (`excess` = 0), so no rung carries source-specific
sequence; and an x-axis computable without ever seeing the source (`null_pos_identity`) orders the
rungs just as well, which is why monotonicity in identity is not the test — `excess` is.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import edlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))

import arcadia_pub as pub  # noqa: E402
import arcadia_style as acs  # noqa: E402
import build_figure_data as bfd  # noqa: E402  (the run path and block count, defined once)
from controls import make_control_sequences as mcs  # noqa: E402  (pure sequence ops; no torch)
from controls.plot_control_wasserstein import PUB_RUNG_LABELS  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

import figure_data  # noqa: E402

RUN = bfd.MAMMAL

# The nested pair: missense_subset edits a strict SUBSET of the bases synonymous_recode edited, so
# it moves fewer nucleotides while damaging the protein the recode kept. Both sit far up the
# identity axis, and neither is a nucleotide-composition rung — keeping them out of the
# nucleotide-only fit is the whole point of separating these lists.
NESTED_PAIR = ("synonymous_recode", "missense_subset")
# Ladder order: most sequence composition destroyed first, then the nested pair.
LADDER = ["kmer6_shuffle", "kmer4_shuffle", "dinuc_shuffle", "gc_match", *NESTED_PAIR]
NUCLEOTIDE_RUNGS = [c for c in LADDER if c not in NESTED_PAIR]
assert set(LADDER) <= set(mcs.CDSMASK_CONTROLS), "ladder names must match the control registry"

# Figure 4's two series, under the names this file's tables use. Read from
# figure_data/exp2_control_preservation.csv so figure 13's y-axis cannot drift from figure 4's.
RHO_COLS = {
    "rho_between_w2": "Between families (Wasserstein)",
    "rho_within_angular": "Within families (angular)",
}
# Swept as x-axes. `null_pos_identity` and `pos_identity_chance` are NEGATIVE CONTROLS on the axis
# itself: both are computable without ever looking at the source sequence, so if rho correlates
# with them as well as it does with identity-to-source, that correlation is not evidence that
# retained source sequence drives rho.
IDENTITY_METRICS = (
    "pos_identity",
    "edit_identity",
    "pos_excess",
    "aa_identity",
    "null_pos_identity",
    "pos_identity_chance",
)

DEFAULT_LAYER = 15  # blocks15 — the layer-selection pick for the Evo2 gene panels
# A nucleotide-rung fit evaluated more than this many fitted-ranges from its data is flagged rather
# than reported as a prediction.
EXTRAP_LIMIT = 3.0
# Height of one publication row, in points; figure 4 uses the same.
PUB_ROW_H = 330.0

# Figure 4's key spells out what each nested rung does to the protein. This figure's key is read
# beside a scatter where that is not the distinction being drawn, so the parentheticals come off.
# Overridden here rather than in PUB_RUNG_LABELS, which figure 4 also reads.
LABEL = {
    **PUB_RUNG_LABELS,
    "synonymous_recode": "Synonymous recode",
    "missense_subset": "Missense at recode sites",
}


# ── sequence metrics


def _codes(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)


def _translate(s: str) -> str:
    """Frame-0 translation; unknown codons (N, partial trailing codon) become 'X'."""
    return "".join(mcs._CODON.get(s[i : i + 3], "X") for i in range(0, len(s) - 2, 3))


def _base_freqs(s: str) -> dict[str, float]:
    c = Counter(s)
    n = len(s) or 1
    return {b: v / n for b, v in c.items()}


def pair_metrics(a: str, b: str, in_frame: bool) -> dict:
    """Identity of `b` to `a` by every measure we report."""
    ca, cb = _codes(a), _codes(b)
    n = min(len(ca), len(cb))
    eq = ca[:n] == cb[:n]
    out = {
        "len_src": len(a),
        "len_delta": len(b) - len(a),
        "pos_identity": float(eq.mean()) if n else np.nan,
        # Global edit identity normalized by the longer sequence.
        "edit_identity": 1.0
        - edlib.align(a, b, task="distance", mode="NW")["editDistance"] / max(len(a), len(b), 1),
        # What two independent sequences of these two compositions would score by chance.
        "pos_identity_chance": float(
            sum(pa * _base_freqs(b).get(base, 0.0) for base, pa in _base_freqs(a).items())
        ),
    }
    if in_frame:
        ncod = n // 3
        if ncod:
            by_pos = eq[: ncod * 3].reshape(ncod, 3).mean(axis=0)
            out |= {
                "p1_identity": float(by_pos[0]),
                "p2_identity": float(by_pos[1]),
                "p3_identity": float(by_pos[2]),
            }
        pa, pb = _translate(a), _translate(b)
        m = min(len(pa), len(pb))
        out["aa_identity"] = float(np.mean(_codes(pa[:m]) == _codes(pb[:m]))) if m else np.nan
    return out


# ── independent control draws


def draw_control(
    control: str, seqs: dict[str, str], fam_of: dict[str, str], seed_tag: str
) -> dict[str, str]:
    """One independent draw of `control` over `seqs`, seeded per (seed_tag, control, key)."""
    if control in NESTED_PAIR:
        by_fam: dict[str, list[str]] = defaultdict(list)
        for g, s in seqs.items():
            by_fam[fam_of[g]].append(s)
        usage = mcs.build_family_codon_usage(by_fam)
    out: dict[str, str] = {}
    for g, s in seqs.items():
        rng = random.Random(zlib.crc32(f"{seed_tag}:{control}:{g}".encode()))
        if control == "gc_match":
            out[g] = mcs.gc_match(s, rng)
        elif control == "dinuc_shuffle":
            out[g] = mcs.dinuc_shuffle(s, rng)
        elif control == "kmer4_shuffle":
            out[g] = mcs.klet_shuffle(s, 4, rng)
        elif control == "kmer6_shuffle":
            out[g] = mcs.klet_shuffle(s, 6, rng)
        elif control in NESTED_PAIR:
            # The recode partner comes from its OWN per-(tag, gene) stream, so a missense_subset
            # draw at seed tag T is matched to the synonymous_recode draw at the same tag T — the
            # two rungs stay paired sequence by sequence inside every draw, including the two
            # independent draws that form the self-pair null.
            recoded = mcs.synonymous_recode(
                s,
                usage[fam_of[g]],
                random.Random(zlib.crc32(f"{seed_tag}:synonymous_recode:{g}".encode())),
            )
            out[g] = (
                recoded if control == "synonymous_recode" else mcs.missense_subset(s, recoded, rng)
            )
        else:
            raise ValueError(f"unknown control {control}")
    return out


# ── stage 1: identity


def load_sources() -> tuple[dict[str, str], dict[str, str]]:
    """The mammalian ortholog CDS-masked panel: the coding substring each control rewrites.

    Identity must be measured on the CODING substring, not the transcript span: the control
    shuffles only coding positions and writes them back in place, so a span-level identity would
    be diluted by the untouched introns and UTRs.
    """
    cache = _REPO / "data/cache/mammal_embed"
    rungs = [c for c in LADDER if (cache / f"transcript_cdsmask_{c}").is_dir()]
    if not rungs:
        raise SystemExit(f"no control embedding caches under {cache}")
    keys = set.intersection(
        *[{p.stem for p in (cache / f"transcript_cdsmask_{c}").glob("*__*.npy")} for c in rungs]
    )
    positions = json.loads((_REPO / "data/cache/mammal_cds_positions.json").read_text())
    loci = _REPO / "data/mammalian_orthologs/loci"
    nat, fam_of = {}, {}
    for key in sorted(keys):
        f = loci / f"{key}.json"
        if key not in positions or not f.exists():
            continue
        d = json.loads(f.read_text())
        span = d["locus_seq"].upper()
        pos = sorted(p for p in positions[key] if 0 <= p < len(span))
        nat[key] = "".join(span[i] for i in pos)
        fam_of[key] = d["family"]
    return nat, fam_of


def applicable_rungs() -> list[str]:
    """Which rungs the panel actually embedded, read off its own control table."""
    tables = sorted(RUN.glob("blocks*/controls/control_within_scores.csv"))
    if not tables:
        raise SystemExit(f"no controls/control_within_scores.csv under {RUN}")
    present = set(pd.read_csv(tables[0])["condition"])
    return [c for c in LADDER if c in present]


def measure_identity(max_seqs: int | None) -> pd.DataFrame:
    """Measure source identity for every control rung in the panel."""
    nat, fam_of = load_sources()
    if max_seqs and len(nat) > max_seqs:
        keep = sorted(nat)[:: max(1, len(nat) // max_seqs)][:max_seqs]  # deterministic thinning
        nat = {k: nat[k] for k in keep}
        fam_of = {k: fam_of[k] for k in keep}
    print(
        f"{len(nat)} sequences, {len(set(fam_of.values()))} families, "
        f"median length {int(np.median([len(s) for s in nat.values()]))} bp"
    )

    rows = []
    for control in applicable_rungs():
        # Draw A/B are independent of each other by construction (different seed tags), which is
        # what makes identity(A, B) a null for the constraint-forced part of identity(source, A).
        A = draw_control(control, nat, fam_of, "drawA")
        B = draw_control(control, nat, fam_of, "drawB")
        for g, s in nat.items():
            m = pair_metrics(s, A[g], True)
            null = pair_metrics(A[g], B[g], True)
            rows.append(
                {
                    "condition": control,
                    "key": g,
                    "family": fam_of[g],
                    **m,
                    **{
                        f"null_{k}": v for k, v in null.items() if k not in ("len_src", "len_delta")
                    },
                }
            )
        print(f"  {control:18} n={sum(r['condition'] == control for r in rows):5d}  done")
    return pd.DataFrame(rows)


def summarise_identity(per_seq: pd.DataFrame) -> pd.DataFrame:
    """Per-rung means, plus the two derived quantities the argument turns on:
    `pos_excess` / `edit_excess` = identity to source minus the self-pair null."""
    num = [
        c for c in per_seq.columns if per_seq[c].dtype.kind == "f" or c in ("len_src", "len_delta")
    ]
    g = per_seq.groupby("condition")
    out = g[num].mean()
    out["pos_identity_sd"] = g["pos_identity"].std()
    out["n_seqs"] = g.size()
    # Standard error of the PAIRED per-sequence excess, so "excess is zero" can be read against its
    # own noise rather than asserted.
    exc = per_seq.assign(_e=per_seq["pos_identity"] - per_seq["null_pos_identity"]).groupby(
        "condition"
    )["_e"]
    out["pos_excess_se"] = exc.std() / np.sqrt(exc.size())
    out["pos_excess"] = out["pos_identity"] - out["null_pos_identity"]
    out["edit_excess"] = out["edit_identity"] - out["null_edit_identity"]
    if {"p1_identity", "p2_identity", "p3_identity"} <= set(out.columns):
        # Of the bases a rung DID change, what share sits at codon position 3? For a synonymous
        # recode this is the headline "the recode only touches wobble bases" claim, stated as the
        # quantity that claim is usually quoted as.
        ch = [1.0 - out[f"p{i}_identity"] for i in (1, 2, 3)]
        out["frac_bases_changed"] = sum(ch) / 3.0
        out["frac_changes_at_p3"] = ch[2] / (ch[0] + ch[1] + ch[2]).replace(0.0, np.nan)
    return out.reset_index()


def summarise_by_family(per_seq: pd.DataFrame) -> pd.DataFrame:
    """Per (rung, family) identity — the input to the within-rung test, which never compares one
    rung to another: inside ONE rung, does a family that retained more identity score higher?"""
    g = per_seq.groupby(["condition", "family"])
    out = g[["pos_identity", "edit_identity", "aa_identity"]].mean()
    out["n_seqs"] = g.size()
    return out.reset_index()


# ── stage 2: join rho and test monotonicity


def load_rho() -> pd.DataFrame:
    """Figure 4's two series per (layer, condition): between-family Wasserstein, and within-family
    angular averaged over families."""
    prep = figure_data.table("exp2_control_preservation")
    between = prep[prep.axis == "between_family"].set_index(["layer", "condition"])["rho"]
    within = prep[prep.axis == "within_family"].groupby(["layer", "condition"])["rho"].mean()
    out = pd.concat({"rho_between_w2": between, "rho_within_angular": within}, axis=1)
    return out.reset_index()


def _fit_residual(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Predicted y at x0 from a least-squares line through (x, y)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return np.nan
    m, b = np.polyfit(x[ok], y[ok], 1)
    return float(m * x0 + b)


def monotonicity_stats(ident: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    """Per (layer, rho column, identity metric): is rho monotone in identity across rungs?"""
    idm = ident.set_index("condition")
    out = []
    for layer, rg in rho.groupby("layer"):
        r = rg.set_index("condition")
        for rho_col in RHO_COLS:
            for x_col in IDENTITY_METRICS:
                if x_col not in idm.columns:
                    continue
                conds = [
                    c
                    for c in LADDER
                    if c in idm.index
                    and c in r.index
                    and np.isfinite(idm.loc[c, x_col])
                    and np.isfinite(r.loc[c, rho_col])
                ]
                if len(conds) < 3:
                    continue
                x = np.array([idm.loc[c, x_col] for c in conds])
                y = np.array([r.loc[c, rho_col] for c in conds])
                # The nucleotide-only subset is BOTH nested-pair members removed. Dropping only
                # the recode leaves missense_subset — a protein-damaging rung sitting at 0.88
                # identity — inside a set whose spread is supposed to be the 0.016 the four
                # shuffles cover, which silently inflates the fitted range 39-fold and reverses
                # the sign of the reported extrapolation.
                nuc = [i for i, c in enumerate(conds) if c not in NESTED_PAIR]
                rho_all = spearmanr(x, y)
                rho_nuc = spearmanr(x[nuc], y[nuc]) if len(nuc) >= 3 else (np.nan, np.nan)
                pred, resid, extrap = (np.nan, np.nan, np.nan)
                if "synonymous_recode" in conds and len(nuc) >= 2:
                    j = conds.index("synonymous_recode")
                    pred = _fit_residual(x[nuc], y[nuc], x[j])
                    resid = y[j] - pred
                    span = float(np.ptp(x[nuc]))
                    extrap = ((x[j] - max(x[nuc])) / span) if span > 0 else np.inf
                out.append(
                    {
                        "layer": layer,
                        "rho_col": rho_col,
                        "identity_metric": x_col,
                        "n_rungs": len(conds),
                        "spearman_all_rungs": rho_all[0],
                        "p_all_rungs": rho_all[1],
                        "spearman_nucleotide_rungs": rho_nuc[0],
                        "p_nucleotide_rungs": rho_nuc[1],
                        "n_nucleotide_rungs": len(nuc),
                        "identity_range_nucleotide_rungs": float(np.ptp(x[nuc])) if nuc else np.nan,
                        "rho_range_nucleotide_rungs": float(np.ptp(y[nuc])) if nuc else np.nan,
                        "recode_rho_predicted": pred,
                        "recode_rho_residual": resid,
                        "recode_extrapolation_ranges": extrap,
                    }
                )
    return pd.DataFrame(out)


def within_rung_stats(by_family: pd.DataFrame, layer: int) -> pd.DataFrame:
    """The test that does not rely on comparing rungs to each other: WITHIN one rung, across
    families, does a family whose sequences retained more identity get a higher rho?"""
    prep = figure_data.table("exp2_control_preservation")
    fam = prep[(prep.axis == "within_family") & (prep.layer == layer)][
        ["condition", "family", "rho"]
    ]
    m = by_family.merge(fam, on=["condition", "family"], how="inner")
    out = []
    for cond, g in m.groupby("condition"):
        ok = g[["pos_identity", "rho"]].dropna()
        if len(ok) < 5 or np.ptp(ok["pos_identity"]) == 0:
            continue
        s = spearmanr(ok["pos_identity"], ok["rho"])
        out.append(
            {
                "layer": layer,
                "condition": cond,
                "rho_col": "rho_within_angular",
                "n_families": len(ok),
                "spearman_identity_vs_rho": s[0],
                "p": s[1],
            }
        )
    return pd.DataFrame(out)


# ── figure


def _title(ax, text: str) -> None:
    """Panel title. Left-aligned in the diagnostic grid, CENTRED in pub mode: matplotlib keeps a
    separate `_left_title` artist for `loc="left"`, and `arcadia_pub.enforce_type` /`audit_glyphs`
    only walk `ax.title`, so a left-located title silently escapes both the 15 pt type rule and
    the missing-glyph check."""
    if pub.is_on():
        ax.set_title(text)
    else:
        ax.set_title(text, fontsize=8, loc="left", x=0.0)


def _colour(cond: str):
    return acs.CONTROL_COLORS.get(cond, acs.CONTROL_FALLBACK)


def _scatter(lo, hi, idm, rl, stats, conds, rho_col, title, layer) -> None:
    """rho against positional identity to source, on a broken x-axis.

    `natural` is deliberately absent: its identity is 1.0 and its preservation rho is 1.0, both by
    construction, so plotting it would stretch the y-axis over half its range to show a point that
    carries no information about the ladder.
    """
    xs = np.array([idm.loc[c, "pos_identity"] for c in conds])
    ys = np.array([rl.loc[c, rho_col] for c in conds])
    nuc = [i for i, c in enumerate(conds) if c not in NESTED_PAIR]
    pair = [i for i, c in enumerate(conds) if c in NESTED_PAIR]
    for ax in (lo, hi):
        for i, c in enumerate(conds):
            ax.scatter(
                xs[i],
                ys[i],
                s=60 if not pub.is_on() else 90,
                color=_colour(c),
                zorder=3,
                edgecolor=acs.apc.white,
                linewidth=0.5,
                marker="D" if c in NESTED_PAIR else "o",
            )
        if len(nuc) >= 2 and np.ptp(xs[nuc]) > 0:
            m, b = np.polyfit(xs[nuc], ys[nuc], 1)
            grid = np.linspace(0.2, 1.05, 60)
            ax.plot(grid, m * grid + b, ls=":", lw=0.7, color=acs.GRID)
            fx = np.linspace(min(xs[nuc]), max(xs[nuc]), 20)
            ax.plot(fx, m * fx + b, ls="-", lw=1.4, color=acs.ANNOTATION)
    if not pub.is_on():
        # Label the cluster on the zoomed axis. The nucleotide rungs can sit within 0.01 of each
        # other in BOTH coordinates, so labels go in an evenly spaced column with leader lines
        # rather than as per-point offsets, which collide. In pub mode the shared key does this.
        for rank, i in enumerate(sorted(nuc, key=lambda i: ys[i])):
            lo.annotate(
                LABEL.get(conds[i], conds[i]),
                xy=(xs[i], ys[i]),
                xycoords="data",
                xytext=(0.62, 0.08 + 0.42 * rank / max(len(nuc) - 1, 1)),
                textcoords="axes fraction",
                fontsize=5.5,
                va="center",
                ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=acs.GRID, shrinkA=0, shrinkB=3),
            )
        for rank, i in enumerate(sorted(pair, key=lambda i: ys[i])):
            hi.annotate(
                LABEL.get(conds[i], conds[i]),
                xy=(xs[i], ys[i]),
                xycoords="data",
                xytext=(0.10, 0.12 + 0.64 * rank / max(len(pair) - 1, 1)),
                textcoords="axes fraction",
                fontsize=5.5,
                va="center",
                ha="left",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=acs.GRID, shrinkA=0, shrinkB=3),
            )
    pad = max(np.ptp(xs[nuc]) * 0.55, 0.004) if nuc else 0.01
    lo.set_xlim(min(xs[nuc]) - pad, max(xs[nuc]) + pad * (2.6 if not pub.is_on() else 0.8))
    hi.set_xlim((min(xs[pair]) - 0.06) if pair else 0.9, 1.02)
    span = max(np.ptp(ys), 0.05)
    lo.set_ylim(min(ys) - 0.12 * span, max(ys) + 0.22 * span)
    lo.spines["right"].set_visible(False)
    hi.spines["left"].set_visible(False)
    hi.tick_params(labelleft=False, left=False)
    for ax, xf in ((lo, 1.0), (hi, 0.0)):  # break marks
        ax.plot(
            [xf, xf],
            [0, 1],
            transform=ax.transAxes,
            clip_on=False,
            color=acs.apc.black,
            lw=0.8,
            ls=(0, (2, 3)),
        )
    st = stats[
        (stats.layer == layer)
        & (stats.rho_col == rho_col)
        & (stats.identity_metric == "pos_identity")
    ]
    if len(st) and not pub.is_on():
        s = st.iloc[0]
        # Review aid, diagnostic render only: the publication panel quotes these in its caption.
        # The panel leaves its upper two thirds empty — every rung sits in the bottom band of a
        # y-axis shared with the recode — so the note goes there.
        lo.text(
            0.03,
            0.97,
            f"ρ_S(identity, ρ) all rungs = {s.spearman_all_rungs:+.2f} "
            f"(p = {s.p_all_rungs:.3f})\n"
            f"nucleotide rungs only ({int(s.n_nucleotide_rungs)}) = "
            f"{s.spearman_nucleotide_rungs:+.2f} (p = {s.p_nucleotide_rungs:.3f})\n"
            f"identity spread there = {s.identity_range_nucleotide_rungs:.4f}\n"
            f"recode sits {s.recode_extrapolation_ranges:.1f}× that spread beyond them\n"
            f"— line: fit over the nucleotide rungs (solid),\n"
            f"   then extrapolated (dotted, NOT a prediction)",
            transform=lo.transAxes,
            va="top",
            fontsize=5.5,
            bbox=dict(fc=acs.apc.white, ec=acs.GRID, lw=0.4, pad=2),
        )
    lo.set_ylabel("Spearman ρ (control vs natural)")
    lo.set_xlabel("positional identity to source", x=1.0, ha="center")
    _title(lo, title)


def _identity_bars(ax, idm, conds) -> None:
    """Identity to source against the identity the rung's own constraint forces."""
    x = np.arange(len(conds))
    w = 0.36
    ax.bar(
        x - w / 2,
        [idm.loc[c, "pos_identity"] for c in conds],
        w,
        color=[_colour(c) for c in conds],
        label="identity to source",
    )
    ax.bar(
        x + w / 2,
        [idm.loc[c, "null_pos_identity"] for c in conds],
        w,
        facecolor="none",
        edgecolor=acs.ANNOTATION,
        hatch="///",
        linewidth=0.6,
        label="two independent draws (null)",
    )
    for i, c in enumerate(conds):
        # Per rung, not one shared rule: the composition floor differs slightly by rung, and the
        # whole point of the panel is that the bars sit ON their own floor.
        ax.hlines(
            idm.loc[c, "pos_identity_chance"],
            i - 0.5,
            i + 0.5,
            color=acs.GRID,
            lw=1.0,
            ls=(0, (2, 2)),
            label="composition chance" if i == 0 else None,
        )
    ax.set_ylabel("positional nucleotide identity")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    _rung_ticks(ax, conds)


def _excess_bars(ax, idm, conds) -> None:
    """Identity minus the self-pair null: the part that is retained SOURCE sequence."""
    ex = np.array([idm.loc[c, "pos_excess"] for c in conds])
    se = np.array([idm.loc[c, "pos_excess_se"] for c in conds])
    ax.bar(
        range(len(conds)),
        ex,
        color=[_colour(c) for c in conds],
        yerr=1.96 * se,
        error_kw=dict(lw=0.8, capsize=2, ecolor=acs.ANNOTATION),
    )
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.6)
    # Five of the six bars are 0.000 at this scale, which is the result — but an invisible bar
    # cannot be read against its own error bar, so the values are written on. Six numbers do not
    # fit across this panel at 15 pt, so the publication render labels the one bar that is not
    # zero and states the rest as a single line.
    for i, e in enumerate(ex):
        ax.annotate(
            f"{e:+.4f}",
            (i, e),
            textcoords="offset points",
            xytext=(0, 6 if e >= 0 else -12),
            ha="center",
            fontsize=5.5,
            color=acs.ANNOTATION,
        )
    ax.set_ylabel("identity to source − self-pair null")
    lim = max(2.1 * np.nanmax(np.abs(ex) + 1.96 * se), 0.004)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.6, len(conds) - 0.4)
    _rung_ticks(ax, conds)


def _rung_ticks(ax, conds) -> None:
    """Rung names under the bars. Diagnostic only: the bar panels are not in the publication
    figure, where six labels of this length set at 15 pt would be taller than the panel."""
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABEL.get(c, c) for c in conds], rotation=30, ha="right", fontsize=5.5)


def _layer_panel(ax, stats, rho_col, title) -> None:
    """Is the monotonicity a layer-specific accident? Spearman across rungs, every block."""

    def series(metric, column):
        d = stats[(stats.rho_col == rho_col) & (stats.identity_metric == metric)]
        d = d.sort_values("layer")
        return d.layer, d[column]

    # The self-pair null is drawn over the identity series deliberately: it is computable without
    # ever seeing the source gene, so wherever the two curves coincide, the correlation with
    # identity-to-source carries no information about retained source sequence.
    for metric, column, style, colour, label in (
        ("pos_identity", "spearman_all_rungs", "-o", acs.apc.amber, "all rungs, identity"),
        (
            "null_pos_identity",
            "spearman_all_rungs",
            "--",
            acs.apc.dragon,
            "all rungs, SELF-PAIR NULL identity",
        ),
        (
            "pos_identity",
            "spearman_nucleotide_rungs",
            "-s",
            acs.apc.dusk,
            "four shuffles, identity",
        ),
        (
            "pos_excess",
            "spearman_nucleotide_rungs",
            "-^",
            acs.CONTROL_FALLBACK,
            "four shuffles, EXCESS identity",
        ),
    ):
        x, y = series(metric, column)
        ax.plot(x, y, style, ms=2.5, lw=1.0, color=colour, label=label)
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Evo2 block")
    ax.set_ylabel("Spearman(identity, ρ)")
    _title(ax, title)
    ax.legend(fontsize=5.5, loc="lower left")


def figure(ident: pd.DataFrame, rho: pd.DataFrame, stats: pd.DataFrame, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    set_pub_style(title_size=10, tick_size=8)
    idm = ident.set_index("condition")
    rl = rho[rho.layer == layer].set_index("condition")
    conds = [c for c in LADDER if c in idm.index and c in rl.index]
    if len(conds) < 3:
        raise SystemExit("fewer than three rungs have both identity and rho")

    on = pub.is_on()
    # The publication figure is the two rho-vs-identity scatters and nothing else — it answers one
    # question, "is this rho just retained source nucleotides", and the reader answers it by
    # looking at the x-axis. The identity decomposition that PROVES the answer (identity equals the
    # rung's own self-pair null, so excess is zero) is a number, not a shape: it belongs in the
    # table and the caption. The diagnostic render keeps it, along with the per-block check.
    letters = iter("ABCDEF")
    if on:
        fig = plt.figure(figsize=pub.size(pub.FULL, PUB_ROW_H))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        cells = [gs[0, i] for i in range(len(RHO_COLS))]
    else:
        fig = plt.figure(figsize=(14.5, 8.0))
        gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.34)
        cells = [gs[row, 1] for row in range(len(RHO_COLS))]
        ax = fig.add_subplot(gs[0, 0])
        _identity_bars(ax, idm, conds)
        _title(ax, f"{next(letters)}. Identity to the unaltered gene is the rung's own constraint")
        ax.legend(fontsize=5.5, loc="upper left")
        ax = fig.add_subplot(gs[1, 0])
        _excess_bars(ax, idm, conds)
        _title(ax, f"{next(letters)}. Retained source identity, ±95% CI")

    for cell, (rho_col, name) in zip(cells, RHO_COLS.items(), strict=True):
        sub = cell.subgridspec(1, 2, width_ratios=[3, 2], wspace=0.06)
        lo = fig.add_subplot(sub[0])
        hi = fig.add_subplot(sub[1], sharey=lo)
        prefix = "" if on else f"{next(letters)}. "
        _scatter(lo, hi, idm, rl, stats, conds, rho_col, f"{prefix}{name}", layer)
    if not on:
        for row, (rho_col, name) in enumerate(RHO_COLS.items()):
            _layer_panel(
                fig.add_subplot(gs[row, 2]), stats, rho_col, f"{next(letters)}. {name}, per block"
            )
        fig.suptitle(
            f"Is the control ladder's ρ just nucleotide identity to source?  "
            f"[mammal CDS-masked panel, blocks{layer}]",
            fontsize=11,
            y=0.98,
        )
        for ext in ("png", "pdf"):
            fig.savefig(RUN / f"control_rho_vs_identity.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {RUN}/control_rho_vs_identity.{{png,pdf}}")
        return

    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker="D" if c in NESTED_PAIR else "o",
            ms=8,
            color=_colour(c),
            label=LABEL.get(c, c),
        )
        for c in conds
    ]
    legend, key_h = pub.key_below(
        fig, handles, [h.get_label() for h in handles], title="Control", width=pub.FULL
    )
    height = PUB_ROW_H + key_h
    fig.set_size_inches(pub.FULL / 72.0, height / 72.0)
    margin = 30.0  # the guide's panel inset
    fig.subplots_adjust(
        left=110.0 / pub.FULL,
        right=1.0 - margin / pub.FULL,
        # Room under the bottom row for its axis label and numerals, then the key.
        bottom=(key_h + margin + 55.0) / height,
        top=1.0 - (margin + 26.0) / height,  # panel titles sit above the axes box
        wspace=0.30,
    )
    legend.set_bbox_to_anchor((margin / pub.FULL, margin / height), transform=fig.transFigure)
    pub.finish(fig, "control_identity_vs_rho", directory=RUN)


# ── report


def write_report(
    ident: pd.DataFrame,
    rho: pd.DataFrame,
    stats: pd.DataFrame,
    within: pd.DataFrame,
    layer: int,
) -> None:
    idm = ident.set_index("condition")
    rl = rho[rho.layer == layer].set_index("condition")
    conds = [c for c in LADDER if c in idm.index]
    L = [
        "# Nucleotide identity of each composition control to its source — mammal CDS-masked panel",
        "",
        f"Run: `{RUN}`  ·  headline layer: `blocks{layer}`",
        "",
        "ρ is **figure 4's ρ**, read from `figure_data/exp2_control_preservation.csv`: "
        "between-family Wasserstein and within-family angular preservation, both against the "
        "natural geometry. Nothing here recomputes geometry.",
        "",
        "Control sequences are fresh generator draws. The embedded draw for the four shuffle rungs "
        "predates the deterministic-seeding fix and cannot be regenerated, so identity here "
        "describes the rung's GENERATOR rather than the exact strings that were embedded; identity "
        "is a distributional property of the constraint, so this is a reproducibility limit, not a "
        "validity one.",
        "",
        "## Per-rung identity to source",
        "",
        "| rung | n | mean len | pos. identity | self-pair null | chance floor | **excess** | "
        "edit identity | aa identity | p1 | p2 | p3 | bp changed | of those, at p3 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        r = idm.loc[c]

        def g(k, r=r):
            return f"{r[k]:.3f}" if k in r and np.isfinite(r[k]) else "–"

        def pc(k, r=r):
            return f"{100 * r[k]:.1f}%" if k in r and np.isfinite(r[k]) else "–"

        L.append(
            f"| {LABEL.get(c, c)} | {int(r['n_seqs'])} | {r['len_src']:.0f} | "
            f"{g('pos_identity')} | {g('null_pos_identity')} | {g('pos_identity_chance')} | "
            f"**{r['pos_excess']:+.4f}** | {g('edit_identity')} | {g('aa_identity')} | "
            f"{g('p1_identity')} | {g('p2_identity')} | {g('p3_identity')} | "
            f"{pc('frac_bases_changed')} | {pc('frac_changes_at_p3')} |"
        )
    L += [
        "",
        "`self-pair null` is the identity between two INDEPENDENT draws of the same control: the "
        "identity its constraint forces with no information from the specific source sequence. "
        "`excess` = identity to source − that null, i.e. the identity attributable to retained "
        "source sequence.",
        "",
        f"## ρ against identity at blocks{layer}",
        "",
        "| rung | pos. identity | excess | " + " | ".join(RHO_COLS.values()) + " |",
        "|---|---|---|" + "---|" * len(RHO_COLS),
    ]
    for c in conds + ["natural"]:
        if c not in rl.index:
            continue
        pi = f"{idm.loc[c, 'pos_identity']:.3f}" if c in idm.index else "1.000 (by definition)"
        ex = f"{idm.loc[c, 'pos_excess']:+.4f}" if c in idm.index else "–"
        vals = " | ".join(f"{rl.loc[c, k]:.3f}" for k in RHO_COLS)
        L.append(f"| {LABEL.get(c, c)} | {pi} | {ex} | {vals} |")
    L += [
        "",
        "## Monotonicity tests at this layer",
        "",
        "| ρ series | identity metric | ρ_S all rungs (p) | ρ_S nucleotide rungs, n (p) | "
        "identity spread there | ρ spread there | recode ρ predicted | recode residual | "
        "extrapolation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, s in stats[stats.layer == layer].iterrows():
        ex = s.recode_extrapolation_ranges
        flag = "†" if not np.isfinite(ex) or ex > EXTRAP_LIMIT else ""
        L.append(
            f"| {s.rho_col} | {s.identity_metric} | {s.spearman_all_rungs:+.3f} "
            f"({s.p_all_rungs:.3f}) | "
            f"{s.spearman_nucleotide_rungs:+.3f}, n={int(s.n_nucleotide_rungs)} "
            f"({s.p_nucleotide_rungs:.3f}) | "
            f"{s.identity_range_nucleotide_rungs:.4f} | {s.rho_range_nucleotide_rungs:.4f} | "
            f"{s.recode_rho_predicted:+.3f}{flag} | {s.recode_rho_residual:+.3f}{flag} | "
            f"{ex:.1f}× the fitted range |"
        )
    L += [
        "",
        f"† The fitted line is calibrated over the four shuffle rungs' identity spread and then "
        f"evaluated at the recode's identity, which is more than {EXTRAP_LIMIT:.0f} such spreads "
        f'outside it. Read those two columns as "the identity axis has nowhere near enough spread '
        f'among the nucleotide rungs to say anything about the recode", never as a quantitative '
        f"prediction — several of the predictions fall outside ρ ∈ [−1, 1].",
        "",
        "`null_pos_identity` and `pos_identity_chance` are on this list as negative controls for "
        "the x-axis: both can be computed without ever looking at the source sequence, so if ρ "
        "correlates with them as well as it does with identity-to-source, the correlation is not "
        "evidence that retained source sequence drives ρ.",
    ]
    if len(within):
        L += [
            "",
            "## Within each rung, across families",
            "",
            "| rung | ρ series | families | Spearman(family identity, family ρ) | p |",
            "|---|---|---|---|---|",
        ]
        for _, s in within.iterrows():
            L.append(
                f"| {LABEL.get(s.condition, s.condition)} | {s.rho_col} | {int(s.n_families)} "
                f"| {s.spearman_identity_vs_rho:+.3f} | {s.p:.3g} |"
            )
    (RUN / "control_sequence_identity.md").write_text("\n".join(L) + "\n")
    print(f"  Saved {RUN}/control_sequence_identity.md")


# ── driver


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage",
        choices=["identity", "figure", "all"],
        default="all",
        help="identity: measure sequences and write the summary tables (the slow part). "
        "figure: join figure 4's ρ and render, from figure_data/.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="headline block for the scatter panels and the report tables "
        "(the per-block panels always cover every block).",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="thin the panel to at most this many sequences (deterministic; for a quick pass "
        "over the 11k-locus panel). Writes to a side file, never over the published tables.",
    )
    ap.add_argument("--pub", action="store_true", help="render at publication geometry.")
    args = ap.parse_args()

    if args.pub:
        pub.enable()

    per_seq_p = RUN / "control_sequence_identity_per_seq.csv.gz"
    if args.stage in ("identity", "all"):
        per_seq = measure_identity(args.max_seqs)
        if args.max_seqs:
            side = RUN / "control_sequence_identity_thinned.csv.gz"
            per_seq.to_csv(side, index=False)
            print(f"  thinned pass -> {side} (published tables untouched)")
            return
        per_seq.to_csv(per_seq_p, index=False)
        summarise_identity(per_seq).to_csv(RUN / "control_sequence_identity.csv", index=False)
        summarise_by_family(per_seq).to_csv(
            RUN / "control_sequence_identity_by_family.csv", index=False
        )
        print(f"  Saved {RUN}/control_sequence_identity{{,_by_family}}.csv (+ per-sequence rows)")
        if args.stage == "identity":
            return

    ident = figure_data.table("exp2_control_identity")
    by_family = figure_data.table("exp2_control_identity_by_family")
    rho = load_rho()
    stats = monotonicity_stats(ident, rho)
    stats.to_csv(RUN / "control_rho_vs_identity_stats.csv", index=False)
    within = within_rung_stats(by_family, args.layer)
    if len(within):
        within.to_csv(RUN / "control_identity_within_rung_stats.csv", index=False)
    figure(ident, rho, stats, args.layer)
    write_report(ident, rho, stats, within, args.layer)

    key = stats[(stats.layer == args.layer) & (stats.identity_metric == "pos_identity")]
    print(
        "\n"
        + key[
            [
                "rho_col",
                "spearman_all_rungs",
                "spearman_nucleotide_rungs",
                "identity_range_nucleotide_rungs",
                "recode_rho_predicted",
                "recode_extrapolation_ranges",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
