"""
How much of the source sequence does each composition control retain, and does the ladder's rho just
track that? Sequence-level only — no model, no GPU.
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
sys.path.insert(0, str(_REPO / "scripts" / "evo2"))
sys.path.insert(0, str(_REPO / "scripts" / "controls"))
# Labels and colors shared by the control figures.
import arcadia_style as acs  # noqa: E402
import make_control_sequences as mcs  # noqa: E402  (pure sequence ops; no torch)
from control_comparison_figure import CONDITIONS as _CONDITIONS  # noqa: E402

LABEL = {k: lbl for k, lbl, _, _ in _CONDITIONS}
COLOUR = {k: c for k, _, _, c in _CONDITIONS}
# Order controls by composition preservation, followed by the nested recode/missense pair.
LADDER = [
    "kmer6_shuffle",
    "kmer4_shuffle",
    "codon_shuffle",
    "dinuc_shuffle",
    "gc_match",
    "synonymous_recode",
    "missense_subset",
]
NESTED_PAIR = ("synonymous_recode", "missense_subset")
# Fit the identity-to-ρ relationship on nucleotide-only controls.
NUCLEOTIDE_RUNGS = [c for c in LADDER if c not in NESTED_PAIR]

DEFAULT_LAYER = 15  # blocks.15 — the layer-selection pick for the Evo2 gene panels.
# A nucleotide-rung fit evaluated more than this many fitted-ranges away from its data is flagged
# rather than reported as a prediction (see monotonicity_stats).
EXTRAP_LIMIT = 3.0


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
        elif control == "codon_shuffle":
            out[g] = mcs.codon_shuffle(s, rng)
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


# ── panels: sequences + where their rho tables live


def _read_fastas(d: Path, families: list[str]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for fam in families:
        cur, buf = None, []
        for line in (d / f"{fam}.fasta").read_text().splitlines():
            if line.startswith(">"):
                if cur:
                    seqs[cur] = "".join(buf)
                cur, buf = line[1:].split("|")[0], []
            elif line.strip():
                buf.append(line.strip().upper())
        if cur:
            seqs[cur] = "".join(buf)
    return seqs


def load_xkingdom():
    """
    Cross-kingdom KEGG panel: natural + the ON-DISK control FASTAs that were actually embedded.
    """
    nat, order = mcs.load_family_fastas()
    fam_of = {g: f for f, hdrs in order.items() for _, g in hdrs}
    fams = list(order)
    embedded = {
        c: _read_fastas(mcs.CONTROL_ROOT / c, fams)
        for c in LADDER
        if (mcs.CONTROL_ROOT / c).is_dir()
    }
    return nat, fam_of, embedded, True


def load_human_cds():
    """Matched human paralog panel, CDS input."""
    import sample_human_genes as ss  # noqa: E402  (repo panel definition; no torch)

    genes, fams, _, _ = ss.load_matched_panel(None, require_both=False, resolve_loci=False)
    cds = json.loads((_REPO / "data/cache/cds_sequences.json").read_text())
    keep = [(g, f) for g, f in zip(genes, fams, strict=False) if g in cds]
    nat = {g: cds[g].upper() for g, _ in keep}
    return nat, {g: f for g, f in keep}, {}, True


def load_mammal_cdsmask():
    """Mammalian ortholog CDS-masked arm."""
    cache = _REPO / "data/cache/mammal_embed"
    rungs = [c for c in LADDER if (cache / f"transcript_cdsmask_{c}").is_dir()]
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
    return nat, fam_of, {}, True


def _latest(pattern: str) -> Path:
    hits = sorted(_REPO.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise SystemExit(f"no results dir matches {pattern}")
    return hits[0]


PANELS = {
    "xkingdom": {
        "loader": load_xkingdom,
        "run": lambda: _latest("results/*_evo2-gene-families"),
        "layer_dir": "blocks{L}",
        "recovery_col": "spearman_geodesic_taxonomy",
        "natural_recovery": ("axisB_within_family_correlations.csv", "spearman_geodesic_taxonomy"),
        "recovery_name": "host taxonomy",
        "exact_sequences": True,
    },
    "human_cds": {
        "loader": load_human_cds,
        "run": lambda: _latest("results/*_evo2-human-panel-cds"),
        "layer_dir": "blocks{L}",
        "recovery_col": "rho_geodesic_patristic",
        "natural_recovery": ("within_family_patristic.csv", "spearman_geodesic_patristic"),
        "recovery_name": "patristic gene tree",
        "exact_sequences": False,
    },
    "mammal_cdsmask": {
        "loader": load_mammal_cdsmask,
        "run": lambda: _latest("results/*_mammalian-orthologs-transcript_cdsmask"),
        "layer_dir": "blocks{L}",
        "recovery_col": "rho_geodesic_speciestree",
        "natural_recovery": ("within_family_speciestree.csv", "spearman_geodesic_speciestree"),
        "recovery_name": "species tree",
        "exact_sequences": False,
    },
}


# ── stage 1: identity


def measure_identity(panel: str, max_seqs: int | None, verify_on_disk: bool) -> pd.DataFrame:
    """Measure source identity for every control rung in a panel."""
    spec = PANELS[panel]
    nat, fam_of, embedded, in_frame = spec["loader"]()
    if max_seqs and len(nat) > max_seqs:
        keep = sorted(nat)[:: max(1, len(nat) // max_seqs)][:max_seqs]  # deterministic thinning
        nat = {k: nat[k] for k in keep}
        fam_of = {k: fam_of[k] for k in keep}
        embedded = {c: {k: v[k] for k in keep if k in v} for c, v in embedded.items()}
    print(
        f"[{panel}] {len(nat)} sequences, {len(set(fam_of.values()))} families, "
        f"median length {int(np.median([len(s) for s in nat.values()]))} bp"
        f"{' (embedded control FASTAs on disk)' if embedded else ''}"
    )

    rungs = [c for c in LADDER if c in embedded] or _applicable_rungs(panel, in_frame)
    rows = []
    for control in rungs:
        # Draw A/B are independent of each other by construction (different seed tags), which is
        # what makes identity(A, B) a null for the constraint-forced part of identity(source, A).
        A = draw_control(control, nat, fam_of, "drawA")
        B = draw_control(control, nat, fam_of, "drawB")
        primary = embedded.get(control) or A
        source_of_primary = "embedded_fasta" if control in embedded else "fresh_draw"
        for g, s in nat.items():
            if g not in primary:
                continue
            m = pair_metrics(s, primary[g], in_frame)
            null = pair_metrics(A[g], B[g], in_frame)
            rows.append(
                {
                    "panel": panel,
                    "condition": control,
                    "key": g,
                    "family": fam_of[g],
                    "primary_from": source_of_primary,
                    **m,
                    **{
                        f"null_{k}": v for k, v in null.items() if k not in ("len_src", "len_delta")
                    },
                    # identity(source, fresh draw A): the same quantity as `pos_identity` but
                    # from a draw we control, so the two can be compared where the primary is
                    # an embedded FASTA -- a check that a fresh draw is exchangeable with it.
                    "redraw_pos_identity": pair_metrics(s, A[g], False)["pos_identity"],
                }
            )
        print(f"  {control:18} n={sum(r['condition'] == control for r in rows):5d}  done")

    if verify_on_disk and embedded:
        _verify_on_disk(nat, fam_of, embedded)
    return pd.DataFrame(rows)


def _applicable_rungs(panel: str, in_frame: bool) -> list[str]:
    """Which rungs a panel actually embedded, read off its control table rather than assumed."""
    run = PANELS[panel]["run"]()
    tables = sorted(
        run.glob(f"{PANELS[panel]['layer_dir'].format(L='*')}/controls/control_within_scores.csv")
    )
    if not tables:
        raise SystemExit(f"{panel}: no controls/control_within_scores.csv under {run}")
    present = set(pd.read_csv(tables[0])["condition"])
    return [c for c in LADDER if c in present]


def _verify_on_disk(nat: dict, fam_of: dict, embedded: dict) -> None:
    """
    Confirm the on-disk control FASTAs are the ones make_control_sequences generates, by replaying
    its
    exact RNG discipline.
    """
    _, order = mcs.load_family_fastas()
    usage = mcs.build_family_codon_usage(
        {f: [nat[g] for _, g in h] for f, h in order.items() if all(g in nat for _, g in h)}
    )
    for control, disk in embedded.items():
        rng = random.Random(mcs.SEED)
        same = total = 0
        for fam, hdrs in order.items():
            for _, g in hdrs:
                s = nat.get(g)
                if s is None:
                    continue
                if control == "gc_match":
                    c = mcs.gc_match(s, rng)
                elif control == "dinuc_shuffle":
                    c = mcs.dinuc_shuffle(s, rng)
                elif control == "kmer4_shuffle":
                    c = mcs.klet_shuffle(s, 4, rng)
                elif control == "kmer6_shuffle":
                    c = mcs.klet_shuffle(s, 6, rng)
                elif control == "codon_shuffle":
                    c = mcs.codon_shuffle(s, rng)
                else:
                    c = mcs.synonymous_recode(s, usage[fam], rng)
                total += 1
                same += int(disk.get(g) == c)
        print(
            f"  [verify] {control:18} {same}/{total} on-disk sequences reproduce from SEED="
            f"{mcs.SEED}"
        )


def summarise_identity(per_seq: pd.DataFrame) -> pd.DataFrame:
    """Per-rung means, plus the two derived quantities the argument turns on:
    `pos_excess` / `edit_excess` = identity to source minus the self-pair null."""
    num = [
        c for c in per_seq.columns if per_seq[c].dtype.kind == "f" or c in ("len_src", "len_delta")
    ]
    g = per_seq.groupby(["panel", "condition"])
    out = g[num].mean()
    out["pos_identity_sd"] = g["pos_identity"].std()
    out["n_seqs"] = g.size()
    # Standard error of the PAIRED per-sequence excess, so "excess is zero" can be read against its
    # own noise rather than asserted.
    exc = per_seq.assign(_e=per_seq["pos_identity"] - per_seq["null_pos_identity"]).groupby(
        ["panel", "condition"]
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
    # Same draw-vs-draw spread, as the error bar on a single draw's mean identity.
    out["redraw_delta"] = out["redraw_pos_identity"] - out["pos_identity"]
    return out.reset_index()


# ── stage 2: join rho and test monotonicity


def load_rho(panel: str) -> pd.DataFrame:
    """
    Tidy (condition, layer, rho_preservation, rho_recovery) for every layer of one panel, from that
    panel's own controls/control_within_scores.csv, averaged over families.
    """
    spec = PANELS[panel]
    run = spec["run"]()
    rows = []
    for d in sorted(run.glob(spec["layer_dir"].format(L="*"))):
        layer = int("".join(ch for ch in d.name if ch.isdigit()))
        t = d / "controls" / "control_within_scores.csv"
        if not t.exists():
            continue
        df = pd.read_csv(t)
        rec = spec["recovery_col"] if spec["recovery_col"] in df.columns else None
        for cond, grp in df.groupby("condition"):
            rows.append(
                {
                    "panel": panel,
                    "layer": layer,
                    "condition": cond,
                    "n_families": int(grp["family"].nunique()),
                    "rho_preservation": grp["rho_geodesic_vs_natural"].mean(),
                    "rho_recovery": grp[rec].mean() if rec else np.nan,
                }
            )
        nat_file, nat_col = spec["natural_recovery"]
        nat_p = d / nat_file
        nat_rec = np.nan
        if nat_p.exists():
            n = pd.read_csv(nat_p)
            fams = set(df["family"])
            nat_rec = n.loc[n["family"].isin(fams), nat_col].mean()
        have = {r["condition"] for r in rows if r["layer"] == layer}
        if "natural" in have:  # some panels already write a natural row; fill its recovery
            for r in rows:
                if (
                    r["layer"] == layer
                    and r["condition"] == "natural"
                    and not np.isfinite(r["rho_recovery"])
                ):
                    r["rho_recovery"] = nat_rec
        else:
            rows.append(
                {
                    "panel": panel,
                    "layer": layer,
                    "condition": "natural",
                    "n_families": int(df["family"].nunique()),
                    "rho_preservation": 1.0,
                    "rho_recovery": nat_rec,
                }
            )
    if not rows:
        raise SystemExit(f"{panel}: no control tables under {run}")
    return pd.DataFrame(rows)


def _fit_residual(x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> tuple[float, float]:
    """(predicted y at x0, y0 - predicted) from a least-squares line through (x, y)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return np.nan, np.nan
    m, b = np.polyfit(x[ok], y[ok], 1)
    pred = m * x0 + b
    return float(pred), float(y0 - pred)


def monotonicity_stats(ident: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    """Per (panel, layer, rho column, identity metric): is rho monotone in identity across rungs?"""
    out = []
    for (panel,), ig in ident.groupby(["panel"]):
        idm = ig.set_index("condition")
        for (layer,), rg in rho[rho["panel"] == panel].groupby(["layer"]):
            r = rg.set_index("condition")
            for rho_col in ("rho_preservation", "rho_recovery"):
                # null_pos_identity and pos_identity_chance are swept alongside the real metrics as
                # negative controls on the x-axis itself: both are computable without ever looking
                # at
                # the source sequence, so if ρ correlates with them just as well, the correlation is
                # not evidence that retained source sequence drives ρ.
                for x_col in (
                    "pos_identity",
                    "edit_identity",
                    "pos_excess",
                    "aa_identity",
                    "null_pos_identity",
                    "pos_identity_chance",
                ):
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
                    nuc = [i for i, c in enumerate(conds) if c != "synonymous_recode"]
                    rho_all = spearmanr(x, y)
                    rho_nuc = spearmanr(x[nuc], y[nuc]) if len(nuc) >= 3 else (np.nan, np.nan)
                    pred, resid, extrap = (np.nan, np.nan, np.nan)
                    if "synonymous_recode" in conds and nuc:
                        j = conds.index("synonymous_recode")
                        pred, resid = _fit_residual(x[nuc], y[nuc], x[j], y[j])
                        span = float(np.ptp(x[nuc]))
                        extrap = ((x[j] - max(x[nuc])) / span) if span > 0 else np.inf
                    out.append(
                        {
                            "panel": panel,
                            "layer": layer,
                            "rho_col": rho_col,
                            "identity_metric": x_col,
                            "n_rungs": len(conds),
                            "spearman_all_rungs": rho_all[0],
                            "p_all_rungs": rho_all[1],
                            "spearman_nucleotide_rungs": rho_nuc[0],
                            "p_nucleotide_rungs": rho_nuc[1],
                            "identity_range_nucleotide_rungs": float(np.ptp(x[nuc]))
                            if nuc
                            else np.nan,
                            "rho_range_nucleotide_rungs": float(np.ptp(y[nuc])) if nuc else np.nan,
                            "recode_rho_predicted": pred,
                            "recode_rho_residual": resid,
                            "recode_extrapolation_ranges": extrap,
                        }
                    )
    return pd.DataFrame(out)


def within_rung_stats(per_seq: pd.DataFrame, panel: str, layer: int) -> pd.DataFrame:
    """
    The test that does not rely on comparing rungs to each other: WITHIN one rung, across families,
    does a family whose sequences retained more identity get a higher ρ?
    """
    spec = PANELS[panel]
    d = spec["run"]() / spec["layer_dir"].format(L=layer) / "controls" / "control_within_scores.csv"
    if not d.exists():
        return pd.DataFrame()
    rho = pd.read_csv(d)
    rec = spec["recovery_col"] if spec["recovery_col"] in rho.columns else None
    fam_id = (
        per_seq.groupby(["condition", "family"])[["pos_identity", "edit_identity"]]
        .mean()
        .reset_index()
    )
    m = fam_id.merge(rho, on=["condition", "family"], how="inner")
    out = []
    for cond, g in m.groupby("condition"):
        for rho_col in ["rho_geodesic_vs_natural"] + ([rec] if rec else []):
            ok = g[["pos_identity", rho_col]].dropna()
            if len(ok) < 5 or np.ptp(ok["pos_identity"]) == 0:
                continue
            s = spearmanr(ok["pos_identity"], ok[rho_col])
            out.append(
                {
                    "panel": panel,
                    "layer": layer,
                    "condition": cond,
                    "rho_col": rho_col,
                    "n_families": len(ok),
                    "spearman_identity_vs_rho": s[0],
                    "p": s[1],
                }
            )
    return pd.DataFrame(out)


# ── figures


def _style() -> None:
    """Apply the repo's figure style."""
    for mod, fn in (("arcadia_style", "setup"), ("plot_utils", "set_pub_style")):
        try:
            getattr(__import__(mod), fn)()
            return
        except Exception as e:  # noqa: BLE001 — style is cosmetic; report and carry on
            print(f"  [style] {mod}.{fn}() unavailable ({type(e).__name__}: {e})")
    print("  [style] falling back to matplotlib defaults")


def _scatter_rho_vs_identity(
    lo,
    hi,
    panel: str,
    idm: pd.DataFrame,
    rl: pd.DataFrame,
    stats: pd.DataFrame,
    conds: list[str],
    layer: int,
    rho_col: str,
    title: str,
) -> None:
    """rho against positional identity to source, on a broken x-axis."""
    xs = np.array([idm.loc[c, "pos_identity"] for c in conds])
    ys = np.array([rl.loc[c, rho_col] for c in conds])
    nuc = [i for i, c in enumerate(conds) if c not in NESTED_PAIR]
    pair = [i for i, c in enumerate(conds) if c in NESTED_PAIR]
    nat_y = rl.loc["natural", rho_col] if "natural" in rl.index else np.nan
    for ax in (lo, hi):
        for i, c in enumerate(conds):
            ax.scatter(
                xs[i],
                ys[i],
                s=60,
                color=COLOUR.get(c, acs.CONTROL_FALLBACK),
                zorder=3,
                edgecolor=acs.apc.white,
                linewidth=0.5,
                marker="D" if c in NESTED_PAIR else "o",
            )
        if np.isfinite(nat_y):
            ax.scatter(
                [1.0],
                [nat_y],
                s=60,
                facecolor="none",
                zorder=3,
                edgecolor=COLOUR.get("natural", acs.REFERENCE_LINE),
                linewidth=1.2,
            )
        if len(nuc) >= 2 and np.ptp(xs[nuc]) > 0:
            m, b = np.polyfit(xs[nuc], ys[nuc], 1)
            ax.plot(
                np.linspace(0.2, 1.05, 60),
                m * np.linspace(0.2, 1.05, 60) + b,
                ls=":",
                lw=0.7,
                color=acs.GRID,
            )
            fx = np.linspace(min(xs[nuc]), max(xs[nuc]), 20)
            ax.plot(fx, m * fx + b, ls="-", lw=1.4, color=acs.ANNOTATION)
    if np.isfinite(nat_y):
        hi.annotate(
            "natural\n(reference)",
            (1.0, nat_y),
            textcoords="offset points",
            xytext=(-3, 8),
            fontsize=5.5,
            ha="right",
            color=COLOUR.get("natural", acs.REFERENCE_LINE),
        )
    # Label the cluster on the zoomed axis. The nucleotide rungs can sit within 0.01 of each other
    # in
    # BOTH coordinates, so labels go in an evenly spaced column with leader lines rather than as
    # per-point offsets, which collide.
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
    # The two nested rungs sit close in x (0.77 vs 0.88), so once BOTH are present they have
    # to be pulled apart into a label column. With only one present that column would drag a long
    # leader line across the panel for nothing, so it gets a plain offset instead.
    ranked = sorted(pair, key=lambda i: ys[i])
    for rank, i in enumerate(ranked):
        if len(ranked) == 1:
            hi.annotate(
                LABEL.get(conds[i], conds[i]),
                xy=(xs[i], ys[i]),
                textcoords="offset points",
                xytext=(0, 10),
                fontsize=5.5,
                ha="center",
                fontweight="bold",
            )
            continue
        hi.annotate(
            LABEL.get(conds[i], conds[i]),
            xy=(xs[i], ys[i]),
            xycoords="data",
            xytext=(0.10, 0.12 + 0.64 * rank / (len(ranked) - 1)),
            textcoords="axes fraction",
            fontsize=5.5,
            va="center",
            ha="left",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.4, color=acs.GRID, shrinkA=0, shrinkB=3),
        )
    pad = max(np.ptp(xs[nuc]) * 0.55, 0.004) if nuc else 0.01
    lo.set_xlim(min(xs[nuc]) - pad, max(xs[nuc]) + pad * 2.6)
    hi.set_xlim((min(xs[pair]) - 0.10) if pair else 0.9, 1.05)
    allv = [v for v in list(ys) + [nat_y] if np.isfinite(v)]
    span = max(np.ptp(allv), 0.05)
    lo.set_ylim(min(allv) - 0.12 * span, max(allv) + 0.16 * span)
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
        (stats.panel == panel)
        & (stats.layer == layer)
        & (stats.rho_col == rho_col)
        & (stats.identity_metric == "pos_identity")
    ]
    if len(st):
        s = st.iloc[0]
        lo.text(
            0.03,
            0.97,
            f"ρ_S(identity, ρ) all rungs = {s.spearman_all_rungs:+.2f}\n"
            f"nucleotide rungs only = {s.spearman_nucleotide_rungs:+.2f}\n"
            f"identity spread there = {s.identity_range_nucleotide_rungs:.3f}\n"
            f"nested pair sits {s.recode_extrapolation_ranges:.0f}× that spread away\n"
            f"— line: fit over the nucleotide rungs (solid)\n"
            f"   then extrapolated (dotted, not a prediction)",
            transform=lo.transAxes,
            va="top",
            fontsize=5.5,
            bbox=dict(fc=acs.apc.white, ec=acs.GRID, lw=0.4, pad=2),
        )
    lo.set_ylabel(f"Spearman ρ, {rho_col.split('_')[1]}")
    lo.set_xlabel("positional identity to source", x=1.0, ha="center")
    lo.set_title(title, fontsize=8, loc="left", x=0.0)


def figure(
    panel: str,
    ident: pd.DataFrame,
    rho: pd.DataFrame,
    stats: pd.DataFrame,
    layer: int,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    spec = PANELS[panel]
    idm = ident.set_index("condition")
    rl = rho[(rho["panel"] == panel) & (rho["layer"] == layer)].set_index("condition")
    conds = [c for c in LADDER if c in idm.index and c in rl.index]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.28)
    axes = {(1, j): fig.add_subplot(gs[1, j]) for j in range(3)}
    axes[(0, 0)] = fig.add_subplot(gs[0, 0])
    # B and C get a BROKEN x-axis: the five nucleotide rungs live inside ~2.5 points of identity
    # while recode sits near 0.77, so a single linear axis collapses the whole ladder into one blob.
    broken = {}
    for j in (1, 2):
        sub = gs[0, j].subgridspec(1, 2, width_ratios=[3, 2], wspace=0.06)
        lo = fig.add_subplot(sub[0])
        hi = fig.add_subplot(sub[1], sharey=lo)
        broken[j] = (lo, hi)
    fig.suptitle(
        f"Is the control ladder's ρ just nucleotide identity to source?  [{panel}, blocks.{layer}]",
        fontsize=11,
        y=0.985,
    )

    # (A) the identity decomposition: identity to source vs the self-pair null vs the chance floor.
    ax = axes[(0, 0)]
    x = np.arange(len(conds))
    w = 0.27
    ax.bar(
        x - w,
        [idm.loc[c, "pos_identity"] for c in conds],
        w,
        color=[COLOUR.get(c, acs.CONTROL_FALLBACK) for c in conds],
        label="to source",
    )
    ax.bar(
        x,
        [idm.loc[c, "null_pos_identity"] for c in conds],
        w,
        facecolor="none",
        edgecolor=acs.ANNOTATION,
        hatch="///",
        linewidth=0.6,
        label="two independent draws (null)",
    )
    ax.bar(
        x + w,
        [idm.loc[c, "pos_identity_chance"] for c in conds],
        w,
        color=acs.MISSING,
        label="composition chance",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(c, c) for c in conds], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("positional nucleotide identity")
    ax.set_title(
        "A. Identity to source is the control's own\nconstraint, not retained sequence", fontsize=8
    )
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_ylim(0, 1.05)

    # (B, C) rho vs identity, with the nucleotide-only fit extended to the nested pair.
    for j, rho_col, name in (
        (1, "rho_preservation", "vs natural geodesic (PRESERVATION)"),
        (2, "rho_recovery", f"vs {spec['recovery_name']} (RECOVERY)"),
    ):
        lo, hi = broken[j]
        _scatter_rho_vs_identity(
            lo,
            hi,
            panel,
            idm,
            rl,
            stats,
            conds,
            layer,
            rho_col,
            f"{'B' if j == 1 else 'C'}. ρ {name}   (broken x-axis)",
        )

    # (D) excess identity: the part of identity that is retained SOURCE sequence.
    ax = axes[(1, 0)]
    ex = np.array([idm.loc[c, "pos_excess"] for c in conds])
    se = np.array(
        [idm.loc[c, "pos_excess_se"] if "pos_excess_se" in idm.columns else np.nan for c in conds]
    )
    ax.bar(
        range(len(conds)),
        ex,
        color=[COLOUR.get(c, acs.CONTROL_FALLBACK) for c in conds],
        yerr=1.96 * se,
        error_kw=dict(lw=0.8, capsize=2, ecolor=acs.ANNOTATION),
    )
    ax.axhline(0, color=acs.ZERO_LINE, lw=0.6)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABEL.get(c, c) for c in conds], rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("identity to source − self-pair null")
    lim = max(1.5 * np.nanmax(np.abs(ex) + 1.96 * se), 0.004)
    ax.set_ylim(-lim, lim)
    ax.set_title(
        "D. Retained SOURCE identity, once each rung's own\nconstraint is netted out (±95% CI)",
        fontsize=8,
    )

    # (E, F) is the monotonicity a layer-specific accident? Spearman per layer.
    for ax, rho_col, tag in (
        (axes[(1, 1)], "rho_preservation", "E"),
        (axes[(1, 2)], "rho_recovery", "F"),
    ):
        s = stats[
            (stats.panel == panel)
            & (stats.rho_col == rho_col)
            & (stats.identity_metric == "pos_identity")
        ].sort_values("layer")
        if len(s):
            ax.plot(
                s.layer,
                s.spearman_all_rungs,
                "-o",
                ms=2.5,
                lw=1.0,
                color=acs.apc.amber,
                label="all rungs (recode included)",
            )
            ax.plot(
                s.layer,
                s.spearman_nucleotide_rungs,
                "-s",
                ms=2.5,
                lw=1.0,
                color=acs.apc.dusk,
                label="nucleotide rungs only",
            )
            s2 = stats[
                (stats.panel == panel)
                & (stats.rho_col == rho_col)
                & (stats.identity_metric == "pos_excess")
            ].sort_values("layer")
            if len(s2):
                ax.plot(
                    s2.layer,
                    s2.spearman_all_rungs,
                    "-^",
                    ms=2.5,
                    lw=1.0,
                    color=acs.CONTROL_FALLBACK,
                    label="all rungs, EXCESS identity",
                )
        ax.axhline(0, color=acs.ZERO_LINE, lw=0.6)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Evo2 block")
        ax.set_ylabel("Spearman(identity, ρ) across rungs")
        ax.set_title(f"{tag}. Monotonicity per layer — {rho_col.split('_')[1]}", fontsize=8)
        ax.legend(fontsize=5.5, loc="lower left")

    for ext in ("png", "pdf"):  # no tight_layout: it fights the nested broken-axis subgridspecs
        fig.savefig(out_dir / f"control_rho_vs_identity.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir}/control_rho_vs_identity.{{png,pdf}}")


def figure_table(
    panel: str,
    ident: pd.DataFrame,
    rho: pd.DataFrame,
    stats: pd.DataFrame,
    layer: int,
    out_dir: Path,
) -> None:
    """
    The headline figure: per-rung numbers as a table on the left, the rho-vs-identity scatter on the
    right, so the values and the shape are read together.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    spec = PANELS[panel]
    idm = ident.set_index("condition")
    rl = rho[(rho["panel"] == panel) & (rho["layer"] == layer)].set_index("condition")
    conds = [c for c in LADDER if c in idm.index and c in rl.index]
    if not conds:
        print("  [skip] control_identity_table: no rung has both identity and ρ")
        return
    show = conds + (["natural"] if "natural" in rl.index else [])

    fig = plt.figure(figsize=(15.2, 0.42 * len(show) + 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.10)
    axT = fig.add_subplot(gs[0])
    axT.axis("off")

    head = [
        "control",
        "nt identity\nto source",
        "self-pair\nnull",
        "excess\n(retained source)",
        "edit-distance\nidentity",
        "amino-acid\nidentity",
        "mean ρ\npreservation",
        f"mean ρ recovery\n(vs {spec['recovery_name']})",
    ]
    rows, colours = [], []
    for c in show:
        nat = c == "natural"
        r = None if nat else idm.loc[c]

        def g(k, nat=nat, r=r):
            return "1.000" if nat else (f"{r[k]:.3f}" if k in r and np.isfinite(r[k]) else "–")

        def rr(k, c=c):
            return f"{rl.loc[c, k]:.3f}" if np.isfinite(rl.loc[c, k]) else "–"

        rows.append(
            [
                LABEL.get(c, c),
                g("pos_identity"),
                g("null_pos_identity"),
                "–" if nat else f"{r['pos_excess']:+.4f}",
                g("edit_identity"),
                g("aa_identity"),
                rr("rho_preservation"),
                rr("rho_recovery"),
            ]
        )
        # Tint the name cell with the rung's colour; shade the whole row for the nested pair.
        row_bg = acs.apc.dawn if c in NESTED_PAIR else acs.apc.white
        colours.append([COLOUR.get(c, acs.CONTROL_FALLBACK)] + [row_bg] * (len(head) - 1))
    tbl = axT.table(
        cellText=rows,
        colLabels=head,
        cellColours=colours,
        colWidths=[0.31] + [0.099] * (len(head) - 1),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.4)
    tbl.scale(1, 2.0)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor(acs.GRID)
        if i == 0:
            cell.set_text_props(fontweight="bold", fontsize=6.0)
            cell.set_facecolor(acs.apc.gray)
        else:
            if j == 0:
                cell.set_text_props(ha="left", fontsize=6.2, color=acs.apc.white, fontweight="bold")
            if j == 3:  # the excess column is what carries the argument
                cell.set_text_props(fontweight="bold")
    axT.set_title(
        f"Per-control identity to source and the ρ it produces  —  {panel}, blocks.{layer}",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    note = (
        "excess = identity to source − self-pair null (the identity between two independent "
        "draws of the SAME control). It isolates the identity attributable to retained SOURCE\n"
        "sequence, as opposed to the identity the control's own constraint forces. ρ are means "
        "over families at this block."
    )
    if len([c for c in NESTED_PAIR if c in conds]) == 2:
        note += (
            "\nShaded rows are the nested pair: missense_subset edits a strict SUBSET of the "
            "bases synonymous_recode edited, so it moves FEWER nucleotides while damaging\n"
            "the protein the recode kept. A lower ρ there cannot be nucleotide loss."
        )
    axT.text(
        0.0, -0.02, note, transform=axT.transAxes, fontsize=5.9, va="top", color=acs.ANNOTATION
    )

    sub = gs[1].subgridspec(1, 2, width_ratios=[3, 2], wspace=0.06)
    lo = fig.add_subplot(sub[0])
    hi = fig.add_subplot(sub[1], sharey=lo)
    _scatter_rho_vs_identity(
        lo,
        hi,
        panel,
        idm,
        rl,
        stats,
        conds,
        layer,
        "rho_preservation",
        "ρ (control vs natural geodesic) against identity to source",
    )
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"control_identity_table.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir}/control_identity_table.{{png,pdf}}")


def figure_within_rung(panel: str, per_seq: pd.DataFrame, layer: int, out_dir: Path) -> None:
    """Family-level scatter of identity vs ρ inside each rung — the within-condition version of the
    same question, where identity varies for reasons other than the rung's definition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = PANELS[panel]
    t = spec["run"]() / spec["layer_dir"].format(L=layer) / "controls" / "control_within_scores.csv"
    if not t.exists():
        return
    _style()
    rho = pd.read_csv(t)
    fam_id = per_seq.groupby(["condition", "family"])["pos_identity"].mean().reset_index()
    m = fam_id.merge(rho, on=["condition", "family"], how="inner")
    conds = [c for c in LADDER if c in set(m["condition"])]
    fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds, strict=False):
        g = m[m["condition"] == c]
        ax.scatter(
            g["pos_identity"],
            g["rho_geodesic_vs_natural"],
            s=12,
            color=COLOUR.get(c, acs.CONTROL_FALLBACK),
            edgecolor=acs.apc.white,
            linewidth=0.3,
        )
        ok = g[["pos_identity", "rho_geodesic_vs_natural"]].dropna()
        if len(ok) >= 5 and np.ptp(ok["pos_identity"]) > 0:
            s = spearmanr(ok["pos_identity"], ok["rho_geodesic_vs_natural"])
            ax.set_title(f"{LABEL.get(c, c)}\nρ_S={s[0]:+.2f}, p={s[1]:.2g}", fontsize=6.5)
        ax.set_xlabel("family mean identity", fontsize=6)
    axes[0].set_ylabel("ρ (control vs natural)", fontsize=6)
    fig.suptitle(
        f"Within each rung, across families: does more retained identity mean higher ρ?  "
        f"[{panel}, blocks.{layer}]",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"control_identity_within_rung.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_dir}/control_identity_within_rung.{{png,pdf}}")


# ── report


def write_report(
    panel: str,
    ident: pd.DataFrame,
    rho: pd.DataFrame,
    stats: pd.DataFrame,
    within: pd.DataFrame,
    layer: int,
    out_dir: Path,
) -> None:
    spec = PANELS[panel]
    idm = ident.set_index("condition")
    rl = rho[(rho.panel == panel) & (rho.layer == layer)].set_index("condition")
    conds = [c for c in LADDER if c in idm.index]
    L = [
        f"# Nucleotide identity of each composition control to its source — {panel}",
        "",
        f"Run: `{spec['run']()}`  ·  headline layer: `blocks.{layer}`  ·  "
        f"recovery ground truth: {spec['recovery_name']}",
        "Control sequences: "
        + (
            "the embedded FASTAs on disk (exact)"
            if spec["exact_sequences"]
            else "fresh generator draws; the hash-seeded embedded draw cannot be regenerated"
        ),
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
        "## ρ against identity",
        "",
        "| rung | pos. identity | excess | ρ preservation | ρ recovery |",
        "|---|---|---|---|---|",
    ]
    for c in conds + ["natural"]:
        if c not in rl.index:
            continue
        pi = f"{idm.loc[c, 'pos_identity']:.3f}" if c in idm.index else "1.000 (by definition)"
        ex = f"{idm.loc[c, 'pos_excess']:+.4f}" if c in idm.index else "–"
        L.append(
            f"| {LABEL.get(c, c)} | {pi} | {ex} | "
            f"{rl.loc[c, 'rho_preservation']:.3f} | {rl.loc[c, 'rho_recovery']:.3f} |"
        )
    L += [
        "",
        "## Monotonicity tests at this layer",
        "",
        "| ρ | identity metric | ρ_S all rungs | ρ_S nucleotide rungs | identity spread there | "
        "ρ spread there | recode ρ predicted | recode residual | extrapolation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, s in stats[(stats.panel == panel) & (stats.layer == layer)].iterrows():
        ex = s.recode_extrapolation_ranges
        flag = "†" if not np.isfinite(ex) or ex > EXTRAP_LIMIT else ""
        L.append(
            f"| {s.rho_col} | {s.identity_metric} | {s.spearman_all_rungs:+.3f} | "
            f"{s.spearman_nucleotide_rungs:+.3f} | "
            f"{s.identity_range_nucleotide_rungs:.4f} | {s.rho_range_nucleotide_rungs:.4f} | "
            f"{s.recode_rho_predicted:+.3f}{flag} | {s.recode_rho_residual:+.3f}{flag} | "
            f"{ex:.1f}× the fitted range |"
        )
    L += [
        "",
        f"† The fitted line is calibrated over the nucleotide rungs' identity spread and then "
        f"evaluated at recode's identity, which is more than {EXTRAP_LIMIT:.0f} such spreads "
        f'outside it. Read the predicted/residual columns as "the identity axis has nowhere near '
        f'enough spread among the nucleotide rungs to say anything about recode", not as a '
        f"quantitative prediction.",
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
            "| rung | ρ column | families | Spearman(family identity, family ρ) | p |",
            "|---|---|---|---|---|",
        ]
        for _, s in within.iterrows():
            L.append(
                f"| {LABEL.get(s.condition, s.condition)} | {s.rho_col} | {int(s.n_families)} "
                f"| {s.spearman_identity_vs_rho:+.3f} | {s.p:.3g} |"
            )
    (out_dir / "control_sequence_identity.md").write_text("\n".join(L) + "\n")
    print(f"  Saved {out_dir}/control_sequence_identity.md")


# ── driver


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--panel", nargs="+", default=list(PANELS), choices=list(PANELS))
    ap.add_argument(
        "--stage",
        choices=["identity", "figure", "all"],
        default="all",
        help="identity: measure sequences and cache the CSVs (the slow part). "
        "figure: join ρ and plot from the cached CSVs.",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help="headline block for the scatter panels and the report table "
        "(the per-layer panels always cover every layer).",
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="thin each panel to at most this many sequences (deterministic; for a "
        "quick pass on the 11k-locus mammal panel).",
    )
    ap.add_argument(
        "--verify-on-disk",
        action="store_true",
        help="(xkingdom) replay make_control_sequences' RNG discipline and report how "
        "many on-disk control sequences it reproduces.",
    )
    args = ap.parse_args()

    for panel in args.panel:
        run = PANELS[panel]["run"]()
        print(f"\n{'=' * 78}\n{panel}  ->  {run}\n{'=' * 78}")
        per_seq_p = run / "control_sequence_identity_per_seq.csv.gz"
        summary_p = run / "control_sequence_identity.csv"

        if args.stage in ("identity", "all"):
            per_seq = measure_identity(panel, args.max_seqs, args.verify_on_disk)
            per_seq.to_csv(per_seq_p, index=False)
            summarise_identity(per_seq).to_csv(summary_p, index=False)
            print(f"  Saved {summary_p} (+ per-sequence rows)")
        if args.stage == "identity":
            continue
        if not per_seq_p.exists():
            raise SystemExit(f"{panel}: {per_seq_p} missing — run --stage identity first")

        # Re-derive the summary from the cached per-sequence rows rather than reading the CSV, so a
        # change to summarise_identity lands on a --stage figure re-run without re-measuring.
        per_seq = pd.read_csv(per_seq_p)
        ident = summarise_identity(per_seq)
        ident.to_csv(summary_p, index=False)
        rho = load_rho(panel)
        rho.to_csv(run / "control_rho_by_layer.csv", index=False)
        stats = monotonicity_stats(ident, rho)
        stats.to_csv(run / "control_rho_vs_identity_stats.csv", index=False)
        within = within_rung_stats(per_seq, panel, args.layer)
        if len(within):
            within.to_csv(run / "control_identity_within_rung_stats.csv", index=False)
        figure_table(panel, ident, rho, stats, args.layer, run)
        figure(panel, ident, rho, stats, args.layer, run)
        figure_within_rung(panel, per_seq, args.layer, run)
        write_report(panel, ident, rho, stats, within, args.layer, run)

        cols = [
            "condition",
            "n_seqs",
            "pos_identity",
            "null_pos_identity",
            "pos_excess",
            "edit_identity",
            "aa_identity",
        ]
        print("\n" + ident[[c for c in cols if c in ident.columns]].round(4).to_string(index=False))
        key = stats[(stats.layer == args.layer) & (stats.identity_metric == "pos_identity")]
        print(
            "\n"
            + key[
                [
                    "rho_col",
                    "spearman_all_rungs",
                    "spearman_nucleotide_rungs",
                    "identity_range_nucleotide_rungs",
                    "rho_range_nucleotide_rungs",
                    "recode_rho_residual",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
