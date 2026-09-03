"""Where does the recovery gain come from — toward platypus, or merely away from human? CPU only."""

from __future__ import annotations
import argparse
import gzip
import importlib.util
import json
import random
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from alignment_metrics import nt_aligner, read_fasta  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sn = _load("sites_nt")
diagnostic_sites_nt, site_pairs_nt = _sn.diagnostic_sites_nt, _sn.site_pairs_nt
_rs = _load("stage4_rescore_nt")
private_bp_subset = _rs.private_bp_subset
ORTHO_DIR, PLATYPUS = _rs.ORTHO_DIR, _rs.PLATYPUS

BASELINE = "unsteered"
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
GC_BASES = frozenset("GC")
# `shared_not_private` is the EXCLUSIVE complement: platypus differs from human, but at least one
# other sampled mammal carries the platypus base too. private and shared_not_private partition
# platy_not_human, so the pair separates "platypus-specific" from "merely non-human" -- which the
# nested pair private c platy_not_human cannot, since 29% of the loose set IS the strict set.
SITE_SETS = ("private", "shared_not_private", "platy_not_human")
GC_CLASSES = ("gc_up", "gc_down", "gc_neutral")
_G: dict = {}


# site annotation
def gc_class(h: str, p: str) -> str:
    """How the human -> platypus substitution at a private site moves GC content."""
    hg, pg = h in GC_BASES, p in GC_BASES
    if hg == pg:
        return "gc_neutral"
    return "gc_up" if pg else "gc_down"


def ortholog_votes(
    plat_cds: str, cont_offset: int, sites: dict[int, tuple[str, str]], orthologs: dict[str, str]
) -> tuple[dict[int, dict[str, int]], int]:
    """
    {site index -> {base: how many sampled mammals carry it}} and the number of voting species.
    """
    votes: dict[int, dict[str, int]] = {i: {} for i in sites}
    voters = 0
    if not sites or not orthologs:
        return votes, 0
    for sp, ocds in orthologs.items():
        if sp == PLATYPUS or not ocds:
            continue
        try:
            a = nt_aligner().align(ocds, plat_cds)[0]
        except Exception:  # noqa: BLE001
            continue
        voters += 1
        o_aln, p_aln = str(a[0]), str(a[1])
        pcur = 0
        for oc, pc in zip(o_aln, p_aln, strict=False):
            if pc == "-":
                continue
            key = pcur - cont_offset
            if key in votes and oc in CODE:
                votes[key][oc] = votes[key].get(oc, 0) + 1
            pcur += 1
    return votes, voters


def consensus_base(vote: dict[str, int]) -> int:
    """Modal base among the voting mammals as a code, or -1 for no votes / a tie for modal."""
    if not vote:
        return -1
    top = max(vote.values())
    winners = [b for b, n in vote.items() if n == top]
    return CODE[winners[0]] if len(winners) == 1 else -1


def gen_at_target(gen: str, target: str) -> np.ndarray:
    """
    Base the generation contributes at each position of `target`, as codes 0-3, -1 = not covered.
    """
    out = np.full(len(target), -1, dtype=np.int8)
    if not gen or not target:
        return out
    a = nt_aligner().align(gen, target)[0]
    g_aln, t_aln = str(a[0]), str(a[1])
    tcur = 0
    for gc, tc in zip(g_aln, t_aln, strict=False):
        if tc == "-":
            continue
        out[tcur] = CODE.get(gc, -1)
        tcur += 1
    return out


# the composition-matched shuffle null
def analytic_null(y: np.ndarray, p: np.ndarray, strat: np.ndarray) -> float:
    """EXACT expected recovery when base calls are permuted within (human base, codon position)."""
    if y.size == 0:
        return np.nan
    tot = 0.0
    for s in np.unique(strat):
        m = strat == s
        ys, ps = y[m], p[m]
        cnt = np.bincount(ys, minlength=4)
        tot += float(cnt[ps].sum()) / ys.size
    return 100.0 * tot / y.size


def permutation_null(
    y: np.ndarray, p: np.ndarray, strat: np.ndarray, n_shuffle: int, rng: np.random.Generator
) -> np.ndarray:
    """Recovery percentage under `n_shuffle` within-stratum permutations of the base calls."""
    if y.size == 0 or n_shuffle <= 0:
        return np.full(max(n_shuffle, 1), np.nan)
    hits = np.zeros(n_shuffle, dtype=np.int32)
    for s in np.unique(strat):
        m = strat == s
        ys, ps = y[m], p[m]
        if ys.size == 1:
            hits += int(ys[0] == ps[0])
            continue
        perm = rng.permuted(np.tile(ys, (n_shuffle, 1)), axis=1)
        hits += (perm == ps).sum(axis=1, dtype=np.int32)
    return 100.0 * hits / y.size


# per-record metrics
def record_metrics(calls: np.ndarray, site: dict, n_shuffle: int, rng: np.random.Generator) -> dict:
    """A / L / C / K, the GC-class breakdown and the shuffle null for ONE generation x site set."""
    idx, h, p = site["idx"], site["h"], site["p"]
    n_sites = idx.size
    out = {"n_sites": int(n_sites)}
    if n_sites == 0:
        return {
            **out,
            "n_covered": 0,
            "K": np.nan,
            "A_cov": np.nan,
            "A_all": np.nan,
            "L": np.nan,
            "C": np.nan,
            "n_eq_P": 0,
            "n_ne_H": 0,
            "A_null": np.nan,
            "A_null_perm_mean": np.nan,
            "A_null_perm_sd": np.nan,
            "excess": np.nan,
            "A_M": np.nan,
            "C_M": np.nan,
            "frac_M_eq_P": np.nan,
            "n_consensus_sites": 0,
            **{f"{c}_{k}": np.nan for c in GC_CLASSES for k in ("A", "n")},
        }

    y = calls[idx]
    cov = y >= 0
    n_cov = int(cov.sum())
    eq_p_all = int(((y == p) & cov).sum())
    out.update(
        {
            "n_covered": n_cov,
            "K": 100.0 * n_cov / n_sites,
            "n_eq_P": eq_p_all,
            # A_all counts alignment gaps as misses; A_cov uses covered sites only.
            "A_all": 100.0 * eq_p_all / n_sites,
        }
    )

    if n_cov == 0:
        out.update(
            {
                "A_cov": np.nan,
                "L": np.nan,
                "C": np.nan,
                "n_ne_H": 0,
                "A_null": np.nan,
                "A_null_perm_mean": np.nan,
                "A_null_perm_sd": np.nan,
                "excess": np.nan,
                "A_M": np.nan,
                "C_M": np.nan,
                "frac_M_eq_P": np.nan,
                "n_consensus_sites": 0,
                **{f"{c}_{k}": np.nan for c in GC_CLASSES for k in ("A", "n")},
            }
        )
        return out

    yc, hc, pc = y[cov], h[cov], p[cov]
    n_eq_p = int((yc == pc).sum())
    n_ne_h = int((yc != hc).sum())
    out.update(
        {
            "A_cov": 100.0 * n_eq_p / n_cov,
            "L": 100.0 * n_ne_h / n_cov,
            "C": (100.0 * n_eq_p / n_ne_h) if n_ne_h else np.nan,
            "n_ne_H": n_ne_h,
        }
    )

    # ---- the same two rates against the MAMMALIAN CONSENSUS base instead of the platypus base ----
    # The question this answers: when the model leaves the human base, is it going to platypus
    # specifically, or just to whatever the other mammals have? On private sites M != P by
    # construction, so A_M and A_cov are competing hypotheses there; on shared sites they often
    # coincide, which is exactly why the two site sets have to be looked at separately.
    mc = site["m"][cov]
    ok = mc >= 0  # no votes, or a tie for modal: no consensus
    out["n_consensus_sites"] = int(ok.sum())
    if ok.any():
        n_eq_m = int((yc[ok] == mc[ok]).sum())
        n_ne_h_ok = int((yc[ok] != hc[ok]).sum())
        out["A_M"] = 100.0 * n_eq_m / int(ok.sum())
        out["C_M"] = (100.0 * n_eq_m / n_ne_h_ok) if n_ne_h_ok else np.nan
        # how often the consensus IS the platypus base, on the sites being scored -- the number that
        # says whether A_M and A_cov are even distinguishable on this site set
        out["frac_M_eq_P"] = 100.0 * float((mc[ok] == pc[ok]).mean())
    else:
        out["A_M"] = out["C_M"] = out["frac_M_eq_P"] = np.nan

    # GC class of the H -> P substitution, on covered sites only (same denominator as A_cov)
    cls = site["gc_class"][cov]
    for ci, name in enumerate(GC_CLASSES):
        m = cls == ci
        n = int(m.sum())
        out[f"{name}_n"] = n
        out[f"{name}_A"] = (100.0 * float((yc[m] == pc[m]).sum()) / n) if n else np.nan

    strat = site["strat"][cov]
    out["A_null"] = analytic_null(yc, pc, strat)
    if n_shuffle > 1:
        perm = permutation_null(yc, pc, strat, n_shuffle, rng)
        out["A_null_perm_mean"] = float(perm.mean())
        out["A_null_perm_sd"] = float(perm.std(ddof=1))
    else:
        out["A_null_perm_mean"] = out["A_null_perm_sd"] = np.nan
    out["excess"] = out["A_cov"] - out["A_null"]
    return out


# per-gene worker
def build_sites(gene: str, human: str, target: str, cp: str, cont_offset: int) -> dict:
    """The predefined site sets for one gene, with H, P, codon position and GC class per site."""
    pairs = site_pairs_nt(human, target)
    diag = diagnostic_sites_nt(human, target)
    if set(pairs) != set(diag) or any(pairs[i][1] != diag[i] for i in diag):
        raise AssertionError(f"{gene}: site_pairs_nt disagrees with diagnostic_sites_nt")

    f = ORTHO_DIR / f"{gene}.fasta"
    orthologs = read_fasta(f, header_field=1, uppercase=False) if f.exists() else {}
    auta, voters = private_bp_subset(cp, cont_offset, diag, orthologs)
    votes, voters_v = ortholog_votes(cp, cont_offset, pairs, orthologs)

    # The private set derived from the per-species votes must be the SAME set the published scorer
    # produces. Deriving it a second way and checking is the only thing standing between "we added a
    # consensus base" and "we silently redefined the headline site set".
    priv_from_votes = {i for i in pairs if pairs[i][1] not in votes[i]}
    if priv_from_votes != set(auta) or voters_v != voters:
        raise AssertionError(
            f"{gene}: private set from ortholog votes ({len(priv_from_votes)} sites, {voters_v} "
            f"voters) disagrees with private_bp_subset ({len(auta)}, {voters})"
        )

    idx_all = np.array(sorted(pairs), dtype=np.int32)
    shared = set(pairs) - priv_from_votes
    out = {"gene": gene, "n_voting_species": voters, "sets": {}}
    for name, keys in (
        ("private", priv_from_votes),
        ("shared_not_private", shared),
        ("platy_not_human", set(pairs)),
    ):
        idx = np.array([i for i in idx_all if i in keys], dtype=np.int32)
        h = np.array([CODE[pairs[i][0]] for i in idx], dtype=np.int8)
        p = np.array([CODE[pairs[i][1]] for i in idx], dtype=np.int8)
        m = np.array([consensus_base(votes[i]) for i in idx], dtype=np.int8)
        cls = np.array(
            [GC_CLASSES.index(gc_class(pairs[i][0], pairs[i][1])) for i in idx], dtype=np.int8
        )
        cpos = (idx % 3).astype(np.int8)
        out["sets"][name] = {
            "idx": idx,
            "h": h,
            "p": p,
            "m": m,
            "gc_class": cls,
            "codon_pos": cpos,
            # 12 strata: (human base, codon position). The null preserves both.
            "strat": (h.astype(np.int16) * 3 + cpos).astype(np.int16),
        }
    # a gene with NO ortholog evidence has an empty private set for a different reason than a gene
    # whose diagnostic sites are all shared -- carried through so the tables can tell them apart
    out["has_ortholog_evidence"] = bool(voters > 0)
    return out


def _init(payload: dict) -> None:
    _G.update(payload)


def _one_gene(gene: str) -> tuple[list[dict], list[dict], list[dict]]:
    plan = _G["plan"][gene]
    cp, ch = _G["cp"][gene], _G["ch"][gene]
    nt = plan["n_tokens"]
    cont_offset = plan["off_p"] + 90
    target = cp[cont_offset:][:nt]
    human = ch[plan["off_h"] + 90 :][:nt]
    site = build_sites(gene, human, target, cp, cont_offset)
    n_shuffle, seed = _G["n_shuffle"], _G["seed"]

    site_rows = [
        {
            "gene": gene,
            "site_set": name,
            "n_sites": int(s["idx"].size),
            "n_voting_species": site["n_voting_species"],
            "has_ortholog_evidence": site["has_ortholog_evidence"],
            "n_tokens": nt,
            **{f"{c}_n": int((s["gc_class"] == i).sum()) for i, c in enumerate(GC_CLASSES)},
        }
        for name, s in site["sets"].items()
    ]

    # ---- controls: exact by construction, so any deviation is a bug, not a result -------------
    ctrl_rows = []
    if _G["controls"]:
        rng = np.random.default_rng(seed)
        shuf = list(human)
        random.Random(seed).shuffle(shuf)  # mononucleotide shuffle: composition kept,
        controls = {
            "human_window": human,  # sequence destroyed -> the chance floor
            "platypus_target": target,
            "shuffled_human": "".join(shuf),
        }
        for cname, seq in controls.items():
            calls = gen_at_target(seq, target)
            for name, s in site["sets"].items():
                m = record_metrics(calls, s, n_shuffle if cname == "shuffled_human" else 0, rng)
                ctrl_rows.append({"gene": gene, "control": cname, "site_set": name, **m})

    rows = []
    if not _G["controls_only"]:
        for rec in _G["gens"].get(gene, []):
            calls = gen_at_target(rec["seq"], target)
            # Seeded per (gene, condition, sample): the null is reproducible record by record and
            # independent of how genes happen to fall across workers. crc32, NOT hash() -- Python
            # salts string hashing per process, so hash() would reseed differently on every run.
            rng = np.random.default_rng(
                zlib.crc32(f"{seed}|{gene}|{rec['condition']}|{rec['sample']}".encode())
            )
            for name, s in site["sets"].items():
                m = record_metrics(calls, s, n_shuffle, rng)
                rows.append(
                    {
                        "gene": gene,
                        "condition": rec["condition"],
                        "sample": rec["sample"],
                        "site_set": name,
                        "n_voting_species": site["n_voting_species"],
                        **m,
                    }
                )
    return rows, site_rows, ctrl_rows


# aggregation
METRICS = [
    "A_cov",
    "A_all",
    "L",
    "C",
    "K",
    "excess",
    "A_null",
    "A_M",
    "C_M",
    "frac_M_eq_P",
    *[f"{c}_A" for c in GC_CLASSES],
]


def per_gene_table(df: pd.DataFrame) -> pd.DataFrame:
    """Average samples within each gene and retain sample counts."""
    keys = ["gene", "condition", "site_set"]
    agg = (
        df.groupby(keys, dropna=False)
        .agg(
            n_samples=("sample", "nunique"),
            n_sites=("n_sites", "mean"),
            n_voting_species=("n_voting_species", "first"),
            **{m: (m, "mean") for m in METRICS},
            **{f"{c}_n": (f"{c}_n", "mean") for c in GC_CLASSES},
        )
        .reset_index()
    )
    return agg


def levels_table(per: pd.DataFrame) -> pd.DataFrame:
    """ABSOLUTE levels of A, L, C, K per condition -- mean and SD over genes, not deltas."""
    rows = []
    for (site_set, cond), d in per.groupby(["site_set", "condition"]):
        row = {
            "site_set": site_set,
            "condition": cond,
            "n_genes": int(d.gene.nunique()),
            "n_samples_median": float(d.n_samples.median()),
            "n_sites_per_gene": float(d.n_sites.mean()),
        }
        for m in ("A_cov", "A_all", "L", "C", "K", "A_null", "excess", "A_M", "C_M", "frac_M_eq_P"):
            v = d[m].dropna()
            row[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row[f"{m}_median"] = float(v.median()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site_set", "condition"])


def describe(delta: pd.Series) -> dict:
    """Observed spread of a per-gene delta, without resampling or inference."""
    d = delta.dropna()
    if d.empty:
        return {
            "n_genes": 0,
            "mean": np.nan,
            "median": np.nan,
            "sd": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "frac_improved": np.nan,
        }
    return {
        "n_genes": int(d.size),
        "mean": float(d.mean()),
        "median": float(d.median()),
        "sd": float(d.std(ddof=1)) if d.size > 1 else np.nan,
        "q25": float(d.quantile(0.25)),
        "q75": float(d.quantile(0.75)),
        "frac_improved": float((d > 0).mean()),
    }


def summarise(per: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired per-gene deltas against the SAME gene's unsteered generation, per condition x site
    set.
    """
    rows, gc_rows = [], []
    for site_set, d in per.groupby("site_set"):
        base = d[d.condition == BASELINE].set_index("gene")
        if base.empty:
            raise SystemExit(f"no {BASELINE} baseline for site set {site_set}")
        for cond, c in d[d.condition != BASELINE].groupby("condition"):
            c = c.set_index("gene")
            common = base.index.intersection(c.index)
            row = {
                "site_set": site_set,
                "condition": cond,
                "n_genes_total": len(common),
                "n_samples_median": float(c.loc[common, "n_samples"].median()),
                "n_sites_mean": float(c.loc[common, "n_sites"].mean()),
            }
            for m in METRICS:
                delta = c.loc[common, m] - base.loc[common, m]
                st = describe(delta)
                row[f"d_{m}"] = st["mean"]
                row[f"d_{m}_median"] = st["median"]
                row[f"d_{m}_sd"] = st["sd"]
                row[f"d_{m}_q25"], row[f"d_{m}_q75"] = st["q25"], st["q75"]
                row[f"d_{m}_frac_improved"] = st["frac_improved"]
                row[f"d_{m}_n_genes"] = st["n_genes"]
                row[f"lvl_{m}"] = float(c.loc[common, m].mean())
                row[f"base_{m}"] = float(base.loc[common, m].mean())
            rows.append(row)
            for cls in GC_CLASSES:
                st = describe(c.loc[common, f"{cls}_A"] - base.loc[common, f"{cls}_A"])
                gc_rows.append(
                    {
                        "site_set": site_set,
                        "condition": cond,
                        "gc_class": cls,
                        "n_sites_per_gene": float(c.loc[common, f"{cls}_n"].mean()),
                        "level": float(c.loc[common, f"{cls}_A"].mean()),
                        "base_level": float(base.loc[common, f"{cls}_A"].mean()),
                        **{f"d_A_{k}": v for k, v in st.items()},
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(gc_rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--out", default="site_directionality")
    ap.add_argument("--n-shuffle", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--genes", nargs="*", default=None)
    ap.add_argument(
        "--controls-only",
        action="store_true",
        help="run only the four structural controls and stop",
    )
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument(
        "--tables-only",
        action="store_true",
        help="rebuild the aggregate tables from the saved per_record.csv.gz, skipping "
        "all realignment -- for adding or changing a table without a 10-minute rerun",
    )
    args = ap.parse_args()
    d = args.run / args.dir
    out = args.run / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.tables_only:
        src = out / "per_record.csv.gz"
        if not src.exists():
            raise SystemExit(f"no {src}; run without --tables-only first")
        df = pd.read_csv(src)
        per = per_gene_table(df)
        per.to_csv(out / "per_gene.csv", index=False)
        summ, gc = summarise(per)
        summ.sort_values(["site_set", "condition"]).to_csv(out / "summary.csv", index=False)
        gc.to_csv(out / "gc_class.csv", index=False)
        lv = levels_table(per)
        lv.to_csv(out / "levels_by_condition.csv", index=False)
        print(f"rebuilt tables from {len(df)} saved records -> {out}")
        return

    ch = read_fasta(args.run / "stage1" / "cds_human.fasta", uppercase=False)
    cp = read_fasta(args.run / "stage1" / "cds_platypus.fasta", uppercase=False)
    plan_df = pd.read_csv(d / "scoring_plan.csv")
    plan_df = plan_df[plan_df.usable]
    # codon position = idx % 3 is only meaningful if the continuation starts in frame
    bad = plan_df[(plan_df.off_p % 3 != 0) | (plan_df.off_h % 3 != 0)]
    if len(bad):
        raise SystemExit(
            f"{len(bad)} genes have an out-of-frame offset; codon position would be "
            f"wrong: {list(bad.gene)[:5]}"
        )
    plan = {
        r.gene: {"n_tokens": int(r.n_tokens), "off_h": int(r.off_h), "off_p": int(r.off_p)}
        for r in plan_df.itertuples()
    }

    gens: dict[str, list[dict]] = defaultdict(list)
    n_read = 0
    try:
        with gzip.open(d / "generations.jsonl.gz", "rt") as fh:
            for line in fh:
                r = json.loads(line)
                gens[r["gene"]].append(r)
                n_read += 1
    except EOFError:
        print(f"  NOTE generations.jsonl.gz truncated after {n_read} records; using what survived")

    genes = [g for g in plan if g in gens and g in ch and g in cp]
    if args.genes:
        genes = [g for g in genes if g in set(args.genes)]
    print(
        f"genes {len(genes)} | generation records {n_read} | "
        f"ortholog fastas {sum(1 for g in genes if (ORTHO_DIR / f'{g}.fasta').exists())}"
        f"/{len(genes)} | shuffles {args.n_shuffle}"
    )

    payload = {
        "plan": plan,
        "ch": ch,
        "cp": cp,
        "gens": gens,
        "n_shuffle": args.n_shuffle,
        "seed": args.seed,
        "controls_only": args.controls_only,
        "controls": not args.no_controls,
    }
    rows, site_rows, ctrl_rows = [], [], []
    with Pool(args.workers, initializer=_init, initargs=(payload,)) as pool:
        for i, (r, s, c) in enumerate(pool.imap_unordered(_one_gene, genes, chunksize=1), 1):
            rows.extend(r)
            site_rows.extend(s)
            ctrl_rows.extend(c)
            if i % 25 == 0:
                print(f"  {i}/{len(genes)} genes", flush=True)

    sites = pd.DataFrame(site_rows)
    sites.to_csv(out / "sites_per_gene.csv", index=False)
    print("\nPredefined sites per gene (mean over genes):")
    print(
        sites.groupby("site_set")[["n_sites", *[f"{c}_n" for c in GC_CLASSES]]]
        .mean()
        .round(1)
        .to_string()
    )

    # ---- controls -----------------------------------------------------------------------------
    if ctrl_rows:
        ctrl = pd.DataFrame(ctrl_rows)
        ctrl.to_csv(out / "controls_per_gene.csv", index=False)
        cs = (
            ctrl.groupby(["control", "site_set"])[
                ["A_cov", "A_all", "L", "C", "K", "A_null", "excess"]
            ]
            .mean()
            .round(3)
        )
        cs.to_csv(out / "controls.csv")
        print(
            "\nCONTROLS (mean over genes) -- human_window must read A=L=0, "
            "platypus_target must read 100:"
        )
        print(cs.to_string())
        fails = []
        for ss in SITE_SETS:
            hw = ctrl[(ctrl.control == "human_window") & (ctrl.site_set == ss)]
            pt = ctrl[(ctrl.control == "platypus_target") & (ctrl.site_set == ss)]
            if hw.A_cov.abs().max() > 1e-9 or hw.L.abs().max() > 1e-9:
                fails.append(
                    f"{ss}: human window scores A={hw.A_cov.max():.4f} L={hw.L.max():.4f}, "
                    f"must be 0"
                )
            for col in ("A_cov", "L", "C", "K"):
                if pt[col].notna().any() and abs(pt[col].min() - 100) > 1e-9:
                    fails.append(
                        f"{ss}: platypus target scores {col}={pt[col].min():.4f}, must be 100"
                    )
        if fails:
            raise SystemExit(
                "STRUCTURAL CONTROL FAILED -- no result is reportable:\n  " + "\n  ".join(fails)
            )
        print("  all structural controls pass exactly")
    if args.controls_only:
        print(f"\n-> {out}")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no generations scored")
    df.to_csv(out / "per_record.csv.gz", index=False, compression="gzip")

    # the permutation null must agree with its closed form; disagreement means the shuffle is wrong
    ok = df[["A_null", "A_null_perm_mean", "A_null_perm_sd"]].dropna()
    if len(ok):
        z = (ok.A_null_perm_mean - ok.A_null) / (
            ok.A_null_perm_sd / np.sqrt(args.n_shuffle) + 1e-12
        )
        worst = float(np.abs(z).max())
        print(
            f"\nshuffle null vs closed form: max |z| = {worst:.2f} over {len(ok)} records "
            f"(mean abs diff {float((ok.A_null_perm_mean - ok.A_null).abs().mean()):.4f} pp)"
        )
        if worst > 8:
            raise SystemExit("permutation null disagrees with its closed form -- shuffle is wrong")

    per = per_gene_table(df)
    per.to_csv(out / "per_gene.csv", index=False)
    summ, gc = summarise(per)
    summ = summ.sort_values(["site_set", "condition"])
    summ.to_csv(out / "summary.csv", index=False)
    gc.to_csv(out / "gc_class.csv", index=False)
    levels_table(per).to_csv(out / "levels_by_condition.csv", index=False)

    (out / "config.json").write_text(
        json.dumps(
            {
                "run": str(args.run),
                "arm_dir": args.dir,
                "n_shuffle": args.n_shuffle,
                "seed": args.seed,
                "site_sets": list(SITE_SETS),
                "baseline": BASELINE,
                "generation_records_read": n_read,
                "genes": len(genes),
                "null_strata": "(human base, codon position) = 12",
                "null_point_estimate": "closed form; permutations are a check only",
                "resampling": "none -- observed per-gene deltas only (user, 2026-08-22)",
                "definitions": {
                    "A_cov": "(Y==P)/covered",
                    "A_all": "(Y==P)/all sites (published defn)",
                    "L": "(Y!=H)/covered",
                    "C": "(Y==P)/(Y!=H)",
                    "K": "covered/all sites",
                    "excess": "A_cov - expected A under within-(H,codon-pos) shuffle",
                },
            },
            indent=2,
        )
        + "\n"
    )

    pd.set_option("display.width", 250)
    for ss in SITE_SETS:
        s = summ[summ.site_set == ss]
        cols = ["condition", "n_genes_total", "d_A_cov", "d_L", "d_C", "d_K", "d_excess"]
        print(f"\n=== {ss} — mean per-gene change vs unsteered (pp) ===")
        print(s[cols].to_string(index=False, float_format=lambda z: f"{z:.2f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
