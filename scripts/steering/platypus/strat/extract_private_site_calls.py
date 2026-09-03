"""Emit SITE-LEVEL calls at private sites for the steering arms. CPU only, no generation.

`site_directionality.py` computes C = (Y==P)/(Y!=H) per (gene, condition, sample) and writes only
those aggregates, so a site-level question -- which substitution classes carry C, and what C would
be under a composition-matched permutation of the platypus base -- cannot be asked of its output.

This re-derives the same per-site arrays that script builds internally, from the SAME saved
generations, and writes one row per (gene, condition, sample, site). It calls
`site_directionality.build_sites` and `gen_at_target` directly rather than reimplementing them, so
the site set, the alignment and the base calls are identical by construction; the caller can then
reproduce C exactly and check that it matches the published table.
"""

from __future__ import annotations
import argparse
import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

BASES = np.array(["A", "C", "G", "T"])
_G: dict = {}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sd = _load("site_directionality")


def _init(payload: dict) -> None:
    _G.update(payload)


def _one_gene(gene: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-site call rows (private set) plus the site DEFINITIONS for both site sets.

    The definitions are emitted once per site rather than once per site x record: the
    private-vs-non-private base-composition comparison needs the non-private sites, but not the
    12,840 generated calls at each of them.
    """
    plan = _G["plan"][gene]
    cp, ch = _G["cp"][gene], _G["ch"][gene]
    nt = plan["n_tokens"]
    cont_offset = plan["off_p"] + 90
    target = cp[cont_offset:][:nt]
    human = ch[plan["off_h"] + 90 :][:nt]
    site = sd.build_sites(gene, human, target, cp, cont_offset)

    defs = pd.concat(
        [
            pd.DataFrame(
                {
                    "gene": gene,
                    "site_set": name,
                    "idx": v["idx"],
                    "h": v["h"],
                    "p": v["p"],
                    "codon_pos": v["codon_pos"],
                    "gc_class": v["gc_class"],
                }
            )
            for name, v in site["sets"].items()
            if name in ("private", "shared_not_private")
        ],
        ignore_index=True,
    )

    s = site["sets"]["private"]
    if s["idx"].size == 0:
        return pd.DataFrame(), defs

    frames = []
    for rec in _G["gens"].get(gene, []):
        if rec["condition"] not in _G["conditions"]:
            continue
        calls = sd.gen_at_target(rec["seq"], target)
        y = calls[s["idx"]]
        frames.append(
            pd.DataFrame(
                {
                    "gene": gene,
                    "condition": rec["condition"],
                    "sample": np.int16(rec["sample"]),
                    "idx": s["idx"],
                    "h": s["h"],
                    "p": s["p"],
                    "y": y,
                    "codon_pos": s["codon_pos"],
                    "gc_class": s["gc_class"],
                }
            )
        )
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), defs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, default=ROOT / "results/2026-08-08_platypus-strat-400")
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/gc_composition_null/")
    ap.add_argument(
        "--conditions",
        nargs="*",
        default=["unsteered", "add_a0.5", "add_a1.0", "add_a2.0", "add_a3.0", "add_a4.0"],
    )
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--genes", nargs="*", default=None)
    args = ap.parse_args()

    d = args.run / args.dir
    out = args.out or (args.run / "gc_composition_null")
    out.mkdir(parents=True, exist_ok=True)

    # Same inputs, same gates as site_directionality.main -- including the `usable` filter, which
    # is what keeps this gene set identical to the published one.
    ch = sd.read_fasta(args.run / "stage1" / "cds_human.fasta", uppercase=False)
    cp = sd.read_fasta(args.run / "stage1" / "cds_platypus.fasta", uppercase=False)
    plan_df = pd.read_csv(d / "scoring_plan.csv")
    plan_df = plan_df[plan_df.usable]
    bad = plan_df[(plan_df.off_p % 3 != 0) | (plan_df.off_h % 3 != 0)]
    if len(bad):
        raise SystemExit(f"{len(bad)} genes have an out-of-frame offset: {list(bad.gene)[:5]}")
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
                if r["condition"] in set(args.conditions):
                    gens[r["gene"]].append(r)
                    n_read += 1
    except EOFError:
        print(f"  NOTE generations.jsonl.gz truncated after {n_read} records; using what survived")

    genes = [g for g in plan if g in gens and g in ch and g in cp]
    if args.genes:
        genes = [g for g in genes if g in set(args.genes)]
    print(f"genes {len(genes)} | generation records kept {n_read} | conditions {args.conditions}")

    payload = {"plan": plan, "ch": ch, "cp": cp, "gens": gens, "conditions": set(args.conditions)}
    frames, defs = [], []
    with Pool(args.workers, initializer=_init, initargs=(payload,)) as pool:
        for i, (f, dfn) in enumerate(pool.imap_unordered(_one_gene, genes, chunksize=1), 1):
            if len(f):
                frames.append(f)
            if len(dfn):
                defs.append(dfn)
            if i % 25 == 0:
                print(f"  {i}/{len(genes)} genes", flush=True)

    sitedefs = pd.concat(defs, ignore_index=True)
    for c in ("h", "p"):
        sitedefs[f"{c}_base"] = BASES[sitedefs[c].to_numpy()]
    sitedefs.to_csv(out / "site_definitions.csv.gz", index=False, compression="gzip")
    print(f"wrote {len(sitedefs):,} site definitions -> {out / 'site_definitions.csv.gz'}")

    df = pd.concat(frames, ignore_index=True)
    # Bases as letters alongside the codes, so the table is readable without this script.
    for c in ("h", "p"):
        df[f"{c}_base"] = BASES[df[c].to_numpy()]
    df["y_base"] = np.where(df.y >= 0, BASES[np.clip(df.y, 0, 3).to_numpy()], "-")
    df["covered"] = df.y >= 0
    df["departure"] = df.covered & (df.y != df.h)
    df["hit"] = df.covered & (df.y == df.p)

    path = out / "private_site_calls.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    print(f"\nwrote {len(df):,} site rows -> {path}")
    print(df.groupby("condition").agg(sites=("idx", "size"), genes=("gene", "nunique")).to_string())


if __name__ == "__main__":
    main()
