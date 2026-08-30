"""Rescore stage 4 from the saved generations, on the nucleotide alignment. CPU only."""

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

from top_recon_steer_sweep import nt_aligner  # noqa: E402

_spec = importlib.util.spec_from_file_location("sites_nt", Path(__file__).with_name("sites_nt.py"))
_sn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sn)
diagnostic_sites_nt = _sn.diagnostic_sites_nt

ORTHO_DIR = ROOT / "data" / "platypus_strat_orthologs" / "cds"
PLATYPUS = "ornithorhynchus_anatinus"
BASES = frozenset("ACGT")
_G: dict = {}


def read_fasta_multi(p: Path) -> dict[str, str]:
    """header -> seq, keyed on the SECOND field (species) of `gene|species|id` headers."""
    d: dict[str, str] = {}
    k = None
    for ln in p.read_text().splitlines():
        if ln.startswith(">"):
            parts = ln[1:].split("|")
            k = parts[1] if len(parts) > 1 else parts[0]
            d[k] = ""
        elif k:
            d[k] += ln.strip()
    return d


def autapomorphic_subset(
    plat_cds: str, cont_offset: int, diag: dict[int, str], orthologs: dict[str, str]
) -> tuple[dict[int, str], int]:
    """Drop diagnostic sites whose platypus base is shared by ANY other mammal."""
    if not diag or not orthologs:
        return {}, 0
    shared: set[int] = set()
    voters = 0
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
            if key in diag and oc == pc:
                shared.add(key)
            pcur += 1
    return {i: b for i, b in diag.items() if i not in shared}, voters


def score_sites(
    gen: str, target: str, diag: dict[int, str], auta: dict[int, str]
) -> dict[str, float]:
    """Both site-set recoveries from ONE nucleotide alignment of generation vs target."""
    if not gen or not target:
        return {}
    a = nt_aligner().align(gen, target)[0]
    g_aln, t_aln = str(a[0]), str(a[1])
    tcur = 0
    dh = dt = ah = at = 0
    for gc, tc in zip(g_aln, t_aln, strict=False):
        if tc == "-":
            continue
        if tcur in diag:
            dt += 1
            if gc == diag[tcur]:
                dh += 1
            if tcur in auta:
                at += 1
                if gc == auta[tcur]:
                    ah += 1
        tcur += 1
    return {
        "pct_diagnostic_correct": (100 * dh / dt) if dt else np.nan,
        "n_diagnostic_scorable": dt,
        "pct_autapomorphy_correct": (100 * ah / at) if at else np.nan,
        "n_autapomorphy_scorable": at,
    }


def _init(payload: dict) -> None:
    _G.update(payload)


def _one_gene(gene: str) -> list[dict]:
    plan = _G["plan"][gene]
    cp, ch = _G["cp"][gene], _G["ch"][gene]
    off_p, off_h, nt = plan["off_p"], plan["off_h"], plan["n_tokens"]
    cont_offset = off_p + 90
    target = cp[cont_offset:][:nt]
    human = ch[off_h + 90 :][:nt]
    diag = diagnostic_sites_nt(human, target)
    f = ORTHO_DIR / f"{gene}.fasta"
    orthologs = read_fasta_multi(f) if f.exists() else {}
    auta, voters = autapomorphic_subset(cp, cont_offset, diag, orthologs)
    out = []
    for rec in _G["gens"].get(gene, []):
        s = score_sites(rec["seq"], target, diag, auta)
        if not s:
            continue
        out.append(
            {
                "gene": gene,
                "condition": rec["condition"],
                "sample": rec["sample"],
                "n_voting_species": voters,
                **s,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--dir", default="stage4_cds_mean_blocks27")
    ap.add_argument("--out", default="stage4_scores_nt.csv")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--genes", nargs="*", default=None)
    args = ap.parse_args()
    d = args.run / args.dir

    def rf(p: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        k = None
        for ln in p.read_text().splitlines():
            if ln.startswith(">"):
                k = ln[1:].split("|")[0]
                out[k] = ""
            elif k:
                out[k] += ln.strip()
        return out

    ch, cp = (
        rf(args.run / "stage1" / "cds_human.fasta"),
        rf(args.run / "stage1" / "cds_platypus.fasta"),
    )
    plan_df = pd.read_csv(d / "scoring_plan.csv")
    plan_df = plan_df[plan_df.usable]
    plan = {
        r.gene: {"n_tokens": int(r.n_tokens), "off_h": int(r.off_h), "off_p": int(r.off_p)}
        for r in plan_df.itertuples()
    }

    # generations.jsonl.gz may be TRUNCATED (the n=400 run was killed by an instance shutdown mid
    # write). Read to the truncation point and carry on: every record before it is intact, and the
    # join against the existing score table below reports anything that went missing.
    gens: dict[str, list[dict]] = defaultdict(list)
    n_read = 0
    try:
        with gzip.open(d / "generations.jsonl.gz", "rt") as fh:
            for line in fh:
                r = json.loads(line)
                gens[r["gene"]].append(r)
                n_read += 1
    except EOFError:
        print(
            f"  NOTE generations.jsonl.gz truncated after {n_read} records (shutdown); "
            f"using what survived"
        )

    genes = [g for g in plan if g in gens and g in ch and g in cp]
    if args.genes:
        genes = [g for g in genes if g in set(args.genes)]
    n_ortho = sum(1 for g in genes if (ORTHO_DIR / f"{g}.fasta").exists())
    print(
        f"genes {len(genes)} | generation records {n_read} | ortholog fastas {n_ortho}/{len(genes)}"
    )

    payload = {"plan": plan, "ch": ch, "cp": cp, "gens": gens}
    rows: list[dict] = []
    with Pool(args.workers, initializer=_init, initargs=(payload,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_one_gene, genes, chunksize=1), 1):
            rows.extend(res)
            if i % 20 == 0:
                print(f"  {i}/{len(genes)} genes", flush=True)
    new = pd.DataFrame(rows)

    # carry over everything not keyed to a site position
    old = pd.read_csv(d / "stage4_scores.csv")
    keep = [
        c
        for c in old.columns
        if c not in ("pct_private_correct", "n_private_in_window", "n_diag_scorable")
    ]
    merged = old[keep].merge(new, on=["gene", "condition", "sample"], how="left")
    miss = int(merged.pct_diagnostic_correct.isna().sum())
    merged.to_csv(d / args.out, index=False)
    print(f"\n[wrote] {d / args.out}  rows {len(merged)}  unrescored {miss}")
    for col in ("pct_diagnostic_correct", "pct_autapomorphy_correct"):
        m = merged.groupby("condition")[col].mean().round(2)
        print(f"\n{col} by condition:\n{m.to_string()}")


if __name__ == "__main__":
    main()
