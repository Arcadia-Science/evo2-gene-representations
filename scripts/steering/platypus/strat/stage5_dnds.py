"""Stage 5d — dN, dS and omega from PAML codeml on stage 5b's trees. CPU, parallel."""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))
sys.path.insert(0, str(ROOT / "scripts"))

import check_tools  # noqa: E402
from alignment_metrics import read_fasta  # noqa: E402

CODEML = ROOT / "data" / "tools" / "codeml"
YN00 = ROOT / "data" / "tools" / "yn00"
TRIMAL = ROOT / "data" / "tools" / "trimal"
HUMAN, PLATYPUS = "homo_sapiens", "ornithorhynchus_anatinus"

# dS above this on a single branch is treated as saturated / uninformative (design Stage 5).
DS_SATURATION = 1.5

CTL = """seqfile = codon.phy
treefile = {tree}
outfile = {out}
noisy = 0
verbose = 1
runmode = 0
seqtype = 1
CodonFreq = 2
clock = 0
model = {model}
NSsites = 0
icode = 0
fix_kappa = 0
kappa = 2
fix_omega = 0
omega = 0.4
cleandata = 0
method = {method}
"""

# `method = 1` (one branch at a time) everywhere. Validated against codeml's default optimiser on
# ENSG00000004534 (1108 codons, 24 taxa): lnL -20068.434635 vs -20068.434638 and omega 0.17576 vs
# 0.17576 -- identical to 3e-6 in lnL and to 5 decimals in omega -- for 12 s instead of >7 min.
# method=0 is unaffordable on the tail of this panel (longest alignment is 4855 codons).
METHOD = {"m0": 1, "m2": 1, "fr": 1}

# m2 nests m0, so lnL_m2 >= lnL_m0 at a true optimum. But on genes where freeing the platypus omega
# buys no likelihood at all -- one terminal branch often carries no information -- the two lnLs are
# EQUAL and their difference is pure floating-point noise whose sign is arbitrary. With lnL ~ -2e4,
# double precision gives ~1e-11 relative, so noise of ~1e-5 absolute is expected. Only a
# meaningfully negative LR indicates a real optimiser failure.
LRT_NEG_TOL = 1e-3

YN00_CTL = """seqfile = codon.phy
outfile = yn00.out
verbose = 1
icode = 0
weighting = 0
commonf3x4 = 0
ndata = 1
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- codon alignment
def build_codon_alignment(gene: str, run: Path, wd: Path) -> tuple[list[str], int]:
    """
    Back-translate stage 5b's protein alignment and write codeml's PHYLIP, human/platypus first.
    """
    from Bio.Seq import Seq

    tdir = run / "stage5" / "trees" / gene
    cds = run / "stage5" / "seqs" / f"{gene}.fasta"
    codon_fa = wd / "codon.fasta"

    cmd = [
        str(TRIMAL),
        "-in",
        str(tdir / "aln_mafft.fasta"),
        "-backtrans",
        str(cds),
        "-gappyout",
        "-ignorestopcodon",
        "-fasta",
        "-out",
        str(codon_fa),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not codon_fa.exists():
        raise RuntimeError(f"trimal -backtrans failed: {r.stderr.strip()[:300]}")

    codon = read_fasta(codon_fa, header_field=None, uppercase=False)
    prot = read_fasta(
        tdir / "aln.fasta", header_field=None, uppercase=False
    )  # trimAl-trimmed protein alignment (stage 5b)
    lens = {len(v) for v in codon.values()}
    if len(lens) != 1:
        raise RuntimeError(f"ragged codon alignment: {sorted(lens)}")
    n_nt = lens.pop()
    n_aa = len(next(iter(prot.values())))
    if n_nt != 3 * n_aa:
        raise RuntimeError(f"codon len {n_nt} != 3 x protein len {n_aa}")
    if set(codon) != set(prot):
        raise RuntimeError("taxon sets differ between codon and protein alignments")

    # Frame check on the two taxa the whole experiment turns on.
    for taxon in (HUMAN, PLATYPUS):
        s = codon[taxon]
        back = "".join(
            "-" if s[i : i + 3] == "---" else str(Seq(s[i : i + 3]).translate())
            for i in range(0, n_nt, 3)
        )
        if back != prot[taxon]:
            raise RuntimeError(f"back-translation != protein alignment for {taxon}")

    order = [HUMAN, PLATYPUS] + sorted(k for k in codon if k not in (HUMAN, PLATYPUS))
    with open(wd / "codon.phy", "w") as f:
        f.write(f" {len(order)} {n_nt}\n")
        for k in order:
            f.write(f"{k}\n{codon[k]}\n")

    # Plain topology for m0/fr; platypus terminal labelled #1 for the two-ratio model.
    top = (tdir / "topology.nwk").read_text().strip()
    if PLATYPUS not in top:
        raise RuntimeError("platypus absent from topology")
    (wd / "tree_plain.nwk").write_text(f" {len(order)} 1\n{top}\n")
    labelled = re.sub(rf"\b{PLATYPUS}\b", f"{PLATYPUS} #1", top)
    (wd / "tree_lab.nwk").write_text(f" {len(order)} 1\n{labelled}\n")
    return order, n_nt // 3


# --------------------------------------------------------------------------- codeml
def run_codeml(wd: Path, model: int, tree: str, tag: str, timeout: int) -> Path:
    """Run one codeml model in its own subdirectory so parallel models never share scratch files."""
    sub = wd / tag
    sub.mkdir(exist_ok=True)
    for f in ("codon.phy", "tree_plain.nwk", "tree_lab.nwk"):
        shutil.copy(wd / f, sub / f)
    (sub / "codeml.ctl").write_text(
        CTL.format(tree=tree, out=f"{tag}.mlc", model=model, method=METHOD[tag])
    )
    r = subprocess.run(
        [str(CODEML), "codeml.ctl"],
        cwd=sub,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    mlc = sub / f"{tag}.mlc"
    if r.returncode != 0:
        raise RuntimeError(
            f"codeml {tag} exited {r.returncode}: {(r.stdout + r.stderr).strip()[-200:]}"
        )
    if not mlc.exists():
        raise RuntimeError(f"codeml {tag} produced no mlc")
    if "Time used:" not in mlc.read_text()[-2000:]:
        raise RuntimeError(f"codeml {tag} mlc is truncated (no completion line)")
    return mlc


def run_yn00(wd: Path, timeout: int) -> dict:
    """Yang & Nielsen (2000) counting-method pairwise dN and dS."""
    sub = wd / "yn00"
    sub.mkdir(exist_ok=True)
    shutil.copy(wd / "codon.phy", sub / "codon.phy")
    (sub / "yn00.ctl").write_text(YN00_CTL)
    r = subprocess.run(
        [str(YN00)],
        cwd=sub,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    out = sub / "yn00.out"
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"yn00 exited {r.returncode}: {(r.stdout + r.stderr).strip()[-200:]}")

    pairs = []
    for line in out.read_text().splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
            r"(-?[\d.]+)\s+(-?[\d.]+)\s*\+-\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\+-\s*"
            r"(-?[\d.]+)\s*$",
            line,
        )
        if m:
            g = m.groups()
            pairs.append(
                {
                    "a": int(g[0]),
                    "b": int(g[1]),
                    "omega": float(g[6]),
                    "dN": float(g[7]),
                    "dS": float(g[9]),
                }
            )
    if not pairs:
        raise RuntimeError("yn00 produced no parsable pairwise rows")

    rec: dict = {"yn_n_pairs": len(pairs)}
    focal = [p for p in pairs if {p["a"], p["b"]} == {1, 2}]
    if focal:
        rec.update(
            {
                "dN_hp_yn": focal[0]["dN"],
                "dS_hp_yn": focal[0]["dS"],
                "omega_hp_yn": focal[0]["omega"],
            }
        )
    bg = [p for p in pairs if p["a"] not in (1, 2) and p["b"] not in (1, 2)]
    if bg:
        import statistics as st

        rec["dN_background_yn"] = st.mean(p["dN"] for p in bg)
        rec["dS_background_yn"] = st.mean(p["dS"] for p in bg)
        rec["omega_background_yn"] = (
            st.median(p["omega"] for p in bg if 0 <= p["omega"] < 10) if bg else float("nan")
        )
        rec["yn_n_background_pairs"] = len(bg)
    return rec


def parse_mlc(mlc: Path) -> dict:
    """Pull lnL, tree lengths, omega(s) and the per-branch dN/dS table out of a codeml mlc file."""
    text = mlc.read_text()
    out: dict = {}

    m = re.search(r"lnL\(ntime:\s*(\d+)\s+np:\s*(\d+)\):\s*(-?[\d.]+)", text)
    if m:
        out["ntime"], out["np"], out["lnL"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    for key, pat in (
        ("tree_len", r"^tree length =\s*([\d.]+)"),
        ("tree_dN", r"^tree length for dN:\s*([\d.]+)"),
        ("tree_dS", r"^tree length for dS:\s*([\d.]+)"),
        ("kappa", r"^kappa \(ts/tv\) =\s*([\d.]+)"),
    ):
        m = re.search(pat, text, re.M)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"^omega \(dN/dS\) =\s*([\d.]+)", text, re.M)
    if m:
        out["omega"] = float(m.group(1))
    # Two-ratio model: "w (dN/dS) for branches:  <background> <foreground>"
    m = re.search(r"^w \(dN/dS\) for branches:\s*([\d.\s]+)$", text, re.M)
    if m:
        out["omega_branches"] = [float(x) for x in m.group(1).split()]

    # Per-branch table. Columns: branch  t  N  S  dN/dS  dN  dS  N*dN  S*dS
    rows = []
    for line in text.splitlines():
        m = re.match(
            r"^\s*(\d+)\.\.(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
            r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
            line,
        )
        if m:
            g = m.groups()
            rows.append(
                {
                    "parent": int(g[0]),
                    "child": int(g[1]),
                    "t": float(g[2]),
                    "N": float(g[3]),
                    "S": float(g[4]),
                    "omega": float(g[5]),
                    "dN": float(g[6]),
                    "dS": float(g[7]),
                }
            )
    out["branches"] = rows
    return out


def summarise_branches(rows: list[dict], prefix: str) -> dict:
    """Split the branch table into human terminal (tip 1), platypus terminal (tip 2), background."""
    out: dict = {}
    if not rows:
        return out
    focal = {1: "human", 2: "platypus"}
    bg = [r for r in rows if r["child"] not in focal]
    for tip, name in focal.items():
        hit = [r for r in rows if r["child"] == tip]
        if hit:
            r = hit[0]
            out[f"dN_{name}_{prefix}"] = r["dN"]
            out[f"dS_{name}_{prefix}"] = r["dS"]
            out[f"omega_{name}_{prefix}"] = r["omega"]
    out[f"dN_background_{prefix}"] = sum(r["dN"] for r in bg)
    out[f"dS_background_{prefix}"] = sum(r["dS"] for r in bg)
    # Length-weighted background omega: sum(N*dN)/sum(N) over sum(S*dS)/sum(S) is fragile when dS~0,
    # so use the ratio of summed synonymous/non-synonymous divergence, which is what a 2-ratio model
    # estimates anyway.
    sdN, sdS = out[f"dN_background_{prefix}"], out[f"dS_background_{prefix}"]
    out[f"omega_background_{prefix}"] = sdN / sdS if sdS > 1e-9 else float("nan")
    out[f"n_branches_{prefix}"] = len(rows)
    return out


def do_gene(
    gene: str,
    run: Path,
    models: list[str],
    timeout: int,
    max_codons_fr: int,
    min_codons_per_branch: float,
) -> dict:
    wd = run / "stage5" / "dnds" / gene
    wd.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "gene": gene,
        "status": "ok",
        "fr_status": "not_requested",
        "yn_status": "not_requested",
    }
    branch_rows: list[dict] = []
    try:
        order, n_codons = build_codon_alignment(gene, run, wd)
        rec["n_taxa"] = len(order)
        rec["n_codons"] = n_codons

        if "yn" in models:
            try:
                rec.update(run_yn00(wd, timeout=600))
                rec["yn_status"] = "ok"
            except Exception as e:  # noqa: BLE001
                rec["yn_status"] = f"failed:{type(e).__name__}:{e}"[:120]

        if "m0" in models:
            p = parse_mlc(run_codeml(wd, 0, "tree_plain.nwk", "m0", timeout))
            rec.update(
                {
                    "lnL_m0": p.get("lnL"),
                    "np_m0": p.get("np"),
                    "omega_m0": p.get("omega"),
                    "kappa_m0": p.get("kappa"),
                    "tree_len_m0": p.get("tree_len"),
                    "tree_dN_m0": p.get("tree_dN"),
                    "tree_dS_m0": p.get("tree_dS"),
                }
            )
            rec.update(summarise_branches(p["branches"], "m0"))
            branch_rows += [{**r, "gene": gene, "model": "m0"} for r in p["branches"]]

        if "m2" in models:
            p = parse_mlc(run_codeml(wd, 2, "tree_lab.nwk", "m2", timeout))
            rec.update(
                {
                    "lnL_m2": p.get("lnL"),
                    "np_m2": p.get("np"),
                    "tree_dN_m2": p.get("tree_dN"),
                    "tree_dS_m2": p.get("tree_dS"),
                }
            )
            wb = p.get("omega_branches") or []
            if len(wb) >= 2:
                rec["omega_bg_m2"], rec["omega_plat_m2"] = wb[0], wb[1]
            rec.update(summarise_branches(p["branches"], "m2"))
            branch_rows += [{**r, "gene": gene, "model": "m2"} for r in p["branches"]]

        if "fr" in models:
            # Free-ratio fits its own omega on every branch: 2n-3 branches on n_codons sites. Slow
            # slow at BOTH ends -- long alignments cost per-iteration time, and SHORT ones are
            # nearly unidentifiable so the optimiser thrashes (measured: 8 min on 1108 codons/24
            # taxa, still running after 4 min on 117 codons/23 taxa). Gate on codons per branch
            # parameter, not just on length, and record every skip.
            n_branches = 2 * len(order) - 3
            per_branch = n_codons / max(n_branches, 1)
            if n_codons > max_codons_fr:
                rec["fr_status"] = f"skipped_too_long({n_codons}>{max_codons_fr})"
            elif per_branch < min_codons_per_branch:
                rec["fr_status"] = (
                    f"skipped_unidentifiable({per_branch:.1f} codons/branch < "
                    f"{min_codons_per_branch})"
                )
            else:
                try:
                    p = parse_mlc(run_codeml(wd, 1, "tree_plain.nwk", "fr", timeout))
                    rec.update(
                        {
                            "lnL_fr": p.get("lnL"),
                            "np_fr": p.get("np"),
                            "tree_dN_fr": p.get("tree_dN"),
                            "tree_dS_fr": p.get("tree_dS"),
                        }
                    )
                    rec.update(summarise_branches(p["branches"], "fr"))
                    branch_rows += [{**r, "gene": gene, "model": "fr"} for r in p["branches"]]
                    rec["fr_status"] = "ok"
                except subprocess.TimeoutExpired:
                    rec["fr_status"] = f"timeout({timeout}s)"
                except Exception as e:  # noqa: BLE001
                    rec["fr_status"] = f"failed:{type(e).__name__}:{e}"[:120]

        # LRT of platypus-specific omega (m2) against one-ratio (m0), df = 1. See LRT_NEG_TOL: a LR
        # of ~-1e-5 is float noise on a gene where the platypus branch carries no information, not a
        # failure, and it correctly yields p = 1. Only a materially negative LR is a real problem.
        if rec.get("lnL_m0") is not None and rec.get("lnL_m2") is not None:
            from scipy import stats

            lr = 2.0 * (rec["lnL_m2"] - rec["lnL_m0"])
            rec["lrt_m2_vs_m0"] = lr
            rec["lrt_p_m2_vs_m0"] = float(stats.chi2.sf(max(lr, 0.0), 1))
            rec["m2_converged"] = bool(lr >= -LRT_NEG_TOL)

        # Saturation flag -- platypus-terminal dS near saturation makes that column uninformative.
        ds_plat = rec.get("dS_platypus_fr", rec.get("dS_platypus_m0"))
        rec["dS_saturated"] = bool(ds_plat is not None and ds_plat > DS_SATURATION)

    except subprocess.TimeoutExpired:
        rec["status"] = f"timeout({timeout}s)"
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"failed:{type(e).__name__}:{e}"[:200]
    return {"rec": rec, "branches": branch_rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument(
        "--models", nargs="+", default=["yn", "m0", "m2", "fr"], choices=["yn", "m0", "m2", "fr"]
    )
    ap.add_argument("--timeout", type=int, default=3600, help="per codeml model, seconds")
    ap.add_argument(
        "--max-codons-fr",
        type=int,
        default=3000,
        help="skip free-ratio above this alignment length; skips are RECORDED "
        "in fr_status and counted in the summary, never silently dropped",
    )
    ap.add_argument(
        "--min-codons-per-branch-fr",
        type=float,
        default=20.0,
        help="skip free-ratio when the alignment gives fewer than this many codons per "
        "branch parameter -- it is unidentifiable and the optimiser thrashes",
    )
    ap.add_argument("--genes", nargs="+", help="subset, for validation")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="keep genes already present in dnds.csv with status ok",
    )
    args = ap.parse_args()

    # One wording for every external tool, and one place that knows how to install each. The
    # message this replaced pointed at data/tools/paml_src/paml-4.10.10/src -- a build tree that
    # `data/` being git-ignored means a clone never has.
    for tool in ("codeml", "yn00", "trimal"):
        check_tools.require(tool)

    stage5 = args.run / "stage5"
    ts = pd.read_csv(stage5 / "tree_stats.csv")
    genes = args.genes or ts.loc[ts.status == "ok", "gene"].tolist()

    out_csv = stage5 / "dnds.csv"
    prev = pd.DataFrame()
    if args.resume and out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev.loc[prev.status == "ok", "gene"])
        genes = [g for g in genes if g not in done]
        log(f"resume: {len(done)} genes already ok, {len(genes)} to run")

    if not genes:
        log("nothing to do")
        return

    # Longest alignments first: they set the wall clock, so start them before the pool drains.
    order = ts.set_index("gene").aln_len_trimmed.to_dict()
    genes = sorted(genes, key=lambda g: -order.get(g, 0))
    log(
        f"{len(genes)} genes, models={args.models}, jobs={args.jobs}, "
        f"longest first ({order.get(genes[0], 0):.0f} aa)"
    )

    recs, branches, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {
            ex.submit(
                do_gene,
                g,
                args.run,
                args.models,
                args.timeout,
                args.max_codons_fr,
                args.min_codons_per_branch_fr,
            ): g
            for g in genes
        }
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            recs.append(res["rec"])
            branches += res["branches"]
            r = res["rec"]
            log(
                f"[{n}/{len(genes)}] {r['gene']} {r['status']} yn={r['yn_status']} "
                f"fr={r['fr_status']} w_m0={r.get('omega_m0')} "
                f"dN/dS_yn={r.get('dN_background_yn')}/{r.get('dS_background_yn')}"
            )

    df = pd.DataFrame(recs)
    if not prev.empty:
        df = pd.concat([prev[~prev.gene.isin(df.gene)], df], ignore_index=True)
    df = df.sort_values("gene")
    df.to_csv(out_csv, index=False)

    if branches:
        bdf = pd.DataFrame(branches)
        bpath = stage5 / "dnds_branches.csv"
        if args.resume and bpath.exists():
            old = pd.read_csv(bpath)
            bdf = pd.concat([old[~old.gene.isin(bdf.gene)], bdf], ignore_index=True)
        bdf.to_csv(bpath, index=False)

    ok = df[df.status == "ok"]
    log(f"done in {(time.time() - t0) / 60:.1f} min -- {len(ok)}/{len(df)} ok")
    if len(ok):
        log(f"yn00: {dict(ok.yn_status.value_counts())}")
        log(f"free-ratio: {dict(ok.fr_status.value_counts())}")
        if "m2_converged" in ok.columns:
            bad_conv = int((~ok.m2_converged.fillna(False)).sum())
            log(
                f"m2 non-convergent (lnL_m2 < lnL_m0, omega estimates untrustworthy): "
                f"{bad_conv}/{len(ok)}"
            )
        for c in ("omega_m0", "dN_background_yn", "dS_background_yn", "omega_background_yn"):
            if c in ok.columns and ok[c].notna().any():
                log(f"  {c:22s} median {ok[c].median():.4f}  n={int(ok[c].notna().sum())}")
        if "dS_saturated" in ok.columns:
            log(
                f"  platypus dS > {DS_SATURATION} in {int(ok.dS_saturated.sum())}/{len(ok)}"
                " genes -- use BACKGROUND dS as the negative control"
            )
    bad = df[df.status != "ok"]
    if len(bad):
        log(f"FAILED {len(bad)}: {dict(bad.status.value_counts())}")

    (stage5 / "stage5d_config.json").write_text(
        json.dumps(
            {
                "models": args.models,
                "method": "1 (one branch at a time)",
                "CodonFreq": "2 (F3x4)",
                "cleandata": 0,
                "icode": 0,
                "codon_alignment": "trimal -backtrans of stage5b aln_mafft.fasta with -gappyout, "
                "i.e. the exact columns tree_stats.csv was estimated from",
                "topology": "stage5b topology.nwk (per-gene pruned MamPhy; no lengths)",
                "tip_order": "human=1, platypus=2 (forced), background = all non-focal branches",
                "dS_saturation_threshold": DS_SATURATION,
                "max_codons_fr": args.max_codons_fr,
                "timeout_s": args.timeout,
                "n_ok": int(len(ok)),
                "n_total": int(len(df)),
            },
            indent=2,
        )
    )
    log(f"-> {out_csv}")


if __name__ == "__main__":
    main()
