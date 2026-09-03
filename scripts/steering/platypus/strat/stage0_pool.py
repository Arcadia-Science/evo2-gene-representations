"""Stage 0 — build the block-disjoint, conservation-stratified human/platypus candidate pool."""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import check_tools  # noqa: E402
import paths  # noqa: E402

BIOMART = "https://www.ensembl.org/biomart/martservice"
REST = "https://rest.ensembl.org"
SEQS = paths.STRAT_SEQS
HUMAN_PEP = SEQS / "Homo_sapiens.GRCh38.pep.all.fa.gz"

# Homolog attributes live on their own BioMart attribute page, so they cannot be mixed with
# gene-level attributes (biotype etc) in one query. Biotype comes from the GTF at stage 1 instead.
QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1"
       uniqueRows="1" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="with_oanatinus_homolog" excluded="0"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="oanatinus_homolog_ensembl_gene"/>
    <Attribute name="oanatinus_homolog_orthology_type"/>
    <Attribute name="oanatinus_homolog_orthology_confidence"/>
    <Attribute name="oanatinus_homolog_perc_id"/>
    <Attribute name="oanatinus_homolog_perc_id_r1"/>
  </Dataset>
</Query>
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensembl_release() -> int:
    with urllib.request.urlopen(
        f"{REST}/info/data?content-type=application/json", timeout=60
    ) as fh:
        return int(json.load(fh)["releases"][0])


def biomart(cache: Path) -> pd.DataFrame:
    """One BioMart query, cached. Retries because Ensembl flaps."""
    if not cache.exists():
        data = urllib.parse.urlencode({"query": QUERY}).encode()
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(BIOMART, data=data, timeout=900) as fh:
                    txt = fh.read().decode()
                if "Query ERROR" in txt or len(txt) < 1000:
                    raise RuntimeError(f"BioMart returned an error or empty body: {txt[:400]}")
                cache.write_text(txt)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                log(f"  BioMart attempt {attempt + 1} failed: {exc}")
                time.sleep(15)
        else:
            raise RuntimeError(f"BioMart failed after 4 attempts: {last}")
    df = pd.read_csv(cache, sep="\t")
    df.columns = [
        "gene_id",
        "plat_gene_id",
        "orthology_type",
        "confidence",
        "perc_id_hp",
        "perc_id_r1",
    ]
    return df


def read_pep(path: Path) -> dict[str, tuple[str, str]]:
    """Longest protein per human gene -> (protein_id, sequence). Ensembl pep headers carry gene:."""
    import gzip

    best: dict[str, tuple[str, str]] = {}
    gene = pid = None
    buf: list[str] = []

    def flush() -> None:
        if gene and buf:
            seq = "".join(buf).replace("*", "")
            if gene not in best or len(seq) > len(best[gene][1]):
                best[gene] = (pid, seq)

    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                buf = []
                pid = line[1:].split()[0]
                gene = None
                for tok in line.split():
                    if tok.startswith("gene:"):
                        gene = tok.split(":", 1)[1].split(".")[0]
            else:
                buf.append(line.strip())
    flush()
    return best


def mmseqs_version() -> str:
    """Exact mmseqs build, recorded in provenance so the block definition is reproducible."""
    try:
        out = subprocess.run(
            ["mmseqs", "version"], check=True, capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        # This message was already the good one; it is now shared with every other tool.
        raise SystemExit(check_tools.missing_message("mmseqs")) from None
    return out.strip().splitlines()[-1].strip()


def mmseqs_blocks(
    pep: dict[str, tuple[str, str]], work: Path, min_id: float, cov: float, threads: int
) -> dict[str, int]:
    """Single-linkage-ish similarity clustering -> gene_id -> block index."""
    work.mkdir(parents=True, exist_ok=True)
    fa = work / "pool.faa"
    with fa.open("w") as fh:
        for g, (_pid, seq) in sorted(pep.items()):
            fh.write(f">{g}\n{seq}\n")
    pref = work / "clu"
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(fa),
        str(pref),
        str(work / "tmp"),
        "--min-seq-id",
        str(min_id),
        "-c",
        str(cov),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "0",
        "--threads",
        str(threads),
        "-v",
        "1",
    ]
    log(f"  mmseqs: {' '.join(cmd[:6])} ... (min_id={min_id}, cov={cov})")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    tsv = pref.with_name(pref.name + "_cluster.tsv")
    reps: dict[str, str] = {}
    for line in tsv.read_text().splitlines():
        rep, mem = line.split("\t")[:2]
        reps[mem] = rep
    order = {r: i for i, r in enumerate(sorted(set(reps.values())))}
    return {g: order[r] for g, r in reps.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-strata", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--min-seq-id", type=float, default=0.30)
    ap.add_argument("--cov", type=float, default=0.50)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rel = ensembl_release()
    log(f"Ensembl release {rel}")
    # Checked up front: mmseqs is mandatory below, and BioMart takes minutes.
    mmseqs_ver = mmseqs_version()
    log(f"mmseqs version {mmseqs_ver}")

    # ---- 1. pool ------------------------------------------------------------------------------
    df = biomart(args.out / "biomart_platypus_homologs.tsv")
    log(f"BioMart rows: {len(df)}  human genes: {df.gene_id.nunique()}")
    n_all = df.gene_id.nunique()

    keep = df[(df.orthology_type == "ortholog_one2one") & (df.confidence == 1)].copy()
    keep = keep.dropna(subset=["perc_id_hp"])
    # A human gene must map to exactly one platypus gene to be a clean 1:1 pair.
    counts = keep.gene_id.value_counts()
    keep = keep[keep.gene_id.isin(counts[counts == 1].index)]
    log(f"one2one + high-confidence + unique: {len(keep)} genes (from {n_all})")

    # ---- 2. homology blocks -------------------------------------------------------------------
    paths.require(
        HUMAN_PEP,
        "the human peptide FASTA (Ensembl release 116)",
        "download the release-116 human/platypus CDS, peptide and GTF files from "
        "https://ftp.ensembl.org/pub/release-116/ -- see REPRODUCING.md",
        "GLM_STRAT_SEQS",
    )
    pep = read_pep(HUMAN_PEP)
    log(f"human proteome: {len(pep)} genes with a protein")
    keep = keep[keep.gene_id.isin(pep)].copy()
    sub = {g: pep[g] for g in keep.gene_id}
    blocks = mmseqs_blocks(sub, args.out / "mmseqs", args.min_seq_id, args.cov, args.threads)
    keep["block"] = keep.gene_id.map(blocks)
    nb = keep.block.nunique()
    sizes = keep.groupby("block").size()
    log(
        f"blocks: {nb} over {len(keep)} genes  (singletons {int((sizes == 1).sum())}, "
        f"max size {int(sizes.max())})"
    )

    # ---- 3. block -> stratum, representative = member nearest the block median ----------------
    med = keep.groupby("block").perc_id_hp.median().rename("block_perc_id")
    keep = keep.join(med, on="block")
    keep["d_med"] = (keep.perc_id_hp - keep.block_perc_id).abs()
    rep = keep.sort_values(["block", "d_med", "gene_id"]).groupby("block", as_index=False).first()

    # Representative selection uses the block MEDIAN (above) so that no block contributes its most
    # extreme member -- that would manufacture between-stratum range. But the stratum LABEL uses the
    # representative's OWN perc_id, because that is the gene actually embedded and steered.
    # Labelling
    # by block median instead leaves 9.5% of representatives outside their nominal stratum and blurs
    # the per-stratum geometry contrasts (H1a/H1b) for no benefit.
    edges = np.quantile(rep.perc_id_hp, np.linspace(0, 1, args.n_strata + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rep["stratum"] = pd.cut(
        rep.perc_id_hp, bins=edges, labels=range(args.n_strata), include_lowest=True
    ).astype(int)

    # ---- 4. frozen order ----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    frames = []
    for _s, grp in rep.groupby("stratum"):
        g = grp.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).copy()
        g["priority"] = range(len(g))
        frames.append(g)
    order = pd.concat(frames).sort_values(["stratum", "priority"])
    cols = [
        "stratum",
        "priority",
        "block",
        "gene_id",
        "plat_gene_id",
        "perc_id_hp",
        "block_perc_id",
        "perc_id_r1",
    ]
    order[cols].to_csv(args.out / "frozen_order.csv", index=False)
    keep.to_csv(args.out / "pool_with_blocks.csv", index=False)

    prov = {
        "ensembl_release": rel,
        "seed": args.seed,
        "n_strata": args.n_strata,
        "mmseqs_version": mmseqs_ver,
        "mmseqs_min_seq_id": args.min_seq_id,
        "mmseqs_cov": args.cov,
        "n_biomart_genes": int(n_all),
        "n_one2one_hiconf": int(len(keep)),
        "n_blocks": int(nb),
        "n_singleton_blocks": int((sizes == 1).sum()),
        "max_block_size": int(sizes.max()),
        "stratum_edges": [float(x) for x in edges],
        "per_stratum_blocks": {
            int(k): int(v) for k, v in rep.stratum.value_counts().sort_index().items()
        },
        "per_stratum_perc_id": {
            int(s): [
                float(g.perc_id_hp.min()),
                float(g.perc_id_hp.median()),
                float(g.perc_id_hp.max()),
            ]
            for s, g in rep.groupby("stratum")
        },
    }
    (args.out / "stage0_config.json").write_text(json.dumps(prov, indent=2))
    log(json.dumps(prov["per_stratum_perc_id"], indent=2))
    log(f"frozen order written: {len(order)} blocks -> {args.out / 'frozen_order.csv'}")


if __name__ == "__main__":
    sys.exit(main())
