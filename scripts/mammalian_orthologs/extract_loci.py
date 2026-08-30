"""Extract the transcript-span genomic locus (+ CDS) for every resolved mammalian ortholog, from each species' own Ensembl assembly, and apply per-locus QC."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "mammalian_orthologs"
LOCI_CACHE = OUT_DIR / "loci"
ENSEMBL = "https://rest.ensembl.org"
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Global request throttle shared across worker threads. Ensembl REST hard-limits at ~15 req/s and
# punishes bursts with long Retry-After 429s (seen: 88s per locus at 6 workers × 4 calls). A single
# spacing lock caps the *global* request rate regardless of worker count, keeping us just under.
_RL_LOCK = threading.Lock()
_RL_LAST = [0.0]
_RL_GAP = 0.09  # ~11 req/s


def _throttle() -> None:
    with _RL_LOCK:
        wait = _RL_GAP - (time.time() - _RL_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _RL_LAST[0] = time.time()

# QC thresholds
N_MAX = 0.02        # max ambiguous-base fraction of the locus
GAP_RUN = 50        # an N-run >= this many bp = assembly gap through the gene
LEN_LO, LEN_HI = 0.2, 5.0   # plausible length band relative to the group's median (flag only)


def _get(path: str, tries: int = 8, want_text: bool = False):
    """GET with robust retry."""
    url = f"{ENSEMBL}{path}"
    hdr = {"Content-Type": "text/plain"} if want_text else {"Content-Type": "application/json"}
    for attempt in range(tries):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=30) as r:
                raw = r.read().decode()
                return raw if want_text else json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            wait = float(e.headers.get("Retry-After", 0)) if e.code == 429 else 0
            time.sleep(max(wait, min(2 ** attempt, 20)) + random.uniform(0, 1.5))
        except Exception:
            time.sleep(min(2 ** attempt, 20) + random.uniform(0, 1.5))
    return None


def _canonical_transcript(gene_json: dict) -> dict | None:
    """Pick the canonical transcript from an expanded /lookup gene record."""
    txs = gene_json.get("Transcript", [])
    if not txs:
        return None
    canon = [t for t in txs if t.get("is_canonical")]
    return canon[0] if canon else max(txs, key=lambda t: t.get("end", 0) - t.get("start", 0))


def fetch_locus(species: str, gene_id: str) -> dict | None:
    """Resolve one ortholog gene -> canonical-transcript locus record (span + CDS + QC covariates).
    Cached per (group,species) by the caller; this does the network work."""
    g = _get(f"/lookup/id/{gene_id}?expand=1;content-type=application/json")
    if g is None:
        return {"error": "lookup_failed"}
    tx = _canonical_transcript(g)
    if tx is None:
        return {"error": "no_transcript"}
    chrom, strand = g["seq_region_name"], ("+" if g.get("strand", 1) == 1 else "-")
    tx_start, tx_end = tx["start"], tx["end"]
    biotype = tx.get("biotype", g.get("biotype", ""))
    # coding (Translation) span, for the start/stop check window
    exons = sorted([[e["start"], e["end"]] for e in tx.get("Exon", [])])
    # transcript-span genomic string on the transcript's strand
    region = f"{chrom}:{tx_start}..{tx_end}:{1 if strand == '+' else -1}"
    locus_seq = _get(f"/sequence/region/{species}/{region}?content-type=text/plain", want_text=True)
    cds_seq = _get(f"/sequence/id/{tx['id']}?type=cds;content-type=text/plain", want_text=True)
    # overlapping genes within the locus (synteny/neighbor QC): count genes other than this one
    ov = _get(f"/overlap/region/{species}/{chrom}:{tx_start}..{tx_end}?feature=gene;content-type=application/json")
    n_overlap = 0
    if isinstance(ov, list):
        n_overlap = sum(1 for f in ov if f.get("id") != gene_id and f.get("biotype") == "protein_coding")
    return {
        "species": species, "gene_id": gene_id, "transcript_id": tx["id"],
        "chrom": chrom, "tx_start": tx_start, "tx_end": tx_end, "strand": strand,
        "biotype": biotype, "n_exons": len(exons),
        "locus_seq": (locus_seq or "").upper(), "cds_seq": (cds_seq or "").upper(),
        "n_overlap_genes": n_overlap,
    }


def _resolve_human_gene_id(symbol: str) -> str | None:
    d = _get(f"/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json")
    return d.get("id") if d else None


def _n_frac_and_run(seq: str) -> tuple[float, int]:
    if not seq:
        return 1.0, 0
    n = seq.count("N")
    # longest run of N
    run = mx = 0
    for c in seq:
        if c == "N":
            run += 1; mx = max(mx, run)
        else:
            run = 0
    return n / len(seq), mx


def qc_locus(rec: dict) -> tuple[bool, list[str]]:
    """Return (pass, flags). Hard fails vs soft flags noted in comments."""
    flags = []
    err = rec.get("error")
    if isinstance(err, str) and err:   # pandas fills the 'error' column with NaN (a float, truthy!)
        return False, [err]            # for success rows — only a real string error counts
    cds, locus = rec.get("cds_seq") or "", rec.get("locus_seq") or ""
    if rec.get("biotype") != "protein_coding":
        flags.append("not_protein_coding")           # hard
    if not locus:
        flags.append("no_locus_seq")                  # hard
    if not cds:
        flags.append("no_cds")                        # hard
    else:
        if len(cds) % 3 != 0:
            flags.append("cds_not_mult3")              # hard
        if not cds.startswith("ATG"):
            flags.append("no_start_codon")             # hard
        if cds[-3:] not in STOP_CODONS:
            flags.append("no_stop_codon")              # hard
    nf, run = _n_frac_and_run(locus)
    if nf > N_MAX:
        flags.append(f"n_frac_{nf:.3f}")               # hard
    if run >= GAP_RUN:
        flags.append(f"gap_run_{run}")                 # hard
    hard = {"not_protein_coding", "no_locus_seq", "no_cds", "cds_not_mult3",
            "no_start_codon", "no_stop_codon"}
    is_hard = any(f in hard or f.startswith("n_frac_") or f.startswith("gap_run_") for f in flags)
    return (not is_hard), flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None, help="Subset of families (default: all).")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    res = pd.read_csv(OUT_DIR / "ortholog_resolution.csv")
    if args.families:
        res = res[res["family"].isin(args.families)]
    # add human query gene as a member of each group (species=homo_sapiens)
    from resolve_orthologs import SPECIES  # noqa: E402  (same dir on sys.path via -m or path insert)
    hum_rows = (res[["family", "human_gene"]].drop_duplicates()
                .assign(species="homo_sapiens", clade="primate", common_name="Human",
                        ortholog_gene_id="", type="self"))
    work = pd.concat([res, hum_rows], ignore_index=True)
    print(f"{len(work)} loci to extract ({work['human_gene'].nunique()} groups, "
          f"{work['family'].nunique()} families)", flush=True)

    LOCI_CACHE.mkdir(parents=True, exist_ok=True)
    # Resolve human gene IDs concurrently and cache them on disk.
    hid_cache = OUT_DIR / "human_gene_ids.json"
    human_ids: dict[str, str] = json.loads(hid_cache.read_text()) if hid_cache.exists() else {}
    missing_h = [g for g in sorted(hum_rows["human_gene"].unique()) if g not in human_ids]
    if missing_h:
        print(f"resolving {len(missing_h)} human gene ids (parallel, workers={args.workers})...",
              flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for g, gid in zip(missing_h, ex.map(_resolve_human_gene_id, missing_h)):
                if gid:  # only persist RESOLVED ids; None (Ensembl outage) retries next run
                    human_ids[g] = gid
        hid_cache.write_text(json.dumps(human_ids))
    print(f"human gene ids ready ({sum(1 for v in human_ids.values() if v)}/{len(human_ids)} resolved)",
          flush=True)

    def _key(row) -> Path:
        return LOCI_CACHE / f"{row.human_gene}__{row.species}.json"

    def _extract(row):
        cache = _key(row)
        if cache.exists():
            return
        gid = human_ids.get(row.human_gene) if row.species == "homo_sapiens" else row.ortholog_gene_id
        rec = {"error": "no_gene_id"} if not gid else fetch_locus(row.species, gid)
        rec.update({"family": row.family, "human_gene": row.human_gene,
                    "species": row.species, "clade": row.clade})
        # Cache ONLY final records: a success, or the permanent 'no_gene_id'. Transient failures
        # (lookup_failed/no_transcript from a 503 spell) are NOT cached, so they retry next run
        # instead of poisoning the dataset.
        if not rec.get("error") or rec.get("error") == "no_gene_id":
            cache.write_text(json.dumps(rec))

    rows = list(work.itertuples(index=False))
    todo = [r for r in rows if not _key(r).exists()]
    print(f"extracting {len(todo)} uncached loci (workers={args.workers})...")
    done = [0]
    def _run(r):
        _extract(r); done[0] += 1
        if done[0] % 100 == 0:
            print(f"  {done[0]}/{len(todo)}", flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_run, todo))
    print("extraction done; QC + manifest.")
    build_manifest(work)


def build_manifest(work: pd.DataFrame) -> None:
    recs = []
    for f in LOCI_CACHE.glob("*.json"):
        recs.append(json.loads(f.read_text()))
    df = pd.DataFrame(recs)
    # Guard columns that only exist on successful records (error rows lack seqs/coords).
    for c in ["locus_seq", "cds_seq", "biotype", "gene_id", "transcript_id", "chrom",
              "tx_start", "tx_end", "strand", "n_overlap_genes", "error"]:
        if c not in df.columns:
            df[c] = None
    df["locus_seq"] = df["locus_seq"].fillna("")
    df["cds_seq"] = df["cds_seq"].fillna("")
    # group-median locus length for the plausible-length flag
    med = df.assign(_ll=df["locus_seq"].fillna("").str.len()).groupby("human_gene")["_ll"].median()
    man_rows = []
    for _, r in df.iterrows():
        ok, flags = qc_locus(r)
        ll = len(r.get("locus_seq") or "")
        m = med.get(r["human_gene"], 0)
        if m and not (LEN_LO * m <= ll <= LEN_HI * m):
            flags = flags + ["length_outlier"]  # soft flag only
        nf, run = _n_frac_and_run(r.get("locus_seq") or "")
        man_rows.append({
            "family": r.get("family"), "group": r["human_gene"],
            "species": r["species"], "clade": r.get("clade"),
            "gene_id": r.get("gene_id"), "transcript_id": r.get("transcript_id"),
            "chrom": r.get("chrom"), "tx_start": r.get("tx_start"), "tx_end": r.get("tx_end"),
            "strand": r.get("strand"), "biotype": r.get("biotype"),
            "locus_len": ll, "cds_len": len(r.get("cds_seq") or ""),
            "n_frac": round(nf, 4), "max_n_run": run, "n_overlap_genes": r.get("n_overlap_genes"),
            "qc_pass": ok, "qc_flags": ";".join(str(x) for x in flags),
        })
    man = pd.DataFrame(man_rows)
    man.to_csv(OUT_DIR / "loci_manifest.csv", index=False)
    # write per-family FASTAs of QC-passing loci
    for kind, col in [("transcript", "locus_seq"), ("cds", "cds_seq")]:
        d = OUT_DIR / "seqs" / kind
        d.mkdir(parents=True, exist_ok=True)
        by_fam = {}
        for _, r in df.iterrows():
            ok, _ = qc_locus(r)
            if not ok:
                continue
            fam = r.get("family")
            by_fam.setdefault(fam, []).append(
                (f"{r['human_gene']}|{r['species']}|{r.get('gene_id')}", r.get(col) or ""))
        for fam, entries in by_fam.items():
            with open(d / f"{fam}.fasta", "w") as fh:
                for hid, seq in entries:
                    fh.write(f">{hid}\n{seq}\n")
    npass = int(man["qc_pass"].sum())
    print(f"\nWrote {OUT_DIR/'loci_manifest.csv'}: {len(man)} loci, {npass} QC-pass "
          f"({100*npass/len(man):.0f}%)")
    # post-QC sizes per family
    p = man[man["qc_pass"]]
    print("\n=== POST-QC loci per family ===")
    print(p.groupby("family").size().to_string())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
