"""Fetch 1:1 mammalian ortholog CDS for the platypus-strat panel, so platypus AUTAPOMORPHIES can be
defined for it. CPU + network only, no GPU.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))

from resolve_orthologs import SPECIES, _get, fetch_homologies  # noqa: E402

OUT_DIR = ROOT / "data" / "platypus_strat_orthologs"
CACHE = ROOT / "data" / "cache" / "strat_ortholog_cds"
PLATYPUS = "ornithorhynchus_anatinus"
BASES = set("ACGT")
STOPS = {"TAA", "TAG", "TGA"}
N_MAX = 0.01

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def read_fasta(p: Path) -> dict[str, str]:
    d: dict[str, str] = {}
    k = None
    for ln in p.read_text().splitlines():
        if ln.startswith(">"):
            k = ln[1:].split("|")[0]
            d[k] = ""
        elif k:
            d[k] += ln.strip()
    return d


def qc(cds: str) -> tuple[bool, str]:
    """Same gates as extract_loci: a CDS that is not in frame cannot be codon-aligned, and
    codon_blocks (the aligner used to call autapomorphies) assumes complete codons."""
    if not cds:
        return False, "empty"
    flags = []
    if len(cds) % 3:
        flags.append("len%3")
    if not cds.startswith("ATG"):
        flags.append("no_atg")
    if cds[-3:] not in STOPS:
        flags.append("no_stop")
    n_frac = sum(1 for c in cds if c not in BASES) / len(cds)
    if n_frac > N_MAX:
        flags.append(f"n_frac={n_frac:.3f}")
    return (not flags), ",".join(flags)


def fetch_cds(gene_id: str, species: str) -> dict:
    """Canonical-transcript CDS for one ortholog. Cached per (gene, species)."""
    f = CACHE / f"{gene_id}__{species}.json"
    if f.exists():
        return json.loads(f.read_text())
    rec: dict = {"gene_id": gene_id, "species": species, "cds": "", "tx": "", "err": ""}
    look = _get(f"/lookup/id/{gene_id}?expand=1;content-type=application/json")
    if not look:
        rec["err"] = "lookup_failed"
    else:
        txs = look.get("Transcript", []) or []
        # Ensembl marks exactly one canonical transcript; fall back to the longest CDS-bearing one
        # so a species is not silently dropped when the flag is absent in this release.
        canon = [t for t in txs if t.get("is_canonical")]
        pick = canon[0] if canon else max(txs, key=lambda t: t.get("length", 0), default=None)
        if not pick:
            rec["err"] = "no_transcript"
        else:
            rec["tx"] = pick["id"]
            seq = _get(f"/sequence/id/{pick['id']}?type=cds;content-type=application/json")
            if not seq or "seq" not in seq:
                rec["err"] = "cds_failed"
            else:
                rec["cds"] = seq["seq"].upper()
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    time.sleep(0.12)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--genes", nargs="*", default=None)
    args = ap.parse_args()

    sym_path = args.run / "stage1" / "gene_symbols.tsv"
    rows = list(csv.DictReader(sym_path.open(), delimiter="\t"))
    panel = {r["gene"]: r["symbol"] for r in rows if r.get("symbol")}
    if args.genes:
        panel = {g: s for g, s in panel.items() if g in set(args.genes)}
    log(f"panel: {len(panel)} genes with symbols (from {sym_path})")

    stage1_plat = read_fasta(args.run / "stage1" / "cds_platypus.fasta")
    (OUT_DIR / "cds").mkdir(parents=True, exist_ok=True)

    # ---- 1. Cached Compara homologies
    todo = list(panel.items())
    homs: dict[str, dict[str, str]] = {}
    done = [0]

    def resolve_one(item: tuple[str, str]) -> None:
        gene, symbol = item
        try:
            # Key homology queries by Ensembl gene ID to avoid ambiguous symbols.
            hs = fetch_homologies(symbol, gene_id=gene)
        except Exception as e:  # noqa: BLE001
            log(f"  {gene} {symbol}: homology failed {e}")
            hs = []
        keep: dict[str, str] = {}
        for h in hs:
            sp, ty = h.get("species"), h.get("type")
            if sp in SPECIES and ty == "ortholog_one2one" and sp not in keep:
                keep[sp] = h.get("id", "")
        homs[gene] = keep
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"  homology {done[0]}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(resolve_one, todo))
    n_sp = [len(v) for v in homs.values()]
    log(
        f"homologies done: median {sorted(n_sp)[len(n_sp) // 2]} species/gene, "
        f"{sum(1 for v in n_sp if v == 0)} genes with none"
    )

    # ---- 2. CDS per (gene, species) ------------------------------------------------------------
    jobs = [(g, sp, gid) for g, keep in homs.items() for sp, gid in keep.items() if gid]
    log(f"fetching CDS for {len(jobs)} (gene, species) pairs...")
    man: list[dict] = []
    got = [0]

    def cds_one(job: tuple[str, str, str]) -> None:
        gene, sp, gid = job
        rec = fetch_cds(gid, sp)
        ok, flags = qc(rec["cds"])
        man.append(
            {
                "gene": gene,
                "species": sp,
                "ortholog_gene_id": gid,
                "transcript": rec["tx"],
                "cds_len": len(rec["cds"]),
                "qc_pass": ok,
                "qc_flags": flags or rec.get("err", ""),
            }
        )
        got[0] += 1
        if got[0] % 250 == 0:
            log(f"  cds {got[0]}/{len(jobs)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(cds_one, jobs))

    # ---- 3. write per-gene fastas + manifests --------------------------------------------------
    by_gene: dict[str, list[dict]] = {}
    for m in man:
        by_gene.setdefault(m["gene"], []).append(m)
    agree = []
    for gene, ms in by_gene.items():
        lines = []
        for m in sorted(ms, key=lambda x: x["species"]):
            if not m["qc_pass"]:
                continue
            rec = json.loads((CACHE / f"{m['ortholog_gene_id']}__{m['species']}.json").read_text())
            lines.append(f">{gene}|{m['species']}|{m['ortholog_gene_id']}\n{rec['cds']}")
            if m["species"] == PLATYPUS:
                s1 = stage1_plat.get(gene, "")
                agree.append(
                    {
                        "gene": gene,
                        "stage1_len": len(s1),
                        "ensembl_len": len(rec["cds"]),
                        "identical": s1 == rec["cds"],
                    }
                )
        if lines:
            (OUT_DIR / "cds" / f"{gene}.fasta").write_text("\n".join(lines) + "\n")

    with (OUT_DIR / "ortholog_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "gene",
                "species",
                "ortholog_gene_id",
                "transcript",
                "cds_len",
                "qc_pass",
                "qc_flags",
            ],
        )
        w.writeheader()
        w.writerows(sorted(man, key=lambda m: (m["gene"], m["species"])))
    if agree:
        with (OUT_DIR / "platypus_agreement.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["gene", "stage1_len", "ensembl_len", "identical"])
            w.writeheader()
            w.writerows(sorted(agree, key=lambda a: a["gene"]))

    npass = sum(1 for m in man if m["qc_pass"])
    per = sorted(sum(1 for m in ms if m["qc_pass"]) for ms in by_gene.values())
    log(f"\nCDS: {npass}/{len(man)} passed QC")
    log(
        f"genes with a fasta: {sum(1 for ms in by_gene.values() if any(m['qc_pass'] for m in ms))}"
        f"/{len(panel)}"
    )
    if per:
        log(f"voting species per gene: median {per[len(per) // 2]}, min {per[0]}, max {per[-1]}")
    if agree:
        log(
            "platypus stage1 vs Ensembl identical: "
            f"{sum(a['identical'] for a in agree)}/{len(agree)}"
        )
    log(f"[wrote] {OUT_DIR}")


if __name__ == "__main__":
    main()
