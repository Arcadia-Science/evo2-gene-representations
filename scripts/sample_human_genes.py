"""Assemble and sample the Evo2 human-paralog panel."""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gene_families import (  # noqa: E402
    HGNC_FAMILY_SPEC,
    PFAM_ACCESSIONS,
    family_members,
    family_order,
)

ENSEMBL_BASE = "https://rest.ensembl.org"

SAMPLING_DIR = ROOT / "data" / "sampling"
MASTER_TABLE = SAMPLING_DIR / "master_gene_table.tsv"
OUT_LOCI = SAMPLING_DIR / "shared_loci.tsv"
GENE_RECORDS_CACHE = ROOT / "data" / "cache" / "gene_records.json"  # symbol -> Ensembl gene record
LOCI_CACHE = ROOT / "data" / "cache" / "gene_loci.json"  # transcript_id -> hg38 locus
CDS_CACHE = ROOT / "data" / "cache" / "cds_sequences.json"

# Local (no-REST) inputs for `build-local`: Ensembl release-111 genome + GTF (chrom names
# "1".."MT").
LOCAL_GTF = ROOT / "data" / "genome" / "Homo_sapiens.GRCh38.111.gtf.gz"
LOCAL_FASTA = ROOT / "data" / "genome" / "GRCh38.primary_assembly.fa"

EVO2_MAX_LEN = 100_000  # clip/center the Evo2 genomic string to this many bp

# master_gene_table.tsv schema (order matters — written verbatim).
COLUMNS = [
    "family_label",
    "species",
    "gene_id",
    "transcript_id",
    "protein_id",
    "symbol",
    "relationship_type",
    "anchor_gene_id",
    "source_ids",
    "cds_sequence_available",
    "genomic_coordinates_available",
    "include_evo2",
    "inclusion_reason",
]

# Family display order for the panel (largest → smallest; intentionally differs from the canonical
# gene_families.family_order("human"), which is grouped by biology rather than size).
PANEL_FAMILY_ORDER = [
    "olfactory_receptors",
    "cytochrome_p450",
    "hox",
    "ras_gtpases",
    "carbonic_anhydrase",
    "globins",
    "opsins",
    "nitric_oxide_synthase",
    "heme_oxygenase",
    "glutathione_peroxidase",
    "peroxidase",
    "peroxiredoxin",
    "glutaredoxin",
    "glutathione_s_transferase",
    "aldehyde_dehydrogenase",
    "aldo_keto_reductase",
    "sulfotransferase",
    "udp_glucuronosyltransferase",
    "nadph_oxidase",
    "arachidonate_lipoxygenase",
    "flavin_monooxygenase",
    "steap_metalloreductase",
    "matrix_metalloproteinase",
    "adam_metallopeptidase",
    "adamts_metallopeptidase",
    "m14_carboxypeptidase",
    "alcohol_dehydrogenase",
    "metallothionein",
    "alkaline_phosphatase",
    "histone_deacetylase_classI",
    "ectonucleotide_pyrophosphatase",
    "phosphodiesterase",
    "ferritin",
    "rab_gtpase",
    "arf_gtpase",
    "rho_gtpase",
    "guanylate_binding_protein",
    "taste2_receptor",
    "serotonin_receptor",
    "adrenoceptor",
    "glutamate_metabotropic",
    "dopamine_receptor",
    "muscarinic_receptor",
    "histamine_receptor",
    "p2y_receptor",
    "cxc_chemokine_receptor",
    "serine_protease",
    "histone_h4",
]


def _b(x: bool) -> str:
    """Lower-case boolean for the TSV."""
    return "true" if x else "false"


# Stage 1 — build-table: HGNC membership + per-gene availability -> master_gene_table.tsv


def fetch_gene_record(symbol: str, sleep: float) -> dict:
    """Look up a human gene symbol; return {status, record}."""
    url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json&expand=1"
    )
    data = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):  # definitive: symbol not in Ensembl
                return {"status": "not_found", "record": None}
            wait = 10 * 2**attempt
            print(f"  Ensembl HTTP {e.code} for {symbol}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001 — transient (timeout / DNS / reset)
            wait = 10 * 2**attempt
            print(f"  Ensembl request failed for {symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {symbol} after 5 attempts")

    time.sleep(sleep)
    chrom = data.get("seq_region_name", "")
    strand = "+" if data.get("strand", 1) == 1 else "-"
    canonical = (data.get("canonical_transcript") or "").split(".")[0]
    for tx in data.get("Transcript", []):
        if tx.get("id") == canonical and "Translation" in tx:
            tr = tx["Translation"]
            return {
                "status": "ok",
                "record": {
                    "gene_id": data["id"],
                    "transcript_id": canonical,
                    "protein_id": tr["id"],
                    "chrom": chrom,
                    "cds_start": tr["start"],
                    "cds_end": tr["end"],
                    "strand": strand,
                    "cds_len": tr["end"] - tr["start"],
                },
            }
    return {"status": "no_canonical_cds", "record": None}


def _load_records_cache() -> dict:
    return json.loads(GENE_RECORDS_CACHE.read_text()) if GENE_RECORDS_CACHE.exists() else {}


def _save_records_cache(cache: dict) -> None:
    GENE_RECORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GENE_RECORDS_CACHE.write_text(json.dumps(cache, indent=2) + "\n")


def resolve_paralog_rows(
    families: list[str],
    gene_families: dict[str, list[str]],
    cache: dict,
    sleep: float,
) -> tuple[list[dict], list[dict]]:
    """Resolve HGNC paralog members → master rows (with availability flags) + dropped rows."""
    rows: list[dict] = []
    dropped: list[dict] = []

    for family in families:
        members = gene_families.get(family, [])
        hgnc_ids = ";".join(
            f"HGNC:{g}" for g in HGNC_FAMILY_SPEC.get(family, {}).get("group_ids", [])
        )
        pfam = PFAM_ACCESSIONS.get(family, "")
        print(f"\n=== {family} ({len(members)} HGNC members) ===")

        for sym in members:
            if sym not in cache:
                cache[sym] = fetch_gene_record(sym, sleep)
            status, rec = cache[sym]["status"], cache[sym]["record"]

            if status != "ok":
                dropped.append(
                    {
                        "gene": sym,
                        "family": family,
                        "reason": "no_canonical_cds"
                        if status == "no_canonical_cds"
                        else "gene_not_found",
                    }
                )
                continue

            coords_ok = True
            src = ";".join(
                x for x in (hgnc_ids, f"Pfam:{pfam}" if pfam else "", rec["gene_id"]) if x
            )
            reason = "human paralog; canonical hg38 CDS"

            rows.append(
                {
                    "family_label": family,
                    "species": "homo_sapiens",
                    "gene_id": rec["gene_id"],
                    "transcript_id": rec["transcript_id"],
                    "protein_id": rec["protein_id"],
                    "symbol": sym,
                    "relationship_type": "human_paralog",
                    "anchor_gene_id": "",
                    "source_ids": src,
                    "cds_sequence_available": _b(coords_ok),
                    "genomic_coordinates_available": _b(coords_ok),
                    "include_evo2": _b(coords_ok),
                    "inclusion_reason": reason,
                }
            )
        _save_records_cache(cache)  # checkpoint after each family (long OR fetch is resumable)

    return rows, dropped


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], dropped: list[dict], families: list[str], date: str) -> dict:
    def by(rs, key):
        out: dict[str, int] = {}
        for r in rs:
            out[r[key]] = out.get(r[key], 0) + 1
        return out

    included = [r for r in rows if r["include_evo2"] == "true"]
    drop_reasons: dict[str, int] = {}
    for d in dropped:
        drop_reasons[d["reason"]] = drop_reasons.get(d["reason"], 0) + 1

    return {
        "generated_date": date,
        "config": {
            "families": families,
            "source": "HGNC gene groups via gene_families.py + Ensembl REST (hg38 canonical CDS)",
        },
        "counts": {
            "master_rows": len(rows),
            "included": len(included),
            "dropped": len(dropped),
        },
        "by_family": by(rows, "family_label"),
        "include_evo2": sum(r["include_evo2"] == "true" for r in rows),
        "dropped_by_reason": drop_reasons,
        "notes": [
            "Human paralogs only; build the cross-kingdom panel with gene_families.py.",
            "Availability flags are metadata-derived (a human canonical CDS implies CDS+coords).",
        ],
    }


def cmd_build_table(args: argparse.Namespace) -> None:
    gene_families = family_members("human")
    cache = _load_records_cache()

    rows, dropped = resolve_paralog_rows(args.families, gene_families, cache, args.sleep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "master_gene_table.tsv", rows)
    with open(args.out_dir / "dropped_genes.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "family", "reason"], delimiter="\t")
        w.writeheader()
        w.writerows(dropped)

    summary = summarize(rows, dropped, args.families, args.date)
    (args.out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        "\nWrote master_gene_table.tsv, dropped_genes.tsv, and sampling_summary.json "
        f"to {args.out_dir}/"
    )
    print(f"  master rows           : {len(rows)}")
    print(f"  included              : {summary['counts']['included']}")
    print(f"  dropped               : {len(dropped)}  {summary['dropped_by_reason']}")


# Stage 2 — the matched panel + shared-locus resolution (importable + resolve-loci CLI)


def _get_json(url: str, sleep: float):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            wait = 10 * 2**attempt
            print(f"  HTTP {e.code} on {url}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001 — transient
            wait = 10 * 2**attempt
            print(f"  request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Ensembl request failed after retries: {url}")


def fetch_locus(transcript_id: str, sleep: float) -> dict | None:
    """Resolve a canonical transcript to its hg38 locus (span + exon + coding-exon intervals)."""
    data = _get_json(
        f"{ENSEMBL_BASE}/lookup/id/{transcript_id}?content-type=application/json&expand=1", sleep
    )
    if data is None:
        return None
    exons = sorted([[e["start"], e["end"]] for e in data.get("Exon", [])])
    coding: list[list[int]] = []
    if "Translation" in data:
        cs, ce = data["Translation"]["start"], data["Translation"]["end"]
        for s, e in exons:
            lo, hi = max(s, cs), min(e, ce)
            if lo <= hi:
                coding.append([lo, hi])
    return {
        "transcript_id": transcript_id,
        "gene_id": data.get("Parent", ""),
        "chrom": data["seq_region_name"],
        "strand": "+" if data.get("strand", 1) == 1 else "-",
        "tx_start": data["start"],  # transcript span (UTR + introns + exons)
        "tx_end": data["end"],
        "exon_intervals": exons,
        "coding_intervals": coding,
    }


def load_or_fetch_loci(transcripts: list[str], sleep: float) -> dict[str, dict | None]:
    cache = json.loads(LOCI_CACHE.read_text()) if LOCI_CACHE.exists() else {}
    missing = [t for t in transcripts if t not in cache]
    if missing:
        print(f"Resolving {len(missing)} transcript loci from Ensembl...")
        for i, t in enumerate(missing, 1):
            cache[t] = fetch_locus(t, sleep)
            if i % 25 == 0:
                LOCI_CACHE.parent.mkdir(parents=True, exist_ok=True)
                LOCI_CACHE.write_text(json.dumps(cache, indent=2) + "\n")
                print(f"  {i}/{len(missing)} resolved")
        LOCI_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LOCI_CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    return {t: cache[t] for t in transcripts}


def shared_interval(locus: dict, span: str = "transcript") -> tuple[str, int, int, str]:
    """Return the hg38 transcript interval."""
    if span != "transcript":
        raise ValueError("only span='transcript' is implemented; gene/flanking are future views")
    return locus["chrom"], locus["tx_start"], locus["tx_end"], locus["strand"]


def evo2_genomic_sequence(
    locus: dict,
    max_len: int = EVO2_MAX_LEN,
    span: str = "transcript",
    sleep: float = 0.34,
) -> str:
    """Genomic nucleotide string over the shared interval, on the gene's strand (for Evo2)."""
    chrom, start, end, strand = shared_interval(locus, span)
    if max_len and (end - start) > max_len:
        mid = (start + end) // 2
        start, end = mid - max_len // 2, mid + max_len // 2
    strand_int = 1 if strand == "+" else -1
    url = (
        f"{ENSEMBL_BASE}/sequence/region/human/{chrom}:{start}..{end}:{strand_int}"
        f"?content-type=text/plain"
    )
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=60) as r:
                seq = r.read().decode().strip().upper()
            time.sleep(sleep)
            return seq
        except Exception as e:  # noqa: BLE001
            wait = 10 * 2**attempt
            print(f"  region fetch failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch genomic region for {locus['transcript_id']}")


def _read_master_rows(families: list[str] | None) -> list[tuple[str, str, str]]:
    """(symbol, transcript_id, family) for human-paralog rows of the master table."""
    if not MASTER_TABLE.exists():
        raise FileNotFoundError(
            f"{MASTER_TABLE} not found — run "
            "`uv run python scripts/sample_human_genes.py build-table` first."
        )
    rows = []
    with open(MASTER_TABLE, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["relationship_type"] != "human_paralog":
                continue
            if families and r["family_label"] not in families:
                continue
            if r["include_evo2"] != "true":
                continue
            rows.append((r["symbol"], r["transcript_id"], r["family_label"]))
    return rows


def matched_panel(families: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Return included ``(symbol, transcript_id, family)`` rows."""
    return _read_master_rows(families)


def load_matched_panel(
    families: list[str] | None = None, sleep: float = 0.0, resolve_loci: bool = True
):
    """Return genes, family labels, loci, and family order for the human panel."""
    rows = matched_panel(families)
    fam_order = [f for f in PANEL_FAMILY_ORDER if any(r[2] == f for r in rows)]
    rows.sort(key=lambda r: (fam_order.index(r[2]), r[0]))
    genes = [r[0] for r in rows]
    fams = [r[2] for r in rows]
    tx = [r[1] for r in rows]
    if not resolve_loci:
        return genes, fams, {}, fam_order
    loci = load_or_fetch_loci(tx, sleep=sleep)
    locus_of = {g: loci[t] for g, t in zip(genes, tx, strict=False) if loci.get(t)}
    keep = [(g, f) for g, f in zip(genes, fams, strict=False) if g in locus_of]
    return [g for g, _ in keep], [f for _, f in keep], locus_of, fam_order


def cmd_resolve_loci(args: argparse.Namespace) -> None:
    members = matched_panel(args.families)
    tx_ids = sorted({t for _, t, _ in members})
    loci = load_or_fetch_loci(tx_ids, args.sleep)

    OUT_LOCI.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "symbol",
        "family_label",
        "gene_id",
        "transcript_id",
        "chrom",
        "strand",
        "tx_start",
        "tx_end",
        "tx_span_bp",
        "n_exons",
        "n_coding_exons",
    ]
    spans, unresolved = [], []
    with open(OUT_LOCI, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for sym, tx, fam in members:
            loc = loci.get(tx)
            if loc is None:
                unresolved.append((sym, tx))
                continue
            span_bp = loc["tx_end"] - loc["tx_start"]
            spans.append(span_bp)
            w.writerow(
                {
                    "symbol": sym,
                    "family_label": fam,
                    "gene_id": loc["gene_id"],
                    "transcript_id": tx,
                    "chrom": loc["chrom"],
                    "strand": loc["strand"],
                    "tx_start": loc["tx_start"],
                    "tx_end": loc["tx_end"],
                    "tx_span_bp": span_bp,
                    "n_exons": len(loc["exon_intervals"]),
                    "n_coding_exons": len(loc["coding_intervals"]),
                }
            )

    spans.sort()
    med = spans[len(spans) // 2] if spans else 0
    print(f"\nWrote {OUT_LOCI}  ({len(spans)} loci; {len(unresolved)} unresolved)")
    if spans:
        print(f"  transcript span bp: median {med:,}  min {spans[0]:,}  max {spans[-1]:,}")
        print(
            f"  spans > EVO2_MAX_LEN ({EVO2_MAX_LEN:,}): {sum(s > EVO2_MAX_LEN for s in spans)} "
            f"(centered+clipped for Evo2)"
        )
    if unresolved:
        print(f"  unresolved transcripts: {unresolved[:5]}{' ...' if len(unresolved) > 5 else ''}")


# Stage 3 — prefetch-cds: fill cds_sequences.json for the matched panel's baselines


def fetch_cds_sequence(gene_symbol: str) -> str:
    """Fetch the canonical CDS nucleotide sequence for a human gene from the Ensembl REST API."""
    # Step 1: look up canonical transcript ID
    lookup_url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}?content-type=application/json"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(lookup_url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 10 * 2**attempt
            print(f"  Lookup failed for {gene_symbol} ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {gene_symbol}")

    transcript_id = data["canonical_transcript"].split(".")[0]

    # Step 2: fetch CDS sequence for that transcript
    seq_url = f"{ENSEMBL_BASE}/sequence/id/{transcript_id}?type=cds&content-type=text/plain"
    for attempt in range(5):
        try:
            req = urllib.request.Request(seq_url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8").strip().upper()
        except Exception as e:
            wait = 10 * 2**attempt
            print(
                f"  Seq fetch failed for {gene_symbol}/{transcript_id} ({e}), "
                f"retrying in {wait}s..."
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch CDS sequence for {gene_symbol}")


def cmd_prefetch_cds(args: argparse.Namespace) -> None:
    """Idempotently fetch the matched panel's CDS into data/cache/cds_sequences.json."""
    cds = json.loads(CDS_CACHE.read_text()) if CDS_CACHE.exists() else {}
    genes = [g for g, _, _ in matched_panel()]
    missing = [g for g in genes if not cds.get(g)]
    print(
        f"cds_sequences.json: {len(genes) - len(missing)}/{len(genes)} cached; "
        f"fetching {len(missing)}"
    )
    for i, g in enumerate(missing, 1):
        try:
            s = fetch_cds_sequence(g)
            if s:
                cds[g] = s
        except Exception as e:  # noqa: BLE001 — skip a gene that won't resolve, keep going
            print(f"  skip {g}: {e}")
        if i % 25 == 0:
            CDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            CDS_CACHE.write_text(json.dumps(cds))
            print(f"  {i}/{len(missing)}")
    CDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CDS_CACHE.write_text(json.dumps(cds))
    have = sum(1 for g in genes if cds.get(g))
    print(f"done; {have}/{len(genes)} matched genes now have CDS ({len(cds)} total in cache)")


# Stage 1b — build-local: derive the master table + CDS from LOCAL release-111 files (no REST)


def _symbol_to_gene_id(gtf_path: Path, symbols: set[str]) -> dict[str, str]:
    """
    Map human gene SYMBOL -> release-111 gene_id via the GTF gene_name (one gzip pass, no REST).
    """
    import gzip

    sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))
    from extract_loci_bulk import _attrs  # noqa: E402

    out: dict[str, str] = {}
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            a = _attrs(line.rstrip("\n").split("\t")[8])
            nm = a.get("gene_name")
            if nm in symbols:
                out[nm] = a.get("gene_id")  # last-wins on the rare duplicated gene_name
    return out


def _derive_cds(model: dict, fa) -> str | None:
    """Assemble a canonical transcript CDS in coding orientation."""
    from Bio.Seq import Seq

    chrom = model["chrom"]
    if not model["cds"] or chrom not in fa:
        return None
    cds = "".join(fa[chrom][s - 1 : e].seq.upper() for s, e in model["cds"])
    if model["strand"] == "-":
        cds = str(Seq(cds).reverse_complement())
    return cds


def cmd_build_local(args: argparse.Namespace) -> None:
    """
    Derive master_gene_table.tsv + fill cds_sequences.json entirely from local release-111 files.
    """
    from pyfaidx import Fasta

    sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))
    from extract_loci_bulk import parse_gtf  # noqa: E402

    if not LOCAL_GTF.exists() or not LOCAL_FASTA.exists():
        sys.exit(f"local inputs missing: {LOCAL_GTF} / {LOCAL_FASTA}")

    gene_families = family_members("human")
    fam_order = family_order("human")

    # (family, symbol) pairs in human family order, then symbol.
    ordered: list[tuple[str, str]] = []
    all_symbols: set[str] = set()
    for fam in fam_order:
        for sym in sorted(gene_families.get(fam, [])):
            ordered.append((fam, sym))
            all_symbols.add(sym)
    print(f"[build-local] {len(fam_order)} families, {len(all_symbols)} unique symbols")

    # Step 2 — symbol -> gene_id (one gzip pass over the local GTF).
    sym2gid = _symbol_to_gene_id(LOCAL_GTF, all_symbols)
    print(
        f"[build-local] matched {len(sym2gid)}/{len(all_symbols)} symbols to a release-111 gene_id"
    )

    # Step 3 — parse the GTF once for the matched gene_ids -> canonical transcript models.
    needed_ids = set(sym2gid.values())
    print(f"[build-local] parsing GTF for {len(needed_ids)} gene_ids...")
    models, _ = parse_gtf(LOCAL_GTF, needed_ids)

    # Step 4 — derive the spliced CDS per gene from the FASTA.
    fa = Fasta(str(LOCAL_FASTA), rebuild=False)

    rows: list[dict] = []
    dropped: list[dict] = []
    derived: dict[str, str] = {}  # symbol -> local CDS (for cache fill + validation)
    per_family: dict[str, dict] = {}  # family -> counts

    for fam, sym in ordered:
        pf = per_family.setdefault(fam, {"members": 0, "include_evo2": 0, "dropped": 0})
        pf["members"] += 1
        gid = sym2gid.get(sym)
        if gid is None:  # symbol not present in the GTF gene_name column
            dropped.append({"gene": sym, "family": fam, "reason": "symbol_not_in_gtf"})
            pf["dropped"] += 1
            continue

        model = models.get(gid)
        cds = _derive_cds(model, fa) if model is not None else None
        pfam = PFAM_ACCESSIONS.get(fam, "")
        src = ";".join(x for x in (f"Pfam:{pfam}" if pfam else "", gid) if x)
        tx_id = model["tx_id"] if model is not None else ""
        chrom = model["chrom"] if model is not None else ""

        if cds is not None:
            derived[sym] = cds
            pf["include_evo2"] += 1
            reason = "human paralog; local canonical CDS derived from release-111 GTF+FASTA"
        else:
            if model is None:
                reason = "no transcript model in release-111 GTF"
            elif not model["cds"]:
                reason = "canonical transcript has no CDS"
            else:
                reason = f"chrom {chrom} absent from genome FASTA"
        ok = cds is not None

        rows.append(
            {
                "family_label": fam,
                "species": "homo_sapiens",
                "gene_id": gid,
                "transcript_id": tx_id,
                "protein_id": "",
                "symbol": sym,
                "relationship_type": "human_paralog",
                "anchor_gene_id": "",
                "source_ids": src,
                "cds_sequence_available": _b(ok),
                "genomic_coordinates_available": _b(ok),
                "include_evo2": _b(ok),
                "inclusion_reason": reason,
            }
        )

    # Step 7 — VALIDATION GATE: compare local derivation vs the cached REST CDS (do NOT overwrite).
    existing = json.loads(CDS_CACHE.read_text()) if CDS_CACHE.exists() else {}
    shared = [s for s in derived if s in existing]
    match = [s for s in shared if derived[s] == existing[s]]
    differ = [s for s in shared if derived[s] != existing[s]]
    print("\n" + "=" * 70)
    print("VALIDATION GATE — local CDS vs cached REST CDS (existing keys, no overwrite)")
    print(f"  compared : {len(shared)}   exact match : {len(match)}   differ : {len(differ)}")
    if shared:
        print(f"  match rate: {len(match) / len(shared):.1%}")
    if differ:
        print("  *** DIFFERENCES (up to 10) — local vs cached lengths: ***")
        for s in differ[:10]:
            print(f"    {s}: local={len(derived[s])}bp  cached={len(existing[s])}bp")
    else:
        print("  all shared genes match exactly — local derivation validated against REST.")
    print("=" * 70 + "\n")

    # Step 6 — ADD new CDS to the cache; NEVER overwrite an existing entry.
    added = [s for s in derived if s not in existing]
    for s in added:
        existing[s] = derived[s]
    CDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CDS_CACHE.write_text(json.dumps(existing))

    # Step 5 — write master table + dropped + summary.
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / "master_gene_table.tsv", rows)
    with open(out_dir / "dropped_genes.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "family", "reason"], delimiter="\t")
        w.writeheader()
        w.writerows(dropped)

    n_evo2 = sum(r["include_evo2"] == "true" for r in rows)
    summary = {
        "generated_date": args.date,
        "config": {
            "families": fam_order,
            "source": "Local Ensembl 111 GTF/FASTA; canonical or longest-CDS transcript",
            "gtf": str(LOCAL_GTF),
            "fasta": str(LOCAL_FASTA),
        },
        "counts": {
            "master_rows": len(rows),
            "include_evo2": n_evo2,
            "dropped": len(dropped),
            "symbols_matched_gene_id": len(sym2gid),
            "symbols_total": len(all_symbols),
        },
        "by_family": {f: c for f, c in per_family.items()},
        "validation_gate": {
            "compared": len(shared),
            "exact_match": len(match),
            "differ": len(differ),
            "match_rate": (len(match) / len(shared)) if shared else None,
            "differing_genes": differ[:10],
        },
        "cds_cache": {"added": len(added), "total": len(existing)},
    }
    (out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Report — per-family and dropped rollup.
    drop_reasons: dict[str, int] = {}
    for d in dropped:
        drop_reasons[d["reason"]] = drop_reasons.get(d["reason"], 0) + 1
    print(f"Wrote master_gene_table.tsv + dropped_genes.tsv + sampling_summary.json to {out_dir}/")
    print(f"  master rows            : {len(rows)}")
    print(f"  include_evo2           : {n_evo2}")
    print(f"  dropped (no gene_id)   : {len(dropped)}  {drop_reasons}")
    print(f"  cds_sequences.json     : +{len(added)} added, {len(existing)} total")


# CLI


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser(
        "build-table", help="HGNC membership + availability -> master_gene_table.tsv"
    )
    pt.add_argument(
        "--families",
        nargs="+",
        default=family_order("human"),
        choices=family_order("human"),
        help="Families to resolve (default: all 9).",
    )
    pt.add_argument(
        "--date", default="unspecified", help="Stamp written into sampling_summary.json."
    )
    pt.add_argument("--out-dir", type=Path, default=SAMPLING_DIR)
    pt.add_argument("--sleep", type=float, default=0.35, help="Ensembl REST throttle (s).")

    pl = sub.add_parser("resolve-loci", help="matched genes -> shared_loci.tsv (review artifact)")
    pl.add_argument("--families", nargs="+", default=None, help="Subset of families (smoke).")
    pl.add_argument("--sleep", type=float, default=0.34, help="Ensembl REST throttle (s).")

    sub.add_parser("prefetch-cds", help="fill cds_sequences.json for the matched panel")

    pbl = sub.add_parser(
        "build-local", help="derive master table + CDS from LOCAL release-111 GTF+FASTA (no REST)"
    )
    pbl.add_argument(
        "--date", default="unspecified", help="Stamp written into sampling_summary.json."
    )
    pbl.add_argument("--out-dir", type=Path, default=SAMPLING_DIR)

    args = ap.parse_args()
    if args.cmd == "build-table":
        cmd_build_table(args)
    elif args.cmd == "resolve-loci":
        cmd_resolve_loci(args)
    elif args.cmd == "prefetch-cds":
        cmd_prefetch_cds(args)
    elif args.cmd == "build-local":
        cmd_build_local(args)


if __name__ == "__main__":
    main()
