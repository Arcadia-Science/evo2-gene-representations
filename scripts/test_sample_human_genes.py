"""Assemble + sample the matched human-paralog panel (the apples-to-apples Evo2-vs-GPN-Star set).

This is the human side of the comparison: starting from the human paralog family membership defined
in scripts/gene_families.py, it works out which genes BOTH models can actually embed, then resolves
the single shared locus each model samples. (The cross-kingdom Evo2 ortholog panel is separate —
scripts/gene_families.py build-ortholog — and needs none of this reconciliation.)

Three stages, each a CLI subcommand; the resolved panel + sampling views are also importable
(consumed by scripts/evo2/embed_and_geodesic_paralog.py and scripts/gpnstar/embed_and_geodesic.py):

  build-table   HGNC paralog membership (gene_families) + per-gene Ensembl canonical-CDS lookup +
                multiz100way chromosome coverage -> data/sampling/master_gene_table.tsv. Each gene is
                flagged include_evo2 (a human canonical CDS exists) and include_gpnstar (hg38 coords +
                chrom present in the multiz panel); the genes with BOTH are the comparable set.
  resolve-loci  For the matched genes, resolve the canonical transcript's hg38 span (UTR + exons +
                introns) -> data/sampling/shared_loci.tsv (a review artifact). The shared locus is the
                single-source definition both models sample (Evo2 = genomic string; GPN = multiz windows).
  prefetch-cds  Fill data/cache/cds_sequences.json with the matched panel's CDS for the within-family
                / k-mer baselines.

(Supersedes the old build_master_gene_table.py + sequence_sampling.py + prefetch_matched_cds.py.)

Usage:
    uv run python scripts/test_sample_human_genes.py build-table [--families ...] [--date 2026-06-29]
    uv run python scripts/test_sample_human_genes.py resolve-loci [--families globins]
    uv run python scripts/test_sample_human_genes.py prefetch-cds
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gene_families import HGNC_FAMILY_SPEC, PFAM_ACCESSIONS, family_members, family_order  # noqa: E402

ENSEMBL_BASE = "https://rest.ensembl.org"

SAMPLING_DIR = ROOT / "data" / "sampling"
MASTER_TABLE = SAMPLING_DIR / "master_gene_table.tsv"
OUT_LOCI = SAMPLING_DIR / "shared_loci.tsv"
GENE_RECORDS_CACHE = ROOT / "data" / "cache" / "gene_records.json"  # symbol -> Ensembl gene record
LOCI_CACHE = ROOT / "data" / "cache" / "gene_loci.json"            # transcript_id -> hg38 locus
CDS_CACHE = ROOT / "data" / "cache" / "cds_sequences.json"
MULTIZ_ZARR = ROOT / "data" / "multiz100way.zarr"

# Sampling defaults — tune at embed time, not here.
GPNSTAR_WINDOW = 1536  # multiz window width (≈ GPN-Star max_position_embeddings)
GPNSTAR_MAX_WINDOWS = 16  # cap windows per gene (mean-pooled)
EVO2_MAX_LEN = 100_000  # clip/center the Evo2 genomic string to this many bp

# master_gene_table.tsv schema (order matters — written verbatim).
COLUMNS = [
    "family_label", "species", "gene_id", "transcript_id", "protein_id", "symbol",
    "relationship_type", "anchor_gene_id", "source_ids", "cds_sequence_available",
    "genomic_coordinates_available", "gpnstar_msa_available", "include_evo2",
    "include_gpnstar", "inclusion_reason",
]

# Family display order for the panel (largest → smallest; intentionally differs from the canonical
# gene_families.family_order("human"), which is grouped by biology rather than size).
PANEL_FAMILY_ORDER = [
    "olfactory_receptors", "cytochrome_p450", "hox", "ras_gtpases",
    "carbonic_anhydrase", "globins", "opsins", "nitric_oxide_synthase", "heme_oxygenase",
]


def _b(x: bool) -> str:
    """Lower-case boolean for the TSV."""
    return "true" if x else "false"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — build-table: HGNC membership + per-gene availability -> master_gene_table.tsv
# ══════════════════════════════════════════════════════════════════════════════


def fetch_gene_record(symbol: str, sleep: float) -> dict:
    """Look up a human gene symbol; return {status, record}.

    status ∈ {ok, no_canonical_cds, not_found}. ``record`` (for ok) carries the canonical-CDS
    coordinates plus the gene/transcript/protein Ensembl ids. Transient network errors retry with
    backoff, then raise (a real outage should not be cached as a permanent drop).
    """
    url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json&expand=1"
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
                    "gene_id": data["id"], "transcript_id": canonical, "protein_id": tr["id"],
                    "chrom": chrom, "cds_start": tr["start"], "cds_end": tr["end"],
                    "strand": strand, "cds_len": tr["end"] - tr["start"],
                },
            }
    return {"status": "no_canonical_cds", "record": None}


def _load_records_cache() -> dict:
    return json.loads(GENE_RECORDS_CACHE.read_text()) if GENE_RECORDS_CACHE.exists() else {}


def _save_records_cache(cache: dict) -> None:
    GENE_RECORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GENE_RECORDS_CACHE.write_text(json.dumps(cache, indent=2) + "\n")


def multiz_chroms() -> set[str]:
    """Top-level chromosome groups present in the multiz100way zarr (drop zarr dotfiles)."""
    if not MULTIZ_ZARR.exists():
        print(f"  WARNING: {MULTIZ_ZARR} not found — gpnstar_msa_available will be false.")
        return set()
    return {p.name for p in MULTIZ_ZARR.iterdir() if p.is_dir() and not p.name.startswith(".")}


def resolve_paralog_rows(
    families: list[str], gene_families: dict[str, list[str]], chroms: set[str],
    cache: dict, sleep: float,
) -> tuple[list[dict], list[dict]]:
    """Resolve HGNC paralog members → master rows (with availability flags) + dropped rows."""
    rows: list[dict] = []
    dropped: list[dict] = []

    for family in families:
        members = gene_families.get(family, [])
        hgnc_ids = ";".join(f"HGNC:{g}" for g in HGNC_FAMILY_SPEC.get(family, {}).get("group_ids", []))
        pfam = PFAM_ACCESSIONS.get(family, "")
        print(f"\n=== {family} ({len(members)} HGNC members) ===")

        for sym in members:
            if sym.startswith("MT-"):  # mitochondrial: absent from the nuclear multiz
                dropped.append({"gene": sym, "family": family, "reason": "mt_locus"})
                continue

            if sym not in cache:
                cache[sym] = fetch_gene_record(sym, sleep)
            status, rec = cache[sym]["status"], cache[sym]["record"]

            if status != "ok":
                dropped.append({"gene": sym, "family": family,
                                "reason": "no_canonical_cds" if status == "no_canonical_cds" else "gene_not_found"})
                continue

            # A human canonical CDS exists ⇒ both the CDS sequence (Evo2) and the hg38 coordinates
            # (GPN window) are available; GPN additionally needs the chrom in the multiz panel.
            coords_ok = True
            msa_ok = rec["chrom"] in chroms
            src = ";".join(x for x in (hgnc_ids, f"Pfam:{pfam}" if pfam else "", rec["gene_id"]) if x)
            reason = "human paralog; canonical hg38 CDS"
            if not msa_ok:
                reason += f"; chrom {rec['chrom']} absent from multiz100way (GPN-excluded)"

            rows.append({
                "family_label": family, "species": "homo_sapiens", "gene_id": rec["gene_id"],
                "transcript_id": rec["transcript_id"], "protein_id": rec["protein_id"], "symbol": sym,
                "relationship_type": "human_paralog", "anchor_gene_id": "", "source_ids": src,
                "cds_sequence_available": _b(coords_ok), "genomic_coordinates_available": _b(coords_ok),
                "gpnstar_msa_available": _b(msa_ok), "include_evo2": _b(coords_ok),
                "include_gpnstar": _b(coords_ok and msa_ok), "inclusion_reason": reason,
            })
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

    matched = [r for r in rows if r["include_evo2"] == "true" and r["include_gpnstar"] == "true"]
    drop_reasons: dict[str, int] = {}
    for d in dropped:
        drop_reasons[d["reason"]] = drop_reasons.get(d["reason"], 0) + 1

    return {
        "generated_date": date,
        "config": {
            "families": families,
            "source": "HGNC gene groups via gene_families.py + Ensembl REST (hg38 canonical CDS)",
            "multiz_chroms_present": sorted(multiz_chroms()),
        },
        "counts": {
            "master_rows": len(rows),
            "matched_both_models": len(matched),
            "dropped": len(dropped),
        },
        "by_family": by(rows, "family_label"),
        "include_gpnstar": sum(r["include_gpnstar"] == "true" for r in rows),
        "include_evo2": sum(r["include_evo2"] == "true" for r in rows),
        "dropped_by_reason": drop_reasons,
        "notes": [
            "Human paralogs only; the cross-kingdom Evo2 panel lives in gene_families.py (build-ortholog).",
            "GPN-Star is human-anchored; include_gpnstar requires hg38 coords + chrom in multiz100way.",
            "Availability flags are metadata-derived (a human canonical CDS implies CDS+coords).",
        ],
    }


def cmd_build_table(args: argparse.Namespace) -> None:
    gene_families = family_members("human")
    chroms = multiz_chroms()
    cache = _load_records_cache()

    print(f"multiz100way chroms: {len(chroms)} present")
    rows, dropped = resolve_paralog_rows(args.families, gene_families, chroms, cache, args.sleep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "master_gene_table.tsv", rows)
    with open(args.out_dir / "dropped_genes.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gene", "family", "reason"], delimiter="\t")
        w.writeheader()
        w.writerows(dropped)

    summary = summarize(rows, dropped, args.families, args.date)
    (args.out_dir / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nWrote master_gene_table.tsv + dropped_genes.tsv + sampling_summary.json to {args.out_dir}/")
    print(f"  master rows           : {len(rows)}")
    print(f"  matched (both models) : {summary['counts']['matched_both_models']}")
    print(f"  include_evo2 / gpnstar: {summary['include_evo2']} / {summary['include_gpnstar']}")
    print(f"  dropped               : {len(dropped)}  {summary['dropped_by_reason']}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — the matched panel + shared-locus resolution (importable + resolve-loci CLI)
# ══════════════════════════════════════════════════════════════════════════════


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
    """Resolve a canonical transcript to its hg38 locus (span + exon + coding-exon intervals).

    Keying on the transcript_id (the same one chosen in master_gene_table.tsv) keeps the locus
    identical to the gene the table selected. Returns None if the transcript is not found.
    """
    data = _get_json(
        f"{ENSEMBL_BASE}/lookup/id/{transcript_id}?content-type=application/json&expand=1", sleep)
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
    """The hg38 interval both models sample. ``span='transcript'`` = the transcript body."""
    if span != "transcript":
        raise ValueError("only span='transcript' is implemented; gene/flanking are future views")
    return locus["chrom"], locus["tx_start"], locus["tx_end"], locus["strand"]


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def gpnstar_windows(
    locus: dict, window: int = GPNSTAR_WINDOW, max_windows: int = GPNSTAR_MAX_WINDOWS,
    span: str = "transcript",
) -> list[tuple[int, int]]:
    """Tile <= max_windows windows of `window` bp across the shared interval (for GPN-Star).

    A span shorter than one window yields a single centered window; longer spans are tiled and then
    evenly subsampled to max_windows. The model mean-pools the windows' embeddings.
    """
    _, start, end, _ = shared_interval(locus, span)
    length = end - start
    if length <= window:
        c = (start + end) // 2
        return [(max(0, c - window // 2), max(0, c - window // 2) + window)]
    n = math.ceil(length / window)
    centers = [start + window // 2 + i * window for i in range(n)]
    centers[-1] = min(centers[-1], end - window // 2)
    if len(centers) > max_windows:
        pick = sorted(set(round(x) for x in _linspace(0, len(centers) - 1, max_windows)))
        centers = [centers[i] for i in pick]
    return [(max(0, c - window // 2), max(0, c - window // 2) + window) for c in centers]


def evo2_genomic_sequence(
    locus: dict, max_len: int = EVO2_MAX_LEN, span: str = "transcript", sleep: float = 0.34,
) -> str:
    """Genomic nucleotide string over the shared interval, on the gene's strand (for Evo2).

    If the span exceeds ``max_len`` the interval is centered on the transcript midpoint and clipped
    to ``max_len`` bp. Sequence is GRCh38 (Ensembl /sequence/region), the same reference assembly
    underlying the multiz-100way human row GPN-Star reads.
    """
    chrom, start, end, strand = shared_interval(locus, span)
    if max_len and (end - start) > max_len:
        mid = (start + end) // 2
        start, end = mid - max_len // 2, mid + max_len // 2
    strand_int = 1 if strand == "+" else -1
    url = (f"{ENSEMBL_BASE}/sequence/region/human/{chrom}:{start}..{end}:{strand_int}"
           f"?content-type=text/plain")
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


def _read_master_rows(families: list[str] | None, require_both: bool) -> list[tuple[str, str, str]]:
    """(symbol, transcript_id, family) for human-paralog rows of the master table."""
    if not MASTER_TABLE.exists():
        raise FileNotFoundError(
            f"{MASTER_TABLE} not found — run "
            "`uv run python scripts/test_sample_human_genes.py build-table` first."
        )
    rows = []
    with open(MASTER_TABLE, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["relationship_type"] != "human_paralog":
                continue
            if families and r["family_label"] not in families:
                continue
            if require_both and not (r["include_evo2"] == "true" and r["include_gpnstar"] == "true"):
                continue
            rows.append((r["symbol"], r["transcript_id"], r["family_label"]))
    return rows


def matched_panel(families: list[str] | None = None, require_both: bool = True
                  ) -> list[tuple[str, str, str]]:
    """(symbol, transcript_id, family) for the matched human-paralog panel.

    require_both=True returns only genes with BOTH include_evo2 AND include_gpnstar — the identical
    gene set both models embed (the comparable panel), so Evo2-human and GPN-human within/between
    results are apples-to-apples. Order follows the master table.
    """
    return _read_master_rows(families, require_both)


def load_matched_panel(families: list[str] | None = None, sleep: float = 0.0):
    """(genes, family_labels, {gene: locus}, family_order) for the matched panel.

    Shared by the Evo2 and GPN-Star human embedders so both run on the identical gene set/order.
    Genes are ordered by PANEL_FAMILY_ORDER then symbol; loci come from the cached locus table
    (sleep=0 since all panel loci are already resolved). Returns only genes with a usable locus.
    """
    rows = matched_panel(families)
    fam_order = [f for f in PANEL_FAMILY_ORDER if any(r[2] == f for r in rows)]
    rows.sort(key=lambda r: (fam_order.index(r[2]), r[0]))
    genes = [r[0] for r in rows]
    fams = [r[2] for r in rows]
    tx = [r[1] for r in rows]
    loci = load_or_fetch_loci(tx, sleep=sleep)
    locus_of = {g: loci[t] for g, t in zip(genes, tx) if loci.get(t)}
    keep = [(g, f) for g, f in zip(genes, fams) if g in locus_of]
    return [g for g, _ in keep], [f for _, f in keep], locus_of, fam_order


def cmd_resolve_loci(args: argparse.Namespace) -> None:
    members = matched_panel(args.families, require_both=False)
    tx_ids = sorted({t for _, t, _ in members})
    loci = load_or_fetch_loci(tx_ids, args.sleep)

    OUT_LOCI.parent.mkdir(parents=True, exist_ok=True)
    cols = ["symbol", "family_label", "gene_id", "transcript_id", "chrom", "strand",
            "tx_start", "tx_end", "tx_span_bp", "n_exons", "n_coding_exons", "n_gpnstar_windows"]
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
            w.writerow({
                "symbol": sym, "family_label": fam, "gene_id": loc["gene_id"],
                "transcript_id": tx, "chrom": loc["chrom"], "strand": loc["strand"],
                "tx_start": loc["tx_start"], "tx_end": loc["tx_end"], "tx_span_bp": span_bp,
                "n_exons": len(loc["exon_intervals"]), "n_coding_exons": len(loc["coding_intervals"]),
                "n_gpnstar_windows": len(gpnstar_windows(loc, args.window, args.max_windows)),
            })

    spans.sort()
    med = spans[len(spans) // 2] if spans else 0
    print(f"\nWrote {OUT_LOCI}  ({len(spans)} loci; {len(unresolved)} unresolved)")
    if spans:
        print(f"  transcript span bp: median {med:,}  min {spans[0]:,}  max {spans[-1]:,}")
        print(f"  spans > EVO2_MAX_LEN ({EVO2_MAX_LEN:,}): {sum(s > EVO2_MAX_LEN for s in spans)} "
              f"(centered+clipped for Evo2)")
    if unresolved:
        print(f"  unresolved transcripts: {unresolved[:5]}{' ...' if len(unresolved) > 5 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — prefetch-cds: fill cds_sequences.json for the matched panel's baselines
# ══════════════════════════════════════════════════════════════════════════════


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
    print(f"cds_sequences.json: {len(genes) - len(missing)}/{len(genes)} cached; fetching {len(missing)}")
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


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("build-table", help="HGNC membership + availability -> master_gene_table.tsv")
    pt.add_argument("--families", nargs="+", default=family_order("human"),
                    choices=family_order("human"), help="Families to resolve (default: all 9).")
    pt.add_argument("--date", default="unspecified", help="Stamp written into sampling_summary.json.")
    pt.add_argument("--out-dir", type=Path, default=SAMPLING_DIR)
    pt.add_argument("--sleep", type=float, default=0.35, help="Ensembl REST throttle (s).")

    pl = sub.add_parser("resolve-loci", help="matched genes -> shared_loci.tsv (review artifact)")
    pl.add_argument("--families", nargs="+", default=None, help="Subset of families (smoke).")
    pl.add_argument("--sleep", type=float, default=0.34, help="Ensembl REST throttle (s).")
    pl.add_argument("--window", type=int, default=GPNSTAR_WINDOW)
    pl.add_argument("--max-windows", type=int, default=GPNSTAR_MAX_WINDOWS)

    sub.add_parser("prefetch-cds", help="fill cds_sequences.json for the matched panel")

    args = ap.parse_args()
    if args.cmd == "build-table":
        cmd_build_table(args)
    elif args.cmd == "resolve-loci":
        cmd_resolve_loci(args)
    elif args.cmd == "prefetch-cds":
        cmd_prefetch_cds(args)


if __name__ == "__main__":
    main()
