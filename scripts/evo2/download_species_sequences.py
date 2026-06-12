"""Download 10 short genomic windows (2000 bp each) for 500 bacterial species via NCBI Entrez."""

import csv
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Tuple

from Bio import Entrez
from tqdm import tqdm

# NCBI requires a contact email; set NCBI_EMAIL in your environment.
Entrez.email = os.environ.get("NCBI_EMAIL", "")
_api_key = os.environ.get("NCBI_API_KEY", "")
if _api_key:
    Entrez.api_key = _api_key

SLEEP = 0.34
WINDOWS_PER_SPECIES = 10
FETCH_BP = 4000
KEEP_BP = 2000


# ---------------------------------------------------------------------------
# NCBI helpers
# ---------------------------------------------------------------------------

def assembly_to_contig(ncbi_accession: str) -> Tuple[str, int]:
    """
    Given an NCBI assembly accession (GCA_XXXXXXX.X or GCF_XXXXXXX.X),
    return (primary_contig_accession, contig_length).

    Strategy:
      1. Call the NCBI Datasets v2 REST API sequence_reports endpoint.
         Results come back sorted by length descending, so reports[0] is
         always the longest sequence.
      2. Prefer role='assembled-molecule' (full chromosomes) over
         'unplaced-scaffold' (WGS contigs). If the assembly has no
         assembled-molecule entries (pure WGS draft), fall back to the
         longest scaffold.
      3. Within a chosen record, prefer the RefSeq accession (NC_/NZ_)
         over the GenBank accession when present — RefSeq IDs are more
         stable for efetch.
    """
    url = (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        f"{ncbi_accession}/sequence_reports?page_size=20"
    )
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    reports = data.get("reports", [])
    if not reports:
        raise ValueError(f"No sequence reports found for {ncbi_accession}")

    # Prefer chromosome-level assembled molecules; fall back to longest scaffold.
    # The API already returns all records sorted by length descending, so
    # reports[0] of either filtered list is always the longest.
    chromosome_reports = [r for r in reports if r.get("role") == "assembled-molecule"]
    best = chromosome_reports[0] if chromosome_reports else reports[0]

    # Prefer RefSeq accession (NC_* / NZ_*) when available; else GenBank.
    accession = best.get("refseq_accession") or best.get("genbank_accession")
    if not accession:
        raise ValueError(f"No accession found in sequence report: {best}")

    return accession, int(best["length"])


def fetch_window(contig_accession: str, start: int, end: int) -> str:
    """
    Fetch a genomic window from NCBI nuccore and return the raw FASTA string.

    Parameters
    ----------
    contig_accession : str
        A nucleotide accession such as 'NC_000913.3' or 'NZ_BCQD01000001.1'.
    start : int
        Start position, 1-based inclusive (NCBI seq_start convention).
    end : int
        End position, 1-based inclusive (NCBI seq_stop convention).

    Returns
    -------
    str
        The raw FASTA text returned by NCBI, including the header line.
    """
    handle = Entrez.efetch(
        db="nuccore",
        id=contig_accession,
        rettype="fasta",
        retmode="text",
        seq_start=start,
        seq_stop=end,
    )
    fasta = handle.read()
    handle.close()
    return fasta


# ---------------------------------------------------------------------------
# Window calculation helpers
# ---------------------------------------------------------------------------

def compute_window_start(ncbi_accession: str, window_idx: int, contig_length: int) -> int:
    """
    Deterministically compute a 1-based window start position.

    Uses an MD5-derived seed so that the same (ncbi_accession, window_idx)
    pair always yields the same position, regardless of run order.
    """
    seed_bytes = hashlib.md5((ncbi_accession + str(window_idx)).encode()).digest()
    seed_int = int.from_bytes(seed_bytes[:4], "big")
    max_start = max(1, contig_length - FETCH_BP)
    return (seed_int % max_start) + 1  # 1-based for NCBI


def parse_fasta_sequence(fasta_text: str) -> str:
    """
    Extract the nucleotide sequence (no header, no newlines) from a raw FASTA string.
    """
    lines = fasta_text.strip().splitlines()
    seq_lines = [ln for ln in lines if ln and not ln.startswith(">")]
    return "".join(seq_lines)


# ---------------------------------------------------------------------------
# Per-species download
# ---------------------------------------------------------------------------

def download_species(
    ncbi_accession: str,
    out_path: Path,
) -> None:
    """
    Fetch 10 genomic windows for one species and write them to out_path.

    Raises on any error (caller catches and logs the warning).
    """
    # 1. Resolve assembly → primary contig
    contig_accession, contig_length = assembly_to_contig(ncbi_accession)
    time.sleep(SLEEP)

    if contig_length < FETCH_BP:
        raise ValueError(
            f"Contig {contig_accession} is only {contig_length} bp "
            f"(need >= {FETCH_BP} bp)"
        )

    sequences = []
    for i in range(WINDOWS_PER_SPECIES):
        window_start = compute_window_start(ncbi_accession, i, contig_length)
        window_end = window_start + FETCH_BP - 1  # 1-based inclusive

        fasta_text = fetch_window(contig_accession, window_start, window_end)
        time.sleep(SLEEP)

        seq = parse_fasta_sequence(fasta_text)
        if len(seq) < KEEP_BP:
            raise ValueError(
                f"Fetched sequence too short ({len(seq)} bp) for window "
                f"{contig_accession}:{window_start}-{window_end}"
            )

        # Keep only the last 2000 bp of the 4000 bp fetch window
        seq_2000 = seq[-KEEP_BP:]
        header = (
            f">{ncbi_accession}|{i}|{contig_accession}:{window_start}-{window_end}"
        )
        sequences.append((header, seq_2000))

    # Write all 10 sequences to FASTA
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for header, seq in sequences:
            fh.write(f"{header}\n{seq}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    manifest_path = repo_root / "data" / "species" / "gtdb_500_manifest.csv"
    seq_dir = repo_root / "data" / "species" / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Read manifest
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    print(f"Manifest loaded: {len(rows)} species")

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    for row in tqdm(rows, desc="Species", unit="sp"):
        ncbi_accession = row["ncbi_accession"].strip()
        out_path = seq_dir / f"{ncbi_accession}.fasta"

        if out_path.exists():
            n_skipped += 1
            continue

        try:
            download_species(ncbi_accession, out_path)
            n_downloaded += 1
        except Exception as exc:
            tqdm.write(f"WARNING: failed {ncbi_accession} — {exc}")
            n_failed += 1

    print(
        f"\nDone. downloaded={n_downloaded}, skipped={n_skipped}, failed={n_failed}"
    )


if __name__ == "__main__":
    main()
