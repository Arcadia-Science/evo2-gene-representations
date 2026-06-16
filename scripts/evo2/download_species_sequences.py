"""Download genome-wide sampled 4000 bp windows for GTDB bacterial species.

For each species, this script downloads one NCBI Datasets genome package, samples
4000 bp windows locally across all eligible contigs, and stores only the sampled
windows. By default it samples ~5% of the manifest genome size per species.

The full 4000 bp window is stored. At embedding time the first 2000 bp prime the
autoregressive model and only the final 2000 bp are pooled.
"""

import argparse
import csv
import gzip
import hashlib
import io
import math
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm

DATASETS_DOWNLOAD_URL = (
    "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{accession}/download"
)

DEFAULT_COVERAGE = 0.05
DEFAULT_SLEEP = 0.34
FETCH_BP = 4000
KEEP_BP = 2000
MIN_WINDOWS_PER_SPECIES = 1


class Contig(NamedTuple):
    accession: str
    sequence: str
    length: int
    n_start_positions: int


class Window(NamedTuple):
    index: int
    contig_accession: str
    start: int
    end: int
    contig_length: int
    sequence: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        default="data/species/gtdb_500_manifest.csv",
        help="Species manifest CSV.",
    )
    p.add_argument(
        "--sequences-dir",
        default="data/species/sequences_5pct",
        help="Output directory for sampled per-species FASTA files.",
    )
    p.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help="Target fraction of each genome to sample.",
    )
    p.add_argument(
        "--window-bp",
        type=int,
        default=FETCH_BP,
        help="Length of each sampled genomic window.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Delay after each genome-package request.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for transient NCBI download failures.",
    )
    p.add_argument(
        "--max-species",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild sampled windows even if complete outputs already exist.",
    )
    return p.parse_args()


def n_windows_for_genome(genome_size: int, coverage: float, window_bp: int) -> int:
    if genome_size <= 0:
        raise ValueError(f"Genome size must be positive, got {genome_size}")
    return max(MIN_WINDOWS_PER_SPECIES, math.ceil(genome_size * coverage / window_bp))


def genome_package_url(ncbi_accession: str) -> str:
    accession = urllib.parse.quote(ncbi_accession, safe="")
    params: list[tuple[str, str]] = [("include_annotation_type", "GENOME_FASTA")]
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params.append(("api_key", api_key))
    return DATASETS_DOWNLOAD_URL.format(accession=accession) + "?" + urllib.parse.urlencode(params)


def request_headers() -> dict[str, str]:
    email = os.environ.get("NCBI_EMAIL", "")
    user_agent = "glm-latent-mapping/0.1"
    if email:
        user_agent += f" ({email})"
    return {"Accept": "application/zip", "User-Agent": user_agent}


def download_genome_package(ncbi_accession: str, max_retries: int) -> bytes:
    url = genome_package_url(ncbi_accession)
    req = urllib.request.Request(url, headers=request_headers())

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            if not zipfile.is_zipfile(io.BytesIO(data)):
                preview = data[:200].decode("utf-8", errors="replace")
                raise ValueError(f"NCBI response was not a zip file: {preview!r}")
            return data
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == max_retries:
                body = exc.read()[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"HTTP {exc.code} from NCBI for {ncbi_accession}: {body}"
                ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {ncbi_accession}: {exc}") from exc

        time.sleep(min(60.0, 2.0**attempt))

    raise RuntimeError(f"Failed to download {ncbi_accession}")


def parse_fasta_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts).upper()))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    return records


def load_genome_contigs(package_bytes: bytes, window_bp: int) -> list[Contig]:
    contigs: list[Contig] = []
    with zipfile.ZipFile(io.BytesIO(package_bytes)) as zf:
        fasta_names = sorted(
            name
            for name in zf.namelist()
            if not name.endswith("/")
            and name.lower().endswith((".fna", ".fa", ".fasta", ".fna.gz", ".fa.gz", ".fasta.gz"))
        )
        if not fasta_names:
            raise ValueError("No genome FASTA found in NCBI Datasets package")

        for name in fasta_names:
            raw = zf.read(name)
            if name.lower().endswith(".gz"):
                text = gzip.decompress(raw).decode("utf-8", errors="replace")
            else:
                text = raw.decode("utf-8", errors="replace")

            for header, seq in parse_fasta_records(text):
                accession = header.split()[0] if header else f"contig_{len(contigs)}"
                seq = "".join(seq.split()).upper()
                length = len(seq)
                if length >= window_bp:
                    contigs.append(
                        Contig(
                            accession=accession,
                            sequence=seq,
                            length=length,
                            n_start_positions=length - window_bp + 1,
                        )
                    )

    if not contigs:
        raise ValueError(f"No contigs at least {window_bp} bp long")
    return sorted(contigs, key=lambda c: c.accession)


def sample_windows(
    ncbi_accession: str,
    contigs: list[Contig],
    n_windows: int,
    coverage: float,
    window_bp: int,
) -> list[Window]:
    seed_material = f"{ncbi_accession}|{coverage:.12g}|{window_bp}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = random.Random(seed)

    total_positions = sum(c.n_start_positions for c in contigs)
    windows: list[Window] = []
    for idx in range(n_windows):
        pick = rng.randrange(total_positions)
        offset = 0
        chosen = contigs[-1]
        start0 = 0
        for contig in contigs:
            next_offset = offset + contig.n_start_positions
            if pick < next_offset:
                chosen = contig
                start0 = pick - offset
                break
            offset = next_offset

        seq = chosen.sequence[start0 : start0 + window_bp]
        start = start0 + 1
        end = start0 + window_bp
        windows.append(
            Window(
                index=idx,
                contig_accession=chosen.accession,
                start=start,
                end=end,
                contig_length=chosen.length,
                sequence=seq,
            )
        )
    return windows


def count_fasta_records(fasta_path: Path) -> int:
    if not fasta_path.exists():
        return 0
    count = 0
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def metadata_is_complete(meta_path: Path, expected_n: int, coverage: float, window_bp: int) -> bool:
    if not meta_path.exists():
        return False
    with open(meta_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != expected_n:
        return False
    for row in rows:
        if int(row["n_windows"]) != expected_n:
            return False
        if int(row["window_bp"]) != window_bp:
            return False
        if abs(float(row["coverage_target"]) - coverage) > 1e-12:
            return False
    return True


def outputs_are_complete(
    fasta_path: Path,
    meta_path: Path,
    expected_n: int,
    coverage: float,
    window_bp: int,
) -> bool:
    return count_fasta_records(fasta_path) == expected_n and metadata_is_complete(
        meta_path, expected_n, coverage, window_bp
    )


def write_outputs(
    ncbi_accession: str,
    fasta_path: Path,
    meta_path: Path,
    windows: list[Window],
    genome_size: int,
    eligible_bp: int,
    coverage: float,
    window_bp: int,
) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fasta = fasta_path.with_name(f"{fasta_path.name}.tmp")
    tmp_meta = meta_path.with_name(f"{meta_path.name}.tmp")

    with open(tmp_fasta, "w") as fh:
        for win in windows:
            header = f">{ncbi_accession}|{win.index}|{win.contig_accession}:{win.start}-{win.end}"
            fh.write(f"{header}\n{win.sequence}\n")

    fieldnames = [
        "ncbi_accession",
        "window_idx",
        "contig_accession",
        "start",
        "end",
        "contig_length",
        "window_bp",
        "genome_size_manifest",
        "eligible_genome_bp",
        "coverage_target",
        "n_windows",
    ]
    with open(tmp_meta, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for win in windows:
            writer.writerow(
                {
                    "ncbi_accession": ncbi_accession,
                    "window_idx": win.index,
                    "contig_accession": win.contig_accession,
                    "start": win.start,
                    "end": win.end,
                    "contig_length": win.contig_length,
                    "window_bp": window_bp,
                    "genome_size_manifest": genome_size,
                    "eligible_genome_bp": eligible_bp,
                    "coverage_target": coverage,
                    "n_windows": len(windows),
                }
            )

    tmp_fasta.replace(fasta_path)
    tmp_meta.replace(meta_path)


def download_species(row: dict[str, str], args: argparse.Namespace) -> tuple[str, int]:
    ncbi_accession = row["ncbi_accession"].strip()
    genome_size = int(float(row["genome_size"]))
    expected_n = n_windows_for_genome(genome_size, args.coverage, args.window_bp)

    seq_dir = Path(args.sequences_dir)
    fasta_path = seq_dir / f"{ncbi_accession}.fasta"
    meta_path = seq_dir / f"{ncbi_accession}.windows.csv"
    if not args.force and outputs_are_complete(
        fasta_path, meta_path, expected_n, args.coverage, args.window_bp
    ):
        return "skipped", expected_n

    package_bytes = download_genome_package(ncbi_accession, args.max_retries)
    time.sleep(args.sleep)

    contigs = load_genome_contigs(package_bytes, args.window_bp)
    windows = sample_windows(
        ncbi_accession=ncbi_accession,
        contigs=contigs,
        n_windows=expected_n,
        coverage=args.coverage,
        window_bp=args.window_bp,
    )
    eligible_bp = sum(c.length for c in contigs)
    write_outputs(
        ncbi_accession=ncbi_accession,
        fasta_path=fasta_path,
        meta_path=meta_path,
        windows=windows,
        genome_size=genome_size,
        eligible_bp=eligible_bp,
        coverage=args.coverage,
        window_bp=args.window_bp,
    )
    return "downloaded", expected_n


def main() -> None:
    args = parse_args()
    if args.coverage <= 0:
        raise ValueError(f"coverage must be positive, got {args.coverage}")
    if args.window_bp <= 0:
        raise ValueError(f"window-bp must be positive, got {args.window_bp}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.max_species is not None:
        rows = rows[: args.max_species]

    total_expected_windows = sum(
        n_windows_for_genome(int(float(row["genome_size"])), args.coverage, args.window_bp)
        for row in rows
    )
    print(f"Manifest loaded: {len(rows)} species")
    print(
        f"Target coverage: {args.coverage:.3%}; "
        f"expected windows={total_expected_windows:,}; "
        f"stored bp={total_expected_windows * args.window_bp:,}"
    )
    print(f"Output directory: {args.sequences_dir}")

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    n_windows = 0

    for row in tqdm(rows, desc="Species", unit="sp"):
        ncbi_accession = row["ncbi_accession"].strip()
        try:
            status, n = download_species(row, args)
            n_windows += n
            if status == "skipped":
                n_skipped += 1
            else:
                n_downloaded += 1
        except Exception as exc:
            tqdm.write(f"WARNING: failed {ncbi_accession} - {exc}")
            n_failed += 1

    print(
        "\nDone. "
        f"downloaded={n_downloaded}, skipped={n_skipped}, failed={n_failed}, "
        f"windows={n_windows:,}"
    )

    # A handful of NCBI failures is normal; a large fraction usually means rate
    # limiting / network trouble, which would silently leave the embed step with
    # far fewer species than expected. Surface it loudly and exit non-zero so the
    # pipeline runner stops instead of proceeding on a degraded download.
    fail_frac = n_failed / len(rows) if rows else 0.0
    if n_failed:
        print(
            f"WARNING: {n_failed}/{len(rows)} species failed to download ({fail_frac:.0%}).",
            file=sys.stderr,
        )
    if fail_frac > 0.2:
        sys.exit(
            f"ERROR: {fail_frac:.0%} of species failed (> 20% threshold). NCBI is "
            "likely rate-limiting — set NCBI_API_KEY and/or re-run (downloads are "
            "resumable; completed species are skipped)."
        )


if __name__ == "__main__":
    main()
