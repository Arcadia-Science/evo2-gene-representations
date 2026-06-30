"""Download CDS sequences for a gene family from NCBI RefSeq and deduplicate with MMseqs2."""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gene_families import family_members  # noqa: E402

GENE_FAMILIES = family_members("human")

# NCBI requires a contact email; set NCBI_EMAIL in your environment.
Entrez.email = os.environ.get("NCBI_EMAIL", "")

SLEEP = 0.34
BATCH_SIZE = 50


def check_mmseqs2():
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs2 not found in PATH. Install it with: conda install -c bioconda mmseqs2"
        )


def fetch_gene_records(gene, max_candidates):
    query = f"{gene}[Gene Name] AND txid7776[Organism] AND srcdb_refseq[PROP] AND biomol_mrna[PROP]"
    handle = Entrez.esearch(db="nucleotide", term=query, usehistory="y", retmax=0)
    sr = Entrez.read(handle)
    handle.close()
    time.sleep(SLEEP)

    count = int(sr["Count"])
    webenv = sr["WebEnv"]
    query_key = sr["QueryKey"]

    records = []
    fetch_limit = min(count, max_candidates)
    for start in range(0, fetch_limit, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, fetch_limit - start)
        handle = Entrez.efetch(
            db="nucleotide",
            rettype="gb",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        batch = list(SeqIO.parse(handle, "genbank"))
        handle.close()
        records.extend(batch)
        time.sleep(SLEEP)

    return records


def extract_cds(record):
    # organism lives on the source feature; annotations["organism"] is a reliable fallback
    organism = record.annotations.get("organism", "unknown")

    for feat in record.features:
        if feat.type != "CDS":
            continue
        # skip incomplete CDS (positions with < or >)
        loc_str = str(feat.location)
        if "<" in loc_str or ">" in loc_str:
            continue

        cds_seq = str(feat.extract(record.seq))
        length = len(cds_seq)
        if length < 100 or length > 15000:
            continue

        gene_name = feat.qualifiers.get("gene", [""])[0]
        # only return the first valid CDS per record
        return {
            "accession": record.id,
            "organism": organism,
            "gene_name": gene_name,
            "cds_seq": cds_seq,
            "cds_length": length,
        }
    return None


def write_fasta(entries, path):
    with open(path, "w") as f:
        for e in entries:
            species = e["organism"].replace(" ", "_")
            header = f">{e['accession']}|{e['gene_name']}|{species}"
            f.write(f"{header}\n{e['cds_seq']}\n")


def run_mmseqs(input_fasta, output_prefix, tmp_dir):
    cmd = [
        "mmseqs",
        "easy-linclust",
        str(input_fasta),
        str(output_prefix),
        str(tmp_dir),
        "--dbtype",
        "2",
        "--min-seq-id",
        "0.90",
        "--cov-mode",
        "1",
        "-c",
        "0.90",
        "--cluster-mode",
        "2",
        "--kmer-per-seq",
        "80",
        "--spaced-kmer-mode",
        "0",
        "--threads",
        "4",
        "--remove-tmp-files",
        "1",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def append_manifest(manifest_path, family, entries):
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["family", "accession", "species", "gene_name", "cds_length"]
        )
        if write_header:
            writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "family": family,
                    "accession": e["accession"],
                    "species": e["organism"],
                    "gene_name": e["gene_name"],
                    "cds_length": e["cds_length"],
                }
            )


def parse_rep_fasta(rep_fasta_path):
    entries = []
    for record in SeqIO.parse(str(rep_fasta_path), "fasta"):
        # header format: accession|gene_name|species_with_underscores
        parts = record.id.split("|")
        accession = parts[0] if len(parts) > 0 else record.id
        gene_name = parts[1] if len(parts) > 1 else ""
        species = parts[2].replace("_", " ") if len(parts) > 2 else "unknown"
        entries.append(
            {
                "accession": accession,
                "organism": species,
                "gene_name": gene_name,
                "cds_seq": str(record.seq),
                "cds_length": len(record.seq),
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description="Download CDS sequences for a gene family.")
    parser.add_argument("--family", required=True, choices=list(GENE_FAMILIES.keys()))
    parser.add_argument("--max-seqs", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=1500)
    args = parser.parse_args()

    check_mmseqs2()

    genes = GENE_FAMILIES[args.family]
    per_gene_limit = math.ceil(args.candidates / len(genes))

    data_dir = Path("data/sequences")
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp_base = Path("tmp")
    tmp_base.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for gene in genes:
        print(f"  Fetching {gene} (up to {per_gene_limit} candidates)...")
        records = fetch_gene_records(gene, per_gene_limit)
        gene_entries = []
        for rec in records:
            if not (rec.id.startswith("NM_") or rec.id.startswith("XM_")):
                continue
            entry = extract_cds(rec)
            if entry is not None:
                gene_entries.append(entry)
        print(f"    {gene}: {len(gene_entries)} valid CDS sequences")
        all_entries.extend(gene_entries)

    # Keep only the longest CDS per (species, gene) before MMseqs2 so isoforms
    # from the same species don't consume representative slots.
    best: dict = {}
    for e in all_entries:
        key = (e["organism"].lower(), e["gene_name"].lower())
        if key not in best or e["cds_length"] > best[key]["cds_length"]:
            best[key] = e
    all_entries = list(best.values())[: args.candidates]
    print(f"Total candidates after per-species dedup: {len(all_entries)}")

    if len(all_entries) == 0:
        print("ERROR: no candidate sequences found.", file=sys.stderr)
        sys.exit(1)

    tmp_dir = tmp_base / f"mmseqs_{args.family}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidate_fasta = tmp_dir / "candidates.fasta"
    write_fasta(all_entries, candidate_fasta)

    output_prefix = tmp_dir / "dedup"
    print("Running MMseqs2 easy-linclust at 90% identity...")
    run_mmseqs(candidate_fasta, output_prefix, tmp_dir / "mmseqs_tmp")

    rep_fasta = Path(str(output_prefix) + "_rep_seq.fasta")
    rep_entries = parse_rep_fasta(rep_fasta)
    print(f"After dedup: {len(rep_entries)} representative sequences")

    if len(rep_entries) < 10:
        print(
            f"WARNING: only {len(rep_entries)} sequences survived deduplication for {args.family}"
        )

    final_entries = rep_entries[: args.max_seqs]

    final_fasta = data_dir / f"{args.family}.fasta"
    write_fasta(final_entries, final_fasta)
    print(f"Wrote {len(final_entries)} sequences to {final_fasta}")

    manifest_path = data_dir / "manifest.csv"
    append_manifest(manifest_path, args.family, final_entries)
    print(f"Manifest updated: {manifest_path}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
