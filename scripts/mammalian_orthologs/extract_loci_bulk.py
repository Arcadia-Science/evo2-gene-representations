"""Local (no-REST) locus extraction from Ensembl release-116 bulk files."""

from __future__ import annotations
import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq
from pyfaidx import Fasta

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from assemble_datasets import LOCI_CACHE, OUT_DIR, build_manifest  # noqa: E402

GENOMES = paths.MAMMAL_GENOMES


def _attrs(field: str) -> dict:
    """Parse a GTF attribute column: key "value"; key "value"; ..."""
    out = {}
    for part in field.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(" ")
        out[k] = v.strip().strip('"')
    return out


def _fast_gid(attr: str) -> str | None:
    """Cheap gene_id extraction without full-parsing the attribute column (GTF has millions of
    lines; only ~thousands belong to needed genes, so filter before the expensive _attrs)."""
    i = attr.find('gene_id "')
    if i < 0:
        return None
    j = attr.find('"', i + 9)
    return attr[i + 9 : j] if j > 0 else None


def parse_gtf(gtf_path: Path, needed: set[str]):
    """One pass over a species GTF."""
    # per gene: collect candidate transcripts -> pick canonical/longest at the end
    cand: dict[str, dict] = {}  # gene_id -> {tx_id -> record}
    all_genes: dict[str, list] = {}
    op = gzip.open if gtf_path.suffix == ".gz" else open
    with op(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            feat, attr = f[2], f[8]
            if feat == "gene":
                a = _attrs(attr)
                all_genes.setdefault(f[0], []).append(
                    (int(f[3]), int(f[4]), a.get("gene_id"), a.get("gene_biotype", ""))
                )
                continue
            if feat not in ("transcript", "CDS", "start_codon", "stop_codon"):
                continue
            gid = _fast_gid(attr)  # cheap filter before the expensive full parse
            if gid not in needed:
                continue
            chrom, start, end, strand = f[0], int(f[3]), int(f[4]), f[6]
            a = _attrs(attr)
            tid = a.get("transcript_id")
            if feat == "transcript":
                rec = cand.setdefault(gid, {}).setdefault(
                    tid,
                    {
                        "chrom": chrom,
                        "strand": strand,
                        "tx_start": start,
                        "tx_end": end,
                        "cds": [],
                        "has_start": False,
                        "has_stop": False,
                        "biotype": a.get("transcript_biotype", ""),
                        "tx_id": tid,
                        "canonical": "Ensembl_canonical" in attr,
                    },
                )
                rec["tx_start"], rec["tx_end"] = start, end
            elif feat == "CDS":
                cand.setdefault(gid, {}).setdefault(tid, _blank(chrom, strand, tid))["cds"].append(
                    (start, end)
                )
            elif feat == "start_codon":
                cand.setdefault(gid, {}).setdefault(tid, _blank(chrom, strand, tid))[
                    "has_start"
                ] = True
            elif feat == "stop_codon":
                # Ensembl GTF EXCLUDES the stop codon from CDS features — add it so the derived CDS
                # is complete (ends in a stop, length %3 == 0).
                r = cand.setdefault(gid, {}).setdefault(tid, _blank(chrom, strand, tid))
                r["has_stop"] = True
                r["cds"].append((start, end))
    models = {}
    for gid, txs in cand.items():
        # prefer the canonical transcript; else the one with the longest CDS
        canon = [t for t in txs.values() if t.get("canonical")]
        pick = (
            canon[0]
            if canon
            else max(txs.values(), key=lambda t: sum(e - s + 1 for s, e in t["cds"]))
        )
        pick["cds"] = sorted(pick["cds"])
        models[gid] = pick
    return models, all_genes


def _blank(chrom, strand, tid):
    return {
        "chrom": chrom,
        "strand": strand,
        "tx_start": None,
        "tx_end": None,
        "cds": [],
        "has_start": False,
        "has_stop": False,
        "biotype": "",
        "tx_id": tid,
        "canonical": False,
    }


def _count_overlap(all_genes, chrom, gid, s, e) -> int:
    return sum(
        1
        for gs, ge, ggid, bt in all_genes.get(chrom, [])
        if ggid != gid and bt == "protein_coding" and gs <= e and ge >= s
    )


def extract_species(
    sp: str, prov: dict, needed_ids: dict[str, str], res_rows: pd.DataFrame
) -> None:
    """needed_ids: gene_id -> group(human_gene) for this species. res_rows: rows for this species
    (family/clade). Writes loci/<group>__<sp>.json for each."""
    d = GENOMES / sp
    info = prov["species"][sp]
    gtf = d / info["gtf"]
    fa_path = d / info["dna"]
    if not (gtf.exists() and fa_path.exists()):
        print(f"  {sp}: bulk files missing, skip", flush=True)
        return
    models, all_genes = parse_gtf(gtf, set(needed_ids))
    fa = Fasta(str(fa_path), rebuild=False)
    meta = {(r.human_gene): (r.family, r.clade) for r in res_rows.itertuples()}
    for gid, group in needed_ids.items():
        key = LOCI_CACHE / f"{group}__{sp}.json"
        if key.exists():
            continue
        fam, clade = meta.get(group, (None, "primate" if sp == "homo_sapiens" else None))
        m = models.get(gid)
        rec = {"family": fam, "human_gene": group, "species": sp, "clade": clade}
        if m is None or not m["cds"] or m["chrom"] not in fa:
            rec["error"] = "no_model" if m is None else "no_cds_or_chrom"
            key.write_text(json.dumps(rec))
            continue
        chrom, strand = m["chrom"], m["strand"]
        # transcript-span locus (5'UTR+introns+exons+3'UTR)
        locus = fa[chrom][m["tx_start"] - 1 : m["tx_end"]].seq.upper()
        # spliced CDS: concat coding exons in genomic order, revcomp whole if minus strand
        cds = "".join(fa[chrom][s - 1 : e].seq.upper() for s, e in m["cds"])
        if strand == "-":
            locus = str(Seq(locus).reverse_complement())
            cds = str(Seq(cds).reverse_complement())
        rec.update(
            {
                "gene_id": gid,
                "transcript_id": m["tx_id"],
                "chrom": chrom,
                "tx_start": m["tx_start"],
                "tx_end": m["tx_end"],
                "strand": strand,
                "biotype": m["biotype"],
                "n_exons": None,
                "locus_seq": locus,
                "cds_seq": cds,
                "has_start_codon": m["has_start"],
                "has_stop_codon": m["has_stop"],
                "n_overlap_genes": _count_overlap(
                    all_genes, chrom, gid, m["tx_start"], m["tx_end"]
                ),
            }
        )
        key.write_text(json.dumps(rec))
    print(f"  {sp}: {len(needed_ids)} loci written", flush=True)


def human_gene_ids_from_gtf(prov: dict, symbols: set[str]) -> dict[str, str]:
    """Map human gene SYMBOL -> release-116 gene_id via the human GTF gene_name (no REST)."""
    info = prov["species"]["homo_sapiens"]
    gtf = GENOMES / "homo_sapiens" / info["gtf"]
    out = {}
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            a = _attrs(line.rstrip("\n").split("\t")[8])
            nm = a.get("gene_name")
            if nm in symbols:
                out[nm] = a.get("gene_id")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()
    paths.require(
        GENOMES / "provenance.json",
        "the Ensembl genome set for the 24-mammal panel",
        "run scripts/mammalian_orthologs/download_bulk.py (experiment 1, stage A2)",
        "GLM_MAMMAL_GENOMES",
    )
    prov = json.loads((GENOMES / "provenance.json").read_text())
    res = pd.read_csv(OUT_DIR / "ortholog_resolution.csv")
    if args.families:
        res = res[res["family"].isin(args.families)]
    LOCI_CACHE.mkdir(parents=True, exist_ok=True)

    groups = sorted(res["human_gene"].unique())
    hmap = human_gene_ids_from_gtf(prov, set(groups))
    print(f"resolved {len(hmap)}/{len(groups)} human gene ids from GTF", flush=True)

    # per species: gene_id -> group
    for sp in prov["species"]:
        if sp == "homo_sapiens":
            needed = {gid: g for g, gid in hmap.items()}
            rows = res  # for family/clade lookup by group
        else:
            sub = res[res["species"] == sp]
            needed = {
                r.ortholog_gene_id: r.human_gene for r in sub.itertuples() if r.ortholog_gene_id
            }
            rows = sub
        if not needed:
            continue
        extract_species(sp, prov, needed, rows)
    print("EXTRACTION DONE; building manifest.", flush=True)
    build_manifest(pd.DataFrame())


if __name__ == "__main__":
    main()
