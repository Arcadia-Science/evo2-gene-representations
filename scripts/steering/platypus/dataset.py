"""Stage 1a — build the paired human/platypus ortholog dataset for the paired-prefix steering study."""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from steer_lib import codon_blocks, codon_diagnostics  # noqa: E402

DATA = ROOT / "data" / "mammalian_orthologs"
CDS_DIR = DATA / "seqs" / "cds"
HUMAN = "homo_sapiens"
PLATYPUS = "ornithorhynchus_anatinus"
OPOSSUM = "monodelphis_domestica"  # Stage-4 wrong-species control; recorded here, not gated on

PREFIX_BP = 90
BASES = set("ACGT")

# Families require enough platypus-covered members for leave-one-gene-out estimation.
FAMILIES = [
    "hox",
    "rab_gtpase",
    "ras_gtpases",
    "cytochrome_p450",
    "arf_gtpase",
    "rho_gtpase",
]

# Apply the same inclusion rule to both species.
TRANSCRIPT_RULE = (
    "Ensembl canonical transcript (REST `is_canonical`, or GTF tag `Ensembl_canonical`); "
    "if no canonical flag is present, the transcript with the longest CDS. "
    "Applied identically to human and platypus (mammalian_orthologs/extract_loci.py:89-104, "
    "extract_loci_bulk.py:103-114)."
)
ORTHOLOGY_SOURCE = (
    "Ensembl Compara one-to-one orthologs of the human gene, resolved by "
    "scripts/mammalian_orthologs/resolve_orthologs.py into "
    "data/mammalian_orthologs/ortholog_resolution.csv; only rows with type == 'ortholog_one2one' "
    "are admitted here."
)


# --------------------------------------------------------------------------- loading
def read_family_fasta(family: str) -> dict[tuple[str, str], tuple[str, str]]:
    """{(gene, species): (cds, gene_id)} from a family fasta (header `gene|species|gene_id`)."""
    path = CDS_DIR / f"{family}.fasta"
    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not path.exists():
        return out
    hid: str | None = None
    seq: list[str] = []

    def flush() -> None:
        if hid is None:
            return
        parts = hid.split("|")
        if len(parts) < 3:
            return
        out[(parts[0], parts[1])] = ("".join(seq).upper(), parts[2])

    for line in path.read_text().splitlines():
        if line.startswith(">"):
            flush()
            hid, seq = line[1:].strip(), []
        elif line.strip():
            seq.append(line.strip())
    flush()
    return out


def read_orthology() -> dict[tuple[str, str, str], str]:
    """{(family, human_gene, species): ortholog_gene_id} for one2one rows only."""
    out = {}
    with open(DATA / "ortholog_resolution.csv") as fh:
        for r in csv.DictReader(fh):
            if r["type"] == "ortholog_one2one":
                out[(r["family"], r["human_gene"], r["species"])] = r["ortholog_gene_id"]
    return out


def read_loci() -> dict[tuple[str, str, str], dict]:
    """{(family, gene, species): manifest row} — the canonical-transcript record."""
    out = {}
    with open(DATA / "loci_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            out[(r["family"], r["group"], r["species"])] = r
    return out


def read_gene_panel() -> set[str]:
    """Genes present in the pre-existing powered steering panel (annotation only, not a filter)."""
    path = ROOT / "scripts" / "steering" / "gene_panel.csv"
    if not path.exists():
        return set()
    with open(path) as fh:
        return {r["gene"] for r in csv.DictReader(fh)}


# --------------------------------------------------------------------------- QC
def check_cds(cds: str, tag: str) -> str | None:
    """Return an exclusion reason for one CDS, or None if it passes."""
    if not cds:
        return f"missing_cds_{tag}"
    if set(cds) - BASES:
        bad = "".join(sorted(set(cds) - BASES))[:8]
        return f"nonstandard_bases_{tag}[{bad}]"
    if len(cds) < PREFIX_BP:
        return f"too_short_{tag}({len(cds)}bp)"
    if len(cds) % 3 != 0:
        return f"frame_not_multiple_of_3_{tag}({len(cds)}bp)"
    if not cds.startswith("ATG"):
        return f"no_start_codon_{tag}({cds[:3]})"
    return None


def qc_pair(human_cds: str, platypus_cds: str) -> tuple[str | None, dict]:
    """Paired-prefix QC gate. Returns (exclusion_reason_or_None, alignment_metadata)."""
    meta: dict = {}
    for cds, tag in ((human_cds, "human"), (platypus_cds, "platypus")):
        reason = check_cds(cds, tag)
        if reason:
            return reason, meta

    n_codons = PREFIX_BP // 3
    blocks_h, blocks_t, ph, pt = codon_blocks(human_cds, platypus_cds)
    if len(blocks_h) == 0:
        return "no_codon_alignment", meta

    (hs, he) = (int(blocks_h[0][0]), int(blocks_h[0][1]))
    (ts, te) = (int(blocks_t[0][0]), int(blocks_t[0][1]))
    matched = sum(int(e - s) for s, e in blocks_h)
    n_same = 0
    for (bh_s, bh_e), (bt_s, _bt_e) in zip(blocks_h, blocks_t, strict=True):
        for k in range(int(bh_e - bh_s)):
            n_same += ph[int(bh_s) + k] == pt[int(bt_s) + k]
    meta.update(
        aln_first_block_human=f"{hs}-{he}",
        aln_first_block_platypus=f"{ts}-{te}",
        aln_n_matched_codons=matched,
        aln_n_blocks=len(blocks_h),
        aln_indel_frac=round(1.0 - matched / max(len(ph), len(pt)), 6),
        aa_identity=round(n_same / max(matched, 1), 6),
    )
    if hs != 0 or ts != 0:
        return f"prefix_not_homologous(first_block_starts_h{int(hs)}_p{int(ts)})", meta
    if he < n_codons or te < n_codons:
        return f"indel_within_prefix(first_block_ends_h{int(he)}_p{int(te)}<{n_codons})", meta

    # In-frame stop inside the prompt would make the reading frame ambiguous for the continuation.
    for aa_seq, tag in ((ph, "human"), (pt, "platypus")):
        if "*" in aa_seq[:n_codons]:
            return f"internal_stop_in_prefix_{tag}", meta
    return None, meta


# --------------------------------------------------------------------------- build
def build(families: list[str]) -> tuple[list[dict], list[dict]]:
    orth = read_orthology()
    loci = read_loci()
    panel = read_gene_panel()

    retained: list[dict] = []
    excluded: list[dict] = []

    for family in families:
        fa = read_family_fasta(family)
        genes = sorted({g for (g, sp) in fa if sp == HUMAN})
        for gene in genes:
            human = fa.get((gene, HUMAN))
            plat = fa.get((gene, PLATYPUS))
            row_base = {"family": family, "gene": gene}

            if plat is None:
                excluded.append({**row_base, "reason": "no_platypus_cds"})
                continue
            orth_id = orth.get((family, gene, PLATYPUS))
            if orth_id is None:
                excluded.append({**row_base, "reason": "no_one2one_evidence"})
                continue
            if orth_id != plat[1]:
                excluded.append({
                    **row_base,
                    "reason": f"orthology_gene_id_mismatch(resolution={orth_id},fasta={plat[1]})",
                })
                continue

            lh = loci.get((family, gene, HUMAN))
            lp = loci.get((family, gene, PLATYPUS))
            if lh is None or lp is None:
                miss = "human" if lh is None else "platypus"
                excluded.append({**row_base, "reason": f"no_transcript_record_{miss}"})
                continue

            reason, meta = qc_pair(human[0], plat[0])
            if reason:
                excluded.append({**row_base, "reason": reason, **meta})
                continue

            diag, cons = codon_diagnostics(human[0], plat[0])
            n_after = sum(1 for d in diag if d["h_idx"] >= PREFIX_BP)
            retained.append({
                **row_base,
                "human_gene_id": human[1],
                "human_transcript_id": lh["transcript_id"],
                "human_cds_len": len(human[0]),
                "human_qc_flags": lh["qc_flags"],
                "platypus_gene_id": plat[1],
                "platypus_transcript_id": lp["transcript_id"],
                "platypus_cds_len": len(plat[0]),
                "platypus_qc_flags": lp["qc_flags"],
                "orthology_type": "ortholog_one2one",
                "prefix_human": human[0][:PREFIX_BP],
                "prefix_platypus": plat[0][:PREFIX_BP],
                "prefix_n_diff": sum(
                    1 for a, b in zip(human[0][:PREFIX_BP], plat[0][:PREFIX_BP], strict=True)
                    if a != b),
                "n_diag_total": len(diag),
                "n_diag_after_prefix": n_after,
                "n_conserved": len(cons),
                "in_gene_panel": gene in panel,
                "has_opossum": (gene, OPOSSUM) in fa,
                **meta,
            })

    return retained, excluded


def write_fasta(path: Path, rows: list[dict], species: str, cds_by_key: dict) -> None:
    """Full CDS for every retained pair, header `gene|species|gene_id|transcript_id`."""
    tag = "human" if species == HUMAN else "platypus"
    with open(path, "w") as fh:
        for r in rows:
            seq = cds_by_key[(r["family"], (r["gene"], species))]
            fh.write(f">{r['gene']}|{species}|{r[tag + '_gene_id']}|{r[tag + '_transcript_id']}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "2026-07-28_evo2-platypus-paired" / "stage1")
    ap.add_argument("--families", nargs="*", default=FAMILIES)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    retained, excluded = build(args.families)

    # full CDS for every retained pair, keyed for the fasta writer
    cds_by_key: dict = {}
    for family in args.families:
        for (gene, sp), (seq, _gid) in read_family_fasta(family).items():
            cds_by_key[(family, (gene, sp))] = seq

    fields = list(retained[0].keys()) if retained else []
    with open(args.out_dir / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(retained)

    exc_fields = sorted({k for r in excluded for k in r})
    with open(args.out_dir / "excluded.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=exc_fields)
        w.writeheader()
        w.writerows(excluded)

    write_fasta(args.out_dir / "cds_human.fasta", retained, HUMAN, cds_by_key)
    write_fasta(args.out_dir / "cds_platypus.fasta", retained, PLATYPUS, cds_by_key)

    by_family: dict[str, dict[str, int]] = {}
    for f in args.families:
        by_family[f] = {
            "candidate": sum(1 for r in retained if r["family"] == f)
            + sum(1 for r in excluded if r["family"] == f),
            "retained": sum(1 for r in retained if r["family"] == f),
        }
    summary = {
        "prefix_bp": PREFIX_BP,
        "families": args.families,
        "n_candidate": len(retained) + len(excluded),
        "n_retained": len(retained),
        "n_excluded": len(excluded),
        "n_retained_in_gene_panel": sum(1 for r in retained if r["in_gene_panel"]),
        "n_retained_with_opossum": sum(1 for r in retained if r["has_opossum"]),
        "by_family": by_family,
        "transcript_rule": TRANSCRIPT_RULE,
        "orthology_source": ORTHOLOGY_SOURCE,
        "human_species": HUMAN,
        "target_species": PLATYPUS,
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"candidate={summary['n_candidate']}  retained={summary['n_retained']}  "
          f"excluded={summary['n_excluded']}")
    for f, d in by_family.items():
        print(f"  {f:18s} candidate={d['candidate']:3d}  retained={d['retained']:3d}")
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
