"""Build cross-kingdom gene-family CDS sets for the Evo2 gene-family pipeline, from KEGG Orthology.

Design: scripts/evo2/gene_family_design.md.

Each family is a curated SET of KEGG KO ids (KEGG's cross-kingdom functional
orthologs — the analog of the HGNC group ids used for the GPN-Star families). For
each family:

  1. link/genes/<KO>  -> every member gene (org:gene) across all KEGG organisms.
  2. map each org -> (domain, group) via the KEGG BRITE organism taxonomy (br08610).
  3. stratified-subsample across taxa to ~TARGET_PER_FAMILY (maximise tree-of-life
     spread, so the within-family phylogeny axis has signal).
  4. batch-fetch nucleotide CDS (/ntseq) — Evo2 is a DNA model, so we need CDS, not
     protein.
  5. (optional) MMseqs2 dedup to drop near-identical strain sequences.
  6. write per-family FASTA + a combined manifest.

Usage:
    uv run python scripts/evo2/build_gene_families_evo2.py
    uv run python scripts/evo2/build_gene_families_evo2.py --target 400 --no-dedup
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

KEGG = "https://rest.kegg.jp"
SLEEP = 0.34  # KEGG fair-use throttle
NTSEQ_BATCH = 10  # KEGG `get` accepts up to 10 entries per call

OUT_DIR = Path("data/evo2_gene_families")
BRITE_CACHE = Path("data/cache/kegg_br08610.txt")
TARGET_PER_FAMILY = 400
SUBSAMPLE_SEED = 0
OVERSAMPLE = 1.6  # fetch this × target as candidates so dedup still leaves ~target

# family -> {KO: subtype-label}. The label is a finer annotation carried into the
# manifest (e.g. type1/type2 for the opsin convergence control); for most families
# it just mirrors the KO's gene name. Edit/extend these sets to change membership.
FAMILY_KOS: dict[str, dict[str, str]] = {
    # Flagship: bacterial glbN → vertebrate Hb/Mb/Ngb/Cygb. Spans all of life.
    "globins": {
        "K06886": "bacterial_glbN",
        "K13822": "HBA",
        "K13823": "HBB",
        "K13824": "HBG",
        "K13825": "HBE",
        "K21892": "MB",
        "K21893": "NGB",
        "K21894": "CYGB",
    },
    # Heme-copper oxidase superfamily, catalytic SUBUNIT I only (homologous across
    # aa3 / cbb3 / cytochrome-o / archaeal aa3). NOT cytochrome bd (different fold).
    "heme_copper_oxidase": {
        "K02256": "aa3_COX1",
        "K02274": "aa3_coxA",
        "K00404": "cbb3_ccoN",
        "K02298": "cyo_cyoB",
        "K24009": "aa3_soxB",
        "K24011": "aa3_soxM",
    },
    # Cytochrome P450 superfamily. Representative cross-kingdom CYP KO set — KEGG
    # splits P450 into many CYP-specific KOs; expand this list for more breadth.
    "cytochrome_p450": {
        "K00490": "CYP4F",
        "K07408": "CYP1A1",
        "K07409": "CYP1A2",
        "K07410": "CYP1B1",
        "K07411": "CYP2A",
        "K07413": "CYP2C",
        "K07415": "CYP2E1",
        "K07424": "CYP3A",
        "K00517": "CYP_other",
    },
    # Olfactory receptors — one lumped KO (~198k genes); heavily subsampled. Vertebrate.
    "olfactory_receptors": {"K04257": "OLFR"},
    # Opsins as a CONVERGENCE CONTROL: animal type-2 (homologous to each other) +
    # microbial type-1 (a non-homologous, convergent family). Subtype tag preserved
    # so the analysis can test homology-vs-function grouping.
    "opsins": {
        "K04250": "type2",
        "K04251": "type2",
        "K04252": "type2",
        "K04253": "type2",
        "K04254": "type2",
        "K04255": "type2",
        "K04256": "type2",
        "K04641": "type1",
        "K04642": "type1",
        "K04643": "type1",
    },
    # Control: small-GTPase RAS family (eukaryotic, tight, well understood).
    "ras_gtpases": {"K07827": "KRAS", "K02833": "HRAS"},
}


# ── HTTP ──────────────────────────────────────────────────────────────────────


def kegg_get(path: str) -> str:
    """GET {KEGG}/{path} as text, with retries; '' on a 404-style empty body."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{KEGG}/{path}", timeout=60) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"  KEGG GET {path} failed after retries ({e})")
                return ""
            time.sleep(2 * 2**attempt)
    return ""


# ── BRITE organism taxonomy (org code -> domain/group) ──────────────────────────

_LEAF = re.compile(r"^([a-z][a-z0-9]{2,4})\s{2,}(.+)$")  # "hsa  Homo sapiens (human)"


def load_org_taxonomy() -> dict[str, tuple[str, str]]:
    """Parse BRITE br08610 -> {org_code: (domain, group)}.

    domain = top level (Eukaryota / Bacteria / Archaea); group = a mid-level taxon
    used for stratification (e.g. Metazoa, Proteobacteria).
    """
    if BRITE_CACHE.exists():
        text = BRITE_CACHE.read_text()
    else:
        print("Downloading KEGG BRITE organism taxonomy (br08610)...")
        text = kegg_get("get/br:br08610")
        BRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BRITE_CACHE.write_text(text)
        time.sleep(SLEEP)

    org_tax: dict[str, tuple[str, str]] = {}
    path: dict[str, str] = {}  # level-letter -> label
    for line in text.splitlines():
        if not line or line[0] < "A" or line[0] > "Z":
            continue
        letter, content = line[0], line[1:].strip()
        if not content:
            continue
        m = _LEAF.match(content)
        if m:  # leaf: an organism
            org = m.group(1)
            domain = path.get("A", "Unknown")
            # group = deepest mid-level label below domain (B preferred, else C/D)
            group = path.get("B") or path.get("C") or domain
            org_tax[org] = (domain, group)
        else:  # interior taxon node: update the path, clear deeper levels
            path[letter] = content
            for deeper in [chr(c) for c in range(ord(letter) + 1, ord("Z") + 1)]:
                path.pop(deeper, None)
    print(f"  Parsed taxonomy for {len(org_tax)} KEGG organisms")
    return org_tax


# ── Members per family ──────────────────────────────────────────────────────────


def ko_members(ko: str) -> list[str]:
    """All member genes (org:gene) linked to a KO via link/genes."""
    text = kegg_get(f"link/genes/{ko}")
    time.sleep(SLEEP)
    members = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and ":" in parts[1]:
            members.append(parts[1])  # "org:gene"
    return members


def stratified_sample(
    members: list[tuple[str, str, str]],  # (org_gene, ko, label)
    org_tax: dict[str, tuple[str, str]],
    n_target: int,
    seed: int,
) -> list[tuple[str, str, str, str, str]]:
    """Round-robin across (domain, group) buckets to maximise taxonomic spread.

    Returns (org_gene, ko, label, domain, group) for the chosen members.
    """
    import random

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for org_gene, ko, label in members:
        org = org_gene.split(":")[0]
        domain, group = org_tax.get(org, ("Unknown", "Unknown"))
        buckets[(domain, group)].append((org_gene, ko, label, domain, group))

    for b in buckets.values():
        rng.shuffle(b)

    # Round-robin: take one from each non-empty bucket per pass until target reached.
    order = sorted(buckets)
    chosen: list = []
    while len(chosen) < n_target and any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                chosen.append(buckets[k].pop())
                if len(chosen) >= n_target:
                    break
    return chosen


# ── CDS fetch ────────────────────────────────────────────────────────────────


def fetch_ntseq(org_genes: list[str]) -> dict[str, str]:
    """Batch-fetch nucleotide CDS for org:gene ids -> {org_gene: sequence}."""
    seqs: dict[str, str] = {}
    for i in range(0, len(org_genes), NTSEQ_BATCH):
        batch = org_genes[i : i + NTSEQ_BATCH]
        text = kegg_get(f"get/{'+'.join(batch)}/ntseq")
        time.sleep(SLEEP)
        cur_id, cur_seq = None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]  # ">hsa:3043 K13823 ..." -> "hsa:3043"
                cur_seq = []
            elif line.strip():
                cur_seq.append(line.strip().upper())
        if cur_id:
            seqs[cur_id] = "".join(cur_seq)
        print(f"    fetched {min(i + NTSEQ_BATCH, len(org_genes))}/{len(org_genes)} CDS", end="\r")
    print()
    return seqs


# ── MMseqs2 dedup (optional) ─────────────────────────────────────────────────


def mmseqs_dedup(fasta_in: Path, tmp_dir: Path, min_id: float = 0.90) -> set[str]:
    """Return the set of representative FASTA ids after easy-linclust, or all ids if
    mmseqs is unavailable."""
    if shutil.which("mmseqs") is None:
        print("  mmseqs not found — skipping dedup (install via bioconda to enable).")
        return {r.split()[0] for r in fasta_in.read_text().splitlines() if r.startswith(">")}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix = tmp_dir / "dedup"
    subprocess.run(
        [
            "mmseqs",
            "easy-linclust",
            str(fasta_in),
            str(prefix),
            str(tmp_dir / "tmp"),
            "--dbtype",
            "2",
            "--min-seq-id",
            str(min_id),
            "--cov-mode",
            "1",
            "-c",
            "0.8",
            "--cluster-mode",
            "2",
            "--threads",
            "4",
            "--remove-tmp-files",
            "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    rep = Path(str(prefix) + "_rep_seq.fasta")
    reps = {line[1:].split()[0] for line in rep.read_text().splitlines() if line.startswith(">")}
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return reps


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--target", type=int, default=TARGET_PER_FAMILY, help="CDS per family after dedup"
    )
    p.add_argument("--families", nargs="+", default=list(FAMILY_KOS), choices=list(FAMILY_KOS))
    p.add_argument("--no-dedup", action="store_true", help="Skip MMseqs2 dedup")
    p.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    org_tax = load_org_taxonomy()

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "family",
                "ko",
                "ko_label",
                "org_gene",
                "organism",
                "domain",
                "group",
                "cds_len",
            ],
        )
        writer.writeheader()

        for family in args.families:
            print(f"\n=== {family} ===")
            # 1. gather members across the family's KOs
            members: list[tuple[str, str, str]] = []
            for ko, label in FAMILY_KOS[family].items():
                m = ko_members(ko)
                print(f"  {ko} ({label}): {len(m)} genes")
                members.extend((og, ko, label) for og in m)

            # kegg_get returns '' on failure (incl. 404 / outage), so an empty
            # member list usually means KEGG was unreachable rather than a real
            # empty family. Warn loudly instead of silently building a tiny family.
            if not members:
                print(
                    f"  WARNING: 0 members for '{family}' — KEGG may be down or the "
                    "KO ids changed; skipping this family.",
                    file=sys.stderr,
                )
                continue

            # 2+3. stratified-subsample (oversample to survive dedup)
            n_candidates = int(args.target * OVERSAMPLE)
            chosen = stratified_sample(members, org_tax, n_candidates, args.seed)
            n_dom = len({c[3] for c in chosen})
            print(
                f"  {len(chosen)} candidates across {n_dom} domains "
                f"({len({c[4] for c in chosen})} taxonomic groups)"
            )

            # 4. fetch CDS
            seqs = fetch_ntseq([c[0] for c in chosen])
            chosen = [c for c in chosen if c[0] in seqs and len(seqs[c[0]]) >= 100]

            # 5. dedup
            cand_fasta = OUT_DIR / f"{family}.candidates.fasta"
            with open(cand_fasta, "w") as fh:
                for og, ko, label, _dom, _grp in chosen:
                    fh.write(f">{og}|{family}|{ko}|{label}\n{seqs[og]}\n")
            if args.no_dedup:
                keep_ids = {c[0] for c in chosen}
            else:
                print("  MMseqs2 dedup (90% id)...")
                reps = mmseqs_dedup(cand_fasta, OUT_DIR / f"tmp_{family}")
                keep_ids = {rid.split("|")[0] for rid in reps}
            chosen = [c for c in chosen if c[0] in keep_ids][: args.target]
            cand_fasta.unlink(missing_ok=True)

            # 6. write final FASTA + manifest rows
            fam_fasta = OUT_DIR / f"{family}.fasta"
            with open(fam_fasta, "w") as fh:
                for og, ko, label, dom, grp in chosen:
                    organism = og.split(":")[0]
                    fh.write(f">{og}|{family}|{ko}|{label}\n{seqs[og]}\n")
                    writer.writerow(
                        {
                            "family": family,
                            "ko": ko,
                            "ko_label": label,
                            "org_gene": og,
                            "organism": organism,
                            "domain": dom,
                            "group": grp,
                            "cds_len": len(seqs[og]),
                        }
                    )
            mf.flush()
            print(f"  -> {len(chosen)} CDS written to {fam_fasta}")

    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
