"""
Stage 5a — resolve 1:1 orthologs for the panel genes across the 24-mammal topology, and pull CDS.
"""

from __future__ import annotations
import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from alignment_metrics import read_fasta  # noqa: E402

REST = "https://rest.ensembl.org"
CACHE = ROOT / "data" / "cache" / "strat_homology"
MAMMAL_CDS = Path("/opt/dlami/nvme/strat_seqs/mammals")
PANEL_CDS = {"homo_sapiens": "cds_human.fasta", "ornithorhynchus_anatinus": "cds_platypus.fasta"}
TREE = ROOT / "data" / "mammalian_orthologs" / "tree" / "species_tree.nwk"
HUMAN, PLATYPUS = "homo_sapiens", "ornithorhynchus_anatinus"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def topology_species() -> list[str]:
    return sorted(set(re.findall(r"'([a-z_]+)'", TREE.read_text())))


def homologs(gene: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{gene}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    # The species is REQUIRED in the path: /homology/id/<species>/<id>. Without it REST returns 404
    # for every gene, which silently looks like "this gene has no orthologs".
    url = (
        f"{REST}/homology/id/{HUMAN}/{gene}?type=orthologues;format=condensed;"
        "content-type=application/json"
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as fh:
                d = json.load(fh)
            cf.write_text(json.dumps(d))
            return d
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # genuinely absent from Compara: terminal, do not retry
                log(f"  {gene}: 404 (not in Compara homology)")
                cf.write_text(json.dumps({"data": []}))
                return {"data": []}
            log(f"  {gene} attempt {attempt + 1}: {exc}")
            time.sleep(8)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            log(f"  {gene} attempt {attempt + 1}: {exc}")
            time.sleep(8)
    return {"data": []}


def read_fasta_by_gene(path: Path, wanted: set[str]) -> dict[str, str]:
    """Longest CDS per gene id, from an Ensembl cds.all.fa.gz (headers carry `gene:ENSG...`)."""
    best: dict[str, str] = {}
    gid = None
    buf: list[str] = []

    def flush() -> None:
        if gid and buf:
            s = "".join(buf).upper()
            if gid not in best or len(s) > len(best[gid]):
                best[gid] = s

    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                flush()
                buf = []
                gid = None
                m = re.search(r"gene:(\S+)", line)
                if m:
                    g = m.group(1).split(".")[0]
                    gid = g if g in wanted else None
            elif gid:
                buf.append(line.strip())
    flush()
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run / "stage5")
    (out / "seqs").mkdir(parents=True, exist_ok=True)

    species = topology_species()
    log(f"{len(species)} species in the fixed topology")
    pairs = pd.read_csv(args.run / "stage1" / "pairs.csv")
    genes = list(pairs.gene)

    # ---- 1. orthology ------------------------------------------------------------------------
    rows = []
    for i, g in enumerate(genes):
        d = homologs(g)
        data = d.get("data") or []
        homs = data[0].get("homologies", []) if data else []
        seen: set[str] = set()
        for h in homs:
            sp = h.get("species")
            if sp not in species or h.get("type") != "ortholog_one2one" or sp in seen:
                continue
            seen.add(sp)
            rows.append(
                {"gene": g, "species": sp, "ortholog_gene_id": h.get("id"), "type": h.get("type")}
            )
        if (i + 1) % 25 == 0:
            log(f"  homology {i + 1}/{len(genes)}")
    orth = pd.DataFrame(rows)
    # human is the query, so it is never in its own homology list
    orth = pd.concat(
        [
            orth,
            pd.DataFrame(
                [
                    {
                        "gene": r.gene,
                        "species": HUMAN,
                        "ortholog_gene_id": r.gene_id,
                        "type": "self",
                    }
                    for r in pairs.itertuples()
                ]
            ),
        ],
        ignore_index=True,
    )
    orth.to_csv(out / "ortholog_resolution.csv", index=False)
    log(f"orthologs: {len(orth)} (gene, species) 1:1 records")

    # ---- 2. presence matrix + candidate cores -------------------------------------------------
    pres = orth.pivot_table(index="gene", columns="species", aggfunc="size", fill_value=0).gt(0)
    for sp in species:
        if sp not in pres.columns:
            pres[sp] = False
    pres = pres[species]
    pres.to_csv(out / "presence_matrix.csv")
    cov = pres.mean().sort_values(ascending=False)
    log("\nper-species coverage of the 100 panel genes:\n" + (cov * 100).round(1).to_string())

    cores = []
    ranked = [s for s in cov.index if s not in (HUMAN, PLATYPUS)]
    for k in range(2, len(species) + 1):
        core = [HUMAN, PLATYPUS] + ranked[: k - 2]
        n_complete = int(pres[core].all(axis=1).sum())
        cores.append(
            {
                "n_taxa": len(core),
                "n_genes_complete": n_complete,
                "genes_x_taxa": n_complete * len(core),
                "core": ";".join(sorted(core)),
            }
        )
    cdf = pd.DataFrame(cores)
    cdf.to_csv(out / "core_candidates.csv", index=False)
    log(
        "\ncandidate common cores (human+platypus mandatory, then by coverage):\n"
        + cdf[["n_taxa", "n_genes_complete", "genes_x_taxa"]].to_string(index=False)
    )
    best = cdf.loc[cdf.genes_x_taxa.idxmax()]
    log(
        f"\nmax genes x taxa: {int(best.n_taxa)} taxa x {int(best.n_genes_complete)} genes "
        f"= {int(best.genes_x_taxa)}"
    )

    # ---- 3. CDS per gene ----------------------------------------------------------------------
    panel = {sp: read_fasta(args.run / "stage1" / f) for sp, f in PANEL_CDS.items()}
    wanted_by_sp: dict[str, set[str]] = {}
    for r in orth.itertuples():
        if r.species in PANEL_CDS:
            continue
        wanted_by_sp.setdefault(r.species, set()).add(str(r.ortholog_gene_id))

    seqs: dict[str, dict[str, str]] = {}
    for sp, want in wanted_by_sp.items():
        f = MAMMAL_CDS / f"{sp}.cds.fa.gz"
        if not f.exists():
            log(f"  MISSING {f.name} -- skipping {sp}")
            continue
        seqs[sp] = read_fasta_by_gene(f, want)
        log(f"  {sp}: {len(seqs[sp])}/{len(want)} ortholog CDS")

    n_written = 0
    per_gene = []
    for g in genes:
        recs: list[tuple[str, str]] = []
        for sp in species:
            if sp == HUMAN:
                s = panel[HUMAN].get(g)
            elif sp == PLATYPUS:
                s = panel[PLATYPUS].get(g)
            else:
                oid = orth[(orth.gene == g) & (orth.species == sp)].ortholog_gene_id
                s = seqs.get(sp, {}).get(str(oid.iloc[0])) if len(oid) else None
            if s and len(s) >= 90 and len(s) % 3 == 0:
                recs.append((sp, s))
        if len(recs) >= 4:
            with (out / "seqs" / f"{g}.fasta").open("w") as fh:
                for sp, s in recs:
                    fh.write(f">{sp}\n{s}\n")
            n_written += 1
        per_gene.append(
            {
                "gene": g,
                "n_taxa_cds": len(recs),
                "has_human": any(sp == HUMAN for sp, _ in recs),
                "has_platypus": any(sp == PLATYPUS for sp, _ in recs),
            }
        )
    pg = pd.DataFrame(per_gene)
    pg.to_csv(out / "cds_per_gene.csv", index=False)
    log(
        f"\nwrote {n_written} per-gene CDS fasta; n_taxa min {pg.n_taxa_cds.min()} "
        f"med {pg.n_taxa_cds.median()} max {pg.n_taxa_cds.max()}"
    )
    (out / "stage5a_config.json").write_text(
        json.dumps(
            {
                "ensembl_release": 116,
                "n_species_topology": len(species),
                "transcript_rule": {
                    "human/platypus": "panel canonical CDS (stage 1)",
                    "other": "longest CDS of the 1:1 ortholog gene",
                },
                "n_genes_with_fasta": n_written,
                "best_core": {"n_taxa": int(best.n_taxa), "n_genes": int(best.n_genes_complete)},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
