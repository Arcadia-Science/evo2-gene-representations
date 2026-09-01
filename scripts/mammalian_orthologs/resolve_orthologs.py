"""Resolve high-confidence 1:1 mammalian orthologs for every human gene in the panel via Ensembl
Compara, one ortholog group per human gene (paralogs kept separate).
"""

from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gene_families import family_members, family_order  # noqa: E402

OUT_DIR = ROOT / "data" / "mammalian_orthologs"
CACHE_DIR = ROOT / "data" / "cache" / "mammal_homology"
ENSEMBL = "https://rest.ensembl.org"
SLEEP = 0.12  # polite; Ensembl allows ~15 req/s

# Ensembl species ID to common name and clade for the 24-mammal panel.
SPECIES: dict[str, tuple[str, str]] = {
    "homo_sapiens": ("Human", "primate"),
    "pan_troglodytes": ("Chimpanzee", "primate"),
    "gorilla_gorilla": ("Gorilla", "primate"),
    "pongo_abelii": ("Orangutan", "primate"),
    "macaca_mulatta": ("Rhesus macaque", "primate"),
    "callithrix_jacchus": ("Marmoset", "primate"),
    "mus_musculus": ("Mouse", "glires"),
    "rattus_norvegicus": ("Rat", "glires"),
    "cavia_porcellus": ("Guinea pig", "glires"),
    "oryctolagus_cuniculus": ("Rabbit", "glires"),
    "bos_taurus": ("Cow", "laurasiatheria"),
    "ovis_aries": ("Sheep", "laurasiatheria"),
    "capra_hircus": ("Goat", "laurasiatheria"),
    "sus_scrofa": ("Pig", "laurasiatheria"),
    "equus_caballus": ("Horse", "laurasiatheria"),
    "canis_lupus_familiaris": ("Dog", "laurasiatheria"),
    "ailuropoda_melanoleuca": ("Panda", "laurasiatheria"),
    "felis_catus": ("Cat", "laurasiatheria"),
    "mustela_putorius_furo": ("Ferret", "laurasiatheria"),
    "myotis_lucifugus": ("Microbat", "laurasiatheria"),
    "loxodonta_africana": ("Elephant", "afrotheria"),
    "dasypus_novemcinctus": ("Armadillo", "xenarthra"),
    "monodelphis_domestica": ("Opossum", "marsupial"),
    "ornithorhynchus_anatinus": ("Platypus", "monotreme"),
}


# Compara homology queries use a longer, configurable timeout.
HTTP_TIMEOUT = int(os.environ.get("ENSEMBL_HTTP_TIMEOUT", "180"))


def _get(path: str, tries: int = 5) -> dict | None:
    """GET Ensembl JSON with retry/backoff on 429/5xx. None on definitive 400/404."""
    url = f"{ENSEMBL}{path}"
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            wait = float(e.headers.get("Retry-After", 2**attempt)) if e.code == 429 else 2**attempt
            time.sleep(wait)
        except Exception:
            time.sleep(2**attempt)
    return None


def fetch_homologies(symbol: str, gene_id: str | None = None) -> list[dict]:
    """Cached Compara orthologues for a human gene."""
    key = gene_id or symbol
    cache = CACHE_DIR / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    # target_taxon=40674 (Mammalia) shrinks the server-side homology set ~24s->7s per gene and keeps
    # only mammalian orthologs (our whole panel is mammalian anyway).
    q = (
        "?type=orthologues;format=condensed;content-type=application/json;"
        "compara=vertebrates;target_taxon=40674"
    )
    if gene_id:
        data = _get(f"/homology/id/homo_sapiens/{gene_id}{q}")
    else:
        data = _get(f"/homology/symbol/homo_sapiens/{symbol}{q}")
    # Do not cache failed requests as empty ortholog sets; callers may retry them.
    if data is None:
        return []
    entries = data.get("data") or []
    if gene_id:
        # Exactly one entry is expected; still filter, so a surprise never silently becomes the
        # wrong gene's orthologs again.
        entries = [e for e in entries if e.get("id") == gene_id]
    elif len(entries) > 1:
        # Symbol path with a genuinely ambiguous symbol. There is no correct answer without a gene
        # id, so say so loudly and take the richest entry -- which recovers the HSD17B7 case -- but
        # this is a heuristic, not a fix. The fix is to pass gene_id.
        print(
            f"  AMBIGUOUS SYMBOL {symbol}: maps to {len(entries)} Ensembl genes "
            f"{[e.get('id') for e in entries]}; taking the one with the most homologies. "
            f"Pass gene_id to disambiguate."
        )
        entries = sorted(entries, key=lambda e: len(e.get("homologies") or []), reverse=True)
    homs = (entries[0].get("homologies") or []) if entries else []
    # Leave empty responses uncached because they may represent transient server failures.
    if not homs:
        return []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(homs))
    time.sleep(SLEEP)
    return homs


def resolve() -> None:
    gene_families = family_members("human")
    order = family_order("human")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parallel prefetch (fetch is 100% network/server-latency-bound; parsing is trivial). Cached, so
    # this is idempotent and resumable. Bounded pool stays within Ensembl fair-use.
    from concurrent.futures import ThreadPoolExecutor

    all_genes = [g for fam in order for g in gene_families[fam]]
    todo = [g for g in all_genes if not (CACHE_DIR / f"{g}.json").exists()]
    print(f"Prefetching {len(todo)}/{len(all_genes)} uncached homology queries (pool=6)...")
    done = [0]

    def _pf(g):
        fetch_homologies(g)
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"  prefetched {done[0]}/{len(todo)}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(_pf, todo))
    print("Prefetch complete; parsing.")

    rows = []
    species_set = set(SPECIES)
    for fam in order:
        genes = gene_families[fam]
        print(f"\n=== {fam}  ({len(genes)} human genes) ===")
        for i, g in enumerate(genes):
            homs = fetch_homologies(g)
            # One2one per target species (condensed schema: species/id/type at top level).
            per_sp: dict[str, str] = {}
            for h in homs:
                sp = h.get("species")
                if sp in species_set and h.get("type") == "ortholog_one2one" and sp not in per_sp:
                    per_sp[sp] = h.get("id")
            for sp, gid in per_sp.items():
                rows.append(
                    {
                        "family": fam,
                        "human_gene": g,
                        "species": sp,
                        "clade": SPECIES[sp][1],
                        "common_name": SPECIES[sp][0],
                        "ortholog_gene_id": gid,
                        "type": "ortholog_one2one",
                    }
                )
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(genes)} genes resolved")
    # write
    import csv

    res = OUT_DIR / "ortholog_resolution.csv"
    with open(res, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "human_gene",
                "species",
                "clade",
                "common_name",
                "ortholog_gene_id",
                "type",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {res}  ({len(rows)} one2one ortholog loci, human query gene excluded)")

    # per-family sizes. n_loci counts non-human orthologs; +1 human locus per group is added when
    # loci are actually extracted (the human gene is the query, always present).
    import pandas as pd

    df = pd.DataFrame(rows)
    sizes = []
    for fam in order:
        n_genes = len(gene_families[fam])
        sub = df[df["family"] == fam]
        groups = sub["human_gene"].nunique()
        n_nonhuman = len(sub)
        n_total = n_nonhuman + n_genes  # + human locus per human gene
        sizes.append(
            {
                "family": fam,
                "n_human_genes": n_genes,
                "groups_with_orthologs": groups,
                "loci_nonhuman": n_nonhuman,
                "loci_total_incl_human": n_total,
                "mean_species_per_group": round(n_total / n_genes, 1) if n_genes else 0,
            }
        )
    sz = pd.DataFrame(sizes)
    sz.to_csv(OUT_DIR / "family_sizes.csv", index=False)
    print(f"Wrote {OUT_DIR / 'family_sizes.csv'}")
    print("\n" + sz.to_string(index=False))
    tot = int(sz["loci_total_incl_human"].sum())
    print(f"\nTOTAL loci (incl. human, one2one, pre-locus-QC): {tot}")
    print(
        "Note: still to be trimmed by locus QC (complete start/stop, N-content, length, "
        "syntenic-neighbor); %id + synteny added at extraction."
    )


if __name__ == "__main__":
    resolve()
