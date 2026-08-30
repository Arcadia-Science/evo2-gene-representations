"""Reproducible cofactor annotations pulled from ChEBI (scaffold + active metal)."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path("data/cache/cofactor_annotations.json")

# ── Cofactor token to stable ChEBI accession
# Derive scaffold and metal properties from these identifiers. Protein-contributed ligands are
# not part of the cofactor structure.
COFACTOR_CHEBI: dict[str, str] = {
    "heme_b":          "CHEBI:26355",  # heme b
    "heme_a":          "CHEBI:24479",  # heme a
    "heme_b_thiolate": "CHEBI:26355",  # heme b (thiolate ligand is protein-contributed)
    "heme_substrate":  "CHEBI:26355",  # heme b (as substrate of heme oxygenase)
    "nonheme_diiron":  "CHEBI:47411",  # mu-oxodiiron (oxo-bridged, non-heme, non-Fe/S)
    "nickel_f430":     "CHEBI:28265",  # coenzyme F430
    "femoco":          "CHEBI:30409",  # iron-sulfur-molybdenum cofactor (FeMoco)
    "fe4s4":           "CHEBI:49883",  # tetra-mu3-sulfido-tetrairon ([4Fe-4S] core)
    "iron":            "CHEBI:29033",  # iron(2+)
    "copper":          "CHEBI:29036",  # copper(2+)
    "zinc":            "CHEBI:29105",  # zinc(2+)
    "magnesium":       "CHEBI:18420",  # magnesium(2+)
    "bh4":             "CHEBI:15372",  # 5,6,7,8-tetrahydrobiopterin
    "retinal":         "CHEBI:15035",  # retinal
    # ── Redox and detoxification cofactors
    "fad":             "CHEBI:16238",  # FAD (flavin; NOX/DUOX, FMO, STEAP)
    "nad":             "CHEBI:15846",  # NAD+ (ALDH, alcohol dehydrogenase)
    "nadp":            "CHEBI:18009",  # NADP+ (aldo-keto reductase)
    "glutathione":     "CHEBI:16856",  # glutathione (GST, glutaredoxin)
}

# Metal / metalloid elements to recognise in a molecular formula. Anything here that appears
# in a cofactor's ChEBI formula is counted as one of its catalytic metals.
METAL_ELEMENTS: frozenset[str] = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Mo", "W", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "Pt", "Au", "Hg", "Al", "Ga", "Sn", "Pb",
})

CHEBI_COMPOUND = "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{num}/"
OLS_ANCESTORS = (
    "https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/"
    "{iri}/hierarchicalAncestors?size=500"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (research)", "Accept": "application/json"}


# ── ChEBI fetchers
def _get_json(url: str, *, timeout: int = 20, retries: int = 4) -> dict:
    """GET + parse JSON with a short per-request timeout and bounded retries."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception as err:  # noqa: BLE001 — network/HTTP/JSON, all retryable
            last_err = err
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff
    raise RuntimeError(f"failed after {retries} attempts: {url}") from last_err


def fetch_formula(accession: str) -> str | None:
    """Molecular formula string for a ChEBI accession (None if the entity is an
    abstract class with no specified structure)."""
    num = accession.split(":")[1]
    data = _get_json(CHEBI_COMPOUND.format(num=num))
    chem = data.get("chemical_data")
    if isinstance(chem, dict):
        return chem.get("formula")
    if isinstance(chem, list):  # occasionally a list of {formula/charge/mass}
        for e in chem:
            if isinstance(e, dict) and e.get("formula"):
                return e["formula"]
    return None


def fetch_ancestors(accession: str) -> list[str]:
    """ChEBI is_a hierarchical-ancestor accessions (self included)."""
    num = accession.split(":")[1]
    iri = urllib.parse.quote(
        urllib.parse.quote(f"http://purl.obolibrary.org/obo/CHEBI_{num}", safe=""),
        safe="",
    )
    data = _get_json(OLS_ANCESTORS.format(iri=iri))
    terms = data.get("_embedded", {}).get("terms", [])
    anc = {t["obo_id"] for t in terms if t.get("obo_id")}
    anc.add(accession)  # a node is its own (trivial) ancestor, for Dice self-similarity
    return sorted(anc)


def parse_metals(formula: str | None) -> list[str]:
    """Metal element symbols present in a molecular formula."""
    if not formula:
        return []
    elems = re.findall(r"[A-Z][a-z]?", formula)
    return sorted({e for e in elems if e in METAL_ELEMENTS})


# ── cache build / load
def _write_cache(ann: dict[str, dict]) -> None:
    """Atomically persist the (possibly partial) annotation cache."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(CACHE.suffix + ".tmp")
    tmp.write_text(json.dumps(ann, indent=1))
    tmp.replace(CACHE)  # atomic rename — a killed run never leaves a half-written cache


def build_annotations(existing: dict[str, dict] | None = None) -> dict[str, dict]:
    """Pull scaffold ancestors + metal set for every seed cofactor from ChEBI."""
    out: dict[str, dict] = dict(existing or {})
    for token, accession in COFACTOR_CHEBI.items():
        if token in out and out[token].get("chebi") == accession:
            print(f"  {token:16s} {accession:12s} (cached)")
            continue
        formula = fetch_formula(accession)
        time.sleep(0.3)
        ancestors = fetch_ancestors(accession)
        time.sleep(0.3)
        metals = parse_metals(formula)
        out[token] = {
            "chebi": accession,
            "formula": formula,
            "metals": metals,
            "ancestors": ancestors,
        }
        _write_cache(out)  # persist progress before moving on
        print(f"  {token:16s} {accession:12s} metals={metals or '—'} "
              f"ancestors={len(ancestors)}")
    return out


def load_cofactor_annotations(refresh: bool = False) -> dict[str, dict]:
    """Load cofactor annotations, building + caching from ChEBI on first use."""
    cached: dict[str, dict] = {}
    if not refresh and CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if set(COFACTOR_CHEBI) <= set(cached):
            return cached
    print("Fetching cofactor annotations from ChEBI (OLS4 + compound API)...")
    return build_annotations(existing={} if refresh else cached)


# ── similarity
def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def scaffold_similarity(x: str, y: str, ann: dict[str, dict]) -> float:
    """Sørensen–Dice overlap of the two cofactors' ChEBI ancestor sets (Wu–Palmer-style
    scaffold similarity, depth = ancestor count)."""
    return _dice(set(ann[x]["ancestors"]), set(ann[y]["ancestors"]))


def metal_similarity(x: str, y: str, ann: dict[str, dict]) -> float:
    """Jaccard overlap of the two cofactors' metal-element sets."""
    mx, my = set(ann[x]["metals"]), set(ann[y]["metals"])
    if not mx or not my:  # a metal-free cofactor shares no metal
        return 0.0
    return len(mx & my) / len(mx | my)


def cofactor_similarity(x: str, y: str, ann: dict[str, dict]) -> float:
    """Combined cofactor similarity in [0, 1]: mean of scaffold (Dice) and metal (Jaccard)
    similarity. Identical cofactor tokens score 1.0."""
    if x == y:
        return 1.0
    return 0.5 * scaffold_similarity(x, y, ann) + 0.5 * metal_similarity(x, y, ann)


# ── CLI: (re)build cache, show derived properties + pairwise similarity
def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-pull from ChEBI")
    args = ap.parse_args()

    ann = load_cofactor_annotations(refresh=args.refresh)
    print(f"\nCache: {CACHE}")
    print(f"{'token':16s} {'chebi':12s} {'formula':22s} metals")
    for tok in COFACTOR_CHEBI:
        a = ann[tok]
        print(f"{tok:16s} {a['chebi']:12s} {str(a['formula'] or '—'):22s} "
              f"{a['metals'] or '—'}")

    toks = list(COFACTOR_CHEBI)
    print("\nPairwise cofactor similarity (0.5·scaffold + 0.5·metal):")
    print(" " * 17 + " ".join(f"{t[:8]:>8s}" for t in toks))
    for x in toks:
        row = " ".join(f"{cofactor_similarity(x, y, ann):8.2f}" for y in toks)
        print(f"{x:16s} {row}")


if __name__ == "__main__":
    _main()
