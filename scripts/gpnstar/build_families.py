"""Build the GPN-Star gene-family definitions from HGNC gene groups.

Rationale (see scripts/gpnstar/README.md):
  The earlier family set mixed orthology-aware families (globins, HOX) with
  groups defined only by a shared domain. To make the latent-space question
  truly about *gene-family* learning rather than domain recognition, every
  family here is pulled from a curated **HGNC gene group** (orthology-aware,
  not a raw Pfam/domain hit). Membership is therefore reproducible and complete
  rather than a hand-picked subset.

Model scope — IMPORTANT:
  GPN-Star is genome-anchored: it embeds an hg38 locus + its multiz-100way MSA
  column stack, so it can only see **human nuclear loci**. The per-family N is
  bounded by the human paralog count, not by how many sequences exist across
  species. The cross-kingdom families the advisor proposed (bacterial→vertebrate
  globins, heme-copper oxidases) belong to the **Evo2** stage, which is sequence-
  native. Two families that the advisor listed therefore degrade in GPN-Star:
    * globins — HGNC only groups the hemoglobin subunits (group 940); myoglobin
      (MB), cytoglobin (CYGB) and neuroglobin (NGB) carry no HGNC group, so we
      add them explicitly to recover the true globin gene family.
    * heme-copper oxidase — HGNC group 643 is the complex-IV *assembly* (non-
      homologous subunits), and the catalytic COX1-like subunit is mitochondrial
      (MT-CO1, absent from the nuclear multiz). It is not a valid within-family
      set in human, so it is omitted here and deferred to Evo2.

Output: ``families_data.json`` (gene lists), the single source of truth loaded
by ``families.py``. Re-run after changing FAMILY_SPEC:

    uv run python scripts/gpnstar/build_families.py
"""

import json
import random
import urllib.request
from pathlib import Path

HGNC_FETCH = "https://rest.genenames.org/fetch/gene_group_id/{gid}"
OUT_PATH = Path(__file__).resolve().parent / "families_data.json"

# Subsample seed for families capped by ``max_members`` (only ORs today).
SUBSAMPLE_SEED = 0

# family -> spec. ``group_ids`` are HGNC gene-group IDs (resolved against the
# live HGNC REST API). ``extra_symbols`` are appended verbatim (for genes HGNC
# leaves ungrouped). ``symbol_prefix`` restricts a broad group to one cluster.
# ``max_members`` triggers a seeded subsample (keeps the matrices balanced).
FAMILY_SPEC: dict[str, dict] = {
    "globins": {
        "group_ids": [940],  # Hemoglobin subunits
        "extra_symbols": ["MB", "CYGB", "NGB"],  # ungrouped in HGNC
    },
    "opsins": {
        "group_ids": [215],  # Opsin receptors
    },
    "olfactory_receptors": {
        # Olfactory receptor families 1,2,4,5,6,7,8,9,10,11,12,13,14,51,52,56.
        "group_ids": [
            147,
            149,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            159,
            160,
            162,
            163,
            164,
            165,
            167,
        ],
        "max_members": 40,  # ~416 protein-coding -> subsample
    },
    "cytochrome_p450": {
        # CYP families 1,2,3,4,7,8,11,17,19,20,21,24,26,27,39,46,51.
        "group_ids": [
            1000,
            1001,
            1002,
            1003,
            1005,
            1006,
            1007,
            1008,
            1009,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1016,
            1017,
        ],
    },
    # ── well-separated positive controls (kept from the original set) ──
    "hox": {
        "group_ids": [518],  # HOXL subclass homeoboxes
        "symbol_prefix": "HOX",  # -> the 39 canonical HOX genes
    },
    "ras_gtpases": {
        "group_ids": [389],  # RAS type GTPase family
    },
}

# Output ordering for all downstream matrices/figures.
FAMILY_ORDER = [
    "globins",
    "opsins",
    "olfactory_receptors",
    "cytochrome_p450",
    "hox",
    "ras_gtpases",
]


def fetch_group_members(gid: int) -> list[dict]:
    req = urllib.request.Request(HGNC_FETCH.format(gid=gid), headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["response"]["docs"]


def build_family(name: str, spec: dict) -> list[str]:
    symbols: set[str] = set()
    for gid in spec["group_ids"]:
        for doc in fetch_group_members(gid):
            if doc.get("status") != "Approved":
                continue
            if doc.get("locus_group") != "protein-coding gene":
                continue  # drop pseudogenes / RNA genes
            sym = doc.get("symbol", "")
            if not sym or sym.startswith("MT-"):
                continue  # mitochondrial loci are absent from the nuclear multiz
            if "symbol_prefix" in spec and not sym.startswith(spec["symbol_prefix"]):
                continue
            symbols.add(sym)

    symbols.update(spec.get("extra_symbols", []))
    members = sorted(symbols)

    max_n = spec.get("max_members")
    if max_n and len(members) > max_n:
        rng = random.Random(SUBSAMPLE_SEED)
        members = sorted(rng.sample(members, max_n))
        print(f"  {name}: subsampled {max_n} of {len(symbols)} (seed={SUBSAMPLE_SEED})")
    else:
        print(f"  {name}: {len(members)} members")
    return members


def main() -> None:
    print("Building gene families from HGNC gene groups...")
    gene_families = {name: build_family(name, FAMILY_SPEC[name]) for name in FAMILY_ORDER}

    total = sum(len(v) for v in gene_families.values())
    data = {
        "source": "HGNC gene groups (rest.genenames.org)",
        "subsample_seed": SUBSAMPLE_SEED,
        "family_order": FAMILY_ORDER,
        "family_spec": FAMILY_SPEC,
        "gene_families": gene_families,
    }
    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH}  ({total} genes across {len(FAMILY_ORDER)} families)")


if __name__ == "__main__":
    main()
