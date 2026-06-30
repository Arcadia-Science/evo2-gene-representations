"""Gene-family definitions shared across the gpnstar pipeline.

Single source of truth for the human gene families analysed by the GPN-Star
scripts. Imported by ``embed_and_geodesic_genes.py`` (embedding + ground-truth
matrices), ``layer_sweep.py``, and the figure scripts (colors). Keeping these
here prevents the lists from drifting between files.

The member lists (``GENE_FAMILIES``) are generated from curated **HGNC gene
groups** by ``build_families.py`` and stored in ``families_data.json`` — every
family is orthology-aware rather than a raw domain hit. Re-run that script to
refresh membership. The metadata dicts below (order, Pfam, colors) are
hand-maintained and must stay in sync with the family keys.
"""

import json
from pathlib import Path

# ── Generated membership (HGNC gene groups; see build_families.py) ──────────────
_DATA = json.loads((Path(__file__).resolve().parent / "families_data.json").read_text())

# Family -> member gene symbols.
GENE_FAMILIES: dict[str, list[str]] = _DATA["gene_families"]

# Canonical family ordering for all matrices/figures.
FAMILY_ORDER: list[str] = _DATA["family_order"]

# Representative Pfam accession per family (for the JSD ground-truth matrix).
# opsins and olfactory_receptors are both class-A GPCRs (7tm clan) — their JSD
# separation is intrinsically small; the within-family metric is the primary one.
PFAM_ACCESSIONS: dict[str, str] = {
    "globins": "PF00042",  # Globin
    "opsins": "PF00001",  # 7tm_1 (rhodopsin-like GPCR)
    "olfactory_receptors": "PF13853",  # 7tm_4 (olfactory receptor)
    "cytochrome_p450": "PF00067",  # p450
    "hox": "PF00046",  # Homeodomain
    "ras_gtpases": "PF00071",  # Ras
}

# Per-family plot colors (used by the figure scripts).
FAMILY_COLORS: dict[str, str] = {
    "globins": "#E63946",
    "opsins": "#F4A261",
    "olfactory_receptors": "#2A9D8F",
    "cytochrome_p450": "#457B9D",
    "hox": "#6A4C93",
    "ras_gtpases": "#E9C46A",
}

# Fail loudly if the metadata dicts drift from the generated membership.
assert set(FAMILY_ORDER) == set(GENE_FAMILIES) == set(PFAM_ACCESSIONS) == set(FAMILY_COLORS), (
    "families.py metadata is out of sync with families_data.json — "
    "update PFAM_ACCESSIONS / FAMILY_COLORS / FAMILY_ORDER after re-running build_families.py"
)
