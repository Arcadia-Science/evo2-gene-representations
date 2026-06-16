"""Ground-truth baselines for the Evo2 gene-family analysis.

The geodesic distances from embed_and_geodesic_gene_families.py are scored against
ground truths computed directly from the pulled CDS and their source taxa — no
human-specific external DB (unlike the GPN-Star gene pipeline, those baselines do
not exist cross-kingdom). See scripts/evo2/gene_family_design.md §5.

Two baselines back this pipeline (the "sequence identity first" tier of the design):

  1. k-mer sequence divergence  — Axis B workhorse. Alignment-free cosine distance
     between nucleotide k-mer frequency vectors. Defined in the shared
     scripts/sequence_baselines.py and re-exported here so the Evo2 and GPN-Star
     pipelines score against an IDENTICAL k-mer baseline.
  2. taxonomic-rank distance    — Axis B cross-check (defined below). Coarse rank
     distance from the KEGG BRITE (domain, group, organism) annotation already in
     the manifest: does within-family embedding distance track the *species* tree?

The gold-standard per-family tree (align -> FastTree -> patristic) is deferred to a
later "trees + taxonomy" pass per the design build order; this module is imported by
embed_and_geodesic_gene_families.py as a sibling.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# k-mer divergence is shared with the GPN-Star pipeline (one implementation, so the
# two models' k-mer baselines are directly comparable).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sequence_baselines import kmer_distance_matrix  # noqa: E402,F401  (re-exported)

# ── taxonomic-rank distance ──────────────────────────────────────────────────

# Coarse step distance up the (domain -> group -> organism) hierarchy KEGG BRITE
# gives us. Finer than nothing, and enough to ask "does within-family embedding
# distance track host taxonomy?" without an NCBI-taxonomy join.
_TAX_SAME_ORG = 0.0
_TAX_SAME_GROUP = 1.0
_TAX_SAME_DOMAIN = 2.0
_TAX_DIFF_DOMAIN = 3.0


def taxonomic_distance_matrix(meta: pd.DataFrame) -> np.ndarray:
    """(N, N) coarse taxonomic-rank distance from manifest domain/group/organism.

    `meta` rows must align positionally with the embedding rows. Columns used:
    organism, domain, group. Unknown ("Unknown") domains compare as a distinct
    domain, so they never spuriously read as close to anything.
    """
    org = meta["organism"].to_numpy()
    dom = meta["domain"].to_numpy()
    grp = meta["group"].to_numpy()
    n = len(meta)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if org[i] == org[j]:
                d = _TAX_SAME_ORG
            elif dom[i] != dom[j] or dom[i] == "Unknown":
                d = _TAX_DIFF_DOMAIN
            elif grp[i] == grp[j] and grp[i] != "Unknown":
                d = _TAX_SAME_GROUP
            else:
                d = _TAX_SAME_DOMAIN
            D[i, j] = D[j, i] = d
    return D
