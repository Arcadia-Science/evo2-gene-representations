"""Taxonomic-rank distance baseline for the Evo2 cross-kingdom gene-family analysis.

Axis-B cross-check for the cross-kingdom KEGG ortholog panel: does within-family
embedding distance track the *species* tree? Coarse step distance up the
(domain -> group -> organism) hierarchy that the KEGG BRITE annotation in the
manifest provides. This is the one genuinely cross-kingdom-only ground truth — the
human matched panel has no taxonomy axis — so it lives apart from the model-agnostic
sequence baselines (k-mer / Pfam-JSD / seq-identity / patristic) in this directory.

Imported by scripts/evo2/embed_and_geodesic_ortholog.py. The shared k-mer baseline
it is scored alongside lives in scripts/baselines/kmer_sequence_divergence.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
