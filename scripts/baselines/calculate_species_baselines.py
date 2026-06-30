"""Ground-truth baseline for the Evo2 species analysis: GTDB patristic distances.

The geodesic distances from embed_and_geodesic_species.py are compared against the
phylogenetic (patristic) distances between the same species in the GTDB bac120 tree.
This module downloads that tree and computes the pairwise patristic distance matrix;
embed_and_geodesic_species.py imports it (sibling module in scripts/evo2/).
"""

import urllib.request
from pathlib import Path

import numpy as np

# GTDB r226.0 — matches the Goodfire tree-of-life reproduction
# (tree_of_life_reproduction.zip), which builds its patristic ground truth from
# bac120_r226.tree. (Earlier this pipeline pinned r220 to match Evo 2's stated
# training release; we follow Goodfire's r226 choice instead. See download_species_manifest.py.)
TREE_URL = "https://data.gtdb.ecogenomic.org/releases/release226/226.0/bac120_r226.tree"


def download_tree(tree_path: Path) -> None:
    """Download bac120.tree if not already present."""
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading GTDB tree from {TREE_URL} ...")
    urllib.request.urlretrieve(TREE_URL, str(tree_path))
    print(f"  Saved to {tree_path}  ({tree_path.stat().st_size / 1e6:.1f} MB)")


def compute_patristic_distances(
    tree_path: Path,
    gtdb_accessions: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise patristic distances for the given GTDB accessions.

    Returns (D, present_mask) where present_mask[i] is True iff gtdb_accessions[i]
    is a leaf in the tree. Distances for absent taxa are left as 0 and the caller
    should restrict any downstream comparison to present taxa via present_mask.

    The GTDB bac120 tree has ~100K leaves. dendropy's phylogenetic_distance_matrix()
    materializes ALL pairwise distances among every leaf (~10^10 pairs for the full
    tree), which exhausts memory. So we first prune the tree down to just our target
    taxa, then compute the (small) distance matrix on the pruned tree. Tree leaf
    names match gtdb_accession (e.g. "GB_GCA_000001405.15").
    """
    import dendropy

    print("  Parsing GTDB tree with dendropy...")
    # preserve_underscores=True: Newick treats unquoted underscores as spaces by
    # default, which would turn "RS_GCF_..." leaf labels into "RS GCF ..." and break
    # matching against our gtdb_accession values.
    tree = dendropy.Tree.get(path=str(tree_path), schema="newick", preserve_underscores=True)

    target_set = set(gtdb_accessions)
    all_labels = {t.label for t in tree.taxon_namespace}
    present = sorted(target_set & all_labels)
    missing = target_set - all_labels
    if missing:
        print(
            f"  WARNING: {len(missing)}/{len(target_set)} accessions not found in "
            f"tree (their patristic distances left as 0): "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )

    print(f"  Pruning tree to {len(present)} target taxa...")
    tree.retain_taxa_with_labels(present)

    print("  Computing patristic distance matrix on pruned tree...")
    pdm = tree.phylogenetic_distance_matrix()

    # taxon_namespace still references the original taxa; map labels -> taxon objects
    # that survive on the pruned tree.
    taxon_map = {t.label: t for t in tree.taxon_namespace if t.label in target_set}

    N = len(gtdb_accessions)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        t_i = taxon_map.get(gtdb_accessions[i])
        if t_i is None:
            continue
        for j in range(i + 1, N):
            t_j = taxon_map.get(gtdb_accessions[j])
            if t_j is None:
                continue
            d = pdm.patristic_distance(t_i, t_j)
            D[i, j] = D[j, i] = max(0.0, d)

    present_mask = np.array([a in taxon_map for a in gtdb_accessions], dtype=bool)
    return D, present_mask
