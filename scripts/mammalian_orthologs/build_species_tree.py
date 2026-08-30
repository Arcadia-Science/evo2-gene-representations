"""Build the mammalian species-tree patristic reference from VertLife posterior trees."""

from __future__ import annotations

from pathlib import Path

import dendropy
import numpy as np
import pandas as pd

TREE_DIR = Path("data/mammalian_orthologs/tree")
NEX = TREE_DIR / "vertlife_pruned" / "output.nex"

# VertLife binomial (spaces) -> our Ensembl species id
V2E = {
    "Homo sapiens": "homo_sapiens", "Pan troglodytes": "pan_troglodytes",
    "Gorilla gorilla": "gorilla_gorilla", "Pongo abelii": "pongo_abelii",
    "Macaca mulatta": "macaca_mulatta", "Callithrix jacchus": "callithrix_jacchus",
    "Mus musculus": "mus_musculus", "Rattus norvegicus": "rattus_norvegicus",
    "Cavia porcellus": "cavia_porcellus", "Oryctolagus cuniculus": "oryctolagus_cuniculus",
    "Bos taurus": "bos_taurus", "Ovis aries": "ovis_aries", "Capra hircus": "capra_hircus",
    "Sus scrofa": "sus_scrofa", "Equus caballus": "equus_caballus",
    "Canis lupus": "canis_lupus_familiaris", "Ailuropoda melanoleuca": "ailuropoda_melanoleuca",
    "Felis catus": "felis_catus", "Mustela putorius": "mustela_putorius_furo",
    "Myotis lucifugus": "myotis_lucifugus", "Loxodonta africana": "loxodonta_africana",
    "Dasypus novemcinctus": "dasypus_novemcinctus", "Monodelphis domestica": "monodelphis_domestica",
    "Ornithorhynchus anatinus": "ornithorhynchus_anatinus",
}


def main() -> None:
    trees = dendropy.TreeList.get(path=str(NEX), schema="nexus")
    species = sorted(V2E.values())
    idx = {s: i for i, s in enumerate(species)}
    n = len(species)
    stack = np.zeros((len(trees), n, n))
    for t, tree in enumerate(trees):
        pdm = tree.phylogenetic_distance_matrix()
        taxa = {tx.label: tx for tx in tree.taxon_namespace}
        for a in species_present(V2E, taxa):
            for b in species_present(V2E, taxa):
                if a is b:
                    continue
                ea, eb = V2E[a.label], V2E[b.label]
                stack[t, idx[ea], idx[eb]] = pdm.patristic_distance(a, b)
    mean = stack.mean(axis=0)
    sd = stack.std(axis=0)
    pd.DataFrame(mean, index=species, columns=species).to_csv(TREE_DIR / "species_patristic.csv")
    pd.DataFrame(sd, index=species, columns=species).to_csv(TREE_DIR / "species_patristic_sd.csv")
    # representative tree, relabeled to Ensembl ids
    rep = trees[0]
    for tx in rep.taxon_namespace:
        if tx.label in V2E:
            tx.label = V2E[tx.label]
    rep.write(path=str(TREE_DIR / "species_tree.nwk"), schema="newick")

    print(f"built patristic from {len(trees)} posterior trees, {n} species")
    print("\nsanity — closest & farthest pairs (My):")
    iu = np.triu_indices(n, 1)
    flat = [(mean[i, j], species[i], species[j]) for i, j in zip(*iu)]
    flat.sort()
    for d, a, b in flat[:3]:
        print(f"  closest  {a:24}{b:24} {d:.1f} My")
    for d, a, b in flat[-3:]:
        print(f"  farthest {a:24}{b:24} {d:.1f} My")
    print(f"\nmax posterior SD on any pair: {sd.max():.2f} My "
          f"(mean SD {sd[iu].mean():.2f})  -> {TREE_DIR}/species_patristic.csv")


def species_present(v2e, taxa):
    return [taxa[label] for label in v2e if label in taxa]


if __name__ == "__main__":
    main()
