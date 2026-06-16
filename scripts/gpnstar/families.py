"""Gene-family definitions shared across the gpnstar pipeline.

Single source of truth for the 10 human gene families analysed by the GPN-Star
scripts. Imported by ``embed_and_geodesic_gpnstar.py`` (embedding + ground-truth
matrices), ``download_sequences.py`` (NCBI CDS download), and the figure scripts
(colors). Keeping these here prevents the lists from drifting between files.
"""

# Family -> member gene symbols.
GENE_FAMILIES: dict[str, list[str]] = {
    "globins": ["HBB", "HBA1", "MB", "NGB"],
    "hox": ["HOXA1", "HOXA2", "HOXB1", "HOXB2", "HOXC4", "HOXD4", "HOXD10"],
    "ras_gtpases": ["KRAS", "HRAS", "NRAS", "RRAS"],
    "cytochrome_p450": ["CYP1A1", "CYP1A2", "CYP2D6", "CYP3A4", "CYP3A5"],
    "c2h2_zinc_fingers": ["SP1", "SP3", "KLF4", "KLF2", "WT1"],
    "aquaporins": ["AQP1", "AQP2", "AQP3", "AQP4", "AQP5"],
    "sirtuins": ["SIRT1", "SIRT2", "SIRT3", "SIRT4", "SIRT5", "SIRT6", "SIRT7"],
    "toll_like_receptors": ["TLR1", "TLR2", "TLR3", "TLR4", "TLR5", "TLR7", "TLR9"],
    "wnt_ligands": ["WNT1", "WNT2", "WNT3", "WNT4", "WNT5A", "WNT7A", "WNT10B"],
    "kinesins": ["KIF1A", "KIF1B", "KIF2A", "KIF5B", "KIF5C", "KIF11"],
}

# Canonical family ordering for all matrices/figures.
FAMILY_ORDER: list[str] = [
    "globins",
    "hox",
    "ras_gtpases",
    "cytochrome_p450",
    "c2h2_zinc_fingers",
    "aquaporins",
    "sirtuins",
    "toll_like_receptors",
    "wnt_ligands",
    "kinesins",
]

# Representative Pfam accession per family (for the JSD ground-truth matrix).
PFAM_ACCESSIONS: dict[str, str] = {
    "globins": "PF00042",
    "hox": "PF00046",
    "ras_gtpases": "PF00071",
    "cytochrome_p450": "PF00067",
    "c2h2_zinc_fingers": "PF00096",
    "aquaporins": "PF00230",
    "sirtuins": "PF02146",
    "toll_like_receptors": "PF01582",
    "wnt_ligands": "PF00110",
    "kinesins": "PF00225",
}

# Per-family plot colors (used by the figure scripts).
FAMILY_COLORS: dict[str, str] = {
    "globins":             "#E63946",
    "hox":                 "#F4A261",
    "ras_gtpases":         "#2A9D8F",
    "cytochrome_p450":     "#457B9D",
    "c2h2_zinc_fingers":   "#A8DADC",
    "aquaporins":          "#6A4C93",
    "sirtuins":            "#52B788",
    "toll_like_receptors": "#8B8B00",
    "wnt_ligands":         "#E9C46A",
    "kinesins":            "#264653",
}
