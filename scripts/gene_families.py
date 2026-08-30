"""Single source of truth for gene-family SELECTION, shared by both models."""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES_DATA_JSON = ROOT / "scripts" / "families_data.json"  # human membership artifact
EVO2_DATA_DIR = ROOT / "data" / "evo2_gene_families"  # evo2 FASTA + manifest live here

# ── HTTP / build constants ──────────────────────────────────────────────────────
HGNC_FETCH = "https://rest.genenames.org/fetch/gene_group_id/{gid}"
KEGG = "https://rest.kegg.jp"
SLEEP = 0.34  # KEGG fair-use throttle
NTSEQ_BATCH = 10  # KEGG `get` accepts up to 10 entries per call
BRITE_CACHE = ROOT / "data" / "cache" / "kegg_br08610.txt"
TARGET_PER_FAMILY = 400
SUBSAMPLE_SEED = 0
OVERSAMPLE = 1.6  # fetch this × target as candidates so dedup still leaves ~target


# Definitions
# Human paralog panel: HGNC gene-group specification
# group_ids are HGNC gene-group IDs (resolved against the live HGNC REST API). extra_symbols are
# appended verbatim (for genes HGNC leaves ungrouped). symbol_prefix restricts a broad group to one
# cluster. max_members triggers a seeded subsample (none set today; machinery kept generic).
HGNC_FAMILY_SPEC: dict[str, dict] = {
    "globins": {
        "group_ids": [940],  # Hemoglobin subunits
        "extra_symbols": ["MB", "CYGB", "NGB"],  # ungrouped in HGNC
    },
    "opsins": {"group_ids": [215]},  # Opsin receptors
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
    # positive controls
    "hox": {"group_ids": [518], "symbol_prefix": "HOX"},
    "ras_gtpases": {"group_ids": [389]},
    # gas-sensing paralog families
    "carbonic_anhydrase": {"group_ids": [460]},
    "nitric_oxide_synthase": {"group_ids": [3878]},
    "heme_oxygenase": {"group_ids": [3873]},
    # redox-homeostasis paralog families
    "peroxiredoxin": {"group_ids": [953]},
    "glutathione_peroxidase": {"group_ids": [3475]},
    # Glutaredoxin domain-containing group; restrict to GLRX/GLRX2/GLRX3/GLRX5
    "glutaredoxin": {"group_ids": [1468], "symbol_prefix": "GLRX"},
    "peroxidase": {"group_ids": [3476]},  # Heme peroxidase family (EPX/LPO/MPO/PXDN/PXDNL/TPO)
    # redox and detoxification metabolism
    # Enzyme oxidoreductases + phase-I/II detox conjugation.
    "glutathione_s_transferase": {"group_ids": [567]},  # Soluble GSTs (GSTA/M/P/T/O/Z/K + HPGDS)
    "aldehyde_dehydrogenase": {"group_ids": [398]},  # ALDH superfamily (NAD(P)+ oxidoreductase)
    # AKR group; restrict to the true aldo-keto.
    "aldo_keto_reductase": {"group_ids": [399], "symbol_prefix": "AKR"},
    "sulfotransferase": {"group_ids": [762]},  # Cytosolic sulfotransferases (SULT; phase-II)
    "udp_glucuronosyltransferase": {"group_ids": [363]},  # UGT (phase-II glucuronidation)
    "nadph_oxidase": {"group_ids": [3535]},  # NOX/DUOX (heme+FAD, O2 -> superoxide/H2O2)
    "arachidonate_lipoxygenase": {"group_ids": [407]},  # ALOX (non-heme Fe + O2)
    "flavin_monooxygenase": {"group_ids": [1667]},  # FMO1-5 (FAD monooxygenase)
    "steap_metalloreductase": {"group_ids": [1324]},  # STEAP (metalloreductase, Fe/Cu)
    # zinc and metal metalloenzymes
    "matrix_metalloproteinase": {"group_ids": [891]},  # M10 matrixins (Zn endopeptidase)
    "adam_metallopeptidase": {"group_ids": [47]},  # ADAM (M12B reprolysin Zn protease)
    "adamts_metallopeptidase": {"group_ids": [50]},  # ADAMTS (M12B + thrombospondin repeats)
    "m14_carboxypeptidase": {"group_ids": [1321]},  # M14 Zn carboxypeptidases
    "alcohol_dehydrogenase": {"group_ids": [397]},  # Zn-binding ADH (NAD oxidoreductase)
    "metallothionein": {"group_ids": [638]},  # Zn/Cu-binding metallothioneins
    "alkaline_phosphatase": {"group_ids": [1072]},  # Zn/Mg alkaline phosphatases
    "histone_deacetylase_classI": {"group_ids": [989]},  # Zn-dependent class-I HDACs
    "ectonucleotide_pyrophosphatase": {"group_ids": [1821]},  # ENPP (Zn nucleotide PDE)
    "phosphodiesterase": {"group_ids": [681]},  # cyclic-nucleotide PDEs (bimetal Zn/Mn)
    # Fe / Fe-S
    "ferritin": {"group_ids": [1455]},  # ferritin chains (Fe storage, ferroxidase)
    # P-loop GTPase superfamily (nested homology vs ras_gtpases) ──
    "rab_gtpase": {"group_ids": [388]},  # RAB small GTPases (Ras superfamily)
    "arf_gtpase": {"group_ids": [357]},  # ARF/ARL small GTPases
    "rho_gtpase": {"group_ids": [390]},  # RHO family GTPases
    "guanylate_binding_protein": {"group_ids": [1825]},  # GBP large IFN-inducible GTPases
    # GPCR / receptor superfamily (within-7TM structure vs opsins/ORs) ──
    "taste2_receptor": {"group_ids": [1162]},  # TAS2R bitter-taste (class-A-like)
    "serotonin_receptor": {"group_ids": [170]},  # 5-HT GPCRs (HTR1/2/4/5/6/7; class A)
    "adrenoceptor": {"group_ids": [169]},  # adrenergic receptors (class A)
    "glutamate_metabotropic": {"group_ids": [281]},  # mGluR (class-C GPCR outgroup)
    "dopamine_receptor": {"group_ids": [181]},  # dopamine receptors (class A)
    "muscarinic_receptor": {"group_ids": [180]},  # muscarinic ACh receptors (class A)
    "histamine_receptor": {"group_ids": [187]},  # histamine receptors (class A)
    "p2y_receptor": {"group_ids": [213]},  # P2Y purinergic receptors (class A)
    "cxc_chemokine_receptor": {"group_ids": [1094]},  # CXC chemokine receptors (class A)
    # mechanism contrast + distant controls ──
    "serine_protease": {"group_ids": [738]},  # trypsin/chymotrypsin-like (proteolysis, non-metal)
    "histone_h4": {"group_ids": [1939]},  # H4 histones (near-identical paralogs; DNA packaging)
}

HUMAN_FAMILY_ORDER: list[str] = [
    "globins",
    "opsins",
    "olfactory_receptors",
    "cytochrome_p450",
    "hox",
    "ras_gtpases",
    "carbonic_anhydrase",
    "nitric_oxide_synthase",
    "heme_oxygenase",
    "peroxiredoxin",
    "glutathione_peroxidase",
    "glutaredoxin",
    "peroxidase",
    "glutathione_s_transferase",
    "aldehyde_dehydrogenase",
    "aldo_keto_reductase",
    "sulfotransferase",
    "udp_glucuronosyltransferase",
    "nadph_oxidase",
    "arachidonate_lipoxygenase",
    "flavin_monooxygenase",
    "steap_metalloreductase",
    "matrix_metalloproteinase",
    "adam_metallopeptidase",
    "adamts_metallopeptidase",
    "m14_carboxypeptidase",
    "alcohol_dehydrogenase",
    "metallothionein",
    "alkaline_phosphatase",
    "histone_deacetylase_classI",
    "ectonucleotide_pyrophosphatase",
    "phosphodiesterase",
    "ferritin",
    "rab_gtpase",
    "arf_gtpase",
    "rho_gtpase",
    "guanylate_binding_protein",
    "taste2_receptor",
    "serotonin_receptor",
    "adrenoceptor",
    "glutamate_metabotropic",
    "dopamine_receptor",
    "muscarinic_receptor",
    "histamine_receptor",
    "p2y_receptor",
    "cxc_chemokine_receptor",
    "serine_protease",
    "histone_h4",
]

# ── Evo2 panel (cross-kingdom): KEGG Orthology spec ──────────────────────────────
# family -> {KO: subtype-label}. The label is a finer annotation carried into the manifest
# (e.g. type1/type2 for the opsin convergence control). Edit/extend to change membership.
FAMILY_KOS: dict[str, dict[str, str]] = {
    # Flagship: bacterial glbN → vertebrate Hb/Mb/Ngb/Cygb. Spans all of life.
    "globins": {
        "K06886": "bacterial_glbN",
        "K13822": "HBA",
        "K13823": "HBB",
        "K13824": "HBG",
        "K13825": "HBE",
        "K21892": "MB",
        "K21893": "NGB",
        "K21894": "CYGB",
    },
    # Heme-copper oxidase superfamily, catalytic SUBUNIT I only (homologous across aa3 / cbb3 /
    # cytochrome-o / archaeal aa3). NOT cytochrome bd (different fold).
    "heme_copper_oxidase": {
        "K02256": "aa3_COX1",
        "K02274": "aa3_coxA",
        "K00404": "cbb3_ccoN",
        "K02298": "cyo_cyoB",
        "K24009": "aa3_soxB",
        "K24011": "aa3_soxM",
    },
    # Cytochrome P450 superfamily. Representative cross-kingdom CYP KO set.
    "cytochrome_p450": {
        "K00490": "CYP4F",
        "K07408": "CYP1A1",
        "K07409": "CYP1A2",
        "K07410": "CYP1B1",
        "K07411": "CYP2A",
        "K07413": "CYP2C",
        "K07415": "CYP2E1",
        "K07424": "CYP3A",
        "K00517": "CYP_other",
    },
    # Olfactory receptors — one lumped KO (~198k genes); heavily subsampled. Vertebrate.
    "olfactory_receptors": {"K04257": "OLFR"},
    # Opsins as a CONVERGENCE CONTROL: animal type-2 (homologous) + microbial type-1 (convergent).
    "opsins": {
        "K04250": "type2",
        "K04251": "type2",
        "K04252": "type2",
        "K04253": "type2",
        "K04254": "type2",
        "K04255": "type2",
        "K04256": "type2",
        "K04641": "type1",
        "K04642": "type1",
        "K04643": "type1",
    },
    # Control: small-GTPase RAS family (eukaryotic, tight, well understood).
    "ras_gtpases": {"K07827": "KRAS", "K02833": "HRAS"},
    # ── Gas sensing and metabolism─
    # Convergent O2 carrier: hemerythrin binds O2 with a non-heme di-iron centre — a DIFFERENT fold
    # from the globin heme pocket (functional-but-not-sequence analog).
    "hemerythrin": {"K07216": "hr"},
    # CO2 hydratases — three independently evolved, sequence-unrelated classes (same EC 4.2.1.1).
    "carbonic_anhydrase_alpha": {"K01672": "CA", "K18245": "CA2", "K18246": "CA4"},
    "carbonic_anhydrase_beta": {"K01673": "cynT_can", "K01674": "cah"},
    "carbonic_anhydrase_gamma": {"K01726": "gammaCA", "K01743": "cam"},
    # N2 fixation — nitrogenase iron protein, a single conserved marker (bacteria+archaea).
    "nitrogenase": {"K02588": "nifH"},
    # Anaerobic CH4 metabolism — methyl-coenzyme M reductase alpha (archaeal marker).
    "methyl_coenzyme_m_reductase": {"K00399": "mcrA"},
    # NO production — shared oxygenase domain across bacterial nos and animal NOS1/2/3.
    "nitric_oxide_synthase": {
        "K00491": "bacterial_nos",
        "K13240": "NOS1",
        "K13241": "NOS2",
        "K13242": "NOS3",
    },
    # CO production — canonical heme-oxygenase fold (animal HMOX1/2 + bacterial/plant HOs).
    "heme_oxygenase": {
        "K00510": "HMOX1",
        "K21418": "HMOX2",
        "K21480": "HO_ferredoxin",
        "K07215": "pigA_hemO",
    },
    # CH4 oxidation convergence pair: soluble di-iron mmoX vs particulate copper pmoA.
    "methane_monooxygenase": {"K16157": "sMMO_mmoX", "K10944": "pMMO_pmoA"},
}

EVO2_FAMILY_ORDER: list[str] = list(FAMILY_KOS)

# ── Pfam accession per family (single union map; family-intrinsic homology id) ───────
# Keys span both panels, which agree on shared-family accessions.
PFAM_ACCESSIONS: dict[str, str] = {
    # human + shared
    "globins": "PF00042",  # Globin
    "opsins": "PF00001",  # 7tm_1 (rhodopsin-like GPCR)
    "olfactory_receptors": "PF13853",  # 7tm_4 (olfactory receptor)
    "cytochrome_p450": "PF00067",  # p450
    "hox": "PF00046",  # Homeodomain
    "ras_gtpases": "PF00071",  # Ras
    "carbonic_anhydrase": "PF00194",  # alpha carbonic anhydrase (human panel name)
    "nitric_oxide_synthase": "PF02898",  # NO_synthase oxygenase domain
    "heme_oxygenase": "PF01126",  # Heme_oxygenase
    # redox-homeostasis additions (human)
    "peroxiredoxin": "PF00578",  # AhpC/TSA (peroxiredoxin)
    "glutathione_peroxidase": "PF00255",  # Glutathione peroxidase
    "glutaredoxin": "PF00462",  # Glutaredoxin
    "peroxidase": "PF03098",  # Animal haem peroxidase (MPO/LPO/EPX/TPO/PXDN)
    # Tier 1 — redox / detox metabolism
    "glutathione_s_transferase": "PF02798",  # GST N-terminal (thioredoxin-like)
    "aldehyde_dehydrogenase": "PF00171",  # Aldedh
    "aldo_keto_reductase": "PF00248",  # Aldo_ket_red
    "sulfotransferase": "PF00685",  # Sulfotransfer_1
    "udp_glucuronosyltransferase": "PF00201",  # UDPGT
    "nadph_oxidase": "PF01794",  # Ferric_reduct (NOX/DUOX)
    "arachidonate_lipoxygenase": "PF00305",  # Lipoxygenase
    "flavin_monooxygenase": "PF00743",  # FMO-like
    "steap_metalloreductase": "PF03807",  # F420_oxidored (STEAP oxidoreductase)
    # Tier 2 — Zn / metal metalloenzymes
    "matrix_metalloproteinase": "PF00413",  # Peptidase_M10 (Matrixin)
    "adam_metallopeptidase": "PF01421",  # Reprolysin (Peptidase_M12B)
    "adamts_metallopeptidase": "PF01421",  # Reprolysin (M12B; shares catalytic domain w/ ADAM)
    "m14_carboxypeptidase": "PF00246",  # Peptidase_M14 (Zn carboxypeptidase)
    "alcohol_dehydrogenase": "PF00107",  # ADH_zinc_N
    "metallothionein": "PF00131",  # Metallothionein
    "alkaline_phosphatase": "PF00245",  # Alk_phosphatase
    "histone_deacetylase_classI": "PF00850",  # Hist_deacetyl
    "ectonucleotide_pyrophosphatase": "PF01663",  # Phosphodiest (Type I PDE/nucleotidase)
    "phosphodiesterase": "PF00233",  # PDEase_I (cyclic-nucleotide)
    # Tier 3 — Fe / Fe-S
    "ferritin": "PF00210",  # Ferritin-like domain
    # Tier 4 — P-loop GTPase superfamily
    "rab_gtpase": "PF00071",  # Ras (Rab is a Ras-superfamily GTPase)
    "arf_gtpase": "PF00025",  # Arf
    "rho_gtpase": "PF00071",  # Ras (Rho family)
    "guanylate_binding_protein": "PF02263",  # GBP N-terminal
    # Tier 5 — GPCR / receptor superfamily
    "taste2_receptor": "PF05296",  # TAS2R
    "serotonin_receptor": "PF00001",  # 7tm_1 (class-A GPCR)
    "adrenoceptor": "PF00001",  # 7tm_1
    "glutamate_metabotropic": "PF00003",  # 7tm_3 (class-C GPCR)
    "dopamine_receptor": "PF00001",  # 7tm_1
    "muscarinic_receptor": "PF00001",  # 7tm_1
    "histamine_receptor": "PF00001",  # 7tm_1
    "p2y_receptor": "PF00001",  # 7tm_1
    "cxc_chemokine_receptor": "PF00001",  # 7tm_1
    # Tier 6 — mechanism contrast + controls
    "serine_protease": "PF00089",  # Trypsin (chymotrypsin-like serine protease)
    "histone_h4": "PF00125",  # Core histone fold
    # evo2 cross-kingdom-only
    "heme_copper_oxidase": "PF00115",  # COX1 (cytochrome c / quinol oxidase subunit I)
    "hemerythrin": "PF01814",  # Hemerythrin (non-heme di-iron O2 carrier)
    "carbonic_anhydrase_alpha": "PF00194",  # Eukaryotic-type (alpha) carbonic anhydrase
    "carbonic_anhydrase_beta": "PF00484",  # beta carbonic anhydrase (Pro_CA)
    "carbonic_anhydrase_gamma": "PF28366",  # gamma-CA (approximate: no single clean Pfam)
    "nitrogenase": "PF00142",  # Fer4_NifH (nitrogenase iron protein)
    "methyl_coenzyme_m_reductase": "PF02249",  # MCR_alpha (N-term)
    "methane_monooxygenase": "PF02332",  # sMMO mmoX representative (family also mixes pMMO)
}

# ── Per-panel plot colors ────────────────────────────────────────────────────────
# Store Arcadia color names so this module remains importable without matplotlib.
# Hue identifies chemistry blocks; lightness orders families within each block.
FAMILY_COLORS: dict[str, dict[str, str]] = {
    "human": {
        # Heme / O2 chemistry — red_shades
        "globins": "cinnabar",
        "cytochrome_p450": "dragon",
        "nitric_oxide_synthase": "amber",
        "heme_oxygenase": "tangerine",
        "peroxidase": "melon",
        # Thiol / peroxide redox — yellow_shades
        "peroxiredoxin": "umber",
        "glutathione_peroxidase": "mustard",
        "glutaredoxin": "canary",
        "glutathione_s_transferase": "sun",
        "nadph_oxidase": "oat",
        # Phase-I / phase-II detox — pink_shades
        "aldehyde_dehydrogenase": "azalea",
        "aldo_keto_reductase": "candy",
        "sulfotransferase": "rose",
        "udp_glucuronosyltransferase": "dress",
        "flavin_monooxygenase": "putty",
        # Non-heme Fe — the rust anchors, darkest first
        "arachidonate_lipoxygenase": "redwood",
        "steap_metalloreductase": "terracotta",
        "ferritin": "tumbleweed",
        # Zn / metal-dependent enzymes — purple_shades then teal_shades
        "carbonic_anhydrase": "concord",
        "matrix_metalloproteinase": "tanzanite",
        "adam_metallopeptidase": "aster",
        "adamts_metallopeptidase": "wish",
        "m14_carboxypeptidase": "iris",
        "alcohol_dehydrogenase": "depths",
        "metallothionein": "asparagus",
        "alkaline_phosphatase": "seaweed",
        "histone_deacetylase_classI": "teal",
        "ectonucleotide_pyrophosphatase": "glass",
        "phosphodiesterase": "mint",
        # P-loop GTPases — green_shades
        "ras_gtpases": "yucca",
        "rab_gtpase": "fern",
        "arf_gtpase": "matcha",
        "rho_gtpase": "lime",
        "guanylate_binding_protein": "edamame",
        # GPCRs — blue_shades then cool_gray_shades
        "opsins": "dusk",
        "olfactory_receptors": "lapis",
        "taste2_receptor": "aegean",
        "serotonin_receptor": "vital",
        "adrenoceptor": "sky",
        "glutamate_metabotropic": "steel",
        "dopamine_receptor": "marine",
        "muscarinic_receptor": "cloud",
        "histamine_receptor": "dove",
        "p2y_receptor": "ice",
        "cxc_chemokine_receptor": "denim",
        # Structural / positive controls — neutrals, so they read as "not a chemistry block"
        "hox": "charcoal",
        "serine_protease": "bark",
        "histone_h4": "stone",
    },
    "evo2": {
        # Heme / O2 chemistry — red_shades, then canary for the sixth
        "globins": "cinnabar",
        "hemerythrin": "dragon",
        "heme_copper_oxidase": "amber",
        "cytochrome_p450": "tangerine",
        "nitric_oxide_synthase": "melon",
        "heme_oxygenase": "canary",
        # Carbonic anhydrases — teal_shades, in class order (the axis is α → β → γ)
        "carbonic_anhydrase_alpha": "depths",
        "carbonic_anhydrase_beta": "asparagus",
        "carbonic_anhydrase_gamma": "seaweed",
        # Sensory GPCRs — blue
        "opsins": "dusk",
        "olfactory_receptors": "vital",
        # GTPase
        "ras_gtpases": "matcha",
        # C1 / N2 metabolism — purple_shades
        "nitrogenase": "concord",
        "methyl_coenzyme_m_reductase": "tanzanite",
        "methane_monooxygenase": "aster",
    },
}

# Integrity: every family in a panel's order has a color and a Pfam accession.
for _panel, _order in (("human", HUMAN_FAMILY_ORDER), ("evo2", EVO2_FAMILY_ORDER)):
    assert set(_order) <= set(PFAM_ACCESSIONS), f"{_panel}: families missing a Pfam accession"
    assert set(_order) == set(FAMILY_COLORS[_panel]), (
        f"{_panel}: family_order vs FAMILY_COLORS mismatch"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. ACCESSORS
# ══════════════════════════════════════════════════════════════════════════════

_HUMAN_MEMBERS: dict[str, list[str]] | None = None


def family_order(panel: str) -> list[str]:
    """Canonical family ordering for a panel ('human' curated; 'evo2' = sorted family names)."""
    if panel == "human":
        return list(HUMAN_FAMILY_ORDER)
    if panel == "evo2":
        return sorted(FAMILY_KOS)  # matches the evo2 pipeline's runtime sorted(manifest.family)
    raise ValueError(f"unknown panel {panel!r} (expected 'human' or 'evo2')")


def family_colors(panel: str) -> dict[str, str]:
    """Per-family plot colors for a panel, as Arcadia HexCodes."""
    import arcadia_style  # noqa: PLC0415 - deliberately lazy; see docstring

    return {fam: arcadia_style.resolve(name) for fam, name in FAMILY_COLORS[panel].items()}


def family_members(panel: str = "human") -> dict[str, list[str]]:
    """Family -> member gene list."""
    if panel == "human":
        global _HUMAN_MEMBERS
        if _HUMAN_MEMBERS is None:
            if not FAMILIES_DATA_JSON.exists():
                raise FileNotFoundError(
                    f"{FAMILIES_DATA_JSON} not found — run "
                    "`uv run python scripts/gene_families.py build-human-paralog` first."
                )
            data = json.loads(FAMILIES_DATA_JSON.read_text())
            members = data["gene_families"]
            # Fail loudly if the generated membership drifts from the curated metadata keys.
            assert set(members) == set(HUMAN_FAMILY_ORDER), (
                "families_data.json is out of sync with HUMAN_FAMILY_ORDER — re-run "
                "build-human-paralog after editing HGNC_FAMILY_SPEC."
            )
            _HUMAN_MEMBERS = members
        return _HUMAN_MEMBERS
    if panel == "evo2":
        import csv

        manifest = EVO2_DATA_DIR / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} not found — run "
                "`uv run python scripts/gene_families.py build-ortholog` first."
            )
        out: dict[str, list[str]] = {}
        with open(manifest, newline="") as fh:
            for row in csv.DictReader(fh):
                out.setdefault(row["family"], []).append(row["org_gene"])
        return out
    raise ValueError(f"unknown panel {panel!r} (expected 'human' or 'evo2')")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Human panel: HGNC gene groups -> families_data.json ──────────────────────────


def build_human_paralogs() -> None:
    """
    Resolve the HGNC gene groups in HGNC_FAMILY_SPEC -> families_data.json (gene symbol lists).
    """
    import random
    import urllib.request

    def fetch_group_members(gid: int) -> list[dict]:
        req = urllib.request.Request(
            HGNC_FETCH.format(gid=gid), headers={"Accept": "application/json"}
        )
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
                if not sym:
                    continue
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

    print("Building human gene families from HGNC gene groups...")
    gene_families = {
        name: build_family(name, HGNC_FAMILY_SPEC[name]) for name in HUMAN_FAMILY_ORDER
    }
    total = sum(len(v) for v in gene_families.values())
    data = {
        "source": "HGNC gene groups (rest.genenames.org)",
        "subsample_seed": SUBSAMPLE_SEED,
        "family_order": HUMAN_FAMILY_ORDER,
        "family_spec": HGNC_FAMILY_SPEC,
        "gene_families": gene_families,
    }
    FAMILIES_DATA_JSON.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"\nWrote {FAMILIES_DATA_JSON}  ({total} genes across {len(HUMAN_FAMILY_ORDER)} families)"
    )


# ── Evo2 panel: KEGG KOs -> per-family CDS FASTA + manifest.csv ──────────────────

import re  # noqa: E402  (module-level: used by the evo2 builder's BRITE parser)

_LEAF = re.compile(r"^([a-z][a-z0-9]{2,4})\s{2,}(.+)$")  # "hsa  Homo sapiens (human)"


def _kegg_get(path: str) -> str:
    """GET {KEGG}/{path} as text, with retries; '' on a 404-style empty body."""
    import time
    import urllib.request

    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{KEGG}/{path}", timeout=60) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                print(f"  KEGG GET {path} failed after retries ({e})")
                return ""
            time.sleep(2 * 2**attempt)
    return ""


def _load_org_taxonomy() -> dict[str, tuple[str, str]]:
    """Parse BRITE br08610 -> {org_code: (domain, group)} (domain = Eukaryota/Bacteria/Archaea)."""
    import time

    if BRITE_CACHE.exists():
        text = BRITE_CACHE.read_text()
    else:
        print("Downloading KEGG BRITE organism taxonomy (br08610)...")
        text = _kegg_get("get/br:br08610")
        BRITE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BRITE_CACHE.write_text(text)
        time.sleep(SLEEP)

    org_tax: dict[str, tuple[str, str]] = {}
    path: dict[str, str] = {}  # level-letter -> label
    for line in text.splitlines():
        if not line or line[0] < "A" or line[0] > "Z":
            continue
        letter, content = line[0], line[1:].strip()
        if not content:
            continue
        m = _LEAF.match(content)
        if m:  # leaf: an organism
            org = m.group(1)
            domain = path.get("A", "Unknown")
            group = path.get("B") or path.get("C") or domain
            org_tax[org] = (domain, group)
        else:  # interior taxon node: update the path, clear deeper levels
            path[letter] = content
            for deeper in [chr(c) for c in range(ord(letter) + 1, ord("Z") + 1)]:
                path.pop(deeper, None)
    print(f"  Parsed taxonomy for {len(org_tax)} KEGG organisms")
    return org_tax


def _ko_members(ko: str) -> list[str]:
    """All member genes (org:gene) linked to a KO via link/genes."""
    import time

    text = _kegg_get(f"link/genes/{ko}")
    time.sleep(SLEEP)
    members = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and ":" in parts[1]:
            members.append(parts[1])  # "org:gene"
    return members


def _stratified_sample(members, org_tax, n_target, seed):
    """Round-robin across (domain, group) buckets to maximise taxonomic spread.

    members: (org_gene, ko, label); returns (org_gene, ko, label, domain, group).
    """
    import random
    from collections import defaultdict

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for org_gene, ko, label in members:
        org = org_gene.split(":")[0]
        domain, group = org_tax.get(org, ("Unknown", "Unknown"))
        buckets[(domain, group)].append((org_gene, ko, label, domain, group))
    for b in buckets.values():
        rng.shuffle(b)
    order = sorted(buckets)
    chosen: list = []
    while len(chosen) < n_target and any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                chosen.append(buckets[k].pop())
                if len(chosen) >= n_target:
                    break
    return chosen


def _fetch_ntseq(org_genes: list[str]) -> dict[str, str]:
    """Batch-fetch nucleotide CDS for org:gene ids -> {org_gene: sequence}."""
    import time

    seqs: dict[str, str] = {}
    for i in range(0, len(org_genes), NTSEQ_BATCH):
        batch = org_genes[i : i + NTSEQ_BATCH]
        text = _kegg_get(f"get/{'+'.join(batch)}/ntseq")
        time.sleep(SLEEP)
        cur_id, cur_seq = None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur_seq)
                cur_id = line[1:].split()[0]  # ">hsa:3043 K13823 ..." -> "hsa:3043"
                cur_seq = []
            elif line.strip():
                cur_seq.append(line.strip().upper())
        if cur_id:
            seqs[cur_id] = "".join(cur_seq)
        print(f"    fetched {min(i + NTSEQ_BATCH, len(org_genes))}/{len(org_genes)} CDS", end="\r")
    print()
    return seqs


def _mmseqs_dedup(fasta_in: Path, tmp_dir: Path, min_id: float = 0.90) -> set[str]:
    """Representative FASTA ids after easy-linclust, or all ids if mmseqs is unavailable."""
    import shutil
    import subprocess

    if shutil.which("mmseqs") is None:
        print("  mmseqs not found — skipping dedup (install via bioconda to enable).")
        return {r.split()[0] for r in fasta_in.read_text().splitlines() if r.startswith(">")}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix = tmp_dir / "dedup"
    subprocess.run(
        [
            "mmseqs",
            "easy-linclust",
            str(fasta_in),
            str(prefix),
            str(tmp_dir / "tmp"),
            "--dbtype",
            "2",
            "--min-seq-id",
            str(min_id),
            "--cov-mode",
            "1",
            "-c",
            "0.8",
            "--cluster-mode",
            "2",
            "--threads",
            "4",
            "--remove-tmp-files",
            "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    rep = Path(str(prefix) + "_rep_seq.fasta")
    reps = {line[1:].split()[0] for line in rep.read_text().splitlines() if line.startswith(">")}
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return reps


def build_orthologs(
    target: int = TARGET_PER_FAMILY,
    families: list[str] | None = None,
    no_dedup: bool = False,
    seed: int = SUBSAMPLE_SEED,
) -> None:
    """Resolve the KEGG KOs in FAMILY_KOS -> per-family CDS FASTA + manifest.csv
    (data/evo2_gene_families/).
    """
    import csv
    import sys

    families = families or list(FAMILY_KOS)
    EVO2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    org_tax = _load_org_taxonomy()

    manifest_path = EVO2_DATA_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "family",
                "ko",
                "ko_label",
                "org_gene",
                "organism",
                "domain",
                "group",
                "cds_len",
            ],
        )
        writer.writeheader()

        for family in families:
            print(f"\n=== {family} ===")
            members: list[tuple[str, str, str]] = []
            for ko, label in FAMILY_KOS[family].items():
                m = _ko_members(ko)
                print(f"  {ko} ({label}): {len(m)} genes")
                members.extend((og, ko, label) for og in m)

            # _kegg_get returns '' on failure, so an empty member list usually means KEGG was
            # unreachable rather than a real empty family — warn instead of building a tiny family.
            if not members:
                print(
                    f"  WARNING: 0 members for '{family}' — KEGG may be down or KO ids changed; "
                    "skipping.",
                    file=sys.stderr,
                )
                continue

            n_candidates = int(target * OVERSAMPLE)
            chosen = _stratified_sample(members, org_tax, n_candidates, seed)
            n_dom = len({c[3] for c in chosen})
            print(
                f"  {len(chosen)} candidates across {n_dom} domains "
                f"({len({c[4] for c in chosen})} taxonomic groups)"
            )

            seqs = _fetch_ntseq([c[0] for c in chosen])
            chosen = [c for c in chosen if c[0] in seqs and len(seqs[c[0]]) >= 100]

            cand_fasta = EVO2_DATA_DIR / f"{family}.candidates.fasta"
            with open(cand_fasta, "w") as fh:
                for og, ko, label, _dom, _grp in chosen:
                    fh.write(f">{og}|{family}|{ko}|{label}\n{seqs[og]}\n")
            if no_dedup:
                keep_ids = {c[0] for c in chosen}
            else:
                print("  MMseqs2 dedup (90% id)...")
                reps = _mmseqs_dedup(cand_fasta, EVO2_DATA_DIR / f"tmp_{family}")
                keep_ids = {rid.split("|")[0] for rid in reps}
            chosen = [c for c in chosen if c[0] in keep_ids][:target]
            cand_fasta.unlink(missing_ok=True)

            fam_fasta = EVO2_DATA_DIR / f"{family}.fasta"
            with open(fam_fasta, "w") as fh:
                for og, ko, label, dom, grp in chosen:
                    fh.write(f">{og}|{family}|{ko}|{label}\n{seqs[og]}\n")
                    writer.writerow(
                        {
                            "family": family,
                            "ko": ko,
                            "ko_label": label,
                            "org_gene": og,
                            "organism": og.split(":")[0],
                            "domain": dom,
                            "group": grp,
                            "cds_len": len(seqs[og]),
                        }
                    )
            mf.flush()
            print(f"  -> {len(chosen)} CDS written to {fam_fasta}")

    print(f"\nManifest: {manifest_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-human-paralog", help="HGNC gene groups -> families_data.json")
    pe = sub.add_parser("build-ortholog", help="KEGG KOs -> per-family CDS FASTA + manifest.csv")
    pe.add_argument(
        "--target", type=int, default=TARGET_PER_FAMILY, help="CDS per family after dedup"
    )
    pe.add_argument("--families", nargs="+", default=list(FAMILY_KOS), choices=list(FAMILY_KOS))
    pe.add_argument("--no-dedup", action="store_true", help="Skip MMseqs2 dedup")
    pe.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    args = ap.parse_args()

    if args.cmd == "build-human-paralog":
        build_human_paralogs()
    elif args.cmd == "build-ortholog":
        build_orthologs(
            target=args.target, families=args.families, no_dedup=args.no_dedup, seed=args.seed
        )


if __name__ == "__main__":
    main()
