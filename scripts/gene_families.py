"""Single source of truth for gene-family SELECTION: the 48 HGNC human paralog families
that scripts/mammalian_orthologs/resolve_orthologs.py expands to 1:1 orthologs across 24
mammals."""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILIES_DATA_JSON = ROOT / "scripts" / "families_data.json"  # human membership artifact

# ── HGNC build constants ────────────────────────────────────────────────────────
_FETCH = "https://rest.genenames.org/fetch/gene_group_id/{gid}"
SUBSAMPLE_SEED = 0  # seeds the max_members subsample in build_human_paralogs


# Definitions
# Human paralog panel: HGNC gene-group specification
# group_ids are HGNC gene-group IDs (resolved against the live HGNC REST API). extra_symbols are
# appended verbatim (for genes HGNC leaves ungrouped). symbol_prefix restricts a broad group to one
# cluster. max_members triggers a seeded subsample (none set today; machinery kept generic).
_FAMILY_SPEC: dict[str, dict] = {
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
}

# Integrity: every family in the panel order has a color and a Pfam accession.
assert set(HUMAN_FAMILY_ORDER) <= set(PFAM_ACCESSIONS), "families missing a Pfam accession"
assert set(HUMAN_FAMILY_ORDER) == set(FAMILY_COLORS["human"]), (
    "family_order vs FAMILY_COLORS mismatch"
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ACCESSORS
# ══════════════════════════════════════════════════════════════════════════════

_HUMAN_MEMBERS: dict[str, list[str]] | None = None


def family_order(panel: str) -> list[str]:
    """Canonical family ordering for the human paralog panel."""
    if panel == "human":
        return list(HUMAN_FAMILY_ORDER)
    raise ValueError(f"unknown panel {panel!r} (expected 'human')")


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
                "build-human-paralog after editing _FAMILY_SPEC."
            )
            _HUMAN_MEMBERS = members
        return _HUMAN_MEMBERS
    raise ValueError(f"unknown panel {panel!r} (expected 'human')")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Human panel: HGNC gene groups -> families_data.json ──────────────────────────


def build_human_paralogs() -> None:
    """
    Resolve the HGNC gene groups in _FAMILY_SPEC -> families_data.json (gene symbol lists).
    """
    import random
    import urllib.request

    def fetch_group_members(gid: int) -> list[dict]:
        req = urllib.request.Request(_FETCH.format(gid=gid), headers={"Accept": "application/json"})
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
    gene_families = {name: build_family(name, _FAMILY_SPEC[name]) for name in HUMAN_FAMILY_ORDER}
    total = sum(len(v) for v in gene_families.values())
    data = {
        "source": "HGNC gene groups (rest.genenames.org)",
        "subsample_seed": SUBSAMPLE_SEED,
        "family_order": HUMAN_FAMILY_ORDER,
        "family_spec": _FAMILY_SPEC,
        "gene_families": gene_families,
    }
    FAMILIES_DATA_JSON.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"\nWrote {FAMILIES_DATA_JSON}  ({total} genes across {len(HUMAN_FAMILY_ORDER)} families)"
    )


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
    args = ap.parse_args()

    if args.cmd == "build-human-paralog":
        build_human_paralogs()


if __name__ == "__main__":
    main()
