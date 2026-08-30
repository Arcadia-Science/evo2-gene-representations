"""Multi-axis between-family ground-truth baselines for the Evo2 gene-family run."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.cofactor_annotations import (  # noqa: E402
    cofactor_similarity as _chebi_cofactor_similarity,
    load_cofactor_annotations,
)
from gene_families import PFAM_ACCESSIONS  # noqa: E402  (single source of family Pfam accessions)
from geodesic_utils import mantel_test, upper_triangle  # noqa: E402

DATA_DIR = Path("data/evo2_gene_families")
EMBED_META = DATA_DIR / "embeddings" / "metadata.csv"

# Curated per-family annotations; the data-derived baselines check them.
#   cofactors  scored via ChEBI-derived cofactor_similarity, role-blind
#   ec         EC number ('non-enzyme' for receptors/carriers)
#   homology_group, gas, process  panel metadata only — their baselines were removed
FAMILY_ANNOTATIONS: dict[str, dict] = {
    "globins": dict(
        homology_group="globin", cofactors=["heme_b"], ec="non-enzyme",
        gas="O2", process="O2_transport"),
    "hemerythrin": dict(
        homology_group="hemerythrin", cofactors=["nonheme_diiron"], ec="non-enzyme",
        gas="O2", process="O2_transport"),
    "heme_copper_oxidase": dict(
        homology_group="HCO", cofactors=["heme_a", "copper"], ec="7.1.1.9",
        gas="O2", process="O2_respiration"),
    "cytochrome_p450": dict(
        homology_group="p450", cofactors=["heme_b_thiolate"], ec="1.14.14.1",
        gas="O2", process="redox_monooxygenation"),
    "nitric_oxide_synthase": dict(
        homology_group="nos", cofactors=["heme_b_thiolate", "bh4"], ec="1.14.13.39",
        gas="NO", process="NO_signalling"),
    "heme_oxygenase": dict(
        homology_group="heme_oxygenase", cofactors=["heme_substrate"], ec="1.14.14.18",
        gas="CO", process="heme_catabolism"),
    "carbonic_anhydrase_alpha": dict(
        homology_group="alpha_CA", cofactors=["zinc"], ec="4.2.1.1",
        gas="CO2", process="CO2_hydration"),
    "carbonic_anhydrase_beta": dict(
        homology_group="beta_CA", cofactors=["zinc"], ec="4.2.1.1",
        gas="CO2", process="CO2_hydration"),
    "carbonic_anhydrase_gamma": dict(
        homology_group="gamma_CA", cofactors=["zinc", "iron"], ec="4.2.1.1",
        gas="CO2", process="CO2_hydration"),
    "methane_monooxygenase": dict(
        homology_group="mmo", cofactors=["nonheme_diiron", "copper"], ec="1.14.13.25",
        gas="CH4", process="CH4_oxidation"),
    "methyl_coenzyme_m_reductase": dict(
        homology_group="mcr", cofactors=["nickel_f430"], ec="2.8.4.1",
        gas="CH4", process="methanogenesis"),
    "nitrogenase": dict(
        homology_group="ploop_ntpase", cofactors=["femoco", "fe4s4"], ec="1.18.6.1",
        gas="N2", process="N2_fixation"),
    "ras_gtpases": dict(
        homology_group="ploop_gtpase", cofactors=["magnesium"], ec="3.6.5.2",
        gas="none", process="signal_transduction"),
    "olfactory_receptors": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "opsins": dict(
        homology_group="opsin_mixed", cofactors=["retinal"], ec="non-enzyme",
        gas="none", process="phototransduction"),
    # Human-paralog panel additions.
    # This carbonic anhydrase family is the alpha class (PF00194).
    "carbonic_anhydrase": dict(
        homology_group="alpha_CA", cofactors=["zinc"], ec="4.2.1.1",
        gas="CO2", process="CO2_hydration"),
    # hox: homeobox developmental TFs — the panel's unrelated negative control.
    "hox": dict(
        homology_group="homeobox", cofactors=[], ec="non-enzyme",
        gas="none", process="developmental_regulation"),
    # Redox, detoxification, metalloenzyme, GTPase, and GPCR families.
    "peroxiredoxin": dict(
        homology_group="peroxiredoxin", cofactors=[], ec="1.11.1.24",
        gas="none", process="peroxide_detox"),
    "glutathione_peroxidase": dict(  # catalytic selenocysteine (a residue, not a bound cofactor)
        homology_group="gpx", cofactors=[], ec="1.11.1.9",
        gas="none", process="peroxide_detox"),
    "glutaredoxin": dict(
        homology_group="glutaredoxin", cofactors=["glutathione"], ec="1.20.4.1",
        gas="none", process="thiol_redox"),
    "peroxidase": dict(
        homology_group="heme_peroxidase", cofactors=["heme_b"], ec="1.11.1.7",
        gas="none", process="peroxide_detox"),
    "glutathione_s_transferase": dict(
        homology_group="gst", cofactors=["glutathione"], ec="2.5.1.18",
        gas="none", process="detox_conjugation"),
    "aldehyde_dehydrogenase": dict(
        homology_group="aldh", cofactors=["nad"], ec="1.2.1.3",
        gas="none", process="aldehyde_oxidation"),
    "aldo_keto_reductase": dict(
        homology_group="akr", cofactors=["nadp"], ec="1.1.1.21",
        gas="none", process="carbonyl_reduction"),
    "sulfotransferase": dict(
        homology_group="sult", cofactors=[], ec="2.8.2.1",
        gas="none", process="detox_conjugation"),
    "udp_glucuronosyltransferase": dict(
        homology_group="ugt", cofactors=[], ec="2.4.1.17",
        gas="none", process="detox_conjugation"),
    "nadph_oxidase": dict(
        homology_group="nox", cofactors=["heme_b", "fad"], ec="1.6.3.1",
        gas="O2", process="ROS_generation"),
    "arachidonate_lipoxygenase": dict(
        homology_group="alox", cofactors=["iron"], ec="1.13.11.31",
        gas="O2", process="lipid_peroxidation"),
    "flavin_monooxygenase": dict(
        homology_group="fmo", cofactors=["fad"], ec="1.14.13.8",
        gas="none", process="redox_monooxygenation"),
    "steap_metalloreductase": dict(
        homology_group="steap", cofactors=["fad", "heme_b"], ec="1.16.1.-",
        gas="none", process="metal_reduction"),
    "matrix_metalloproteinase": dict(
        homology_group="mmp_m10", cofactors=["zinc"], ec="3.4.24.-",
        gas="none", process="proteolysis"),
    "adam_metallopeptidase": dict(
        homology_group="adam_m12b", cofactors=["zinc"], ec="3.4.24.-",
        gas="none", process="proteolysis"),
    "adamts_metallopeptidase": dict(
        homology_group="adamts_m12b", cofactors=["zinc"], ec="3.4.24.-",
        gas="none", process="proteolysis"),
    "m14_carboxypeptidase": dict(
        homology_group="cpa_m14", cofactors=["zinc"], ec="3.4.17.-",
        gas="none", process="proteolysis"),
    "alcohol_dehydrogenase": dict(
        homology_group="adh_zn", cofactors=["zinc", "nad"], ec="1.1.1.1",
        gas="none", process="alcohol_oxidation"),
    "metallothionein": dict(
        homology_group="metallothionein", cofactors=["zinc", "copper"], ec="non-enzyme",
        gas="none", process="metal_binding"),
    "alkaline_phosphatase": dict(
        homology_group="alp", cofactors=["zinc", "magnesium"], ec="3.1.3.1",
        gas="none", process="phosphate_hydrolysis"),
    "histone_deacetylase_classI": dict(
        homology_group="hdac", cofactors=["zinc"], ec="3.5.1.98",
        gas="none", process="deacetylation"),
    "ectonucleotide_pyrophosphatase": dict(
        homology_group="enpp", cofactors=["zinc"], ec="3.6.1.9",
        gas="none", process="nucleotide_hydrolysis"),
    "phosphodiesterase": dict(
        homology_group="pde", cofactors=["zinc", "magnesium"], ec="3.1.4.17",
        gas="none", process="cyclic_nucleotide_hydrolysis"),
    "ferritin": dict(
        homology_group="ferritin", cofactors=["nonheme_diiron"], ec="1.16.3.1",
        gas="none", process="iron_storage"),
    "rab_gtpase": dict(
        homology_group="ploop_gtpase", cofactors=["magnesium"], ec="3.6.5.2",
        gas="none", process="signal_transduction"),
    "arf_gtpase": dict(
        homology_group="ploop_gtpase", cofactors=["magnesium"], ec="3.6.5.2",
        gas="none", process="signal_transduction"),
    "rho_gtpase": dict(
        homology_group="ploop_gtpase", cofactors=["magnesium"], ec="3.6.5.2",
        gas="none", process="signal_transduction"),
    "guanylate_binding_protein": dict(
        homology_group="gbp_gtpase", cofactors=["magnesium"], ec="3.6.5.-",
        gas="none", process="signal_transduction"),
    "taste2_receptor": dict(
        homology_group="tas2r", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "serotonin_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "adrenoceptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "glutamate_metabotropic": dict(
        homology_group="classC_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "dopamine_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "muscarinic_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "histamine_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "p2y_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "cxc_chemokine_receptor": dict(
        homology_group="classA_GPCR", cofactors=[], ec="non-enzyme",
        gas="none", process="GPCR_signalling"),
    "serine_protease": dict(  # Ser-His-Asp triad, no metal — mechanism contrast to the M10/M12B/M14 zinc-proteases
        homology_group="serine_protease", cofactors=[], ec="3.4.21.-",
        gas="none", process="proteolysis"),
    "histone_h4": dict(
        homology_group="histone", cofactors=[], ec="non-enzyme",
        gas="none", process="chromatin_packaging"),
}

# Pfam accession per family (keys the Pfam-JSD and GO baselines) is imported from the shared
# Pfam accessions shared by both panels.


# Cofactor similarity, derived from ChEBI: 0.5*scaffold + 0.5*metal, where scaffold is the
# Sørensen-Dice overlap of the two cofactors' ChEBI ancestor sets and metal the Jaccard overlap of
# the elements parsed from the formula. Continuous in [0, 1] rather than a hand-picked class label.
# Shared biochemical role is out of scope — EC and GO cover function, Pfam-JSD covers homology.
_COFACTOR_ANN: dict[str, dict] | None = None


def _cofactor_ann() -> dict[str, dict]:
    """Lazily load (and cache in-process) the ChEBI-derived cofactor annotations."""
    global _COFACTOR_ANN
    if _COFACTOR_ANN is None:
        _COFACTOR_ANN = load_cofactor_annotations()
    return _COFACTOR_ANN


def cofactor_similarity(x: str, y: str) -> float:
    """Scaled ChEBI-derived cofactor similarity in [0, 1] (see cofactor_annotations)."""
    return _chebi_cofactor_similarity(x, y, _cofactor_ann())


# ── matrix builders


def _sym(fams: list[str], pair_fn) -> np.ndarray:
    """F×F symmetric matrix from a pairwise distance fn (diagonal forced to 0)."""
    F = len(fams)
    D = np.zeros((F, F), dtype=np.float64)
    for i, j in itertools.combinations(range(F), 2):
        D[i, j] = D[j, i] = pair_fn(fams[i], fams[j])
    return D


def cofactor_matrix(fams: list[str]) -> np.ndarray:
    """1 − max cofactor similarity over the two families' cofactor sets."""
    def pair(a, b):
        ca = FAMILY_ANNOTATIONS[a]["cofactors"]
        cb = FAMILY_ANNOTATIONS[b]["cofactors"]
        if not ca or not cb:  # a cofactor-less family (receptors) shares no chemistry
            return 1.0
        return 1.0 - max(cofactor_similarity(x, y) for x in ca for y in cb)
    return _sym(fams, pair)


def ec_matrix(fams: list[str]) -> np.ndarray:
    """1 − shared EC prefix depth / 4. Non-enzymes: 0 to each other, 1 to enzymes."""
    def pair(a, b):
        ea, eb = FAMILY_ANNOTATIONS[a]["ec"], FAMILY_ANNOTATIONS[b]["ec"]
        if ea == "non-enzyme" or eb == "non-enzyme":
            return 0.0 if ea == eb else 1.0
        la, lb = ea.split("."), eb.split(".")
        shared = 0
        for x, y in zip(la, lb):
            if x == y:
                shared += 1
            else:
                break
        return 1.0 - shared / 4.0
    return _sym(fams, pair)


def kmer_between_matrix(fams: list[str], seq_families: np.ndarray, kmer_seq: np.ndarray) -> np.ndarray:
    """Aggregate the run's sequence-level k-mer distance to between-family means."""
    F = len(fams)
    idx = {f: np.where(seq_families == f)[0] for f in fams}
    D = np.zeros((F, F), dtype=np.float64)
    for i in range(F):
        for j in range(i, F):
            ia, ib = idx[fams[i]], idx[fams[j]]
            if i == j:
                sub = kmer_seq[np.ix_(ia, ia)]
                tri = sub[np.triu_indices(len(ia), k=1)]
                D[i, j] = tri.mean() if len(tri) else 0.0
            else:
                D[i, j] = D[j, i] = kmer_seq[np.ix_(ia, ib)].mean()
    return D


def gc_matrix(fams: list[str], gc_by_family: dict[str, float]) -> np.ndarray:
    """|mean GC fraction difference| between families."""
    return _sym(fams, lambda a, b: abs(gc_by_family[a] - gc_by_family[b]))


# ── GO molecular-function
# Terms come from GO's curated pfam2go mapping, keyed by Pfam accession like Pfam-JSD. A few
# families have no pfam2go entry and get a minimal curated fill of their own canonical terms,
# rather than dropping out of the analogy test. Distance = 1 - Jaccard over the MF term set.
PFAM2GO_URL = "http://current.geneontology.org/ontology/external2go/pfam2go"
PFAM2GO_CACHE = Path("data/cache/pfam2go.txt")
GO_ASPECT_CACHE = Path("data/cache/go_aspects.json")
QUICKGO_TERMS = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{ids}"

# Canonical GO terms for families pfam2go does not map at the Pfam level (fill only).
CURATED_GO_FILL: dict[str, list[str]] = {
    "carbonic_anhydrase_alpha": ["GO:0004089", "GO:0008270"],  # carbonate dehydratase; Zn
    "carbonic_anhydrase_gamma": ["GO:0004089", "GO:0008270"],
    "carbonic_anhydrase": ["GO:0004089", "GO:0008270"],
    "hemerythrin": ["GO:0005344", "GO:0019825", "GO:0015671"],  # O2 carrier/binding; O2 transport
    "methane_monooxygenase": ["GO:0004497", "GO:0016705"],  # monooxygenase activity
}


def load_pfam2go() -> dict[str, list[str]]:
    """{Pfam accession -> [GO ids]} from GO's curated pfam2go mapping (cached)."""
    import re
    if not PFAM2GO_CACHE.exists():
        PFAM2GO_CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(PFAM2GO_URL, headers={"User-Agent": "Mozilla/5.0 (research)"})
        PFAM2GO_CACHE.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    out: dict[str, list[str]] = {}
    for ln in PFAM2GO_CACHE.read_text().splitlines():
        m = re.match(r"Pfam:(PF\d+)\s+.*?> GO:.+? ; (GO:\d+)", ln)
        if m:
            out.setdefault(m.group(1), []).append(m.group(2))
    return out


def go_aspects(go_ids: set[str]) -> dict[str, str]:
    """{GO id -> aspect} (molecular_function / biological_process / cellular_component)
    via QuickGO, cached. Only the panel's handful of unique terms are queried."""
    cache: dict[str, str] = {}
    if GO_ASPECT_CACHE.exists():
        cache = json.loads(GO_ASPECT_CACHE.read_text())
    need = sorted(go_ids - set(cache))
    for i in range(0, len(need), 100):
        chunk = need[i : i + 100]
        url = QUICKGO_TERMS.format(ids=",".join(chunk))
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            j = json.load(urllib.request.urlopen(req, timeout=60))
            for r in j.get("results", []):
                if r.get("aspect"):
                    cache[r["id"]] = r["aspect"]
        except Exception as e:  # noqa: BLE001
            print(f"  QuickGO aspect fetch failed ({e})")
        time.sleep(0.3)
    GO_ASPECT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GO_ASPECT_CACHE.write_text(json.dumps(cache, indent=0))
    return cache


def go_matrices(fams: list[str], accessions: dict[str, str]) -> dict[str, np.ndarray]:
    """Return {'go_mf': D}: 1 − Jaccard over each family's GO molecular-function term set."""
    pf2go = load_pfam2go()
    # Per-family raw GO set = pfam2go(acc) ∪ curated fill where pfam2go is empty.
    raw: dict[str, set[str]] = {}
    for f in fams:
        terms = set(pf2go.get(accessions.get(f, ""), []))
        if not terms:
            terms = set(CURATED_GO_FILL.get(f, []))
        raw[f] = terms
    aspect = go_aspects(set().union(*raw.values()) if raw else set())

    def matrix(want: str) -> np.ndarray:
        sets = {f: {g for g in raw[f] if aspect.get(g) == want} for f in fams}

        def pair(a, b):
            sa, sb = sets[a], sets[b]
            if not sa or not sb:
                return 1.0
            return 1.0 - len(sa & sb) / len(sa | sb)
        return _sym(fams, pair)

    return {"go_mf": matrix("molecular_function")}


# ── FASTA GC


def _gc(seq: str) -> tuple[int, int]:
    s = seq.upper()
    return s.count("G") + s.count("C"), s.count("A") + s.count("T")


def gc_by_family(fams: list[str], seq_source: str, fam_of: dict[str, str]) -> dict[str, float]:
    """Mean GC fraction from cross-kingdom FASTAs or the human CDS cache.
    Returns {} if no source is available."""
    gc = {f: 0 for f in fams}
    at = {f: 0 for f in fams}
    if seq_source == "evo2":
        for f in fams:
            fasta = DATA_DIR / f"{f}.fasta"
            if not fasta.exists():
                continue
            for line in fasta.read_text().splitlines():
                if line.startswith(">") or not line:
                    continue
                g, a = _gc(line)
                gc[f] += g
                at[f] += a
    else:  # human: gene-keyed CDS JSON
        human_cds = Path("data/cache/cds_sequences.json")
        if not human_cds.exists():
            return {}
        seqs = json.loads(human_cds.read_text())
        for gene, seq in seqs.items():
            f = fam_of.get(gene)
            if f in gc:
                g, a = _gc(seq)
                gc[f] += g
                at[f] += a
    return {f: (gc[f] / (gc[f] + at[f]) if (gc[f] + at[f]) else 0.0) for f in fams}


def load_between_kmer(run_dir: Path, fams: list[str], seq_families: np.ndarray | None,
                      kmer_seq: np.ndarray | None) -> np.ndarray | None:
    """Load a family-level k-mer matrix or aggregate a sequence-level matrix.
    Returns None if neither is available."""
    fam_csv = run_dir / "kmer_distance_family.csv"
    if fam_csv.exists():
        df = pd.read_csv(fam_csv, index_col=0).reindex(index=fams, columns=fams)
        return df.values.astype(np.float64)
    if kmer_seq is not None and seq_families is not None:
        return kmer_between_matrix(fams, seq_families, kmer_seq)
    return None


# ── targeted convergent-pair rank test
# A global rho over 105 pairs cannot reflect a handful of analogy pairs. The sharper question is
# where a convergent pair RANKS in geodesic closeness among all pairs, vs where homology places it.
# Pairs whose families are absent from a run are skipped, so one list serves every panel.
CONVERGENT_PAIRS: list[tuple[str, str, str]] = [
    # Convergent: same chemistry/role, independent origin (homology says FAR).
    ("carbonic_anhydrase_alpha", "carbonic_anhydrase_beta", "convergent_CO2"),
    ("carbonic_anhydrase_alpha", "carbonic_anhydrase_gamma", "convergent_CO2"),
    ("carbonic_anhydrase_beta", "carbonic_anhydrase_gamma", "convergent_CO2"),
    ("globins", "hemerythrin", "convergent_O2"),
    ("methane_monooxygenase", "methyl_coenzyme_m_reductase", "convergent_CH4"),
    # Convergent peroxide-detox and proteolysis controls.
    ("peroxiredoxin", "glutathione_peroxidase", "convergent_peroxide_detox"),
    ("peroxiredoxin", "peroxidase", "convergent_peroxide_detox"),
    ("glutathione_peroxidase", "peroxidase", "convergent_peroxide_detox"),
    ("matrix_metalloproteinase", "serine_protease", "convergent_proteolysis"),
    ("m14_carboxypeptidase", "serine_protease", "convergent_proteolysis"),
    # Heme/O2 chemistry cluster (non-homologous, shared cofactor).
    ("globins", "cytochrome_p450", "heme_cluster"),
    ("globins", "nitric_oxide_synthase", "heme_cluster"),
    ("globins", "heme_copper_oxidase", "heme_cluster"),
    ("globins", "heme_oxygenase", "heme_cluster"),
    # Homology positive control (should be close on homology too).
    ("olfactory_receptors", "opsins", "homologous_GPCR"),
    # Unrelated negative controls (far on every axis).
    ("globins", "hox", "negative_control"),
    ("ras_gtpases", "hox", "negative_control"),
    ("globins", "ras_gtpases", "negative_control"),
]


def convergent_pair_report(fams: list[str], mats: dict[str, np.ndarray]) -> pd.DataFrame:
    """For each curated pair present in the run, its percentile rank (0=closest, 100=farthest) among all off-diagonal family pairs, under the geodesic and each biological axis."""
    from scipy.stats import rankdata

    pos = {f: i for i, f in enumerate(fams)}
    iu = np.triu_indices(len(fams), k=1)
    pair_ix = {frozenset({fams[i], fams[j]}): p for p, (i, j) in enumerate(zip(*iu))}

    # Percentile rank of every pair within each matrix's upper triangle (avg ties).
    pct = {}
    for name, D in mats.items():
        u = D[iu]
        # Skip constant or NaN-containing matrices: rankdata would rank NaN as the
        # largest value, silently corrupting every pair's percentile for that baseline.
        if not np.isfinite(u).all() or np.ptp(u) == 0:
            continue
        pct[name] = 100.0 * (rankdata(u, method="average") - 1) / (len(u) - 1)

    rows = []
    for a, b, kind in CONVERGENT_PAIRS:
        if a not in pos or b not in pos:
            continue
        p = pair_ix[frozenset({a, b})]
        row = {"family_a": a, "family_b": b, "relationship": kind}
        for name in mats:
            row[f"{name}_pctile"] = round(float(pct[name][p]), 1) if name in pct else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ── scoring

# axis tag -> {baseline name -> matrix}; filled in main and scored uniformly.
AXIS_OF = {
    "pfam_jsd": "1_homology",
    "cofactor": "2_mechanism",
    "ec_number": "2_mechanism",
    "go_mf": "2_mechanism",
    "kmer": "control",
    "gc_content": "control",
}

# Retire the uninformative categorical homology baseline.
DEPRECATED_BASELINES = {"homology_tier"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seq-source", choices=["evo2", "human"], default="evo2",
                    help="Which panel this run is from (drives GC source and k-mer).")
    ap.add_argument("--n-perms", type=int, default=9999, help="Mantel permutations (F≈15 → cheap).")
    ap.add_argument("--distances-from", default=None,
                    help="Reuse the layer-independent per-axis distance matrices "
                         "(betweenfam_*_distances.csv) from this donor run dir; only the "
                         "Spearman/Mantel scores against THIS run's centroid geodesic are "
                         "recomputed. For an all-layer sweep — the biological axes do not "
                         "depend on the embedding layer.")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    cen_files = list(run_dir.glob("*_centroid_distances.csv"))
    if not cen_files:
        sys.exit(f"No *_centroid_distances.csv in {run_dir}")
    centroid = pd.read_csv(cen_files[0], index_col=0)
    fams = centroid.index.tolist()
    missing = [f for f in fams if f not in FAMILY_ANNOTATIONS]
    if missing:
        print(f"WARN: no curated annotation for {len(missing)} families; excluding from "
              f"between-family axes: {missing}", file=sys.stderr)
        fams = [f for f in fams if f in FAMILY_ANNOTATIONS]
        centroid = centroid.reindex(index=fams, columns=fams)
    geo = centroid.values.astype(np.float64)

    # Per-CDS metadata: family label and panel-specific member identifier.
    meta_path = run_dir / "metadata.csv" if (run_dir / "metadata.csv").exists() else EMBED_META
    meta = pd.read_csv(meta_path)
    id_col = "gene" if args.seq_source == "human" else "org_gene"
    fam_of = dict(zip(meta[id_col], meta["family"])) if id_col in meta.columns else {}
    seq_families = meta["family"].to_numpy()
    # Prefer the sequence-level k-mer matrix when its order matches the metadata.
    kmer_seq = None
    if (run_dir / "kmer_distance.npy").exists():
        kmer_seq = np.load(run_dir / "kmer_distance.npy")
        if kmer_seq.shape[0] != len(seq_families):
            kmer_seq = None  # order would not align; fall back to the family-level CSV

    if args.distances_from:
        # Sweep fast path: the per-axis matrices are sequence/annotation-derived and
        # identical at every layer, so load them from a donor run instead of rebuilding
        # (skips the GO/QuickGO + Pfam fetches). Only the scoring below is layer-dependent.
        donor = Path(args.distances_from)
        mats = {}
        for csv in sorted(donor.glob("betweenfam_*_distances.csv")):
            name = csv.name[len("betweenfam_"):-len("_distances.csv")]
            if name in DEPRECATED_BASELINES:
                continue  # donor may still hold a deprecated matrix on disk; skip it
            if name not in AXIS_OF:
                continue  # not one of ours: ot_between_family_sweep.py writes its own
                # betweenfam_ot_*_distances.csv into the same run dir, and those have no
                # axis — scoring them below would KeyError and lose the whole scores file.
            mats[name] = pd.read_csv(csv, index_col=0).reindex(index=fams, columns=fams).values
        if not mats:
            sys.exit(f"--distances-from {donor}: no betweenfam_*_distances.csv to reuse")
        print(f"Reusing {len(mats)} cached between-family distance matrices from {donor}")
    else:
        print(f"Scoring {len(fams)} families ({args.seq_source}) against multi-axis baselines\n")
        mats = {
            "cofactor": cofactor_matrix(fams),
            "ec_number": ec_matrix(fams),
        }
        gc = gc_by_family(fams, args.seq_source, fam_of)
        if gc:
            mats["gc_content"] = gc_matrix(fams, gc)
        else:
            print("  [skip] gc_content: no sequence source available")
        kmer_fam = load_between_kmer(run_dir, fams, seq_families, kmer_seq)
        if kmer_fam is not None:
            mats["kmer"] = kmer_fam
        # Pfam HMM JSD — the continuous homology baseline, and the only one on the panel.
        jsd_path = run_dir / "pfam_jsd_distances.csv"
        if jsd_path.exists():
            mats["pfam_jsd"] = (
                pd.read_csv(jsd_path, index_col=0).reindex(index=fams, columns=fams).values
            )
        print("Fetching GO terms (pfam2go + QuickGO aspect, cached)...")
        mats.update(go_matrices(fams, PFAM_ACCESSIONS))

    # Remove retired matrices loaded from donor directories.
    for _dep in DEPRECATED_BASELINES:
        mats.pop(_dep, None)

    # Long-form matrix dump (every baseline, all family pairs) + per-baseline scores.
    long_rows, score_rows = [], []
    geo_u = upper_triangle(geo)
    for name, D in mats.items():
        D_u = upper_triangle(D)
        # Report NaN, not a spurious rho/p, when a baseline is uninformative: NaN cells from a
        # reindex coverage mismatch slip past a bare ==0 guard (np.ptp returns NaN, and nan>=nan is
        # False, giving a spurious p), and a constant matrix has no variance to correlate.
        if not np.isfinite(D_u).all():
            rho, p_sp, p_mantel = float("nan"), float("nan"), float("nan")
            print(f"  [skip] {name}: NaN cells (family-coverage mismatch in source matrix)")
        elif np.ptp(D_u) == 0:
            rho, p_sp, p_mantel = float("nan"), float("nan"), float("nan")
            print(f"  [skip] {name}: constant distance matrix (uninformative on this panel)")
        else:
            rho, p_sp = spearmanr(geo_u, D_u)
            _, p_mantel = mantel_test(geo, D, n_perms=args.n_perms)
        score_rows.append({
            "baseline": name, "axis": AXIS_OF[name],
            "spearman_rho": rho, "p_spearman": p_sp, "p_mantel": p_mantel,
        })
        dfm = pd.DataFrame(D, index=fams, columns=fams)
        dfm.to_csv(run_dir / f"betweenfam_{name}_distances.csv")
        for i, j in itertools.combinations(range(len(fams)), 2):
            long_rows.append({"baseline": name, "family_a": fams[i], "family_b": fams[j], "distance": D[i, j]})

    pd.DataFrame(long_rows).to_csv(run_dir / "between_family_baseline_distances.csv", index=False)
    scores = pd.DataFrame(score_rows).sort_values(["axis", "baseline"]).reset_index(drop=True)
    scores.to_csv(run_dir / "between_family_baseline_scores.csv", index=False)

    print("\nBetween-family geodesic-centroid correlation, by axis:")
    print(f"  {'axis':<12} {'baseline':<16} {'ρ':>8} {'p_spearman':>12} {'p_mantel':>10}")
    for _, r in scores.iterrows():
        print(f"  {r.axis:<12} {r.baseline:<16} {r.spearman_rho:>+8.3f} "
              f"{r.p_spearman:>12.2e} {r.p_mantel:>10.4f}")
    print(f"\nSaved {run_dir}/between_family_baseline_scores.csv")
    print(f"Saved {run_dir}/between_family_baseline_distances.csv (+ per-baseline matrices)")

    # Targeted convergent-pair rank test (geodesic closeness percentile vs homology).
    report = convergent_pair_report(fams, {"geodesic": geo, **mats})
    if not report.empty:
        report.to_csv(run_dir / "convergent_pair_ranks.csv", index=False)
        cols = ["family_a", "family_b", "relationship",
                "geodesic_pctile", "pfam_jsd_pctile", "cofactor_pctile",
                "ec_number_pctile"]
        show = [c for c in cols if c in report.columns]
        print("\nConvergent-pair rank test (percentile among all family pairs; 0=closest):")
        print(report[show].to_string(index=False))
        print(f"Saved {run_dir}/convergent_pair_ranks.csv")
    print("Done.")


if __name__ == "__main__":
    main()
