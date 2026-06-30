"""Multi-axis between-family ground-truth baselines for the Evo2 gene-family run.

The between-family (Axis A) result is otherwise scored against only Pfam JSD, which is
null. This module decomposes the between-family ground truth into two orthogonal axes
(homology, mechanism), so we can ask *which kind* of relatedness the latent geometry
tracks — evolutionary homology vs biochemical analogy:

  Axis 1  Homology   — curated superfamily/fold tier, Pfam JSD (added separately),
                       eggNOG OG distance (added separately; heavy).
  Axis 2  Mechanism  — cofactor distance, EC-number distance, GO molecular-function overlap.
  Control Composition— k-mer (aggregated from the run's sequence-level matrix), GC.

Each baseline is an F×F family distance matrix scored by Spearman ρ + a Mantel p-value
against the latent family-centroid geodesic. The panel's convergent families (the three
non-homologous carbonic anhydrases; globins vs hemerythrin) are far on Axis 1 but close
on Axis 2 — so a geometry that tracks Axis 2 on those pairs is the headline
"captures analogy beyond homology" result.

Operates on a finished run dir (like scripts/baselines/pfam_hmm_jsd.py): writes
between_family_baseline_distances.csv (every matrix, long form) and
between_family_baseline_scores.csv (ρ + Mantel p per baseline, tagged by axis).

Usage:
    uv run python scripts/baselines/between_family_baselines.py \
        --run-dir results/YYYY-MM-DD_evo2-gene-families
"""

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
from gene_families import PFAM_ACCESSIONS  # noqa: E402  (single source of family Pfam accessions)
from geodesic_utils import mantel_test, upper_triangle  # noqa: E402

DATA_DIR = Path("data/evo2_gene_families")
EMBED_META = DATA_DIR / "embeddings" / "metadata.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Curated per-family annotations (between_family_baselines_design.md §3).
# This table is the input to vet; the data-derived baselines check it.
#
#   homology_group : families sharing a label are tier-1 homologous. Every family is
#                    its own group here (the convergent families are deliberately NOT
#                    grouped); cross-family homology lives in HOMOLOGY_EDGES below.
#   cofactors      : tokens scored by COFACTOR_SIM (shared chemistry, ancestry-blind).
#   ec             : EC number ('non-enzyme' for receptors/carriers).
#   gas            : the gas the family binds/produces/processes ('none' if not gas) —
#                    descriptive panel metadata; the context/process axis was removed.
#   process        : coarse biological-role leaf — descriptive only (see gas).
# ─────────────────────────────────────────────────────────────────────────────
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
    # ── GPN-Star panel additions (same biology, different family subset) ──────────
    # "carbonic_anhydrase" (GPN) is the alpha class specifically (PF00194) — identical
    # annotation to carbonic_anhydrase_alpha above.
    "carbonic_anhydrase": dict(
        homology_group="alpha_CA", cofactors=["zinc"], ec="4.2.1.1",
        gas="CO2", process="CO2_hydration"),
    # hox: homeobox developmental TFs — the panel's unrelated negative control.
    "hox": dict(
        homology_group="homeobox", cofactors=[], ec="non-enzyme",
        gas="none", process="developmental_regulation"),
}

# Pfam accession per family (keys the Pfam-JSD and GO baselines) is imported from the shared
# scripts/gene_families.py (PFAM_ACCESSIONS, above) — the single source for both panels.

# Cross-family homology not captured by a shared homology_group, as (tier, pair) rules.
# tier 1 = homologous (same superfamily/clan); tier 2 = distant shared fold division.
HOMOLOGY_EDGES: dict[frozenset[str], int] = {
    # OLFR and animal (type-2) opsins are both class-A GPCRs (Pfam clan CL0192).
    frozenset({"olfactory_receptors", "opsins"}): 1,
    # NifH (nitrogenase Fe protein) and Ras are both P-loop (Walker-A) NTPases — deep,
    # distant homology; nothing else about them is shared.
    frozenset({"nitrogenase", "ras_gtpases"}): 2,
}

# Cofactor chemical similarity in [0, 1] for non-identical tokens (1.0 implicit on the
# diagonal; 0.0 implicit for unlisted pairs). Hemes are mutually near; both di-iron and
# heme are Fe-O2 chemistry → partial; copper co-occurs with heme in the O2 oxidases.
COFACTOR_SIM: dict[frozenset[str], float] = {
    frozenset({"heme_b", "heme_a"}): 0.85,
    frozenset({"heme_b", "heme_b_thiolate"}): 0.9,
    frozenset({"heme_b", "heme_substrate"}): 0.85,
    frozenset({"heme_a", "heme_b_thiolate"}): 0.8,
    frozenset({"heme_a", "heme_substrate"}): 0.8,
    frozenset({"heme_b_thiolate", "heme_substrate"}): 0.8,
    frozenset({"heme_b", "nonheme_diiron"}): 0.4,
    frozenset({"heme_b_thiolate", "nonheme_diiron"}): 0.4,
    frozenset({"heme_a", "copper"}): 0.3,
    frozenset({"zinc", "iron"}): 0.3,
    frozenset({"femoco", "fe4s4"}): 0.6,
}

# ── matrix builders ──────────────────────────────────────────────────────────


def _sym(fams: list[str], pair_fn) -> np.ndarray:
    """F×F symmetric matrix from a pairwise distance fn (diagonal forced to 0)."""
    F = len(fams)
    D = np.zeros((F, F), dtype=np.float64)
    for i, j in itertools.combinations(range(F), 2):
        D[i, j] = D[j, i] = pair_fn(fams[i], fams[j])
    return D


def homology_tier_matrix(fams: list[str]) -> np.ndarray:
    """Ordinal homology distance: 1 same group/clan, 2 distant shared fold, 3 unrelated."""
    def pair(a, b):
        edge = HOMOLOGY_EDGES.get(frozenset({a, b}))
        if edge is not None:
            return float(edge)
        ga = FAMILY_ANNOTATIONS[a]["homology_group"]
        gb = FAMILY_ANNOTATIONS[b]["homology_group"]
        return 1.0 if ga == gb else 3.0
    return _sym(fams, pair)


def cofactor_matrix(fams: list[str]) -> np.ndarray:
    """1 − max chemical similarity over the two families' cofactor sets."""
    def sim(x, y):
        return 1.0 if x == y else COFACTOR_SIM.get(frozenset({x, y}), 0.0)

    def pair(a, b):
        ca = FAMILY_ANNOTATIONS[a]["cofactors"]
        cb = FAMILY_ANNOTATIONS[b]["cofactors"]
        if not ca or not cb:  # a cofactor-less family (receptors) shares no chemistry
            return 1.0
        return 1.0 - max(sim(x, y) for x in ca for y in cb)
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
    """Aggregate the run's sequence-level k-mer distance to between-family means.

    Diagonal = mean within-family pair distance; off-diagonal = mean cross-family pair.
    """
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


# ── GO molecular-function (data-derived + curated fill) ──────────────────────
#
# GO terms per family come from GO's curated pfam2go mapping (family-intrinsic, like
# Pfam JSD: keyed by the family's Pfam accession). Namespace (MF/BP/CC) is resolved
# via QuickGO. pfam2go has no Pfam-level mapping for a few families on this panel
# (α-/γ-CA, hemerythrin, sMMO) — those would otherwise drop out of the analogy test, so
# they get a minimal CURATED fill of their canonical GO terms (their own well-established
# function, not fabricated cross-family overlap). go_mf joins the mechanism axis.
# Distance = 1 − Jaccard over the molecular-function term set.
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


# ── FASTA GC ─────────────────────────────────────────────────────────────────


def _gc(seq: str) -> tuple[int, int]:
    s = seq.upper()
    return s.count("G") + s.count("C"), s.count("A") + s.count("T")


def gc_by_family(fams: list[str], seq_source: str, fam_of: dict[str, str]) -> dict[str, float]:
    """Mean GC fraction per family. Evo2: per-family FASTAs; GPN: cds_sequences.json
    keyed by gene (membership from `fam_of`). Returns {} if no source is available."""
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
    else:  # gpn: gene-keyed CDS json
        gpn_cds = Path("data/cache/cds_sequences.json")
        if not gpn_cds.exists():
            return {}
        seqs = json.loads(gpn_cds.read_text())
        for gene, seq in seqs.items():
            f = fam_of.get(gene)
            if f in gc:
                g, a = _gc(seq)
                gc[f] += g
                at[f] += a
    return {f: (gc[f] / (gc[f] + at[f]) if (gc[f] + at[f]) else 0.0) for f in fams}


def load_between_kmer(run_dir: Path, fams: list[str], seq_families: np.ndarray | None,
                      kmer_seq: np.ndarray | None) -> np.ndarray | None:
    """Between-family k-mer matrix: prefer a precomputed family-level CSV (GPN-Star),
    else aggregate the run's sequence-level k-mer matrix (Evo2). None if neither."""
    fam_csv = run_dir / "kmer_distance_family.csv"
    if fam_csv.exists():
        df = pd.read_csv(fam_csv, index_col=0).reindex(index=fams, columns=fams)
        return df.values.astype(np.float64)
    if kmer_seq is not None and seq_families is not None:
        return kmer_between_matrix(fams, seq_families, kmer_seq)
    return None


# ── targeted convergent-pair rank test ───────────────────────────────────────
#
# A global ρ over 105 pairs cannot reflect a handful of analogy pairs. The sharper
# question: where does a convergent pair RANK in geodesic closeness among all pairs,
# vs where homology places it? "The three CAs rank in the closest decile despite zero
# homology" is the directly-testable analogy claim. Each entry is (family_a, family_b,
# relationship-type); pairs whose families are absent from a run are skipped, so the
# same list serves the Evo2 and GPN-Star panels.
CONVERGENT_PAIRS: list[tuple[str, str, str]] = [
    # Convergent: same chemistry/role, independent origin (homology says FAR).
    ("carbonic_anhydrase_alpha", "carbonic_anhydrase_beta", "convergent_CO2"),
    ("carbonic_anhydrase_alpha", "carbonic_anhydrase_gamma", "convergent_CO2"),
    ("carbonic_anhydrase_beta", "carbonic_anhydrase_gamma", "convergent_CO2"),
    ("globins", "hemerythrin", "convergent_O2"),
    ("methane_monooxygenase", "methyl_coenzyme_m_reductase", "convergent_CH4"),
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
    """For each curated pair present in the run, its percentile rank (0=closest,
    100=farthest) among all off-diagonal family pairs, under the geodesic and each
    biological axis. The geodesic percentile vs the homology percentile is the readout."""
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


# ── scoring ──────────────────────────────────────────────────────────────────

# axis tag -> {baseline name -> matrix}; filled in main and scored uniformly.
AXIS_OF = {
    "homology_tier": "1_homology",
    "pfam_jsd": "1_homology",
    "cofactor": "2_mechanism",
    "ec_number": "2_mechanism",
    "go_mf": "2_mechanism",
    "kmer": "control",
    "gc_content": "control",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seq-source", choices=["evo2", "gpn"], default="evo2",
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
        sys.exit(f"No curated annotation for families: {missing}")
    geo = centroid.values.astype(np.float64)

    # Per-CDS metadata: family label + member id (org_gene for Evo2, gene for GPN-Star).
    meta_path = run_dir / "metadata.csv" if (run_dir / "metadata.csv").exists() else EMBED_META
    meta = pd.read_csv(meta_path)
    id_col = "gene" if args.seq_source == "gpn" else "org_gene"
    fam_of = dict(zip(meta[id_col], meta["family"])) if id_col in meta.columns else {}
    seq_families = meta["family"].to_numpy()
    # Sequence-level k-mer matrix (Evo2 .npy); GPN uses a precomputed family-level CSV.
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
            mats[name] = pd.read_csv(csv, index_col=0).reindex(index=fams, columns=fams).values
        if not mats:
            sys.exit(f"--distances-from {donor}: no betweenfam_*_distances.csv to reuse")
        print(f"Reusing {len(mats)} cached between-family distance matrices from {donor}")
    else:
        print(f"Scoring {len(fams)} families ({args.seq_source}) against multi-axis baselines\n")
        mats = {
            "homology_tier": homology_tier_matrix(fams),
            "cofactor": cofactor_matrix(fams),
            "ec_number": ec_matrix(fams),
        }
        gc = gc_by_family(fams, args.seq_source, fam_of)
        if gc:  # empty if no sequence source on disk (e.g. GPN without cds_sequences.json)
            mats["gc_content"] = gc_matrix(fams, gc)
        else:
            print("  [skip] gc_content: no sequence source available")
        kmer_fam = load_between_kmer(run_dir, fams, seq_families, kmer_seq)
        if kmer_fam is not None:
            mats["kmer"] = kmer_fam
        # Pfam HMM JSD — the *continuous* (graded) homology baseline (the categorical tier
        # is near-degenerate on a mutually-non-homologous panel). Computed by
        # scripts/baselines/pfam_hmm_jsd.py; fold it into the homology axis here if present.
        jsd_path = run_dir / "pfam_jsd_distances.csv"
        if jsd_path.exists():
            mats["pfam_jsd"] = (
                pd.read_csv(jsd_path, index_col=0).reindex(index=fams, columns=fams).values
            )
        print("Fetching GO terms (pfam2go + QuickGO aspect, cached)...")
        mats.update(go_matrices(fams, PFAM_ACCESSIONS))

    # Long-form matrix dump (every baseline, all family pairs) + per-baseline scores.
    long_rows, score_rows = [], []
    geo_u = upper_triangle(geo)
    for name, D in mats.items():
        D_u = upper_triangle(D)
        # Report NaN (not a spurious correlation / Mantel p) when a baseline is
        # uninformative. Two cases: (a) NaN cells from a reindex family-coverage
        # mismatch — np.ptp would return NaN, *not* 0, so it slips past a bare ==0 guard
        # and mantel_test would return a spurious p≈1e-3 (nan>=nan is always False); and
        # (b) a constant matrix (a baseline on which no two families share an annotation →
        # all-disjoint, every off-diagonal equal).
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
                "geodesic_pctile", "homology_tier_pctile", "cofactor_pctile",
                "ec_number_pctile"]
        show = [c for c in cols if c in report.columns]
        print("\nConvergent-pair rank test (percentile among all family pairs; 0=closest):")
        print(report[show].to_string(index=False))
        print(f"Saved {run_dir}/convergent_pair_ranks.csv")
    print("Done.")


if __name__ == "__main__":
    main()
