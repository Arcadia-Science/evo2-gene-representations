"""Multi-axis between-family ground-truth baselines for the Evo2 gene-family run."""

from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geodesic_utils import mantel_test, upper_triangle  # noqa: E402

# ── matrix builders


def _sym(fams: list[str], pair_fn) -> np.ndarray:
    """F×F symmetric matrix from a pairwise distance fn (diagonal forced to 0)."""
    F = len(fams)
    D = np.zeros((F, F), dtype=np.float64)
    for i, j in itertools.combinations(range(F), 2):
        D[i, j] = D[j, i] = pair_fn(fams[i], fams[j])
    return D


def kmer_between_matrix(
    fams: list[str], seq_families: np.ndarray, kmer_seq: np.ndarray
) -> np.ndarray:
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


# ── FASTA GC


def _gc(seq: str) -> tuple[int, int]:
    s = seq.upper()
    return s.count("G") + s.count("C"), s.count("A") + s.count("T")


def gc_by_family(fams: list[str], fam_of: dict[str, str]) -> dict[str, float]:
    """Mean GC fraction from the human CDS cache. Returns {} if no source is available."""
    gc = {f: 0 for f in fams}
    at = {f: 0 for f in fams}
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


def load_between_kmer(
    run_dir: Path, fams: list[str], seq_families: np.ndarray | None, kmer_seq: np.ndarray | None
) -> np.ndarray | None:
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
    # Non-homologous families with shared heme/O2 chemistry.
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
    """For each curated pair present in the run, its percentile rank (0=closest, 100=farthest) among
    all off-diagonal family pairs, under the geodesic and each biological axis.
    """
    from scipy.stats import rankdata

    pos = {f: i for i, f in enumerate(fams)}
    iu = np.triu_indices(len(fams), k=1)
    pair_ix = {frozenset({fams[i], fams[j]}): p for p, (i, j) in enumerate(zip(*iu, strict=False))}

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

# Baseline name to axis label.
AXIS_OF = {
    "pfam_jsd": "1_homology",
    "kmer": "control",
    "gc_content": "control",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-perms", type=int, default=9999, help="Mantel permutations (F≈15 → cheap).")
    ap.add_argument(
        "--distances-from",
        default=None,
        help="Reuse the layer-independent per-axis distance matrices "
        "(betweenfam_*_distances.csv) from this donor run dir; only the "
        "Spearman/Mantel scores against THIS run's centroid geodesic are "
        "recomputed. For an all-layer sweep — the biological axes do not "
        "depend on the embedding layer.",
    )
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    cen_files = list(run_dir.glob("*_centroid_distances.csv"))
    if not cen_files:
        sys.exit(f"No *_centroid_distances.csv in {run_dir}")
    centroid = pd.read_csv(cen_files[0], index_col=0)
    fams = centroid.index.tolist()
    geo = centroid.values.astype(np.float64)

    # Per-CDS metadata: family label and panel-specific member identifier.
    meta = pd.read_csv(run_dir / "metadata.csv")
    id_col = "gene"
    fam_of = dict(zip(meta[id_col], meta["family"], strict=False)) if id_col in meta.columns else {}
    seq_families = meta["family"].to_numpy()
    # Prefer the sequence-level k-mer matrix when its order matches the metadata.
    kmer_seq = None
    if (run_dir / "kmer_distance.npy").exists():
        kmer_seq = np.load(run_dir / "kmer_distance.npy")
        if kmer_seq.shape[0] != len(seq_families):
            kmer_seq = None  # order would not align; fall back to the family-level CSV

    if args.distances_from:
        # Sweep fast path: the matrices are sequence-derived and identical at every layer.
        donor = Path(args.distances_from)
        mats = {}
        for csv in sorted(donor.glob("betweenfam_*_distances.csv")):
            name = csv.name[len("betweenfam_") : -len("_distances.csv")]
            if name not in AXIS_OF:
                continue
            mats[name] = pd.read_csv(csv, index_col=0).reindex(index=fams, columns=fams).values
        if not mats:
            sys.exit(f"--distances-from {donor}: no betweenfam_*_distances.csv to reuse")
        print(f"Reusing {len(mats)} cached between-family distance matrices from {donor}")
    else:
        print(f"Scoring {len(fams)} families against multi-axis baselines\n")
        mats = {}
        gc = gc_by_family(fams, fam_of)
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
        score_rows.append(
            {
                "baseline": name,
                "axis": AXIS_OF[name],
                "spearman_rho": rho,
                "p_spearman": p_sp,
                "p_mantel": p_mantel,
            }
        )
        dfm = pd.DataFrame(D, index=fams, columns=fams)
        dfm.to_csv(run_dir / f"betweenfam_{name}_distances.csv")
        for i, j in itertools.combinations(range(len(fams)), 2):
            long_rows.append(
                {"baseline": name, "family_a": fams[i], "family_b": fams[j], "distance": D[i, j]}
            )

    pd.DataFrame(long_rows).to_csv(run_dir / "between_family_baseline_distances.csv", index=False)
    scores = pd.DataFrame(score_rows).sort_values(["axis", "baseline"]).reset_index(drop=True)
    scores.to_csv(run_dir / "between_family_baseline_scores.csv", index=False)

    print("\nBetween-family geodesic-centroid correlation, by axis:")
    print(f"  {'axis':<12} {'baseline':<16} {'ρ':>8} {'p_spearman':>12} {'p_mantel':>10}")
    for _, r in scores.iterrows():
        print(
            f"  {r.axis:<12} {r.baseline:<16} {r.spearman_rho:>+8.3f} "
            f"{r.p_spearman:>12.2e} {r.p_mantel:>10.4f}"
        )
    print(f"\nSaved {run_dir}/between_family_baseline_scores.csv")
    print(f"Saved {run_dir}/between_family_baseline_distances.csv (+ per-baseline matrices)")

    # Targeted convergent-pair rank test (geodesic closeness percentile vs homology).
    report = convergent_pair_report(fams, {"geodesic": geo, **mats})
    if not report.empty:
        report.to_csv(run_dir / "convergent_pair_ranks.csv", index=False)
        cols = [
            "family_a",
            "family_b",
            "relationship",
            "geodesic_pctile",
            "pfam_jsd_pctile",
        ]
        show = [c for c in cols if c in report.columns]
        print("\nConvergent-pair rank test (percentile among all family pairs; 0=closest):")
        print(report[show].to_string(index=False))
        print(f"Saved {run_dir}/convergent_pair_ranks.csv")
    print("Done.")


if __name__ == "__main__":
    main()
