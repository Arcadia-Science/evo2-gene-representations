"""Score between-family separation and independent distance baselines for mammalian orthologs."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from gene_families import PFAM_ACCESSIONS  # noqa: E402
from geodesic_utils import compute_centroid_geodesic, mantel_test  # noqa: E402
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402
from mammal_score import load_cds  # noqa: E402
from pfam_hmm_jsd import compute_pfam_jsd  # noqa: E402

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE_ROOT = ROOT / "data" / "cache" / "mammal_embed"
AXIS = {
    "pfam_jsd": "1_homology",
    "gc_content": "control",
    "kmer": "control",
}
MIN_MEMBERS = 8  # a family needs at least this many embedded loci to have a stable centroid


def load_embedded(arm: str, manifest: str = "complete_manifest.csv"):
    cache = CACHE_ROOT / arm
    man = pd.read_csv(OUT / manifest)
    man["key"] = man.group + "__" + man.species
    vecs, rows = [], []
    for _, r in man.iterrows():
        p = cache / f"{r['key']}.npy"
        if p.exists():
            vecs.append(np.load(p))
            rows.append(r)
    return np.stack(vecs, axis=1), pd.DataFrame(rows).reset_index(drop=True)


def kmer_between(meta, cds, fams):
    """Family×family mean member-pair k-mer distance (composition axis) on this panel's CDS."""
    keys = meta["key"].tolist()
    D = kmer_distance_matrix([cds[k] for k in keys], k=6)
    fam_of = meta["family"].to_numpy()
    F = len(fams)
    M = np.zeros((F, F))
    for i, a in enumerate(fams):
        ia = np.where(fam_of == a)[0]
        for j, b in enumerate(fams):
            if i == j:
                continue
            ib = np.where(fam_of == b)[0]
            M[i, j] = D[np.ix_(ia, ib)].mean()
    return M


def gc_between(meta, cds, fams):
    def gc(s):
        s = s.upper()
        return (s.count("G") + s.count("C")) / max(1, len(s))

    fam_gc = {f: np.mean([gc(cds[k]) for k in meta[meta.family == f]["key"]]) for f in fams}
    return np.array([[abs(fam_gc[a] - fam_gc[b]) for b in fams] for a in fams])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", default="transcript", choices=["transcript", "cds", "transcript_cdsmask"]
    )
    ap.add_argument(
        "--manifest",
        default="complete_manifest.csv",
        help="locus manifest (e.g. complete_manifest_cap400.csv for the 400/family cap)",
    )
    ap.add_argument("--tag", default="", help="suffix appended to the run-dir arm name (e.g. _400)")
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(32)))
    ap.add_argument("--n-perms", type=int, default=9999)
    args = ap.parse_args()

    stack, meta = load_embedded(args.arm, args.manifest)
    cds = load_cds()
    counts = meta.family.value_counts()
    fams = sorted([f for f in counts.index if counts[f] >= MIN_MEMBERS])
    sub_idx = _cols_for(
        meta, args.arm, fams, args.manifest
    )  # stack cols for kept families, meta order
    meta = meta[meta.family.isin(fams)].reset_index(drop=True)
    print(f"between-family over {len(fams)} families: {fams}", flush=True)

    # layer-independent axes (computed once)
    print(
        "computing transferable axes (Pfam-JSD, k-mer, and GC)...",
        flush=True,
    )
    axes = {
        "pfam_jsd": compute_pfam_jsd(fams, PFAM_ACCESSIONS),
        "kmer": kmer_between(meta, cds, fams),
        "gc_content": gc_between(meta, cds, fams),
    }
    fam_arr = meta.family.to_numpy()
    sweep_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}{args.tag}"
    iu = np.triu_indices(len(fams), 1)
    for L in args.layers:
        emb = stack[L][sub_idx]
        cen = compute_centroid_geodesic(emb, fam_arr, fams)
        run = sweep_root / f"blocks{L}"
        run.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cen, index=fams, columns=fams).to_csv(
            run / "evo2_mammal_centroid_distances.csv"
        )
        rows = []
        for base, M in axes.items():
            x, y = cen[iu], M[iu]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
                continue
            rho = spearmanr(x[ok], y[ok]).statistic
            p = mantel_test(cen, M, n_perms=args.n_perms)[1]
            rows.append({"baseline": base, "axis": AXIS[base], "spearman_rho": rho, "p_mantel": p})
        pd.DataFrame(rows).to_csv(run / "between_family_baseline_scores.csv", index=False)
    print(
        f"wrote between_family_baseline_scores.csv to {len(args.layers)} run dirs -> {sweep_root}",
        flush=True,
    )


def _cols_for(meta, arm, fams, manifest="complete_manifest.csv"):
    """Column indices into the embedded stack (load_embedded builds it in manifest order,
    over present loci) for the rows of the kept families, in that same order."""
    full = pd.read_csv(OUT / manifest)
    full["key"] = full.group + "__" + full.species
    cache = CACHE_ROOT / arm
    present = [
        (k, f)
        for k, f in zip(full["key"], full["family"], strict=False)
        if (cache / f"{k}.npy").exists()
    ]
    return [i for i, (k, f) in enumerate(present) if f in fams]


if __name__ == "__main__":
    main()
