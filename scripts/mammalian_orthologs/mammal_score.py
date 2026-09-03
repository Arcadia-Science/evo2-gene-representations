"""Score whether within-group Evo2 geodesics recover mammalian evolutionary distances."""

from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
import paths  # noqa: E402
from geodesic_utils import compute_geodesic, find_min_connected_k  # noqa: E402
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402
from protein_alignment_patristic import (  # noqa: E402
    align_members,
    tree_patristic,
)

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE_ROOT = ROOT / "data" / "cache" / "mammal_embed"
MIN_SP = 10
# Baseline key to output filename and correlation column.
BASELINES = {
    "speciestree": ("within_family_speciestree.csv", "spearman_geodesic_speciestree"),
    "patristic": ("within_family_patristic.csv", "spearman_geodesic_patristic"),
    "kmer": ("kmer_within_family_correlations.csv", "spearman_geodesic_kmer"),
    "gc": ("within_family_gc.csv", "spearman_geodesic_gc"),
}


def gc_distance_matrix(seqs: list[str]) -> np.ndarray:
    """Return pairwise absolute GC-fraction differences for a group's CDS sequences."""
    gc = np.array(
        [
            (sum(c in "GCgc" for c in s) / n) if (n := sum(c in "ACGTacgt" for c in s)) else np.nan
            for s in seqs
        ]
    )
    return np.abs(gc[:, None] - gc[None, :])


def load_embedded(arm: str):
    cache = CACHE_ROOT / arm
    man = pd.read_csv(OUT / "complete_manifest.csv")
    man["key"] = man["group"] + "__" + man["species"]
    vecs, rows = [], []
    for _, r in man.iterrows():
        p = cache / f"{r['key']}.npy"
        if p.exists():
            vecs.append(np.load(p))
            rows.append(r)
    return np.stack(vecs, axis=1), pd.DataFrame(rows).reset_index(drop=True)


def load_cds() -> dict[str, str]:
    seqs = {}
    for fa in (OUT / "seqs" / "complete" / "cds").glob("*.fasta"):
        hid, chunk = None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if hid:
                    seqs[hid] = "".join(chunk)
                parts = line[1:].split("|")
                hid = f"{parts[0]}__{parts[1]}"
                chunk = []
            elif line.strip():
                chunk.append(line.strip())
        if hid:
            seqs[hid] = "".join(chunk)
    return seqs


def group_ground_truths(meta: pd.DataFrame, cds: dict, pat: pd.DataFrame) -> dict:
    """Per group (>=MIN_SP embedded species): layer-independent distance matrices over its
    embedded members, in a fixed member order.
    """
    gt = {}
    for (fam, group), sub in meta.groupby(["family", "group"]):
        members = sub["key"].tolist()  # group__species
        species = sub["species"].tolist()
        if len(members) < MIN_SP:
            continue
        n = len(members)
        d = {
            "family": fam,
            "members": members,
            "n": n,
            "speciestree": pat.loc[species, species].values,
            "kmer": kmer_distance_matrix([cds[m] for m in members], k=6),
            "gc": gc_distance_matrix([cds[m] for m in members]),
        }
        # One MAFFT + FastTree scratch dir per group, over a ~3 h stage. $GLM_TMPDIR (or TMPDIR)
        # moves it off a small /tmp; tmp_dir() returns None for the system default.
        with tempfile.TemporaryDirectory(dir=paths.tmp_dir()) as tmp:
            res = align_members(members, {m: cds[m] for m in members}, Path(tmp))
            if res:
                tp = tree_patristic(res, members, Path(tmp))
                if tp is not None:
                    d["patristic"] = tp
        gt[group] = d
    return gt


def score_all_layers(stack, meta, gt, layers, arm, distance="geodesic"):
    sweep_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{arm}"
    per_group_rows = []
    kpos = {k: i for i, k in enumerate(meta["key"])}
    for L in layers:
        if distance == "geodesic":
            _, W = find_min_connected_k(stack[L], k_min=3)
            geo = pd.DataFrame(compute_geodesic(W), index=meta["key"], columns=meta["key"])
        rows = []  # per (group, baseline) this layer
        for group, d in gt.items():
            if distance == "geodesic":
                G = geo.loc[d["members"], d["members"]].values
            else:
                # Direct pairwise angular distance provides the graph-free comparison.
                V = stack[L][[kpos[m] for m in d["members"]]]
                U = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
                G = np.arccos(np.clip(U @ U.T, -1.0, 1.0)) / np.pi
            iu = np.triu_indices(d["n"], 1)
            for base in BASELINES:
                if base not in d:
                    continue
                g, b = G[iu], d[base][iu]
                ok = np.isfinite(g) & np.isfinite(b)
                if ok.sum() >= 6 and np.ptp(g[ok]) > 0 and np.ptp(b[ok]) > 0:
                    rho = spearmanr(g[ok], b[ok]).statistic
                    rows.append(
                        {"family": d["family"], "group": group, "baseline": base, "rho": rho}
                    )
                    per_group_rows.append({**rows[-1], "layer": L})
        df = pd.DataFrame(rows)
        run = sweep_root / f"blocks{L}"
        run.mkdir(parents=True, exist_ok=True)
        for base, (fname, col) in BASELINES.items():
            fam_mean = df[df.baseline == base].groupby("family")["rho"].mean().reset_index()
            # The metric is named in BOTH the filename and the column. Naming only the file was
            # the old behaviour and it read as "geodesic" to anyone who opened the CSV.
            out, col_out = fname, col
            if distance != "geodesic":
                out = fname.replace(".csv", "_angular.csv")
                col_out = col.replace("spearman_geodesic_", "spearman_angular_")
            fam_mean.columns = ["family", col_out]
            fam_mean.to_csv(run / out, index=False)
    return pd.DataFrame(per_group_rows), sweep_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", default="transcript", choices=["transcript", "cds", "transcript_cdsmask"]
    )
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(32)))
    ap.add_argument(
        "--distance",
        default="both",
        choices=["geodesic", "angular", "both"],
        help="distance metric to score; both reuses one alignment pass and writes geodesic "
        "tables plus the _angular tables used by publication figures",
    )
    args = ap.parse_args()

    stack, meta = load_embedded(args.arm)
    pat = pd.read_csv(OUT / "tree" / "species_patristic.csv", index_col=0)
    cds = load_cds()
    print(
        f"{stack.shape[1]} embedded loci ({args.arm}); building per-group ground truths "
        f"(MAFFT/FastTree, once)...",
        flush=True,
    )
    gt = group_ground_truths(meta, cds, pat)
    print(
        f"{len(gt)} analyzable groups (>=+{MIN_SP} species); scoring {len(args.layers)} layers",
        flush=True,
    )

    distances = ["geodesic", "angular"] if args.distance == "both" else [args.distance]
    for distance in distances:
        res, sweep_root = score_all_layers(stack, meta, gt, args.layers, args.arm, distance)
        suffix = "" if distance == "geodesic" else "_angular"
        res.to_csv(sweep_root / f"per_group_scores_{args.arm}{suffix}.csv", index=False)
        best = res.groupby("layer")["rho"].mean().idxmax()
        print(f"\n=== {distance}: mean within-group rho at best layer {best} ===")
        print(res[res.layer == best].groupby("baseline")["rho"].mean().round(3).to_string())
    print(f"per-layer run dirs -> {sweep_root}/blocks<L>/")


if __name__ == "__main__":
    main()
