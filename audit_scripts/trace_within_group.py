"""AUDIT (read-only): show exactly what mammal_score.py's within-group block does.

Prints, for one family at one layer:

  * every ortholog group in the family, its members, and whether MIN_SP kept or dropped it
  * for one chosen group: the Evo2 angular distance matrix and each baseline matrix
  * the upper-triangle vectors that go into scipy.stats.spearmanr, and the resulting rho
  * the per-group rho for every group, the family mean, and a diff against the PUBLISHED value
    in figure_data/exp1_within_family_by_layer.csv

    uv run python audit_scripts/trace_within_group.py                       # globins, layer 15
    uv run python audit_scripts/trace_within_group.py --family ferritin
    uv run python audit_scripts/trace_within_group.py --family globins --layer 3 --group NGB

Nothing is written. The only deviation from the production path is that the embedding cache is
loaded for ONE FAMILY instead of all 48 -- which is exact for `--distance angular`, because that
branch touches only `d["members"]` (mammal_score.py:120-122). It would NOT be exact for the
geodesic branch, which builds a k-NN graph over every locus in the panel (mammal_score.py:112-113).
"""

from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))

# The production code, imported -- not reimplemented.
from mammal_score import BASELINES, MIN_SP, gc_distance_matrix, load_cds  # noqa: E402
from kmer_sequence_divergence import kmer_distance_matrix  # noqa: E402
from protein_alignment_patristic import align_members, tree_patristic  # noqa: E402

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE = ROOT / "data" / "cache" / "mammal_embed" / "transcript_cdsmask"
PUBLISHED = ROOT / "figure_data" / "exp1_within_family_by_layer.csv"
# figure_data metric label -> the BASELINES key it came from
METRIC_OF = {
    "speciestree": "species tree (mammal)",
    "patristic": "patristic tree",
    "kmer": "k-mer composition (CDS, k=6)",
    "gc": "GC content (control)",
}
SHORT = {"homo_sapiens": "human"}


def short(sp: str) -> str:
    return SHORT.get(sp, sp[:3] + "_" + sp.split("_")[-1][:6])


def show(M: np.ndarray, labels: list[str], title: str) -> None:
    df = pd.DataFrame(np.round(M, 3), index=labels, columns=labels)
    print(f"\n  {title}   shape {M.shape}")
    print("  " + df.to_string().replace("\n", "\n  "))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="globins")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--group", default=None, help="which group to print matrices for")
    args = ap.parse_args()

    # ---- step 1: the manifest, and what `group` actually is -------------------------------
    man = pd.read_csv(OUT / "complete_manifest.csv")
    man["key"] = man["group"] + "__" + man["species"]          # mammal_score.py:49
    fam = man[man.family == args.family]
    if fam.empty:
        sys.exit(f"no family {args.family!r}; try one of {sorted(man.family.unique())}")

    print("=" * 100)
    print(f"FAMILY {args.family!r}   layer {args.layer}   arm transcript_cdsmask")
    print("=" * 100)
    print(f"\nmanifest rows for this family: {len(fam)}")
    print(f"distinct `group` values:       {fam.group.nunique()}   <- one per HUMAN PARALOG")
    print(f"distinct `species` values:     {fam.species.nunique()}")
    print("\n`group` is assigned at assemble_datasets.py:118 as  \"group\": r[\"human_gene\"]")
    print("so one group = one human paralog, and its rows are that gene across species:\n")
    print(fam.groupby("group").agg(n_species=("species", "nunique")).sort_values(
        "n_species", ascending=False).to_string())

    # ---- step 2: keep only what was actually embedded (mammal_score.py:51-55) --------------
    fam = fam[[(CACHE / f"{k}.npy").exists() for k in fam.key]].reset_index(drop=True)
    vec = {k: np.load(CACHE / f"{k}.npy")[args.layer] for k in fam.key}
    print(f"\nembedded loci for this family: {len(fam)}  (rows without a cached .npy are skipped)")

    # ---- step 3: MIN_SP, exactly as mammal_score.py:82-86 ---------------------------------
    pat = pd.read_csv(OUT / "tree" / "species_patristic.csv", index_col=0)
    cds = load_cds()
    print(f"\n--- MIN_SP = {MIN_SP} filter (mammal_score.py:85) ---")
    gt: dict[str, dict] = {}
    for (f_, group), sub in fam.groupby(["family", "group"]):        # mammal_score.py:82
        members = sub["key"].tolist()                                # mammal_score.py:83
        species = sub["species"].tolist()                            # mammal_score.py:84
        if len(members) < MIN_SP:                                    # mammal_score.py:85-86
            print(f"  DROP  {group:<8} n={len(members):>2}  {[short(s) for s in species]}")
            continue
        d = {
            "family": f_,
            "members": members,
            "species": species,
            "n": len(members),
            "speciestree": pat.loc[species, species].values,         # mammal_score.py:92
            "kmer": kmer_distance_matrix([cds[m] for m in members], k=6),   # :93
            "gc": gc_distance_matrix([cds[m] for m in members]),            # :94
        }
        with tempfile.TemporaryDirectory() as tmp:                   # mammal_score.py:96-101
            res = align_members(members, {m: cds[m] for m in members}, Path(tmp))
            if res:
                tp = tree_patristic(res, members, Path(tmp))
                if tp is not None:
                    d["patristic"] = tp
        gt[group] = d
        miss = [b for b in BASELINES if b not in d]
        print(f"  KEEP  {group:<8} n={d['n']:>2}"
              + (f"   (no {miss} baseline)" if miss else ""))

    if not gt:
        sys.exit("no group in this family survived MIN_SP")

    # ---- step 4: the matrices for one group ----------------------------------------------
    pick = args.group or min(gt, key=lambda g: gt[g]["n"])
    if pick not in gt:
        sys.exit(f"group {pick!r} not kept; kept groups are {sorted(gt)}")
    d = gt[pick]
    lab = [short(s) for s in d["species"]]
    print("\n" + "=" * 100)
    print(f"GROUP {pick!r}  --  n = {d['n']} species. Every matrix below is species x species,")
    print("for this ONE gene. No paralog of the family enters it.")
    print("=" * 100)
    print(f"\n  members (the `key` column, gene__species):\n    " + "\n    ".join(d["members"]))

    V = np.stack([vec[m] for m in d["members"]])                     # mammal_score.py:120
    U = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)   # :121
    G = np.arccos(np.clip(U @ U.T, -1.0, 1.0)) / np.pi               # :122
    show(G, lab, "Evo2 angular distance  D^G = arccos(clip(U @ U.T))/pi   (mammal_score.py:120-122)")
    for b in BASELINES:
        if b in d:
            show(d[b], lab, f"baseline {b!r}")

    # ---- step 5: upper triangle -> spearman (mammal_score.py:123-133) ----------------------
    iu = np.triu_indices(d["n"], 1)                                  # mammal_score.py:123
    print(f"\n  upper triangle (mammal_score.py:123): {len(iu[0])} unique pairs "
          f"= n(n-1)/2 = {d['n']}*{d['n'] - 1}/2")
    print("\n  the vectors handed to scipy.stats.spearmanr, first 8 pairs:")
    head = pd.DataFrame({"pair": [f"{lab[i]} x {lab[j]}" for i, j in zip(*iu)][:8],
                         "evo2_angular": np.round(G[iu][:8], 4)})
    for b in BASELINES:
        if b in d:
            head[b] = np.round(d[b][iu][:8], 4)
    print("  " + head.to_string(index=False).replace("\n", "\n  "))

    print(f"\n  --- rho for group {pick!r} ---")
    for b in BASELINES:
        if b not in d:
            print(f"    {b:<12}  no baseline matrix -> this (group, baseline) cell is skipped")
            continue
        g_, b_ = G[iu], d[b][iu]
        ok = np.isfinite(g_) & np.isfinite(b_)                       # mammal_score.py:128
        gate = ok.sum() >= 6 and np.ptp(g_[ok]) > 0 and np.ptp(b_[ok]) > 0   # :129
        if not gate:
            print(f"    {b:<12}  FILTERED OUT (mammal_score.py:129): "
                  f"usable pairs={ok.sum()}, ptp_evo2={np.ptp(g_[ok]):.3g}, ptp_base={np.ptp(b_[ok]):.3g}")
            continue
        print(f"    {b:<12}  usable pairs={ok.sum():>3}   rho = {spearmanr(g_[ok], b_[ok]).statistic:+.6f}")

    # ---- step 6: every group, then the family mean, vs the published value ------------------
    print("\n" + "=" * 100)
    print("PER-GROUP rho, THEN THE FAMILY MEAN  (mammal_score.py:130-139)")
    print("=" * 100)
    rows = []
    for group, dd in gt.items():
        Vg = np.stack([vec[m] for m in dd["members"]])
        Ug = Vg / np.clip(np.linalg.norm(Vg, axis=1, keepdims=True), 1e-12, None)
        Gg = np.arccos(np.clip(Ug @ Ug.T, -1.0, 1.0)) / np.pi
        iug = np.triu_indices(dd["n"], 1)
        for b in BASELINES:
            if b not in dd:
                continue
            g_, b_ = Gg[iug], dd[b][iug]
            ok = np.isfinite(g_) & np.isfinite(b_)
            if ok.sum() >= 6 and np.ptp(g_[ok]) > 0 and np.ptp(b_[ok]) > 0:
                rows.append({"group": group, "n_species": dd["n"], "baseline": b,
                             "rho": spearmanr(g_[ok], b_[ok]).statistic})
    per = pd.DataFrame(rows)
    print("\n" + per.pivot(index=["group", "n_species"], columns="baseline",
                           values="rho").round(4).to_string())

    mean = per.groupby("baseline")["rho"].mean()                     # mammal_score.py:139
    pub = pd.read_csv(PUBLISHED).query("layer == @args.layer and family == @args.family")
    print("\n  family mean of the per-group rho, vs the published table:\n")
    print(f"    {'baseline':<12} {'n_groups':>8} {'recomputed':>12} {'published':>12} {'diff':>12}")
    for b in BASELINES:
        if b not in set(per.baseline):
            print(f"    {b:<12} {0:>8} {'-':>12} {'(absent)':>12} {'-':>12}")
            continue
        got = float(mean[b])
        hit = pub[pub.metric == METRIC_OF[b]]["rho"]
        want = float(hit.iloc[0]) if len(hit) else np.nan
        n = int((per.baseline == b).sum())
        print(f"    {b:<12} {n:>8} {got:>12.6f} {want:>12.6f} {got - want:>12.3e}")
    print(f"\n  published rows read from {PUBLISHED.relative_to(ROOT)}")
    print("  (layer, metric, family, rho) -- the same table figures 2 and 3 plot.\n")


if __name__ == "__main__":
    main()
