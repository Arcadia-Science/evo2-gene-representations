"""Stratified per-paralog cap of the mammalian ortholog manifest to <=CAP loci per family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "mammalian_orthologs"


def even_alloc(sizes: dict[str, int], budget: int, rng: np.random.Generator) -> dict[str, int]:
    """Water-fill `budget` units across groups (key -> size) as evenly as possible, each capped by its size."""
    alloc = {g: 0 for g in sizes}
    order = [g for g in sizes if sizes[g] > 0]
    rng.shuffle(order)
    open_groups = list(order)
    remaining = budget
    while remaining > 0 and open_groups:
        share = remaining // len(open_groups)
        if share == 0:                       # fewer slots left than open groups: 1 each until dry
            for g in open_groups[:remaining]:
                alloc[g] += 1
            break
        still = []
        for g in open_groups:
            take = min(share, sizes[g] - alloc[g])
            alloc[g] += take
            remaining -= take
            if alloc[g] < sizes[g]:
                still.append(g)
        open_groups = still
    return alloc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="complete_manifest.csv")
    args = ap.parse_args()

    man = pd.read_csv(OUT / args.manifest)
    man["key"] = man.group + "__" + man.species

    keep: list[int] = []
    report = []
    for fi, (fam, fdf) in enumerate(man.groupby("family")):   # groupby sorts keys -> deterministic
        n = len(fdf)
        if n <= args.cap:
            keep.extend(fdf.index.tolist())
            report.append((fam, fdf.group.nunique(), n, n))
            continue
        rng = np.random.default_rng(args.seed * 100_000 + fi)  # per-family, reproducible
        sizes = fdf.groupby("group").size().to_dict()
        alloc = even_alloc(sizes, args.cap, rng)
        for g, k in alloc.items():
            gidx = fdf[fdf.group == g].index.to_numpy()
            if k >= len(gidx):
                keep.extend(gidx.tolist())
            elif k > 0:
                keep.extend(rng.choice(gidx, size=k, replace=False).tolist())
        report.append((fam, fdf.group.nunique(), n, sum(alloc.values())))

    capped = man.loc[sorted(keep)].reset_index(drop=True)
    out = OUT / f"complete_manifest_cap{args.cap}.csv"
    capped.to_csv(out, index=False)

    rep = pd.DataFrame(report, columns=["family", "paralogs", "loci_full", "loci_capped"])
    rep = rep.sort_values("loci_full", ascending=False)
    print(rep.to_string(index=False))
    print(f"\ntotal loci: {len(man)} -> {len(capped)}   (cap={args.cap}, seed={args.seed})")
    print(f"written {out}")


if __name__ == "__main__":
    main()
