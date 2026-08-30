"""Embed mammalian ortholog loci and assemble a cached layer stack for scoring."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))
from embed_and_geodesic_paralog import (  # noqa: E402  reuse the windowed Evo2 embedder + shuffles
    CONTROL_FNS, EMBED_DIM, EVO2_WINDOW, N_BLOCKS, embed_all_blocks, load_model,
)

# genomic-applicable composition controls (transcript loci have no reading frame, so codon_shuffle/
# synonymous_recode are excluded): gc_match, dinuc_shuffle, kmer4_shuffle, kmer6_shuffle.
GENOMIC_CONTROLS = ["gc_match", "dinuc_shuffle", "kmer4_shuffle", "kmer6_shuffle"]

MAX_WINDOWS = 24  # Bound long-locus cost by subsampling across the full span.
                  # Mean pooling gives each locus an equal-window estimator.
                  # 24 windows sample 192 kb; only about 3% of loci exceed
                  # 12 windows, so the added cost is modest.


def embed_capped(seq: str, model, device: str, window: int = EVO2_WINDOW,
                 max_windows: int = MAX_WINDOWS) -> np.ndarray:
    """(N_BLOCKS, 4096): second-half-pooled per window, averaged across windows. Contiguous tiling
    when it fits in max_windows; else evenly-spaced window centres across the locus."""
    L = len(seq)
    if L <= window:
        out = embed_all_blocks(seq, model, device)
        torch.cuda.empty_cache()
        return out
    n = math.ceil(L / window)
    if n <= max_windows:
        bounds = [(round(i * L / n), round((i + 1) * L / n)) for i in range(n)]
    else:
        centres = np.linspace(window // 2, L - window // 2, max_windows)
        bounds = [(int(c - window // 2), int(c + window // 2)) for c in centres]
    accum = np.zeros((N_BLOCKS, EMBED_DIM), dtype=np.float64)
    for a, b in bounds:
        accum += embed_all_blocks(seq[a:b], model, device)
        torch.cuda.empty_cache()
    return (accum / len(bounds)).astype(np.float32)

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE_ROOT = ROOT / "data" / "cache" / "mammal_embed"


def load_fastas(dataset: str, arm: str, families: list[str] | None):
    """-> list of (locus_id, family, group, species, seq)."""
    d = OUT / "seqs" / dataset / arm
    rows = []
    for fa in sorted(d.glob("*.fasta")):
        fam = fa.stem
        if families and fam not in families:
            continue
        hid, seq = None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if hid:
                    rows.append((hid, fam, *hid.split("|")[:2], "".join(seq)))
                hid, seq = line[1:], []
            elif line.strip():
                seq.append(line.strip())
        if hid:
            rows.append((hid, fam, *hid.split("|")[:2], "".join(seq)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["balanced_core", "complete", "or_track"])
    ap.add_argument("--arm", required=True, choices=["transcript", "cds"])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=EVO2_WINDOW)
    ap.add_argument("--control", default=None, choices=GENOMIC_CONTROLS,
                    help="Composition control: shuffle each locus before embedding (cache tagged).")
    args = ap.parse_args()

    rows = load_fastas(args.dataset, args.arm, args.families)
    # SHARED per-locus cache keyed by group__species (an embedding is dataset-independent), so
    # embedding 'complete' also serves balanced_core / or_track — no double work. Controls get a
    # <arm>_<control> cache dir.
    cache = CACHE_ROOT / (f"{args.arm}_{args.control}" if args.control else args.arm)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (cache / f"{r[2]}__{r[3]}.npy").exists()]
    print(f"{len(rows)} loci ({args.dataset}/{args.arm}, control={args.control}); "
          f"{len(todo)} to embed", flush=True)

    import random
    device = "cuda"
    model = load_model() if todo else None
    skipped = []
    for locus_id, fam, group, species, seq in tqdm(todo, desc="embed"):
        if args.control:  # deterministic per-locus composition shuffle
            rng = random.Random(f"{args.control}:{group}__{species}".__hash__() & 0xFFFFFFFF)
            seq = CONTROL_FNS[args.control](seq, rng)
        try:
            vec = embed_capped(seq, model, device, args.window)  # (N_BLOCKS, 4096), windows capped
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:  # one retry after clearing the allocator
                vec = embed_capped(seq, model, device, args.window)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped.append(locus_id)
                print(f"  OOM skip (not cached, retries next run): {locus_id} len={len(seq)}", flush=True)
                continue
        np.save(cache / f"{group}__{species}.npy", vec.astype(np.float32))
        torch.cuda.empty_cache()  # per-locus reset to curb fragmentation across the long run
    if skipped:
        print(f"OOM-skipped {len(skipped)} loci (rerun to retry): {skipped[:5]}", flush=True)

    # assemble stack + metadata (only loci with a cached embedding)
    meta, stack = [], []
    for locus_id, fam, group, species, seq in rows:
        p = cache / f"{group}__{species}.npy"
        if not p.exists():
            continue
        stack.append(np.load(p))
        meta.append({"locus_id": locus_id, "family": fam, "group": group, "species": species})
    arr = np.stack(stack, axis=1) if stack else np.zeros((N_BLOCKS, 0, EMBED_DIM), np.float32)
    # per-dataset assembled stack, drawn from the shared per-locus cache
    np.save(cache / f"{args.dataset}_layer_stack.npy", arr.astype(np.float32))
    pd.DataFrame(meta).to_csv(cache / f"{args.dataset}_metadata.csv", index=False)
    print(f"saved {args.dataset}_layer_stack {arr.shape} + metadata ({len(meta)} loci) -> {cache}",
          flush=True)


if __name__ == "__main__":
    main()
