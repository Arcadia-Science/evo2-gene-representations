"""Condition C — embed the Evo2-human GENOMIC TRANSCRIPT SPAN, pool only the CDS positions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))
from evo2_embedding import LAYER_NAMES, load_model, MODEL_NAME  # noqa: E402
from embed_and_geodesic_paralog import N_BLOCKS, EMBED_DIM, EVO2_WINDOW  # noqa: E402

GENOMIC_CACHE = ROOT / "data" / "cache" / "evo2_human_genomic.json"
CDS_POS = ROOT / "data" / "cache" / "human_cds_positions.json"
# `--meta-src` selects the gene set and ordering for this arm.
META_SRC = ROOT / "results" / "2026-07-01_evo2-human-panel" / "blocks15" / "metadata.csv"
CACHE_DIR = ROOT / "data" / "cache" / "evo2_human_cdspool_transcript"


def _forward_positions(seq: str, model, device: str, want_local: list[int]) -> dict[str, np.ndarray]:
    """Forward over `seq`; return {layer: (len(want_local), H)} hidden states at the given 0-based
    positions, in the given order."""
    input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
    idx = torch.tensor(want_local, dtype=torch.long, device=device)
    with torch.no_grad():
        _, emb = model(input_ids, return_embeddings=True, layer_names=list(LAYER_NAMES))
    out = {ln: emb[ln][0].float().index_select(0, idx).cpu().numpy().astype(np.float32)
           for ln in LAYER_NAMES}
    del input_ids, emb, idx
    torch.cuda.empty_cache()
    return out


def embed_cds_masked(seq: str, cds_pos: list[int], model, device: str,
                     window: int = EVO2_WINDOW) -> np.ndarray:
    """(N_BLOCKS, H): hidden states at CDS positions (transcript order) across contiguous windows, second half mean-pooled."""
    L = len(seq)
    cds = np.array(sorted(p for p in cds_pos if 0 <= p < L))
    if L <= window:
        bounds = [(0, L)]
    else:
        n = math.ceil(L / window)
        bounds = [(round(i * L / n), round((i + 1) * L / n)) for i in range(n)]
    collected = {ln: [] for ln in LAYER_NAMES}
    for a, b in bounds:
        in_win = cds[(cds >= a) & (cds < b)]
        if len(in_win) == 0:
            continue
        got = _forward_positions(seq[a:b], model, device, (in_win - a).tolist())
        for ln in LAYER_NAMES:
            collected[ln].append(got[ln])
    out = np.zeros((N_BLOCKS, EMBED_DIM), dtype=np.float32)
    for i, ln in enumerate(LAYER_NAMES):
        allpos = np.concatenate(collected[ln], axis=0)  # (n_cds, H), transcript order
        half = len(allpos) // 2                          # pool_second_half rule, on CDS positions
        out[i] = allpos[half:].mean(axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--meta-src", default=str(META_SRC),
                    help="metadata.csv whose gene column defines this arm's gene set")
    args = ap.parse_args()

    genomic = json.loads(GENOMIC_CACHE.read_text())
    cds_pos = json.loads(CDS_POS.read_text())
    meta_src = pd.read_csv(args.meta_src)  # gene, family — same set/order as the transcript run (B)

    # keep the transcript run's gene order, restricted to genes with a validated CDS mask
    rows = [(g, f) for g, f in zip(meta_src["gene"].astype(str), meta_src["family"])
            if g in cds_pos and g in genomic]
    dropped = [g for g in meta_src["gene"].astype(str) if g not in cds_pos]
    genes = [g for g, _ in rows]
    fams = [f for _, f in rows]
    print(f"{len(genes)} genes with validated CDS masks "
          f"({len(dropped)} dropped, no/failed mask): {dropped[:8]}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stack_path = CACHE_DIR / "layer_stack.npy"
    config = {"model": MODEL_NAME, "pool": "second_half_of_cds", "input": "transcript_span",
              "n_blocks": N_BLOCKS, "window": EVO2_WINDOW, "genes": genes}
    cfg_path = CACHE_DIR / "config.json"
    if not args.force and stack_path.exists() and cfg_path.exists() \
            and json.loads(cfg_path.read_text()) == config:
        print("  Reusing cached CDS-masked stack (config matches).")
        return

    model = load_model()
    device = "cuda"
    stack = np.zeros((N_BLOCKS, len(genes), EMBED_DIM), dtype=np.float32)
    for i, g in enumerate(tqdm(genes, desc="CDS-masked transcript embed")):
        stack[:, i, :] = embed_cds_masked(genomic[g], cds_pos[g], model, device)
        if (i + 1) % args.checkpoint_every == 0:
            tmp = stack_path.with_suffix(".tmp.npy")
            np.save(tmp, stack); tmp.replace(stack_path)
            tqdm.write(f"    [checkpoint] {i + 1}/{len(genes)}")
    np.save(stack_path, stack)
    pd.DataFrame({"gene": genes, "family": fams}).to_csv(CACHE_DIR / "metadata.csv", index=False)
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"  Saved CDS-masked layer stack {stack.shape} -> {stack_path}", flush=True)


if __name__ == "__main__":
    main()
