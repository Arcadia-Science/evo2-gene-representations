"""Shared Evo2 embedding engine for the gene-panel analyses (paralog / ortholog + the layer sweep).

One place for "load Evo2 → tokenize → return_embeddings forward → second-half mean-pool", so the
three gene-panel scripts don't each re-implement it (and don't import primitives from each other):

  * embed_and_geodesic_ortholog.py  — one KEGG CDS per row (cross-kingdom)
  * embed_and_geodesic_paralog.py   — GRCh38 genomic string per human gene (matched panel)
  * layer_sweep.py                  — all 32 blocks at once, for layer selection

It also holds the Evo2-7B block taxonomy (N_BLOCKS / layer_type / LAYER_NAMES) the sweep + paralog
embedder share. This is the Evo2 analog of scripts/geodesic_utils.py — pure embedding mechanics, no
analysis logic.

NOTE: the SPECIES pipeline (embed_and_geodesic_species.py) deliberately does NOT use this engine —
it loads a different checkpoint (evo2_7b_262k, to match Goodfire's weights) and pools the tail of a
fixed 4096-bp window, so it keeps its own embedding code.
"""

from __future__ import annotations

import numpy as np
import torch

# 1M-context 7B StripedHyena — the gene-panel default (the species pipeline uses evo2_7b_262k).
MODEL_NAME = "evo2_7b"
# Residual-stream tap leaving block 24 — the production default (validated on the species pipeline:
# lifts geodesic-vs-phylo Pearson ~0.36 → ~0.72 vs the MLP-delta tap).
EMBED_LAYER = "blocks.24"

# Evo2 7B has 32 blocks (0..31). Layer types from configs/evo2-7b-8k.yml:
#   attn: 3,10,17,24,31   hcl: 2,6,9,13,16,20,23,27,30
#   hcm:  1,5,8,12,15,19,22,26,29   hcs: 0,4,7,11,14,18,21,25,28
N_BLOCKS = 32
ATTN_IDX = {3, 10, 17, 24, 31}
HCL_IDX = {2, 6, 9, 13, 16, 20, 23, 27, 30}
HCM_IDX = {1, 5, 8, 12, 15, 19, 22, 26, 29}
HCS_IDX = {0, 4, 7, 11, 14, 18, 21, 25, 28}
LAYER_NAMES = [f"blocks.{i}" for i in range(N_BLOCKS)]


def layer_type(i: int) -> str:
    if i in ATTN_IDX:
        return "attn"
    if i in HCL_IDX:
        return "hyena-long"
    if i in HCM_IDX:
        return "hyena-med"
    if i in HCS_IDX:
        return "hyena-short"
    return "other"


def load_model(model_name: str = MODEL_NAME):
    """Lazily import + load an Evo2 checkpoint (the ~14 GB download happens on first run)."""
    from evo2 import Evo2  # local import: the heavy model package is only needed when embedding

    print(f"  Loading {model_name} (downloads on first run ~14 GB)...")
    return Evo2(model_name)


def pool_second_half(win_emb):
    """Mean-pool the SECOND HALF of the token positions of a (L, H) hidden-state tensor.

    Evo2 is autoregressive (causal): a position's hidden state has attended over every earlier
    token, so by the back half of the sequence the representation is well past the autoregressive
    burn-in — the gene's identity is "burned in" — regardless of how long the gene is. Pooling the
    second half drops the low-left-context first half while keeping a length-proportional,
    length-consistent window for every gene. The WHOLE sequence is still fed to the model; this only
    selects which positions' hidden states are averaged, so no input sequence is truncated.
    """
    half = win_emb.shape[0] // 2
    return win_emb[half:].mean(dim=0)


def _forward_pool(seq: str, model, device: str, layer_names: list[str]) -> dict[str, np.ndarray]:
    """One forward pass over `seq`; return {layer_name: second-half-pooled (H,) float32 vector}."""
    input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
    with torch.no_grad():
        _, emb = model(input_ids, return_embeddings=True, layer_names=list(layer_names))
    out = {ln: pool_second_half(emb[ln][0].float()).cpu().numpy().astype(np.float32) for ln in layer_names}
    del input_ids, emb
    return out


def embed_one(seq: str, model, device: str, layer: str = EMBED_LAYER) -> np.ndarray:
    """Embed one sequence at a single residual-stream tap → (H,) second-half-pooled vector."""
    return _forward_pool(seq, model, device, [layer])[layer]


def embed_all_blocks(seq: str, model, device: str) -> np.ndarray:
    """Embed one sequence at EVERY block in a single forward → (N_BLOCKS, H), second-half pooled."""
    pooled = _forward_pool(seq, model, device, LAYER_NAMES)
    return np.stack([pooled[ln] for ln in LAYER_NAMES], axis=0)
