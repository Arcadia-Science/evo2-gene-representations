"""Shared Evo2 model loading and all-layer position extraction."""

from __future__ import annotations

import numpy as np
import torch

MODEL_NAME = "evo2_7b"
N_BLOCKS = 32
LAYER_NAMES = [f"blocks.{i}" for i in range(N_BLOCKS)]


def load_model(model_name: str = MODEL_NAME):
    """Load an Evo2 checkpoint, downloading it on first use."""
    from evo2 import Evo2

    print(f"  Loading {model_name} (downloads on first run ~14 GB)...")
    return Evo2(model_name)


def forward_positions(
    seq: str,
    model,
    device: str,
    positions: list[int],
) -> dict[str, np.ndarray]:
    """Return all-block hidden states at selected zero-based sequence positions."""
    input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
    idx = torch.tensor(positions, dtype=torch.long, device=device)
    with torch.no_grad():
        _, embeddings = model(input_ids, return_embeddings=True, layer_names=LAYER_NAMES)
    out = {
        layer: embeddings[layer][0].float().index_select(0, idx).cpu().numpy().astype(np.float32)
        for layer in LAYER_NAMES
    }
    del input_ids, embeddings, idx
    torch.cuda.empty_cache()
    return out
