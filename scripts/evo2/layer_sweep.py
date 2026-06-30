"""Dense layer-stack embedder for the Evo2 gene-family layer-selection benchmark.

Evo2's `model(input_ids, return_embeddings=True, layer_names=[...])` captures ANY
number of residual-stream taps in a SINGLE forward pass (the layer_names just add
capture hooks). So — unlike a model where each layer costs its own forward pass —
there is no compute reason to subsample: we tap EVERY block (blocks.0 … blocks.31
for the 7B) and mean-pool each on the fly. This mirrors the GPN-Star sweep, which
already grabs all 17 `output_hidden_states` per forward pass.

Pooling matches the production gene-family embedder exactly (mean of the SECOND HALF
of the token positions; the low-left-context first half is dropped as autoregressive
burn-in — see evo2_embedding.pool_second_half), so the swept layers
are byte-for-byte the production embeddings — a built-in sanity check.

Output is a layer stack (n_layers, N, H) cached under data/cache/evo2_layer_sweep/,
consumed by scripts/layer_selection/layer_selection.py (the model-agnostic scoring engine).

Usage:
    uv run python scripts/evo2/layer_sweep.py
    uv run python scripts/evo2/layer_sweep.py --force-reembed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # evo2/ siblings (script dir)
# Embedding mechanics + block taxonomy come from evo2_embedding; the ortholog data loaders
# (manifest + FASTA) from the ortholog embedder. Bare-name imports so they don't collide
# with the installed `evo2` model package.
from evo2_embedding import MODEL_NAME, N_BLOCKS, embed_all_blocks, layer_type, load_model  # noqa: E402
from embed_and_geodesic_ortholog import MANIFEST_PATH, load_family_sequences  # noqa: E402

CACHE_DIR = Path("data/cache/evo2_layer_sweep")
STACK_PATH = CACHE_DIR / "layer_stack.npy"  # (n_layers, N, H)
META_PATH = CACHE_DIR / "metadata.csv"
CONFIG_PATH = CACHE_DIR / "config.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST_PATH)
    org_genes = manifest["org_gene"].tolist()
    N = len(manifest)
    print(f"  {N} CDS across {manifest['family'].nunique()} families")

    config = {
        "model": MODEL_NAME,
        "pool": "second_half",
        "n_blocks": N_BLOCKS,
        "org_genes": org_genes,
    }
    if (
        not args.force_reembed
        and STACK_PATH.exists()
        and CONFIG_PATH.exists()
        and json.loads(CONFIG_PATH.read_text()) == config
    ):
        print("  Reusing cached layer stack (config matches).")
        return

    seqs = load_family_sequences(manifest["family"].tolist())
    missing = [g for g in org_genes if g not in seqs]
    if missing:
        raise ValueError(f"{len(missing)} manifest org_genes missing from FASTA: {missing[:5]}")

    model = load_model()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stack = np.zeros((N_BLOCKS, N, 4096), dtype=np.float32)
    for i, og in enumerate(tqdm(org_genes, desc=f"Embedding CDS (all {N_BLOCKS} blocks)")):
        stack[:, i, :] = embed_all_blocks(seqs[og], model, device)
        if (i + 1) % args.checkpoint_every == 0:
            tmp = STACK_PATH.with_suffix(".tmp.npy")
            np.save(tmp, stack)
            tmp.replace(STACK_PATH)
            tqdm.write(f"    [checkpoint] {i + 1}/{N}")

    np.save(STACK_PATH, stack)
    manifest.to_csv(META_PATH, index=False)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"\n  Saved layer stack {stack.shape} -> {STACK_PATH}")
    print(f"  Layer types: {[layer_type(i) for i in range(N_BLOCKS)]}")


if __name__ == "__main__":
    main()
