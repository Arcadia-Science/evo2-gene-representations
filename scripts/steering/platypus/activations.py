"""Stage 1b — extract Evo2 residual-stream activations at the FINAL position of each 90-bp prefix."""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

import steer_lib as S  # noqa: E402

PREFIX_BP = 90
BLOCKS = [f"blocks.{i}" for i in range(32)]
SPECIES = ["human", "platypus"]  # column order of the saved tensor
BASES = set("ACGT")


def _verify_tap_matches_hook(model, seq: str, layer: str = "blocks.20") -> float:
    """Assert `layer_names=` returns the same tensor a forward hook on that block sees."""
    import torch

    captured = {}

    def hook(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy()

    handle = model.model.get_submodule(layer).register_forward_hook(hook)
    try:
        ids = S._tokenize(model, seq, "cuda")
        with torch.no_grad():
            _logits, emb = model(ids, return_embeddings=True, layer_names=[layer])
        tapped = emb[layer].detach().float().cpu().numpy()
    finally:
        handle.remove()
    return float(np.abs(captured["h"] - tapped).max())


def _assert_no_hooks(model) -> None:
    """No forward hooks anywhere on the blocks: activations must be captured un-steered."""
    for name in BLOCKS:
        mod = model.model.get_submodule(name)
        n = len(getattr(mod, "_forward_hooks", {})) + len(getattr(mod, "_forward_pre_hooks", {}))
        if n:
            raise AssertionError(f"{name} has {n} hook(s) registered during extraction")


def extract(pairs: list[dict], out_dir: Path) -> None:
    import torch

    model = S.load_model()
    _assert_no_hooks(model)

    probe = pairs[0]["prefix_human"]
    tap_diff = _verify_tap_matches_hook(model, probe)
    ids_probe = S._tokenize(model, probe, "cuda")
    n_tok = int(ids_probe.shape[-1])
    if n_tok != PREFIX_BP:
        raise AssertionError(
            f"tokenizer emitted {n_tok} tokens for a {PREFIX_BP}-bp prefix (BOS/EOS?); "
            "the final-position index assumption is invalid")
    _assert_no_hooks(model)

    genes = [r["gene"] for r in pairs]
    acts: np.ndarray | None = None
    norms: list[dict] = []

    for gi, r in enumerate(pairs):
        for si, sp in enumerate(SPECIES):
            seq = r[f"prefix_{sp}"]
            if len(seq) != PREFIX_BP or set(seq) - BASES:
                raise AssertionError(f"{r['gene']}/{sp}: bad prefix {seq!r}")
            ids = S._tokenize(model, seq, "cuda")
            assert ids.shape[0] == 1 and ids.shape[-1] == PREFIX_BP, ids.shape
            with torch.no_grad():
                _logits, emb = model(ids, return_embeddings=True, layer_names=BLOCKS)
            for li, layer in enumerate(BLOCKS):
                h = emb[layer][0].float().cpu().numpy()  # (Lseq, H)
                assert h.shape[0] == PREFIX_BP, (layer, h.shape)
                vec = S.pool(h, "last")  # final prompt position, index 89
                if acts is None:
                    shape = (len(pairs), len(SPECIES), len(BLOCKS), vec.shape[0])
                    acts = np.zeros(shape, np.float32)
                acts[gi, si, li] = vec
                norms.append({"gene": r["gene"], "family": r["family"], "species": sp,
                              "layer": li, "norm": float(np.linalg.norm(vec))})
            del emb
        if (gi + 1) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"  {gi + 1}/{len(pairs)}", flush=True)

    assert acts is not None
    # every (gene, species, layer) slice has identical shape by construction; assert no NaN/Inf
    if not np.isfinite(acts).all():
        bad = np.argwhere(~np.isfinite(acts).all(axis=-1))
        raise AssertionError(f"non-finite activations at {len(bad)} (gene,species,layer) slices")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "activations_last_pos.npz",
        acts=acts,
        genes=np.array(genes),
        families=np.array([r["family"] for r in pairs]),
        species=np.array(SPECIES),
        layers=np.array(BLOCKS),
    )
    with open(out_dir / "activation_norms.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gene", "family", "species", "layer", "norm"])
        w.writeheader()
        w.writerows(norms)

    (out_dir / "activation_validation.json").write_text(json.dumps({
        "n_genes": len(genes),
        "shape": list(acts.shape),
        "shape_meaning": "[gene, species(human,platypus), layer(blocks.0-31), hidden]",
        "hidden_dim": int(acts.shape[-1]),
        "prefix_bp": PREFIX_BP,
        "tokens_per_prefix": n_tok,
        "tokenizer_prepends_bos": n_tok != PREFIX_BP,
        "final_position_index": n_tok - 1,
        "pool_mode": "last",
        "batch_size": 1,
        "species_share_a_batch": False,
        "hooks_registered_during_extraction": 0,
        "layer_names_vs_hook_max_abs_diff": tap_diff,
        "all_finite": True,
    }, indent=2))
    print(f"acts {acts.shape} -> {out_dir}/activations_last_pos.npz")
    print(f"layer_names vs hook max|diff| = {tap_diff:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-dir", type=Path,
                    default=ROOT / "results" / "2026-07-28_evo2-platypus-paired" / "stage1")
    args = ap.parse_args()
    with open(args.stage1_dir / "pairs.csv") as fh:
        pairs = list(csv.DictReader(fh))
    print(f"{len(pairs)} retained pairs -> {2 * len(pairs)} forwards x {len(BLOCKS)} layers")
    extract(pairs, args.stage1_dir)


if __name__ == "__main__":
    main()
