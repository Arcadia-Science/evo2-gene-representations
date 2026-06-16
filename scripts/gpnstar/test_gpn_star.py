"""
GPN-Star model verification script.

Two-stage test:
  1. Synthetic sanity check — loads both 200M models, runs a forward pass with
     random MSA tokens, and validates output shapes/values. No alignment data needed.
  2. Real VEP benchmark — runs variant effect prediction on songlab/clinvar_vs_benign,
     scores variants, and reports AUROC against pathogenic/benign labels.
     Requires multiz100way zarr alignment data (see --help for download instructions).

Usage:
    # Stage 1 only (synthetic check):
    uv run python scripts/test_gpn_star.py

    # Stage 1 + 2 (needs zarr data, see MSA DATA DOWNLOAD NOTES below):
    uv run python scripts/test_gpn_star.py --vep
    uv run python scripts/test_gpn_star.py --vep --alignments vertebrate
    uv run python scripts/test_gpn_star.py --vep --alignments vertebrate mammalian
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

import gpn.star.model  # registers GPNStar with AutoModel/AutoConfig
from transformers import AutoConfig, AutoModelForMaskedLM

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "vertebrate": {
        "hf_id": "songlab/gpn-star-hg38-v100-200m",
        "n_species": 100,
        "msa_path": "data/multiz100way.zarr",
    },
    "mammalian": {
        "hf_id": "songlab/gpn-star-hg38-m447-200m",
        "n_species": 447,
        "msa_path": "data/multiz447way.zarr",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def download_model(hf_id: str, cache_dir: Path) -> Path:
    print(f"    Downloading {hf_id} (cached after first run)...")
    local_path = snapshot_download(
        repo_id=hf_id,
        cache_dir=str(cache_dir),
        # Skip non-PyTorch formats to save bandwidth
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    return Path(local_path)


def load_model_compat(model_path: Path):
    """Load a GPN-Star model, working around transformers 5.x meta-device init.

    Adapted from load_model() in gpn/star/model.py (https://github.com/songlab-cal/gpn),
    originally by Gonzalo Benegas et al., MIT License.

    Transformers >=4.44 calls the model constructor inside accelerate's
    init_empty_weights() context, which intercepts ALL tensor creation
    (including torch.tensor(np_array)) and places tensors on the meta device.
    GPNStarPhyloInfo.__init__ loads numpy arrays and calls .item() on them,
    which raises 'Tensor.item() cannot be called on meta tensors'.

    Fix: instantiate the model directly (outside of from_pretrained's context)
    and then load the safetensors weights manually.

    TODO: verify this approach
    """
    import os

    from safetensors.torch import load_file

    from gpn.star.model import GPNStarForMaskedLM

    config = AutoConfig.from_pretrained(str(model_path))

    # The checkpoint stores phylo_dist_path as an absolute path from the
    # original training machine; fall back to the local copy.
    if not os.path.exists(config.phylo_dist_path):
        fallback = os.path.join(str(model_path), "phylo_dist")
        if os.path.exists(fallback):
            config.phylo_dist_path = fallback
        else:
            raise FileNotFoundError(
                f"phylo_dist not found at '{config.phylo_dist_path}' "
                f"or fallback '{fallback}'"
            )

    # Direct instantiation avoids the meta-device context in from_pretrained
    model = GPNStarForMaskedLM(config) 

    weights_path = os.path.join(str(model_path), "model.safetensors")
    state_dict = load_file(weights_path)

    # cls.predictions.decoder.bias / .weight are tied to input embeddings and
    # are not stored separately in the checkpoint. Use strict=False and assert
    # that no unexpected keys are missing beyond the declared tied set.
    load_result = model.load_state_dict(state_dict, strict=False)
    tied = set(model._tied_weights_keys)
    unexpected_missing = [k for k in load_result.missing_keys if k not in tied]
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing keys after load: {unexpected_missing}")

    # Restore tied weights (decoder bias <- decoder weight bias; decoder weight <- embeddings)
    model.tie_weights()

    return model


# ── Stage 1: synthetic forward pass ──────────────────────────────────────────


def run_synthetic_check(name: str, cfg: dict, model_dir: Path) -> bool:
    """
    Load the model, run a forward pass with random MSA tokens, and check outputs.
    """
    hf_id, n_species = cfg["hf_id"], cfg["n_species"]
    print(f"  Synthetic check: {name}  ({hf_id})")
    print(f"{'─'*60}")

    local_path = download_model(hf_id, model_dir)

    print("    Loading model weights...")
    model = load_model_compat(local_path)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    cfg_obj = model.config
    # TODO: remove these print statements after checking
    print(f"    Parameters : {n_params:,}")
    print(
        f"    Architecture: hidden={cfg_obj.hidden_size}, "
        f"layers={cfg_obj.num_hidden_layers}, "
        f"heads={cfg_obj.num_attention_heads}, "
        f"max_pos={cfg_obj.max_position_embeddings}"
    )

    # Build synthetic batch.
    # Vocab: -=0  A=1  C=2  G=3  T=4  ?=5
    # batch size B, sequence length L, target species T, total species N
    # input_ids  : (B, L, T)  — target species tokens (mask the variant position in real use)
    # source_ids : (B, L, N)  — all-species MSA tokens
    # target_species : (B, T) — which species index is the target (0 = hg38)
    B, L, T = 2, 64, 1
    N = n_species
    rng = np.random.default_rng(42)
    input_ids = torch.from_numpy(rng.integers(1, 5, size=(B, L, T))).long()
    source_ids = torch.from_numpy(rng.integers(1, 5, size=(B, L, N))).long()
    target_species = torch.zeros(B, T, dtype=torch.long) # target species index 0 = hg38 (human reference)

    print(
        f"    Batch shapes: input_ids={tuple(input_ids.shape)}, "
        f"source_ids={tuple(source_ids.shape)}, "
        f"target_species={tuple(target_species.shape)}"
    )

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            source_ids=source_ids,
            target_species=target_species,
        )

    logits = out.logits  # expected: (B, L, T, vocab_size)
    vocab_size = cfg_obj.vocab_size

    # Shape check
    expected_shape = (B, L, T, vocab_size)
    assert logits.shape == expected_shape, (
        f"Unexpected logits shape: got {tuple(logits.shape)}, expected {expected_shape}"
    )

    # Numerical sanity
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf — model may not have loaded correctly"

    # Spot-check: nucleotide probabilities at the centre position of the first example
    centre = L // 2
    acgt_probs = logits[0, centre, 0, 1:5].softmax(dim=-1).numpy()
    print(f"    Output shape : {tuple(logits.shape)}  ✓")
    print(f"    ACGT probs at centre (example 0): {np.round(acgt_probs, 3)}")
    print(f"    All finite   : {torch.isfinite(logits).all().item()}  ✓")
    print(f"  PASSED")
    return True


# ── Stage 2: real VEP benchmark ───────────────────────────────────────────────


def run_vep_benchmark(name: str, cfg: dict, msa_path: str, model_dir: Path, window_size: int = 512, chrom: str | None = None) -> None:
    """Run VEP on songlab/clinvar_vs_benign and report AUROC.

    Adapted from gpn/star/vep.py in the GPN-Star repository (https://github.com/songlab-cal/gpn), 
    originally by Gonzalo Benegas et al., MIT License.
    """
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score

    from gpn.star.data import GenomeMSA
    from gpn.star.vep import VEPInference
    from gpn.star.inference import run_inference

    hf_id = cfg["hf_id"]

    print(f"\n{'─'*60}")
    print(f"  VEP benchmark: {name}")
    print(f"  Dataset       : songlab/clinvar_vs_benign")
    print(f"  MSA path      : {msa_path}")
    print(f"  Window size   : {window_size} bp")
    print(f"{'─'*60}")

    print("    Loading alignment (zarr)...")
    genome_msa = GenomeMSA(msa_path, n_species=cfg["n_species"])

    local_path = download_model(hf_id, model_dir)
    # VEPInference calls AutoModelForMaskedLM.from_pretrained internally;
    # wrap in MLMforVEPModel manually to use our compat loader instead.
    from gpn.star.vep import MLMforVEPModel

    vep_model = MLMforVEPModel.__new__(MLMforVEPModel)
    torch.nn.Module.__init__(vep_model)
    vep_model.model = load_model_compat(local_path)
    vep_model.model.eval()

    inference = VEPInference.__new__(VEPInference)
    from gpn.data import Tokenizer, ReverseComplementer
    # TODO: is there some check we can do to verify that the model loading workaround is correct? 

    inference.model = vep_model
    inference.genome_msa_list = [genome_msa]
    inference.window_size = window_size
    inference.disable_aux_features = False   
    inference.reverse_complementer = ReverseComplementer()
    inference.tokenizer = Tokenizer()

    print("    Loading clinvar_vs_benign dataset...")
    dataset = load_dataset("songlab/clinvar_vs_benign", split="test")
    if chrom is not None:
        dataset = dataset.filter(lambda x: x["chrom"] == chrom)
        print(f"    Filtered to chrom {chrom}: {len(dataset)} variants")
    else:
        print(f"    Variants: {len(dataset)}")

    labels = np.array(dataset["label"])  # extract before Trainer wraps the dataset
    os.environ["WANDB_DISABLED"] = "true"
    print("    Running VEP inference...")
    scores_df = run_inference(dataset, inference, per_device_batch_size=32)
    # LLR convention is log P(alt) - log P(ref); negate so higher = more pathogenic
    auroc = roc_auc_score(labels, -scores_df["score"])
    print(f"\n    AUROC: {auroc:.4f}")
    print(f"    (Published GPN-Star v100 AUROC on clinvar_vs_benign: ~0.89–0.91)")

    scores_df["label"] = labels
    date_str = datetime.date.today().isoformat()
    out_dir = Path("results") / f"{date_str}_gpnstar-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vep_scores.csv"
    scores_df.to_csv(out_path, index=False)
    print(f"    Scores saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vep",
        action="store_true",
        help="Also run Stage 2: real VEP benchmark (requires --msa-path)",
    )
    p.add_argument(
        "--alignments",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="Which alignment(s) to benchmark in Stage 2: vertebrate, mammalian, or both (default: both).",
    )
    p.add_argument(
        "--models-dir",
        default="models",
        help="Directory to cache HuggingFace model downloads (default: models/)",
    )
    p.add_argument(
        "--chrom",
        default=None,
        help="Restrict VEP benchmark to a single chromosome (e.g. 22). ",
    )
    p.add_argument(
        "--synthetic-only",
        dest="synthetic_models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="Run synthetic check on a subset of models",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.models_dir)
    model_dir.mkdir(exist_ok=True)

    print("\n== Testing GPN-Star model ==")

    # Stage 1
    print("\n[Stage 1] Synthetic forward-pass checks")
    passed, failed = [], []
    for name in args.synthetic_models:
        try:
            run_synthetic_check(name, MODELS[name], model_dir)
            passed.append(name)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    print(f"\n  Stage 1 summary: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed models: {failed}")
        sys.exit(1)

    # ── Stage 2 ──
    if args.vep:
        print("\n[Stage 2] Real VEP benchmark")
        vep_passed, vep_failed = [], []
        for name in args.alignments:
            cfg = MODELS[name]
            msa_path = cfg["msa_path"]
            if not Path(msa_path).exists():
                print(
                    f"\n  SKIPPED {name}: MSA not found at {msa_path}\n"
                    f"  Download with: python scripts/download_msa.py --out {msa_path}"
                )
                vep_failed.append(name)
                continue
            try:
                run_vep_benchmark(name=name, cfg=cfg, msa_path=msa_path, model_dir=model_dir, chrom=args.chrom)
                vep_passed.append(name)
            except Exception as e:
                print(f"\n  FAILED {name}: {e}")
                vep_failed.append(name)

        print(f"\n  Stage 2 summary: {len(vep_passed)} passed, {len(vep_failed)} failed/skipped")
        if vep_failed and not vep_passed:
            raise RuntimeError(f"All Stage 2 benchmarks failed: {vep_failed}")

    print("\n== Done ==\n")


if __name__ == "__main__":
    main()
