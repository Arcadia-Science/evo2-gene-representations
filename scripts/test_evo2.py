"""
Evo2 model verification script.

Two-stage test:
  1. Synthetic sanity check — loads evo2_7b, runs a forward pass on a random
     DNA sequence, and validates output logits and embedding shapes/values.
  2. VEP benchmark — scores variants from songlab/clinvar_vs_benign using
     log-likelihood ratio (LLR) and reports AUROC.

Usage:
    # Stage 1 only:
    uv run python scripts/test_evo2.py

    # Stage 1 + VEP benchmark:
    uv run python scripts/test_evo2.py --vep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "evo2_7b"

# Intermediate layer recommended for embeddings per the Evo2 paper.
# 7B has 32 blocks (0-31); layer 24 is ~75% depth — a good default.
EMBED_LAYER = "blocks.24.mlp.l3"

DNA_VOCAB = list("ACGT")


def random_dna(length: int, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(DNA_VOCAB, size=length))


# ── Stage 1: synthetic forward pass ──────────────────────────────────────────


def run_synthetic_check(device: str) -> bool:
    from evo2 import Evo2

    print(f"\n{'─'*60}")
    print(f"  Synthetic check: {MODEL_NAME}")
    print(f"{'─'*60}")

    print(f"    Loading {MODEL_NAME} (downloads on first run ~14 GB)...")
    # Evo2 is a thin wrapper, not an nn.Module: it places its StripedHyena on GPU
    # during construction and runs in inference_mode, so there is no .to()/.eval().
    model = Evo2(MODEL_NAME)

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"    Parameters : {n_params:,}")

    # Tokenize a short random sequence
    seq = random_dna(512)
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq), dtype=torch.int
    ).unsqueeze(0).to(device)
    print(f"    Input shape: {tuple(input_ids.shape)}  (1 × {len(seq)} tokens)")

    # ── Forward pass: logits only ──
    with torch.no_grad():
        logits, _ = model(input_ids)

    print(f"    Logits shape: {tuple(logits.shape)}")
    assert logits.ndim == 3, f"Expected 3D logits (B, L, vocab), got {logits.ndim}D"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"

    # Spot-check: next-token probabilities at position 10
    probs = logits[0, 10].softmax(dim=-1)
    top_token = probs.argmax().item()
    print(f"    Top next-token at pos 10: id={top_token}, prob={probs[top_token]:.3f}  ✓")

    # ── Forward pass: embeddings ──
    print(f"\n    Extracting embeddings from layer '{EMBED_LAYER}'...")
    with torch.no_grad():
        _, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=[EMBED_LAYER],
        )

    emb = embeddings[EMBED_LAYER]
    print(f"    Embedding shape : {tuple(emb.shape)}  (B × L × d_model)")
    assert emb.ndim == 3, f"Expected 3D embedding (B, L, d), got {emb.ndim}D"
    assert torch.isfinite(emb).all(), "Embeddings contain NaN or Inf"

    # Mean-pool to get a sequence-level vector
    seq_vec = emb[0].mean(dim=0)
    print(f"    Sequence vector : shape={tuple(seq_vec.shape)}, "
          f"norm={seq_vec.norm().item():.2f}  ✓")

    print(f"  PASSED")
    return True


# ── Stage 2: VEP benchmark ────────────────────────────────────────────────────


def llr_score(model, tokenizer, chrom, pos, ref, alt, window: int, device: str) -> float:
    """Zero-shot LLR for a single SNV: log P(alt) - log P(ref) at masked position."""
    # pos is 1-based VCF; we don't have the genome sequence here so we score
    # based on the log-likelihood of the ref vs alt token at that position
    # using a window of surrounding context from the MSA-free sequence.
    # For a fair comparison with GPN-Star we'd need genomic context, but here
    # we demonstrate the API using the variant's ref allele as the centre token.
    half = window // 2
    # Build a dummy context: ref allele flanked by random DNA
    rng = np.random.default_rng(int(pos))
    left = "".join(rng.choice(DNA_VOCAB, size=half))
    right = "".join(rng.choice(DNA_VOCAB, size=half))

    def score_seq(seq: str) -> float:
        ids = torch.tensor(tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(ids)
        log_probs = logits[0].log_softmax(dim=-1)
        # Score = mean log-prob of the full sequence (proxy for likelihood)
        token_ids = ids[0]
        return log_probs[torch.arange(len(token_ids) - 1), token_ids[1:]].mean().item()

    ref_seq = left + ref + right
    alt_seq = left + alt + right
    return score_seq(alt_seq) - score_seq(ref_seq)


def run_vep_benchmark(device: str, n_variants: int = 500) -> None:
    from datasets import load_dataset
    from sklearn.metrics import roc_auc_score
    from evo2 import Evo2

    print(f"\n{'─'*60}")
    print(f"  VEP benchmark: {MODEL_NAME}")
    print(f"  Dataset: songlab/clinvar_vs_benign  (first {n_variants} variants)")
    print(f"{'─'*60}")

    print(f"    Loading {MODEL_NAME}...")
    model = Evo2(MODEL_NAME)  # self-places on GPU; no .to()/.eval()

    print("    Loading clinvar_vs_benign...")
    ds = load_dataset("songlab/clinvar_vs_benign", split="test")
    df = ds.to_pandas().head(n_variants)
    print(f"    Scoring {len(df)} variants...")

    scores = []
    for _, row in df.iterrows():
        s = llr_score(
            model, model.tokenizer,
            row.chrom, row.pos, row.ref, row.alt,
            window=64, device=device,
        )
        scores.append(s)

    auroc = roc_auc_score(df["label"].tolist(), scores)
    print(f"\n    AUROC: {auroc:.4f}  (on {n_variants} variants, random-context proxy)")
    print("    NOTE: this uses random flanking context, not actual genomic sequence.")
    print("    Full genomic context requires the hg38 reference FASTA.")

    import pandas as pd
    out = Path("data/evo2_clinvar_vep_scores.parquet")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"score": scores, "label": df["label"].tolist()}).to_parquet(out, index=False)
    print(f"    Scores saved to {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vep", action="store_true", help="Also run Stage 2 VEP benchmark")
    p.add_argument("--n-variants", type=int, default=500,
                   help="Number of clinvar variants to score in VEP benchmark (default: 500)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n== Evo2 model verification ==")
    print(f"   PyTorch version : {torch.__version__}")
    print(f"   Device          : {args.device}")

    # ── Stage 1 ──
    print("\n[Stage 1] Synthetic forward-pass check")
    try:
        run_synthetic_check(args.device)
    except Exception as e:
        print(f"\n  Stage 1: FAILED — {e}")
        raise

    # ── Stage 2 ──
    if args.vep:
        print("\n[Stage 2] VEP benchmark")
        try:
            run_vep_benchmark(args.device, n_variants=args.n_variants)
            print("\n  Stage 2: PASSED")
        except Exception as e:
            print(f"\n  Stage 2: FAILED — {e}")
            raise

    print("\n== Done ==\n")


if __name__ == "__main__":
    main()
