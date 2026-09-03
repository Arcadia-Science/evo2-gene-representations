"""Core operations used by the active Evo2 steering pipeline."""

from __future__ import annotations
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))

BASE_ID = {base: ord(base) for base in "ACGT"}


def load_model():
    """Load the shared Evo2 7B model."""
    from evo2_embedding import load_model as _load_model

    return _load_model()


def _tokenize(model, seq: str, device: str) -> torch.Tensor:
    return torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)


def _steer_hook(vec: torch.Tensor):
    """Add `vec` to a block's residual stream at every position."""

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            return (hidden + vec.to(hidden.dtype).to(hidden.device),) + tuple(out[1:])
        return out + vec.to(out.dtype).to(out.device)

    return hook


@contextmanager
def steering_multi(model, layer_vecs: dict[str, np.ndarray], device: str = "cuda"):
    """Apply already-scaled steering vectors at multiple blocks."""
    handles = []
    try:
        for layer, vec in layer_vecs.items():
            if vec is None:
                continue
            tensor = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=device)
            handles.append(
                model.model.get_submodule(layer).register_forward_hook(_steer_hook(tensor))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def _overwrite_hook(vec: torch.Tensor, alpha: float, keep_hidden: bool, preserve_norm: bool):
    """Replace or interpolate a residual stream with `vec`."""

    def hook(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        target = vec.to(hidden.dtype).to(hidden.device)
        new = (
            (1.0 - alpha) * hidden + alpha * target
            if keep_hidden
            else (alpha * target).expand_as(hidden).clone()
        )
        if preserve_norm:
            original_norm = hidden.norm(dim=-1, keepdim=True)
            new = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-8) * original_norm
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    return hook


@contextmanager
def overwriting(
    model,
    layer_vecs: dict[str, np.ndarray],
    alpha: float,
    keep_h: bool,
    preserve_norm: bool,
    device: str = "cuda",
):
    """Apply replacement or interpolation hooks at multiple blocks."""
    handles = []
    try:
        for layer, vec in layer_vecs.items():
            if vec is None:
                continue
            tensor = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=device)
            hook = _overwrite_hook(tensor, float(alpha), keep_h, preserve_norm)
            handles.append(model.model.get_submodule(layer).register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def norm_matched_random(vec: np.ndarray, seed: int) -> np.ndarray:
    """Return a seeded Gaussian vector with the same L2 norm as `vec`."""
    rng = np.random.default_rng(seed)
    random = rng.standard_normal(vec.shape).astype(np.float32)
    return random / (np.linalg.norm(random) + 1e-8) * np.linalg.norm(vec)


def codon_blocks(human_cds: str, target_cds: str):
    """Codon-align two CDS sequences and return paired nucleotide blocks."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.Seq import Seq

    def protein(cds: str) -> str:
        aa = str(Seq(cds[: len(cds) // 3 * 3]).translate())
        return aa[:-1] if aa.endswith("*") else aa

    human_protein, target_protein = protein(human_cds), protein(target_cds)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10.0, -0.5
    alignment = aligner.align(human_protein, target_protein)[0]
    human_blocks, target_blocks = alignment.aligned
    return human_blocks, target_blocks, human_protein, target_protein


def codon_diagnostics(human_cds: str, target_cds: str):
    """Return differing and conserved sites from a codon-aware CDS alignment."""
    human_blocks, target_blocks, human_protein, target_protein = codon_blocks(human_cds, target_cds)

    diagnostic, conserved = [], []
    for (human_start, human_end), (target_start, _target_end) in zip(
        human_blocks, target_blocks, strict=False
    ):
        for offset in range(human_end - human_start):
            human_codon_index = human_start + offset
            target_codon_index = target_start + offset
            human_codon = human_cds[3 * human_codon_index : 3 * human_codon_index + 3]
            target_codon = target_cds[3 * target_codon_index : 3 * target_codon_index + 3]
            if len(human_codon) < 3 or len(target_codon) < 3:
                continue
            aa_change = human_protein[human_codon_index] != target_protein[target_codon_index]
            for codon_position in range(3):
                human_base = human_codon[codon_position]
                target_base = target_codon[codon_position]
                if human_base not in BASE_ID or target_base not in BASE_ID:
                    continue
                human_index = 3 * human_codon_index + codon_position
                if human_base != target_base:
                    diagnostic.append(
                        {
                            "h_idx": human_index,
                            "h_base": human_base,
                            "t_base": target_base,
                            "aa_change": bool(aa_change),
                        }
                    )
                else:
                    conserved.append(human_index)
    return diagnostic, conserved
