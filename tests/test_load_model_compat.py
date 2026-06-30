"""
Tests for load_model_compat() in scripts/test_gpn_star.py.

The function works around a transformers >=4.44 regression where every tensor
created inside __init__ (including numpy loads) lands on the meta device, making
.item() calls crash.  The fix is to:
  1. Instantiate GPNStarForMaskedLM directly (outside from_pretrained's context)
  2. Load weights from safetensors with strict=False
  3. Call tie_weights() to restore cls.predictions.decoder.bias -> cls.predictions.bias

Bugs this test suite is designed to catch:
  - Weights never loaded          → model has random-init values, not checkpoint values
  - Tied weight not restored       → decoder.bias is garbage / not tied to predictions.bias
  - Wrong assignment / key drift   → weight loaded into wrong slot
  - Meta-device tensor left behind → forward pass crashes
  - Unexpected missing keys        → silent partial load
  - Phylo-dist path fallback broken → model can't construct phylo info on new machine
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# ── Make load_model_compat importable from scripts/ ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from test_gpn_star import load_model_compat  # noqa: E402

import gpn.star.model  # registers GPNStar with AutoConfig/AutoModel  # noqa: E402
from gpn.star.model import GPNStarForMaskedLM  # noqa: E402
from transformers import AutoConfig  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_MODEL_DIR = _REPO_ROOT / "models"
_HF_ID = "songlab/gpn-star-hg38-v100-200m"
_N_SPECIES = 100

# Layers to spot-check for correct weight values. Chosen to cover first layer,
# last layer, embedding, and prediction head — maximising the chance of catching
# an off-by-one assignment or a partial load.
_SPOT_CHECK_KEYS = [
    "model.encoder.layer.0.attention.col_attention.self.key.weight",
    "model.encoder.layer.0.attention.row_attention.self.query.weight",
    "model.encoder.layer.15.output.dense.weight",   # last transformer layer
    "model.target_embedding.input_embed.weight",
    "model.source_embedding.embed.weight",
    "cls.predictions.transform.dense.weight",
    "cls.predictions.decoder.weight",               # in _tied_weights_keys but IS in checkpoint
]

# ── Module-scoped fixtures (model loaded once for the session) ────────────────


@pytest.fixture(scope="module")
def model_path():
    """Resolve the local HuggingFace snapshot (no download; must already be cached)."""
    try:
        path = snapshot_download(
            repo_id=_HF_ID,
            cache_dir=str(_MODEL_DIR),
            local_files_only=True,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
    except Exception as e:
        pytest.skip(
            f"Model snapshot not found in cache for {_HF_ID} under {_MODEL_DIR} "
            f"(local_files_only=True): {e}"
        )
    return Path(path)


@pytest.fixture(scope="module")
def loaded_model(model_path):
    """GPNStarForMaskedLM loaded with load_model_compat (the function under test)."""
    model = load_model_compat(model_path)
    model.eval()
    return model


@pytest.fixture(scope="module")
def checkpoint_weights(model_path):
    """Raw tensors straight from the safetensors file — ground truth for comparison."""
    return load_file(str(model_path / "model.safetensors"))


@pytest.fixture(scope="module")
def random_model(model_path):
    """A freshly-initialized (untrained) model — used to confirm loaded ≠ random."""
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(str(model_path))
    config.phylo_dist_path = str(model_path / "phylo_dist")
    model = GPNStarForMaskedLM(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def synthetic_batch():
    """Deterministic small batch for forward-pass tests (no MSA data needed)."""
    rng = np.random.default_rng(42)
    B, L, T, N = 1, 32, 1, _N_SPECIES
    return {
        "input_ids":      torch.from_numpy(rng.integers(1, 5, (B, L, T))).long(),
        "source_ids":     torch.from_numpy(rng.integers(1, 5, (B, L, N))).long(),
        "target_species": torch.zeros(B, T, dtype=torch.long),
    }


# ── Smoke ─────────────────────────────────────────────────────────────────────


def test_load_succeeds(loaded_model):
    assert isinstance(loaded_model, GPNStarForMaskedLM)


def test_no_meta_tensors(loaded_model):
    """Catch the original bug: tensors stranded on meta device crash .item()."""
    meta = [n for n, p in loaded_model.named_parameters() if p.device.type == "meta"]
    assert not meta, f"Parameters still on meta device: {meta}"


# ── Weight-loading integrity ──────────────────────────────────────────────────


@pytest.mark.parametrize("key", _SPOT_CHECK_KEYS)
def test_weights_match_checkpoint(key, loaded_model, checkpoint_weights):
    """Each spot-checked parameter must exactly equal the safetensors value.

    Detects: weights never loaded, weights loaded into wrong layer, silent dtype cast.
    """
    model_param = dict(loaded_model.named_parameters())[key]
    expected = checkpoint_weights[key]
    assert torch.equal(model_param, expected), (
        f"{key}: loaded value differs from checkpoint. "
        f"Max abs diff: {(model_param - expected).abs().max().item():.3e}"
    )


def test_loaded_differs_from_random_init(loaded_model, random_model):
    """Trained and randomly-initialized models must have clearly different weights.

    Detects: load_state_dict silently no-opped and the model kept its random init.
    """
    key = "model.encoder.layer.0.attention.col_attention.self.key.weight"
    loaded_w = dict(loaded_model.named_parameters())[key]
    random_w = dict(random_model.named_parameters())[key]
    max_diff = (loaded_w - random_w).abs().max().item()
    assert max_diff > 1e-3, (
        f"Loaded weights are indistinguishable from random init (max diff={max_diff:.2e}). "
        "The checkpoint was probably never applied."
    )


# ── Tied weights ──────────────────────────────────────────────────────────────


def test_decoder_bias_is_tied_tensor(loaded_model):
    """cls.predictions.decoder.bias must be the *same tensor object* as cls.predictions.bias.

    Detects: tie_weights() not called — decoder.bias stays at random-init values.
    The bias controls absolute output probabilities for each nucleotide token;
    an un-tied bias silently shifts all predictions away from the trained values.
    """
    assert loaded_model.cls.predictions.decoder.bias is loaded_model.cls.predictions.bias, (
        "decoder.bias and cls.predictions.bias are not the same tensor. "
        "Likely fix: call model.tie_weights() after load_state_dict."
    )


def test_decoder_bias_values_match_checkpoint(loaded_model, checkpoint_weights):
    """After tie_weights(), decoder.bias must equal cls.predictions.bias from the checkpoint.

    Detects: tied but pointing at the wrong tensor (e.g., a freshly allocated one).
    """
    expected = checkpoint_weights["cls.predictions.bias"]
    actual = loaded_model.cls.predictions.decoder.bias
    assert torch.equal(actual, expected), (
        f"decoder.bias value mismatch. Max diff: {(actual - expected).abs().max().item():.3e}"
    )


# ── Missing / unexpected keys ─────────────────────────────────────────────────


def test_only_declared_tied_keys_missing(model_path):
    """After strict=False load, missing_keys must be a subset of _tied_weights_keys.

    Detects: a weight that should exist in the checkpoint is absent — indicating
    model or checkpoint version drift, or a renamed parameter.
    """
    config = AutoConfig.from_pretrained(str(model_path))
    config.phylo_dist_path = str(model_path / "phylo_dist")
    model = GPNStarForMaskedLM(config)
    state_dict = load_file(str(model_path / "model.safetensors"))
    result = model.load_state_dict(state_dict, strict=False)
    tied = set(model._tied_weights_keys)
    unexpected_missing = [k for k in result.missing_keys if k not in tied]
    assert not unexpected_missing, f"Unexpected missing keys: {unexpected_missing}"


def test_no_unexpected_keys_in_checkpoint(model_path):
    """Checkpoint must not contain keys that don't correspond to any model parameter.

    Detects: stale keys from an old model version, or a mismatched checkpoint/code.
    """
    config = AutoConfig.from_pretrained(str(model_path))
    config.phylo_dist_path = str(model_path / "phylo_dist")
    model = GPNStarForMaskedLM(config)
    state_dict = load_file(str(model_path / "model.safetensors"))
    result = model.load_state_dict(state_dict, strict=False)
    assert not result.unexpected_keys, f"Unexpected keys in checkpoint: {result.unexpected_keys}"


# ── Forward pass ──────────────────────────────────────────────────────────────


def test_forward_pass_runs(loaded_model, synthetic_batch):
    with torch.no_grad():
        out = loaded_model(**synthetic_batch)
    assert out.logits is not None


def test_output_shape(loaded_model, synthetic_batch):
    """Logits must be (B, L, T, vocab_size)."""
    B, L, T = 1, 32, 1
    with torch.no_grad():
        out = loaded_model(**synthetic_batch)
    expected_shape = (B, L, T, loaded_model.config.vocab_size)
    assert tuple(out.logits.shape) == expected_shape, (
        f"Got {tuple(out.logits.shape)}, expected {expected_shape}"
    )


def test_output_is_finite(loaded_model, synthetic_batch):
    """NaN / Inf in logits usually means a broken phylo-dist bias or un-initialized weight."""
    with torch.no_grad():
        out = loaded_model(**synthetic_batch)
    assert torch.isfinite(out.logits).all(), "Logits contain NaN or Inf"


def test_forward_pass_is_deterministic(loaded_model, synthetic_batch):
    """Same input → identical logits on consecutive calls (eval mode, no dropout)."""
    with torch.no_grad():
        out1 = loaded_model(**synthetic_batch)
        out2 = loaded_model(**synthetic_batch)
    assert torch.equal(out1.logits, out2.logits), "Forward pass is not deterministic"


def test_loaded_vs_random_logits_differ(loaded_model, random_model, synthetic_batch):
    """Loaded-model and random-init logits must diverge clearly.

    Detects: checkpoint never applied → model outputs look like random-init model.
    """
    with torch.no_grad():
        logits_loaded = loaded_model(**synthetic_batch).logits
        logits_random = random_model(**synthetic_batch).logits
    max_diff = (logits_loaded - logits_random).abs().max().item()
    assert max_diff > 0.01, (
        f"Logits from loaded model are nearly identical to random init "
        f"(max diff={max_diff:.2e}). Checkpoint was likely not applied."
    )


# ── Architecture config ───────────────────────────────────────────────────────


def test_config_architecture(loaded_model):
    """Sanity-check the 200M vertebrate model's key architectural parameters."""
    cfg = loaded_model.config
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 16
    assert cfg.num_attention_heads == 16
    assert cfg.vocab_size == 6


# ── Phylo-dist path fallback ──────────────────────────────────────────────────


def test_phylo_dist_fallback_is_used(loaded_model, model_path):
    """The original config.json stores an absolute training-machine path that does not
    exist here; load_model_compat must silently fall back to model_path/phylo_dist.

    Verify by checking that the loaded model's config now points to the local path.
    """
    config = AutoConfig.from_pretrained(str(model_path))
    original_path = config.phylo_dist_path
    # If the original path exists this test is vacuous; warn but don't fail
    if Path(original_path).exists():
        pytest.skip(f"Original phylo_dist_path exists on this machine: {original_path}")

    local_fallback = str(model_path / "phylo_dist")
    assert loaded_model.config.phylo_dist_path == local_fallback, (
        f"Expected config.phylo_dist_path to be updated to local fallback "
        f"'{local_fallback}', got '{loaded_model.config.phylo_dist_path}'"
    )


def test_phylo_dist_missing_raises(model_path, tmp_path):
    """When both the config path and the local fallback are absent, raise FileNotFoundError.

    The error is raised before safetensors loading, so we only need a minimal
    fake model directory containing config.json (no weights file needed).
    """
    # Copy config.json to tmp dir and patch phylo_dist_path to a nonexistent location.
    # Do NOT copy phylo_dist/, so the fallback is also absent.
    cfg_data = json.loads((model_path / "config.json").read_text())
    cfg_data["phylo_dist_path"] = "/nonexistent/path/phylo_dist"
    (tmp_path / "config.json").write_text(json.dumps(cfg_data))

    with pytest.raises(FileNotFoundError):
        load_model_compat(tmp_path)
