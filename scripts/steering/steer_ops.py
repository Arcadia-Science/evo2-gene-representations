"""Non-additive steering interventions for the Evo2 species-steering experiment."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


# --------------------------------------------------------------------------- pure tensor math
def _unit(u: torch.Tensor) -> torch.Tensor:
    """Normalise a direction (H,) to unit L2 length (defensive; set-point assumes ‖û‖=1)."""
    return u / (u.norm() + 1e-8)


def _clamp(h: torch.Tensor, u: torch.Tensor, c_target: float, alpha: float) -> torch.Tensor:
    """Set the projection along unit direction û to c_target, α-interpolated, per position."""
    proj = (h * u).sum(dim=-1, keepdim=True)          # (...,1)  per-token projection
    return h + alpha * (c_target - proj) * u          # û broadcasts over the feature dim


def _ablate(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Remove the component along unit direction û entirely (directional ablation):
        h' = h − ⟨h,û⟩·û   ⇒  ⟨h',û⟩ = 0."""
    proj = (h * u).sum(dim=-1, keepdim=True)
    return h - proj * u


def _subspace_clamp(h: torch.Tensor, U: torch.Tensor, c_target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Interpolate residual coordinates toward a target in an orthonormal subspace."""
    coords = h @ U                                    # (...,k)  = Uᵀh per token
    delta = alpha * (c_target - coords)               # (...,k)
    return h + delta @ U.transpose(-2, -1)            # back to (...,H)


# --------------------------------------------------------------------------- hook factories
def _tuple_out(out, new_h: torch.Tensor):
    """Rewrite a block's output preserving its tuple tail (matches steer_lib._steer_hook)."""
    return (new_h,) + tuple(out[1:]) if isinstance(out, tuple) else new_h


def _op_hook(fn):
    """Wrap a pure h->h' function as a forward hook operating on out[0] (the residual stream)."""
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        return _tuple_out(out, fn(h))
    return hook


# --------------------------------------------------------------------------- context managers
@contextmanager
def clamp_direction(model, layer: str, direction: np.ndarray, c_target: float,
                    alpha: float = 1.0, device: str = "cuda"):
    """SET the residual-stream projection along `direction` to the scalar set-point `c_target`."""
    handle = None
    if direction is not None and alpha != 0.0:
        u = _unit(torch.as_tensor(np.asarray(direction), dtype=torch.float32, device=device))
        c = float(c_target)
        handle = model.model.get_submodule(layer).register_forward_hook(
            _op_hook(lambda h: _clamp(h, u.to(h.dtype).to(h.device), c, alpha)))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def ablate_direction(model, layer: str, direction: np.ndarray, device: str = "cuda"):
    """Directional ablation control: remove the component along `direction` at every position
    (h' = h − ⟨h,û⟩·û). Zeros the species projection without steering toward any target."""
    handle = None
    if direction is not None:
        u = _unit(torch.as_tensor(np.asarray(direction), dtype=torch.float32, device=device))
        handle = model.model.get_submodule(layer).register_forward_hook(
            _op_hook(lambda h: _ablate(h, u.to(h.dtype).to(h.device))))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


@contextmanager
def subspace_clamp(model, layer: str, U: np.ndarray, c_target: np.ndarray,
                   alpha: float = 1.0, device: str = "cuda"):
    """Clamp residual-stream coordinates in an orthonormal species subspace."""
    handle = None
    if U is not None and c_target is not None and alpha != 0.0:
        Ut = torch.as_tensor(np.asarray(U), dtype=torch.float32, device=device)          # (H,k)
        ct = torch.as_tensor(np.asarray(c_target), dtype=torch.float32, device=device)   # (k,)
        handle = model.model.get_submodule(layer).register_forward_hook(
            _op_hook(lambda h: _subspace_clamp(h, Ut.to(h.dtype).to(h.device),
                                               ct.to(h.dtype).to(h.device), alpha)))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


# --------------------------------------------------------------------------- synthetic math tests
def _test(tol: float = 1e-4) -> None:
    """CPU-only unit test of the operator MATH on fake tensors (no Evo2, no GPU)."""
    torch.manual_seed(0)
    B, L, H, k = 1, 7, 32, 4
    h = torch.randn(B, L, H, dtype=torch.float64)

    # --- clamp along a unit direction: ⟨h',û⟩ must equal c_target at every position ---
    u = _unit(torch.randn(H, dtype=torch.float64))
    c_target = 1.234
    hc = _clamp(h, u, c_target, alpha=1.0)
    proj = (hc * u).sum(dim=-1)                              # (B,L)
    r_clamp = (proj - c_target).abs().max().item()
    assert r_clamp < tol, f"clamp projection residual {r_clamp}"

    # α=0 is a no-op; α=0.5 lands exactly halfway from the original projection to c_target
    assert torch.allclose(_clamp(h, u, c_target, 0.0), h)
    proj0 = (h * u).sum(dim=-1)
    half = (_clamp(h, u, c_target, 0.5) * u).sum(dim=-1)
    r_half = (half - (proj0 + 0.5 * (c_target - proj0))).abs().max().item()
    assert r_half < tol, f"alpha-interp residual {r_half}"

    # --- directional ablation: projection must be exactly zero ---
    ha = _ablate(h, u)
    r_abl = (ha * u).sum(dim=-1).abs().max().item()
    assert r_abl < tol, f"ablation residual {r_abl}"

    # --- subspace clamp: Uᵀh' must equal c_target (k,) at every position ---
    U, _ = torch.linalg.qr(torch.randn(H, k, dtype=torch.float64))   # orthonormal columns (H,k)
    cvec = torch.randn(k, dtype=torch.float64)
    hs = _subspace_clamp(h, U, cvec, alpha=1.0)
    coords = hs @ U                                          # (B,L,k)
    r_sub = (coords - cvec).abs().max().item()
    assert r_sub < tol, f"subspace clamp residual {r_sub}"
    # off-subspace component is untouched: (I-UUᵀ)h' == (I-UUᵀ)h
    perp = lambda x: x - (x @ U) @ U.transpose(-2, -1)
    r_perp = (perp(hs) - perp(h)).abs().max().item()
    assert r_perp < tol, f"subspace off-axis leakage {r_perp}"

    print("steer_ops synthetic math tests PASSED")
    print(f"  clamp    |⟨h',û⟩ − c_target|_max      = {r_clamp:.2e}")
    print(f"  clamp    α-interp residual_max          = {r_half:.2e}")
    print(f"  ablate   |⟨h',û⟩|_max                   = {r_abl:.2e}")
    print(f"  subspace |Uᵀh' − c_target|_max          = {r_sub:.2e}")
    print(f"  subspace off-axis leakage_max           = {r_perp:.2e}")


if __name__ == "__main__":
    _test()
