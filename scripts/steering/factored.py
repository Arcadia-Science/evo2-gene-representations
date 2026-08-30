"""Factor pooled Evo2 embeddings into gene and learned species components."""

from __future__ import annotations

import numpy as np


def abstract_gene_vector(embeds: dict[str, np.ndarray]) -> np.ndarray:
    """The species-invariant gene component = mean of the pooled embeddings over species, (H,)."""
    return np.mean(np.stack([np.asarray(v, dtype=np.float64) for v in embeds.values()]), axis=0)


def species_deviations(embeds: dict[str, np.ndarray]):
    """(species list, deviations (S,H)) where deviation[i] = embeds[sp_i] − abstract_gene.
    Deviations are mean-zero across species by construction (so already 'centered' for PCA)."""
    sp = list(embeds.keys())
    abstract = abstract_gene_vector(embeds)
    dev = np.stack([np.asarray(embeds[s], dtype=np.float64) - abstract for s in sp])  # (S,H)
    return sp, dev, abstract


def species_subspace(embeds: dict[str, np.ndarray], k: int = 3):
    """PCA of the species deviations."""
    sp, dev, abstract = species_deviations(embeds)
    # SVD of the (already mean-zero) deviations: dev = Us · diag(s) · Vt ; principal dirs = rows of Vt.
    _, s, Vt = np.linalg.svd(dev, full_matrices=False)
    kk = int(min(k, Vt.shape[0], np.count_nonzero(s > 1e-12)))
    U = Vt[:kk].T.copy()                                   # (H,kk), orthonormal columns
    coords = {name: dev[i] @ U for i, name in enumerate(sp)}   # (kk,) per species
    var = s[:kk] ** 2
    evr = var / (np.sum(s ** 2) + 1e-12)
    return {"abstract": abstract, "U": U, "coords": coords, "evr": evr, "species": sp}


def species_direction(embeds: dict[str, np.ndarray], target: str, reference: np.ndarray | None = None):
    """Unit species direction toward `target` and its scalar set-point, for steer_ops.clamp_direction."""
    ref = abstract_gene_vector(embeds) if reference is None else np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(embeds[target], dtype=np.float64)
    d = tgt - ref
    u = d / (np.linalg.norm(d) + 1e-8)
    return u.astype(np.float32), float(tgt @ u)


def anisotropic_axis(store: dict[str, np.ndarray], layer_idx: int) -> np.ndarray:
    """Unit vector along the panel-wide SHARED (anisotropic) component at one layer."""
    M = np.stack([v[layer_idx] for v in store.values()]).astype(np.float64)
    mu = M.mean(axis=0)
    return mu / (np.linalg.norm(mu) + 1e-12)


def species_direction_orth(embeds: dict[str, np.ndarray], target: str, axis: np.ndarray):
    """ANISOTROPY-ORTHOGONALISED species direction — the control for "is this steering the species, or just pushing along the shared norm axis?"."""
    u, _ = species_direction(embeds, target)
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    u = np.asarray(u, dtype=np.float64)
    u_perp = u - (u @ a) * a
    n = np.linalg.norm(u_perp)
    if n < 1e-8:                      # û was essentially the anisotropic axis itself
        return None, 0.0
    u_perp = u_perp / n
    tgt = np.asarray(embeds[target], dtype=np.float64)
    return u_perp.astype(np.float32), float(tgt @ u_perp)


# --------------------------------------------------------------------------- self-check (no Evo2)
def _test(tol: float = 1e-6) -> None:
    """CPU-only sanity check on fake embeddings: subspace round-trips, coords match, abstract = mean."""
    rng = np.random.default_rng(0)
    H, S, k = 40, 6, 3
    names = [f"sp{i}" for i in range(S)]
    embeds = {n: rng.standard_normal(H) for n in names}

    abstract = abstract_gene_vector(embeds)
    assert np.allclose(abstract, np.mean(np.stack(list(embeds.values())), axis=0))

    dec = species_subspace(embeds, k=k)
    U, coords = dec["U"], dec["coords"]
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=tol), "U columns not orthonormal"
    # reconstruction inside the k-subspace: abstract + U·coords ≈ projection of embeds onto subspace
    for n in names:
        dev = embeds[n] - abstract
        recon_dev = U @ coords[n]
        # coords are exactly Uᵀdev, so U·coords is the orthogonal projection of dev onto span(U)
        r = np.linalg.norm(recon_dev - (U @ (U.T @ dev)))
        assert r < tol, f"coords/projection mismatch {r}"

    u, c = species_direction(embeds, "sp2")
    assert abs(np.linalg.norm(u) - 1.0) < tol, "direction not unit"
    assert abs(float(np.asarray(embeds["sp2"]) @ u) - c) < 1e-4, "set-point mismatch"

    # orthogonalised direction: unit, exactly perpendicular to the axis, set-point consistent
    ax = rng.standard_normal(H)
    up, cp = species_direction_orth(embeds, "sp2", ax)
    assert abs(np.linalg.norm(up) - 1.0) < tol, "orth direction not unit"
    r_perp = abs(float(np.asarray(up, dtype=np.float64) @ (ax / np.linalg.norm(ax))))
    assert r_perp < 1e-6, f"orth direction not perpendicular to axis ({r_perp})"
    assert abs(float(np.asarray(embeds["sp2"]) @ np.asarray(up, dtype=np.float64)) - cp) < 1e-4

    print("factored self-check PASSED")
    print(f"  orth: û⊥·â = {r_perp:.2e}, ‖û⊥‖ = {np.linalg.norm(up):.6f}")
    print(f"  species={S} H={H} k_req={k} k_eff={U.shape[1]}  evr={np.round(dec['evr'], 3).tolist()}")
    print(f"  U orthonormal, abstract==mean, coords==Uᵀdev, set-point consistent")


if __name__ == "__main__":
    _test()
