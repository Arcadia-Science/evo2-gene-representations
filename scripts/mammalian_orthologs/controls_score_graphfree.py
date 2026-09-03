"""Graph-free composition-control preservation for the mammalian ortholog panel, both axes."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import paths  # noqa: E402
from controls.make_control_sequences import ORDINARY_CONTROLS  # noqa: E402
from ot_between_family import angular_distance, compute_ot_matrices, l2_normalize  # noqa: E402

ARM = "transcript_cdsmask"
RUN = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{ARM}"
CA = ROOT / "data" / "cache" / "mammal_embed"
OT_CACHE = ROOT / "data" / "cache" / "mammal_controls_ot" / ARM
MANIFEST = ROOT / "data" / "mammalian_orthologs" / "complete_manifest.csv"

LAYERS = list(range(32))
MIN_SP = 10  # ortholog groups with fewer represented species are excluded from within-family


def load_panel() -> tuple[list[str], pd.DataFrame]:
    """The locus keys that have an embedding on disk, and their manifest rows in that order."""
    man = pd.read_csv(MANIFEST)
    man["key"] = man.group + "__" + man.species
    keys = [k for k in man.key if (CA / ARM / f"{k}.npy").exists()]
    return keys, man.set_index("key").loc[keys]


def ready_conditions(keys: list[str]) -> list[str]:
    """Natural plus every control rung whose embedding cache is COMPLETE."""
    return [ARM] + [
        f"{ARM}_{c}"
        for c in ORDINARY_CONTROLS
        if len(list((CA / f"{ARM}_{c}").glob("*.npy"))) == len(keys)
    ]


def load_stack(cond: str, keys: list[str]) -> np.ndarray:
    """(N, 32, H) for one condition — the single expensive read."""
    return np.stack([np.load(CA / cond / f"{k}.npy", mmap_mode="r") for k in keys])


# ── between-family: Wasserstein
def score_between(keys: list[str], meta: pd.DataFrame) -> Path:
    OT_CACHE.mkdir(parents=True, exist_ok=True)
    paths.require(
        RUN / "blocks15" / "betweenfam_ot_metadata.json",
        "the W2 family order (an EXPERIMENT 1 output)",
        "run experiment 1 stage B3 (baselines/ot_between_family_sweep.py)",
        "GLM_SCRATCH",
    )
    fam_order = json.loads((RUN / "blocks15" / "betweenfam_ot_metadata.json").read_text())[
        "fam_order"
    ]
    fam_arr = meta.family.to_numpy()
    F = len(fam_order)
    iu = np.triu_indices(F, 1)
    conds = ready_conditions(keys)
    print(f"{len(keys)} loci, {F} families, conditions: {conds}", flush=True)

    nat_was = {}
    for L in LAYERS:  # the natural Wasserstein matrices are already on disk
        p = RUN / f"blocks{L}" / "betweenfam_ot_wasserstein_distances.csv"
        if p.exists():
            nat_was[L] = pd.read_csv(p, index_col=0).values[iu]

    rows = []
    for cond in conds:
        print(f"\n=== {cond} — loading stack ===", flush=True)
        X = load_stack(cond, keys)
        print(f"    {X.shape}, {X.nbytes / 1e9:.1f} GB", flush=True)
        for L in LAYERS:
            XL = X[:, L, :].astype(np.float64)
            f = OT_CACHE / f"{cond}_L{L}.npy"
            if f.exists():
                Wm = np.load(f)
            else:
                Wm = compute_ot_matrices(XL, fam_arr, fam_order, alphas=()).matrices["wasserstein"]
                np.save(f, Wm)
            rec = {"condition": cond, "layer": L}
            if cond != ARM and L in nat_was:
                rec["rho_wasserstein"] = spearmanr(nat_was[L], Wm[iu]).statistic
            rows.append(rec)
            print(
                f"    L{L:<2}"
                + (
                    f" wasserstein={rec['rho_wasserstein']:.4f}" if "rho_wasserstein" in rec else ""
                ),
                flush=True,
            )
        del X

    df = pd.DataFrame(rows)
    out = ROOT / "results" / f"_ot_control_preservation_{ARM}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)
    return out


# ── within-family: direct angular
def angular_upper(V: np.ndarray) -> np.ndarray:
    """Strict upper triangle of the pairwise angular distance among a group's members."""
    U = l2_normalize(V)
    A = angular_distance(U, U)
    return A[np.triu_indices(len(V), 1)]


def score_within(keys: list[str], meta: pd.DataFrame) -> Path:
    kpos = {k: i for i, k in enumerate(keys)}
    groups = {}
    for (fam, grp), sub in meta.reset_index().groupby(["family", "group"]):
        if len(sub) >= MIN_SP:
            groups[grp] = (fam, sub["key"].tolist())
    fams = sorted(meta.family.unique())
    print(f"{len(keys)} loci, {len(groups)} groups >={MIN_SP}sp, {len(fams)} families", flush=True)

    ready = ready_conditions(keys)
    print(f"conditions: {[c.replace(ARM + '_', '') or 'natural' for c in ready]}", flush=True)

    # reduced angular vectors per (condition, layer, group) — tiny, so hold them all
    red: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    idx = {g: [kpos[k] for k in ks] for g, (_, ks) in groups.items()}
    for cond in ready:
        print(f"  loading {cond} ...", flush=True)
        X = load_stack(cond, keys)
        red[cond] = {
            L: {g: angular_upper(X[ix, L, :].astype(np.float64)) for g, ix in idx.items()}
            for L in LAYERS
        }
        del X
        print(f"    reduced {cond}", flush=True)

    rows = []
    for cond in ready:
        if cond == ARM:
            continue
        for L in LAYERS:
            per_fam: dict[str, list[float]] = {f: [] for f in fams}
            for g, (fam, _) in groups.items():
                a, b = red[ARM][L][g], red[cond][L][g]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 6 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
                    per_fam[fam].append(spearmanr(a[ok], b[ok]).statistic)
            vals = [np.mean(v) for v in per_fam.values() if v]
            rows.append(
                {
                    "condition": cond.replace(f"{ARM}_", ""),
                    "layer": L,
                    "rho_within_angular": float(np.mean(vals)) if vals else np.nan,
                    "n_families": len(vals),
                }
            )
        print(f"  scored {cond}", flush=True)

    out = ROOT / "results" / f"_angular_control_preservation_{ARM}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}", flush=True)
    print(
        pd.DataFrame(rows)
        .query("8 <= layer <= 27")
        .groupby("condition")["rho_within_angular"]
        .mean()
        .round(3)
        .to_string(),
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--axis",
        choices=["between", "within", "both"],
        default="both",
        help="which control-preservation axis to score (default: both)",
    )
    args = ap.parse_args()

    keys, meta = load_panel()
    if args.axis in ("between", "both"):
        print("\n########## between-family (Wasserstein) ##########", flush=True)
        score_between(keys, meta)
    if args.axis in ("within", "both"):
        print("\n########## within-family (angular) ##########", flush=True)
        score_within(keys, meta)


if __name__ == "__main__":
    main()
