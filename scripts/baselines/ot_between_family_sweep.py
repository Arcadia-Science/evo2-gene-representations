"""Score the centroid-free OT metrics across an all-layer sweep, alongside the geodesic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))

from geodesic_utils import mantel_test, upper_triangle  # noqa: E402
from ot_between_family import DEFAULT_ALPHAS, compute_ot_matrices  # noqa: E402
from between_family_baselines import CONVERGENT_PAIRS  # noqa: E402

# ── experiment registry
# Each experiment maps a per-layer sweep dir to the embedding cache the centroid
# analysis consumed. panel drives the loader (human = single layer_stack.npy;
# mammal = per-locus npy assembled by mammal_between.load_embedded).
HUMAN_CACHES = {
    "human-cds": "data/cache/evo2_human_layer_sweep_cds",
    "human-transcript": "data/cache/evo2_human_layer_sweep",
    "human-cdsmask": "data/cache/evo2_human_cdspool_transcript",
    # 47-family expansion: SAME cache as human-cds (the incremental embed extended it from 580 to
    # 1122 genes); the run dir's centroid matrix is what selects the 47-family order, so the two
    # experiments differ only in which run dir — and therefore which family set — they score.
    "human-cds-48fam": "data/cache/evo2_human_layer_sweep_cds",
}
HUMAN_RUNS = {
    "human-cds": "results/2026-07-15_evo2-human-panel-cds",
    "human-transcript": "results/2026-07-01_evo2-human-panel",
    "human-cdsmask": "results/2026-07-20_evo2-human-cdspool-transcript",
    "human-cds-48fam": "results/2026-07-22_evo2-human-panel-cds",
}
HUMAN_CENTROID = "evo2_human_centroid_distances.csv"
# (arm, run_dir, manifest) — manifest lets a capped variant reuse the SAME cache under a _400 run.
MAMMAL_RUNS = {
    "mammal-cds": ("cds", "results/2026-07-16_mammalian-orthologs-cds", "complete_manifest.csv"),
    "mammal-transcript": ("transcript", "results/2026-07-16_mammalian-orthologs-transcript",
                          "complete_manifest.csv"),
    "mammal-cdsmask": ("transcript_cdsmask",
                       "results/2026-07-16_mammalian-orthologs-transcript_cdsmask",
                       "complete_manifest.csv"),
    "mammal-cds-400": ("cds", "results/2026-07-16_mammalian-orthologs-cds_400",
                       "complete_manifest_cap400.csv"),
    "mammal-transcript-400": ("transcript", "results/2026-07-16_mammalian-orthologs-transcript_400",
                              "complete_manifest_cap400.csv"),
    "mammal-cdsmask-400": ("transcript_cdsmask",
                           "results/2026-07-16_mammalian-orthologs-transcript_cdsmask_400",
                           "complete_manifest_cap400.csv"),
}
MAMMAL_CENTROID = "evo2_mammal_centroid_distances.csv"

EXPERIMENTS = list(HUMAN_RUNS) + list(MAMMAL_RUNS)


# ── panel loaders: return (stack, families_array, fam_order, baseline_mats, axis_of) ──
# stack: (n_layers, N, D); families_array aligned to stack columns; fam_order + baselines
# are layer-independent and shared across the sweep.


def _fam_order_from_centroid(run: Path, centroid_name: str) -> list[str]:
    """Canonical family row/col order = the panel's existing centroid matrix order."""
    blocks = sorted(run.glob("blocks*"))
    for b in blocks:
        f = b / centroid_name
        if f.exists():
            return pd.read_csv(f, index_col=0).index.tolist()
    raise SystemExit(f"no {centroid_name} under {run}/blocks*")


def load_human(exp: str):
    cache = ROOT / HUMAN_CACHES[exp]
    run = ROOT / HUMAN_RUNS[exp]
    stack = np.load(cache / "layer_stack.npy")           # (n_layers, N, D)
    meta = pd.read_csv(cache / "metadata.csv")           # aligned to stack columns
    fams = meta["family"].to_numpy()
    fam_order = _fam_order_from_centroid(run, HUMAN_CENTROID)
    # Baseline F×F matrices + axis map: already on disk per layer (layer-independent);
    # pull from the first layer that has them.
    blocks = sorted(run.glob("blocks*"))
    src = next(b for b in blocks if (b / "between_family_baseline_scores.csv").exists())
    axis_of = dict(zip(*[pd.read_csv(src / "between_family_baseline_scores.csv")[c]
                         for c in ("baseline", "axis")]))
    baseline_mats = {}
    for name in axis_of:
        f = src / f"betweenfam_{name}_distances.csv"
        if f.exists():
            baseline_mats[name] = (pd.read_csv(f, index_col=0)
                                   .reindex(index=fam_order, columns=fam_order).values)
    return stack, fams, fam_order, baseline_mats, axis_of, run, HUMAN_CENTROID


def load_mammal(exp: str):
    import mammal_between as mb  # noqa: E402  (reuse its loaders + axis builders exactly)

    arm, run_rel, manifest = MAMMAL_RUNS[exp]
    run = ROOT / run_rel
    stack, meta = mb.load_embedded(arm, manifest)        # (n_layers, N, D), meta aligned
    fam_order = _fam_order_from_centroid(run, MAMMAL_CENTROID)
    fams = meta["family"].to_numpy()
    # Recompute the transferable axes exactly as mammal_between.main does (layer-independent;
    # cached fetches make this cheap). fam_order is the ≥8-member set the centroid used.
    cds = mb.load_cds()
    sub = meta[meta.family.isin(fam_order)].reset_index(drop=True)
    axis_of = mb.AXIS
    baseline_mats = {
        "pfam_jsd": mb.compute_pfam_jsd(fam_order, mb.PFAM_ACCESSIONS),
        "cofactor": mb.cofactor_matrix(fam_order),
        "ec_number": mb.ec_matrix(fam_order),
        "go_mf": mb.go_matrices(fam_order, mb.PFAM_ACCESSIONS)["go_mf"],
        "kmer": mb.kmer_between(sub, cds, fam_order),
        "gc_content": mb.gc_between(sub, cds, fam_order),
    }
    return stack, fams, fam_order, baseline_mats, axis_of, run, MAMMAL_CENTROID


# ── scoring


def score_matrix(D: np.ndarray, baseline_mats: dict, axis_of: dict, approach: str,
                 n_perms: int) -> list[dict]:
    """Spearman ρ + Mantel p of one approach's F×F distances vs every baseline."""
    D_u = upper_triangle(D)
    rows = []
    for name, B in baseline_mats.items():
        B_u = upper_triangle(B)
        ok = np.isfinite(D_u) & np.isfinite(B_u)
        if ok.sum() < 3 or np.ptp(D_u[ok]) == 0 or np.ptp(B_u[ok]) == 0:
            rho, p_mantel = float("nan"), float("nan")
        else:
            rho = spearmanr(D_u[ok], B_u[ok]).statistic
            p_mantel = mantel_test(D, B, n_perms=n_perms)[1]
        rows.append({"approach": approach, "baseline": name,
                     "axis": axis_of.get(name), "spearman_rho": rho, "p_mantel": p_mantel})
    return rows


def convergent_ranks(fam_order: list[str], approaches: dict[str, np.ndarray]) -> pd.DataFrame:
    """Percentile rank (0=closest) of each curated convergent pair, per approach."""
    pos = {f: i for i, f in enumerate(fam_order)}
    iu = np.triu_indices(len(fam_order), 1)
    pair_ix = {frozenset({fam_order[i], fam_order[j]}): p
               for p, (i, j) in enumerate(zip(*iu))}
    pct = {}
    for name, D in approaches.items():
        u = D[iu]
        if not np.isfinite(u).all() or np.ptp(u) == 0:
            continue
        pct[name] = 100.0 * (rankdata(u, method="average") - 1) / (len(u) - 1)
    rows = []
    for a, b, kind in CONVERGENT_PAIRS:
        if a not in pos or b not in pos:
            continue
        p = pair_ix[frozenset({a, b})]
        row = {"family_a": a, "family_b": b, "relationship": kind}
        for name in approaches:
            row[f"{name}_pctile"] = round(float(pct[name][p]), 1) if name in pct else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(exp: str, layers: list[int] | None, alphas: tuple[float, ...],
                   n_perms: int, solver: str, reg: float | None) -> None:
    if exp in HUMAN_RUNS:
        stack, fams, fam_order, base, axis_of, run, cen_name = load_human(exp)
    else:
        stack, fams, fam_order, base, axis_of, run, cen_name = load_mammal(exp)

    n_layers = stack.shape[0]
    todo = layers if layers is not None else list(range(n_layers))
    print(f"[{exp}] {len(fam_order)} families, {len(todo)} layers -> {run}")
    print(f"  families: {fam_order}")

    for L in todo:
        blocks = run / f"blocks{L}"
        if not blocks.exists():
            print(f"  blocks{L}: run dir missing, skip")
            continue
        cen_f = blocks / cen_name
        if not cen_f.exists():
            print(f"  blocks{L}: no {cen_name}, skip")
            continue
        geo = pd.read_csv(cen_f, index_col=0).reindex(index=fam_order, columns=fam_order).values

        res = compute_ot_matrices(stack[L], fams, fam_order, alphas=alphas,
                                  solver=solver, reg=reg)
        res.to_run_dir(blocks)

        approaches = {"geodesic": geo, **res.matrices}
        rows = []
        for name, D in approaches.items():
            rows += score_matrix(D, base, axis_of, name, n_perms)
        pd.DataFrame(rows).to_csv(blocks / "between_family_ot_scores.csv", index=False)
        convergent_ranks(fam_order, approaches).to_csv(
            blocks / "convergent_pair_ranks_ot.csv", index=False)
        print(f"  blocks{L}: scored {len(approaches)} approaches × {len(base)} baselines "
              f"(OT compute {res.meta['runtime_seconds']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=EXPERIMENTS, help="single experiment to score")
    ap.add_argument("--all", action="store_true", help="score every registered experiment")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="layer indices (default: all layers present in the cache)")
    ap.add_argument("--alphas", nargs="*", type=float, default=list(DEFAULT_ALPHAS))
    ap.add_argument("--n-perms", type=int, default=9999)
    ap.add_argument("--solver", choices=["exact", "entropic"], default="exact")
    ap.add_argument("--reg", type=float, default=None, help="entropic regularization (solver=entropic)")
    args = ap.parse_args()
    if not args.experiment and not args.all:
        ap.error("pass --experiment <name> or --all")
    exps = EXPERIMENTS if args.all else [args.experiment]
    for exp in exps:
        run_experiment(exp, args.layers, tuple(args.alphas), args.n_perms,
                       args.solver, args.reg)


if __name__ == "__main__":
    main()
