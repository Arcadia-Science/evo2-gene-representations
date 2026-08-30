"""Between-family control ladder scored by all-gene Wasserstein (W2) at every layer — the W2
counterpart of the run's `control_rho_by_layer.csv` figures.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Each worker gets ONE BLAS thread. Without this every worker would spawn 16 OpenMP threads and the
# pool would thrash. Must be set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from mammal_controls_score import load_stack, present_keys  # noqa: E402
from ot_between_family import compute_ot_matrices  # noqa: E402

DATA = ROOT / "data" / "mammalian_orthologs"
RUN = ROOT / "results" / "2026-07-16_mammalian-orthologs-transcript_cdsmask"
ARM = "transcript_cdsmask"
N_BLOCKS = 32

LADDER = [
    "gc_match",
    "dinuc_shuffle",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "synonymous_recode",
    "missense_subset",
]
MISSENSE = "missense_subset"
RECODE = "synonymous_recode"

# Measured nucleotide / amino-acid identity to source on THIS arm. Shuffles from
# results/.../control_sequence_identity.csv (pos_identity); the recode/missense pair measured
# directly off the run's own crc32-per-locus seeding, since the identity CSV has no missense row
# until the run's scoring step refreshes it.
NT_IDENTITY = {
    "gc_match": 0.257,
    "dinuc_shuffle": 0.262,
    "kmer4_shuffle": 0.266,
    "kmer6_shuffle": 0.273,
    "synonymous_recode": 0.771,
    "missense_subset": 0.876,
}
AA_IDENTITY = {
    "gc_match": 0.058,
    "dinuc_shuffle": 0.061,
    "kmer4_shuffle": 0.067,
    "kmer6_shuffle": 0.076,
    "synonymous_recode": 1.000,
    "missense_subset": 0.723,
}

MIN_LOCI_PER_FAMILY = 8

# Flag late blocks with degenerate activation norms; blocks 30 and 31 are byte-identical.
# Keep them in per-layer outputs but separate them in summaries.
DEGENERATE_BLOCKS = (28, 29, 30, 31)
DUPLICATE_OF = {31: 30}

# Set once per process by _init so workers can read the big stack copy-on-write instead of
# receiving it through the pickling channel.
_STACK: np.ndarray | None = None
_FAMS: np.ndarray | None = None
_ORDER: list[str] | None = None


def _init(stack, fam_arr, fam_order):
    global _STACK, _FAMS, _ORDER
    _STACK, _FAMS, _ORDER = stack, fam_arr, fam_order


def _w2_for_layer(layer: int) -> tuple[int, np.ndarray]:
    """Upper triangle of the F x F all-gene W2 matrix for one layer of the resident stack."""
    iu = np.triu_indices(len(_ORDER), 1)
    res = compute_ot_matrices(_STACK[layer], _FAMS, _ORDER, alphas=())
    return layer, res.matrices["wasserstein"][iu]


def all_layer_w2(arm_dir: str, shared, fam_arr, fam_order, layers, workers: int) -> dict:
    """{layer: upper-triangle W2} for one condition, loading its cache exactly once."""
    t0 = time.perf_counter()
    stack, present = load_stack(arm_dir, shared, layers)
    if stack is None or len(present) != len(shared):
        raise RuntimeError(
            f"{arm_dir}: expected {len(shared)} loci, got {0 if stack is None else len(present)}"
        )
    load_s = time.perf_counter() - t0
    # `layers` selected the rows at load time, so stack row i is layers[i], not block i.
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(stack, fam_arr, fam_order)) as pool:
        out = dict(pool.map(_w2_for_layer, range(len(layers))))
    del stack
    print(
        f"    load {load_s:5.1f}s  +  {len(layers)} layers in "
        f"{time.perf_counter() - t0 - load_s:6.1f}s",
        flush=True,
    )
    return {layers[i]: v for i, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(N_BLOCKS)))
    ap.add_argument("--controls", nargs="+", default=LADDER, choices=LADDER)
    ap.add_argument("--manifest", default="complete_manifest_cap400.csv")
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(10, (os.cpu_count() or 4) - 6)),
        help="parallel layer solves; default leaves 6 cores for the GPU embedder + OS",
    )
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    man = pd.read_csv(DATA / args.manifest)
    keys = sorted(man.group + "__" + man.species)
    fam_of = dict(zip(man.group + "__" + man.species, man.family, strict=False))

    arms = {None: ARM, **{c: f"{ARM}_{c}" for c in args.controls}}
    have = {c: set(present_keys(a, keys)) for c, a in arms.items()}
    shared = sorted(set.intersection(*have.values()))
    print(f"manifest {args.manifest}   workers {args.workers}   layers {len(args.layers)}")
    for c, _a in arms.items():
        print(f"  {'natural' if c is None else c:<18} {len(have[c]):>6,} cached")

    # The ceiling is the NATURAL cache: loci absent from it were never embedded in any arm (they
    # failed CDS-mask QC), so they are not a cache still filling and must not trigger the INTERIM
    # caveat. Comparing against the manifest's row count instead marks a finished run as partial.
    n_target = len(have[None])
    fam_arr0 = np.array([fam_of[k] for k in shared])
    counts = pd.Series(fam_arr0).value_counts()
    fam_order = sorted(counts[counts >= MIN_LOCI_PER_FAMILY].index)
    keep = np.isin(fam_arr0, fam_order)
    shared = [k for k, m in zip(shared, keep, strict=False) if m]
    fam_arr = fam_arr0[keep]
    print(
        f"\nshared locus set: {len(shared):,} of {len(keys):,}   "
        f"families >= {MIN_LOCI_PER_FAMILY} loci: {len(fam_order)} of {man.family.nunique()}   "
        f"family pairs: {len(fam_order) * (len(fam_order) - 1) // 2}\n"
    )

    t0 = time.perf_counter()
    print("  natural ...", flush=True)
    nat = all_layer_w2(ARM, shared, fam_arr, fam_order, args.layers, args.workers)

    rows = []
    for c in args.controls:
        print(f"  {c} ...", flush=True)
        w = all_layer_w2(arms[c], shared, fam_arr, fam_order, args.layers, args.workers)
        for L in args.layers:
            rows.append(
                {
                    "layer": L,
                    "condition": c,
                    "metric": "wasserstein",
                    "manifest": args.manifest,
                    "n_loci": len(shared),
                    "n_families": len(fam_order),
                    "n_target": n_target,
                    "n_family_pairs": len(fam_order) * (len(fam_order) - 1) // 2,
                    "between_preservation": spearmanr(nat[L], w[L]).correlation,
                    "nt_identity": NT_IDENTITY[c],
                    "aa_identity": AA_IDENTITY[c],
                }
            )
        t = pd.DataFrame(rows)  # checkpoint after every condition, so a kill loses at most one
        out = RUN / "controls_pilot"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"between_family_ot_by_layer{'_' + args.tag if args.tag else ''}"
        t.to_csv(out / f"{stem}.csv", index=False)

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")
    piv = t.pivot(index="layer", columns="condition", values="between_preservation")
    piv = piv[[c for c in LADDER if c in piv.columns]]
    print("\n" + piv.round(3).to_string())
    bad = sorted(set(piv.index) & set(DEGENERATE_BLOCKS))
    if bad:
        print(
            f"\n[note] blocks {bad}: residual norms reach ~1e12, and blocks 30/31 are "
            f"byte-identical (the Evo2 tap writes the last block twice), so the last two points "
            f"are ONE measurement plotted twice. Reported as their own row below."
        )
    if MISSENSE in piv and RECODE in piv:
        d = (piv[MISSENSE] - piv[RECODE]).drop(index=bad, errors="ignore").dropna()
        print(
            f"\nmissense − recode across {len(d)} mid-stack layers: mean {d.mean():+.3f}  "
            f"range [{d.min():+.3f}, {d.max():+.3f}]  lower in {(d < 0).sum()}/{len(d)}"
        )
        # The split is not post-hoc: the early blocks are where the shuffles themselves are still
        # near-natural, i.e. where this axis has little dynamic range to resolve anything.
        early, late = d.loc[:9], d.loc[10:]
        if len(early) and len(late):
            print(
                f"    blocks 0-9   mean {early.mean():+.4f}  lower in "
                f"{(early < 0).sum()}/{len(early)}"
            )
            print(
                f"    blocks 10-27 mean {late.mean():+.4f}  lower in {(late < 0).sum()}/{len(late)}"
            )
        print("  missense holds MORE nucleotide (0.876 vs 0.771), so a DROP cannot be nucleotide")
        print("  loss. No drop is weak evidence — 72% of the protein is still intact.")
    figure(t, out / stem, len(shared), len(fam_order))
    print(f"\n[wrote] {out}/{stem}.csv + .png/.pdf")


def figure(t: pd.DataFrame, path: Path, n_shared: int, n_fams: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mod, fn in (("arcadia_style", "setup"), ("plot_utils", "set_pub_style")):
        try:
            getattr(__import__(mod), fn)()
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [style] {mod}.{fn}() unavailable ({type(e).__name__}: {e})")
    import arcadia_style as acs

    col = {c: acs.CONTROL_COLORS.get(c, acs.CONTROL_FALLBACK) for c in t.condition.unique()}
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
    for c in [x for x in LADDER if x in set(t.condition)]:
        s = t[t.condition == c].sort_values("layer")
        ax.plot(
            s.layer,
            s.between_preservation,
            "-o",
            ms=3,
            lw=2.2 if c == MISSENSE else 1.3,
            color=col[c],
            zorder=3 if c == MISSENSE else 2,
            label=f"{c} (nt {NT_IDENTITY[c]:.2f}, aa {AA_IDENTITY[c]:.2f})",
        )
    ax.axhline(1.0, ls="--", lw=1.0, color=acs.REFERENCE_LINE, label="natural (self = 1.0)")
    ax.set_xlabel("Evo2 block")
    ax.set_ylabel(r"Spearman $\rho$ vs natural all-gene $W_2$")
    ax.set_title("Between-family preservation by layer, all-gene Wasserstein", fontsize=9)
    ax.legend(fontsize=6, loc="best")

    # The INTERIM caveat is driven by DATA, not hardcoded: it belongs on the figure only while a
    # control cache is still filling, and silently outliving that is how a provisional number gets
    # quoted as final.
    n_target = int(t.get("n_target", pd.Series([n_shared])).iloc[0])
    partial = n_shared < n_target
    fig.suptitle(
        f"Between-family control ladder by Wasserstein, {n_shared:,} loci / {n_fams} "
        + (
            f"families. INTERIM: {n_target - n_shared:,} loci of a control cache still filling."
            if partial
            else "families."
        ),
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
