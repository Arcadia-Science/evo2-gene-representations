"""Composition-control preservation for the mammalian ortholog panel, at every layer."""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
from pathlib import Path

# One BLAS thread per process: the W2 pass below fans 32 layers out to a fork pool, and without this
# every worker would spawn a full OpenMP team and thrash. Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
from control_tables import write_control_table  # noqa: E402
from geodesic_utils import (  # noqa: E402
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
)
from ot_between_family import compute_ot_matrices  # noqa: E402

# Blocks whose cached activations are numerically degenerate: mean L2 norm runs 1.13e1 at block 27
# to 2.41e12 at 30-31, and blocks 30/31 are byte-identical (the tap writes the last block twice), so
# any "block 31" number is really block 30. Still scored, since dropping rows would break the
# per-layer tables, but flagged so summaries can exclude them.
DEGENERATE_BLOCKS = (28, 29, 30, 31)

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE_ROOT = ROOT / "data" / "cache" / "mammal_embed"
CONTROLS = [
    "gc_match",
    "dinuc_shuffle",
    "kmer4_shuffle",
    "kmer6_shuffle",
    # transcript_cdsmask only — both need a reading frame. missense_subset is nested inside
    # synonymous_recode (it edits only bases the recode edited), so it is the weaker
    # perturbation on BOTH axes at once, not protein alone. Read one-sidedly: rho falling
    # below the recode means protein, since nucleotide loss cannot explain a drop; rho
    # holding is weak evidence, because 72% of the protein survives.
    "synonymous_recode",
    "missense_subset",
    # The matched pair. Both arms edit the same eligible position-3 codons — eligibility
    # depends only on the source codon — so they share edited positions and base-change rate
    # by construction and differ only in whether the protein survives. Two-sided: missense
    # falling below syn means protein, holding means nucleotide. Read them against each
    # other, never against synonymous_recode, whose rate is different.
    "paired_p3_syn",
    "paired_p3_missense",
]
MIN_SP = 10


# Layers are processed in chunks so only CHUNK layers per condition are resident. A full stack is
# 6.4 GB and this needs the natural plus one per control, which with the geodesic matrices exceeded
# the machine and was OOM-killed silently at 48 families.
LAYER_CHUNK = 4  # 6 conditions x 4 layers x 12,294 x 4096 x 4B ~ 4.8 GB resident (6th = the
# cdsmask-only synonymous_recode rung; 5 on arms that do not have it)


def load_stack(arm_dir: str, keys: list[str], layers: list[int] | None = None):
    """(len(layers), N, H) for the given keys, plus the keys actually present."""
    cache = CACHE_ROOT / arm_dir
    vecs, present = [], []
    for k in keys:
        p = cache / f"{k}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r")
            vecs.append(np.array(a if layers is None else a[layers]))
            present.append(k)
    return (np.stack(vecs, axis=1) if vecs else None), present


def present_keys(arm_dir: str, keys: list[str]) -> list[str]:
    """Keys whose .npy exists, without reading any array data (existence check only)."""
    cache = CACHE_ROOT / arm_dir
    return [k for k in keys if (cache / f"{k}.npy").exists()]


def upper(A, idx):
    return A[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]


def between_ut_w2(emb: np.ndarray, fam_arr: np.ndarray, fams: list[str], iu_fam):
    """Upper triangle of the all-gene Wasserstein (W2) between-family matrix for one layer."""
    return compute_ot_matrices(emb, fam_arr, fams, alphas=()).matrices["wasserstein"][iu_fam]


# Set by _w2_init in each forked worker so the (n_layers, N, D) chunk stacks are inherited
# copy-on-write rather than pickled per task — at 11k loci a chunk is ~5.5 GB.
_W2_STACKS: dict | None = None
_W2_ARGS: tuple | None = None


def _w2_init(stacks, fam_arr, fams, iu_fam):
    global _W2_STACKS, _W2_ARGS
    _W2_STACKS, _W2_ARGS = stacks, (fam_arr, fams, iu_fam)


def _w2_task(task):
    cond, layer, j = task
    return (cond, layer), between_ut_w2(_W2_STACKS[cond][j], *_W2_ARGS)


def write_control_summary(sweep_root: Path) -> Path:
    """Aggregate per-family control preservation for Figure 12."""
    rows = []
    for block_dir in sorted(sweep_root.glob("blocks*")):
        layer_text = block_dir.name.removeprefix("blocks")
        if not layer_text.isdigit():
            continue
        layer = int(layer_text)
        table = block_dir / "controls" / "control_within_scores.csv"
        if not table.exists():
            continue
        scores = pd.read_csv(table)
        recovery_col = "rho_geodesic_speciestree"
        for condition, group in scores.groupby("condition"):
            rows.append(
                {
                    "panel": "mammal_cdsmask",
                    "layer": layer,
                    "condition": condition,
                    "n_families": int(group["family"].nunique()),
                    "rho_preservation": group["rho_geodesic_vs_natural"].mean(),
                    "rho_recovery": group[recovery_col].mean(),
                }
            )

        natural_path = block_dir / "within_family_speciestree.csv"
        natural_recovery = np.nan
        if natural_path.exists():
            natural = pd.read_csv(natural_path)
            families = set(scores["family"])
            natural_recovery = natural.loc[
                natural["family"].isin(families), "spearman_geodesic_speciestree"
            ].mean()
        layer_rows = [row for row in rows if row["layer"] == layer]
        natural_row = next((row for row in layer_rows if row["condition"] == "natural"), None)
        if natural_row:
            if not np.isfinite(natural_row["rho_recovery"]):
                natural_row["rho_recovery"] = natural_recovery
        else:
            rows.append(
                {
                    "panel": "mammal_cdsmask",
                    "layer": layer,
                    "condition": "natural",
                    "n_families": int(scores["family"].nunique()),
                    "rho_preservation": 1.0,
                    "rho_recovery": natural_recovery,
                }
            )

    if not rows:
        raise RuntimeError(f"no control score tables under {sweep_root}")
    out = sweep_root / "control_rho_by_layer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # Cache and result paths share the arm name.
    ap.add_argument(
        "--arm", default="transcript", choices=["transcript", "cds", "transcript_cdsmask"]
    )
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(32)))
    ap.add_argument(
        "--between-metric",
        choices=["wasserstein", "centroid_geodesic"],
        default="wasserstein",
        help="between-family family-distance metric (default wasserstein since "
        "2026-08-26; see the module docstring for the measured reason). Switching "
        "this re-derives cached between-family triangles but reuses the expensive "
        "within-family locus geodesics. NOTE _run_cdsmask_missense_subset.sh calls "
        "this script unattended when its embed finishes, so that run will score "
        "under W2 and every between-family number on this arm will change; the "
        "old values are reproducible with --between-metric centroid_geodesic.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(10, (os.cpu_count() or 4) - 6)),
        help="parallel W2 solves; default leaves 6 cores for any concurrent GPU embed",
    )
    args = ap.parse_args()

    man = pd.read_csv(OUT / "complete_manifest.csv")
    man["key"] = man.group + "__" + man.species
    pat = pd.read_csv(OUT / "tree" / "species_patristic.csv", index_col=0)
    # natural loci present (defines the shared locus set) — existence only, no data read yet
    keys = present_keys(args.arm, man["key"].tolist())
    kmeta = man.set_index("key").loc[keys]
    kpos = {k: i for i, k in enumerate(keys)}
    fams = sorted(kmeta.family.unique())
    fam_arr = kmeta.family.to_numpy()
    # ortholog groups with >=MIN_SP species (for within-group preservation/recovery)
    groups = {}
    for (fam, grp), sub in kmeta.reset_index().groupby(["family", "group"]):
        if len(sub) >= MIN_SP:
            groups[grp] = (fam, sub["key"].tolist(), sub["species"].tolist())
    print(
        f"{len(keys)} natural loci; {len(groups)} groups >=+{MIN_SP}sp; {len(fams)} families",
        flush=True,
    )

    # which controls are fully embedded (file-existence check; stacks are loaded per chunk below)
    ready = []
    for c in CONTROLS:
        pk = present_keys(f"{args.arm}_{c}", keys)
        if len(pk) == len(keys):
            ready.append(c)
        else:
            print(f"  control {c}: {len(pk)}/{len(keys)} embedded — skipping for now", flush=True)
    if not ready:
        sys.exit("no complete control stacks yet")

    sweep_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}"

    # Cache the reduced group and centroid triangles, keyed by panel contents.
    grp_order = list(groups)
    # Include the centroid graph rule in cache validation.
    panel_id = hashlib.sha1(
        (
            "\n".join(keys)
            + "||"
            + "\n".join(f"{g}:{','.join(groups[g][1])}" for g in grp_order)
            + "||"
            + ",".join(fams)
        ).encode()
    ).hexdigest()[:16]
    # Track the between-family metric separately so within-family geodesics remain reusable.
    # Keep the persisted `min_connected` marker for cache compatibility.
    cen_k = "min_connected" if args.between_metric == "centroid_geodesic" else "wasserstein_allgene"
    geo_cache = ROOT / "data" / "cache" / "mammal_controls_geo" / args.arm
    geo_cache.mkdir(parents=True, exist_ok=True)
    iu_fam = np.triu_indices(len(fams), 1)

    def reduce_condition(cond: str, emb: np.ndarray | None, layer: int):
        """
        (per-group upper-triangle geodesics, centroid upper triangle) for one (condition, layer).
        """
        # Validate group geodesics and centroid triangles independently.
        p = geo_cache / f"{cond}_L{layer}.npz"
        vecs = cen = None
        if p.exists():
            z = np.load(p, allow_pickle=False)
            if str(z["panel_id"]) == panel_id:
                off = z["offsets"]
                vecs = [z["groups_flat"][off[i] : off[i + 1]] for i in range(len(grp_order))]
                # Treat entries without a centroid rule marker as stale.
                if "centroid_k" in z.files and str(z["centroid_k"]) == cen_k:
                    cen = z["centroid_ut"]
            else:
                print(
                    f"    cache stale for {cond} L{layer} (panel changed) — recomputing", flush=True
                )
        if vecs is not None and cen is not None:
            return {g: v for g, v in zip(grp_order, vecs, strict=False)}, cen
        if emb is None:
            return None
        if vecs is None:  # expensive half missing
            _, W = find_min_connected_k(emb, k_min=3)
            geo = compute_geodesic(W)
            vecs = [upper(geo, [kpos[k] for k in groups[g][1]]) for g in grp_order]
            del geo
        else:
            print(
                f"    {cond} L{layer}: reusing cached locus geodesic, rebuilding between-family "
                f"triangle (metric={cen_k})",
                flush=True,
            )
        # Served from the batched W2 pass when there is one; otherwise computed here. The batch is
        # only a speed path — the value is identical either way.
        cen = w2_batch.pop((cond, layer), None)
        if cen is None:
            cen = (
                compute_centroid_geodesic(emb, fam_arr, fams)[iu_fam]
                if args.between_metric == "centroid_geodesic"
                else between_ut_w2(emb, fam_arr, fams, iu_fam)
            )
        offsets = np.cumsum([0] + [len(v) for v in vecs])
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            panel_id=panel_id,
            centroid_k=cen_k,
            offsets=offsets,
            groups_flat=np.concatenate(vecs),
            centroid_ut=cen,
        )
        tmp.replace(p)
        return {g: v for g, v in zip(grp_order, vecs, strict=False)}, cen

    def chunk_needs(cond: str, chunk: list[int]) -> bool:
        """Does any layer in this chunk still need a geodesic for `cond`? If not, its stack is
        never loaded — which is what makes the incremental re-run cheap in I/O and RAM too."""
        return any(reduce_condition(cond, None, L) is None for L in chunk)

    # Defined before the loop because reduce_condition closes over it and chunk_needs calls that
    # closure before the batch for a chunk has been built.
    w2_batch: dict = {}
    chunks = [args.layers[i : i + LAYER_CHUNK] for i in range(0, len(args.layers), LAYER_CHUNK)]
    for chunk in chunks:
        # Load ONLY the conditions with at least one uncached layer in this chunk.
        need_nat = chunk_needs(args.arm, chunk)
        need_ctl = [c for c in ready if chunk_needs(f"{args.arm}_{c}", chunk)]
        nat_chunk = load_stack(args.arm, keys, chunk)[0] if need_nat else None
        ctrl_chunk = {c: load_stack(f"{args.arm}_{c}", keys, chunk)[0] for c in need_ctl}
        n_hit = (1 + len(ready)) - (int(need_nat) + len(need_ctl))
        print(
            f"  layers {chunk}: loaded {int(need_nat) + len(need_ctl)} conditions "
            f"({n_hit} served from the reduced-geodesic cache)",
            flush=True,
        )

        # Batched W2: exact EMD over ~950 family pairs costs ~30 s per (condition, layer) at this
        # panel size, and the pairs are independent, so the chunk's outstanding (condition, layer)
        # solves are farmed to a fork pool before the sequential loop below consumes them. The
        # centroid path is cheap enough that it stays inline.
        w2_batch.clear()
        if args.between_metric == "wasserstein" and args.workers > 1:
            stacks = {
                c: s
                for c, s in [(args.arm, nat_chunk)]
                + [(f"{args.arm}_{c}", ctrl_chunk[c]) for c in need_ctl]
                if s is not None
            }
            tasks = [
                (cond, L, j)
                for cond in stacks
                for j, L in enumerate(chunk)
                if reduce_condition(cond, None, L) is None
            ]
            if tasks:
                print(
                    f"    W2: {len(tasks)} (condition, layer) solves on "
                    f"{min(args.workers, len(tasks))} workers",
                    flush=True,
                )
                ctx = mp.get_context("fork")
                with ctx.Pool(
                    min(args.workers, len(tasks)),
                    initializer=_w2_init,
                    initargs=(stacks, fam_arr, fams, iu_fam),
                ) as pool:
                    w2_batch.update(dict(pool.map(_w2_task, tasks)))
            del stacks
        for j, L in enumerate(chunk):
            nat_grp, nat_cen_ut = reduce_condition(
                args.arm, nat_chunk[j] if nat_chunk is not None else None, L
            )
            within_rows = [
                {"condition": "natural", "family": f, "rho_geodesic_vs_natural": 1.0} for f in fams
            ]
            # Mirror the former centroid column for reader compatibility; record metric provenance.
            between_rows = [
                {
                    "condition": "natural",
                    "rho_vs_natural_between": 1.0,
                    "rho_vs_natural_centroid": 1.0,
                    "between_metric": args.between_metric,
                    "degenerate_block": L in DEGENERATE_BLOCKS,
                }
            ]
            for c in ready:
                cs = ctrl_chunk.get(c)
                ctrl_grp, ctrl_cen_ut = reduce_condition(
                    f"{args.arm}_{c}", cs[j] if cs is not None else None, L
                )
                # per-group preservation + recovery, aggregated per family
                per_fam_pres, per_fam_rec = {f: [] for f in fams}, {f: [] for f in fams}
                for grp, (fam, _gk, gsp) in groups.items():
                    ng, cg = nat_grp[grp], ctrl_grp[grp]
                    P = pat.loc[gsp, gsp].values[np.triu_indices(len(gsp), 1)]
                    ok = np.isfinite(ng) & np.isfinite(cg)
                    if ok.sum() >= 6 and np.ptp(cg[ok]) > 0 and np.ptp(ng[ok]) > 0:
                        per_fam_pres[fam].append(spearmanr(cg[ok], ng[ok]).statistic)
                    okr = np.isfinite(cg) & np.isfinite(P)
                    if okr.sum() >= 6 and np.ptp(cg[okr]) > 0 and np.ptp(P[okr]) > 0:
                        per_fam_rec[fam].append(spearmanr(cg[okr], P[okr]).statistic)
                for f in fams:
                    within_rows.append(
                        {
                            "condition": c,
                            "family": f,
                            "rho_geodesic_vs_natural": np.mean(per_fam_pres[f])
                            if per_fam_pres[f]
                            else np.nan,
                            "rho_geodesic_speciestree": np.mean(per_fam_rec[f])
                            if per_fam_rec[f]
                            else np.nan,
                        }
                    )
                nz = np.isfinite(nat_cen_ut) & np.isfinite(ctrl_cen_ut)
                brho = (
                    spearmanr(nat_cen_ut[nz], ctrl_cen_ut[nz]).statistic
                    if nz.sum() >= 3
                    else np.nan
                )
                between_rows.append(
                    {
                        "condition": c,
                        "rho_vs_natural_between": brho,
                        "rho_vs_natural_centroid": brho,
                        "between_metric": args.between_metric,
                        "degenerate_block": L in DEGENERATE_BLOCKS,
                    }
                )
            cdir = sweep_root / f"blocks{L}" / "controls"
            cdir.mkdir(parents=True, exist_ok=True)
            # Upsert so incomplete controls do not remove existing rows.
            gen = f"mammal_controls_score:{args.arm}"
            write_control_table(
                cdir / "control_within_scores.csv", pd.DataFrame(within_rows), generator=gen
            )
            write_control_table(
                cdir / "control_between_scores.csv", pd.DataFrame(between_rows), generator=gen
            )
        del nat_chunk, ctrl_chunk  # (either may be None / empty when fully cache-served)
    print(
        f"wrote controls/ scores to {len(args.layers)} run dirs ({ready}) -> {sweep_root}",
        flush=True,
    )
    summary = write_control_summary(sweep_root)
    print(f"wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
