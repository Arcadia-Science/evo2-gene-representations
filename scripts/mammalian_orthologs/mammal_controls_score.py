"""Composition-control preservation for the mammalian ortholog panel, at every layer."""

from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import os
import shutil
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
from controls.make_control_sequences import CDSMASK_CONTROLS  # noqa: E402
from ot_between_family import (  # noqa: E402
    W2_CONTRACT_VERSION,
    angular_distance,
    compute_ot_matrices,
    l2_normalize,
)

# Bump when the cache payload's meaning or layout changes. Entries written by an older schema are
# not reused: a reduced triangle carries no self-description, so an incompatible one is silently
# plausible rather than obviously broken.
CACHE_SCHEMA = "controls-cache-v2"

# Blocks whose cached activations are numerically degenerate: mean L2 norm runs 1.13e1 at block 27
# to 2.41e12 at 30-31, and blocks 30/31 are byte-identical (the tap writes the last block twice), so
# any "block 31" number is really block 30. Still scored, since dropping rows would break the
# per-layer tables, but flagged so summaries can exclude them.
DEGENERATE_BLOCKS = (28, 29, 30, 31)

OUT = ROOT / "data" / "mammalian_orthologs"
CACHE_ROOT = ROOT / "data" / "cache" / "mammal_embed"
MIN_SP = 10


# Layers are processed in chunks because loading every complete condition at once exceeds memory.
LAYER_CHUNK = 4  # Natural + eight complete controls x four layers is about 6.7 GB resident.


def write_control_table(path: Path, new: pd.DataFrame, *, generator: str) -> pd.DataFrame:
    """Upsert the scorer's conditions while preserving other existing rows and a backup."""
    new = new.copy()
    if "condition" not in new.columns:
        raise ValueError(f"{path.name}: new frame has no 'condition' column; refusing to write")
    new["written_at"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d")
    new["generator"] = generator

    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        old = pd.read_csv(path)
        if "condition" in old.columns:
            kept = old[~old["condition"].isin(set(new["condition"].unique()))]
            if not kept.empty:
                for column in ("written_at", "generator"):
                    if column not in kept.columns:
                        kept = kept.assign(**{column: "unknown"})
                new = pd.concat([kept, new], ignore_index=True, sort=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


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
        recovery_col = "rho_angular_speciestree"
        for condition, group in scores.groupby("condition"):
            rows.append(
                {
                    "panel": "mammal_cdsmask",
                    "layer": layer,
                    "condition": condition,
                    "n_families": int(group["family"].nunique()),
                    "rho_preservation": group["rho_angular_vs_natural"].mean(),
                    "rho_recovery": group[recovery_col].mean(),
                }
            )

        natural_path = block_dir / "within_family_speciestree_angular.csv"
        natural_recovery = np.nan
        if natural_path.exists():
            natural = pd.read_csv(natural_path)
            families = set(scores["family"])
            natural_recovery = natural.loc[
                natural["family"].isin(families), "spearman_angular_speciestree"
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
    for c in CDSMASK_CONTROLS:
        pk = present_keys(f"{args.arm}_{c}", keys)
        if len(pk) == len(keys):
            ready.append(c)
        else:
            print(f"  control {c}: {len(pk)}/{len(keys)} embedded — skipping for now", flush=True)
    if not ready:
        sys.exit("no complete control stacks yet")

    sweep_root = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}"

    # Cache reduced angular group triangles and W2 family triangles separately from untouched
    # legacy geodesic artifacts.
    grp_order = list(groups)
    group_indices = {g: [kpos[k] for k in groups[g][1]] for g in grp_order}
    panel_id = hashlib.sha1(
        (
            "angular\n"
            + "\n".join(keys)
            + "||"
            + "\n".join(f"{g}:{','.join(groups[g][1])}" for g in grp_order)
            + "||"
            + ",".join(fams)
        ).encode()
    ).hexdigest()[:16]
    score_cache = ROOT / "data" / "cache" / "mammal_controls_angular" / args.arm
    score_cache.mkdir(parents=True, exist_ok=True)
    iu_fam = np.triu_indices(len(fams), 1)

    # ── cache provenance contract ───────────────────────────────────────────────────────────────
    # A reduced triangle is a bare vector of numbers: it stays numerically plausible when the
    # family coordinate system underneath it changes, so nothing about a wrong cache LOOKS wrong.
    # Every field below must match before an entry is reused, and a mismatch names itself rather
    # than reporting a generic "panel changed".
    expected_group_lens = [
        len(group_indices[g]) * (len(group_indices[g]) - 1) // 2 for g in grp_order
    ]
    contract = {
        "cache_schema": CACHE_SCHEMA,
        "w2_contract": W2_CONTRACT_VERSION,
        "panel_id": panel_id,
        "metric": "angular+wasserstein",
        "between_metric": "wasserstein",
        "fam_order": np.asarray(fams, dtype=object).astype(str),
        "group_order": np.asarray(grp_order, dtype=object).astype(str),
        "offsets": np.cumsum([0] + expected_group_lens),
        "between_len": len(iu_fam[0]),
    }

    NEW_FIELDS = ("cache_schema", "w2_contract", "metric", "fam_order", "group_order")

    def _shape_ok(z) -> str | None:
        """Payload checks that apply to any schema: the vectors must be the panel's shape."""
        if str(z["panel_id"]) != panel_id:
            return f"panel_id {str(z['panel_id'])!r} != {panel_id!r}"
        if str(z["between_metric"]) != "wasserstein":
            return f"between_metric {str(z['between_metric'])!r} != 'wasserstein'"
        if not np.array_equal(z["offsets"], contract["offsets"]):
            return "offsets differ (group structure changed)"
        if "between_ut" not in z.files or len(z["between_ut"]) != contract["between_len"]:
            got = len(z["between_ut"]) if "between_ut" in z.files else "absent"
            return f"between_ut length {got} != {contract['between_len']}"
        if len(z["groups_flat"]) != int(contract["offsets"][-1]):
            return f"groups_flat length {len(z['groups_flat'])} != {int(contract['offsets'][-1])}"
        return None

    def cache_status(z) -> tuple[str, str | None]:
        """('ok', None) reusable, ('upgrade', None) pre-schema but compatible, ('stale', why).

        The pre-schema entries carry panel_id, which already hashes the ordered locus keys, the
        group structure AND the ordered family labels. So a matching panel_id plus the right
        vector lengths does establish that the payload belongs to this panel; what those entries
        lack is the ability to SAY so. Upgrading rewrites the provenance around an unchanged
        payload rather than discarding ~2.4 h of exact-EMD compute. Anything else recomputes.
        """
        if any(f not in z.files for f in ("panel_id", "between_metric", "offsets", "groups_flat")):
            return "stale", "pre-panel_id entry"
        if (why := _shape_ok(z)) is not None:
            return "stale", why
        legacy = [f for f in NEW_FIELDS if f not in z.files]
        if legacy:
            # Partially-written entries are not upgradeable: absence must be all-or-nothing.
            if len(legacy) != len(NEW_FIELDS):
                return "stale", f"half-written provenance, missing {legacy}"
            return "upgrade", None
        for field_ in ("cache_schema", "w2_contract", "metric"):
            if str(z[field_]) != str(contract[field_]):
                return "stale", f"{field_} {str(z[field_])!r} != {str(contract[field_])!r}"
        for field_ in ("fam_order", "group_order"):
            if not np.array_equal(z[field_], contract[field_]):
                return "stale", f"{field_} differs (family or group coordinate system changed)"
        return "ok", None

    def _write_entry(path: Path, vecs: list[np.ndarray], between: np.ndarray) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **contract, groups_flat=np.concatenate(vecs), between_ut=between)
        tmp.replace(path)

    def reduce_condition(cond: str, emb: np.ndarray | None, layer: int):
        """Return per-group angular triangles and the between-family W2 triangle."""
        p = score_cache / f"{cond}_L{layer}.npz"
        vecs = between = None
        if p.exists():
            z = np.load(p, allow_pickle=False)
            status, reason = cache_status(z)
            if status in ("ok", "upgrade"):
                off = z["offsets"]
                vecs = [z["groups_flat"][off[i] : off[i + 1]] for i in range(len(grp_order))]
                between = z["between_ut"]
                if status == "upgrade":
                    z.close()
                    _write_entry(p, vecs, between)
                    print(f"    cache provenance upgraded for {cond} L{layer}", flush=True)
            else:
                print(f"    cache stale for {cond} L{layer}: {reason} — recomputing", flush=True)
        if vecs is not None and between is not None:
            return {g: v for g, v in zip(grp_order, vecs, strict=False)}, between
        if emb is None:
            return None
        if vecs is None:
            vecs = []
            for g in grp_order:
                U = l2_normalize(emb[group_indices[g]])
                A = angular_distance(U, U)
                vecs.append(A[np.triu_indices(len(U), 1)])
        between = w2_batch.pop((cond, layer), None)
        if between is None:
            between = between_ut_w2(emb, fam_arr, fams, iu_fam)
        offsets = np.cumsum([0] + [len(v) for v in vecs])
        if not np.array_equal(offsets, contract["offsets"]):
            raise SystemExit(
                f"{cond} L{layer}: reduced group sizes {offsets.tolist()} do not match the "
                f"panel contract {contract['offsets'].tolist()}"
            )
        _write_entry(p, vecs, between)
        return {g: v for g, v in zip(grp_order, vecs, strict=False)}, between

    def chunk_needs(cond: str, chunk: list[int]) -> bool:
        """Return whether any layer lacks an angular/W2 cache entry for this condition."""
        return any(reduce_condition(cond, None, layer) is None for layer in chunk)

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
            f"({n_hit} served from the angular/W2 cache)",
            flush=True,
        )

        # Batched W2: exact EMD over ~950 family pairs costs ~30 s per (condition, layer) at this
        # panel size, and the pairs are independent, so the chunk's outstanding (condition, layer)
        # solves are farmed to a fork pool before the sequential loop below consumes them.
        w2_batch.clear()
        if args.workers > 1:
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
            nat_grp, nat_between_ut = reduce_condition(
                args.arm, nat_chunk[j] if nat_chunk is not None else None, L
            )
            within_rows = [
                {"condition": "natural", "family": f, "rho_angular_vs_natural": 1.0} for f in fams
            ]
            between_rows = [
                {
                    "condition": "natural",
                    "rho_vs_natural_between": 1.0,
                    "between_metric": "wasserstein",
                    "degenerate_block": L in DEGENERATE_BLOCKS,
                }
            ]
            for c in ready:
                cs = ctrl_chunk.get(c)
                ctrl_grp, ctrl_between_ut = reduce_condition(
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
                            "rho_angular_vs_natural": np.mean(per_fam_pres[f])
                            if per_fam_pres[f]
                            else np.nan,
                            "rho_angular_speciestree": np.mean(per_fam_rec[f])
                            if per_fam_rec[f]
                            else np.nan,
                        }
                    )
                nz = np.isfinite(nat_between_ut) & np.isfinite(ctrl_between_ut)
                brho = (
                    spearmanr(nat_between_ut[nz], ctrl_between_ut[nz]).statistic
                    if nz.sum() >= 3
                    else np.nan
                )
                between_rows.append(
                    {
                        "condition": c,
                        "rho_vs_natural_between": brho,
                        "between_metric": "wasserstein",
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
