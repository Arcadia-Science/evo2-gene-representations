"""Embed mammalian transcript spans while pooling only CDS positions."""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))
sys.path.insert(
    0, str(ROOT / "scripts" / "controls")
)  # composition-shuffle fns (make_control_sequences)
from evo2_embedding import (  # noqa: E402
    LAYER_NAMES,
    N_BLOCKS,
    forward_positions,
    load_model,
)
from make_control_sequences import dinuc_shuffle, gc_match, klet_shuffle  # noqa: E402

LOCI_DIR = ROOT / "data" / "mammalian_orthologs" / "loci"
CDS_POS = ROOT / "data" / "cache" / "mammal_cds_positions.json"
CACHE = ROOT / "data" / "cache" / "mammal_embed" / "transcript_cdsmask"
EMBED_DIM = 4096
EVO2_WINDOW = 8000
CONTROL_FNS = {
    "gc_match": gc_match,
    "dinuc_shuffle": dinuc_shuffle,
    "kmer4_shuffle": lambda seq, rng: klet_shuffle(seq, 4, rng),
    "kmer6_shuffle": lambda seq, rng: klet_shuffle(seq, 6, rng),
}
# The CDS mask adds frame-aware controls to the shared genomic control ladder.
# `missense_subset` changes a strict subset of the bases changed by `synonymous_recode`.
CDSMASK_CONTROLS = [
    "gc_match",
    "dinuc_shuffle",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "synonymous_recode",
    "missense_subset",
    # The MATCHED PAIR: same eligible sites, same 14.8% rate, same codon-position
    # profile (100% p3), differing ONLY in protein outcome. Read against EACH
    # OTHER, never against synonymous_recode (different rate).
    "paired_p3_syn",
    "paired_p3_missense",
]
# Rungs that need per-family codon usage rather than a plain (seq, rng) shuffle.
FAMILY_USAGE_CONTROLS = {"synonymous_recode", "missense_subset", "paired_p3_syn"}
# The between-family panel (families with >=8 embedded loci; the set mammal_between scores).
DEFAULT_FAMILIES = [
    "olfactory_receptors",
    "hox",
    "cytochrome_p450",
    "ras_gtpases",
    "carbonic_anhydrase",
    "opsins",
    "globins",
    "nitric_oxide_synthase",
    "heme_oxygenase",
]
MAX_WINDOWS = 24


def _window_bounds(L: int, window: int = EVO2_WINDOW, max_windows: int = MAX_WINDOWS):
    """Return contiguous windows, capped by evenly spaced sampling for long loci."""
    if L <= window:
        return [(0, L)]
    n = math.ceil(L / window)
    if n <= max_windows:
        return [(round(i * L / n), round((i + 1) * L / n)) for i in range(n)]
    centres = np.linspace(window // 2, L - window // 2, max_windows)
    return [(int(c - window // 2), int(c + window // 2)) for c in centres]


def embed_cds_masked(
    seq: str, cds_pos: list[int], model, device: str, window: int = EVO2_WINDOW
) -> np.ndarray:
    """(N_BLOCKS, H): CDS-position hidden states in transcript order across the (capped) windows,
    second half mean-pooled.
    """
    L = len(seq)
    cds = np.array(sorted(p for p in cds_pos if 0 <= p < L))
    collected = {ln: [] for ln in LAYER_NAMES}
    seen: set[int] = set()
    for a, b in _window_bounds(L, window):
        in_win = [p for p in cds[(cds >= a) & (cds < b)] if p not in seen]
        if not in_win:
            continue
        seen.update(in_win)
        got = forward_positions(seq[a:b], model, device, [p - a for p in in_win])
        for ln in LAYER_NAMES:
            collected[ln].append(got[ln])
    out = np.zeros((N_BLOCKS, EMBED_DIM), dtype=np.float32)
    if not seen:  # no coding positions landed in any sampled window
        return out
    for i, ln in enumerate(LAYER_NAMES):
        allpos = np.concatenate(collected[ln], axis=0)  # (n_cds, H) transcript order
        half = len(allpos) // 2  # pool_second_half rule, on CDS positions
        out[i] = allpos[half:].mean(axis=0)
    return out


def build_family_usage(rows) -> dict[str, dict]:
    """Build per-family codon usage from the complete target set."""
    from collections import defaultdict

    from make_control_sequences import build_family_codon_usage  # noqa: E402  (lazy)

    by_fam: dict[str, list[str]] = defaultdict(list)
    for _key, fam, seq, pos in rows:
        by_fam[fam].append("".join(seq[i] for i in sorted(p for p in pos if 0 <= p < len(seq))))
    return build_family_codon_usage(by_fam)


def shuffle_coding_in_place(
    seq: str,
    cds_pos: list[int],
    control: str,
    key: str,
    fam: str | None = None,
    fam_usage: dict[str, dict] | None = None,
) -> str:
    """The locus span with ONLY its coding positions replaced by shuffled coding content."""
    pos = sorted(p for p in cds_pos if 0 <= p < len(seq))
    coding = "".join(seq[i] for i in pos)
    if control.startswith("paired_p3_"):
        from make_control_sequences import paired_p3  # noqa: E402 (lazy)

        arm = "synonymous" if control == "paired_p3_syn" else "missense"
        if arm == "synonymous" and (fam_usage is None or fam is None):
            raise ValueError(f"{control} needs fam + fam_usage (see build_family_usage)")
        # Eligibility fixes the edited sites; the arm-specific seed selects alternatives.
        shuffled = paired_p3(
            coding,
            fam_usage[fam] if arm == "synonymous" else {},
            random.Random(zlib.crc32(f"{control}:{key}".encode())),
            arm,
        )
    elif control in FAMILY_USAGE_CONTROLS:
        from make_control_sequences import missense_subset, synonymous_recode  # noqa: E402 (lazy)

        if fam_usage is None or fam is None:
            raise ValueError(f"{control} needs fam + fam_usage (see build_family_usage)")
        # Recreate the deterministic recode so `missense_subset` remains nested per locus.
        recoded = synonymous_recode(
            coding, fam_usage[fam], random.Random(zlib.crc32(f"synonymous_recode:{key}".encode()))
        )
        shuffled = (
            recoded
            if control == "synonymous_recode"
            else missense_subset(
                coding, recoded, random.Random(zlib.crc32(f"{control}:{key}".encode()))
            )
        )
    else:
        rng = random.Random(f"{control}:{key}".__hash__() & 0xFFFFFFFF)
        shuffled = CONTROL_FNS[control](coding, rng)
    if len(shuffled) != len(coding):  # length-preserving by construction; assert the contract
        raise ValueError(
            f"{control} changed coding length for {key}: {len(coding)} -> {len(shuffled)}"
        )
    span = list(seq)
    for i, idx in enumerate(pos):
        span[idx] = shuffled[i]
    return "".join(span)


def load_target_loci(families: list[str], keys_from: str | None = None):
    """Load masked loci selected by the assembled manifest."""
    positions = json.loads(CDS_POS.read_text())
    manifest = keys_from or "complete_manifest.csv"
    selected = pd.read_csv(ROOT / "data" / "mammalian_orthologs" / manifest)
    allowed = set(selected.group + "__" + selected.species)
    rows = []
    for f in sorted(LOCI_DIR.glob("*.json")):
        key = f.stem
        if key not in positions or key not in allowed:
            continue
        d = json.loads(f.read_text())
        if d.get("family") not in families:
            continue
        rows.append((key, d["family"], d["locus_seq"], positions[key]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    ap.add_argument("--window", type=int, default=EVO2_WINDOW)
    ap.add_argument(
        "--keys-from",
        default=None,
        help="restrict to group__species keys in this manifest csv (e.g. "
        "complete_manifest_cap400.csv for the 400/family cap)",
    )
    ap.add_argument(
        "--control",
        default=None,
        choices=CDSMASK_CONTROLS,
        help="Composition control: shuffle each locus's CODING content in place before "
        "embedding (introns/UTRs left natural); cache dir is tagged _<control>.",
    )
    args = ap.parse_args()

    rows = load_target_loci(args.families, args.keys_from)
    cache = CACHE.with_name(f"{CACHE.name}_{args.control}") if args.control else CACHE
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (cache / f"{r[0]}.npy").exists()]
    print(
        f"{len(rows)} masked loci in {len(args.families)} families "
        f"(control={args.control}); {len(todo)} to embed -> {cache}",
        flush=True,
    )
    if not todo:
        print("nothing to embed (all cached).")
        return

    # Built from ALL rows (not just todo) so resuming does not change the usage table.
    fam_usage = None
    if args.control in FAMILY_USAGE_CONTROLS:
        fam_usage = build_family_usage(rows)
        print(
            f"codon usage from {len(rows)} natural CDS over {len(fam_usage)} families", flush=True
        )

    device = "cuda"
    model = load_model()
    skipped = []
    for key, fam, seq, pos in tqdm(todo, desc="CDS-masked mammal embed"):
        if args.control:
            seq = shuffle_coding_in_place(seq, pos, args.control, key, fam, fam_usage)
        try:
            vec = embed_cds_masked(seq, pos, model, device, args.window)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                vec = embed_cds_masked(seq, pos, model, device, args.window)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped.append(key)
                print(f"  OOM skip (retries next run): {key} len={len(seq)}", flush=True)
                continue
        # Write via a temp file + atomic rename: this run takes days and is meant to be
        # killable at any moment, and a half-written .npy would still satisfy the
        # `.exists()` resume check above — silently poisoning one locus.
        tmp = cache / f"{key}.tmp.npy"  # must end in .npy or np.save appends another
        np.save(tmp, vec.astype(np.float32))
        tmp.replace(cache / f"{key}.npy")
        torch.cuda.empty_cache()
    if skipped:
        print(f"OOM-skipped {len(skipped)} loci (rerun to retry): {skipped[:5]}", flush=True)
    print(
        f"done: {len([r for r in rows if (cache / f'{r[0]}.npy').exists()])}/{len(rows)} cached",
        flush=True,
    )


if __name__ == "__main__":
    main()
