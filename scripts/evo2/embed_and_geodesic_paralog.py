"""Embed and score the Evo2 human-paralog panel."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))

sys.path.insert(0, str(ROOT / "scripts" / "controls"))  # composition-shuffle fns live with the control scripts
import sample_human_genes as ss  # noqa: E402
from evo2_embedding import MODEL_NAME, N_BLOCKS, embed_all_blocks, layer_type, load_model  # noqa: E402
from make_control_sequences import (  # noqa: E402
    build_family_codon_usage,
    codon_shuffle,
    dinuc_shuffle,
    gc_match,
    klet_shuffle,
    missense_subset,
    synonymous_recode,
)

# Composition-preserving shuffles. The first four apply to ANY nucleotide string (no reading
# frame needed). gc_match: mononucleotide; dinuc: di-nucleotide (Eulerian); kmer4/kmer6:
# 4-mer/6-mer-preserving (Euler k-let).
CONTROL_FNS = {
    "gc_match": lambda s, rng: gc_match(s, rng),
    "dinuc_shuffle": lambda s, rng: dinuc_shuffle(s, rng),
    "kmer4_shuffle": lambda s, rng: klet_shuffle(s, 4, rng),
    "kmer6_shuffle": lambda s, rng: klet_shuffle(s, 6, rng),
    # CDS-only (require frame-0 reading frame; see CDS_ONLY_CONTROLS). codon_shuffle reorders the
    # CDS's own codons (preserves codon counts, scrambles protein); it fits the (s, rng) signature.
    "codon_shuffle": lambda s, rng: codon_shuffle(s, rng),
}
# synonymous_recode and its nonsynonymous partner missense_subset need per-family codon usage, so
# they are not plain (s, rng) fns — apply_control handles them specially. All of these only make
# sense on in-frame CDS input (--input cds).
FAMILY_USAGE_CONTROLS = {"synonymous_recode", "missense_subset"}
CDS_ONLY_CONTROLS = {"codon_shuffle"} | FAMILY_USAGE_CONTROLS
CONTROL_CHOICES = list(CONTROL_FNS) + sorted(FAMILY_USAGE_CONTROLS)
from geodesic_utils import (  # noqa: E402
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
)

CACHE_DIR = Path("data/cache/evo2_human_layer_sweep")
STACK_PATH = CACHE_DIR / "layer_stack.npy"
META_PATH = CACHE_DIR / "metadata.csv"
CONFIG_PATH = CACHE_DIR / "config.json"
SEQ_CACHE = Path("data/cache/evo2_human_genomic.json")  # gene -> GRCh38 genomic string
EMBED_DIM = 4096
DEFAULT_MAX_LEN = 100_000  # matches sample_human_genes.EVO2_MAX_LEN (genomic-string fetch clip)
EVO2_WINDOW = 8000  # max bp per Evo2-7B forward on a 23 GB A10G (attention is O(L^2)); longer
                    # spans are tiled into windows of this size and pooled (see embed_all_layers)


# ── gene set + genomic sequences


def load_or_fetch_genomic(genes: list[str], locus_of: dict, max_len: int) -> dict[str, str]:
    """gene -> GRCh38 genomic string over its transcript span (Ensembl /sequence/region, cached)."""
    cache = json.loads(SEQ_CACHE.read_text()) if SEQ_CACHE.exists() else {}
    missing = [g for g in genes if g not in cache]
    if missing:
        print(f"Fetching {len(missing)} genomic transcript-span strings from Ensembl...")
        for i, g in enumerate(tqdm(missing, desc="Ensembl region"), 1):
            cache[g] = ss.evo2_genomic_sequence(locus_of[g], max_len=max_len)
            if i % 25 == 0:
                SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
                SEQ_CACHE.write_text(json.dumps(cache))
        SEQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SEQ_CACHE.write_text(json.dumps(cache))
    return {g: cache[g] for g in genes}


# ── embedding


def embed_all_layers(seq: str, model, device: str, window: int = EVO2_WINDOW) -> np.ndarray:
    """(N_BLOCKS, 4096): second-half-pooled per window, mean-pooled across contiguous windows."""
    L = len(seq)
    if L <= window:
        out = embed_all_blocks(seq, model, device)
    else:
        n = math.ceil(L / window)
        bounds = [(round(i * L / n), round((i + 1) * L / n)) for i in range(n)]
        accum = np.zeros((N_BLOCKS, EMBED_DIM), dtype=np.float64)
        for a, b in bounds:
            accum += embed_all_blocks(seq[a:b], model, device)
            torch.cuda.empty_cache()
        out = (accum / n).astype(np.float32)
    torch.cuda.empty_cache()
    return out


def _seq_hash(s: str) -> str:
    """First 12 hex of sha1(sequence) — a compact per-gene sequence fingerprint."""
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def run_sweep(genes, fams, seqs, device, force, checkpoint_every, window=EVO2_WINDOW,
              cache_dir=CACHE_DIR, control=None) -> np.ndarray:
    stack_path = cache_dir / "layer_stack.npy"
    meta_path = cache_dir / "metadata.csv"
    config_path = cache_dir / "config.json"
    config = {"model": MODEL_NAME, "pool": "second_half", "n_blocks": N_BLOCKS,
              "window": window, "control": control, "genes": genes}
    hashes = {g: _seq_hash(seqs[g]) for g in genes}

    # Reuse cached vectors by gene and verify sequence hashes when available.
    reusable: dict[str, np.ndarray] = {}
    if not force and stack_path.exists() and meta_path.exists() and config_path.exists():
        cached_cfg = json.loads(config_path.read_text())
        # Tolerate caches written before the "control" key existed (a keyless cache was a
        # natural/control=None embedding) so a schema addition never forces a GPU re-embed.
        cached_cfg.setdefault("control", None)
        scalar = ("model", "pool", "n_blocks", "window", "control")
        if all(cached_cfg.get(k) == config[k] for k in scalar):
            old_stack = np.load(stack_path)
            old_meta = pd.read_csv(meta_path)
            old_genes = old_meta["gene"].astype(str).tolist()
            # A checkpoint rewrites layer_stack.npy but NOT metadata.csv/config.json (both are
            # written only on clean completion), so a crash mid-embed leaves a partial stack whose
            # columns no longer align with the stale metadata's gene order. Reuse-by-column-index
            # would then silently misassign vectors, so refuse a stack whose width != metadata rows
            # (only ever true for a partial cache) and fall back to a full re-embed.
            old_hash = (dict(zip(old_genes, old_meta["seq_hash"].astype(str)))
                        if "seq_hash" in old_meta.columns else None)
            want = set(genes) if old_stack.shape[1] == len(old_genes) else set()
            for i, g in enumerate(old_genes):
                if g not in want:
                    continue
                if old_hash is not None and old_hash.get(g) != hashes.get(g):
                    continue  # sequence changed since cache written — must re-embed
                reusable[g] = old_stack[:, i, :]

    misses = [g for g in genes if g not in reusable]
    N = len(genes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stack = np.zeros((N_BLOCKS, N, EMBED_DIM), dtype=np.float32)
    for i, g in enumerate(genes):
        if g in reusable:
            stack[:, i, :] = reusable[g]
    if misses:  # only touch the GPU / load the model when there is something to embed
        print(f"  window={window} bp/forward, control={control}")
        model = load_model()
        idx = {g: i for i, g in enumerate(genes)}
        for c, g in enumerate(tqdm(misses, desc=f"Embedding ({control or 'natural'}, all {N_BLOCKS} blocks)")):
            stack[:, idx[g], :] = embed_all_layers(seqs[g], model, device, window)
            if (c + 1) % checkpoint_every == 0:
                tmp = stack_path.with_suffix(".tmp.npy")
                np.save(tmp, stack)
                tmp.replace(stack_path)
                tqdm.write(f"    [checkpoint] {c + 1}/{len(misses)}")
    print(f"  reused {N - len(misses)} / embedded {len(misses)} / total {N} genes")
    np.save(stack_path, stack)
    pd.DataFrame({"gene": genes, "family": fams,
                  "seq_hash": [hashes[g] for g in genes]}).to_csv(meta_path, index=False)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"  Saved layer stack {stack.shape} -> {stack_path}")
    return stack


# ── run dir at a chosen layer (geodesic + centroid; baseline-ready)


def write_run_dir(stack, genes, fams, fam_order, layer_idx, model_tag="evo2_human", run_tag="",
                  run_dir=None):
    emb = stack[layer_idx]
    families_arr = np.array(fams)
    if run_dir:  # orchestrator-fixed dir; else stamp the date here
        out = Path(run_dir)
    else:
        date = datetime.date.today().isoformat()
        out = Path("results") / f"{date}_evo2-human-panel-blocks{layer_idx}{run_tag}"
    out.mkdir(parents=True, exist_ok=True)

    _, W = find_min_connected_k(emb, k_min=3)
    geo = compute_geodesic(W)
    np.save(out / f"{model_tag}_geodesic.npy", geo)
    pd.DataFrame(geo, index=genes, columns=genes).to_csv(out / f"{model_tag}_geodesic_labeled.csv")

    cen = compute_centroid_geodesic(emb, families_arr, fam_order)
    pd.DataFrame(cen, index=fam_order, columns=fam_order).to_csv(
        out / f"{model_tag}_centroid_distances.csv"
    )
    # Human-panel metadata uses gene symbols as identifiers.
    pd.DataFrame({"gene": genes, "family": fams}).to_csv(out / "metadata.csv", index=False)
    (out / "family_order.txt").write_text("\n".join(fam_order) + "\n")
    print(f"  Wrote run dir {out}  (geodesic {geo.shape}, {len(fam_order)} families)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--families", nargs="+", default=None, help="Restrict to these families (smoke test).")
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="Center-clip genomic span to this many bp.")
    p.add_argument("--window", type=int, default=EVO2_WINDOW, help="Max bp per Evo2 forward; longer spans are tiled.")
    p.add_argument("--from-layer", type=int, default=None, help="Build a run dir at this block from the sweep cache.")
    p.add_argument("--stack", default=None,
                   help="Score an EXTERNAL layer-stack cache dir (dir with layer_stack.npy + "
                        "metadata.csv), skipping embedding. Use with --from-layer + --run-dir. "
                        "For the CDS-masked-transcript condition (embed_cds_masked_transcript.py).")
    p.add_argument("--run-dir", default=None,
                   help="Explicit output dir (pass from the sweep so the date is fixed once by "
                        "the caller; else this script stamps results/<today>_..., which splits a "
                        "sweep across two folders if it runs past midnight UTC).")
    p.add_argument("--control", default=None, choices=CONTROL_CHOICES,
                   help="Composition control: shuffle each sequence before embedding "
                        "(gc_match / dinuc_shuffle / kmer4_shuffle / kmer6_shuffle; plus "
                        "codon_shuffle / synonymous_recode, which require --input cds). "
                        "Caches + run dir are control-tagged.")
    p.add_argument("--input", default="genomic", choices=["genomic", "cds"],
                   help="What Evo2 reads: 'genomic' transcript span (default) "
                        "or 'cds' (diagnostic — same genes, CDS input, to isolate intron-dilution "
                        "from the paralog gene set). CDS read from data/cache/cds_sequences.json.")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def apply_control(seqs: dict[str, str], control: str,
                  fam_of: dict[str, str] | None = None) -> dict[str, str]:
    """Composition-preserving shuffle of each sequence (deterministic per gene)."""
    import random
    from collections import defaultdict
    out = {}
    if control in FAMILY_USAGE_CONTROLS:
        if fam_of is None:
            raise ValueError(f"{control} requires fam_of (gene -> family)")
        by_fam: dict[str, list[str]] = defaultdict(list)
        for g, s in seqs.items():
            by_fam[fam_of[g]].append(s)
        usage = build_family_codon_usage(by_fam)
        for g, s in seqs.items():
            rng = random.Random(f"synonymous_recode:{g}".__hash__() & 0xFFFFFFFF)
            recoded = synonymous_recode(s, usage[fam_of[g]], rng)
            out[g] = recoded if control == "synonymous_recode" else missense_subset(
                s, recoded, random.Random(f"{control}:{g}".__hash__() & 0xFFFFFFFF))
        return out
    for g, s in seqs.items():
        rng = random.Random(f"{control}:{g}".__hash__() & 0xFFFFFFFF)
        out[g] = CONTROL_FNS[control](s, rng)
    return out


def main():
    args = parse_args()

    if args.stack:  # score an external stack (e.g. CDS-masked-transcript); no embedding
        _, _, _, fam_order = ss.load_matched_panel(None, resolve_loci=False)  # canonical family order
        meta = pd.read_csv(Path(args.stack) / "metadata.csv")
        stack = np.load(Path(args.stack) / "layer_stack.npy")
        if args.families:  # restrict the panel; the geodesic graph is global, so a subset
            keep = meta["family"].isin(args.families).to_numpy()  # is a genuinely different run
            if not keep.any():
                sys.exit(f"--families {args.families}: no genes in {args.stack}/metadata.csv")
            meta, stack = meta[keep].reset_index(drop=True), stack[:, keep, :]
        genes, fams = meta["gene"].astype(str).tolist(), meta["family"].tolist()
        fam_order = [f for f in fam_order if f in set(fams)]
        print(f"[stack] {stack.shape} from {args.stack}; {len(genes)} genes, {len(fam_order)} families")
        if args.from_layer is None:
            sys.exit("--stack requires --from-layer")
        write_run_dir(stack, genes, fams, fam_order, args.from_layer, run_dir=args.run_dir)
        print("Done.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Loading matched human Panel-1 (device={device}; control={args.control})")
    # CDS input pulls sequences from cds_sequences.json and never uses the genomic locus, so skip
    # locus resolution for it (avoids ~hundreds of per-gene Ensembl REST calls + gene drops).
    genes, fams, locus_of, fam_order = ss.load_matched_panel(
        args.families, require_both=False, resolve_loci=(args.input != "cds"))
    print(f"  {len(genes)} genes across {len(fam_order)} families: {fam_order}")

    if args.input == "cds":
        cds = json.loads(Path("data/cache/cds_sequences.json").read_text())
        keep = [(g, f) for g, f in zip(genes, fams) if g in cds]
        genes, fams = [g for g, _ in keep], [f for _, f in keep]
        fam_order = [f for f in fam_order if f in set(fams)]
        seqs = {g: cds[g] for g in genes}
        base_cache, base_tag = CACHE_DIR.parent / "evo2_human_layer_sweep_cds", "-cds"
        print(f"[2] CDS input (diagnostic): {len(genes)} genes with cached CDS")
    else:
        print("[2] Genomic transcript-span strings")
        seqs = load_or_fetch_genomic(genes, locus_of, args.max_len)
        base_cache, base_tag = CACHE_DIR, ""
    lens = sorted(len(seqs[g]) for g in genes)
    print(f"  input lengths: median={lens[len(lens)//2]}  max={lens[-1]} bp")

    cache_dir, run_tag = base_cache, base_tag
    if args.control:
        if args.control in CDS_ONLY_CONTROLS and args.input != "cds":
            sys.exit(f"--control {args.control} requires --input cds (it needs an in-frame reading frame)")
        print(f"[2b] Applying composition control: {args.control}")
        seqs = apply_control(seqs, args.control, fam_of=dict(zip(genes, fams)))
        cache_dir = base_cache.parent / f"{base_cache.name}_{args.control}"
        run_tag = f"{base_tag}-{args.control}"

    print(f"[3] Dense layer sweep ({N_BLOCKS} blocks, second-half pooled, window={args.window} bp)")
    stack = run_sweep(genes, fams, seqs, device, args.force_reembed, args.checkpoint_every,
                      args.window, cache_dir=cache_dir, control=args.control)

    if args.from_layer is not None:
        print(f"[4] Run dir @ blocks.{args.from_layer} ({layer_type(args.from_layer)})")
        write_run_dir(stack, genes, fams, fam_order, args.from_layer, run_tag=run_tag,
                      run_dir=args.run_dir)
    print("Done.")


if __name__ == "__main__":
    main()
