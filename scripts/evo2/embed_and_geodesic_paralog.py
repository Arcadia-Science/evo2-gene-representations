"""Evo2 human Panel-1 (matched-manifest) embedder — the human side of the apples-to-apples
Evo2-vs-GPN-Star comparison (§1 shared locus).

Evo2 reads the **GRCh38 genomic nucleotide string** over each gene's transcript span
[tx_start, tx_end] — the SAME locus GPN-Star tiles with multiz windows — for the 580 genes
embeddable by both models (test_sample_human_genes.matched_panel). This is distinct from the
cross-kingdom pipeline (scripts/evo2/embed_and_geodesic_ortholog.py), which reads KEGG CDS.

Two stages (like the cross-kingdom sweep):
  * default            : dense layer sweep — embed every block, second-half pooled, →
                         data/cache/evo2_human_layer_sweep/{layer_stack.npy,metadata.csv}
                         (consumed by scripts/layer_selection/layer_selection.py --model evo2 --panel human).
  * --from-layer L     : pull block L from the sweep cache, build the angular k-NN geodesic +
                         family-centroid geodesic, and write a run dir whose artifacts match the
                         GPN run-dir contract (metadata.csv with gene/family, *_geodesic.npy,
                         *_centroid_distances.csv, family_order.txt) so the shared human baselines
                         (protein_alignment_patristic_seqid / between_family_baselines, --seq-source gpn)
                         score it exactly as they score GPN-human.

The whole genomic span is fed to Evo2 (1M-context evo2_7b); only the second-half token positions
are pooled (evo2_embedding.pool_second_half). Long spans risk GPU OOM, so the
forward pass retries with a progressively tighter center-clip before giving up on a gene.

Usage:
    uv run python scripts/evo2/embed_and_geodesic_paralog.py                       # full 580-gene sweep
    uv run python scripts/evo2/embed_and_geodesic_paralog.py --families globins    # smoke subset
    uv run python scripts/evo2/embed_and_geodesic_paralog.py --from-layer 15       # run dir @ blocks.15
"""

from __future__ import annotations

import argparse
import datetime
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
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))  # evo2/ siblings (NOT gpnstar) → no name clash

sys.path.insert(0, str(ROOT / "analyses"))  # composition-shuffle fns live with the control scripts
import test_sample_human_genes as ss  # noqa: E402
from evo2_embedding import MODEL_NAME, N_BLOCKS, embed_all_blocks, layer_type, load_model  # noqa: E402
from make_control_sequences import dinuc_shuffle, gc_match, klet_shuffle  # noqa: E402

# Composition-preserving shuffles applicable to a GENOMIC string (no reading frame, so the
# CDS-only codon_shuffle/synonymous_recode are excluded — the protein-level question is answered
# by the cross-kingdom CDS controls). gc_match: mononucleotide; dinuc: di-nucleotide (Eulerian);
# kmer6: 6-mer-preserving (Euler k-let).
CONTROL_FNS = {
    "gc_match": lambda s, rng: gc_match(s, rng),
    "dinuc_shuffle": lambda s, rng: dinuc_shuffle(s, rng),
    "kmer6_shuffle": lambda s, rng: klet_shuffle(s, 6, rng),
}
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
DEFAULT_MAX_LEN = 100_000  # matches test_sample_human_genes.EVO2_MAX_LEN (genomic-string fetch clip)
EVO2_WINDOW = 8000  # max bp per Evo2-7B forward on a 23 GB A10G (attention is O(L^2)); longer
                    # spans are tiled into windows of this size and pooled (see embed_all_layers)


# ── gene set + genomic sequences ──────────────────────────────────────────────


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


# ── embedding ─────────────────────────────────────────────────────────────────


def embed_all_layers(seq: str, model, device: str, window: int = EVO2_WINDOW) -> np.ndarray:
    """(N_BLOCKS, 4096): second-half-pooled per window, mean-pooled across contiguous windows.

    Evo2-7B attention is O(L^2) in memory, so a single forward over a long genomic span OOMs a
    23 GB GPU (and a failed giant forward fragments the allocator, so in-process retry can't
    recover). Instead we cap the forward at `window` bp: a span <= window is one forward; a longer
    span is tiled into ceil(L/window) contiguous windows, each embedded independently and the
    per-block vectors averaged across windows — full-span coverage (like GPN's multi-window
    scheme), no oversized forward. empty_cache between windows/genes keeps the footprint flat.
    """
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


def run_sweep(genes, fams, seqs, device, force, checkpoint_every, window=EVO2_WINDOW,
              cache_dir=CACHE_DIR, control=None) -> np.ndarray:
    stack_path = cache_dir / "layer_stack.npy"
    meta_path = cache_dir / "metadata.csv"
    config_path = cache_dir / "config.json"
    config = {"model": MODEL_NAME, "pool": "second_half", "n_blocks": N_BLOCKS,
              "window": window, "control": control, "genes": genes}
    if not force and stack_path.exists() and config_path.exists():
        cached_cfg = json.loads(config_path.read_text())
        # Tolerate caches written before the "control" key existed (a keyless cache was a
        # natural/control=None embedding) so a schema addition never forces a GPU re-embed.
        cached_cfg.setdefault("control", None)
        if cached_cfg == config:
            print(f"  Reusing cached layer stack ({cache_dir.name}, config matches).")
            return np.load(stack_path)

    print(f"  window={window} bp/forward, control={control}")
    model = load_model()
    cache_dir.mkdir(parents=True, exist_ok=True)
    N = len(genes)
    stack = np.zeros((N_BLOCKS, N, EMBED_DIM), dtype=np.float32)
    for i, g in enumerate(tqdm(genes, desc=f"Embedding ({control or 'natural'}, all {N_BLOCKS} blocks)")):
        stack[:, i, :] = embed_all_layers(seqs[g], model, device, window)
        if (i + 1) % checkpoint_every == 0:
            tmp = stack_path.with_suffix(".tmp.npy")
            np.save(tmp, stack)
            tmp.replace(stack_path)
            tqdm.write(f"    [checkpoint] {i + 1}/{N}")
    np.save(stack_path, stack)
    pd.DataFrame({"gene": genes, "family": fams}).to_csv(meta_path, index=False)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"  Saved layer stack {stack.shape} -> {stack_path}")
    return stack


# ── run dir at a chosen layer (geodesic + centroid; baseline-ready) ─────────────


def write_run_dir(stack, genes, fams, fam_order, layer_idx, model_tag="evo2_human", run_tag=""):
    emb = stack[layer_idx]
    families_arr = np.array(fams)
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
    # metadata.csv with a "gene" column → scored with --seq-source gpn (human ground truth),
    # exactly like GPN-human, so the two models are compared on the same baselines.
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
    p.add_argument("--control", default=None, choices=list(CONTROL_FNS),
                   help="Composition control: shuffle each genomic string before embedding "
                        "(gc_match / dinuc_shuffle / kmer6_shuffle). Caches + run dir are control-tagged.")
    p.add_argument("--input", default="genomic", choices=["genomic", "cds"],
                   help="What Evo2 reads: 'genomic' transcript span (default, matches the GPN locus) "
                        "or 'cds' (diagnostic — same genes, CDS input, to isolate intron-dilution "
                        "from the paralog gene set). CDS read from data/cache/cds_sequences.json.")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def apply_control(seqs: dict[str, str], control: str) -> dict[str, str]:
    """Composition-preserving shuffle of each genomic string (deterministic per gene)."""
    import random
    out = {}
    for g, s in seqs.items():
        rng = random.Random(f"{control}:{g}".__hash__() & 0xFFFFFFFF)
        out[g] = CONTROL_FNS[control](s, rng)
    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Loading matched human Panel-1 (device={device}; control={args.control})")
    genes, fams, locus_of, fam_order = ss.load_matched_panel(args.families)
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
        print(f"[2b] Applying composition control: {args.control}")
        seqs = apply_control(seqs, args.control)
        cache_dir = base_cache.parent / f"{base_cache.name}_{args.control}"
        run_tag = f"{base_tag}-{args.control}"

    print(f"[3] Dense layer sweep ({N_BLOCKS} blocks, second-half pooled, window={args.window} bp)")
    stack = run_sweep(genes, fams, seqs, device, args.force_reembed, args.checkpoint_every,
                      args.window, cache_dir=cache_dir, control=args.control)

    if args.from_layer is not None:
        print(f"[4] Run dir @ blocks.{args.from_layer} ({layer_type(args.from_layer)})")
        write_run_dir(stack, genes, fams, fam_order, args.from_layer, run_tag=run_tag)
    print("Done.")


if __name__ == "__main__":
    main()
