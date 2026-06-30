"""GPN-Star human gene-family embedder + geodesic — the GPN side of the apples-to-apples
Evo2-vs-GPN-Star comparison (§1 shared locus).

GPN-Star tiles multiz windows over each gene's transcript span [tx_start, tx_end] — the SAME
locus the Evo2 human embedder reads as a genomic string — for the 580 genes embeddable by both
models (test_sample_human_genes.load_matched_panel).

This module is BOTH:
  * the embedding-helper library — model loading, transcript-span coords, MSA-window tiling, and
    per-gene multi-window embedding — reused by scripts/gpnstar/layer_sweep.py and
    analyses/embed_and_score_msa_controls.py; and
  * the human embed + geodesic pipeline (run as a script):
      - default        : dense layer sweep — embed every hidden state, full-transcript multi-window,
                         → data/cache/gpnstar_human_layer_sweep/{layer_stack_<model>.npy,metadata.csv}
                         (consumed by scripts/layer_selection/layer_selection.py --model gpnstar --panel human).
      - --from-layer L : pull hidden_state L from the cache, build the angular k-NN geodesic +
                         family-centroid geodesic, and write a run dir (metadata.csv gene/family,
                         *_geodesic.npy, *_centroid_distances.csv, family_order.txt) for the shared
                         human baselines (--seq-source gpn). The geodesic math itself lives in the
                         shared scripts/geodesic_utils.py; this script only orchestrates it.

Usage:
    uv run python scripts/gpnstar/embed_and_geodesic.py                     # full 580-gene sweep
    uv run python scripts/gpnstar/embed_and_geodesic.py --families globins  # smoke subset
    uv run python scripts/gpnstar/embed_and_geodesic.py --from-layer 5      # run dir @ hidden_state.5
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import gpn.star.model  # noqa: F401 — registers GPNStar with AutoModel/AutoConfig
import numpy as np
import pandas as pd
import torch
from gpn.star.data import GenomeMSA
from gpn.star.model import GPNStarModel
from tqdm import tqdm
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "gpnstar"))  # gpnstar siblings only → no name clash

import test_sample_human_genes as ss  # noqa: E402
from geodesic_utils import (  # noqa: E402
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
)

ENSEMBL_BASE = "https://rest.ensembl.org"

MODELS = {
    "vertebrate": {
        "hf_id": "songlab/gpn-star-hg38-v100-200m",
        "n_species": 100,
        "msa_path": "data/multiz100way.zarr",
    },
}

# Multi-window coverage of the FULL genomic transcript span (§1 shared locus): up to this
# many windows of `max_position_embeddings` (=1536) bp tile [tx_start, tx_end] (5'UTR + exons +
# introns + 3'UTR), mean-pooled. The same scheme is used by both the production embedder and
# the layer sweep (scripts/gpnstar/layer_sweep.py), so the two are byte-for-byte aligned.
# 512 windows fully tile up to ~786 kb; the largest gene in the panel (CA10, 528 kb) needs 345,
# so this caps nothing in the current dataset. Total ≈ 4.6k forward passes over the 583-gene set
# (vs 2.9k at a 16-cap) — trivial for the 200M GPN-Star model, so we cover every gene in full.
MAX_WINDOWS = 512

CACHE_DIR = Path("data/cache/gpnstar_human_layer_sweep")
CONFIG_PATH = CACHE_DIR / "config.json"


# ════════════════════════════════════════════════════════════════════════════════
# Embedding-helper library (reused by layer_sweep.py and the MSA-controls analysis)
# ════════════════════════════════════════════════════════════════════════════════

# ── hg38 transcript-span coordinates ──────────────────────────────────────────


def fetch_cds_coords(gene_symbol: str) -> dict:
    url = (
        f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}"
        f"?content-type=application/json&expand=1"
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            wait = 10 * 2**attempt
            print(f"  Ensembl request failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(f"Ensembl lookup failed for {gene_symbol} after 5 attempts")

    chrom = data["seq_region_name"]
    strand = "+" if data["strand"] == 1 else "-"
    canonical_id = data["canonical_transcript"].split(".")[0]

    for tx in data.get("Transcript", []):
        if tx["id"] == canonical_id and "Translation" in tx:
            t = tx["Translation"]
            return {
                "gene": gene_symbol,
                "chrom": chrom,
                # Full genomic transcript span (1-based inclusive): 5'UTR + exons + introns + 3'UTR.
                "tx_start": tx["start"],
                "tx_end": tx["end"],
                "tx_len": tx["end"] - tx["start"],
                # CDS (coding) span, kept for metadata / backward compatibility.
                "cds_start": t["start"],
                "cds_end": t["end"],
                "cds_len": t["end"] - t["start"],
                "strand": strand,
            }
    raise ValueError(f"No canonical CDS found for {gene_symbol}")


def load_or_fetch_coords(all_genes: list[str], cache_path: Path) -> dict[str, dict]:
    """Return {gene: coord} for genes with a canonical CDS in Ensembl.

    Genes without one (no canonical translation, not found) are cached as a
    ``None`` sentinel and dropped from the result, so a single unmappable HGNC
    member can't abort the run. Transient network errors still raise.
    """
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    # Refetch any gene not cached, or cached under the old schema (no transcript span).
    missing = [g for g in all_genes
               if g not in cache or (cache[g] is not None and "tx_start" not in cache[g])]
    if missing:
        print(f"Fetching transcript-span coordinates for {len(missing)} genes from Ensembl...")
        for gene in tqdm(missing, desc="Ensembl lookup"):
            try:
                cache[gene] = fetch_cds_coords(gene)
            except ValueError as e:  # definitive: no canonical CDS / not found
                print(f"  Skipping {gene}: {e}")
                cache[gene] = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    dropped = [g for g in all_genes if cache.get(g) is None]
    if dropped:
        print(f"  {len(dropped)} gene(s) without a canonical CDS were dropped: {dropped}")
    return {g: cache[g] for g in all_genes if cache.get(g) is not None}


# ── Model loading ──────────────────────────────────────────────────────────────


def load_base_model(model_path: Path, device: str) -> tuple[GPNStarModel, int]:
    from safetensors.torch import load_file

    config = AutoConfig.from_pretrained(str(model_path))
    if not os.path.exists(config.phylo_dist_path):
        fallback = os.path.join(str(model_path), "phylo_dist")
        if os.path.exists(fallback):
            config.phylo_dist_path = fallback
        else:
            raise FileNotFoundError(
                f"phylo_dist not found at '{config.phylo_dist_path}' or '{fallback}'"
            )

    model = GPNStarModel(config)
    state_dict = load_file(os.path.join(str(model_path), "model.safetensors"))
    base_state = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(base_state, strict=False)
    model.eval()
    model.to(device)
    return model, config.max_position_embeddings


def resolve_model_path(model_name: str, models_dir: Path) -> Path:
    hf_id = MODELS[model_name]["hf_id"]
    # Check if already downloaded into models_dir as huggingface cache layout
    owner, repo = hf_id.split("/")
    cache_name = f"models--{owner}--{repo}"
    cache_dir = models_dir / cache_name / "snapshots"
    if cache_dir.exists():
        snapshots = sorted(cache_dir.iterdir())
        if snapshots:
            return snapshots[-1]

    # Fall back to snapshot_download
    from huggingface_hub import snapshot_download

    print(f"Downloading {hf_id}...")
    local = snapshot_download(
        repo_id=hf_id,
        cache_dir=str(models_dir),
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    return Path(local)


# ── MSA window tiling + per-gene embedding ─────────────────────────────────────


def sample_span_windows(
    start0: int, end0: int, window: int, max_windows: int = MAX_WINDOWS
) -> list[tuple[int, int]]:
    """Tile <= max_windows windows of `window` bp over the genomic interval [start0, end0).

    A span shorter than one window gets a single centred window. Longer spans are tiled with
    contiguous windows; when more than max_windows would be needed, the window centres are
    sampled evenly across the span (so a long, mostly-intron gene is covered sparsely but
    uniformly). Coordinates are 0-based half-open. The whole transcript span — 5'UTR + exons +
    introns + 3'UTR (§1's shared locus) — is the interval passed in.
    """
    span = end0 - start0
    if span <= window:
        center = (start0 + end0) // 2
        s = max(0, center - window // 2)
        return [(s, s + window)]
    n = math.ceil(span / window)
    centers = [start0 + window // 2 + i * window for i in range(n)]
    centers[-1] = min(centers[-1], end0 - window // 2)
    if len(centers) > max_windows:
        pick = np.unique(np.linspace(0, len(centers) - 1, max_windows).round().astype(int))
        centers = [centers[i] for i in pick]
    return [(max(0, c - window // 2), max(0, c - window // 2) + window) for c in centers]


def embed_gene_multiwindow(genome_msa, model, coord, windows, device, n_states) -> np.ndarray:
    """(n_states, hidden_size): per-layer embedding, mean-pooled over positions then windows."""
    chrom, strand = coord["chrom"], coord["strand"]
    accum = np.zeros((n_states, model.config.hidden_size), dtype=np.float64)
    for win_start, win_end in windows:
        msa = genome_msa.get_msa(chrom, win_start, win_end, strand=strand, tokenize=True)
        msa_t = torch.from_numpy(msa.astype(np.int64)).to(device)
        input_ids = msa_t[:, :1].unsqueeze(0)
        source_ids = msa_t.unsqueeze(0)
        target_species = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                source_ids=source_ids,
                target_species=target_species,
                output_hidden_states=True,
            )
        vecs = np.stack(
            [
                hs[:, :, 0, :].mean(dim=1).squeeze(0).cpu().float().numpy()
                for hs in out.hidden_states
            ]
        )
        accum += vecs
    return (accum / len(windows)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
# Pipeline — dense layer sweep + run-dir build at a chosen layer
# ════════════════════════════════════════════════════════════════════════════════


def stack_path(model: str) -> Path:
    return CACHE_DIR / f"layer_stack_{model}.npy"


def run_sweep(genes, fams, locus_of, model_name, max_windows, device, force, checkpoint_every):
    config = {"model": model_name, "scheme": "transcript-multiwindow",
              "max_windows": max_windows, "genes": genes}
    sp = stack_path(model_name)
    if (not force and sp.exists() and CONFIG_PATH.exists()
            and json.loads(CONFIG_PATH.read_text()) == config):
        print("  Reusing cached human layer stack (config matches).")
        return np.load(sp)

    cfg = MODELS[model_name]
    print(f"  Loading MSA + model ({model_name})")
    genome_msa = GenomeMSA(str(cfg["msa_path"]), n_species=cfg["n_species"])
    model, max_window = load_base_model(resolve_model_path(model_name, Path("models")), device)
    n_states = model.config.num_hidden_layers + 1
    H = model.config.hidden_size

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    N = len(genes)
    stack = np.zeros((n_states, N, H), dtype=np.float32)
    n_win = []
    for i, g in enumerate(tqdm(genes, desc=f"Embedding (≤{max_windows} windows × {n_states} layers)")):
        c = locus_of[g]
        coord = {"chrom": c["chrom"], "strand": c["strand"]}
        windows = sample_span_windows(c["tx_start"] - 1, c["tx_end"], max_window, max_windows)
        n_win.append(len(windows))
        stack[:, i, :] = embed_gene_multiwindow(genome_msa, model, coord, windows, device, n_states)
        if (i + 1) % checkpoint_every == 0:
            tmp = sp.with_suffix(".tmp.npy")
            np.save(tmp, stack)
            tmp.replace(sp)
            tqdm.write(f"    [checkpoint] {i + 1}/{N}")
    np.save(sp, stack)
    pd.DataFrame({"gene": genes, "family": fams}).to_csv(CACHE_DIR / "metadata.csv", index=False)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"  Windows/gene: min={min(n_win)} median={int(np.median(n_win))} max={max(n_win)}")
    print(f"  Saved human layer stack {stack.shape} -> {sp}")
    return stack


def write_run_dir(stack, genes, fams, fam_order, layer_idx, model_name):
    emb = stack[layer_idx]
    families_arr = np.array(fams)
    date = datetime.date.today().isoformat()
    out = Path("results") / f"{date}_gpnstar-human-panel-L{layer_idx}"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"gpnstar_{model_name}"

    _, W = find_min_connected_k(emb, k_min=3)
    geo = compute_geodesic(W)
    np.save(out / f"{tag}_geodesic.npy", geo)
    pd.DataFrame(geo, index=genes, columns=genes).to_csv(out / f"{tag}_geodesic_labeled.csv")

    cen = compute_centroid_geodesic(emb, families_arr, fam_order)
    pd.DataFrame(cen, index=fam_order, columns=fam_order).to_csv(out / f"{tag}_centroid_distances.csv")
    pd.DataFrame({"gene": genes, "family": fams}).to_csv(out / "metadata.csv", index=False)
    (out / "family_order.txt").write_text("\n".join(fam_order) + "\n")
    print(f"  Wrote run dir {out}  (geodesic {geo.shape}, {len(fam_order)} families)")
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vertebrate", choices=list(MODELS))
    p.add_argument("--families", nargs="+", default=None, help="Restrict to these families (smoke test).")
    p.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    p.add_argument("--from-layer", type=int, default=None, help="Build a run dir at this hidden state.")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Loading matched human Panel-1 (device={device})")
    genes, fams, locus_of, fam_order = ss.load_matched_panel(args.families)
    print(f"  {len(genes)} genes across {len(fam_order)} families: {fam_order}")

    print("[2] Dense layer sweep (all hidden states, full-transcript multi-window)")
    stack = run_sweep(genes, fams, locus_of, args.model, args.max_windows,
                      device, args.force_reembed, args.checkpoint_every)

    if args.from_layer is not None:
        print(f"[3] Run dir @ hidden_state.{args.from_layer}")
        write_run_dir(stack, genes, fams, fam_order, args.from_layer, args.model)
    print("Done.")


if __name__ == "__main__":
    main()
