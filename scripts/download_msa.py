"""
Download and prepare MSA alignment data needed for GPN-Star VEP.

Three dtype options:

  synthetic  (fast, no download)
    Builds a minimal fake zarr with random MSA sequences for whichever
    chromosomes/positions appear in songlab/clinvar_vs_benign.  Scores are
    meaningless, but it exercises the full VEP pipeline end-to-end.

  vertebrate  (slow, ~42 GB download + extraction)
    Downloads the actual 100-way vertebrate alignment from
    songlab/multiz100way-pigz and extracts it with pigz.
    Required for scientifically valid VEP scores with the vertebrate model.

  mammalian  (slow, ~121 GB download + extraction)
    Downloads the mammalian 447-way alignment from songlab/hg38_cactus447way
    (three shards: 36.tar.gz, 204.tar.gz, 243.tar.gz) and extracts them into
    a single zarr store.  Required for the GPN-Star mammalian (m447) model.

Usage:
    uv run python scripts/download_msa.py synthetic
    uv run python scripts/download_msa.py vertebrate
    uv run python scripts/download_msa.py mammalian
    uv run python scripts/download_msa.py mammalian --out data/multiz447way.zarr
"""

import argparse
from pathlib import Path

import numpy as np


# ── Synthetic zarr ────────────────────────────────────────────────────────────


def build_synthetic_zarr(out_path: Path, n_species: int = 100, context: int = 512):
    """Create a minimal zarr with random MSA data for clinvar_vs_benign positions.

    np.frombuffer vocab-lookup pattern is adapted from make_msa_chrom in analysis/gpn-star/wga_processing/workflow/rules/msa.smk.
    from https://github.com/songlab-cal/gpn, Gonzalo Benegas et al., MIT License.
    """
    import zarr
    from datasets import load_dataset

    print("Loading songlab/clinvar_vs_benign to determine required positions...")
    ds = load_dataset("songlab/clinvar_vs_benign", split="test") # features: ['chrom', 'pos', 'ref', 'alt', 'label', 'id', 'review_status', 'consequence']
    df = ds.to_pandas()[["chrom", "pos"]] 
    print(f"  {len(df)} variants across {df.chrom.nunique()} chromosomes")

    print(f"\nBuilding synthetic zarr at {out_path}  (n_species={n_species})...")
    out_path.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_path), mode="w")

    rng = np.random.default_rng(42)
    vocab = np.frombuffer(b"ACGT", dtype="S1") # lookup array [b'A' b'C' b'G' b'T']

    for chrom, grp in df.groupby("chrom"):
        min_pos = int(grp.pos.min()) - 1  # convert to 0-based
        max_pos = int(grp.pos.max())  # exclusive end

        # Cover the full range of positions + context on both sides; GenomeMSA extracts needed sequences during inference
        start = max(0, min_pos - context) 
        end = max_pos + context
        length = end - start

        # Random DNA for all species
        data = rng.integers(0, 4, size=(length, n_species), dtype=np.uint8)
        seq = vocab[data]  # (length, n_species) of S1 bytes

        # Pad front with gap characters ('-') to align data to chrom positions
        total_length = end  # minimal length to cover all positions 
        full = np.full((total_length, n_species), b"-", dtype="S1")
        full[start:end] = seq

        store[chrom] = full
        print(f"  {chrom}: {total_length:,} bp × {n_species} species")

    print(f"\nSynthetic zarr written to {out_path}")
    print("NOTE: This MSA is random — Only useful for testing.")


# ── Real download ─────────────────────────────────────────────────────────────


def download_real(out_path: Path):
    """Download and extract the full multiz100way pigz archive (~42 GB)."""
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download

    # Check pigz is available
    if subprocess.run(["which", "pigz"], capture_output=True).returncode != 0:
        print("ERROR: pigz is not installed. Install it with:")
        print("  sudo apt install pigz")
        sys.exit(1)

    print("Downloading songlab/multiz100way-pigz (99.zarr.tar.gz, ~42 GB)...")
    print("WARNING: this file is ~42 GB. Ensure you have sufficient disk space.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = Path(hf_hub_download(
        repo_id="songlab/multiz100way-pigz",
        filename="99.zarr.tar.gz",
        local_dir=str(out_path.parent),
        repo_type="dataset",
    ))

    print(f"\nExtracting to {out_path.parent} ...")
    subprocess.run(
        f"unpigz < {tmp_archive} | tar -x -C {out_path.parent}",
        shell=True,
        check=True,
    )

    extracted = out_path.parent / "99.zarr"
    if extracted.exists() and extracted != out_path:
        extracted.rename(out_path)
        print(f"Renamed {extracted} → {out_path}")

    print(f"\nDone. MSA zarr at {out_path}")


def download_m447(out_path: Path):
    """Download and extract the mammalian 447-way alignment (~121 GB total).

    The dataset songlab/hg38_cactus447way is split across three shards:
      36.tar.gz  (~15 GB), 204.tar.gz (~66 GB), 243.tar.gz (~40 GB).
    Each shard is a standard gzip-compressed tar archive containing zarr
    chunk files for a subset of chromosomes.  All three are extracted into
    the same output directory so that their contents merge into one zarr store.
    """
    import subprocess

    from huggingface_hub import hf_hub_download

    SHARDS = ["36.tar.gz", "204.tar.gz", "243.tar.gz"]
    TOTAL_GB = 121

    print(f"Downloading songlab/hg38_cactus447way ({len(SHARDS)} shards, ~{TOTAL_GB} GB total)...")
    print(f"WARNING: ~{TOTAL_GB} GB download + extraction. Ensure you have sufficient disk space.")
    out_path.mkdir(parents=True, exist_ok=True)

    for shard in SHARDS:
        print(f"\n  Downloading shard {shard}...")
        archive = Path(hf_hub_download(
            repo_id="songlab/hg38_cactus447way",
            filename=shard,
            local_dir=str(out_path.parent),
            repo_type="dataset",
        ))
        print(f"  Extracting {archive} into {out_path} ...")
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(out_path)],
            check=True,
        )
        print(f"  Shard {shard} done.")

    print(f"\nDone. MSA zarr at {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "dtype",
        choices=["synthetic", "vertebrate", "mammalian"],
        help=(
            "Which alignment to download: "
            "'synthetic' builds a fake zarr for smoke-testing (no download); "
            "'vertebrate' downloads the 100-way vertebrate alignment (~42 GB); "
            "'mammalian' downloads the mammalian 447-way alignment (~121 GB)."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output zarr path. "
            "Defaults: synthetic/vertebrate → data/multiz100way.zarr, "
            "mammalian → data/multiz447way.zarr"
        ),
    )
    p.add_argument(
        "--n-species",
        type=int,
        default=100,
        help="Number of species columns in synthetic zarr (default: 100)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve default output path based on dtype
    if args.out is not None:
        out_path = Path(args.out)
    elif args.dtype == "mammalian":
        out_path = Path("data/multiz447way.zarr")
    else:
        out_path = Path("data/multiz100way.zarr")

    if args.dtype == "synthetic":
        build_synthetic_zarr(out_path, n_species=args.n_species)
    elif args.dtype == "vertebrate":
        download_real(out_path)
    else:  # mammalian
        download_m447(out_path)


if __name__ == "__main__":
    main()
