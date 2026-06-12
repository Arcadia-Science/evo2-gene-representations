"""Download GTDB bacterial metadata, stratified-sample 500 species by phylum, write manifest CSV."""

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

METADATA_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz"
TREE_URL = "https://data.gtdb.ecogenomic.org/releases/latest/bac120.tree"

TARGET_N = 500
RANDOM_STATE = 42

DATA_DIR = Path("data/species")
MANIFEST_PATH = DATA_DIR / "gtdb_500_manifest.csv"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists, skipping: {dest}")
        return
    print(f"  Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest}")


def load_metadata(gz_path: Path) -> pd.DataFrame:
    print("Loading metadata TSV...")
    # GTDB metadata files sometimes have a leading comment line starting with '#'.
    # pandas read_csv with comment='#' handles that gracefully.
    df = pd.read_csv(
        gz_path,
        sep="\t",
        comment="#",
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def parse_taxonomy(taxonomy_str: str) -> tuple[str, str]:
    """Return (phylum, genus) as plain strings with rank prefix stripped."""
    parts = taxonomy_str.split(";")
    phylum = next((p[3:] for p in parts if p.startswith("p__")), "")
    genus = next((p[3:] for p in parts if p.startswith("g__")), "")
    return phylum, genus


def extract_species_name(taxonomy_str: str) -> str:
    """Return species name as plain string with 's__' prefix stripped."""
    parts = taxonomy_str.split(";")
    species = next((p[3:] for p in parts if p.startswith("s__")), "")
    return species


def strip_prefix(gtdb_accession: str) -> str:
    """Strip GB_/RS_ prefix from GTDB accession to recover plain NCBI accession."""
    return re.sub(r"^(GB_|RS_)", "", gtdb_accession)


def stratified_sample(df: pd.DataFrame, target: int, random_state: int) -> pd.DataFrame:
    """Sample exactly target rows stratified by gtdb_phylum."""
    phylum_counts = df["gtdb_phylum"].value_counts()
    total = len(df)

    # Compute per-phylum quota (rounded, minimum 1)
    quotas: dict[str, int] = {}
    for phylum, count in phylum_counts.items():
        n = max(1, round(target * count / total))
        quotas[phylum] = min(n, count)  # cannot sample more than available

    current_total = sum(quotas.values())
    delta = target - current_total

    if delta != 0:
        # Sort phyla by count descending (largest first) for tie-breaking
        sorted_phyla = phylum_counts.index.tolist()  # already sorted by value_counts
        # Adjust from the largest phyla
        i = 0
        while delta != 0 and i < len(sorted_phyla):
            phylum = sorted_phyla[i % len(sorted_phyla)]
            available = phylum_counts[phylum]
            if delta > 0:
                if quotas[phylum] < available:
                    quotas[phylum] += 1
                    delta -= 1
            else:
                if quotas[phylum] > 1:
                    quotas[phylum] -= 1
                    delta += 1
            i += 1

    # Sample per phylum
    sampled_frames = []
    for phylum, n in quotas.items():
        group = df[df["gtdb_phylum"] == phylum]
        n_actual = min(n, len(group))
        sampled_frames.append(group.sample(n=n_actual, random_state=random_state))

    result = pd.concat(sampled_frames, ignore_index=True)

    # Trim or top-up if rounding still leaves us off by tiny margin
    if len(result) > target:
        result = result.iloc[:target]
    elif len(result) < target:
        # Fill from unsampled rows, largest phyla first
        sampled_ids = set(result.index)
        remainder = df[~df.index.isin(sampled_ids)].sort_values(
            "gtdb_phylum",
            key=lambda col: col.map(phylum_counts),
            ascending=False,
        )
        extra = remainder.iloc[: target - len(result)]
        result = pd.concat([result, extra], ignore_index=True)

    return result.reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download files ---
    gz_path = DATA_DIR / "bac120_metadata.tsv.gz"
    tree_path = DATA_DIR / "bac120.tree"
    download_file(METADATA_URL, gz_path)
    download_file(TREE_URL, tree_path)

    # --- Load and parse metadata ---
    df = load_metadata(gz_path)

    required_cols = {"accession", "gtdb_taxonomy", "genome_size", "gtdb_representative"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in metadata: {missing}", file=sys.stderr)
        sys.exit(1)

    # Keep only GTDB species representatives. The bac120.tree (our phylogenetic
    # ground truth) contains exactly the representative genomes, so sampling
    # non-representatives yields species with no position in the tree.
    before = len(df)
    df = df[df["gtdb_representative"] == "t"].copy()
    print(f"  Kept {len(df):,} species representatives (of {before:,} total genomes)")

    # Parse taxonomy
    taxonomy_parsed = df["gtdb_taxonomy"].apply(parse_taxonomy)
    df["gtdb_phylum"] = [t[0] for t in taxonomy_parsed]
    df["gtdb_genus"] = [t[1] for t in taxonomy_parsed]
    df["species_name"] = df["gtdb_taxonomy"].apply(extract_species_name)

    # Strip accession prefix
    df["gtdb_accession"] = df["accession"]
    df["ncbi_accession"] = df["accession"].apply(strip_prefix)

    # Drop rows with empty phylum (unassigned)
    before = len(df)
    df = df[df["gtdb_phylum"] != ""].copy()
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after:,} rows with empty/unassigned phylum")

    # Sort for reproducibility
    df = df.sort_values("gtdb_accession").reset_index(drop=True)

    print(f"  {len(df):,} rows with assigned phylum across {df['gtdb_phylum'].nunique()} phyla")

    # --- Stratified sample ---
    print(f"Stratified sampling {TARGET_N} species by phylum...")
    sampled = stratified_sample(df, TARGET_N, RANDOM_STATE)

    assert len(sampled) == TARGET_N, f"Expected {TARGET_N} rows, got {len(sampled)}"

    # --- Write manifest ---
    manifest_cols = [
        "gtdb_accession",
        "ncbi_accession",
        "species_name",
        "gtdb_phylum",
        "gtdb_genus",
        "genome_size",
    ]
    manifest = sampled[manifest_cols].copy()
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote manifest: {MANIFEST_PATH} ({len(manifest)} rows)")

    # --- Print phylum distribution ---
    print("\nPhylum distribution in sample:")
    dist = manifest["gtdb_phylum"].value_counts()
    for phylum, count in dist.items():
        print(f"  {phylum:<50s} {count:>4d}")
    print(f"\nTotal: {len(manifest)}")


if __name__ == "__main__":
    main()
