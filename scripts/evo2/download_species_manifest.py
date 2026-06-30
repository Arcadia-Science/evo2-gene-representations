"""Download GTDB bacterial metadata, sample species, write manifest CSV.

Two sampling strategies (--sampling):
  wide  phylum-proportional across ALL phyla (broad, in-distribution; default)
  deep  dense+equal within the top-N largest families (narrow Goodfire-style footprint)
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Pinned to GTDB r220.0 to match Evo 2's training data (OpenGenome2)
METADATA_URL = (
    "https://data.gtdb.ecogenomic.org/releases/release220/220.0/bac120_metadata_r220.tsv.gz"
)
TREE_URL = "https://data.gtdb.ecogenomic.org/releases/release220/220.0/bac120_r220.tree"

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
    df = pd.read_csv(
        gz_path,
        sep="\t",
        comment="#",  # handle potential leading comment line
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def parse_taxonomy(taxonomy_str: str) -> tuple[str, str, str]:
    """Return (phylum, genus, species) with GTDB rank prefixes stripped."""
    ranks = {p[:3]: p[3:] for p in taxonomy_str.split(";")}
    return ranks.get("p__", ""), ranks.get("g__", ""), ranks.get("s__", "")


def sample_wide(df: pd.DataFrame, target: int, random_state: int) -> pd.DataFrame:
    """Sample exactly target rows stratified by gtdb_phylum, proportional to phylum abundance."""
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
        # fix up rounding by adjusting quotas, prioritizing larger phyla to preserve distribution
        sorted_phyla = phylum_counts.index.tolist()  # already sorted by value_counts
        i = 0
        while delta != 0 and i < len(sorted_phyla):
            phylum = sorted_phyla[i]
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
        # Fill from not-yet-sampled species (matched on accession), largest phyla first
        sampled_ids = set(result["gtdb_accession"])
        remainder = df[~df["gtdb_accession"].isin(sampled_ids)].sort_values(
            "gtdb_phylum",
            key=lambda col: col.map(phylum_counts),
            ascending=False,
        )
        extra = remainder.iloc[: target - len(result)]
        result = pd.concat([result, extra], ignore_index=True)

    return result.reset_index(drop=True)


def sample_deep(
    df: pd.DataFrame, n_families: int, per_family: int, random_state: int
) -> pd.DataFrame:
    """Sample per_family species from each of the top n_families largest families."""
    top_families = df["gtdb_family"].value_counts().head(n_families).index.tolist()

    sampled_frames = []
    for family in top_families:
        group = df[df["gtdb_family"] == family]
        n_actual = min(per_family, len(group))
        sampled_frames.append(group.sample(n=n_actual, random_state=random_state))

    return pd.concat(sampled_frames, ignore_index=True).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sampling",
        choices=["wide", "deep"],
        default="wide",
        help="wide=phylum-proportional across all phyla; deep=dense within top families",
    )
    p.add_argument(
        "--target", type=int, default=TARGET_N, help="total species to sample (wide mode)"
    )
    p.add_argument(
        "--n-families",
        type=int,
        default=24,
        help="number of largest families to draw from (deep mode)",
    )
    p.add_argument(
        "--per-family", type=int, default=100, help="species sampled per family (deep mode)"
    )
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output manifest path (default depends on --sampling)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    # Keep only GTDB species representatives for bac120.tree (phylogenetic baseline)
    before = len(df)
    df = df[df["gtdb_representative"] == "t"].copy()
    print(f"  Kept {len(df):,} species representatives (of {before:,} total genomes)")

    # Parse taxonomy
    df[["gtdb_phylum", "gtdb_genus", "species_name"]] = df["gtdb_taxonomy"].apply(
        lambda s: pd.Series(parse_taxonomy(s))
    )
    df["gtdb_family"] = df["gtdb_taxonomy"].str.extract(r"f__([^;]*)")[0].fillna("")

    # Strip accession prefix
    df["gtdb_accession"] = df["accession"]
    df["ncbi_accession"] = df["accession"].str.replace(r"^(GB_|RS_)", "", regex=True)

    # Drop rows with empty phylum (unassigned)
    before = len(df)
    df = df[df["gtdb_phylum"] != ""].copy()
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after:,} rows with empty/unassigned phylum")

    # Sort for reproducibility (helpful if newer GTDB release used)
    df = df.sort_values("gtdb_accession").reset_index(drop=True)

    print(f"  {len(df):,} rows with assigned phylum across {df['gtdb_phylum'].nunique()} phyla")

    # --- Sample ---
    if args.sampling == "wide":
        print(f"Wide sampling {args.target} species stratified by phylum...")
        sampled = sample_wide(df, args.target, args.seed)
        assert len(sampled) == args.target, f"Expected {args.target} rows, got {len(sampled)}"
        out_path = args.out or DATA_DIR / f"gtdb_{args.target}_manifest.csv"
    else:
        print(f"Deep sampling {args.per_family}/family from top {args.n_families} families...")
        sampled = sample_deep(df, args.n_families, args.per_family, args.seed)
        out_path = args.out or DATA_DIR / f"gtdb_top{args.n_families}_dense_manifest.csv"

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
    manifest.to_csv(out_path, index=False)
    print(f"Wrote manifest: {out_path} ({len(manifest)} rows)")

    # --- Print distribution ---
    group_col = "gtdb_phylum" if args.sampling == "wide" else "gtdb_family"
    print(f"\n{group_col} distribution in sample:")
    for name, count in sampled[group_col].value_counts().items():
        print(f"  {name:<50s} {count:>4d}")
    print(f"\nTotal: {len(manifest)}")


if __name__ == "__main__":
    main()
