"""
Evo2 zero-shot ClinVar variant-effect-prediction (VEP) benchmark.

A faithful reproduction of the Evo2 zero-shot VEP procedure from the official
`notebooks/brca1/brca1_zero_shot_vep.ipynb` (confirmed against the paper, Fig. 3a):
extract an 8192 bp GRCh38 window per SNV, score the reference and variant windows
with `model.score_sequences`, take delta = logL(alt) - logL(ref), and report
AUROC / AUPRC (Pathogenic = positive) against the reported Evo2 values
(AUROC 0.830, AUPRC 0.883).

Dataset: songlab/clinvar_vs_benign (50,164 ClinVar missense SNVs, GRCh38).
Needs the GRCh38 primary assembly (Ensembl release-111) and pyfaidx. The genome
is auto-downloaded to data/genome/GRCh38.primary_assembly.fa on first run if absent
(~880 MB download, ~3.1 GB on disk); see ensure_genome().

Usage:
    # Windowing + allele check only, no model/GPU (quick smoke test):
    uv run python scripts/evo2/test_evo2.py --dry-run --n-variants 50

    # 1000-variant pilot (~85 min on one A10G):
    uv run python scripts/evo2/test_evo2.py --n-variants 1000

    # Full benchmark, all 50,164 variants (~70 h on one A10G):
    uv run python scripts/evo2/test_evo2.py
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GENOME_FASTA = REPO / "data" / "genome" / "GRCh38.primary_assembly.fa"
# Ensembl release-111 primary assembly (gzipped, ~880 MB; ~3.1 GB uncompressed).
# Sequence names are '1'..'22','X','Y','MT' — matching songlab/clinvar_vs_benign.
GENOME_URL = (
    "https://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
)
MODEL_NAME = "evo2_7b"
WINDOW_SIZE = 8192  # matches the official Evo2 brca1 notebook


# ── Reference genome ──────────────────────────────────────────────────────────


def ensure_genome(fasta_path: Path = GENOME_FASTA, url: str = GENOME_URL) -> Path:
    """Return the path to the GRCh38 FASTA, downloading it if it's not already
    on disk at the expected spot. Streams the gzip from Ensembl and decompresses
    in place (no extra deps). The ~3 GB .fai index is built lazily by pyfaidx on
    first use. Safe to call repeatedly — it's a no-op once the file exists."""
    import gzip
    import shutil
    import urllib.request

    if fasta_path.exists():
        return fasta_path

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_gz = fasta_path.with_suffix(fasta_path.suffix + ".gz.part")
    print(f"    genome FASTA not found at {fasta_path}")
    print(f"    downloading GRCh38 (Ensembl release-111, ~880 MB) from {url}")
    print("    this is a one-time download (~3.1 GB uncompressed) ...")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp_gz, "wb") as fh:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 20  # 1 MiB
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(
                        f"\r      {done >> 20} / {total >> 20} MiB ({pct:4.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
        print("    decompressing ...")
        tmp_fa = fasta_path.with_suffix(fasta_path.suffix + ".part")
        with gzip.open(tmp_gz, "rb") as gz, open(tmp_fa, "wb") as out:
            shutil.copyfileobj(gz, out, length=1 << 22)  # 4 MiB blocks
        tmp_fa.rename(fasta_path)  # atomic: only appears complete if we got here
    finally:
        tmp_gz.unlink(missing_ok=True)
    print(f"    genome ready at {fasta_path}")
    return fasta_path


# ── ClinVar zero-shot VEP benchmark ───────────────────────────────────────────


def load_clinvar() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("songlab/clinvar_vs_benign", split="test")
    df = ds.to_pandas()
    # chrom as str ('1'..'22','X'); pos is 1-based.
    df["chrom"] = df["chrom"].astype(str)
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Shuffle and take n rows preserving the Benign/Pathogenic ratio."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for _label, grp in df.groupby("label"):
        k = max(1, round(n * len(grp) / len(df)))
        idx = rng.choice(len(grp), size=min(k, len(grp)), replace=False)
        parts.append(grp.iloc[idx])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


class Genome:
    """Thin pyfaidx wrapper that tolerates 'chr' prefix differences."""

    def __init__(self, fasta_path: Path):
        from pyfaidx import Fasta

        self.fa = Fasta(str(fasta_path), rebuild=False)
        self.keys = set(self.fa.keys())

    def _resolve(self, chrom: str) -> str:
        if chrom in self.keys:
            return chrom
        for cand in (f"chr{chrom}", chrom.replace("chr", "")):
            if cand in self.keys:
                return cand
        raise KeyError(f"chrom {chrom!r} not in genome ({list(self.keys)[:5]}...)")

    def seq(self, chrom: str, start: int, end: int) -> str:
        """0-based, half-open [start, end). Returns uppercase string."""
        key = self._resolve(chrom)
        return str(self.fa[key][start:end]).upper()

    def length(self, chrom: str) -> int:
        return len(self.fa[self._resolve(chrom)])


def build_windows(df: pd.DataFrame, genome: Genome, window: int = WINDOW_SIZE):
    """Build ref/var windows per the brca1 notebook. Returns (records, stats).

    records: list of dicts with keys idx, ref_seq, var_seq, label, allele_ok.
    Variants whose genome ref base does not match the dataset ref are flagged
    (allele_ok=False) and excluded from scoring.
    """
    half = window // 2
    records = []
    n_mismatch = 0
    for i, row in df.iterrows():
        p = int(row["pos"]) - 1  # 0-based position of the SNV
        chrom_len = genome.length(row["chrom"])
        start = max(0, p - half)
        end = min(chrom_len, p + half)
        ref_seq = genome.seq(row["chrom"], start, end)
        snv_pos_in_ref = min(half, p)  # offset of the SNV within the window

        genome_ref = ref_seq[snv_pos_in_ref] if snv_pos_in_ref < len(ref_seq) else "?"
        allele_ok = genome_ref == row["ref"]
        if not allele_ok:
            n_mismatch += 1

        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref + 1 :]
        records.append(
            {
                "idx": i,
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "ref": row["ref"],
                "alt": row["alt"],
                "label": row["label"],
                "ref_seq": ref_seq,
                "var_seq": var_seq,
                "snv_pos_in_ref": snv_pos_in_ref,
                "genome_ref": genome_ref,
                "allele_ok": allele_ok,
            }
        )
    stats = {
        "n": len(records),
        "n_mismatch": n_mismatch,
        "mismatch_rate": n_mismatch / max(1, len(records)),
    }
    return records, stats


def score_records(records, batch_size: int, rc: bool, window: int = WINDOW_SIZE):
    """Score ref/var windows with Evo2. Deduplicates ref windows like the notebook."""
    from evo2 import Evo2

    usable = [r for r in records if r["allele_ok"]]
    print(f"    Loading {MODEL_NAME}...", flush=True)
    model = Evo2(MODEL_NAME)

    # Dedup reference windows.
    ref_seq_to_index = {}
    ref_seqs = []
    for r in usable:
        if r["ref_seq"] not in ref_seq_to_index:
            ref_seq_to_index[r["ref_seq"]] = len(ref_seqs)
            ref_seqs.append(r["ref_seq"])
    var_seqs = [r["var_seq"] for r in usable]
    print(
        f"    Scoring {len(ref_seqs)} unique ref windows + {len(var_seqs)} var windows "
        f"(window={window}, rc={rc}, batch_size={batch_size})...",
        flush=True,
    )

    t0 = time.time()
    ref_scores = model.score_sequences(
        ref_seqs, batch_size=batch_size, average_reverse_complement=rc
    )
    var_scores = model.score_sequences(
        var_seqs, batch_size=batch_size, average_reverse_complement=rc
    )
    dt = time.time() - t0
    print(f"    Scored in {dt / 60:.1f} min", flush=True)

    ref_scores = np.array(ref_scores)
    var_scores = np.array(var_scores)
    for j, r in enumerate(usable):
        ref_idx = ref_seq_to_index[r["ref_seq"]]
        r["ref_score"] = float(ref_scores[ref_idx])
        r["var_score"] = float(var_scores[j])
        r["delta_score"] = r["var_score"] - r["ref_score"]
    return usable


def compute_metrics(scored):
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.DataFrame(scored)
    y_true = (df["label"] == "Pathogenic").astype(int).values
    # Pathogenic variants are expected to LOWER the likelihood -> more negative
    # delta. Prediction score for "pathogenic" is therefore -delta.
    pred = -df["delta_score"].values
    auroc = roc_auc_score(y_true, pred)
    auprc = average_precision_score(y_true, pred)
    return df, auroc, auprc


def run_vep_benchmark(args) -> None:
    """Build windows from the reference genome, score with Evo2, and report
    AUROC/AUPRC. With --dry-run, stops after the windowing + allele-verification
    step without loading the model (no GPU)."""
    window = args.window
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO / "results" / f"{datetime.date.today().isoformat()}_evo2-clinvar"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  ClinVar VEP benchmark: {MODEL_NAME}")
    print(f"    genome : {GENOME_FASTA}")
    print(f"    window : {window} bp")
    print(f"    subset : {args.n_variants or 'ALL'}")
    print(f"    rc     : {args.rc}")
    print(f"{'─' * 60}")

    ensure_genome(GENOME_FASTA)

    df = load_clinvar()
    print(f"    loaded {len(df)} variants ({df.label.value_counts().to_dict()})")
    df = stratified_sample(df, args.n_variants, args.seed)
    print(f"    sampled {len(df)} ({df.label.value_counts().to_dict()})")

    print("    building windows + verifying ref alleles against genome...")
    genome = Genome(GENOME_FASTA)
    records, stats = build_windows(df, genome, window)
    print(
        f"    allele check: {stats['n_mismatch']}/{stats['n']} mismatch "
        f"({stats['mismatch_rate'] * 100:.2f}%)"
    )
    lens = {len(r["ref_seq"]) for r in records}
    print(f"    window lengths present: {sorted(lens)[:5]}{'...' if len(lens) > 5 else ''}")

    if args.dry_run:
        print("\n    [dry-run] windows built and alleles verified; model NOT loaded.")
        sample = [
            {k: r[k] for k in ("chrom", "pos", "ref", "alt", "label", "genome_ref", "allele_ok")}
            for r in records[:5]
        ]
        print(json.dumps(sample, indent=2))
        (out_dir / "dry_run_stats.json").write_text(json.dumps(stats, indent=2))
        return

    scored = score_records(records, batch_size=args.batch_size, rc=args.rc, window=window)
    df_out, auroc, auprc = compute_metrics(scored)

    tag = f"n{len(scored)}{'_rc' if args.rc else ''}_w{window}"
    parquet = out_dir / f"clinvar_vep_{tag}.parquet"
    df_out[
        ["chrom", "pos", "ref", "alt", "label", "ref_score", "var_score", "delta_score"]
    ].to_parquet(parquet, index=False)

    result = {
        "model": MODEL_NAME,
        "window": window,
        "rc": args.rc,
        "n_scored": len(scored),
        "n_excluded_allele_mismatch": stats["n_mismatch"],
        "auroc": auroc,
        "auprc": auprc,
        "reported_auroc": 0.830,
        "reported_auprc": 0.883,
    }
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(result, indent=2))

    print(f"\n    AUROC = {auroc:.4f}   (Evo2 reported 0.830)")
    print(f"    AUPRC = {auprc:.4f}   (Evo2 reported 0.883)")
    print(f"    scores  -> {parquet}")
    print(f"    metrics -> {out_dir / f'metrics_{tag}.json'}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n-variants",
        type=int,
        default=None,
        help="Subset size (default: all 50164). The full set is ~70 h on one A10G.",
    )
    p.add_argument(
        "--window",
        type=int,
        default=WINDOW_SIZE,
        help=f"Window size in bp (default: {WINDOW_SIZE})",
    )
    p.add_argument("--batch-size", type=int, default=1, help="Scoring batch size")
    p.add_argument("--rc", action="store_true", help="Average forward + reverse-complement scores")
    p.add_argument("--seed", type=int, default=0, help="Stratified-sample seed")
    p.add_argument(
        "--dry-run", action="store_true", help="Windowing + allele check only; no model/GPU"
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for VEP scores/metrics (default: results/<today>_evo2-clinvar)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print("\n== Evo2 ClinVar zero-shot VEP benchmark ==")
    try:
        run_vep_benchmark(args)
        print("\n== Done ==\n")
    except Exception as e:
        print(f"\n  FAILED — {e}")
        raise


if __name__ == "__main__":
    main()
