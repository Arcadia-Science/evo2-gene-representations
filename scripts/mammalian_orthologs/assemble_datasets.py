"""Assemble the analysis datasets from the QC'd loci manifest, per the locked decisions: - EXCLUDE
qc-fail AND length_outlier loci (partial/truncated models).
"""

from __future__ import annotations
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "mammalian_orthologs"
LOCI_CACHE = OUT_DIR / "loci"
STOP_CODONS = {"TAA", "TAG", "TGA"}

# QC thresholds
N_MAX = 0.02  # max ambiguous-base fraction of the locus
GAP_RUN = 50  # an N-run >= this many bp = assembly gap through the gene
LEN_LO, LEN_HI = 0.2, 5.0  # plausible length band relative to the group's median (flag only)
CAP = 600
MIN_SP = 10
SEED = 17
LARGE = {"cytochrome_p450", "hox"}  # >CAP, capped in core (Ras 536 < CAP -> kept whole)
OR_FAM = "olfactory_receptors"  # separate capped track
MEDIUM_KEEPALL = {"globins", "opsins", "carbonic_anhydrase"}
ANCHORS = {"nitric_oxide_synthase", "heme_oxygenase"}


def _n_frac_and_run(seq: str) -> tuple[float, int]:
    if not seq:
        return 1.0, 0
    n = seq.count("N")
    # longest run of N
    run = mx = 0
    for c in seq:
        if c == "N":
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    return n / len(seq), mx


def qc_locus(rec: dict) -> tuple[bool, list[str]]:
    """Return (pass, flags). Hard fails vs soft flags noted in comments."""
    flags = []
    err = rec.get("error")
    if isinstance(err, str) and err:  # pandas fills the 'error' column with NaN (a float, truthy!)
        return False, [err]  # for success rows — only a real string error counts
    cds, locus = rec.get("cds_seq") or "", rec.get("locus_seq") or ""
    if rec.get("biotype") != "protein_coding":
        flags.append("not_protein_coding")  # hard
    if not locus:
        flags.append("no_locus_seq")  # hard
    if not cds:
        flags.append("no_cds")  # hard
    else:
        if len(cds) % 3 != 0:
            flags.append("cds_not_mult3")  # hard
        if not cds.startswith("ATG"):
            flags.append("no_start_codon")  # hard
        if cds[-3:] not in STOP_CODONS:
            flags.append("no_stop_codon")  # hard
    nf, run = _n_frac_and_run(locus)
    if nf > N_MAX:
        flags.append(f"n_frac_{nf:.3f}")  # hard
    if run >= GAP_RUN:
        flags.append(f"gap_run_{run}")  # hard
    hard = {
        "not_protein_coding",
        "no_locus_seq",
        "no_cds",
        "cds_not_mult3",
        "no_start_codon",
        "no_stop_codon",
    }
    is_hard = any(f in hard or f.startswith("n_frac_") or f.startswith("gap_run_") for f in flags)
    return (not is_hard), flags


def build_manifest(work: pd.DataFrame) -> None:
    recs = []
    for f in LOCI_CACHE.glob("*.json"):
        recs.append(json.loads(f.read_text()))
    df = pd.DataFrame(recs)
    # Guard columns that only exist on successful records (error rows lack seqs/coords).
    for c in [
        "locus_seq",
        "cds_seq",
        "biotype",
        "gene_id",
        "transcript_id",
        "chrom",
        "tx_start",
        "tx_end",
        "strand",
        "n_overlap_genes",
        "error",
    ]:
        if c not in df.columns:
            df[c] = None
    df["locus_seq"] = df["locus_seq"].fillna("")
    df["cds_seq"] = df["cds_seq"].fillna("")
    # group-median locus length for the plausible-length flag
    med = df.assign(_ll=df["locus_seq"].fillna("").str.len()).groupby("human_gene")["_ll"].median()
    man_rows = []
    for _, r in df.iterrows():
        ok, flags = qc_locus(r)
        ll = len(r.get("locus_seq") or "")
        m = med.get(r["human_gene"], 0)
        if m and not (LEN_LO * m <= ll <= LEN_HI * m):
            flags = flags + ["length_outlier"]  # soft flag only
        nf, run = _n_frac_and_run(r.get("locus_seq") or "")
        man_rows.append(
            {
                "family": r.get("family"),
                "group": r["human_gene"],
                "species": r["species"],
                "clade": r.get("clade"),
                "gene_id": r.get("gene_id"),
                "transcript_id": r.get("transcript_id"),
                "chrom": r.get("chrom"),
                "tx_start": r.get("tx_start"),
                "tx_end": r.get("tx_end"),
                "strand": r.get("strand"),
                "biotype": r.get("biotype"),
                "locus_len": ll,
                "cds_len": len(r.get("cds_seq") or ""),
                "n_frac": round(nf, 4),
                "max_n_run": run,
                "n_overlap_genes": r.get("n_overlap_genes"),
                "qc_pass": ok,
                "qc_flags": ";".join(str(x) for x in flags),
            }
        )
    man = pd.DataFrame(man_rows)
    man.to_csv(OUT_DIR / "loci_manifest.csv", index=False)
    # write per-family FASTAs of QC-passing loci
    for kind, col in [("transcript", "locus_seq"), ("cds", "cds_seq")]:
        d = OUT_DIR / "seqs" / kind
        d.mkdir(parents=True, exist_ok=True)
        by_fam = {}
        for _, r in df.iterrows():
            ok, _ = qc_locus(r)
            if not ok:
                continue
            fam = r.get("family")
            by_fam.setdefault(fam, []).append(
                (f"{r['human_gene']}|{r['species']}|{r.get('gene_id')}", r.get(col) or "")
            )
        for fam, entries in by_fam.items():
            with open(d / f"{fam}.fasta", "w") as fh:
                for hid, seq in entries:
                    fh.write(f">{hid}\n{seq}\n")
    npass = int(man["qc_pass"].sum())
    print(
        f"\nWrote {OUT_DIR / 'loci_manifest.csv'}: {len(man)} loci, {npass} QC-pass "
        f"({100 * npass / len(man):.0f}%)"
    )
    # post-QC sizes per family
    p = man[man["qc_pass"]]
    print("\n=== POST-QC loci per family ===")
    print(p.groupby("family").size().to_string())


def _seq(group: str, species: str, key: str) -> str:
    f = LOCI_CACHE / f"{group}__{species}.json"
    if not f.exists():
        return ""
    return json.loads(f.read_text()).get(key) or ""


def _cap_by_groups(df: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep whole groups until ~cap loci. Prefer groups with more species (better phylogeny signal),
    breaking ties randomly (seeded)."""
    rng = random.Random(seed)
    sizes = df.groupby("group").size()
    groups = list(sizes.index)
    rng.shuffle(groups)
    groups.sort(key=lambda g: sizes[g], reverse=True)  # stable: more-species groups first
    kept, n = [], 0
    for g in groups:
        if n >= cap:
            break
        kept.append(g)
        n += sizes[g]
    return df[df["group"].isin(kept)]


def main() -> None:
    m = pd.read_csv(OUT_DIR / "loci_manifest.csv")
    flags = m["qc_flags"].fillna("")
    keep = m[m["qc_pass"] & ~flags.str.contains("length_outlier")].copy()
    print(f"kept {len(keep)}/{len(m)} loci after excluding qc-fail + length_outlier")

    # analyzable flag (>=MIN_SP species in a group)
    spg = keep.groupby("group")["species"].transform("nunique")
    keep["group_n_species"] = spg
    keep["analyzable"] = spg >= MIN_SP

    # ── COMPLETE ────────────────────────────────────────────────────────────────
    _write("complete", keep)

    # ── BALANCED CORE ─────────────────────────────────────────────────────────────
    core_parts = []
    for fam, sub in keep.groupby("family"):
        if fam == OR_FAM:
            continue  # separate track
        if fam in LARGE:
            sub = _cap_by_groups(sub, CAP, SEED)
        core_parts.append(sub)
    core = pd.concat(core_parts, ignore_index=True)
    _write("balanced_core", core)

    # ── OR TRACK (separate, capped) ───────────────────────────────────────────────
    or_track = _cap_by_groups(keep[keep["family"] == OR_FAM], CAP, SEED)
    _write("or_track", or_track)

    # summary
    print("\n=== dataset sizes (loci | groups | analyzable groups) ===")
    for name, d in [("complete", keep), ("balanced_core", core), ("or_track", or_track)]:
        ag = (
            d[d["analyzable"]]["group"].nunique()
            if "analyzable" in d
            else (d.groupby("group")["species"].nunique() >= MIN_SP).sum()
        )
        print(f"  {name:15} {len(d):>5} | {d['group'].nunique():>4} | {ag:>4}")
        print(d.groupby("family").size().to_string().replace("\n", "\n     "))


def _write(name: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df.to_csv(OUT_DIR / f"{name}_manifest.csv", index=False)
    for kind, col in [("transcript", "locus_seq"), ("cds", "cds_seq")]:
        d = OUT_DIR / "seqs" / name / kind
        d.mkdir(parents=True, exist_ok=True)
        for fam, sub in df.groupby("family"):
            with open(d / f"{fam}.fasta", "w") as fh:
                for _, r in sub.iterrows():
                    s = _seq(r["group"], r["species"], col)
                    if s:
                        fh.write(f">{r['group']}|{r['species']}|{r.get('gene_id')}\n{s}\n")
    print(f"wrote {name}: {len(df)} loci -> {name}_manifest.csv + seqs/{name}/")


if __name__ == "__main__":
    main()
