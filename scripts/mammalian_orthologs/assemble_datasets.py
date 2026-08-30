"""Assemble the analysis datasets from the QC'd loci manifest, per the locked decisions: - EXCLUDE qc-fail AND length_outlier loci (partial/truncated models)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

OUT = Path("data/mammalian_orthologs")
LOCI = OUT / "loci"
CAP = 600
MIN_SP = 10
SEED = 17
LARGE = {"cytochrome_p450", "hox"}          # >CAP, capped in core (Ras 536 < CAP -> kept whole)
OR_FAM = "olfactory_receptors"              # separate capped track
MEDIUM_KEEPALL = {"globins", "opsins", "carbonic_anhydrase"}
ANCHORS = {"nitric_oxide_synthase", "heme_oxygenase"}


def _seq(group: str, species: str, key: str) -> str:
    f = LOCI / f"{group}__{species}.json"
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
    groups.sort(key=lambda g: sizes[g], reverse=True)   # stable: more-species groups first
    kept, n = [], 0
    for g in groups:
        if n >= cap:
            break
        kept.append(g)
        n += sizes[g]
    return df[df["group"].isin(kept)]


def main() -> None:
    m = pd.read_csv(OUT / "loci_manifest.csv")
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
        ag = d[d["analyzable"]]["group"].nunique() if "analyzable" in d else \
            (d.groupby("group")["species"].nunique() >= MIN_SP).sum()
        print(f"  {name:15} {len(d):>5} | {d['group'].nunique():>4} | {ag:>4}")
        print(d.groupby("family").size().to_string().replace("\n", "\n     "))


def _write(name: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df.to_csv(OUT / f"{name}_manifest.csv", index=False)
    for kind, col in [("transcript", "locus_seq"), ("cds", "cds_seq")]:
        d = OUT / "seqs" / name / kind
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
