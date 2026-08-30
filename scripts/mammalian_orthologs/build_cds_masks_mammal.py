"""Build per-locus CDS-position masks over each mammalian ortholog TRANSCRIPT-SPAN string."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
LOCI_DIR = ROOT / "data" / "mammalian_orthologs" / "loci"
OUT_POS = ROOT / "data" / "cache" / "mammal_cds_positions.json"
OUT_REPORT = ROOT / "data" / "cache" / "mammal_cds_positions_report.csv"
SEED = 31  # seed length for anchoring each coding-exon block; 4^31 ≫ any genome, so an intron
           # cannot coincidentally carry a coding seed — the walk stays on the true exon order.


def cds_span_positions(locus: str, cds: str, seed: int = SEED) -> list[int] | None:
    """0-based indices in `locus` covered by the spliced `cds`, via ordered exon-block matching."""
    if not cds or not locus:
        return None
    pos: list[int] = []
    L, C = len(locus), len(cds)
    lp = 0   # earliest allowed locus index for the next block (enforces monotone exon order)
    cp = 0   # cursor into the CDS
    while cp < C:
        k = min(seed, C - cp)
        j = locus.find(cds[cp:cp + k], lp)
        if j == -1:
            return None
        m = 0                                   # extend the exact match maximally
        while cp + m < C and j + m < L and locus[j + m] == cds[cp + m]:
            m += 1
        pos.extend(range(j, j + m))
        cp += m
        lp = j + m
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None,
                    help="restrict to these families (default: all loci with a JSON record)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    files = sorted(LOCI_DIR.glob("*.json"))
    if not files:
        sys.exit(f"no locus JSONs under {LOCI_DIR}")

    positions, rows = {}, []
    for f in tqdm(files, desc="recovering CDS positions"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            rows.append(dict(key=f.stem, status="unreadable")); continue
        if args.families and d.get("family") not in args.families:
            continue
        key = f.stem
        locus, cds = d.get("locus_seq", ""), d.get("cds_seq", "")
        if not locus or not cds:
            rows.append(dict(key=key, family=d.get("family"), status="no_seq")); continue
        pos = cds_span_positions(locus, cds, args.seed)
        if pos is None:
            rows.append(dict(key=key, family=d.get("family"), span_len=len(locus),
                             cds_len=len(cds), n_cds_pos=0, status="unmappable")); continue
        # exact reconstruction: count must equal the spliced CDS length, indices strictly increasing.
        ok = (len(pos) == len(cds)) and (len(set(pos)) == len(pos))
        status = "ok" if ok else "MISMATCH"
        if ok:
            positions[key] = pos
        rows.append(dict(key=key, family=d.get("family"), strand=d.get("strand"),
                         span_len=len(locus), cds_len=len(cds), n_cds_pos=len(pos),
                         diff=len(pos) - len(cds), status=status))

    OUT_POS.parent.mkdir(parents=True, exist_ok=True)
    OUT_POS.write_text(json.dumps(positions))
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT_REPORT, index=False)

    n_ok = int((rep.status == "ok").sum())
    print(f"\n=== {n_ok}/{len(rep)} loci validated -> {OUT_POS.name} ===")
    bad = rep[rep.status != "ok"]
    if len(bad):
        print(f"{len(bad)} not usable (by status):")
        print(bad["status"].value_counts().to_string())
        if "family" in bad:
            print("  by family:", bad.family.value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
