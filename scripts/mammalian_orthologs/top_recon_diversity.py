"""Cross-species sequence diversity for the top reconstruction genes' 24-mammal orthologs."""

from __future__ import annotations
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "mammal_top_recon"
OUT = IN / "mammal_diversity.csv"
RECON = ROOT / "results" / "2026-07-24_evo2-gene-reconstruction" / "top_families.csv"
BASES = set("ACGT")


def read_fasta(fa: Path) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    for line in fa.read_text().splitlines():
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        elif line.strip():
            buf.append(line.strip().upper())
    if name:
        seqs[name] = "".join(buf)
    return seqs


def qc_length_filter(seqs: dict[str, str], lo: float = 0.7, hi: float = 1.4) -> tuple[dict, list]:
    """Remove orthologs with CDS lengths outside the allowed reference range."""
    import statistics as st

    if len(seqs) < 3:
        return seqs, []
    ref = (
        len(seqs["homo_sapiens"])
        if "homo_sapiens" in seqs
        else st.median(len(v) for v in seqs.values())
    )
    keep, dropped = {}, []
    for k, v in seqs.items():
        if ref * lo <= len(v) <= ref * hi:
            keep[k] = v
        else:
            dropped.append(f"{k}({len(v)}bp vs ref {int(ref)})")
    return keep, dropped


def mafft(seqs: dict[str, str]) -> dict[str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as f:
        for k, v in seqs.items():
            f.write(f">{k}\n{v}\n")
        inp = f.name
    out = subprocess.run(["mafft", "--auto", "--quiet", inp], capture_output=True, text=True)
    Path(inp).unlink(missing_ok=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:200])
    aln, name, buf = {}, None, []
    for line in out.stdout.splitlines():
        if line.startswith(">"):
            if name:
                aln[name] = "".join(buf)
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip().upper())
    if name:
        aln[name] = "".join(buf)
    return aln


def diff_cols(a: str, b: str) -> int:
    """Columns where the two aligned seqs differ (substitutions + indels; gap==gap is same)."""
    return sum(1 for x, y in zip(a, b, strict=False) if x != y)


def analyze(gene: str, aln: dict[str, str]) -> dict:
    species = list(aln)
    L = len(next(iter(aln.values())))
    cols = list(zip(*aln.values(), strict=False))
    variable = [c for c in cols if len({x for x in c}) > 1]
    # mean pairwise
    pair_diffs = [diff_cols(aln[a], aln[b]) for a, b in combinations(species, 2)]
    mean_bp_diff = sum(pair_diffs) / len(pair_diffs) if pair_diffs else 0
    mean_pct_id = 100 * (1 - mean_bp_diff / L) if L else 0
    row = {
        "gene": gene,
        "n_species": len(species),
        "aln_len": L,
        "n_variable_cols": len(variable),
        "pct_variable": round(100 * len(variable) / L, 2) if L else 0,
        "mean_pairwise_bp_diff": round(mean_bp_diff, 1),
        "mean_pairwise_pct_id": round(mean_pct_id, 2),
    }
    # private (autapomorphic) bp per species: a column base present in exactly ONE species,
    # attributed to that species. One O(L*n) pass over columns via base singletons.
    private_by_species = {s: 0 for s in species}
    for col in cols:  # col is a tuple of chars, one per species (order == species)
        counts: dict[str, int] = {}
        for ch in col:
            if ch in BASES:
                counts[ch] = counts.get(ch, 0) + 1
        singletons = {ch for ch, c in counts.items() if c == 1}
        if singletons:
            for s, ch in zip(species, col, strict=False):
                if ch in singletons:
                    private_by_species[s] += 1
    mp = max(private_by_species, key=private_by_species.get)
    row["most_private_species"] = mp
    row["most_private_bp"] = private_by_species[mp]

    # most human-divergent
    if "homo_sapiens" in aln:
        h = aln["homo_sapiens"]
        divs = {s: diff_cols(h, aln[s]) for s in species if s != "homo_sapiens"}
        if divs:
            md = max(divs, key=divs.get)
            row["most_divergent_species"] = md
            row["most_divergent_bp_diff_vs_human"] = divs[md]
            row["most_divergent_pct_diff"] = round(100 * divs[md] / L, 2) if L else 0
            row["private_bp_in_most_divergent"] = private_by_species.get(md, 0)
    return row


def main() -> None:
    fastas = sorted(IN.glob("*.fasta"))
    if not fastas:
        sys.exit("no fastas yet")
    recon_map = {}
    if RECON.exists():
        tf = pd.read_csv(RECON)
        # top_families.csv is only top-10; fall back to full ranking if present
        RECON.parent / "top_families.csv"
        recon_map = dict(zip(tf["family"], tf["mean_corrected_nt"], strict=False))
    rows = []
    for fa in fastas:
        gene = fa.stem
        seqs = read_fasta(fa)
        seqs, dropped = qc_length_filter(seqs)
        if len(seqs) < 3:
            print(f"{gene}: only {len(seqs)} species after QC — skip align")
            continue
        try:
            aln = mafft(seqs)
            r = analyze(gene, aln)
            r["n_dropped_qc"] = len(dropped)
            if dropped:
                print(f"  {gene}: dropped {len(dropped)} length-outlier(s): {', '.join(dropped)}")
            r["recon_corrected_nt"] = recon_map.get(gene)
            rows.append(r)
            mp = r.get("most_private_species", "?")
            print(
                f"{gene:8s} n={r['n_species']:2d} aln={r['aln_len']:4d} "
                f"meanID={r['mean_pairwise_pct_id']:5.1f}% | "
                f"most-private={mp.split('_')[0][:9]:9s} "
                f"({r.get('most_private_bp', '?')}bp private) | "
                f"most-divergent={r.get('most_divergent_species', '?').split('_')[0][:9]} "
                f"Δ{r.get('most_divergent_bp_diff_vs_human', '?')}bp"
            )
        except Exception as e:  # noqa: BLE001
            print(f"{gene}: align/analyze failed: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT, index=False)
        print(f"\n-> {OUT}  ({len(df)} genes)")


if __name__ == "__main__":
    main()
