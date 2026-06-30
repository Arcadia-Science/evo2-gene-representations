"""Generate composition-matched control sequence sets for the Evo2 gene-family panel.

Section 3 of the project: test whether Evo2's within-family geodesic signal (which
recovers per-family phylogeny, ρ up to 0.78) is genuine higher-order structure or merely
a reflection of sequence COMPOSITION — which is itself phylogenetically structured (GC
content and k-mer usage vary by clade). Each natural CDS is replaced by a control that
preserves a specific level of composition and destroys everything above it; we then
re-embed and ask whether the geodesic still recovers the *source organism's* taxonomy
(a fixed property preserved by labelling). If it does, the "phylogeny" signal is a
composition artifact; if it collapses, Evo2 reads higher-order features.

Four controls, increasingly destructive (length-preserving except where noted):
  dinuc_shuffle     Altschul–Erikson shuffle — EXACT mono+dinucleotide frequencies,
                    destroys codons/motifs/higher k-mers.
  codon_shuffle     reorder the CDS's own codons — exact codon multiset + amino-acid
                    composition + reading frame, destroys codon order/motifs.
  synonymous_recode keep the protein, resample each codon from the family's codon-usage
                    distribution — preserves protein + family codon bias, changes nt.
  gc_match          random sequence, length- and GC-fraction-matched only.

Output mirrors the natural data dir so the control embedder can consume it:
  data/evo2_gene_families/controls/<control>/{manifest.csv, <family>.fasta}
(metadata — org_gene/family/taxonomy — is copied verbatim; only the sequences change.)

Usage:
    uv run python analyses/make_control_sequences.py            # all 4 controls
    uv run python analyses/make_control_sequences.py --controls dinuc_shuffle
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/evo2_gene_families")
CONTROL_ROOT = DATA_DIR / "controls"
# kmer4/kmer6 preserve the exact 4-mer/6-mer spectrum (k=6 matches the k-mer ground-truth
# baseline) — the high-order rungs of the composition gradient.
CONTROLS = ["dinuc_shuffle", "codon_shuffle", "synonymous_recode", "gc_match",
            "kmer4_shuffle", "kmer6_shuffle"]
SEED = 1234

# Standard genetic code (frame-0 translation for amino-acid grouping).
_CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L",
    "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V",
    "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P",
    "CCC": "P", "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*",
    "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "TGT": "C",
    "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
    "GGG": "G",
}


# ── control generators ───────────────────────────────────────────────────────


def dinuc_shuffle(seq: str, rng: random.Random, max_tries: int = 100) -> str:
    """Altschul–Erikson dinucleotide-preserving shuffle: a uniformly-random sequence
    with EXACTLY the same first/last symbol and mono+dinucleotide frequencies.

    Builds the de Bruijn edge multiset, fixes a random arborescence of "last edges"
    toward the terminal symbol (retrying until it spans), shuffles the rest, and walks
    the Eulerian path. (Treats any alphabet symbol uniformly, so N/ambiguity is safe.)
    """
    s = seq.upper()
    n = len(s)
    if n < 4 or len(set(s)) < 2:
        return s
    last = s[-1]
    verts = set(s)
    edges = defaultdict(list)
    for a, b in zip(s[:-1], s[1:]):
        edges[a].append(b)

    for _ in range(max_tries):
        avail = {x: list(ys) for x, ys in edges.items()}
        last_edge: dict[str, str] = {}
        ok = True
        for x in verts:
            if x == last:
                continue
            if not avail[x]:
                ok = False
                break
            last_edge[x] = avail[x].pop(rng.randrange(len(avail[x])))
        if not ok:
            continue
        # last_edge must form a tree directed into `last` (every vertex reaches it).
        def reaches(x: str) -> bool:
            seen = set()
            while x != last:
                if x in seen or x not in last_edge:
                    return False
                seen.add(x)
                x = last_edge[x]
            return True

        if not all(reaches(x) for x in verts if x != last):
            continue
        for x in verts:
            rng.shuffle(avail[x])
            if x in last_edge:
                avail[x].append(last_edge[x])  # terminal edge consumed last
        out = [s[0]]
        cur = s[0]
        idx = {x: 0 for x in verts}
        for _ in range(n - 1):
            nxt = avail[cur][idx[cur]]
            idx[cur] += 1
            out.append(nxt)
            cur = nxt
        return "".join(out)
    return s  # rare: failed to construct → leave unchanged


def _euler_shuffle(symbols: list, rng: random.Random, max_tries: int = 100) -> list:
    """Altschul–Erikson shuffle on an arbitrary symbol list: a uniformly-random
    reordering that preserves the count of every adjacent (symbol_i, symbol_{i+1}) pair
    and the first/last symbol. (dinuc_shuffle is the nucleotide-symbol special case.)"""
    n = len(symbols)
    if n < 4 or len(set(symbols)) < 2:
        return list(symbols)
    last = symbols[-1]
    verts = set(symbols)
    edges = defaultdict(list)
    for a, b in zip(symbols[:-1], symbols[1:]):
        edges[a].append(b)
    for _ in range(max_tries):
        # default [] for vertices with no outgoing edge (e.g. the terminal (k-1)-mer,
        # which appears only as the last token and has no successor).
        avail = {x: list(edges.get(x, [])) for x in verts}
        last_edge: dict = {}
        ok = True
        for x in verts:
            if x == last:
                continue
            if not avail[x]:
                ok = False
                break
            last_edge[x] = avail[x].pop(rng.randrange(len(avail[x])))
        if not ok:
            continue

        def reaches(x):
            seen = set()
            while x != last:
                if x in seen or x not in last_edge:
                    return False
                seen.add(x)
                x = last_edge[x]
            return True

        if not all(reaches(x) for x in verts if x != last):
            continue
        for x in verts:
            rng.shuffle(avail[x])
            if x in last_edge:
                avail[x].append(last_edge[x])
        out = [symbols[0]]
        cur = symbols[0]
        idx = {x: 0 for x in verts}
        for _ in range(n - 1):
            nxt = avail[cur][idx[cur]]
            idx[cur] += 1
            out.append(nxt)
            cur = nxt
        return out
    return list(symbols)


def klet_shuffle(seq: str, k: int, rng: random.Random) -> str:
    """Shuffle preserving the EXACT k-mer spectrum (k>=2): an Eulerian walk on the graph
    whose nodes are (k-1)-mers and edges are k-mers. At higher k fewer (k-1)-mers repeat,
    so less reordering is possible — the shuffle preserves more and destroys less."""
    s = seq.upper()
    if len(s) < k + 1:
        return s
    toks = [s[i : i + k - 1] for i in range(len(s) - (k - 1) + 1)]  # overlapping (k-1)-mers
    shuf = _euler_shuffle(toks, rng)
    out = shuf[0]
    for t in shuf[1:]:
        out += t[-1]  # each step extends by one nucleotide; overlaps are valid by construction
    return out


def codon_shuffle(seq: str, rng: random.Random) -> str:
    """Reorder the CDS's own codons (frame 0); trailing partial codon stays put."""
    s = seq.upper()
    ncod = len(s) // 3
    codons = [s[i * 3 : i * 3 + 3] for i in range(ncod)]
    rng.shuffle(codons)
    return "".join(codons) + s[ncod * 3 :]


def build_family_codon_usage(seqs_by_family: dict[str, list[str]]) -> dict[str, dict]:
    """family -> {aa: (codons, weights)} from all member CDS (frame 0)."""
    usage: dict[str, dict] = {}
    for fam, seqs in seqs_by_family.items():
        by_aa: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in seqs:
            s = s.upper()
            for i in range(0, len(s) - 2, 3):
                cod = s[i : i + 3]
                aa = _CODON.get(cod)
                if aa is not None:
                    by_aa[aa][cod] += 1
        usage[fam] = {aa: (list(c), list(c.values())) for aa, c in by_aa.items()}
    return usage


def synonymous_recode(seq: str, fam_usage: dict, rng: random.Random) -> str:
    """Keep the protein; resample each codon from the family's synonymous usage."""
    s = seq.upper()
    ncod = len(s) // 3
    out = []
    for i in range(ncod):
        cod = s[i * 3 : i * 3 + 3]
        aa = _CODON.get(cod)
        if aa is None or aa not in fam_usage:
            out.append(cod)
            continue
        codons, weights = fam_usage[aa]
        out.append(rng.choices(codons, weights=weights, k=1)[0])
    return "".join(out) + s[ncod * 3 :]


def gc_match(seq: str, rng: random.Random) -> str:
    """Random length-matched sequence with the same G+C fraction (only)."""
    s = seq.upper()
    valid = sum(c in "ACGT" for c in s)
    gc = sum(c in "GC" for c in s)
    p = gc / valid if valid else 0.5
    return "".join(rng.choice("GC") if rng.random() < p else rng.choice("AT") for _ in s)


# ── FASTA IO ─────────────────────────────────────────────────────────────────


def load_family_fastas() -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Return ({org_gene: cds}, {family: [(header, org_gene), ...] in file order})."""
    seqs: dict[str, str] = {}
    order: dict[str, list[tuple[str, str]]] = {}
    for fasta in sorted(DATA_DIR.glob("*.fasta")):
        fam = fasta.stem
        order[fam] = []
        cur_hdr, cur_id, cur = None, None, []
        for line in fasta.read_text().splitlines():
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur)
                cur_hdr = line[1:]
                cur_id = cur_hdr.split("|")[0]
                cur = []
                order[fam].append((cur_hdr, cur_id))
            elif line.strip():
                cur.append(line.strip().upper())
        if cur_id:
            seqs[cur_id] = "".join(cur)
    return seqs, order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", nargs="+", default=CONTROLS, choices=CONTROLS)
    args = ap.parse_args()

    seqs, order = load_family_fastas()
    seqs_by_family = {fam: [seqs[i] for _, i in hdrs] for fam, hdrs in order.items()}
    print(f"Loaded {len(seqs)} CDS across {len(order)} families")
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    fam_usage_all = build_family_codon_usage(seqs_by_family) if "synonymous_recode" in args.controls else {}

    for control in args.controls:
        rng = random.Random(SEED)  # fresh deterministic stream per control
        out_dir = CONTROL_ROOT / control
        out_dir.mkdir(parents=True, exist_ok=True)
        n_total = 0
        for fam, hdrs in order.items():
            lines = []
            for hdr, og in hdrs:
                s = seqs[og]
                if control == "dinuc_shuffle":
                    c = dinuc_shuffle(s, rng)
                elif control == "kmer4_shuffle":
                    c = klet_shuffle(s, 4, rng)
                elif control == "kmer6_shuffle":
                    c = klet_shuffle(s, 6, rng)
                elif control == "codon_shuffle":
                    c = codon_shuffle(s, rng)
                elif control == "synonymous_recode":
                    c = synonymous_recode(s, fam_usage_all[fam], rng)
                else:  # gc_match
                    c = gc_match(s, rng)
                lines.append(f">{hdr}\n{c}")
                n_total += 1
            (out_dir / f"{fam}.fasta").write_text("\n".join(lines) + "\n")
        manifest.to_csv(out_dir / "manifest.csv", index=False)  # metadata unchanged
        print(f"  {control:18} -> {out_dir}  ({n_total} sequences)")
    print("Done.")


if __name__ == "__main__":
    main()
