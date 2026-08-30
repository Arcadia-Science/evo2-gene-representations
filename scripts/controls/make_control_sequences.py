"""Composition-matched control-sequence operators, shared by every panel's control builder."""

from __future__ import annotations
import random
from collections import defaultdict

# kmer4/kmer6 preserve exact k-mer spectra. `missense_subset` is nested within each recode.
# The paired-p3 arms share edited sites and rates and should be compared only with each other.
CONTROLS = [
    "dinuc_shuffle",
    "codon_shuffle",
    "synonymous_recode",
    "gc_match",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "missense_subset",
    "paired_p3_syn",
    "paired_p3_missense",
]
SEED = 1234

# Standard genetic code (frame-0 translation for amino-acid grouping).
_CODON = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


# ── control generators


def _random_arborescence(edges: dict, verts: set, last, rng: random.Random) -> dict:
    """A UNIFORMLY-random arborescence (spanning in-tree) oriented toward `last`, as
    {vertex -> its chosen "last edge" successor} for every vertex except `last`.
    """
    next_edge: dict = {}
    in_tree = {last}
    for v in verts:
        if v in in_tree:
            continue
        u = v
        while u not in in_tree:  # random walk, erasing loops as it goes
            succ = edges[u]
            next_edge[u] = succ[rng.randrange(len(succ))]
            u = next_edge[u]
        u = v
        while u not in in_tree:  # commit the loop-erased path
            in_tree.add(u)
            u = next_edge[u]
    return next_edge


def _euler_shuffle(symbols: list, rng: random.Random) -> list:
    """Randomize a sequence while preserving its k-mer counts."""
    n = len(symbols)
    if n < 4 or len(set(symbols)) < 2:
        return list(symbols)
    last = symbols[-1]
    verts = set(symbols)
    edges: dict = defaultdict(list)
    for a, b in zip(symbols[:-1], symbols[1:], strict=False):
        edges[a].append(b)
    # Every vertex other than `last` has an outgoing edge: a symbol occurring only at the
    # final position IS `last`. So the walk in _random_arborescence cannot dead-end.
    last_edge = _random_arborescence(edges, verts, last, rng)

    avail = {x: list(edges.get(x, [])) for x in verts}
    for x, e in last_edge.items():
        avail[x].remove(e)  # reserve the tree edge for last
    for x in verts:
        rng.shuffle(avail[x])
        if x in last_edge:
            avail[x].append(last_edge[x])  # terminal edge consumed last
    out = [symbols[0]]
    cur = symbols[0]
    idx = {x: 0 for x in verts}
    for _ in range(n - 1):
        nxt = avail[cur][idx[cur]]
        idx[cur] += 1
        out.append(nxt)
        cur = nxt
    return out


def dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """Altschul–Erikson dinucleotide-preserving shuffle: a uniformly-random sequence
    with EXACTLY the same first/last symbol and mono+dinucleotide frequencies.
    """
    s = seq.upper()
    if len(s) < 4 or len(set(s)) < 2:
        return s
    return "".join(_euler_shuffle(list(s), rng))


def klet_shuffle(seq: str, k: int, rng: random.Random) -> str:
    """Shuffle preserving the EXACT k-mer spectrum (k>=2): an Eulerian walk on the graph
    whose nodes are (k-1)-mers and edges are k-mers.
    """
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


_ALL_CODONS = sorted(_CODON)
_SENSE_CODONS = [c for c in _ALL_CODONS if _CODON[c] != "*"]


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b, strict=False))


# ── Matched synonymous/missense pair
# Both arms edit the same eligible position-3 sites at the same rate; only the protein outcome
# differs.
# Compare these arms with each other, not with `synonymous_recode`, which has a different edit rate.
def _p3_alternatives(cod: str) -> tuple[list[str], list[str]]:
    """(synonymous, missense) position-3 alternatives of `cod`, stops never included."""
    aa = _CODON.get(cod)
    if aa is None or aa == "*":
        return [], []
    syn, mis = [], []
    for b in "ACGT":
        if b == cod[2]:
            continue
        alt = cod[:2] + b
        a2 = _CODON.get(alt)
        if a2 is None or a2 == "*":
            continue
        (syn if a2 == aa else mis).append(alt)
    return syn, mis


def paired_p3(seq: str, fam_usage: dict, rng: random.Random, arm: str) -> str:
    """One arm of the matched pair. `arm` is 'synonymous' or 'missense'."""
    if arm not in ("synonymous", "missense"):
        raise ValueError(f"arm must be 'synonymous' or 'missense', got {arm!r}")
    s = seq.upper()
    ncod = len(s) // 3
    out = []
    for i in range(ncod):
        cod = s[i * 3 : i * 3 + 3]
        syn, mis = _p3_alternatives(cod)
        if not syn or not mis:  # 4-fold (no missense) or 1-fold (no synonym): skip
            out.append(cod)
            continue
        if arm == "missense":
            out.append(rng.choice(mis))
        else:
            usage = fam_usage.get(_CODON[cod])
            if usage is None:
                out.append(rng.choice(syn))
            else:
                codons, weights = usage
                w = [dict(zip(codons, weights, strict=False)).get(c, 0) for c in syn]
                out.append(rng.choices(syn, weights=w, k=1)[0] if sum(w) > 0 else rng.choice(syn))
    return "".join(out) + s[ncod * 3 :]


def missense_subset(seq: str, recoded: str, rng: random.Random) -> str:
    """The NONSYNONYMOUS counterpart of a `synonymous_recode`, changing ONLY bases the recode itself
    changed — a strict subset of the recode's edits, never a base the recode left alone.
    """
    s, r = seq.upper(), recoded.upper()
    ncod = min(len(s), len(r)) // 3
    out = []
    for i in range(ncod):
        src, rec = s[i * 3 : i * 3 + 3], r[i * 3 : i * 3 + 3]
        aa = _CODON.get(src)
        if aa is None or src == rec:
            out.append(src)  # non-ACGT, or the recode left this codon alone
            continue
        allowed = {j for j in range(3) if src[j] != rec[j]}
        cands = [
            c
            for c in _SENSE_CODONS
            if c != src
            and _CODON[c] != aa
            and all(c[j] == src[j] for j in range(3) if j not in allowed)
        ]
        if not cands:
            out.append(src)  # no missense reachable inside the recode's sites
            continue
        best = max(_hamming(c, src) for c in cands)  # use as much of `allowed` as possible
        out.append(rng.choice([c for c in cands if _hamming(c, src) == best]))
    return "".join(out) + s[ncod * 3 :]


def gc_match(seq: str, rng: random.Random) -> str:
    """Random length-matched sequence with the same G+C fraction (only)."""
    s = seq.upper()
    valid = sum(c in "ACGT" for c in s)
    gc = sum(c in "GC" for c in s)
    p = gc / valid if valid else 0.5
    return "".join(rng.choice("GC") if rng.random() < p else rng.choice("AT") for _ in s)
