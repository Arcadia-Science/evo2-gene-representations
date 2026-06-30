"""Per-family within-family ground truth from one protein alignment (Axis B).

This is the single source of truth for the two alignment-based within-family baselines,
emitted from ONE MAFFT alignment per family so the two pipelines (Evo2 cross-kingdom,
GPN-Star human-paralog) are scored against identically-defined metrics:

  * patristic distance  — the gold standard: translate members to protein, align (MAFFT),
    build a tree (FastTree), take patristic (tree-path, rate-corrected) distances. The
    realizable cross-kingdom form of the plan's "PANTHER branch lengths" — curated PANTHER
    trees do not cover archaeal mcrA / bacterial nitrogenase, so the tree is built from the
    actual embedded sequences.
  * sequence identity   — a divergence control: pairwise % identity over ungapped columns
    of the *same* MAFFT alignment (distance = 1 − identity). Derived from the alignment the
    tree is built on, so patristic and seq-identity are one consistent unit and the metric
    is identical across models. Protein identity (not nucleotide) is what makes this usable
    cross-kingdom, where pairwise nucleotide identity would be near-random.

The third within-family baseline, alignment-free k-mer composition, is computed in each
embedding pipeline (it needs no alignment); together the three form a divergence ladder
(tree → pairwise identity → composition).

Model-agnostic: Evo2 member id = org_gene, CDS in data/evo2_gene_families/<family>.fasta;
GPN-Star member id = gene symbol, CDS in data/cache/cds_sequences.json. Aligns to the
geodesic's member order by id, so it never positionally mislabels.

Writes within_family_patristic.csv (family, n, spearman_geodesic_patristic, p) and
within_family_seqid.csv (family, n, spearman_geodesic_seqid, p) to the run dir; the
visualization scripts pick both up as within-correlations series.

Usage:
    uv run python scripts/baselines/protein_alignment_patristic_seqid.py \
        --run-dir results/2026-06-19_evo2-gene-families --seq-source evo2
    uv run python scripts/baselines/protein_alignment_patristic_seqid.py \
        --run-dir results/2026-06-19_gpnstar-vertebrate --seq-source gpn
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/
from geodesic_utils import upper_triangle  # noqa: E402

EVO2_FASTA_DIR = Path("data/evo2_gene_families")
GPN_CDS_JSON = Path("data/cache/cds_sequences.json")
MIN_MEMBERS = 4  # below this a within-family rank correlation is meaningless

_CODON = {  # standard genetic code; '*' = stop, 'X' = unknown/ambiguous
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate(cds: str) -> str:
    """Frame-0 protein; stops/ambiguous → X (alignment treats them as wildcards)."""
    cds = cds.upper().replace("U", "T")
    aa = [_CODON.get(cds[i : i + 3], "X") for i in range(0, len(cds) - 2, 3)]
    return "".join("X" if c == "*" else c for c in aa)


def load_sequences(seq_source: str) -> dict[str, str]:
    """id -> CDS, for whichever panel this run came from."""
    if seq_source == "gpn":
        return json.loads(GPN_CDS_JSON.read_text())
    seqs: dict[str, str] = {}
    for fasta in EVO2_FASTA_DIR.glob("*.fasta"):
        cur_id, cur = None, []
        for line in fasta.read_text().splitlines():
            if line.startswith(">"):
                if cur_id:
                    seqs[cur_id] = "".join(cur)
                cur_id, cur = line[1:].split("|")[0], []
            elif line.strip():
                cur.append(line.strip().upper())
        if cur_id:
            seqs[cur_id] = "".join(cur)
    return seqs


def align_members(ids: list[str], seqs: dict[str, str], workdir: Path) -> tuple[Path, dict[str, str]] | None:
    """MAFFT-align translated members. Returns (aln_path, {seqN-label -> aligned protein})
    in `ids` order, or None on alignment failure. Index labels (seq0..) avoid Newick-special
    chars — member ids like "org:gene" would be truncated at ':' by FastTree."""
    prot = workdir / "prot.fasta"
    with open(prot, "w") as fh:
        for i, g in enumerate(ids):
            fh.write(f">seq{i}\n{translate(seqs[g])}\n")
    aln = workdir / "aln.fasta"
    with open(aln, "w") as out:
        r = subprocess.run(["mafft", "--auto", "--anysymbol", "--quiet", str(prot)],
                           stdout=out, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return None
    aligned: dict[str, str] = {}
    cur, chunk = None, []
    for line in aln.read_text().splitlines():
        if line.startswith(">"):
            if cur is not None:
                aligned[cur] = "".join(chunk)
            cur, chunk = line[1:].strip(), []
        elif line.strip():
            chunk.append(line.strip())
    if cur is not None:
        aligned[cur] = "".join(chunk)
    return aln, aligned


def seqid_distance_matrix(ids: list[str], aligned: dict[str, str]) -> np.ndarray:
    """1 − pairwise fractional identity over ungapped alignment columns, in `ids` order.

    Identity = (columns where both residues are non-gap and equal) / (columns where both
    are non-gap). NaN for any pair with no overlapping ungapped column (essentially never
    for real homologs). Vectorised: per-symbol indicator matmuls give all-pairs match and
    overlap counts at once."""
    n = len(ids)
    rows = [aligned.get(f"seq{i}", "") for i in range(n)]
    L = max((len(r) for r in rows), default=0)
    if L == 0:
        return np.full((n, n), np.nan)
    arr = np.array([list(r.ljust(L, "-")) for r in rows])  # (n, L) chars
    nongap = (arr != "-").astype(np.float64)
    overlap = nongap @ nongap.T  # (n, n): columns where both are non-gap
    matches = np.zeros((n, n), dtype=np.float64)
    for sym in set("".join(rows)) - {"-"}:
        ind = ((arr == sym) & (arr != "-")).astype(np.float64)
        matches += ind @ ind.T
    with np.errstate(invalid="ignore", divide="ignore"):
        ident = np.where(overlap > 0, matches / overlap, np.nan)
    D = 1.0 - ident
    np.fill_diagonal(D, 0.0)
    return D


def tree_patristic(aln: Path, ids: list[str], workdir: Path) -> np.ndarray | None:
    """FastTree on a MAFFT alignment → patristic distances in `ids` order (None on failure)."""
    import dendropy

    label2id = {f"seq{i}": g for i, g in enumerate(ids)}
    tree_nwk = workdir / "tree.nwk"
    with open(tree_nwk, "w") as out:
        r = subprocess.run(["FastTree", "-quiet", "-nopr", str(aln)],
                           stdout=out, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or tree_nwk.stat().st_size == 0:
        return None

    tree = dendropy.Tree.get(path=str(tree_nwk), schema="newick", preserve_underscores=True)
    pdm = tree.phylogenetic_distance_matrix()
    # tree taxa are the seqN labels; map each back to its member id's position.
    taxon = {label2id[t.label]: t for t in tree.taxon_namespace if t.label in label2id}
    n = len(ids)
    D = np.full((n, n), np.nan, dtype=float)
    for a in range(n):
        D[a, a] = 0.0
        for b in range(a + 1, n):
            ta, tb = taxon.get(ids[a]), taxon.get(ids[b])
            if ta is None or tb is None:
                continue
            d = pdm.patristic_distance(ta, tb)
            D[a, b] = D[b, a] = d
    return D


def score_within(geo_sub: np.ndarray, D: np.ndarray):
    """Spearman ρ of the within-family geodesic vs a distance matrix; (ρ, p, n_pairs) or None
    if too few finite pairs / degenerate (constant) distances."""
    if D is None:
        return None
    iu = np.triu_indices(len(geo_sub), k=1)
    d_u, geo_u = D[iu], geo_sub[iu]
    ok = ~np.isnan(d_u)
    if ok.sum() < 6 or np.ptp(d_u[ok]) == 0:
        return None
    rho, p = spearmanr(geo_u[ok], d_u[ok])
    return rho, p, int(ok.sum())


def load_cached_within(cache_dir: Path, fam: str, members: list[str]):
    """Reuse cached layer-independent matrices. Returns (patristic, seqid|None, src).

    Cache layout (shared with the layer-selection engine): <fam>.npy = patristic matrix,
    <fam>.ids.json = its member order, optional <fam>.seqid.npy. Reorders to `members`
    when the member SET matches; returns (None, None, "mafft") on miss/mismatch.
    """
    pat_npy = cache_dir / f"{fam}.npy"
    ids_json = cache_dir / f"{fam}.ids.json"
    if not (pat_npy.exists() and ids_json.exists()):
        return None, None, "mafft"
    cached_ids = json.loads(ids_json.read_text())
    if set(cached_ids) != set(members):
        return None, None, "mafft"
    pos = {g: i for i, g in enumerate(cached_ids)}
    order = [pos[g] for g in members]
    D_pat = np.load(pat_npy)[np.ix_(order, order)]
    seq_npy = cache_dir / f"{fam}.seqid.npy"
    D_seq = np.load(seq_npy)[np.ix_(order, order)] if seq_npy.exists() else None
    return D_pat, D_seq, "cache"


def save_cached_within(cache_dir: Path, fam: str, members: list[str],
                       D_pat: np.ndarray, D_seq: np.ndarray | None) -> None:
    if D_pat is not None:
        np.save(cache_dir / f"{fam}.npy", D_pat)
        (cache_dir / f"{fam}.ids.json").write_text(json.dumps(members))
    if D_seq is not None:
        np.save(cache_dir / f"{fam}.seqid.npy", D_seq)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seq-source", required=True, choices=["evo2", "gpn"])
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geo = df_geo.values
    ids_all = df_geo.index.tolist()
    id_pos = {g: i for i, g in enumerate(ids_all)}

    meta = pd.read_csv(run_dir / "metadata.csv") if (run_dir / "metadata.csv").exists() \
        else pd.read_csv(EVO2_FASTA_DIR / "embeddings" / "metadata.csv")
    id_col = "gene" if args.seq_source == "gpn" else "org_gene"
    fam_of = dict(zip(meta[id_col], meta["family"]))
    seqs = load_sequences(args.seq_source)

    # The per-family patristic & seq-id distance MATRICES are layer-INDEPENDENT
    # (they come from the sequences, not the embedding), so they are cached on disk —
    # the layer-selection engine already populates data/cache/<model>_patristic/. We
    # reuse them and skip the (slow) MAFFT/FastTree pass; a fresh family is computed
    # once and cached. Only the geodesic↔matrix correlation is recomputed per layer.
    cache_dir = Path(f"data/cache/{'evo2' if args.seq_source == 'evo2' else 'gpnstar'}_patristic")
    cache_dir.mkdir(parents=True, exist_ok=True)

    families = sorted(set(meta["family"]))
    pat_rows, seqid_rows = [], []
    print(f"{'family':<26} {'n':>4} {'ρ_patristic':>12} {'ρ_seqid':>10}  src")
    with tempfile.TemporaryDirectory(dir="/opt/dlami/nvme/uv/tmp") as td:
        for fam in families:
            members = [g for g in ids_all if fam_of.get(g) == fam and g in seqs]
            if len(members) < MIN_MEMBERS:
                print(f"{fam:<26} {len(members):>4}  (too few members; skipped)")
                continue
            gi = [id_pos[g] for g in members]
            geo_sub = geo[np.ix_(gi, gi)]

            D_pat, D_seq, src = load_cached_within(cache_dir, fam, members)
            # Recompute via one MAFFT alignment when EITHER matrix is missing — notably when
            # the patristic cache predates the seq-identity companion (D_pat hit, D_seq
            # absent), so seq-id is backfilled and cached rather than left NaN forever. Only
            # the missing matrix is rebuilt: a cached patristic still skips the FastTree pass.
            if D_pat is None or D_seq is None:
                wd = Path(td) / fam
                wd.mkdir(parents=True, exist_ok=True)
                res = align_members(members, seqs, wd)
                if res is None:
                    print(f"{fam:<26} {len(members):>4}  (alignment failed; skipped)")
                    continue
                aln, aligned = res
                if D_seq is None:
                    D_seq = seqid_distance_matrix(members, aligned)
                if D_pat is None:
                    D_pat = tree_patristic(aln, members, wd)
                save_cached_within(cache_dir, fam, members, D_pat, D_seq)
                src = "mafft"

            # Only the geodesic↔matrix correlation depends on the layer.
            seqid = score_within(geo_sub, D_seq) if D_seq is not None else None
            patr = score_within(geo_sub, D_pat)
            if seqid is not None:
                rho, p, npairs = seqid
                seqid_rows.append({"family": fam, "n_members": len(members), "n_pairs": npairs,
                                   "spearman_geodesic_seqid": rho, "p_seqid": p})
            if patr is not None:
                rho, p, npairs = patr
                pat_rows.append({"family": fam, "n_members": len(members), "n_pairs": npairs,
                                 "spearman_geodesic_patristic": rho, "p_patristic": p})
            print(f"{fam:<26} {len(members):>4} "
                  f"{(patr[0] if patr else float('nan')):>+12.3f} "
                  f"{(seqid[0] if seqid else float('nan')):>+10.3f}  {src}")

    pat_out = run_dir / "within_family_patristic.csv"
    seqid_out = run_dir / "within_family_seqid.csv"
    # Always write a header (even with 0 rows) so downstream pd.read_csv never hits
    # EmptyDataError — e.g. when every family reused a cached patristic matrix that
    # carried no seq-identity companion.
    pd.DataFrame(pat_rows, columns=["family", "n_members", "n_pairs",
                                    "spearman_geodesic_patristic", "p_patristic"]).to_csv(pat_out, index=False)
    pd.DataFrame(seqid_rows, columns=["family", "n_members", "n_pairs",
                                      "spearman_geodesic_seqid", "p_seqid"]).to_csv(seqid_out, index=False)
    print(f"\nSaved {pat_out}")
    print(f"Saved {seqid_out}")
    print("Done.")


if __name__ == "__main__":
    main()
