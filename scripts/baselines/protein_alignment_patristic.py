"""Per-family within-family ground truth from one protein alignment."""

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

HUMAN_CDS_JSON = Path("data/cache/cds_sequences.json")
MIN_MEMBERS = 4  # below this a within-family rank correlation is meaningless

_CODON = {  # standard genetic code; '*' = stop, 'X' = unknown/ambiguous
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


def translate(cds: str) -> str:
    """Frame-0 protein; stops/ambiguous → X (alignment treats them as wildcards)."""
    cds = cds.upper().replace("U", "T")
    aa = [_CODON.get(cds[i : i + 3], "X") for i in range(0, len(cds) - 2, 3)]
    return "".join("X" if c == "*" else c for c in aa)


def load_sequences() -> dict[str, str]:
    """id -> CDS for the human paralog panel."""
    return json.loads(HUMAN_CDS_JSON.read_text())


def align_members(ids: list[str], seqs: dict[str, str], workdir: Path) -> Path | None:
    """MAFFT-align translated members."""
    prot = workdir / "prot.fasta"
    with open(prot, "w") as fh:
        for i, g in enumerate(ids):
            fh.write(f">seq{i}\n{translate(seqs[g])}\n")
    aln = workdir / "aln.fasta"
    with open(aln, "w") as out:
        r = subprocess.run(
            ["mafft", "--auto", "--anysymbol", "--quiet", str(prot)],
            stdout=out,
            stderr=subprocess.DEVNULL,
        )
    if r.returncode != 0:
        return None
    return aln


def tree_patristic(aln: Path, ids: list[str], workdir: Path) -> np.ndarray | None:
    """FastTree on a MAFFT alignment → patristic distances in `ids` order (None on failure)."""
    import dendropy

    label2id = {f"seq{i}": g for i, g in enumerate(ids)}
    tree_nwk = workdir / "tree.nwk"
    with open(tree_nwk, "w") as out:
        r = subprocess.run(
            ["FastTree", "-quiet", "-nopr", str(aln)], stdout=out, stderr=subprocess.DEVNULL
        )
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
    """Return the cached patristic matrix reordered to `members`."""
    pat_npy = cache_dir / f"{fam}.npy"
    ids_json = cache_dir / f"{fam}.ids.json"
    if not (pat_npy.exists() and ids_json.exists()):
        return None, "mafft"
    cached_ids = json.loads(ids_json.read_text())
    if set(cached_ids) != set(members):
        return None, "mafft"
    pos = {g: i for i, g in enumerate(cached_ids)}
    order = [pos[g] for g in members]
    D_pat = np.load(pat_npy)[np.ix_(order, order)]
    return D_pat, "cache"


def save_cached_within(
    cache_dir: Path, fam: str, members: list[str], D_pat: np.ndarray | None
) -> None:
    if D_pat is not None:
        np.save(cache_dir / f"{fam}.npy", D_pat)
        (cache_dir / f"{fam}.ids.json").write_text(json.dumps(members))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    geo_files = list(run_dir.glob("*_geodesic_labeled.csv"))
    if not geo_files:
        sys.exit(f"No *_geodesic_labeled.csv in {run_dir}")
    df_geo = pd.read_csv(geo_files[0], index_col=0)
    geo = df_geo.values
    ids_all = df_geo.index.tolist()
    id_pos = {g: i for i, g in enumerate(ids_all)}

    meta = pd.read_csv(run_dir / "metadata.csv")
    fam_of = dict(zip(meta["gene"], meta["family"], strict=False))
    seqs = load_sequences()

    # Patristic matrices are layer-independent, so compute and cache each family once.
    cache_dir = Path("data/cache/human_patristic")
    cache_dir.mkdir(parents=True, exist_ok=True)

    families = sorted(set(meta["family"]))
    pat_rows = []
    print(f"{'family':<26} {'n':>4} {'ρ_patristic':>12}  src")
    with tempfile.TemporaryDirectory(dir="/opt/dlami/nvme/uv/tmp") as td:
        for fam in families:
            members = [g for g in ids_all if fam_of.get(g) == fam and g in seqs]
            if len(members) < MIN_MEMBERS:
                print(f"{fam:<26} {len(members):>4}  (too few members; skipped)")
                continue
            gi = [id_pos[g] for g in members]
            geo_sub = geo[np.ix_(gi, gi)]

            D_pat, src = load_cached_within(cache_dir, fam, members)
            if D_pat is None:
                wd = Path(td) / fam
                wd.mkdir(parents=True, exist_ok=True)
                res = align_members(members, seqs, wd)
                if res is None:
                    print(f"{fam:<26} {len(members):>4}  (alignment failed; skipped)")
                    continue
                D_pat = tree_patristic(res, members, wd)
                save_cached_within(cache_dir, fam, members, D_pat)
                src = "mafft"

            patr = score_within(geo_sub, D_pat)
            if patr is not None:
                rho, p, npairs = patr
                pat_rows.append(
                    {
                        "family": fam,
                        "n_members": len(members),
                        "n_pairs": npairs,
                        "spearman_geodesic_patristic": rho,
                        "p_patristic": p,
                    }
                )
            print(
                f"{fam:<26} {len(members):>4} {(patr[0] if patr else float('nan')):>+12.3f}  {src}"
            )

    pat_out = run_dir / "within_family_patristic.csv"
    pd.DataFrame(
        pat_rows,
        columns=["family", "n_members", "n_pairs", "spearman_geodesic_patristic", "p_patristic"],
    ).to_csv(pat_out, index=False)
    print(f"\nSaved {pat_out}")
    print("Done.")


if __name__ == "__main__":
    main()
