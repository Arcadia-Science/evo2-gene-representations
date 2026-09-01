"""Stage 5b — branch lengths on a FIXED species topology, and the rate predictors. CPU, parallel."""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from alignment_metrics import read_fasta  # noqa: E402

IQTREE = ROOT / "data" / "tools" / "iqtree2"
TRIMAL = ROOT / "data" / "tools" / "trimal"
SPECIES_TREE = ROOT / "data" / "mammalian_orthologs" / "tree" / "species_tree.nwk"
HUMAN, PLATYPUS = "homo_sapiens", "ornithorhynchus_anatinus"
MODEL = "LG+G4"  # fixed across genes on purpose: comparability between genes beats per-gene fit


def protein(cds: str) -> str:
    from Bio.Seq import Seq

    aa = str(Seq(cds[: len(cds) // 3 * 3]).translate())
    return aa[:-1] if aa.endswith("*") else aa


def pruned_topology(taxa: list[str], out: Path) -> int:
    """Species topology restricted to `taxa`, branch lengths stripped (topology only for -te)."""
    import dendropy

    t = dendropy.Tree.get(path=str(SPECIES_TREE), schema="newick", preserve_underscores=True)
    keep = [x for x in taxa]
    t.retain_taxa_with_labels(keep)
    for e in t.edges():
        e.length = None
    out.write_text(
        t.as_string(schema="newick", suppress_rooting=True, unquoted_underscores=True).strip()
        + "\n"
    )
    return len(t.leaf_nodes())


def tree_stats(tree_path: Path) -> dict:
    import dendropy

    t = dendropy.Tree.get(path=str(tree_path), schema="newick", preserve_underscores=True)
    labels = {lf.taxon.label for lf in t.leaf_node_iter()}
    lens = [e.length for e in t.edges() if e.length is not None]
    total = float(sum(lens))
    pdm = t.phylogenetic_distance_matrix()
    tax = {lf.taxon.label: lf.taxon for lf in t.leaf_node_iter()}
    pair = [pdm(a, b) for a in tax.values() for b in tax.values() if a is not b]
    internal = float(
        sum(e.length for e in t.edges() if e.length is not None and e.head_node.child_nodes())
    )
    term = {lf.taxon.label: float(lf.edge.length or 0.0) for lf in t.leaf_node_iter()}

    def dist(a: str, b: str) -> float | None:
        if a in tax and b in tax:
            return float(pdm(tax[a], tax[b]))
        return None

    mean_to_target = None
    if PLATYPUS in tax:
        d = [float(pdm(tax[PLATYPUS], o)) for o in tax.values() if o is not tax[PLATYPUS]]
        mean_to_target = float(sum(d) / len(d)) if d else None

    hb = term.get(HUMAN)
    pb = term.get(PLATYPUS)
    return {
        "n_taxa_tree": len(labels),
        # legacy five
        "tree_len": total,
        "diameter": float(max(pair)) if pair else None,
        "focal_target_dist": dist(HUMAN, PLATYPUS),
        "mean_target_dist": mean_to_target,
        "treeness": internal / total if total > 0 else None,
        # primary predictors
        "human_branch": hb,
        "platypus_branch": pb,
        "background_rate": (total - (hb or 0.0) - (pb or 0.0)) if total > 0 else None,
        "mean_terminal": float(sum(term.values()) / len(term)) if term else None,
    }


def one_gene(fa: Path, outdir: Path) -> dict:
    gene = fa.stem
    gd = outdir / gene
    gd.mkdir(parents=True, exist_ok=True)
    rec = {"gene": gene}
    try:
        cds = read_fasta(fa, header_field=None)
        prot = {sp: protein(s) for sp, s in cds.items()}
        prot = {sp: p for sp, p in prot.items() if len(p) >= 30 and "*" not in p[:-1]}
        if len(prot) < 4:
            return {**rec, "status": "skip", "reason": f"only {len(prot)} usable taxa"}
        if len({p for p in prot.values()}) < 4:
            return {**rec, "status": "skip", "reason": "fewer than 4 unique protein sequences"}

        praw = gd / "prot.fasta"
        with praw.open("w") as fh:
            for sp, p in prot.items():
                fh.write(f">{sp}\n{p}\n")
        aln_raw = gd / "aln_mafft.fasta"
        with aln_raw.open("w") as fh:
            r = subprocess.run(
                ["mafft", "--auto", "--quiet", str(praw)], stdout=fh, stderr=subprocess.PIPE
            )
        if r.returncode != 0:
            return {**rec, "status": "fail", "reason": f"mafft: {r.stderr.decode()[:200]}"}
        a0 = read_fasta(aln_raw, header_field=None)
        rec["aln_len_raw"] = len(next(iter(a0.values())))

        trimmed = gd / "aln.fasta"
        r = subprocess.run(
            [str(TRIMAL), "-in", str(aln_raw), "-out", str(trimmed), "-gappyout"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if r.returncode != 0 or not trimmed.exists():
            shutil.copy(aln_raw, trimmed)
            rec["trimal"] = "failed_used_raw"
        a1 = read_fasta(trimmed, header_field=None)
        rec["aln_len_trimmed"] = len(next(iter(a1.values())))
        if rec["aln_len_trimmed"] < 60:
            return {**rec, "status": "skip", "reason": "trimmed alignment < 60 columns"}

        topo = gd / "topology.nwk"
        n_leaf = pruned_topology(sorted(a1), topo)
        if n_leaf != len(a1):
            return {
                **rec,
                "status": "fail",
                "reason": f"topology has {n_leaf} leaves for {len(a1)} taxa",
            }

        pre = gd / "iq"
        r = subprocess.run(
            [
                str(IQTREE),
                "-s",
                str(trimmed),
                "-st",
                "AA",
                "-m",
                MODEL,
                "-te",
                str(topo),
                "-pre",
                str(pre),
                "-T",
                "1",
                "-quiet",
                "-redo",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        tf = Path(f"{pre}.treefile")
        if r.returncode != 0 or not tf.exists():
            return {**rec, "status": "fail", "reason": f"iqtree: {r.stderr.decode()[:200]}"}
        return {**rec, "status": "ok", **tree_stats(tf)}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "status": "fail", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    s5 = args.run / "stage5"
    outdir = args.outdir or (s5 / "trees")
    outdir.mkdir(parents=True, exist_ok=True)

    fastas = sorted((s5 / "seqs").glob("*.fasta"))
    print(f"[{time.strftime('%H:%M:%S')}] {len(fastas)} genes, {args.jobs} jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one_gene, fa, outdir): fa.stem for fa in fastas}
        for i, fut in enumerate(as_completed(futs)):
            rows.append(fut.result())
            if (i + 1) % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}]   {i + 1}/{len(fastas)}", flush=True)

    df = pd.DataFrame(rows).sort_values("gene")
    df.to_csv(s5 / "tree_stats.csv", index=False)
    ok = df[df.status == "ok"]
    print(f"\nok {len(ok)} / {len(df)}")
    if len(df) > len(ok):
        print(df[df.status != "ok"][["gene", "status", "reason"]].to_string(index=False))
    if len(ok):
        cols = [
            "n_taxa_tree",
            "tree_len",
            "diameter",
            "focal_target_dist",
            "mean_target_dist",
            "treeness",
            "human_branch",
            "platypus_branch",
            "background_rate",
        ]
        print("\n" + ok[cols].describe().loc[["min", "50%", "max"]].round(4).to_string())
    (s5 / "stage5b_config.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "topology": "fixed (-te), VertLife MamPhy pruned per gene, lengths stripped",
                "trim": "trimAl -gappyout",
                "aligner": "mafft --auto",
                "iqtree": "2.3.6",
                "n_ok": int(len(ok)),
                "n_total": int(len(df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
