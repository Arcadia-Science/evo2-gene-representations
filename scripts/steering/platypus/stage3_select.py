"""
Stage 3 — freeze the layer and build the leave-one-gene-out steering vectors. CPU only, no Evo2.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

from alignment_metrics import read_fasta  # noqa: E402

GTPASE = {"rab_gtpase", "ras_gtpases", "arf_gtpase", "rho_gtpase"}


def gc3(s: str) -> float:
    t = s[2::3]
    return (t.count("G") + t.count("C")) / max(len(t), 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    base = ROOT / "results" / "2026-07-28_evo2-platypus-paired"
    ap.add_argument("--stage1-dir", type=Path, default=base / "stage1")
    ap.add_argument("--sweep-dir", type=Path, default=base / "stage6_representation")
    ap.add_argument("--out-dir", type=Path, default=base / "stage3")
    ap.add_argument("--representation", default="cds_mean")
    ap.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=[24, 25, 26, 27],
        help="candidate layers to characterise; the primary is chosen from these",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.sweep_dir / "pooled_representations.npz", allow_pickle=True)
    genes = [str(g) for g in d["genes"]]
    a = d[args.representation]
    n = len(genes)

    with open(args.stage1_dir / "pairs.csv") as fh:
        pairs = list(csv.DictReader(fh))
    if [r["gene"] for r in pairs] != genes:
        raise AssertionError("gene order mismatch between the npz and pairs.csv")
    fams = np.array([r["family"] for r in pairs])
    supers = np.where(np.isin(fams, list(GTPASE)), "small_GTPase", fams)
    has_opossum = np.array([r["has_opossum"] == "True" for r in pairs])

    ch = read_fasta(args.stage1_dir / "cds_human.fasta")
    cp = read_fasta(args.stage1_dir / "cds_platypus.fasta")
    x = np.array([gc3(cp[g]) - gc3(ch[g]) for g in genes])
    xc = x - x.mean()

    per_gene, per_layer, vecs = [], [], {}
    for li in args.layers:
        H = a[:, 0, li, :].astype(np.float64)
        P = a[:, 1, li, :].astype(np.float64)
        D = P - H
        S = D.sum(0)
        v = S / n
        X = np.vstack([H, P])
        mu = X.mean(0)
        muh = mu / (np.linalg.norm(mu) + 1e-300)
        mean_pooled_norm = float(np.linalg.norm(X, axis=1).mean())
        beta = (xc @ (D - D.mean(0))) / (xc @ xc)
        bn = float(np.linalg.norm(beta))

        V = (S - D) / (n - 1)  # (n, H) leave-one-out vectors
        Vn = np.linalg.norm(V, axis=1)
        Vh = V / Vn[:, None]
        Dn = np.linalg.norm(D, axis=1)
        loo_cos = np.einsum("ij,ij->i", D, Vh) / Dn
        q_loo = np.einsum("ij,ij->i", D, Vh)
        frac_cone = (Vh @ muh) ** 2
        abs_gc = np.abs(Vh @ beta) / (bn + 1e-300)

        vecs[f"v_loo_L{li}"] = V.astype(np.float32)
        vecs[f"v_pooled_L{li}"] = v.astype(np.float32)
        vecs[f"muhat_L{li}"] = muh.astype(np.float32)
        # The GC/composition axis itself, not just each gene's |cos| to it. Stage 4's `gc_removed`
        # arm needs the axis to project out; without it the confound can be diagnosed but not
        # intervened on.
        vecs[f"gc_axis_L{li}"] = (beta / (bn + 1e-300)).astype(np.float32)

        for i in range(n):
            per_gene.append(
                {
                    "layer": li,
                    "gene": genes[i],
                    "family": fams[i],
                    "superfamily": supers[i],
                    "has_opossum": bool(has_opossum[i]),
                    "loo_cos": float(loo_cos[i]),
                    "q_loo": float(q_loo[i]),
                    "delta_norm": float(Dn[i]),
                    "v_loo_norm": float(Vn[i]),
                    "alpha_to_match_own_shift": float(q_loo[i] / Vn[i]),
                    "frac_on_cone": float(frac_cone[i]),
                    "abs_cos_gc": float(abs_gc[i]),
                }
            )

        per_layer.append(
            {
                "representation": args.representation,
                "layer": li,
                "v_norm": float(np.linalg.norm(v)),
                "rel_norm_v_over_pooled": float(np.linalg.norm(v)) / mean_pooled_norm,
                "mean_pooled_norm": mean_pooled_norm,
                "loo_cos_median": float(np.median(loo_cos)),
                "loo_cos_min": float(loo_cos.min()),
                "loo_frac_pos": float((loo_cos > 0).mean()),
                "q_loo_median": float(np.median(q_loo)),
                "q_loo_iqr": float(np.percentile(q_loo, 75) - np.percentile(q_loo, 25)),
                "q_loo_frac_pos": float((q_loo > 0).mean()),
                "alpha_median": float(np.median(q_loo / Vn)),
                "alpha_p25": float(np.percentile(q_loo / Vn, 25)),
                "alpha_p75": float(np.percentile(q_loo / Vn, 75)),
                "loo_vector_agreement": float(np.median(Vh @ Vh.T)),
                "frac_on_cone_median": float(np.median(frac_cone)),
                "frac_on_cone_max": float(frac_cone.max()),
                "abs_cos_gc_median": float(np.median(abs_gc)),
            }
        )

    pg = pd.DataFrame(per_gene)
    pl = pd.DataFrame(per_layer)
    pg.to_csv(args.out_dir / "loo_diagnostics.csv", index=False)
    pl.to_csv(args.out_dir / "layer_candidates.csv", index=False)
    np.savez_compressed(
        args.out_dir / "loo_vectors.npz",
        genes=np.array(genes),
        families=fams,
        representation=np.array([args.representation]),
        layers=np.array(args.layers),
        **vecs,
    )

    pd.set_option("display.width", 240)
    print(f"representation={args.representation}  N={n}  candidate layers={args.layers}\n")
    print(
        pl[
            [
                "layer",
                "v_norm",
                "rel_norm_v_over_pooled",
                "loo_cos_median",
                "loo_frac_pos",
                "q_loo_median",
                "q_loo_frac_pos",
                "alpha_median",
                "alpha_p25",
                "alpha_p75",
                "frac_on_cone_median",
                "abs_cos_gc_median",
                "loo_vector_agreement",
            ]
        ].to_string(index=False, float_format=lambda z: f"{z:.4f}")
    )

    print("\nPer-gene LOO cosine, worst 5 per layer (the genes to read with caution in Stage 4):")
    for li in args.layers:
        s = pg[pg.layer == li].nsmallest(5, "loo_cos")
        print(f"  L{li}: " + ", ".join(f"{r.gene}({r.loo_cos:+.2f})" for r in s.itertuples()))

    print("\nLOO cosine by family (median):")
    print(
        pg.pivot_table(
            index="family", columns="layer", values="loo_cos", aggfunc="median"
        ).to_string(float_format=lambda z: f"{z:+.3f}")
    )

    (args.out_dir / "stage3_config.json").write_text(
        json.dumps(
            {
                "representation": args.representation,
                "layers": args.layers,
                "n_genes": n,
                "design": "leave-one-gene-out: gene i is steered by v_-i = mean_{j!=i} D_j",
                "alpha_convention": "alpha=1 injects v_-i; alpha_median reproduces the median "
                "true shift along its own LOO direction",
                "n_with_opossum_wrong_species_control": int(has_opossum.sum()),
            },
            indent=2,
        )
    )
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
