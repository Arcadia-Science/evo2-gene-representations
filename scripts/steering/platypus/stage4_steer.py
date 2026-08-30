"""Stage 4 (paired ~103-gene panel) — does the stage-3 direction actually steer generation? GPU."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))

import steer_lib as S  # noqa: E402
from steer_lib import codon_blocks  # noqa: E402
from top_recon_steer_sweep import score  # noqa: E402  (alignment walk + aa id + indel/stop scoring)

PREFIX_BP = 90
LAYER = "blocks.27"


def read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    hid, seq = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if hid:
                out[hid.split("|")[0]] = "".join(seq).upper()
            hid, seq = line[1:], []
        elif line.strip():
            seq.append(line.strip())
    if hid:
        out[hid.split("|")[0]] = "".join(seq).upper()
    return out


def diagnostic_sites_in_continuation(human_cds: str, plat_cds: str, prefix_bp: int) -> dict[int, str]:
    """{index into plat_cds[prefix_bp:] -> platypus base} for codon-aligned diagnostic sites."""
    blocks_h, blocks_t, _ph, _pt = codon_blocks(human_cds, plat_cds)
    bases = set("ACGT")
    out: dict[int, str] = {}
    for (hs, he), (ts, _te) in zip(blocks_h, blocks_t):
        for k in range(int(he - hs)):
            hi, ti = int(hs) + k, int(ts) + k
            hcod, tcod = human_cds[3 * hi:3 * hi + 3], plat_cds[3 * ti:3 * ti + 3]
            if len(hcod) < 3 or len(tcod) < 3:
                continue
            for c in range(3):
                hb, tb = hcod[c], tcod[c]
                if hb not in bases or tb not in bases or hb == tb:
                    continue
                t_idx = 3 * ti + c
                if t_idx >= prefix_bp:
                    out[t_idx - prefix_bp] = tb
    return out


def pick_genes(pairs: list[dict], loo: pd.DataFrame, n: int) -> list[str]:
    """ALL genes in priority order: one highest-power gene per family, then the worst-LOO genes, then the remainder by diagnostic-site count. Truncated to `n`."""
    power = {r["gene"]: int(r["n_diag_after_prefix"]) for r in pairs}
    fam = {r["gene"]: r["family"] for r in pairs}
    chosen: list[str] = []
    for f in sorted({r["family"] for r in pairs}):
        cands = sorted((g for g in power if fam[g] == f), key=lambda g: -power[g])
        if cands:
            chosen.append(cands[0])
    for g in loo.nsmallest(4, "loo_cos")["gene"]:
        if g not in chosen:
            chosen.append(g)
    for g in sorted(power, key=lambda g: -power[g]):
        if g not in chosen:
            chosen.append(g)
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    base = ROOT / "results" / "2026-07-28_evo2-platypus-paired"
    ap.add_argument("--stage1-dir", type=Path, default=base / "stage1")
    ap.add_argument("--stage3-dir", type=Path, default=base / "stage3")
    ap.add_argument("--out", type=Path, default=base / "stage4")
    ap.add_argument("--layer", default=LAYER)
    ap.add_argument("--n-genes", type=int, default=0, help="0 = all genes, in priority order")
    ap.add_argument("--resume", action="store_true",
                    help="skip (gene, condition) cells already present in stage4_scores.csv")
    ap.add_argument("--genes", nargs="*", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--gen-bp", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    li = int(args.layer.split(".")[1])
    with open(args.stage1_dir / "pairs.csv") as fh:
        pairs = list(csv.DictReader(fh))
    ch = read_fasta(args.stage1_dir / "cds_human.fasta")
    cp = read_fasta(args.stage1_dir / "cds_platypus.fasta")

    d = np.load(args.stage3_dir / "loo_vectors.npz", allow_pickle=True)
    genes_all = [str(g) for g in d["genes"]]
    V = d[f"v_loo_L{li}"]                       # (103, 4096) leave-one-out vectors
    muh = d[f"muhat_L{li}"]                     # the anisotropic axis, for the cone control
    gidx = {g: i for i, g in enumerate(genes_all)}

    loo = pd.read_csv(args.stage3_dir / "loo_diagnostics.csv")
    loo = loo[loo.layer == li]
    n_want = args.n_genes if args.n_genes > 0 else len(genes_all)
    genes = args.genes or pick_genes(pairs, loo, n_want)
    prow = {r["gene"]: r for r in pairs}
    print(f"layer={args.layer}  genes={len(genes)}  alphas={args.alphas}  "
          f"n_samples={args.n_samples}\n{', '.join(genes)}\n")

    # Build the layer-specific GC axis for the GC-removed arm.
    def gc3(s: str) -> float:
        t = s[2::3]
        return (t.count("G") + t.count("C")) / max(len(t), 1)
    x = np.array([gc3(cp[g]) - gc3(ch[g]) for g in genes_all])
    xc = x - x.mean()
    pooled = np.load(base / "stage6_representation" / "pooled_representations.npz",
                     allow_pickle=True)["cds_mean"]
    Dall = (pooled[:, 1, li, :] - pooled[:, 0, li, :]).astype(np.float64)
    beta = (xc @ (Dall - Dall.mean(0))) / (xc @ xc)
    bhat = beta / np.linalg.norm(beta)

    rng = np.random.default_rng(args.seed)
    model = S.load_model()

    scores_path = args.out / "stage4_scores.csv"
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and scores_path.exists():
        prev = pd.read_csv(scores_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.gene, prev.condition))
        print(f"--resume: {len(done)} (gene, condition) cells already complete\n")
    gf = gzip.open(args.out / "generations.jsonl.gz", "at" if args.resume else "wt")
    flog = open(args.out / "failures.log", "a" if args.resume else "w")

    for gi, gene in enumerate(genes):
        hcds, pcds = ch[gene], cp[gene]
        prompt = hcds[:PREFIX_BP]
        target_cont = pcds[PREFIX_BP:]
        diag = diagnostic_sites_in_continuation(hcds, pcds, PREFIX_BP)
        n_tokens = min(len(target_cont), args.gen_bp)
        # Score only the generated target region to avoid length-truncation penalties.
        target_cont = target_cont[:n_tokens]
        diag = {i: b for i, b in diag.items() if i < n_tokens}
        if not diag or n_tokens < 30:
            flog.write(f"{gene}: skipped (n_diag={len(diag)}, n_tokens={n_tokens})\n")
            continue

        i = gidx[gene]
        v = V[i].astype(np.float64)
        vn = float(np.linalg.norm(v))
        # controls, all norm-matched to v so only DIRECTION differs
        j = (i + 51) % len(genes_all)                       # a fixed distant gene, not a resample
        v_cross = V[j].astype(np.float64) / np.linalg.norm(V[j]) * vn
        v_rand = S.norm_matched_random(v, seed=args.seed + i)
        v_cone = v - (v @ muh) * muh                        # cone component projected out
        v_cone = v_cone / np.linalg.norm(v_cone) * vn
        v_gc = v - (v @ bhat) * bhat                        # GC component projected out
        v_gc = v_gc / np.linalg.norm(v_gc) * vn

        conditions: list[tuple[str, np.ndarray | None, float]] = [("unsteered", None, 0.0)]
        for a in args.alphas:
            conditions.append((f"add_a{a}", v, a))
        primary = 1.0 if 1.0 in args.alphas else args.alphas[-1]
        conditions += [
            (f"cross_gene_a{primary}", v_cross, primary),
            (f"random_a{primary}", v_rand, primary),
            (f"add_cone_removed_a{primary}", v_cone, primary),
            (f"add_gc_removed_a{primary}", v_gc, primary),
        ]

        print(f"[{gi + 1}/{len(genes)}] {gene} ({prow[gene]['family']}) "
              f"n_diag_scorable={len(diag)} gen={n_tokens}bp ||v||={vn:.2f}", flush=True)

        for cond, vec, alpha in conditions:
            if (gene, cond) in done:
                continue
            try:
                with S.steering(model, args.layer, vec, alpha):
                    out = model.generate(prompt_seqs=[prompt] * args.n_samples, n_tokens=n_tokens,
                                         temperature=args.temperature, top_k=4, verbose=0)
                for k, cont in enumerate(out.sequences):
                    sc = score(cont, target_cont, diag)
                    rows.append({"gene": gene, "family": prow[gene]["family"], "layer": args.layer,
                                 "condition": cond, "alpha": alpha, "sample": k,
                                 "n_diag_scorable": len(diag), "gen_bp": n_tokens, **sc})
                    gf.write(json.dumps({"gene": gene, "condition": cond, "sample": k,
                                         "seq": cont}) + "\n")
                gf.flush()
                g = pd.DataFrame([r for r in rows if r["gene"] == gene and r["condition"] == cond])
                print(f"    {cond:24s} diag={g['pct_private_correct'].mean():5.1f}%  "
                      f"aa={g['aa_id_to_target'].mean():5.1f}%  "
                      f"indel={g['indel_bp'].mean():6.1f}  stop={g['premature_stop'].mean():.2f}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                flog.write(f"{gene} {cond}: {type(e).__name__}: {e}\n")
                flog.flush()
                torch.cuda.empty_cache()
                continue
            pd.DataFrame(rows).to_csv(scores_path, index=False)
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(scores_path, index=False)
    gf.close()
    flog.close()

    # Paired across-gene summary; gene is the unit of analysis.
    df = pd.DataFrame(rows)
    if not df.empty:
        per = df.groupby(["gene", "condition"])[["pct_private_correct", "aa_id_to_target"]].mean()
        base_ = per.xs("unsteered", level="condition")
        out_rows = []
        for cond in df.condition.unique():
            if cond == "unsteered":
                continue
            c = per.xs(cond, level="condition")
            common = base_.index.intersection(c.index)
            for metric in ("pct_private_correct", "aa_id_to_target"):
                delta = (c.loc[common, metric] - base_.loc[common, metric]).dropna()
                if delta.empty:
                    continue
                try:
                    from scipy.stats import wilcoxon
                    p = float(wilcoxon(delta).pvalue) if len(delta) > 5 else np.nan
                except Exception:  # noqa: BLE001
                    p = np.nan
                out_rows.append({"condition": cond, "metric": metric, "n_genes": len(delta),
                                 "mean_delta": float(delta.mean()),
                                 "median_delta": float(delta.median()),
                                 "n_improved": int((delta > 0).sum()), "wilcoxon_p": p})
        summ = pd.DataFrame(out_rows)
        summ.to_csv(args.out / "stage4_paired_stats.csv", index=False)
        pd.set_option("display.width", 200)
        print("\nPaired across-gene deltas vs unsteered (gene = unit of analysis):")
        print(summ.to_string(index=False, float_format=lambda z: f"{z:.4f}"))

    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
