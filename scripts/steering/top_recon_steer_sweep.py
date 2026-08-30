"""Sweep Evo2 steering layers for genes with the strongest target-species reconstructions."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))  # steer_lib, steer_ops, factored
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))  # evo2_embedding (via steer_lib.load_model)
sys.path.insert(0, str(ROOT / "scripts" / "mammalian_orthologs"))
import factored  # noqa: E402
import steer_lib as S  # noqa: E402
import steer_ops  # noqa: E402
from Bio.Align import PairwiseAligner  # noqa: E402
from Bio.Seq import Seq  # noqa: E402
from top_recon_diversity import mafft, qc_length_filter, read_fasta  # noqa: E402

IN = ROOT / "data" / "mammal_top_recon"
DIV = IN / "mammal_diversity.csv"
OUT = ROOT / "results" / "2026-07-24_evo2-steering-toprecon"
PROMPT_BP = 90
GEN = 500  # steer/generate the first GEN bp of the continuation
BASES = set("ACGT")
BLOCKS = [f"blocks.{i}" for i in range(32)]


def nt_aligner() -> PairwiseAligner:
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 1.0, 0.0
    al.open_gap_score, al.extend_gap_score = -2.0, -0.5
    return al


def protein(s: str) -> str:
    s = "".join(c if c in BASES else "N" for c in s)
    return str(Seq(s[: len(s) // 3 * 3]).translate())


def target_private_coords(cds_qc: dict[str, str], target: str) -> dict[int, str]:
    """{target-CDS 0-based index: base} for columns where the target's base is unique among the QC'd
    orthologs (autapomorphy). Uses the same MSA the diversity table is built from."""
    aln = mafft(cds_qc)
    species = list(aln)
    ti = species.index(target)
    cols = list(zip(*aln.values(), strict=False))
    priv: dict[int, str] = {}
    tcur = 0
    for col in cols:
        tch = col[ti]
        if tch != "-":
            if tch in BASES:
                # singleton base at this column?
                if sum(1 for c in col if c == tch) == 1:
                    priv[tcur] = tch
            tcur += 1
    return priv


def score(gen: str, target_cont: str, priv_in_window: dict[int, str]) -> dict:
    """gen vs target continuation. priv_in_window keys are indices INTO target_cont."""
    al = nt_aligner()
    a = al.align(gen, target_cont)[0] if gen and target_cont else None
    if a is None:
        return {}
    g_aln, t_aln = str(a[0]), str(a[1])
    # walk alignment, map target_cont index -> gen char, count private recovery + indels
    tcur, priv_hit, priv_tot, indel = 0, 0, 0, 0
    for gc, tc in zip(g_aln, t_aln, strict=False):
        if gc == "-" or tc == "-":
            indel += 1
        if tc != "-":
            if tcur in priv_in_window:
                priv_tot += 1
                if gc == priv_in_window[tcur]:
                    priv_hit += 1
            tcur += 1
    cols = max(len(g_aln), 1)
    matches = sum(1 for x, y in zip(g_aln, t_aln, strict=False) if x == y and x != "-")
    # protein identity to target
    pg, pt = protein(gen), protein(target_cont)
    pa = nt_aligner().align(pg, pt)[0] if pg and pt else None
    aa_id = (
        sum(1 for x, y in zip(str(pa[0]), str(pa[1]), strict=False) if x == y and x != "-")
        / max(len(str(pa[0])), 1)
        if pa is not None
        else np.nan
    )
    # premature stop (frame = continuation start; prompt_bp % 3 == 0)
    codons = [gen[i : i + 3] for i in range(0, len(gen) - 2, 3)]
    stopset = {"TAA", "TAG", "TGA"}
    n_stop = sum(1 for c in codons if c in stopset)  # all in-frame stop codons
    stop = any(c in stopset for c in codons[:-1])  # any PREMATURE (internal) stop
    # count indel EVENTS (maximal gap runs in either seq), separate from indel bp
    indel_events = 0
    in_gap = False
    for gc, tc in zip(g_aln, t_aln, strict=False):
        gap = gc == "-" or tc == "-"
        if gap and not in_gap:
            indel_events += 1
        in_gap = gap
    return {
        "pct_private_correct": (100 * priv_hit / priv_tot) if priv_tot else np.nan,
        "n_private_in_window": priv_tot,
        "aa_id_to_target": round(100 * aa_id, 2),
        "nt_id_to_target": round(100 * matches / cols, 2),
        "indel_bp": indel,
        "indel_events": indel_events,
        "n_stop_codons": n_stop,
        "premature_stop": bool(stop),
    }


def main() -> None:
    global OUT, GEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--gen-bp", type=int, default=1000, help="bp of continuation to generate/score")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min-species", type=int, default=5)
    ap.add_argument("--out", default=str(OUT), help="output dir")
    ap.add_argument(
        "--layers", type=int, nargs="+", default=None, help="block indices; default all 32"
    )
    args = ap.parse_args()
    OUT = Path(args.out)
    GEN = args.gen_bp
    OUT.mkdir(parents=True, exist_ok=True)

    div = pd.read_csv(DIV)
    targets = dict(zip(div["gene"], div["most_private_species"], strict=False))
    nsp = dict(zip(div["gene"], div["n_species"], strict=False))
    genes = args.genes or [g for g in div["gene"] if nsp.get(g, 0) >= args.min_species]
    blocks = [f"blocks.{i}" for i in args.layers] if args.layers else BLOCKS

    model = S.load_model()
    import gzip

    import torch

    scores_path = OUT / "steer_sweep_scores.csv"
    done = set()
    if scores_path.exists():
        prev = pd.read_csv(scores_path)
        done = set(zip(prev["gene"], prev["layer"], strict=False))
    rows: list[dict] = []
    flog = open(OUT / "sweep.log", "a")
    gf = gzip.open(OUT / "generations.jsonl.gz", "at")  # save every generation (<1MB gz)
    refs = open(OUT / "references.fasta", "a")  # human + target continuation per gene

    for gene in genes:
        fa = IN / f"{gene}.fasta"
        target = targets.get(gene)
        if not fa.exists() or not isinstance(target, str):
            continue
        cds_all = read_fasta(fa)
        cds, _drop = qc_length_filter(cds_all)
        if S.HUMAN not in cds or target not in cds or len(cds) < args.min_species:
            flog.write(f"{gene}: skip (human/target missing after QC or <{args.min_species} sp)\n")
            flog.flush()
            continue
        human, tgt_seq = cds[S.HUMAN], cds[target]
        prompt = human[:PROMPT_BP]
        target_cont = tgt_seq[PROMPT_BP : PROMPT_BP + GEN]
        n_tokens = len(target_cont)
        if n_tokens < 60:
            continue
        # target private coords (full CDS), then restrict to the generated window and re-index to
        # cont
        priv_full = target_private_coords(cds, target)
        priv_win = {
            p - PROMPT_BP: b for p, b in priv_full.items() if PROMPT_BP <= p < PROMPT_BP + n_tokens
        }
        print(
            f"\n=== {gene} -> {target} | prompt {PROMPT_BP}bp, gen {n_tokens}bp | "
            f"{len(priv_win)} private bp in window | {len(cds)} species ===",
            flush=True,
        )
        refs.write(
            f">{gene}|HUMAN_cont\n{human[PROMPT_BP : PROMPT_BP + n_tokens]}\n"
            f">{gene}|TARGET_{target}_cont\n{target_cont}\n"
            f">{gene}|HUMAN_prompt\n{prompt}\n"
        )
        refs.flush()

        # all-layer pooled embeddings, one forward per species
        embeds_by_layer: dict[str, dict[str, np.ndarray]] = {L: {} for L in blocks}
        for sp, seq in cds.items():
            ids = S._tokenize(model, seq, "cuda")
            with torch.no_grad():
                _lg, emb = model(ids, return_embeddings=True, layer_names=blocks)
            for L in blocks:
                embeds_by_layer[L][sp] = S.pool(emb[L][0].float().cpu().numpy(), "mean")
            del emb
        torch.cuda.empty_cache()

        # conditions: unsteered baseline + each block clamped
        conditions = [("none", None)] + [(L, L) for L in blocks]
        for cond_name, L in conditions:
            if (gene, cond_name) in done:
                continue
            try:
                if L is None:
                    gen_out = model.generate(
                        prompt_seqs=[prompt] * args.n_samples,
                        n_tokens=n_tokens,
                        temperature=args.temperature,
                        top_k=4,
                        verbose=0,
                    )
                else:
                    u_dir, c_pt = factored.species_direction(embeds_by_layer[L], target)
                    with steer_ops.clamp_direction(model, L, u_dir, c_pt, alpha=1.0):
                        gen_out = model.generate(
                            prompt_seqs=[prompt] * args.n_samples,
                            n_tokens=n_tokens,
                            temperature=args.temperature,
                            top_k=4,
                            verbose=0,
                        )
                for k, cont in enumerate(gen_out.sequences):
                    sc = score(cont, target_cont, priv_win)
                    rows.append(
                        {
                            "gene": gene,
                            "target": target,
                            "layer": cond_name,
                            "sample": k,
                            "n_species": len(cds),
                            **sc,
                        }
                    )
                    gf.write(
                        json.dumps(
                            {
                                "gene": gene,
                                "target": target,
                                "layer": cond_name,
                                "sample": k,
                                "seq": cont,
                            }
                        )
                        + "\n"
                    )
                gf.flush()
            except Exception as e:  # noqa: BLE001
                flog.write(f"{gene} {cond_name}: {type(e).__name__}: {e}\n")
                flog.flush()
                torch.cuda.empty_cache()
                continue
            # stream after each condition
            pd.DataFrame(rows).to_csv(scores_path, index=False)
            g = pd.DataFrame([r for r in rows if r["gene"] == gene and r["layer"] == cond_name])
            print(
                f"  {cond_name:9s} priv={g['pct_private_correct'].mean():5.1f}% "
                f"aa={g['aa_id_to_target'].mean():5.1f}% indel={g['indel_bp'].mean():5.1f} "
                f"stop={g['premature_stop'].mean():.2f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    flog.close()
    gf.close()
    refs.close()
    print(f"\nwrote {len(df)} rows -> {scores_path}")


if __name__ == "__main__":
    main()
