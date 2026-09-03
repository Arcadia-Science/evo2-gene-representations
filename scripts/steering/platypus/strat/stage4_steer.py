"""Stage 4 (stratified panel) — does the block-held-out direction steer generation? GPU."""

from __future__ import annotations
import argparse
import gzip
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))
sys.path.insert(0, str(ROOT / "scripts" / "steering" / "platypus"))

import steer_lib as S  # noqa: E402
from alignment_metrics import read_fasta, score_generation  # noqa: E402
from diagnostic_sites import diagnostic_sites_in_continuation  # noqa: E402

PREFIX_BP = 90


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stratum_interleaved(pairs: pd.DataFrame) -> list[str]:
    """One gene per stratum, round-robin, so leading cells span the whole conservation range."""
    by: dict[int, list[str]] = {}
    for s, grp in pairs.groupby("stratum"):
        by[int(s)] = list(grp.gene)
    order: list[str] = []
    while any(by.values()):
        for s in sorted(by):
            if by[s]:
                order.append(by[s].pop(0))
    return order


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--representation", default="cds_mean")
    ap.add_argument(
        "--layers",
        nargs="+",
        default=["blocks.27"],
        help="one layer, or several to inject SIMULTANEOUSLY (band steering). Each layer "
        "gets its OWN v_-i at the stated alpha, per band_steer_sweep.py convention.",
    )
    ap.add_argument(
        "--no-split-alpha",
        action="store_true",
        help="do NOT divide alpha by the number of layers. Default is to divide, so a "
        "band run's summed alpha equals a single-layer run's alpha.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["unsteered", "add", "add_own", "random"],
        help="unsteered|add|add_own|random|cross_gene|cone_removed|gc_removed|"
        "replace|blend|blendnorm  (tier D is --per-gene-layers)",
    )
    ap.add_argument(
        "--per-gene-layers",
        type=Path,
        default=None,
        help="TIER D: CSV with columns gene,layer. Steers each gene at ITS OWN peak "
        "layer instead of --layers. Only run it if stage3_gates.py licensed tier D "
        "-- if nearly every gene peaks at the frozen layer this arm is a re-run of "
        "the frozen-layer condition at full price.",
    )
    ap.add_argument(
        "--cond-suffix",
        default="",
        help="appended to every condition name (not `unsteered`). Use it when the same "
        "operator runs on a different vector -- e.g. the second-half-CDS vector -- "
        "so the merged scores table stays unambiguous.",
    )
    ap.add_argument("--trial-target", type=int, default=400)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument(
        "--min-sites",
        type=int,
        default=20,
        help="diagnostic-site target the generation window is extended toward (up to "
        "--max-gen-bp). A target, not an inclusion cutoff -- that is --min-sites-keep.",
    )
    ap.add_argument(
        "--min-sites-keep",
        type=int,
        default=5,
        help="drop a gene only below this many diagnostic sites. A short CDS can still be steered, "
        "just scored on fewer sites: after the 90 bp prompt PCP4 has 84 bp of continuation and 9 "
        "diagnostic sites, COX17 54 bp and 12, against a panel median of 191. They are kept and "
        "n_diag is recorded in scoring_plan.csv, so anything downstream can weight or exclude by "
        "scoring power instead of losing the gene outright.",
    )
    ap.add_argument("--gen-bp", type=int, default=1000)
    ap.add_argument("--max-gen-bp", type=int, default=2500)
    ap.add_argument("--genes", nargs="*", default=None)
    ap.add_argument("--n-genes", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="logit truncation, applied BEFORE temperature (vortex sample.py), so "
        "temperature only re-weights within these k bytes. top_k=1 SHORT-CIRCUITS "
        "to argmax = greedy decoding, which makes temperature irrelevant and the "
        "output deterministic -- pair it with --min-samples 1 --max-samples 1. "
        "Do NOT pass --temperature 0: sample() divides by it.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    stage1 = args.run / "stage1"
    stage3 = args.run / f"stage3_{args.representation}"
    lis = [int(x.split(".")[1]) for x in args.layers]
    # one label for the whole intervention, so single-layer and band runs are distinguishable in the
    # merged scores table: "blocks.27" vs "blocks.25+26+27"
    layer_label = args.layers[0] if len(lis) == 1 else "blocks." + "+".join(str(i) for i in lis)
    out = args.out or (
        args.run / f"stage4_{args.representation}_{layer_label.replace('.', '').replace('+', '_')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(stage1 / "pairs.csv")
    prow = {r.gene: r for r in pairs.itertuples()}
    ch = read_fasta(stage1 / "cds_human.fasta")
    cp = read_fasta(stage1 / "cds_platypus.fasta")

    # TIER D: each gene is steered at its own peak layer, so the set of layers to load is the union
    # over genes rather than --layers. Everything downstream keys off `lis_of(gene)`, and
    # single-layer
    # and band runs take the unchanged global path.
    per_gene_layers: dict[str, list[int]] = {}
    if args.per_gene_layers:
        pgl = pd.read_csv(args.per_gene_layers)
        per_gene_layers = {str(r.gene): [int(r.layer)] for r in pgl.itertuples()}
        layer_label = "per_gene"
        out = args.out or (args.run / f"stage4_{args.representation}_pergene")
        out.mkdir(parents=True, exist_ok=True)

    def lis_of(gene: str) -> list[int]:
        return per_gene_layers.get(gene, lis)

    all_lis = sorted(set(lis) | {li for v in per_gene_layers.values() for li in v})

    d = np.load(stage3 / "loo_vectors.npz", allow_pickle=True)
    genes_all = [str(g) for g in d["genes"]]
    V = {li: d[f"v_loo_L{li}"] for li in all_lis}  # per-layer leave-one-out vectors
    MUH = {li: d[f"muhat_L{li}"] for li in all_lis}
    # The GC-removed arm requires the cached GC axis.
    GCAX = {li: d[f"gc_axis_L{li}"] for li in all_lis} if f"gc_axis_L{all_lis[0]}" in d else {}
    if "gc_removed" in args.arms and not GCAX:
        raise SystemExit(
            "gc_removed needs gc_axis_L* in loo_vectors.npz -- re-run stage3_select.py"
        )
    gidx = {g: i for i, g in enumerate(genes_all)}


    genes = args.genes or stratum_interleaved(pairs)
    genes = [g for g in genes if g in gidx and g in ch and g in cp]
    if args.n_genes:
        genes = genes[: args.n_genes]

    # ---- per-gene scoring window and sample count, decided before the model loads --------------
    # Prompt offsets align shifted human prompts and platypus diagnostic sites.
    # Missing offset columns denote the codon-zero panel.
    off_h = {r.gene: int(getattr(r, "prefix_offset_h_bp", 0) or 0) for r in pairs.itertuples()}
    off_p = {r.gene: int(getattr(r, "prefix_offset_p_bp", 0) or 0) for r in pairs.itertuples()}
    n_shift = sum(1 for g in genes if off_h.get(g, 0) or off_p.get(g, 0))
    if n_shift:
        log(
            f"shifted-window panel: {n_shift}/{len(genes)} genes have a non-zero prompt offset "
            f"(human median {int(np.median([off_h[g] for g in genes]))} bp)"
        )

    plan: dict[str, dict] = {}
    for g in genes:
        # in TARGET coordinates the forced region ends here, so it is the cut for both the
        # continuation and the diagnostic-site re-keying -- one quantity, used twice
        pref_t = off_p.get(g, 0) + PREFIX_BP
        cont = cp[g][pref_t:]
        diag_all = diagnostic_sites_in_continuation(ch[g], cp[g], pref_t)
        n_tokens = min(len(cont), args.gen_bp)
        n_diag = sum(1 for i in diag_all if i < n_tokens)
        # minimum-site rule: extend the window rather than drop a conserved gene
        while n_diag < args.min_sites and n_tokens < min(len(cont), args.max_gen_bp):
            n_tokens = min(len(cont), n_tokens + 250)
            n_diag = sum(1 for i in diag_all if i < n_tokens)
        ns = int(
            np.clip(
                math.ceil(args.trial_target / max(n_diag, 1)), args.min_samples, args.max_samples
            )
        )
        plan[g] = {
            "n_tokens": n_tokens,
            "n_diag": n_diag,
            "n_samples": ns,
            "off_h": off_h.get(g, 0),
            "off_p": off_p.get(g, 0),
            "diag": {i: b for i, b in diag_all.items() if i < n_tokens},
            "cont": cont[:n_tokens],
        }
        if len(ch[g][off_h.get(g, 0) : off_h.get(g, 0) + PREFIX_BP]) != PREFIX_BP:
            raise AssertionError(
                f"{g}: prompt shorter than {PREFIX_BP} bp at offset "
                f"{off_h.get(g, 0)} -- stage 1 should have gated this"
            )
    usable = [
        g for g in genes if plan[g]["n_diag"] >= args.min_sites_keep and plan[g]["n_tokens"] >= 30
    ]
    dropped = [g for g in genes if g not in usable]
    pd.DataFrame(
        [
            {
                "gene": g,
                **{k: v for k, v in plan[g].items() if k not in ("diag", "cont")},
                "stratum": int(prow[g].stratum),
                "usable": g in usable,
            }
            for g in genes
        ]
    ).to_csv(out / "scoring_plan.csv", index=False)
    thin = [g for g in usable if plan[g]["n_diag"] < args.min_sites]
    log(
        f"{len(usable)}/{len(genes)} genes usable; dropped {len(dropped)} for "
        f"<{args.min_sites_keep} sites: {dropped[:10]}"
    )
    if thin:
        log(
            f"  {len(thin)} kept with <{args.min_sites} diagnostic sites (noisier per-gene "
            f"scores): {[(g, plan[g]['n_diag']) for g in thin[:10]]}"
        )
    ns_arr = np.array([plan[g]["n_samples"] for g in usable])
    log(
        f"n_samples: mean {ns_arr.mean():.1f} min {ns_arr.min()} max {ns_arr.max()}  |  "
        f"cells = {len(usable)} x arms"
    )

    # Per-gene deltas at this layer, for the own-shift-matched dose alpha_i = q_loo_i / ||v_-i||.
    # Loaded once: `pooled_representations.npz` gene order is asserted against pairs.csv by stage 3.
    Dpool = np.load(args.run / "stage2" / "pooled_representations.npz", allow_pickle=True)[
        args.representation
    ]
    if Dpool.shape[0] != len(genes_all):
        raise AssertionError(f"stage2 has {Dpool.shape[0]} genes, stage3 has {len(genes_all)}")
    DALL = {li: (Dpool[:, 1, li, :] - Dpool[:, 0, li, :]).astype(np.float64) for li in all_lis}

    # Band runs divide alpha across layers so the summed dose matches a single-layer run. Note this
    # equalises the summed ALPHA, not the summed relative perturbation: ||v_-i|| grows steeply with
    # depth (~1.4 / 2.4 / 5.3 at L25/26/27), so a 1/3-split band delivers less total
    # residual-relative
    # perturbation than L27 alone. rel_norm per layer is logged below so the realised dose is on
    # record.
    split = 1.0 if (args.no_split_alpha or len(lis) == 1) else 1.0 / len(lis)
    if len(lis) > 1:
        log(
            f"band injection at {len(lis)} layers, alpha split factor {split:.4f}"
            f"{' (DISABLED)' if args.no_split_alpha else ''}"
        )

    model = S.load_model()

    scores_path = out / "stage4_scores.csv"
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and scores_path.exists():
        prev = pd.read_csv(scores_path)
        rows = prev.to_dict("records")
        done = set(zip(prev.gene, prev.condition, strict=False))
        log(f"--resume: {len(done)} cells already complete")
    gf = gzip.open(out / "generations.jsonl.gz", "at" if args.resume else "wt")
    flog = open(out / "failures.log", "a" if args.resume else "w")

    for gi, gene in enumerate(usable):
        p = plan[gene]
        prompt = ch[gene][off_h.get(gene, 0) : off_h.get(gene, 0) + PREFIX_BP]
        i = gidx[gene]
        # Per layer: this gene's own LOO vector, its norm, and the dose that reproduces its own
        # teacher-forced shift. In a band run every layer gets its OWN vector, acting simultaneously
        # Apply h += alpha * v_-i^(L) at each layer in the band.
        g_lis = lis_of(gene)
        g_split = 1.0 if (args.no_split_alpha or len(g_lis) == 1) else 1.0 / len(g_lis)
        v = {li: V[li][i].astype(np.float64) for li in g_lis}
        vn = {li: float(np.linalg.norm(v[li])) for li in g_lis}
        q_loo = {li: float(DALL[li][i] @ (v[li] / vn[li])) for li in g_lis}
        alpha_own = {li: q_loo[li] / vn[li] for li in g_lis}

        def scaled(
            vecs: dict[int, np.ndarray],
            al: dict[int, float],
            g_split: float = g_split,
            g_lis: list[int] = g_lis,
        ) -> dict[str, np.ndarray]:
            """steering_multi takes ALREADY-SCALED vectors, so alpha is folded in here. For a single
            layer this is bit-identical to S.steering(model, layer, vec, alpha).
            """
            return {f"blocks.{li}": vecs[li] * al[li] * g_split for li in g_lis}

        # (name, layer_vecs, op) -- op is None for the additive operator (vectors pre-scaled by
        # `scaled`), or (alpha, keep_h, preserve_norm) for the replace/blend operators, which apply
        # alpha inside the hook because they weight h as well as v.
        conditions: list[tuple[str, dict[str, np.ndarray], tuple | None]] = []
        if "unsteered" in args.arms:
            conditions.append(("unsteered", {}, None))
        for a in args.alphas:
            if "add" in args.arms:
                conditions.append((f"add_a{a}", scaled(v, {li: a for li in g_lis}), None))
        if "add_own" in args.arms:
            conditions.append(("add_own", scaled(v, alpha_own), None))
        if "random" in args.arms:
            # Reuse one random direction per gene and evaluate its null at every dose.
            rv = {li: S.norm_matched_random(v[li], seed=args.seed + i + 1000 * li) for li in g_lis}
            for a in args.alphas:
                conditions.append((f"random_a{a}", scaled(rv, {li: a for li in g_lis}), None))
        if "cross_gene" in args.arms:
            j = (i + 51) % len(genes_all)
            cv = {
                li: V[li][j].astype(np.float64) / np.linalg.norm(V[li][j]) * vn[li] for li in g_lis
            }
            conditions.append(("cross_gene_a1.0", scaled(cv, {li: 1.0 for li in g_lis}), None))
        if "cone_removed" in args.arms:
            cv = {}
            for li in g_lis:
                w = v[li] - (v[li] @ MUH[li]) * MUH[li]
                cv[li] = w / np.linalg.norm(w) * vn[li]
            conditions.append(
                ("add_cone_removed_a1.0", scaled(cv, {li: 1.0 for li in g_lis}), None)
            )
        if "gc_removed" in args.arms:
            # Same construction as cone_removed, against the GC/composition axis instead of the
            # anisotropy axis: project it out, then RESTORE the original norm so the dose is
            # unchanged and the only difference from `add` is direction.
            cv = {}
            for li in g_lis:
                w = v[li] - (v[li] @ GCAX[li]) * GCAX[li]
                cv[li] = w / np.linalg.norm(w) * vn[li]
            conditions.append(("add_gc_removed_a1.0", scaled(cv, {li: 1.0 for li in g_lis}), None))

        # --- operator variants: the residual stream is REPLACED or INTERPOLATED, not added to ---
        # Vectors are passed UNSCALED (alpha lives in the hook). h <- alpha*v for `replace`;
        # h <- (1-alpha)*h + alpha*v for `blend`; `blendnorm` additionally rescales back to ||h||,
        # which isolates the direction change from the ~2.8x norm collapse a bare replace also
        # causes.
        raw = {f"blocks.{li}": v[li] for li in g_lis}
        for a in args.alphas:
            if "replace" in args.arms:
                conditions.append((f"replace_a{a}", raw, (a, False, False)))
            if "blend" in args.arms:
                conditions.append((f"blend_a{a}", raw, (a, True, False)))
            if "blendnorm" in args.arms:
                conditions.append((f"blendnorm_a{a}", raw, (a, True, True)))

        log(
            f"[{gi + 1}/{len(usable)}] {gene} s{int(prow[gene].stratum)} "
            f"pid={prow[gene].perc_id_hp:.1f} n_diag={p['n_diag']} gen={p['n_tokens']}bp "
            f"ns={p['n_samples']} |v|={[round(vn[li], 2) for li in lis]} "
            f"a_own={[round(alpha_own[li], 2) for li in lis]}"
        )

        for cond, layer_vecs, op in conditions:
            cond = cond + args.cond_suffix if cond != "unsteered" else cond
            if (gene, cond) in done:
                continue
            try:
                ctx = (
                    S.steering_multi(model, layer_vecs)
                    if op is None
                    else S.overwriting(
                        model, layer_vecs, alpha=op[0], keep_h=op[1], preserve_norm=op[2]
                    )
                )
                # Sampling draw for THIS cell, keyed on the cell rather than on how far into the
                # loop we are. A single seed at the top of the run would look like a fix and not
                # be one: --resume skips a different set of cells each invocation, so the global
                # RNG stream diverges and the same cell gets a different draw. Keying on
                # (gene, condition) makes a cell reproducible however the run was assembled.
                torch.manual_seed(args.seed + zlib.crc32(f"{gene}:{cond}".encode()))
                with ctx:
                    o = model.generate(
                        prompt_seqs=[prompt] * p["n_samples"],
                        n_tokens=p["n_tokens"],
                        temperature=args.temperature,
                        top_k=args.top_k,
                        verbose=0,
                    )
                for k, cont in enumerate(o.sequences):
                    sc = score_generation(cont, p["cont"], p["diag"])
                    rows.append(
                        {
                            "gene": gene,
                            "stratum": int(prow[gene].stratum),
                            "perc_id_hp": float(prow[gene].perc_id_hp),
                            "block": int(prow[gene].block),
                            # no hook is registered for `unsteered`, so no layer was applied --
                            # recording one would imply these rows are layer-specific. They are
                            # not, which is why this arm can run before the layer is chosen.
                            "layer": "none" if cond == "unsteered" else layer_label,
                            "n_layers": 0 if cond == "unsteered" else len(lis),
                            "representation": args.representation,
                            "condition": cond,
                            "sample": k,
                            # decoding params travel WITH the row: a temperature ladder merges
                            # several runs into one table, and condition names alone would not
                            # say which draw came from which sampler setting
                            "temperature": args.temperature,
                            "top_k": args.top_k,
                            "n_diag_scorable": p["n_diag"],
                            "gen_bp": p["n_tokens"],
                            "n_samples": p["n_samples"],
                            "v_norm": json.dumps({str(li): round(vn[li], 4) for li in lis}),
                            "q_loo": json.dumps({str(li): round(q_loo[li], 4) for li in lis}),
                            "alpha_own": json.dumps(
                                {str(li): round(alpha_own[li], 4) for li in lis}
                            ),
                            **sc,
                        }
                    )
                    gf.write(
                        json.dumps({"gene": gene, "condition": cond, "sample": k, "seq": cont})
                        + "\n"
                    )
                pd.DataFrame(rows).to_csv(scores_path, index=False)
                gf.flush()
            except Exception as exc:  # noqa: BLE001
                flog.write(f"{gene}/{cond}: {type(exc).__name__}: {exc}\n")
                flog.flush()
                log(f"  FAILED {gene}/{cond}: {exc}")

    gf.close()
    flog.close()
    df = pd.DataFrame(rows)
    df.to_csv(scores_path, index=False)
    (out / "stage4_config.json").write_text(
        json.dumps(
            {
                "layers": args.layers,
                "layer_label": layer_label,
                "n_layers": len(lis),
                "injection": "per-layer v_-i applied simultaneously with steering_multi",
                "alpha_split_factor": split,
                "alpha_split_note": (
                    "alpha divided by n_layers so the summed alpha matches a single-layer "
                    "run; equalises summed alpha, not summed residual-relative dose"
                ),
                "representation": args.representation,
                "arms": args.arms,
                "alphas": args.alphas,
                "trial_target": args.trial_target,
                "gen_bp": args.gen_bp,
                "max_gen_bp": args.max_gen_bp,
                "min_sites": args.min_sites,
                "min_sites_keep": args.min_sites_keep,
                "n_genes": len(usable),
                "dropped": dropped,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "decoding": (
                    "greedy (top_k=1 short-circuits to argmax; temperature inert)"
                    if args.top_k == 1
                    else f"top-{args.top_k} truncation then temperature {args.temperature}"
                ),
                "seed": args.seed,
                # What the seed covers, so a reader cannot assume more than it delivers.
                "seeding": (
                    "torch.manual_seed(seed + crc32('<gene>:<condition>')) before each generate "
                    "call, so a cell's draw does not depend on which cells the invocation ran; "
                    "numpy draws (random directions) are seeded per gene and layer. CUDA kernel "
                    "non-determinism is NOT controlled -- torch.use_deterministic_algorithms is "
                    "not enabled -- so bitwise reproducibility is not claimed, only that the "
                    "sampling stream is fixed per cell."
                ),
            },
            indent=2,
        )
    )
    if not df.empty:
        log(
            "\nmean pct_private_correct by condition:\n"
            + df.groupby("condition").pct_private_correct.mean().round(2).to_string()
        )
    log(f"done -> {scores_path}")


if __name__ == "__main__":
    sys.exit(main())
