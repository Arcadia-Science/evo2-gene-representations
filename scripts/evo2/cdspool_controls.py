"""Tier-2 composition controls for condition C (CDS-masked-transcript, human panel)."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))
from control_tables import write_control_table  # noqa: E402
from embed_and_geodesic_paralog import (  # noqa: E402
    CDS_ONLY_CONTROLS,
    EMBED_DIM,
    N_BLOCKS,
    apply_control,
)
from embed_cds_masked_transcript import CDS_POS, GENOMIC_CACHE, embed_cds_masked  # noqa: E402
from geodesic_utils import (  # noqa: E402
    between_preservation_rho,
    compute_centroid_geodesic,
    compute_geodesic,
    find_min_connected_k,
    within_preservation_rho,
)

NAT_CACHE = ROOT / "data" / "cache" / "evo2_human_cdspool_transcript"
CDS_SEQ = ROOT / "data" / "cache" / "cds_sequences.json"
RUN_ROOT = ROOT / "results" / "2026-07-20_evo2-human-cdspool-transcript"
CONTROLS = [
    "gc_match",
    "dinuc_shuffle",
    "kmer4_shuffle",
    "kmer6_shuffle",
    "codon_shuffle",
    "synonymous_recode",
]


def coding_str(span: str, pos: list[int]) -> str:
    return "".join(span[i] for i in pos)


def build_control_seqs(control, genes, fams, genomic, cds_pos, cds_full):
    """gene -> transcript span with its CODING positions replaced by the shuffled coding content."""
    # coding content the readout actually pools (== full CDS for un-clipped genes)
    coding = {g: coding_str(genomic[g], cds_pos[g]) for g in genes}
    frame_ok = {
        g: (control not in CDS_ONLY_CONTROLS) or (coding[g] == cds_full.get(g, "")) for g in genes
    }
    shufflable = [g for g in genes if frame_ok[g]]
    fam_of = dict(zip(genes, fams, strict=False))
    shuffled_coding = apply_control(
        {g: coding[g] for g in shufflable}, control, fam_of={g: fam_of[g] for g in shufflable}
    )
    out = {}
    for g in genes:
        sc = shuffled_coding.get(g, coding[g])  # clipped-frame genes keep natural coding (no-op)
        span = list(genomic[g])
        for i, idx in enumerate(cds_pos[g]):
            span[idx] = sc[i]
        out[g] = "".join(span)
    return out


def embed_all_controls(controls):
    from evo2_embedding import load_model  # lazy import (GPU)

    genomic = json.loads(GENOMIC_CACHE.read_text())
    cds_pos = json.loads(CDS_POS.read_text())
    cds_full = json.loads(CDS_SEQ.read_text())
    meta = pd.read_csv(NAT_CACHE / "metadata.csv")
    genes, fams = meta["gene"].astype(str).tolist(), meta["family"].tolist()

    model, device = load_model(), "cuda"
    for control in controls:
        cache = ROOT / "data" / "cache" / f"evo2_human_cdspool_transcript_{control}"
        cache.mkdir(parents=True, exist_ok=True)
        stack_path = cache / "layer_stack.npy"
        if (
            stack_path.exists()
            and (cache / "config.json").exists()
            and json.loads((cache / "config.json").read_text()).get("genes") == genes
        ):
            print(f"[{control}] cached, skip", flush=True)
            continue
        print(f"[{control}] building shuffled-coding transcript inputs...", flush=True)
        seqs = build_control_seqs(control, genes, fams, genomic, cds_pos, cds_full)
        stack = np.zeros((N_BLOCKS, len(genes), EMBED_DIM), dtype=np.float32)
        for i, g in enumerate(tqdm(genes, desc=f"{control}")):
            stack[:, i, :] = embed_cds_masked(seqs[g], cds_pos[g], model, device)
            if (i + 1) % 50 == 0:
                tmp = stack_path.with_suffix(".tmp.npy")
                np.save(tmp, stack)
                tmp.replace(stack_path)
        np.save(stack_path, stack)
        pd.DataFrame({"gene": genes, "family": fams}).to_csv(cache / "metadata.csv", index=False)
        (cache / "config.json").write_text(json.dumps({"control": control, "genes": genes}))
        print(f"[{control}] saved {stack.shape}", flush=True)


def score(controls, layers):
    nat = np.load(NAT_CACHE / "layer_stack.npy")
    meta = pd.read_csv(NAT_CACHE / "metadata.csv")
    fams_arr = meta["family"].to_numpy()
    fam_order = sorted(set(meta["family"]))
    ctrl_stacks = {}
    for c in controls:
        p = ROOT / "data" / "cache" / f"evo2_human_cdspool_transcript_{c}" / "layer_stack.npy"
        if p.exists():
            ctrl_stacks[c] = np.load(p)
        else:
            print(f"  {c}: no stack, skip", flush=True)
    for L in layers:
        _, Wn = find_min_connected_k(nat[L], k_min=3)
        nat_geo = compute_geodesic(Wn)
        nat_cen = compute_centroid_geodesic(nat[L], fams_arr, fam_order)
        within_rows, between_rows = [], [{"condition": "natural", "rho_vs_natural_centroid": 1.0}]
        for c, cs in ctrl_stacks.items():
            _, Wc = find_min_connected_k(cs[L], k_min=3)
            cgeo = compute_geodesic(Wc)
            per_fam, _ = within_preservation_rho(nat_geo, cgeo, fams_arr, fam_order)
            for fam, n, rho in per_fam:
                within_rows.append(
                    {
                        "condition": c,
                        "family": fam,
                        "n_members": n,
                        "rho_geodesic_patristic": np.nan,
                        "rho_geodesic_vs_natural": rho,
                    }
                )
            ccen = compute_centroid_geodesic(cs[L], fams_arr, fam_order)
            between_rows.append(
                {"condition": c, "rho_vs_natural_centroid": between_preservation_rho(nat_cen, ccen)}
            )
        cdir = RUN_ROOT / f"blocks{L}" / "controls"
        cdir.mkdir(parents=True, exist_ok=True)
        write_control_table(
            cdir / "control_within_scores.csv",
            pd.DataFrame(within_rows),
            generator="cdspool_controls",
        )
        write_control_table(
            cdir / "control_between_scores.csv",
            pd.DataFrame(between_rows),
            generator="cdspool_controls",
        )
    print(f"wrote controls/ scores to {len(layers)} layers -> {RUN_ROOT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", nargs="+", default=CONTROLS)
    ap.add_argument("--layers", nargs="*", type=int, default=list(range(N_BLOCKS)))
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    if not args.score_only:
        embed_all_controls(args.controls)
    score(args.controls, args.layers)


if __name__ == "__main__":
    main()
