"""Stage 2 — embed each human/platypus CDS and pool two co-equal representations. GPU."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))
sys.path.insert(0, str(ROOT / "scripts" / "steering" / "platypus"))

import steer_lib as S  # noqa: E402
from alignment_metrics import read_fasta  # noqa: E402
from steer_lib import codon_blocks  # noqa: E402

SPECIES = ["human", "platypus"]  # column order of the saved tensor
MODES = ["cds_mean", "aligned_mean", "cds_second_half"]
NB = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def aligned_positions(h_cds: str, p_cds: str, context_bp: int) -> tuple[np.ndarray, np.ndarray]:
    """Return paired nucleotide positions with sufficient indel-free alignment context."""
    blocks_h, blocks_t, _ph, _pt = codon_blocks(h_cds, p_cds)
    need = int(np.ceil(context_bp / 3))
    hi: list[int] = []
    ti: list[int] = []
    for (hs, he), (ts, _te) in zip(blocks_h, blocks_t, strict=True):
        hs, he, ts = int(hs), int(he), int(ts)
        for k in range(need, he - hs):  # skip the first `need` codons of each block
            h_cod, t_cod = hs + k, ts + k
            if 3 * h_cod + 3 > len(h_cds) or 3 * t_cod + 3 > len(p_cds):
                continue
            hi.extend((3 * h_cod, 3 * h_cod + 1, 3 * h_cod + 2))
            ti.extend((3 * t_cod, 3 * t_cod + 1, 3 * t_cod + 2))
    return np.asarray(hi, np.int64), np.asarray(ti, np.int64)


class Pooler:
    """Forward hooks that ACCUMULATE each block's residual across one or more windows."""

    def __init__(self, model):
        self.inner = model.model
        self.sum: dict[int, object] = {}
        self.asum: dict[int, object] = {}
        self.hsum: dict[int, object] = {}
        self.n = 0
        self.n_aln = 0
        self.n_half = 0
        self.keep_from = 0
        self.abs_lo = 0
        self.half_from = 0
        self.idx = None
        self.handles: list = []

    def reset(self) -> None:
        self.sum, self.asum, self.hsum = {}, {}, {}
        self.n, self.n_aln, self.n_half = 0, 0, 0

    def set_window(
        self, keep_from: int, local_idx: np.ndarray, abs_lo: int = 0, half_from: int = 0
    ) -> None:
        """`abs_lo` is the ABSOLUTE CDS coordinate of the first retained position in this window."""
        import torch

        self.keep_from = int(keep_from)
        self.abs_lo = int(abs_lo)
        self.half_from = int(half_from)
        self.idx = (
            torch.as_tensor(local_idx, device="cuda", dtype=torch.long) if len(local_idx) else None
        )

    def __enter__(self):
        for li in range(NB):

            def hook(_m, _i, out, li=li):
                import torch

                h = (out[0] if isinstance(out, tuple) else out)[0].float()  # (L, H)
                kept = h[self.keep_from :]
                s = kept.sum(0)
                self.sum[li] = s if li not in self.sum else self.sum[li] + s
                if self.idx is not None:
                    keep = self.idx[self.idx < h.shape[0]]
                    a = h.index_select(0, keep).sum(0)
                    n_a = int(keep.numel())
                else:
                    a, n_a = torch.zeros_like(s), 0
                self.asum[li] = a if li not in self.asum else self.asum[li] + a
                # second half, in ABSOLUTE CDS coordinates (see set_window)
                first_half_in_window = max(0, min(kept.shape[0], self.half_from - self.abs_lo))
                hk = kept[first_half_in_window:]
                hs = hk.sum(0) if hk.shape[0] else torch.zeros_like(s)
                self.hsum[li] = hs if li not in self.hsum else self.hsum[li] + hs
                if li == 0:  # count positions once, not 32 times
                    self.n += int(kept.shape[0])
                    self.n_aln += n_a
                    self.n_half += int(hk.shape[0])

            self.handles.append(
                self.inner.get_submodule(f"blocks.{li}").register_forward_hook(hook)
            )
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def pooled(self) -> dict[str, np.ndarray]:
        cds = np.stack([(self.sum[li] / max(self.n, 1)).cpu().numpy() for li in range(NB)])
        if self.n_aln:
            aln = np.stack([(self.asum[li] / self.n_aln).cpu().numpy() for li in range(NB)])
        else:
            aln = np.full_like(cds, np.nan)
        half = (
            np.stack([(self.hsum[li] / self.n_half).cpu().numpy() for li in range(NB)])
            if self.n_half
            else np.full_like(cds, np.nan)
        )
        return {"cds_mean": cds, "aligned_mean": aln, "cds_second_half": half}


def embed_one(
    model, pool: Pooler, seq: str, aln_idx: np.ndarray, window: int, overlap: int
) -> tuple[dict[str, np.ndarray], int]:
    """Pool one sequence, windowing the forward when it exceeds `window` bp."""
    import torch

    pool.reset()
    half_from = len(seq) // 2  # absolute CDS coordinate where the second half begins
    start, n_win = 0, 0
    while start < len(seq):
        end = min(len(seq), start + window)
        keep_from = 0 if start == 0 else overlap
        lo = start + keep_from
        local = aln_idx[(aln_idx >= lo) & (aln_idx < end)] - start
        pool.set_window(keep_from, local, abs_lo=lo, half_from=half_from)
        ids = S._tokenize(model, seq[start:end], "cuda")
        with torch.no_grad():
            model(ids)
        n_win += 1
        if end == len(seq):
            break
        start = end - overlap
        del ids
        torch.cuda.empty_cache()
    return pool.pooled(), n_win


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context", type=int, default=30, help="bp of indel-free upstream alignment")
    ap.add_argument(
        "--window",
        type=int,
        default=6000,
        help="max bp per forward; longer CDS are tiled (a 10.7 kb CDS OOMs an A10G)",
    )
    ap.add_argument(
        "--overlap",
        type=int,
        default=1000,
        help="context discarded at the start of every window after the first",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    pairs = pd.read_csv(args.stage1 / "pairs.csv")
    ch = read_fasta(args.stage1 / "cds_human.fasta")
    cp = read_fasta(args.stage1 / "cds_platypus.fasta")
    genes = [g for g in pairs.gene if g in ch and g in cp]
    log(f"{len(genes)} genes, context={args.context} bp")

    # ---- alignment index sets (CPU, before the model is loaded) --------------------------------
    idx: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cov: list[dict] = []
    for g in genes:
        hi, ti = aligned_positions(ch[g], cp[g], args.context)
        idx[g] = (hi, ti)
        cov.append(
            {
                "gene": g,
                "n_aligned_bp": len(hi),
                "cds_len_human": len(ch[g]),
                "cds_len_platypus": len(cp[g]),
                "retained_frac": len(hi) / max(len(ch[g]), 1),
            }
        )
    covdf = pd.DataFrame(cov)
    covdf.to_csv(args.out / "aligned_coverage.csv", index=False)
    log(
        f"retained_frac: min {covdf.retained_frac.min():.3f} "
        f"med {covdf.retained_frac.median():.3f} max {covdf.retained_frac.max():.3f}"
    )
    drop = covdf[covdf.n_aligned_bp < 90].gene.tolist()
    if drop:
        log(f"WARNING {len(drop)} genes have <90 aligned bp: {drop[:8]}")

    model = S.load_model()
    acts = {m: np.zeros((len(genes), 2, NB, 4096), np.float32) for m in MODES}
    norms: list[dict] = []

    with Pooler(model) as pool:
        for gi, g in enumerate(genes):
            for si, sp in enumerate(SPECIES):
                seq = ch[g] if sp == "human" else cp[g]
                out, n_win = embed_one(model, pool, seq, idx[g][si], args.window, args.overlap)
                if pool.n != len(seq):
                    raise AssertionError(
                        f"{g}/{sp}: pooled {pool.n} positions for a {len(seq)} bp CDS "
                        "-- window tiling is not contiguous"
                    )
                # The second-half count must equal len(seq) - len(seq)//2 exactly. This is the check
                # that catches the tiling failure mode: a window-local threshold instead of an
                # absolute one gives the right answer on single-window genes and a silently wrong
                # one on the ~6% that tile, which is precisely where nobody would look.
                if pool.n_half != len(seq) - len(seq) // 2:
                    raise AssertionError(
                        f"{g}/{sp}: second-half pooled {pool.n_half} positions, expected "
                        f"{len(seq) - len(seq) // 2} for a {len(seq)} bp CDS ({n_win} windows)"
                    )
                for m in MODES:
                    acts[m][gi, si] = out[m]
                norms.append(
                    {
                        "gene": g,
                        "species": sp,
                        "n_pooled": pool.n,
                        "n_aligned": pool.n_aln,
                        "n_windows": n_win,
                        "cds_len": len(seq),
                        **{f"norm_b27_{m}": float(np.linalg.norm(out[m][27])) for m in MODES},
                    }
                )
                torch.cuda.empty_cache()
            if (gi + 1) % 20 == 0:
                log(f"  {gi + 1}/{len(genes)}")

    for m in MODES:
        finite = np.isfinite(acts[m]).all(axis=-1)
        if not finite.all():
            log(f"WARNING {m}: {int((~finite).sum())} non-finite (gene,species,layer) slices")
    # Use the shared pooled-representation cache name.
    np.savez_compressed(
        args.out / "pooled_representations.npz",
        genes=np.array(genes),
        species=np.array(SPECIES),
        **{m: acts[m] for m in MODES},
    )
    pd.DataFrame(norms).to_csv(args.out / "pooled_norms.csv", index=False)
    (args.out / "stage2_config.json").write_text(
        json.dumps(
            {
                "n_genes": len(genes),
                "context_bp": args.context,
                "modes": MODES,
                "n_blocks": NB,
                "window_bp": args.window,
                "overlap_bp": args.overlap,
                "n_genes_windowed": int(sum(1 for r in norms if r["n_windows"] > 1) / 2),
                "retained_frac": [
                    float(covdf.retained_frac.min()),
                    float(covdf.retained_frac.median()),
                    float(covdf.retained_frac.max()),
                ],
            },
            indent=2,
        )
    )
    log(f"saved {args.out / 'pooled_representations.npz'}")


if __name__ == "__main__":
    sys.exit(main())
