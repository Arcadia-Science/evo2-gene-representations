"""Core operations for Evo2 species steering."""

from __future__ import annotations
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evo2"))

CDS_DIR = ROOT / "data" / "mammalian_orthologs" / "seqs" / "cds"
HUMAN = "homo_sapiens"
BASES = "ACGT"
BASE_ID = {b: ord(b) for b in BASES}  # byte-level tokenizer: token id == ord(char)

# gene -> family fasta stem (both drivers only need these two; extend as needed)
GENE_FAMILY = {
    "HBE1": "globins",
    "HBZ": "globins",
    "HBQ1": "globins",
    "HBB": "globins",
    "CYP27B1": "cytochrome_p450",
    "HOXB1": "hox",
}


# --------------------------------------------------------------------------- data
def load_cds(gene: str) -> dict[str, str]:
    """{species: uppercase CDS} for one gene, scanning its family fasta (header
    group|species|gene_id).
    """
    fam = GENE_FAMILY.get(gene)
    fastas = [CDS_DIR / f"{fam}.fasta"] if fam else sorted(CDS_DIR.glob("*.fasta"))
    out: dict[str, str] = {}
    for fa in fastas:
        if not fa.exists():
            continue
        hid, seq = None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if hid and hid.split("|")[0] == gene:
                    out[hid.split("|")[1]] = "".join(seq).upper()
                hid, seq = line[1:], []
            elif line.strip():
                seq.append(line.strip())
        if hid and hid.split("|")[0] == gene:
            out[hid.split("|")[1]] = "".join(seq).upper()
    return out


# --------------------------------------------------------------------------- model / forward
def load_model():
    from evo2_embedding import load_model as _lm  # reuse the shared evo2_7b loader

    return _lm()


def _tokenize(model, seq: str, device: str) -> torch.Tensor:
    return torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to(device)


def _steer_hook(vec: torch.Tensor):
    """Forward hook that ADDS `vec` (H,) to a block's residual-stream output at every position."""

    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            h = out[0]
            return (h + vec.to(h.dtype).to(h.device),) + tuple(out[1:])
        return out + vec.to(out.dtype).to(out.device)

    return hook


@contextmanager
def steering(model, layer: str, steer_vec: np.ndarray | None, alpha: float, device: str = "cuda"):
    """Keep a steering hook on `layer` active for the whole block (e.g. across autoregressive
    generation). No-op if steer_vec is None or alpha == 0."""
    handle = None
    if steer_vec is not None and alpha != 0.0:
        vec = torch.as_tensor(
            np.asarray(steer_vec) * float(alpha), dtype=torch.float32, device=device
        )
        handle = model.model.get_submodule(layer).register_forward_hook(_steer_hook(vec))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def _field_hook(field: torch.Tensor, state: dict):
    """
    Forward hook that ADDS a POSITION-SPECIFIC row of `field` (Lfield,H) to the residual stream.
    """

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        L = h.shape[1]
        c = state["c"]
        seg = field[c : c + L]
        if seg.shape[0] < L:  # ran off the end of the field -> zero-pad the tail
            seg = torch.cat([seg, seg.new_zeros((L - seg.shape[0], field.shape[1]))], dim=0)
        state["c"] = c + L
        add = seg.to(h.dtype).to(h.device).unsqueeze(0)  # (1,L,H) broadcasts over the batch
        new_h = h + add
        return (new_h,) + tuple(out[1:]) if isinstance(out, tuple) else new_h

    return hook


@contextmanager
def steering_field(model, layer: str, field: np.ndarray | None, alpha: float, device: str = "cuda"):
    """Add a DIFFERENT (position-matched) steering vector at each position of `layer`, scaled by
    alpha.
    """
    handle, state = None, {"c": 0}
    if field is not None and alpha != 0.0:
        f = torch.as_tensor(np.asarray(field) * float(alpha), dtype=torch.float32, device=device)
        handle = model.model.get_submodule(layer).register_forward_hook(_field_hook(f, state))
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def forward(
    model,
    seq: str,
    layer: str,
    steer_vec: np.ndarray | None = None,
    alpha: float = 0.0,
    device: str = "cuda",
):
    """One forward over `seq`, optionally adding alpha*steer_vec at `layer`."""
    input_ids = _tokenize(model, seq, device)
    handle = None
    if steer_vec is not None and alpha != 0.0:
        vec = torch.as_tensor(
            np.asarray(steer_vec) * float(alpha), dtype=torch.float32, device=device
        )
        handle = model.model.get_submodule(layer).register_forward_hook(_steer_hook(vec))
    try:
        with torch.no_grad():
            logits, emb = model(input_ids, return_embeddings=True, layer_names=[layer])
    finally:
        if handle is not None:
            handle.remove()
    # Evo2's wrapper hands back StripedHyena.forward's return verbatim, which is a (logits, ...)
    # tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[0].float().cpu().numpy(), emb[layer][0].float().cpu().numpy()


@contextmanager
def steering_multi(model, layer_vecs: dict[str, np.ndarray], device: str = "cuda"):
    """
    Add a DIFFERENT (already-scaled) steering vector at EACH block in `layer_vecs`, all at once.
    """
    handles = []
    try:
        for layer, vec in layer_vecs.items():
            if vec is None:
                continue
            v = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=device)
            handles.append(model.model.get_submodule(layer).register_forward_hook(_steer_hook(v)))
        yield
    finally:
        for h in handles:
            h.remove()


def _overwrite_hook(vec: torch.Tensor, alpha: float, keep_h: bool, preserve_norm: bool):
    """Forward hook that REPLACES (rather than adds to) the residual stream at every position."""

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(h.dtype).to(h.device)
        new = ((1.0 - alpha) * h + alpha * v) if keep_h else (alpha * v).expand_as(h).clone()
        if preserve_norm:
            n0 = h.norm(dim=-1, keepdim=True)
            new = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-8) * n0
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    return hook


@contextmanager
def overwriting(
    model,
    layer_vecs: dict[str, np.ndarray],
    alpha: float,
    keep_h: bool,
    preserve_norm: bool,
    device: str = "cuda",
):
    """Replace-or-interpolate the residual stream at each block in `layer_vecs`. See
    _overwrite_hook.
    """
    handles = []
    try:
        for layer, vec in layer_vecs.items():
            if vec is None:
                continue
            v = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=device)
            handles.append(
                model.model.get_submodule(layer).register_forward_hook(
                    _overwrite_hook(v, float(alpha), keep_h, preserve_norm)
                )
            )
        yield
    finally:
        for h in handles:
            h.remove()


def forward_multi(
    model, seq: str, layer_vecs: dict[str, np.ndarray], device: str = "cuda"
) -> np.ndarray:
    """
    One forward over `seq` with per-layer steering vectors (already scaled) added at every block in
    `layer_vecs` simultaneously. Returns logits[Lseq,512] (float32, cpu).
    """
    input_ids = _tokenize(model, seq, device)
    any_layer = next(iter(layer_vecs))
    with torch.no_grad(), steering_multi(model, layer_vecs, device):
        logits, _ = model(input_ids, return_embeddings=True, layer_names=[any_layer])
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits[0].float().cpu().numpy()


def pool(hidden: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Pool per-position hidden (Lseq,H) -> (H,)."""
    if mode == "second_half":
        return hidden[hidden.shape[0] // 2 :].mean(axis=0)
    if mode == "last":
        return hidden[-1]
    if mode.startswith("last"):
        k = int(mode[4:])
        if k < 1:
            raise ValueError(f"pool mode {mode!r}: K must be >= 1")
        return hidden[-k:].mean(axis=0)
    return hidden.mean(axis=0)


# --------------------------------------------------------------------------- steering vectors
def build_pooled_embeddings(
    model, cds: dict[str, str], layer: str, pool_mode: str = "mean", device: str = "cuda"
) -> dict[str, np.ndarray]:
    """{species: pooled (H,) residual-stream vector at `layer`} for every ortholog of one gene."""
    return {
        sp: pool(forward(model, seq, layer, device=device)[1], pool_mode) for sp, seq in cds.items()
    }


def steering_vectors(embeds: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """From pooled embeddings build, per target species != human:
    v_h2t     = h(target) - h(human)
    v_centroid= h(target) - mean_over_all_orthologs h(species)
    """
    centroid = np.mean(np.stack(list(embeds.values())), axis=0)
    h = embeds[HUMAN]
    return {
        sp: {"h2t": embeds[sp] - h, "centroid": embeds[sp] - centroid}
        for sp in embeds
        if sp != HUMAN
    }


def norm_matched_random(vec: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(vec.shape).astype(np.float32)
    return r / (np.linalg.norm(r) + 1e-8) * np.linalg.norm(vec)


def shuffled(vec: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(vec)


# --------------------------------------------------------------------------- codon-aware diagnostic
def codon_blocks(human_cds: str, target_cds: str):
    """Codon-align two CDS sequences and return paired nucleotide blocks."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.Seq import Seq

    def protein(cds: str) -> str:
        aa = str(Seq(cds[: len(cds) // 3 * 3]).translate())
        return aa[:-1] if aa.endswith("*") else aa  # drop trailing stop (not in BLOSUM62)

    ph, pt = protein(human_cds), protein(target_cds)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10.0, -0.5
    aln = aligner.align(ph, pt)[0]
    blocks_h, blocks_t = aln.aligned  # matched-codon blocks, in aa coords
    return blocks_h, blocks_t, ph, pt


def codon_diagnostics(human_cds: str, target_cds: str):
    """Codon-align human vs target CDS (translate -> BLOSUM62 global protein align -> matched-codon
    blocks).
    """
    blocks_h, blocks_t, ph, pt = codon_blocks(human_cds, target_cds)

    diagnostic, conserved = [], []
    for (hs, he), (ts, _te) in zip(blocks_h, blocks_t, strict=False):
        for k in range(he - hs):
            hi, ti = hs + k, ts + k
            hcod, tcod = human_cds[3 * hi : 3 * hi + 3], target_cds[3 * ti : 3 * ti + 3]
            if len(hcod) < 3 or len(tcod) < 3:
                continue
            aa_change = ph[hi] != pt[ti]
            for c in range(3):
                hb, tb = hcod[c], tcod[c]
                if hb not in BASE_ID or tb not in BASE_ID:
                    continue
                h_idx = 3 * hi + c
                if hb != tb:
                    diagnostic.append(
                        {"h_idx": h_idx, "h_base": hb, "t_base": tb, "aa_change": bool(aa_change)}
                    )
                else:
                    conserved.append(h_idx)
    return diagnostic, conserved


# --------------------------------------------------------------------------- likelihood readout
def _logprobs_at(logits: np.ndarray, idx: int) -> np.ndarray:
    """log-softmax over the vocab of the distribution predicting sequence position `idx`
    (Evo2 convention: use logits[idx-1])."""
    row = logits[idx - 1]
    row = row - row.max()
    return row - np.log(np.exp(row).sum())


def base_logits(model, human_cds: str, layer: str, device: str = "cuda") -> np.ndarray:
    """
    Un-steered logits for the human CDS (compute ONCE per gene/layer; reuse across conditions).
    """
    return forward(model, human_cds, layer, None, 0.0, device)[0]


def readout(
    logits_s: np.ndarray, logits_b: np.ndarray, human_cds: str, diagnostic, conserved
) -> dict:
    """Self-controlled steering metrics from steered vs baseline logits at diagnostic/conserved
    sites.
    """
    bids = [BASE_ID[b] for b in BASES]
    dlo, rec, aa, syn, spec, tgt_gain, hum_gain, dis_gain = [], [], [], [], [], [], [], []
    for s in diagnostic:
        j = s["h_idx"]
        if j == 0:
            continue
        ls, lb = _logprobs_at(logits_s, j), _logprobs_at(logits_b, j)
        tid, hid = BASE_ID[s["t_base"]], BASE_ID[s["h_base"]]
        others = [i for i in bids if i not in (tid, hid)]
        dlo.append(float(ls[tid] - ls[hid]))
        (aa if s["aa_change"] else syn).append(dlo[-1])
        rec.append(int(np.argmax([ls[i] for i in bids]) == BASES.index(s["t_base"])))
        tg, dg = float(ls[tid] - lb[tid]), float(np.mean([ls[i] - lb[i] for i in others]))
        tgt_gain.append(tg)
        hum_gain.append(float(ls[hid] - lb[hid]))
        dis_gain.append(dg)
        spec.append(tg - dg)
    hb_ids = [BASE_ID.get(b, -1) for b in human_cds]
    cons = [
        float(_logprobs_at(logits_s, j)[hb_ids[j]]) for j in conserved if j > 0 and hb_ids[j] >= 0
    ]
    full = [
        float(_logprobs_at(logits_s, j)[hb_ids[j]])
        for j in range(1, len(human_cds))
        if hb_ids[j] >= 0
    ]

    def m(x):
        return float(np.mean(x)) if len(x) else np.nan

    return {
        "n_diag": len(dlo),
        "spec_effect": m(spec),
        "mean_dlogodds": m(dlo),
        "recovery": m(rec),
        "target_gain": m(tgt_gain),
        "human_drop": m(hum_gain),
        "distractor_gain": m(dis_gain),
        "spec_aa": m([spec[i] for i, s in enumerate(_valid(diagnostic)) if s["aa_change"]]),
        "spec_syn": m([spec[i] for i, s in enumerate(_valid(diagnostic)) if not s["aa_change"]]),
        "dlo_aa": m(aa),
        "dlo_syn": m(syn),
        "conserved_logp": m(cons),
        "seq_logp": m(full),
    }


def _valid(diagnostic):
    """Diagnostic sites that contribute to readout (h_idx>0), in order — for aligning per-site
    lists.
    """
    return [s for s in diagnostic if s["h_idx"] != 0]


# --------------------------------------------------------------------------- per-position field
def codon_position_map(human_cds: str, target_cds: str):
    """Every matched-codon nt position pair (h_idx, t_idx) between human and target CDS."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    from Bio.Seq import Seq

    def protein(cds: str) -> str:
        aa = str(Seq(cds[: len(cds) // 3 * 3]).translate())
        return aa[:-1] if aa.endswith("*") else aa

    ph, pt = protein(human_cds), protein(target_cds)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -10.0, -0.5
    aln = aligner.align(ph, pt)[0]
    blocks_h, blocks_t = aln.aligned
    pairs = []
    for (hs, he), (ts, _te) in zip(blocks_h, blocks_t, strict=False):
        for k in range(he - hs):
            hi, ti = hs + k, ts + k
            for c in range(3):
                pairs.append((3 * hi + c, 3 * ti + c))
    return pairs


def build_field(
    model, human_cds: str, target_cds: str, layer: str, device: str = "cuda"
) -> np.ndarray:
    """Build a position-matched target-minus-human steering field."""
    h_human = forward(model, human_cds, layer, device=device)[1]  # (Lh, H)
    h_target = forward(model, target_cds, layer, device=device)[1]  # (Lt, H)
    H = h_target.shape[1]
    field = np.zeros((len(target_cds), H), dtype=np.float32)
    for h_idx, t_idx in codon_position_map(human_cds, target_cds):
        if t_idx < h_target.shape[0] and h_idx < h_human.shape[0]:
            field[t_idx] = h_target[t_idx] - h_human[h_idx]
    return field
