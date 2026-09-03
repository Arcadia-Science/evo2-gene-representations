"""Stage 1 — walk the frozen block order and build the paired human/platypus dataset. CPU only."""

from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "steering"))
sys.path.insert(0, str(ROOT / "scripts" / "steering" / "platypus"))
sys.path.insert(0, str(ROOT / "scripts"))

import paths  # noqa: E402
from dataset import PREFIX_BP, check_cds, qc_pair  # noqa: E402
from steer_lib import codon_blocks  # noqa: E402


def qc_pair_window(
    human_cds: str, platypus_cds: str, max_start_codon: int
) -> tuple[str | None, dict]:
    """The design's SHIFTED-WINDOW sensitivity gate (§3 Stage 1)."""
    if max_start_codon <= 0:
        reason, meta = qc_pair(human_cds, platypus_cds)
        meta.setdefault("prefix_codon_human", 0)
        meta.setdefault("prefix_codon_platypus", 0)
        return reason, meta

    meta: dict = {}
    for cds, tag in ((human_cds, "human"), (platypus_cds, "platypus")):
        reason = check_cds(cds, tag)
        if reason:
            return reason, meta

    n_codons = PREFIX_BP // 3
    blocks_h, blocks_t, ph, pt = codon_blocks(human_cds, platypus_cds)
    if len(blocks_h) == 0:
        return "no_codon_alignment", meta

    matched = sum(int(e - s) for s, e in blocks_h)
    n_same = 0
    for (bh_s, bh_e), (bt_s, _bt_e) in zip(blocks_h, blocks_t, strict=True):
        for k in range(int(bh_e - bh_s)):
            n_same += ph[int(bh_s) + k] == pt[int(bt_s) + k]
    meta.update(
        aln_first_block_human=f"{int(blocks_h[0][0])}-{int(blocks_h[0][1])}",
        aln_first_block_platypus=f"{int(blocks_t[0][0])}-{int(blocks_t[0][1])}",
        aln_n_matched_codons=matched,
        aln_n_blocks=len(blocks_h),
        aln_indel_frac=round(1.0 - matched / max(len(ph), len(pt)), 6),
        aa_identity=round(n_same / max(matched, 1), 6),
    )

    # earliest block that both starts early enough and is long enough to hold the whole prompt
    for (hs, he), (ts, te) in zip(blocks_h, blocks_t, strict=True):
        hs, he, ts, te = int(hs), int(he), int(ts), int(te)
        if hs > max_start_codon:
            break  # blocks are in ascending order, so no later block can start earlier
        if he - hs < n_codons or te - ts < n_codons:
            continue
        if "*" in ph[hs : hs + n_codons] or "*" in pt[ts : ts + n_codons]:
            continue  # an in-frame stop in the prompt leaves the continuation's frame ambiguous
        meta.update(prefix_codon_human=hs, prefix_codon_platypus=ts)
        return None, meta

    first_ok = next((int(hs) for (hs, he) in blocks_h if int(he - hs) >= n_codons), None)
    if first_ok is None:
        return f"no_indel_free_window({n_codons}codons_anywhere)", meta
    return f"window_starts_too_late(h{first_ok}>{max_start_codon})", meta


SEQS = paths.STRAT_SEQS
CDS = {
    "human": SEQS / "Homo_sapiens.GRCh38.cds.all.fa.gz",
    "platypus": SEQS / "Ornithorhynchus_anatinus.mOrnAna1.p.v1.cds.all.fa.gz",
}
GTF = {
    "human": SEQS / "Homo_sapiens.GRCh38.116.gtf.gz",
    "platypus": SEQS / "Ornithorhynchus_anatinus.mOrnAna1.p.v1.116.gtf.gz",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonical_transcripts(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(gene_id -> canonical transcript_id, transcript_id -> gene_id) from a GTF."""
    canon: dict[str, str] = {}
    t2g: dict[str, str] = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "transcript":
                continue
            attr = f[8]
            g = t = None
            for kv in attr.split(";"):
                kv = kv.strip()
                if kv.startswith("gene_id "):
                    g = kv.split('"')[1].split(".")[0]
                elif kv.startswith("transcript_id "):
                    t = kv.split('"')[1].split(".")[0]
            if not g or not t:
                continue
            t2g[t] = g
            if "Ensembl_canonical" in attr:
                canon[g] = t
    return canon, t2g


def read_cds(path: Path, wanted_tx: set[str]) -> dict[str, str]:
    """transcript_id (unversioned) -> CDS, restricted to `wanted_tx`."""
    out: dict[str, str] = {}
    tid = None
    buf: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if tid and buf:
                    out[tid] = "".join(buf).upper()
                raw = line[1:].split()[0].split(".")[0]
                tid = raw if raw in wanted_tx else None
                buf = []
            elif tid:
                buf.append(line.strip())
    if tid and buf:
        out[tid] = "".join(buf).upper()
    return out


def pick_transcript(
    gene: str, canon: dict[str, str], by_gene: dict[str, list[tuple[str, str]]]
) -> tuple[str | None, str | None, str]:
    """
    Canonical transcript if present and non-empty, else the longest CDS. Same rule both species.
    """
    cands = by_gene.get(gene, [])
    if not cands:
        return None, None, "no_cds_in_release"
    ct = canon.get(gene)
    for tid, seq in cands:
        if tid == ct:
            return tid, seq, "canonical"
    tid, seq = max(cands, key=lambda x: (len(x[1]), x[0]))
    return tid, seq, "longest_cds"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage0", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument(
        "--max-start-codon",
        type=int,
        default=0,
        help="0 = the production codon-0 rule (bit-identical to qc_pair). >0 accepts the "
        "first indel-free 30-codon homologous window starting at or before this "
        "codon -- the design's shifted-window sensitivity arm (30 => within 60).",
    )
    ap.add_argument(
        "--recovered-only",
        action="store_true",
        help="keep ONLY genes that FAIL the codon-0 rule but pass the shifted rule. "
        "Gives a set disjoint from the production panel, so 'do the headline "
        "correlations survive on the recovered genes' is a real replication rather "
        "than a re-measurement of mostly the same genes.",
    )
    args = ap.parse_args()
    if args.recovered_only and args.max_start_codon <= 0:
        raise SystemExit("--recovered-only is meaningless without --max-start-codon > 0")
    args.out.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(args.stage0 / "frozen_order.csv")
    log(f"frozen order: {len(order)} blocks, {order.stratum.nunique()} strata")

    # ---- transcript choice + CDS, both species ------------------------------------------------
    seqs: dict[str, dict[str, tuple[str, str, str]]] = {}
    for sp in ("human", "platypus"):
        for kind, table in (("CDS FASTA", CDS), ("GTF", GTF)):
            paths.require(
                table[sp],
                f"the {sp} {kind} (Ensembl release 116)",
                "download the release-116 human/platypus CDS, peptide and GTF files from "
                "https://ftp.ensembl.org/pub/release-116/ -- see REPRODUCING.md",
                "GLM_STRAT_SEQS",
            )
        canon, t2g = canonical_transcripts(GTF[sp])
        ids = set(order.gene_id) if sp == "human" else set(order.plat_gene_id)
        want_tx = {t for t, g in t2g.items() if g in ids}
        cds = read_cds(CDS[sp], want_tx)
        by_gene: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for t, s in cds.items():
            by_gene[t2g[t]].append((t, s))
        seqs[sp] = {g: pick_transcript(g, canon, by_gene) for g in ids}
        n = sum(1 for v in seqs[sp].values() if v[1])
        log(
            f"{sp}: {n}/{len(ids)} genes with a CDS "
            f"({sum(1 for v in seqs[sp].values() if v[2] == 'canonical')} canonical)"
        )

    # ---- walk the frozen order ----------------------------------------------------------------
    kept: list[dict] = []
    attrition: list[dict] = []
    per_stratum: dict[int, int] = defaultdict(int)

    for row in order.sort_values(["stratum", "priority"]).itertuples():
        s = int(row.stratum)
        if per_stratum[s] >= args.per_stratum:
            continue
        h_tid, h_cds, h_how = seqs["human"][row.gene_id]
        p_tid, p_cds, p_how = seqs["platypus"][row.plat_gene_id]
        base = {
            "stratum": s,
            "priority": int(row.priority),
            "block": int(row.block),
            "gene_id": row.gene_id,
            "plat_gene_id": row.plat_gene_id,
            "perc_id_hp": row.perc_id_hp,
            "human_tx": h_tid,
            "platypus_tx": p_tid,
            "human_tx_rule": h_how,
            "platypus_tx_rule": p_how,
        }
        if not h_cds or not p_cds:
            attrition.append({**base, "reason": f"no_cds({h_how if not h_cds else p_how})"})
            continue
        reason, meta = qc_pair_window(h_cds, p_cds, args.max_start_codon)
        if reason:
            attrition.append({**base, "reason": reason, **meta})
            continue
        # The recovered set is defined by what the PRODUCTION gate rejects, so ask it directly
        # rather
        # than inferring from the offset: a gene can pass codon-0 and still be reported at offset 0.
        if args.recovered_only:
            base_reason, _ = qc_pair(h_cds, p_cds)
            if base_reason is None:
                attrition.append({**base, "reason": "already_in_production_panel", **meta})
                continue
            base["displaced_reason"] = base_reason
        off_h = 3 * int(meta.get("prefix_codon_human", 0))
        off_p = 3 * int(meta.get("prefix_codon_platypus", 0))
        per_stratum[s] += 1
        kept.append(
            {
                **base,
                **meta,
                "reason": "pass",
                "cds_len_human": len(h_cds),
                "cds_len_platypus": len(p_cds),
                "prefix_offset_h_bp": off_h,
                "prefix_offset_p_bp": off_p,
                "prefix_human": h_cds[off_h : off_h + PREFIX_BP],
                "prefix_platypus": p_cds[off_p : off_p + PREFIX_BP],
                "cds_human": h_cds,
                "cds_platypus": p_cds,
            }
        )
        attrition.append({**base, "reason": "pass", **meta})

    df = pd.DataFrame(kept)
    if df.empty:
        raise SystemExit("no genes passed QC -- inspect attrition.csv")
    # `gene` is the join key used by every downstream stage; keep the Ensembl id as provenance.
    df["gene"] = df.gene_id
    df["family"] = "block" + df.block.astype(str)  # block == family for a block-disjoint panel
    # Reserve the opossum-arm availability flag; this panel has no opossum orthologs.
    df["has_opossum"] = False

    pairs_cols = [c for c in df.columns if c not in ("cds_human", "cds_platypus")]
    df[pairs_cols].to_csv(args.out / "pairs.csv", index=False)
    for sp in ("human", "platypus"):
        with (args.out / f"cds_{sp}.fasta").open("w") as fh:
            for r in df.itertuples():
                fh.write(f">{r.gene}|{sp}\n{getattr(r, f'cds_{sp}')}\n")
    att = pd.DataFrame(attrition)
    att.to_csv(args.out / "attrition.csv", index=False)

    # ---- the attrition report IS a result -----------------------------------------------------
    att["passed"] = att.reason == "pass"
    rate = att.groupby("stratum").passed.agg(["sum", "count", "mean"])
    log("\nQC attrition by stratum (pass / examined / rate):\n" + rate.round(3).to_string())
    top = att[~att.passed].reason.str.replace(r"\(.*", "", regex=True).value_counts().head(8)
    log("\ntop rejection reasons:\n" + top.to_string())
    summary = {
        "max_start_codon": int(args.max_start_codon),
        "recovered_only": bool(args.recovered_only),
        "prefix_offset_h_bp": [
            int(df.prefix_offset_h_bp.min()),
            int(df.prefix_offset_h_bp.median()),
            int(df.prefix_offset_h_bp.max()),
        ],
        "n_offset_nonzero": int((df.prefix_offset_h_bp > 0).sum()),
        "n_kept": int(len(df)),
        "per_stratum": {int(k): int(v) for k, v in sorted(per_stratum.items())},
        "n_examined": int(len(att)),
        "pass_rate_by_stratum": {int(k): float(v) for k, v in rate["mean"].items()},
        "examined_by_stratum": {int(k): int(v) for k, v in rate["count"].items()},
        "rejection_reasons": {str(k): int(v) for k, v in top.items()},
        "perc_id_by_stratum": {
            int(s): [
                float(g.perc_id_hp.min()),
                float(g.perc_id_hp.median()),
                float(g.perc_id_hp.max()),
            ]
            for s, g in df.groupby("stratum")
        },
        "cds_len_human": [
            int(df.cds_len_human.min()),
            int(df.cds_len_human.median()),
            int(df.cds_len_human.max()),
        ],
    }
    (args.out / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\nkept {len(df)} genes -> {args.out / 'pairs.csv'}")
    log(json.dumps(summary["perc_id_by_stratum"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
