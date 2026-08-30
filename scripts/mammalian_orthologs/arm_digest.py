"""Readable results digest for one mammalian-ortholog arm, printed as soon as that arm finishes."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY_LAYERS = [0, 5, 8, 13, 15, 19, 24, 31]


def _layers(run: Path, name: str) -> pd.DataFrame:
    """Concatenate a per-layer CSV across blocks*/ with a `layer` column."""
    rows = []
    for f in sorted(glob.glob(str(run / "blocks*" / name))):
        m = re.search(r"blocks(\d+)", f)
        d = pd.read_csv(f)
        d["layer"] = int(m.group(1))
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _controls(run: Path, which: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(run / "blocks*" / "controls" / f"control_{which}_scores.csv"))):
        d = pd.read_csv(f)
        d["layer"] = int(re.search(r"blocks(\d+)", f).group(1))
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def digest(arm: str) -> str:
    run = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{arm}"
    out = [f"# Mammalian ortholog arm: `{arm}`", "", f"Run dir: `{run.relative_to(ROOT)}`"]

    # ── 1. within-group vs the independent species tree
    st = _layers(run, "within_family_speciestree.csv")
    out += ["", "## 1. Within-group recovery vs the INDEPENDENT species tree", ""]
    if st.empty:
        out += ["_(no within_family_speciestree.csv yet)_"]
    else:
        col = "spearman_geodesic_speciestree"
        per_layer = st.groupby("layer")[col].agg(["mean", "median", "count"]).round(3)
        best = per_layer["mean"].idxmax()
        out += [f"**{int(per_layer['count'].iloc[0])} families scored.** "
                f"Best layer by mean rho: **block {best}** (mean {per_layer.loc[best,'mean']:.3f}).", ""]
        out += ["| layer | mean rho | median rho |", "|---|---|---|"]
        for L in [x for x in KEY_LAYERS if x in per_layer.index] + ([best] if best not in KEY_LAYERS else []):
            r = per_layer.loc[L]
            out.append(f"| {L}{' **(best)**' if L == best else ''} | {r['mean']:.3f} | {r['median']:.3f} |")
        top = st[st.layer == best].nlargest(8, col)[["family", col]]
        out += ["", f"Top families at block {best}:", ""]
        out += ["| family | rho |", "|---|---|"]
        out += [f"| {r.family} | {getattr(r, col):.3f} |" for r in top.itertuples()]

    # ── 2. between-family axes
    out += ["", "## 2. Between-family axes (centroid geodesic)", ""]
    bf = _layers(run, "between_family_baseline_scores.csv")
    if bf.empty:
        out += ["_(no between_family_baseline_scores.csv yet)_"]
    else:
        hom = bf[bf.baseline == "pfam_jsd"]
        best = int(hom.loc[hom.spearman_rho.idxmax(), "layer"]) if not hom.empty else int(bf.layer.min())
        sub = bf[bf.layer == best].sort_values("axis")
        out += [f"At the homology-peak layer **block {best}**:", "",
                "| baseline | axis | rho | Mantel p |", "|---|---|---|---|"]
        for r in sub.itertuples():
            p = getattr(r, "p_mantel", float("nan"))
            out.append(f"| {r.baseline} | {r.axis} | {r.spearman_rho:+.3f} | {p:.3f} |")
        n_fam = None
        cen = sorted(glob.glob(str(run / "blocks*" / "evo2_mammal_centroid_distances.csv")))
        if cen:
            n_fam = len(pd.read_csv(cen[0], index_col=0))
        if n_fam:
            out += ["", f"({n_fam} family nodes in the centroid graph — was 9 before the expansion.)"]

    # ── 3. composition controls
    out += ["", "## 3. Composition-control preservation (rho vs the natural geometry)", ""]
    cb = _controls(run, "between")
    if cb.empty:
        out += ["_(no control tables yet)_"]
    else:
        cb = cb[cb.condition != "natural"]
        piv = cb.pivot_table(index="layer", columns="condition", values="rho_vs_natural_centroid")
        keep = [L for L in KEY_LAYERS if L in piv.index]
        out += ["Between-family. Low = real structure beyond that composition level.", ""]
        out += ["| layer | " + " | ".join(piv.columns) + " |",
                "|---" * (len(piv.columns) + 1) + "|"]
        for L in keep:
            out.append(f"| {L} | " + " | ".join(f"{piv.loc[L, c]:.3f}" if pd.notna(piv.loc[L, c]) else "–"
                                               for c in piv.columns) + " |")
        band = piv.loc[[L for L in piv.index if 8 <= L <= 24]].mean().round(3)
        out += ["", "Mean over blocks 8-24: " + ", ".join(f"**{c}** {band[c]:.3f}" for c in band.index)]
        if "written_at" in cb.columns and cb.written_at.nunique() > 1:
            out += ["", f"> NOTE mixed provenance: rows written on {sorted(cb.written_at.unique())} "
                        "— this table spans more than one methods state."]

    cw = _controls(run, "within")
    if not cw.empty and "rho_geodesic_speciestree" in cw.columns:
        cw = cw[cw.condition != "natural"]
        p2 = cw.pivot_table(index="layer", columns="condition", values="rho_geodesic_speciestree")
        band = p2.loc[[L for L in p2.index if 8 <= L <= 24]].mean().round(3)
        out += ["", "Within-group RECOVERY under each control (geodesic vs species tree), "
                "mean over blocks 8-24 — does the shuffle still recover phylogeny?", "",
                ", ".join(f"**{c}** {band[c]:.3f}" for c in band.index)]

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["transcript", "cds", "transcript_cdsmask"])
    args = ap.parse_args()
    text = digest(args.arm)
    run = ROOT / "results" / f"2026-07-16_mammalian-orthologs-{args.arm}"
    run.mkdir(parents=True, exist_ok=True)
    (run / f"DIGEST_{args.arm}.md").write_text(text)
    print(text)
    print(f"[saved] {run / f'DIGEST_{args.arm}.md'}")


if __name__ == "__main__":
    main()
