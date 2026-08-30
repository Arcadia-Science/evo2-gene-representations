"""What are the principal components of the per-gene steering delta, beyond the GC axis? CPU only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))

import arcadia_style as acs  # noqa: E402
from plot_utils import set_pub_style  # noqa: E402

HUMAN_REF, PLAT_REF = "__human_reference__", "__platypus_reference__"
N_PC = 8


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def gene_features(run: Path, genes: list[str]) -> pd.DataFrame:
    """Per-gene sequence features to interpret the PCs against, indexed like `genes`."""
    f = pd.read_csv(run / "composition_profile" / "features_per_gene.csv.gz")
    h = f[f.condition == HUMAN_REF].set_index("gene")
    p = f[f.condition == PLAT_REF].set_index("gene")
    scal = [c for c in ("gc", "gc1", "gc2", "gc3", "cpg_oe", "f_A", "f_C", "f_G", "f_T")
            if c in h.columns]
    rho = [c for c in h.columns if c.startswith("rho_")]
    out = pd.DataFrame(index=pd.Index(genes, name="gene"))
    for c in scal + rho:
        out[f"d_{c}"] = (p[c] - h[c]).reindex(genes)          # platypus minus human
    for c in scal:
        out[f"human_{c}"] = h[c].reindex(genes)               # the starting point itself
    for c in [c for c in p.columns if c.startswith("HP_")]:
        out[c] = p[c].reindex(genes)                          # per-gene between-species JSD
    if "n_tokens" in h.columns:
        out["window_bp"] = h["n_tokens"].reindex(genes)

    # the platypus target's own divergence from human, and how strongly the gene responded
    sc = run / "stage4_cds_mean_blocks27" / "stage4_scores_nt.csv"
    if sc.exists():
        s = pd.read_csv(sc).drop_duplicates("gene").set_index("gene")
        for c in ("perc_id_hp", "stratum"):
            if c in s.columns:
                out[c] = s[c].reindex(genes)
    sd = run / "site_directionality" / "per_gene.csv"
    if sd.exists():
        d = pd.read_csv(sd)
        d = d[d.site_set == "private"]
        b = d[d.condition == "unsteered"].set_index("gene")["A_cov"]
        a = d[d.condition == "add_a1.0"].set_index("gene")["A_cov"]
        out["gain_A_a1"] = (a - b).reindex(genes)
        out["unsteered_A"] = b.reindex(genes)
    lo = run / "stage3_cds_mean" / "loo_diagnostics.csv"
    if lo.exists():
        d = pd.read_csv(lo)
        d = d[d.layer == 27].drop_duplicates("gene").set_index("gene")
        for c in ("delta_norm", "loo_cos", "abs_cos_gc", "frac_on_cone"):
            if c in d.columns:
                out[f"loo_{c}"] = d[c].reindex(genes)
    return out


def analyse_layer(X: np.ndarray, ip: int, ih: int, vecs, layer: int,
                  genes: list[str], feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    D = (X[:, ip, layer, :] - X[:, ih, layer, :]).astype(np.float64)
    Dc = D - D.mean(axis=0, keepdims=True)

    refs = {}
    for name, key in (("gc_axis", f"gc_axis_L{layer}"), ("v_pooled", f"v_pooled_L{layer}"),
                      ("muhat", f"muhat_L{layer}")):
        if key in vecs:
            refs[name] = unit(vecs[key].astype(np.float64))

    rows = []
    for centred, Dm in (("centred", Dc), ("uncentred", D)):
        _U, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        var = S ** 2
        frac = var / var.sum()
        # participation ratio: the effective number of directions the cloud occupies
        pr = float((var.sum() ** 2) / np.sum(var ** 2))
        scores = Dm @ Vt.T                     # (genes, components)
        for k in range(min(N_PC, Vt.shape[0])):
            row = {"layer": layer, "basis": centred, "pc": k + 1,
                   "var_frac": float(frac[k]), "cum_var_frac": float(frac[:k + 1].sum()),
                   "participation_ratio": pr}
            for name, r in refs.items():
                row[f"abs_cos_{name}"] = abs(float(Vt[k] @ r))
            rows.append(row)
        if centred == "centred":
            sc = pd.DataFrame(scores[:, :N_PC],
                              columns=[f"PC{k + 1}" for k in range(min(N_PC, Vt.shape[1]))],
                              index=pd.Index(genes, name="gene"))
    spectrum = pd.DataFrame(rows)

    # ---- what does each PC track across genes? ------------------------------------------------
    cors = []
    for pc in sc.columns:
        for feat in feats.columns:
            a, b = sc[pc], feats[feat]
            ok = a.notna() & b.notna()
            if ok.sum() < 30:
                continue
            r = float(a[ok].rank().corr(b[ok].rank()))
            cors.append({"layer": layer, "pc": pc, "feature": feat, "spearman": r,
                         "n": int(ok.sum())})
    return spectrum, pd.DataFrame(cors)


def figure(spec: pd.DataFrame, cors: pd.DataFrame, out: Path, layer: int) -> None:
    set_pub_style(title_size=8.5, tick_size=6.5)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5),
                             gridspec_kw={"width_ratios": [1, 1, 1.5]})
    s = spec[spec.layer == layer]

    ax = axes[0]
    for basis, col in (("uncentred", acs.SERIES_NULL), ("centred", acs.SERIES_PRIMARY)):
        d = s[s.basis == basis]
        ax.plot(d.pc, 100 * d.var_frac, "o-", color=col, lw=1.2, ms=4,
                label=f"{basis} (PR = {d.participation_ratio.iloc[0]:.1f})")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title(f"Scree — per-gene delta at blocks.{layer}")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1]
    for basis, col in (("uncentred", acs.SERIES_NULL), ("centred", acs.SERIES_PRIMARY)):
        d = s[s.basis == basis]
        ax.plot(d.pc, 100 * d.cum_var_frac, "o-", color=col, lw=1.2, ms=4, label=basis)
    ax.set_ylim(0, 100)
    ax.set_xlabel("top-k principal components")
    ax.set_ylabel("cumulative variance (%)")
    ax.set_title("Cumulative variance")
    ax.legend(fontsize=6, frameon=False)

    ax = axes[2]
    d = s[s.basis == "centred"]
    cols = [c for c in d.columns if c.startswith("abs_cos_")]
    M = d.set_index("pc")[cols].to_numpy().T
    im = ax.imshow(M, cmap=acs.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(d)), d.pc.astype(int), fontsize=6)
    ax.set_yticks(range(len(cols)), [c.replace("abs_cos_", "") for c in cols], fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if M[i, j] < 0.6 else "black")
    ax.set_xlabel("principal component (centred cloud)")
    ax.set_title("|cos| with the pipeline's reference directions")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(out / f"25_delta_pc_structure_L{layer}.{e}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[27, 24])
    ap.add_argument("--rep-dir", default="stage3_cds_mean")
    ap.add_argument("--out", default="pc_structure")
    ap.add_argument("--top", type=int, default=8, help="top correlates to print per PC")
    args = ap.parse_args()
    out = args.run / args.out
    out.mkdir(parents=True, exist_ok=True)

    pooled = np.load(args.run / "stage2" / "pooled_representations.npz", allow_pickle=True)
    X = pooled["cds_mean"]
    genes_all = [str(g) for g in pooled["genes"]]
    sp = [str(x) for x in pooled["species"]]
    ip = next(i for i, s in enumerate(sp)
              if s.lower().startswith("plat") or s.lower().startswith("orn"))
    ih = 1 - ip
    vecs = np.load(args.run / args.rep_dir / "loo_vectors.npz", allow_pickle=True)

    # restrict to the genes actually scored, so the cloud matches the panel every other table uses
    plan = pd.read_csv(args.run / "stage4_cds_mean_blocks27" / "scoring_plan.csv")
    keep = set(plan[plan.usable].gene)
    sel = [i for i, g in enumerate(genes_all) if g in keep]
    genes = [genes_all[i] for i in sel]
    X = X[sel]
    print(f"delta cloud: {len(genes)} genes x {X.shape[-1]} dims  (species: platypus={sp[ip]})")

    feats = gene_features(args.run, genes)
    print(f"interpreting against {feats.shape[1]} per-gene features")

    specs, corrs = [], []
    for layer in args.layers:
        spec, cors = analyse_layer(X, ip, ih, vecs, layer, genes, feats)
        specs.append(spec)
        corrs.append(cors)
        figure(spec, cors, out, layer)

        print(f"\n================ blocks.{layer} ================")
        for basis in ("uncentred", "centred"):
            d = spec[spec.basis == basis]
            print(f"\n{basis} cloud (participation ratio {d.participation_ratio.iloc[0]:.1f}):")
            print(d[["pc", "var_frac", "cum_var_frac"]
                    + [c for c in d.columns if c.startswith("abs_cos_")]]
                  .to_string(index=False, float_format=lambda z: f"{z:.3f}"))
        print(f"\ntop {args.top} sequence correlates of each PC (centred cloud, Spearman):")
        for pc, g in cors.groupby("pc", sort=False):
            g = g.reindex(g.spearman.abs().sort_values(ascending=False).index).head(args.top)
            best = ", ".join(f"{r.feature} {r.spearman:+.2f}" for r in g.itertuples())
            print(f"  {pc}: {best}")

    pd.concat(specs).to_csv(out / "pc_spectrum.csv", index=False)
    pd.concat(corrs).to_csv(out / "pc_feature_correlations.csv", index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
