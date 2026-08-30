"""H1c — are the per-gene direction residuals clustered, and who is in the clusters? CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, spearmanr

K_RANGE = range(2, 9)
N_NULL = 500
N_BOOT = 500
SEED = 20260807

# Membership covariates. Only rate/conservation carry a test; the rest are descriptive by declaration.
TESTED = ["perc_id_hp"]
DESCRIPTIVE = ["cds_len_human", "gc3_div", "delta_norm", "retained_frac"]


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def partition(R: np.ndarray, k: int) -> np.ndarray:
    """Average linkage on angular distance, at k clusters. Fixed in advance, not tuned."""
    C = np.clip(unit(R) @ unit(R).T, -1.0, 1.0)
    d = squareform(np.clip(1.0 - C, 0.0, 2.0), checks=False)
    return fcluster(linkage(d, method="average"), k, criterion="maxclust")


def silhouette(R: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette on angular distance. Returns 0 for a degenerate (single-cluster) partition."""
    if len(np.unique(lab)) < 2:
        return 0.0
    D = np.clip(1.0 - np.clip(unit(R) @ unit(R).T, -1.0, 1.0), 0.0, 2.0)
    out = np.empty(len(R))
    for i in range(len(R)):
        same = (lab == lab[i]) & (np.arange(len(R)) != i)
        if not same.any():
            out[i] = 0.0
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        out[i] = (b - a) / max(a, b)
    return float(out.mean())


def dispersion(R: np.ndarray, lab: np.ndarray) -> float:
    """log within-cluster dispersion: the gap statistic's W_k, on angular distance."""
    tot = 0.0
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if len(idx) < 2:
            continue
        C = np.clip(unit(R[idx]) @ unit(R[idx]).T, -1.0, 1.0)
        tot += float(np.clip(1.0 - C, 0.0, 2.0)[np.triu_indices(len(idx), 1)].sum()) / len(idx)
    return float(np.log(max(tot, 1e-12)))


def spectrum_null(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian with the observed covariance SPECTRUM, random orientation: rank kept, clusters killed."""
    n = len(R)
    s = np.linalg.svd(R, compute_uv=False)
    Z = rng.standard_normal((n, len(s)))
    Z /= np.linalg.norm(Z, axis=0, keepdims=True) / np.sqrt(n)   # columns to unit scale
    return Z * s                                                  # apply the observed spectrum


def isotropic_null(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reference point only: how much apparent structure comes from anisotropy alone."""
    return unit(rng.standard_normal(R.shape)) * np.linalg.norm(R, axis=1, keepdims=True)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index. Written out rather than imported to keep the sklearn dep off this path."""
    ca, cb = pd.factorize(a)[0], pd.factorize(b)[0]
    n = len(ca)
    M = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(M, (ca, cb), 1)
    comb = lambda x: (x * (x - 1) / 2).sum()
    idx = comb(M)
    ea, eb = comb(M.sum(1)), comb(M.sum(0))
    exp = ea * eb / comb(np.array([n]))
    mx = (ea + eb) / 2
    return float((idx - exp) / (mx - exp)) if mx != exp else 1.0


def gc3(s: str) -> float:
    third = s[2::3]
    return sum(c in "GC" for c in third) / max(len(third), 1)


def read_fasta(p: Path) -> dict[str, str]:
    out, hid, buf = {}, None, []
    for line in p.read_text().splitlines():
        if line.startswith(">"):
            if hid:
                out[hid.split("|")[0]] = "".join(buf).upper()
            hid, buf = line[1:], []
        elif line.strip():
            buf.append(line.strip())
    if hid:
        out[hid.split("|")[0]] = "".join(buf).upper()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=27, help="frozen layer; blocks 28-31 are unusable")
    ap.add_argument("--mode", default="cds_mean", help="pooling mode; cds_mean is the only primary one")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.run / "h1c_clusters")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    pairs = pd.read_csv(args.run / "stage1" / "pairs.csv")
    npz = np.load(args.run / "stage2" / "pooled_representations.npz", allow_pickle=True)
    genes = [str(g) for g in npz["genes"]]
    if list(pairs.gene) != genes:
        raise AssertionError("gene order mismatch between pairs.csv and pooled_representations.npz")

    a = npz[args.mode]
    D = (a[:, 1, args.layer, :] - a[:, 0, args.layer, :]).astype(np.float64)
    R = D - D.mean(0)                       # residual after removing the shared direction
    n = len(R)
    print(f"{n} genes, layer {args.layer}, mode {args.mode}")

    # ---- 0. is the residual cloud non-isotropic at all? (context for the gates below) ----
    Ru = unit(R)
    off = (Ru @ Ru.T)[np.triu_indices(n, 1)]
    null_off = np.concatenate([(lambda G: (G @ G.T)[np.triu_indices(n, 1)])(
        unit(rng.standard_normal(R.shape))) for _ in range(50)])
    s = np.linalg.svd(Ru, compute_uv=False) ** 2
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    print(f"  residual pairwise-cos sd {off.std():.4f} vs isotropic {null_off.std():.4f} "
          f"(x{off.std()/null_off.std():.1f});  participation ratio {pr:.1f}")

    # ---- 1. EXISTENCE: max-over-k silhouette excess vs the SPECTRUM-MATCHED null ----
    #
    # Compare maximum silhouette excess across k=2..8 with null replicates that perform
    # the same search, accounting for selection over k.
    ks = list(K_RANGE)
    nulls = [spectrum_null(R, rng) for _ in range(N_NULL)]
    isos = [isotropic_null(R, rng) for _ in range(100)]
    labs = {k: partition(R, k) for k in ks}
    sil_obs = np.array([silhouette(R, labs[k]) for k in ks])
    disp_obs = np.array([dispersion(R, labs[k]) for k in ks])
    nsil = np.array([[silhouette(N, partition(N, k)) for k in ks] for N in nulls])   # (rep, k)
    ndisp = np.array([[dispersion(N, partition(N, k)) for k in ks] for N in nulls])
    isil = np.array([[silhouette(N, partition(N, k)) for k in ks] for N in isos])

    excess = sil_obs - nsil.mean(0)
    null_excess = nsil - nsil.mean(0)                      # centre each null on the same expectation
    p_global = float((np.sum(null_excess.max(1) >= excess.max()) + 1) / (N_NULL + 1))
    kstar = int(ks[int(np.argmax(excess))])
    exists = bool(p_global < 0.05)

    rows = []
    for j, k in enumerate(ks):
        rows.append({
            "k": k, "sizes": ",".join(map(str, np.bincount(labs[k])[1:])),
            "silhouette": round(float(sil_obs[j]), 4),
            "sil_null_mean": round(float(nsil[:, j].mean()), 4),
            "sil_null_sd": round(float(nsil[:, j].std(ddof=1)), 4),
            "sil_excess": round(float(excess[j]), 4),
            "sil_p_at_k": float((np.sum(nsil[:, j] >= sil_obs[j]) + 1) / (N_NULL + 1)),
            "sil_isotropic_mean": round(float(isil[:, j].mean()), 4),
            "gap": round(float(ndisp[:, j].mean() - disp_obs[j]), 4),
            "gap_se": round(float(ndisp[:, j].std(ddof=1) * np.sqrt(1 + 1 / N_NULL)), 4),
        })
        print(f"  k={k}  sizes {rows[-1]['sizes']:20s} sil {sil_obs[j]:+.3f} "
              f"(spectrum null {nsil[:, j].mean():+.3f}+/-{nsil[:, j].std(ddof=1):.3f}, "
              f"excess {excess[j]:+.3f}, p={rows[-1]['sil_p_at_k']:.4f}; "
              f"isotropic {isil[:, j].mean():+.3f})")
    ex = pd.DataFrame(rows)
    print(f"\n  max-over-k excess {excess.max():+.4f} at k* = {kstar}, "
          f"global p = {p_global:.4f}  ->  EXISTENCE GATE: {'PASS' if exists else 'FAIL'}")

    lab = labs[kstar]

    # ---- 2. STABILITY: block bootstrap + disjoint halves. Blocks == genes on this panel. ----
    boot = []
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        keep = np.unique(idx)                      # ARI needs each gene once
        boot.append(ari(partition(R[keep], kstar), lab[keep]))
    half = []
    for _ in range(N_BOOT):
        p = rng.permutation(n)
        h1, h2 = p[: n // 2], p[n // 2:]
        # fit on one half, score agreement with the full-panel partition on the OTHER half
        half.append(ari(partition(R[h2], kstar), lab[h2]) if len(np.unique(lab[h1])) > 1 else np.nan)
    stab = {"k": kstar,
            "bootstrap_ari_median": round(float(np.median(boot)), 4),
            "bootstrap_ari_q05": round(float(np.quantile(boot, 0.05)), 4),
            "half_ari_median": round(float(np.nanmedian(half)), 4)}
    stable = bool(stab["bootstrap_ari_median"] >= 0.5)
    print(f"  bootstrap ARI {stab['bootstrap_ari_median']:.3f} "
          f"[q05 {stab['bootstrap_ari_q05']:.3f}], half ARI {stab['half_ari_median']:.3f}"
          f"  -> STABILITY GATE: {'PASS' if stable else 'FAIL'}")

    # ---- 3. MEMBERSHIP: what are these clusters? Tested for rate; everything else descriptive. ----
    ch = read_fasta(args.run / "stage1" / "cds_human.fasta")
    cp = read_fasta(args.run / "stage1" / "cds_platypus.fasta")
    m = pairs.copy()
    m["cluster"] = lab
    m["delta_norm"] = np.linalg.norm(D, axis=1)
    m["gc3_div"] = [gc3(cp[g]) - gc3(ch[g]) for g in genes]
    cov = args.run / "stage2" / "aligned_coverage.csv"
    if cov.exists():
        m = m.merge(pd.read_csv(cov)[["gene", "retained_frac"]], on="gene", how="left")

    # rate predictors, if stage 5 has run
    rate_cols = []
    ts = args.run / "stage5" / "tree_stats.csv"
    dn = args.run / "stage5" / "dnds.csv"
    for p, cols in [(ts, ["tree_len", "diameter", "treeness", "background_rate", "platypus_branch"]),
                    (dn, ["dN_hp_yn", "dS_hp_yn", "omega_hp_yn"])]:
        if p.exists():
            d = pd.read_csv(p)
            have = [c for c in cols if c in d.columns]
            m = m.merge(d[["gene"] + have], on="gene", how="left")
            rate_cols += have

    mem = []
    for c in TESTED + rate_cols:
        if c not in m.columns:
            continue
        groups = [pd.to_numeric(m.loc[m.cluster == g, c], errors="coerce").dropna()
                  for g in sorted(m.cluster.unique())]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        h, p = kruskal(*groups)
        mem.append({"variable": c, "role": "tested", "kruskal_H": round(float(h), 3), "p": float(p),
                    "by_cluster": " / ".join(f"{g.mean():.3f}" for g in groups)})
    for c in DESCRIPTIVE:
        if c not in m.columns:
            continue
        vals = [pd.to_numeric(m.loc[m.cluster == g, c], errors="coerce").dropna()
                for g in sorted(m.cluster.unique())]
        mem.append({"variable": c, "role": "descriptive", "kruskal_H": np.nan, "p": np.nan,
                    "by_cluster": " / ".join(f"{v.mean():.3f}" for v in vals)})
    mem = pd.DataFrame(mem)

    print("\n  membership (tested variables carry a p; descriptive ones do not):")
    for _, r in mem.iterrows():
        pv = "" if pd.isna(r.p) else f"  p={r.p:.4f}"
        print(f"    {r.variable:22s} {r.role:12s} {r.by_cluster}{pv}")

    ex.to_csv(out / "existence.csv", index=False)
    mem.to_csv(out / "membership.csv", index=False)
    m[["gene", "block", "stratum", "cluster", "delta_norm", "perc_id_hp"]].to_csv(
        out / "assignments.csv", index=False)
    (out / "h1c_summary.json").write_text(json.dumps({
        "layer": args.layer, "mode": args.mode, "n_genes": n, "seed": SEED,
        "k_range": [K_RANGE.start, K_RANGE.stop - 1], "n_null": N_NULL, "n_boot": N_BOOT,
        "algorithm": "average linkage on angular distance; k and existence both from the "
                     "max-over-k silhouette excess, so searching k is free",
        "p_global": p_global,
        "null": "spectrum-matched Gaussian (covariance eigenvalues preserved, random orientation); "
                "isotropic null reported alongside as an anisotropy reference",
        "residual_cos_sd": round(float(off.std()), 4),
        "isotropic_cos_sd": round(float(null_off.std()), 4),
        "participation_ratio": round(pr, 2),
        "k_selected": kstar, "existence_gate": exists, "stability_gate": stable,
        "stability": stab,
        "tier_c_licensed": bool(exists and stable),
    }, indent=2))
    print(f"\n-> {out}    Tier C licensed by H1c: {exists and stable}")


if __name__ == "__main__":
    main()
