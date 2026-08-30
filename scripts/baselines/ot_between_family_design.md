# Centroid-free between-family proximity — optimal-transport metrics

**Status:** implemented (2026-07-22). Companion to `scripts/evo2/between_family_baselines_design.md`.
Code: `scripts/baselines/ot_between_family.py` (core + validation), `ot_between_family_sweep.py`
(all-layer scoring for both panels), `ot_between_family_figure.py` (overlay figure),
`test_ot_between_family.py` (pytest).

## Why

The established between-family metric collapses each family to one L2-normalized **centroid**
direction and takes the angular k-NN geodesic between the F centroids
(`geodesic_utils.compute_centroid_geodesic`). That is size-robust but throws away the *shape*
of each family's gene cloud — its spread, anisotropy, and internal geometry. Two families whose
centroids coincide but whose clouds are oriented differently look identical to the centroid
metric. Optimal transport compares the **full empirical distributions** of genes, so it can see
that structure. We add two centroid-free family-distance matrices and score all three approaches
(geodesic, Wasserstein, FGW) against the *same* homology / mechanism / composition baselines, so
the centroid-free metrics are judged on identical ground truth.

## Shared preprocessing (must match the centroid analysis exactly)

- **Same embeddings.** For each model/panel/layer we read the identical gene embedding array the
  centroid analysis consumes (`stack[layer]`), with the identical global preprocessing. The
  centroid path applies **only per-gene L2-normalization and no feature-wise standardization**, so
  neither do we. An optional `standardize` flag (default off) would fit **one** scaler across all
  genes in that model/panel/layer — never per family.
- **Ground cost = direct angular distance** in the shared embedding space, NOT any k-NN graph,
  shortest path, or existing geodesic:

      theta(x, y) = arccos(clip(cos_sim(x, y), -1, 1)) / pi        ∈ [0, 1]

- **Uniform within-family weights** `p_i = 1/n_f`: every family carries total mass 1 regardless of
  size. Unequal sizes are supported directly; the primary analysis does **not** subsample.

## The two metrics

1. **All-gene Wasserstein (W2)** — primary centroid-free proximity. Cross-family angular matrix
   `A_fg`; squared cost `M = A_fg**2`; `w2 = sqrt(max(ot.emd2(p, q, M), 0))`. Exact network simplex.

2. **Fused Gromov–Wasserstein (FGW)** — `ot.gromov.fused_gromov_wasserstein2(M, C_f, C_g, p, q,
   loss_fun="square_loss", alpha=α)` with within-family structure matrices `C_f = A_ff`, `C_g =
   A_gg` (angular, not squared) and the same `M = A_fg**2`. In POT, `(1-α)` weights direct
   cross-family proximity and `α` weights agreement of within-family geometry. Prespecified grid
   **α = 0.25 / 0.50 / 0.75**, each kept as a separate result — never argmax-ed on baseline
   performance.

## Solver policy

Exact `ot.emd2` (W2) and the standard CG `fused_gromov_wasserstein2` (FGW) are the default and the
only results used for the headline. Entropic solvers (`sinkhorn` / `entropic_fused_gromov…`) exist
**only** as an explicit `--solver entropic --reg <ε>` option for when exact is prohibitive; the
regularization strength and marginal residuals are recorded, and exact/regularized results are
never silently mixed (a metadata `solver.kind` field tags every matrix). Observed exact runtime is
tractable everywhere (human ≈2 s/layer; mammal ≈180 s/layer, dominated by the 2,648-locus
olfactory-receptor FGW), so all shipped results are exact.

## Validation (`--self-test` / pytest)

Hard invariants, asserted: symmetry; finite & nonnegative; ~zero diagonal; **gene-order
invariance**; **deterministic repeatability**; and the solver sanity check **FGW(α=0) reproduces
the Wasserstein-squared objective** given the same cost (observed max diff ~1e-16). FGW **stability
across initializations** is a separate, non-asserted diagnostic: the primary path uses the
deterministic product-coupling init (n_init=1); `--fgw-n-init > 1` restarts CG from seeded random
couplings and records `init_spread` per pair (the CG problem is non-convex, so small spreads are
expected). Note gene-order invariance is a property of the deterministic path only — a seeded
random init is tied to the gene arrangement, so strict invariance is not asserted under restarts.

## Small-family caveat

FGW's quadratic term needs enough genes to resolve within-family geometry. Families with
≤ `TINY_FAMILY_MAX` (=3) genes are **flagged** in `betweenfam_ot_metadata.json` (`tiny_families`),
not dropped: on the **human** panel `nitric_oxide_synthase` (3) and `heme_oxygenase` (2) are tiny,
so any FGW comparison involving them has poorly resolved internal geometry and should be read with
caution. The **mammalian** panel has no tiny families (smallest = heme_oxygenase, 32 loci).

## Optional sensitivity analysis

`subsample_sensitivity()` runs repeated equal-size (taxon/gene-balanced) subsamples and reports the
mean+std of every matrix across repeats — a robustness check for imbalance, *especially* for small
families. The primary result still uses all genes with uniform within-family weights.

## Scoring & outputs (per layer run dir)

- `betweenfam_ot_<name>_distances.csv` — the F×F W2 and FGW-α matrices (family-labelled).
- `betweenfam_ot_metadata.json` — family sizes, preprocessing, POT version, solver settings,
  per-pair convergence (result codes / CG iterations / marginal residuals), tiny-family flags,
  runtime, alphas.
- `between_family_ot_scores.csv` — long form: `approach ∈ {geodesic, wasserstein, fgw_alpha*}` ×
  baseline × {axis, spearman_rho, p_mantel}. The `geodesic` rows reproduce the panel's existing
  `between_family_baseline_scores.csv` **exactly** (Δρ = 0) — a built-in cross-check that the OT
  scoring harness matches the established one.
- `convergent_pair_ranks_ot.csv` — each curated convergent pair's percentile rank per approach.

`ot_between_family_figure.py` overlays the three approaches across layers (one small-multiple per
baseline, filled marker = Mantel p<0.05), the "typical between-family baseline result" per
experiment.

## Coverage of the six requested experiments — all six now exist

| Panel | CDS-only | Full transcript | Transcript + CDS-masking |
|-------|----------|-----------------|--------------------------|
| Human | ✅ `2026-07-15_evo2-human-panel-cds` | ✅ `2026-07-01_evo2-human-panel` | ✅ `2026-07-20_evo2-human-cdspool-transcript` |
| Mammal | ✅ `2026-07-16_mammalian-orthologs-cds` | ✅ `2026-07-16_mammalian-orthologs-transcript` | ✅ `2026-07-16_mammalian-orthologs-transcript_cdsmask` |

**Closed (2026-07-23).** The mammalian CDS-masked-transcript gap described in earlier revisions of
this doc has been filled: `build_cds_masks_mammal.py` builds the per-locus coding-position masks for
all 24 mammals and `embed_cds_masked_mammal.py` re-embedded 4,805 loci / 522 ortholog groups with
CDS-masked pooling into `data/cache/mammal_embed/transcript_cdsmask/`. OT is scored at all 32 layers
under `results/2026-07-16_mammalian-orthologs-transcript_cdsmask/` (and its `_400` variant), so this
is **6 figures**, not 5. Note `mammal_embed.py --arm` itself still offers only `{transcript, cds}` —
the CDS-masked arm is a separate embedder, consumed by the sweep as `arm="transcript_cdsmask"`.

**Panel-size coverage.** Beyond the six input cells above, OT is also scored on the **47-family human
panel** (`human-cds-48fam` → `results/2026-07-22_evo2-human-panel-cds`, all 32 layers). This is the
cell the metric comparison most needs: with only ~9 family nodes the centroid k-NN graph is
near-complete, so geodesic ≈ direct angular distance and the three approaches have little room to
disagree. 48 family nodes is where a centroid-free metric can actually differ from the centroid one.
**Still uncovered:** the cross-kingdom KEGG panel (`2026-06-30_evo2-gene-families`).
