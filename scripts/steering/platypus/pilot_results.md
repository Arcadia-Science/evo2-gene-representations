# Pilot results — conservation-stratified platypus panel (n = 100)

**Document status:** historical pilot record. Use `results_n400.md` for production conclusions and
`experiments/exp2_platypus_steering.sh` for the active workflow.

Everything measured to date. The method these runs implement is in `conservation_stratified_design.md`,
which is kept free of results on purpose. Where a design choice was made *because* of a number below,
the design doc states the choice and points here for the evidence.

**The n = 400 production run is done and reported separately in `results_n400.md`**, which stands on
its own and does not depend on anything here. Where the two disagree, the n = 400 file is the one to
use — this document is a 100-gene pilot.

**Power warning that applies to every section.** The pilot is 100 genes, 20 per stratum — the *same*
ρ ≈ 0.28 detection floor as the 2026-08-04 null it was built to escape. It was designed to test the
machinery, not the continuous H1/H2 hypotheses later evaluated in the n = 400 run.

## Run index

| run | what |
|---|---|
| `results/2026-08-05_platypus-strat-pilot/` | 100-gene codon-0 panel; stages 0–5 complete |
| `results/2026-08-06_platypus-strat-shifted/` | 100-gene shifted-window replication, disjoint from the above; stages 1–4 (`cds_mean` / L27 only); unsteered baseline regenerating |

---

## Stage 0 — pool and stratification

`results/2026-08-05_platypus-strat-pilot/stage0/` (`strat/stage0_pool.py`)

| file | what |
|---|---|
| `stage0_config.json` | seed 20260805, Ensembl 116, stratum edges 64.68 / 72.89 / 80.48 / 88.09 |
| `biomart_platypus_homologs.tsv` | 15,926 human genes with a platypus homolog |
| `pool_with_blocks.csv` | 11,135 one2one hi-conf → **6,961 blocks** (5,027 singletons, max 40) |
| `mmseqs/` | all-vs-all clustering at 30 % id / 50 % cov |
| `frozen_order.csv` | the pre-committed draw, 1,392–1,393 blocks per stratum |

15,926 human genes have a platypus homolog → **11,135** one2one, high-confidence, uniquely-mapping
(against 512 in the 48-family panel, of which 103 survived QC).

## Stage 1 — QC

`<pilot>/stage1/` (`strat/stage1_qc.py`): `pairs.csv` (100 genes, 20/stratum),
`cds_{human,platypus}.fasta`, `attrition.csv`, `stage1_summary.json`.

100 kept of 325 examined. Pass rate **20.6 / 24.1 / 29.0 / 51.3 / 54.1 %** fastest → most conserved —
a **2.6× gradient**, with `indel_within_prefix` accounting for 151 of 225 rejections. Human CDS length
216 / 1,434 / 14,574 bp (min / median / max).

**The shifted-window arm.** `qc_pair_window()` exposed as `--max-start-codon N` and `--recovered-only`.
`--max-start-codon 0` delegates to `qc_pair`, verified bit-identical (0/100 mismatches), and the
shifted rule rejects 0/100 codon-0 passers.

*Result:* **67 %** of position-failures recovered (85/127 blocks examined by both gates), at a roughly
**flat** rate across strata (0.65 / 0.54 / 0.74 / 0.83 / 0.71). So the 2.6× production pass-rate
gradient is mostly a *position* artifact, not a sequence-quality one. The recovered panel is 100 genes,
20/stratum, **overlap with the production panel = 0**; displaced reasons 69 `indel_within_prefix` /
31 `prefix_not_homologous`.

*Caveat:* `--max-start-codon` bounds the **human** window start only. The platypus counterpart can
begin much later where a large insertion intervenes (median 57 bp, max 576 bp).

→ This is why the shifted-window rule is **primary** at n = 400.

## Stage 2 — direction geometry

`<pilot>/stage2/`: `pooled_representations.npz` (100 × 2 species × 32 blocks × 4096, both pooling modes
+ `prefix_last`, 180 MB), `aligned_coverage.csv` (`retained_frac` min 0.529 / med 0.904 / max 0.988),
`pooled_norms.csv`. Geometry from `delta_stats.py` → `geom_cds_mean/`, `geom_aligned_mean/`.

**Blocks 28–31 are unusable, not merely flagged.** ‖v‖ runs 5.3 at block 27 → 1.1e3 at 28 → 1.0e6 at
29 → 8.6e11 at 30, and 31 is byte-identical to 30. Layer selection is confined to ≤ 27.

**Genes in the same stratum deviate from the consensus *together*, not just more.** A binned test
(since retired to `attic/stratum_geometry.py` in favour of H1c) measured
*mean cos(adjacent strata) − mean cos(strata ≥ 3 apart)* = **+0.116, p = 0.0005** at L27 `cds_mean`;
GC-removed +0.144, PC1-removed +0.160; `aligned_mean` +0.094, p = 0.0005. Off-diagonal cosine 0.90
against a split-half ceiling of 0.94.

It is **not** an artifact of fast genes simply being noisier. Under a null that keeps each gene's
residual norm and shuffles residual *directions* across genes, the contrast is +0.004 and the observed
value still lands at p = 0.0005; per-stratum split-half ceilings are near-flat
(0.929 / 0.929 / 0.949 / 0.935 / 0.943), and residual norms vary only 5.89 → 4.53 fastest → conserved.

→ Real structure, but the statistic presumes shared within-stratum deviation instead of testing it,
which is why H1c replaces it. This is the evidence that H1c has something to find.

## H1c — directional clusters (`strat/h1c_clusters.py`, `<pilot>/h1c_clusters/`)

Residual cloud, `cds_mean` L27: pairwise residual-cosine sd **0.260** against **0.016** for isotropic
residuals at matched norms, and a **participation ratio of 13.0** on 100 genes.

**Existence — PASS.** Max-over-k silhouette excess **+0.058 at k\* = 7, global p = 0.0040** against the
spectrum-matched Gaussian null (500 replicates), which is the max-over-k comparison so searching
k ∈ 2…8 costs nothing.

| k | sizes | silhouette | spectrum null | excess | p at k | isotropic null |
|---|---|---|---|---|---|---|
| 2 | 30, 70 | +0.201 | +0.192 ± 0.020 | +0.009 | 0.33 | +0.003 |
| 3 | 30, 13, 57 | +0.158 | +0.153 ± 0.017 | +0.004 | 0.41 | +0.003 |
| 4 | 30, 13, 12, 45 | +0.159 | +0.137 ± 0.018 | +0.022 | 0.12 | +0.003 |
| **5** | 30, 13, 12, 21, 24 | +0.185 | +0.128 ± 0.019 | **+0.057** | **0.0020** | +0.003 |
| **6** | 20, 10, 13, 12, 21, 24 | +0.181 | +0.123 ± 0.019 | **+0.058** | **0.0020** | +0.003 |
| **7** | 20, 10, 13, 12, 21, 16, 8 | +0.179 | +0.120 ± 0.018 | **+0.058** | **0.0020** | +0.004 |
| 8 | 20, 2, 8, 13, 12, 21, 16, 8 | +0.168 | +0.120 ± 0.018 | +0.049 | 0.0020 | +0.004 |

Two things to read off this table. **The structure is fine-grained, not coarse**: there is no
significant 2-, 3- or 4-way split, and the signal only appears at k ≥ 5. And **the isotropic null is
useless here** — it sits at +0.003 against +0.20 observed, so a naive test would have declared clusters
at every k. Almost all of the apparent coarse structure is low-rank anisotropy, and only the
spectrum-matched null can see past it.

**Stability — PASS.** Block-bootstrap ARI **0.608** [q05 0.440], disjoint-half ARI **0.587** at k = 7.
(The failed k = 2 partition was markedly less stable, 0.528 / 0.502.)

**Membership.** Cluster means, clusters in index order, at k = 7 (n ≈ 8–21 each):

| variable | c0 | c1 | c2 | c3 | c4 | c5 | c6 | Kruskal p |
|---|---|---|---|---|---|---|---|---|
| `perc_id_hp` | 80.9 | 64.3 | 64.8 | 69.3 | 84.0 | 80.5 | 85.1 | 1e-5 |
| `tree_len` | 1.11 | 2.69 | 2.35 | 2.50 | 1.65 | 2.38 | 2.30 | 1e-5 |
| `diameter` | 0.46 | 1.15 | 0.90 | 0.95 | 0.63 | 1.38 | 1.62 | 1e-5 |
| `background_rate` | 0.90 | 2.11 | 2.00 | 2.18 | 1.52 | 2.21 | 2.17 | 0.0004 |
| `platypus_branch` | 0.20 | 0.56 | 0.35 | 0.30 | 0.12 | 0.17 | 0.13 | 1e-5 |
| `dN_hp_yn` | 0.109 | 0.221 | 0.207 | 0.169 | 0.093 | 0.123 | 0.078 | 1e-5 |
| `dS_hp_yn` | 1.89 | 1.97 | 1.01 | 1.95 | 1.30 | 1.05 | 0.82 | 0.0001 |
| `omega_hp_yn` | 0.085 | 0.130 | 0.223 | 0.098 | 0.094 | 0.130 | 0.110 | 0.0022 |
| `treeness` | 0.178 | 0.157 | 0.258 | 0.223 | 0.221 | 0.193 | 0.196 | 0.080 |
| `gc3_div` *(desc.)* | 0.280 | 0.172 | 0.077 | −0.009 | −0.039 | 0.112 | 0.110 | — |
| `delta_norm` *(desc.)* | 9.05 | 9.36 | 6.95 | 6.99 | 6.35 | 5.47 | 5.89 | — |
| `cds_len_human` *(desc.)* | 2695 | 952 | 2234 | 2791 | 2788 | 1674 | 1637 | — |

The clusters are strongly rate-structured but **not** a simple conservation ordering: c1/c2/c3 are the
fast end (`perc_id_hp` 64–69) and c0/c4/c5/c6 the conserved end (80–85), yet within each group the
clusters differ on *which* rate axis separates them — c2 is high-ω/low-dS, c6 is high-diameter/low-dN,
c1 has the largest platypus terminal branch. `delta_norm` also tracks the split (9.4 → 5.5).

**Caveats.** k = 7 on n = 100 leaves ~14 genes per cluster, several at 8–13, so membership is
suggestive rather than established, and part 3 was declared exploratory. Tier C needs only k = 3–5
genes per direction, so the sizes are workable, but n = 400 is what would make this solid. Nothing here
is a claim until H2c validates it causally.

*Two null-design errors were found by running this and are recorded in the script's docstring: the
norm-preserving direction-shuffle null is a no-op for angular statistics, and Tibshirani's 1-SE gap
rule selects k = 2 here while the signal is at k ≥ 5. See `design_v2_migration.md` § B.*

`aligned_mean` gives a *different* k = 2 partition from `cds_mean` (52/48 vs 30/70) — recorded because
it is the kind of instability the stability gate exists to catch, though k = 2 is not the selected k.

## Stage 3 — layer freeze

`<pilot>/stage3_cds_mean/`, `stage3_aligned_mean/`: `loo_vectors.npz`, `loo_diagnostics.csv`,
`layer_candidates.csv`, `stage3_config.json`.

**The layer freeze is `blocks.27` / `cds_mean`**, recorded before any stage-4 generation per the
ordering safeguard. Grounds from `layer_candidates.csv`: `loo_cos` median **0.785** with **100 %** of
genes positive (the only layer ≤ 27 to reach 1.00), `frac_on_cone` 0.006, `rel_norm` 0.34,
`abs_cos_gc` 0.45. Blocks 24–26 are runners-up (0.61 / 0.66 / 0.71) and were carried into the band arm.

**The free pre-check gates were never run**, so tiers C and D were never licensed. Neither the cosine
contrast at feasible *k* nor the argmax-layer spread is implemented; gate 4 had no input.

## Stage 4 — generation and scoring

`stage4_{cds_mean,aligned_mean}_blocks27` and `..._blocks25_26_27`, each with `stage4_scores.csv`,
`generations.jsonl.gz`, `scoring_plan.csv`, `stage4_config.json`, `analysis_summary.csv`,
`analysis_gradient.csv`. Shared unsteered baseline `<pilot>/unsteered_seed.csv` (mean
`pct_private_correct` 39.62).

**Only tier A ran** — `unsteered / add α=1 / add_own / random α=1`. No dose-response, no
`cone_removed` / `gc_removed` / `cross_gene`, no H2c, no per-gene peak layer. The gate-5 kill did not
fire: aa cost −1.0 to −2.3 pp, and premature stops *fell* by 1.5–2.1.

| arm | Δ private-bp vs unsteered | vs norm-matched random | Δ aa |
|---|---|---|---|
| `cds_mean` L27 α=1 | +2.06 [1.28, 2.85] | +2.14 | −1.03 |
| `cds_mean` L27 α_own | **+2.78** [1.77, 3.84] | +2.86 | −1.37 |
| `aligned_mean` L27 α=1 | +2.10 [1.27, 2.97] | +1.23 | −1.66 |
| `cds_mean` 25+26+27 α=1 | +1.49 [0.66, 2.34] | +1.67 | −1.16 |
| `aligned_mean` 25+26+27 α=1 | +1.29 [0.52, 2.03] | +0.76 | −1.30 |

**A multi-layer band arm, not in the original design.** `--layers blocks.25 blocks.26 blocks.27`
injects each layer's own `v_-i` simultaneously with α split 1/3 so the *summed* α matches a
single-layer run. This equalises summed α, not summed residual-relative dose — ‖v_-i‖ grows ~1.4 / 2.4
/ 5.3 across those layers, so the band delivers less total perturbation. **Spreading the dose costs
30–40 % of the effect**; single-layer L27 wins.

**The `aligned_mean` random arm is not clean** (+0.87 pp, p = 0.048 at L27; +0.53, p = 0.011 in the
band) where `cds_mean`'s sits at zero as it should. → One of the reasons pooling collapsed to
`cds_mean`.

**Shifted-window replication** — `results/2026-08-06_platypus-strat-shifted/`, `cds_mean` / L27 only,
99 usable of the 100 recovered genes (`ENSG00000183036` dropped, < 20 sites). Steered vs norm-matched
random: `add` α=1 **+3.31 pp** [2.14, 4.48] p = 2.5e-7; `add_own` **+4.32 pp** [3.00, 5.73] p = 3.1e-9
— *larger* than the production panel's +2.14 / +2.86. The headline steering effect is not an artifact
of the codon-0 prefix gate. Stage-2 `retained_frac` median 0.899 vs 0.904, so the recovered genes are
not alignment junk.

Stage 4 also gained a **per-gene prompt offset** for the shifted panel: `off_p + PREFIX_BP` (end of the
forced region in target coordinates) cuts the continuation and re-keys diagnostic sites; `off_h` places
the human prompt. Codon-0 panels take offset 0 and are unaffected.

## Stage 5 — evolutionary rate

`<pilot>/stage5/`: `ortholog_resolution.csv`, `presence_matrix.csv`, `core_candidates.csv`,
`cds_per_gene.csv`, `seqs/`, `trees/`, `tree_stats.csv` (**100/100 ok**, n_taxa 10 / 23 / 24), `dnds/`,
`dnds.csv` (**96/100**), `dnds_branches.csv`, `rate_vs_direction.csv`, `rate_vs_gain.csv`.

**The free-ratio codeml arm is dead — zero usable genes.** 71 `skipped_unidentifiable`
(< 20 codons/branch), 22 `timeout(3600 s)`, 3 `skipped_too_long`. No `*_fr` predictors exist, so
per-branch dS that is not a rescaled branch length — the one quantity that would make the dS control
clean — is unavailable. Also 4/100 genes timed out of codeml entirely and 1 yn00 failed.

### G3 is answered: the 2026-08-04 null was range restriction

Rate predicts geometry on this panel — `diameter` vs `loo_cos` ρ = **−0.47** (p = 1e-6), `tree_len`
−0.39, `background_rate` −0.38; `focal_residual` vs ‖D_i‖ **+0.49** (p = 1.8e-7) — all surviving
partials on `aln_len` / `n_taxa` / log CDS length. Fast genes are less well described by the consensus
direction and carry larger difference vectors. `human_residual`, the symmetric control, stays null.

### The dS negative control FAILS on geometry

dS outpredicts dN and every ω: ‖D_i‖ vs `dS_hp_yn` ρ **+0.539** (p = 1.8e-8) against `dN_hp_yn` +0.356
and all ω ≈ +0.10; `loo_cos` vs `tree_dS_m0` −0.392 against `dN_background` −0.318. The geometry tracks
*how much sequence has changed*, with synonymous divergence carrying more of it than nonsynonymous.
This sits badly against the compositional-controls result (synonymous-recode ρ +0.315 ≫ GC floor
+0.118) and is the most important open problem in the design.

**It is not explained by dS co-scaling with total divergence.** `dS_hp_yn` vs `dN_hp_yn`
ρ = **+0.186 (p = 0.071)**; vs `tree_len` ρ = **+0.163 (p = 0.11)**. dS is near-orthogonal to both
amino-acid change and total tree length — it is its own axis. The surviving caveat is saturation
(human–platypus dS ≈ 1.0–1.6, though `dS_saturated` flags 0/96), which is why `dS_background_yn` and
`tree_dS_m0` are reported alongside; on *gain* those two are flat (p = 0.19 / 0.31) while `dS_hp_yn` is
not, localising the problem to the saturated pairwise estimate. On *geometry* there is no such let-off.

### Rate-statistic correlation structure

The 19 stage-5 statistics are not 19 independent things. Spearman correlation matrix, n = 95:
**effective dimension 3.11**, Kaiser 4, PC1–3 = 81 % of variance.

| axis | var | top loadings | reads as |
|---|---|---|---|
| PC1 | 52 % | `mean_target_dist`, `focal_target_dist`, `tree_dN_m0`, `dN_hp_yn`, `omega_m0` | amino-acid divergence |
| PC2 | 20 % | `tree_dS_m0`, `dS_background_yn`, `dS_hp_yn` (ω opposite sign) | synonymous divergence |
| PC3 | 10 % | `focal_residual`, `platypus_branch` | platypus-lineage acceleration |
| — | — | `treeness` (ρ = 0.053 with `tree_len`) | rate heterogeneity / tree shape |

Other cross-correlations: `omega_hp_yn` vs `dN_hp_yn` +0.754; `dS_background_yn` vs `dS_hp_yn` +0.592;
legacy stats 1–4 intercorrelate +0.51…+0.96.

→ This is what the four confirmatory predictors were declared on.

## Gate 4 — the two Stage-3 pre-checks (`strat/stage3_gates.py`, `<pilot>/stage3_gates/`)

**Tier C — LICENSED.** Cluster-matched vs cross-cluster panel directions, both excluding the focal
gene, both norm-matched. `contrast` = cos(random, random) − cos(same-cluster, other-cluster); positive
means the H1c split carries directional information.

| k | cos(same, other) | cos(random, random) | contrast |
|---|---|---|---|
| 3 | 0.696 | 0.773 | +0.077 |
| 5 | 0.756 | 0.854 | +0.098 |
| 8 | 0.786 | 0.904 | +0.118 |
| 16 | 0.818 | 0.950 | +0.133 |

Note the saturation the gate exists to catch: by k = 16 *both* arms are above 0.81 from each other, so
the arm must run at k = 3–8. Combined with H1c's own gates passing, **Tier C is licensed at k = 5**.

**Tier D — NOT LICENSED.** Per-gene argmax layer of `loo_cos` inside the candidate band L24–27:
**88 % of genes peak at the frozen layer L27** (histogram 24:2, 25:2, 26:8, 27:88), only 12 genes
disagree. The arm would be a re-run of the frozen-layer condition at full price. **~1.9 h of A10G
saved by a CPU check.**

*Trap recorded:* the unrestricted argmax over all layers gives a completely different and completely
spurious answer — 36 genes "peak" at layer 0, 28 at layer 9. `loo_cos` is scale-free, so the embedding
layer scores median 0.758 with median ‖D_i‖ = **0.008**, against 0.785 with ‖D_i‖ = **6.83** at L27.
Human and platypus CDS land on nearly the same mean token embedding and the sliver left over points the
same way for every gene because it is shared codon composition. Layers 0–8 are all this regime. The
gate must search the candidate band only; the unrestricted histogram is a diagnostic, never the gate.

## G1 geometry vs rate, per gene

Block-held-out `loo_cos` (LOO ≡ leave-family-out on this panel), at L27:

| | `cds_mean` | `aligned_mean` |
|---|---|---|
| `loo_cos` vs `perc_id_hp` | ρ = +0.183, p = 0.068 | ρ = +0.202, p = 0.043 |
| ‖D_i‖ vs `perc_id_hp` | ρ = −0.310, p = 0.0017 | ρ = −0.257, p = 0.0099 |

`loo_cos` by stratum, fastest → most conserved: **0.701 / 0.723 / 0.776 / 0.754 / 0.723** — a hump,
peaking in the middle strata, exactly like the steering gain below. The monotone ρ understates it.

### H1a / H1b under the pre-registered shape statistics

`rate_vs_direction_shape.csv`, `cds_mean` L27, four confirmatory axes × three shape statistics, Holm
within each hypothesis family. **Splitting orientation from magnitude was the right call: they behave
oppositely.**

**H1b (magnitude, ‖D_i‖) — all four axes survive Holm, and it is purely LINEAR.**

| predictor | linear | p | quadratic | p | mid−ext | Holm |
|---|---|---|---|---|---|---|
| `focal_residual` | **+1.027** | 1.3e-5 | −0.051 | 0.44 | −0.45 | **5.2e-5** |
| `dS_hp_yn` *(control)* | **+0.884** | 0.0020 | −0.115 | 0.36 | −0.39 | **0.0059** |
| `dN_hp_yn` | **+0.606** | 0.0027 | −0.270 | 0.13 | −0.39 | **0.0059** |
| `treeness` | **−0.408** | 0.029 | −0.039 | 0.75 | −0.45 | **0.029** |

No curvature, no mid-vs-extreme effect: faster genes simply carry proportionally larger difference
vectors. **The dS control fails here too** — +0.884 against dN's +0.606, on the same genes.

**H1a (orientation, `loo_cos`) — only two axes survive, and the significant term is the contrast, not
the slope.** `focal_residual` on the PC1-removed outcome (Holm 0.050; mid−ext +0.092, p = 0.017) and
`treeness` (Holm 0.042; linear +0.051, p = 0.0052). `dN_hp_yn` and `dS_hp_yn` are nominal only.

So the hump is an **orientation** phenomenon and magnitude is a clean monotone one. A single combined
H1 would have averaged these two into mush.

## G2 steerability vs rate

The GLMM was replaced by a **meta-regression** (`strat/stage5_gain.py`): statsmodels is unavailable
here and a hand-rolled GLMM is not worth the correctness risk, and it is not needed — every steered
cell is compared to the unsteered cell of the *same* gene at the *same* sites, so the gene random
effect differences out exactly, leaving a per-gene effect size plus its sampling variance.

```
y_i   = logit(p_steered,i) − logit(p_unsteered,i)
var_i = phi · [ 1/(n_s p_s q_s) + 1/(n_u p_u q_u) ]        delta method, overdispersed
y_i   = b0 + b1·z(rate_i) + e_i,  weights 1/var_i          inverse-variance weighted LS
```

`phi` is estimated from between-sample scatter within each (gene, condition) cell against its binomial
expectation, absorbing both dependence problems (sites within a generation are correlated; samples
share a prompt). Measured **phi = 1.76–2.01** — nominal trials were ~2× more precise than they are.

**Results (n = 100, pilot power).** `b0` confirms steering at z = +6.8 (L27 α=1) to +10.3 (L27 α_own);
band arm z = +4.0 to +6.7. **Nothing survives Bonferroni** (3.5e-4 over 144 tests). But 31/144 are
nominally p < 0.05 against 7.2 expected, with consistent signs: `background_rate`, `tree_len`, dN and ω
all **negative** — faster genes steer *less* well, best p = 0.004–0.008. Best ω is `omega_hp_yn`
−0.054/SD (p = 0.004); best dN −0.052/SD (p = 0.025).

**On gain, the dS control fails only in its saturated arm.** Strongest single predictor is `dS_hp_yn`,
+0.068/SD (p = 1.6e-4), opposite in sign to the dN/ω effects — but `dS_background_yn` (p = 0.19) and
`tree_dS_m0` (p = 0.31) are flat.

### The effect is non-monotonic

Mean gain in pp by stratum, `cds_mean` L27 (0 = fastest, 4 = most conserved):

| arm | s0 | s1 | s2 | s3 | s4 | Spearman vs `perc_id_hp` |
|---|---|---|---|---|---|---|
| `add` α=1 | +0.94 | +1.65 | **+3.57** | **+3.85** | +0.28 | ρ = 0.045, p = 0.66 |
| `add_own` | +1.00 | +2.65 | **+4.52** | **+4.45** | +1.28 | ρ = 0.029, p = 0.77 |
| `random` | −0.27 | +0.57 | −0.55 | +0.57 | −0.72 | ρ = −0.040, p = 0.69 |

The rise s0→s3 and the fall s3→s4 cancel, so Spearman reports ρ ≈ 0 while the middle strata really do
gain ~3 pp more than the edges. Mid (2,3) vs extremes (0,4) differs by **+2.4 to +3.3 pp** across all
8 arm × condition combinations, MWU p = 0.0003 to 0.028 (7 of 8 at p < 0.03); the inverse-variance
weighted versions agree (+2.6/+5.1 mid vs −0.1/+1.5 extremes).

This was **post hoc** on the pilot. The reading to test, not to assert: nothing to steer toward at the
conserved end, direction does not transfer at the fast end.

### H2a re-fitted with the pre-registered quadratic

`rate_vs_gain_confirmatory.csv` + `gain_mid_vs_extreme.csv`. **The quadratic finds what the linear term
cannot.** Every linear term is dead flat (p = 0.72–0.99) while the curvature is real and correctly
signed — negative for dN, i.e. a hump:

| arm / condition | predictor | linear | p | quadratic | p | Holm |
|---|---|---|---|---|---|---|
| L27 `add_own` | `dN_hp_yn` | −0.002 | 0.94 | **−0.060** | 0.0086 | **0.034** |
| L27 `add_a1.0` | `dN_hp_yn` | −0.000 | 0.99 | −0.036 | 0.036 | 0.14 |
| L27 `add_own` | `dS_hp_yn` *(control)* | +0.013 | 0.72 | +0.031 | 0.042 | 0.13 |
| band `add_a1.0` | `dS_hp_yn` *(control)* | −0.010 | 0.73 | +0.029 | 0.021 | 0.084 |

dS's curvature is *opposite in sign* to dN's and does not survive Holm — the first place in the whole
pilot where the negative control behaves as a negative control should.

Mid-vs-extreme, now the pre-registered third statistic rather than a post-hoc note:

| arm | condition | mid | ext | diff | MWU p | weighted |
|---|---|---|---|---|---|---|
| L27 | `add_a1.0` | +3.71 | +0.61 | **+3.09** | 0.0006 | +3.82 vs +0.81 |
| L27 | `add_own` | +4.48 | +1.14 | **+3.34** | 0.012 | +5.11 vs +1.16 |
| band | `add_a1.0` | +2.89 | +0.37 | **+2.52** | 0.0062 | +2.71 vs +0.07 |
| band | `add_own` | +3.65 | +0.80 | **+2.85** | 0.0053 | +3.74 vs +1.05 |

→ Geometry and causal outcome peak together in the middle strata, which is why H1a, H1b and H2a are
all pre-registered at n = 400 with linear + quadratic + mid-vs-extreme rather than a monotone test.

## Gate status

| gate | status |
|---|---|
| 1 after stage 0 | **passed** — 1,392–1,393 blocks per stratum, far above the 200 floor; 5 strata kept |
| 2 after stage 1 | **n/a at pilot n** — the floors are written for the 400-gene draw; the pilot filled 20/stratum everywhere |
| 3 after stage 2 | **passed** — panel split-half 0.87–0.94 at L24–27; the H1c existence half also passes retrospectively |
| 4 after stage 3 | **never evaluated** — neither pre-check implemented, so tiers C and D were not licensed |
| 5 stage 4 leading genes | **did not fire** — aa cost −1.0 to −2.3 pp, premature stops fell |

## Code

`scripts/steering/platypus/strat/` — `stage0_pool.py`, `stage1_qc.py`, `stage2_embed.py`,
`h1c_clusters.py`, `stage3_gates.py`, `stage4_steer.py`, `stage4_analysis.py`, `stage5_orthologs.py`,
`stage5_trees.py`, `stage5_dnds.py`, `stage5_merge.py`, `stage5_gain.py`, `layer_table.py`; reusing
`../delta_stats.py`, `../figures.py`, `../stage3_select.py`. Retired: `attic/stratum_geometry.py`.

## Still outstanding

- **Tier B** dose-response (α = 0.5/2) and confound arms — `gc_removed`, `cluster_panel` and per-gene
  layer steering are **built and verified but deliberately unrun**, held for the n = 400 experiment.
- **Shifted-window unsteered baseline** and its per-stratum replication — the mid-vs-extreme
  replication on that panel depends on it, and that contrast is now a pre-registered test.
- **Nucleotide-tree fallback** for `skip_conserved` genes.
- **Free-ratio codeml**, dead at pilot scale; clean per-branch dS is the one measurement that would
  make the failing dS control interpretable.
- ~~**The n = 400 draw**~~ — run on 2026-08-08, `results/2026-08-08_platypus-strat-400/`, written up in
  `results_n400.md`. Tier B dose and confound arms ran there; free-ratio codeml was not retried.
