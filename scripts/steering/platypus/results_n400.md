# Results — conservation-stratified platypus panel (n = 400)

**Document status:** production result record for Experiment 3. Figure reproduction is maintained by
`experiments/exp3_platypus_steering.sh`; this file preserves the detailed stage-level interpretation.

Everything measured on `results/2026-08-08_platypus-strat-400/`. Method: `conservation_stratified_design.md`
(kept results-free). Each stage below ends with the hypotheses it feeds; § Hypothesis index is the map.

**The run.** 400 human↔platypus 1:1 orthologs, one gene per mmseqs homology block (so leave-one-out =
leave-family-out), 80 per stratum across five quintiles of Compara protein identity. Shifted-window QC
(`--max-start-codon 30`), pooling `cds_mean`, steering at `blocks.27`, additive operator, 1,000 bp at
temperature 0.7. All four stage-4 condition blocks ran on all genes. 08-08 00:02 → 08-09 12:28 (stage 4
is 36 h A10G), plus 6.6 h for H2c run afterwards because its gate did not pass.

**Three headline results:**

1. Steering works with a clean null: **+3.11 pp** private-bp at α = 1 (z = +18.1), random arm flat at
   every dose (−0.09 / −0.03 / −0.13 pp).
2. **Projecting the GC axis out of the steering vector removes 74 % of that effect** (+3.11 → +0.81 pp);
   removing the anisotropy axis costs nothing.
3. **The dS negative control fails on both geometry and gain.** Some dS statistic matches or beats the
   best dN statistic on every geometry outcome, and `dS_background_yn` is the strongest gain predictor
   in the run, surviving Bonferroni where `dN_hp_yn` is dead.

2 and 3 point the same way, and against the protein-level account of what this direction encodes.

---

## Stage 0 — pool and stratification

`stage0/` — symlink to the shared pool built under seed **20260805**, Ensembl **116**
(`strat/stage0_pool.py`). The panel is the first 80 QC-passers per stratum in the frozen order, so no
replacement decision was possible after QC began.

| file | what |
|---|---|
| `stage0_config.json` | seed 20260805, Ensembl 116, stratum edges 64.68 / 72.89 / 80.48 / 88.09 |
| `biomart_platypus_homologs.tsv` | 15,926 human genes with a platypus homolog |
| `pool_with_blocks.csv` | 11,135 one2one high-confidence → **6,961 blocks** (5,027 singletons, max 40) |
| `mmseqs/` | all-vs-all clustering at 30 % identity / 50 % coverage |
| `frozen_order.csv` | the pre-committed draw, 1,392–1,393 blocks per stratum |

**Gate 1 passes** — 1,392+ blocks per stratum against a floor of 200; five strata kept.

*→ sampling frame only; `perc_id_hp` is not a confirmatory predictor anywhere. What the draw
actually produced per stratum: figure `figures/8_strata_composition`.*

## Stage 1 — paired dataset and QC

`stage1/` (`strat/stage1_qc.py --max-start-codon 30`): `pairs.csv`, `cds_{human,platypus}.fasta`,
`attrition.csv`, `stage1_summary.json`.

**400 kept of 558 examined**, exactly 80 per stratum. Human CDS length 189 / 1,440 / 14,574 bp.

| stratum (fastest → most conserved) | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| blocks examined | 136 | 127 | 108 | 92 | 95 |
| pass rate | 58.8 % | 63.0 % | 74.1 % | 87.0 % | 84.2 % |
| `perc_id_hp` min / med / max | 50.2 / 58.8 / 64.5 | 64.7 / 68.8 / 72.8 | 72.9 / 76.7 / 80.5 | 80.7 / 84.0 / 88.0 | 88.1 / 92.1 / 100 |

Pass rate still rises **1.43×** with conservation, so QC is not conservation-neutral even under the
shifted rule: every conclusion is conditional on having an indel-free homologous 30-codon window inside
the first 60 codons. Rejections (158): `window_starts_too_late` 108, `no_start_codon_platypus` 48,
`no_start_codon_human` 2.

**169 genes take a non-zero human prompt offset** (max 87 bp); stage 4 counts 213/400 non-zero on either
side. Open caveat unchanged: `--max-start-codon` bounds the *human* start only, so the platypus window
can begin later where an insertion intervenes.

**Gate 2 passes** — 400 post-QC against a floor of 250; fastest stratum 80 against a floor of 40.

*→ the required attrition-by-stratum reporting: `stage1/attrition.csv`, `stage1_summary.json`;
figure `figures/1_qc_attrition`.*

## Stage 2 — direction geometry

`stage2/` (`strat/stage2_embed.py`): `pooled_representations.npz` (400 × 2 × 32 × 4096, both pooling
modes, **754 MB**), `aligned_coverage.csv`, `pooled_norms.csv`. Statistics from `../delta_stats.py` →
`geom_cds_mean/{layer_stats,per_gene_by_layer}.csv`, `mean_vectors.npz`, `null_distributions.npz`.

12 genes exceeded 6 kb and were tiled (6 kb windows, 1 kb leading context discarded). `retained_frac`
0.485 / 0.900 / 0.996.

**Strong shared direction at `blocks.27`:**

| statistic | value | null |
|---|---|---|
| split-half cos(v_A, v_B), disjoint gene halves | **0.996** [0.992, 0.998] | 0 |
| directional coherence C | **0.741** | isotropic floor 0.05 |
| `loo_frac_pos` | 0.9975 (399/400) | — |
| ‖v‖ | 5.63 | — |

**Blocks 28–31 are unusable**: ‖v‖ 1.19e3 → 1.01e6 → 8.84e11, block 31 byte-identical to 30. Coherence
and `loo_cos` peak spuriously there, so layer selection is confined to ≤ 27.

**`aligned_mean` was embedded but never analysed**, so the post-hoc robustness pass on `cds_mean`'s
positional non-correspondence has not happened.

**Gate 3, first half, passes** — split-half never below ~0.99 in L20–27, against a floor of 0.3.

*Null caveat: the mismatched-pair null reproduces the observed `loo_mean` to 15 digits (p = 1.0 at L27) —
the documented degeneracy in `delta_stats.py`, since permuting pairings leaves v bit-identical. Read the
sign-flip and mismatch-pool columns for `loo`.*

*→ supplies `loo_cos` and ‖D_i‖ for H1a/H1b, and the residual cloud for H1c:
`geom_cds_mean/per_gene_by_layer.csv`, `layer_stats.csv`; figures
`geom_cds_mean/figures/1–8` (per-layer statistics and their nulls).*

## H1c — directional clusters

`strat/h1c_clusters.py` → `h1c_clusters/`. `cds_mean` L27, average linkage on angular distance,
k ∈ 2…8, 500 null and 500 bootstrap replicates. Residual cloud: pairwise cosine sd **0.2625** against
**0.0156** isotropic at matched norms, participation ratio **14.04**.

**Part 1, existence — PASS.** Max-over-k silhouette excess **+0.0568 at k\* = 7, p = 0.0020** vs the
spectrum-matched Gaussian null.

| k | sizes | silhouette | spectrum null | excess | p at k | isotropic null |
|---|---|---|---|---|---|---|
| 2 | 254, 146 | 0.190 | 0.175 ± 0.013 | +0.015 | 0.12 | 0.0013 |
| 3 | 254, 37, 109 | 0.145 | 0.124 ± 0.012 | +0.021 | 0.032 | 0.0008 |
| 4 | 77, 177, 37, 109 | 0.150 | 0.103 ± 0.013 | +0.047 | 0.0020 | 0.0007 |
| 5 | 77, 30, 147, 37, 109 | 0.128 | 0.091 ± 0.013 | +0.037 | 0.0040 | 0.0007 |
| 6 | 77, 30, 147, 37, 8, 101 | 0.116 | 0.083 ± 0.013 | +0.034 | 0.0040 | 0.0007 |
| **7** | 77, 30, 98, 49, 37, 8, 101 | 0.134 | 0.077 ± 0.014 | **+0.057** | **0.0020** | 0.0007 |
| 8 | 77, 30, 98, 49, 8, 29, 8, 101 | 0.125 | 0.073 ± 0.014 | +0.051 | 0.0020 | 0.0008 |

The structure is fine-grained — no significant 2-way split (p = 0.12), signal only from k ≥ 4 — and the
isotropic null is useless, sitting at 0.0007 against 0.19 observed. Most apparent coarse structure is
low-rank anisotropy that only the spectrum-matched null sees past.

**Part 2, stability — FAIL.** Block-bootstrap ARI **0.4846** [q05 **0.3274**], disjoint-half ARI
**0.4559** at k = 7. The cloud has real low-rank structure, but k = 7 hard clusters are not a
description of it that survives resampling. → **Tier C not licensed** (`tier_c_licensed: false`).

**Part 3, membership** (exploratory by declaration). `perc_id_hp` across clusters, Kruskal
**p = 4.8e-30**: 86.9 / 68.6 / 81.5 / 67.2 / 78.1 / 63.7 / 70.1. `delta_norm` 7.70 / 7.80 / 5.93 / 6.79 /
8.68 / 8.01 / 8.95; `gc3_div` 0.142 / 0.010 / −0.015 / 0.031 / 0.116 / 0.213 / 0.204; `retained_frac`
0.826–0.933. The clusters are essentially **conservation bands**:

| cluster (n) | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| 1 (77) | 0 | 3 | 11 | 26 | 37 |
| 2 (30) | 10 | 9 | 8 | 2 | 1 |
| 3 (98) | 3 | 16 | 29 | 23 | 27 |
| 4 (49) | 21 | 12 | 13 | 3 | 0 |
| 5 (37) | 5 | 9 | 5 | 8 | 10 |
| 6 (8) | 4 | 4 | 0 | 0 | 0 |
| 7 (101) | 37 | 27 | 14 | 18 | 5 |

→ this is what makes H2c uninterpretable as run. **Gate 3, second half, passes** on existence; Tier C
needs existence *and* stability.

*→ **H1c**: `existence.csv` (part 1), `h1c_summary.json` → `stability` (part 2),
`membership.csv` + `assignments.csv` (part 3); figure `figures/4_h1c_clusters` (all three panels).*

## Stage 3 — LOO vectors and layer freeze

`stage3_cds_mean/` (`../stage3_select.py`): `loo_vectors.npz` (196 MB — per-gene `v_-i`, pooled vector,
anisotropy axis μ̂, GC axis, all layers), `loo_diagnostics.csv`, `layer_candidates.csv`.

**The freeze is `blocks.27` / `cds_mean`**, recorded in `stage3_config.json` before any stage-4
generation, so it is blind to every stage-4 outcome.

| layer | `loo_cos` median | `loo_frac_pos` | `rel_norm` | `frac_on_cone` | `abs_cos_gc` |
|---|---|---|---|---|---|
| 24 | 0.605 | 0.988 | 0.102 | 0.008 | 0.537 |
| 25 | 0.648 | 0.990 | 0.116 | 0.012 | 0.491 |
| 26 | 0.719 | 0.998 | 0.201 | 0.008 | 0.397 |
| **27** | **0.775** | **0.998** | 0.364 | 0.012 | 0.403 |

L27 leads on alignment and relative dose, sits off the anisotropy cone, and is the last layer before the
norm explosion. Two numbers here are load-bearing later:

- **`loo_vector_agreement` = 0.999995** — every gene's LOO direction is essentially the same direction,
  an arithmetic consequence of removing one gene from a 400-gene mean. This is why `cross_gene` fails.
- **`abs_cos_gc` = 0.403** — the frozen direction sits ~66° from the ΔGC3 axis. That component is what
  `gc_removed` projects out.

Cluster-restricted vectors (H2c) and per-gene peak layers (Tier D) are also built here.

*→ freezes the layer for all of G2; supplies H1b's dose diagnostics and `abs_cos_gc`:
`layer_candidates.csv`, `loo_diagnostics.csv`.*

## Gate 4 — the two free stage-3 pre-checks

`strat/stage3_gates.py` → `stage3_gates/gate4_summary.json`, `tier_c_cosine_contrast.csv`.

**Tier C — contrast clears, gate does not.** `contrast` = cos(random, random) − cos(same-cluster,
other-cluster), norm-matched, focal gene excluded, ~2,000 draws per k:

| k | 3 | 4 | 5 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| cos(same, other) | 0.707 | 0.744 | 0.766 | 0.782 | 0.804 | 0.823 | 0.836 |
| cos(random, random) | 0.781 | 0.827 | 0.855 | 0.878 | 0.906 | 0.935 | 0.951 |
| contrast | +0.074 | +0.083 | +0.089 | +0.096 | +0.102 | +0.111 | +0.115 |

Positive at every k, with the mechanical convergence the gate exists to catch visible by k = 16 (both
arms ≳ 0.84 from each other). But gate 4 is an **AND** with H1c stability, which failed →
`licensed: false`. `run_400.sh` skipped H2c (`SKIP 04d_h2c`); `strat/queue_h2c.sh` ran it afterwards as
exploratory.

**Tier D — NOT LICENSED.** Per-gene argmax of `loo_cos` in L24–27: **88.75 % peak at the frozen layer**
(24:12, 25:5, 26:28, 27:355), `spread_ok: false`. The arm would re-run the frozen-layer condition at
full price — ~3 h of A10G saved by a CPU check.

*Trap:* the **unrestricted** argmax over all 32 layers is spurious — 132 genes "peak" at layer 0 and 134
at layer 9, where ‖D_i‖ ≈ 1e-2 against 7.4 at L27. `loo_cos` is scale-free, so embedding layers score
well on what is essentially shared codon composition. Stored as
`unrestricted_histogram_DIAGNOSTIC_ONLY`; never use it as the gate.

*→ licenses (declines) **H2c** and **H1d**'s causal arm: `gate4_summary.json`.*

## Stage 4 — generation and scoring

`stage4_cds_mean_blocks27/` (`strat/stage4_steer.py`, `stage4_analysis.py`): `stage4_scores.csv`
(27,404 rows = 398 genes × 13 conditions × ~5.3 samples), `generations.jsonl.gz`, `scoring_plan.csv`,
`analysis_summary.csv`, `analysis_gradient.csv`, `failures.log` (empty).

398/400 genes usable (`ENSG00000183036`, `ENSG00000138495` dropped for < 20 diagnostic sites). Adaptive
sampling 5.3 samples/cell (5–16). Unsteered mean `pct_private_correct` **39.73**. Every delta is against
the same gene's unsteered cell at the same sites; CIs bootstrap over genes.

### Primary and dose

| condition | Δ private-bp | CI | vs random at same α | Δ aa | Δ nt | Δ indel bp | Δ stops |
|---|---|---|---|---|---|---|---|
| `add` α = 0.5 | **+1.51** | [1.07, 1.96] | +1.59 | −0.17 | +0.16 | −0.37 | −1.42 |
| `add` α = 1.0 | **+3.11** | [2.53, 3.69] | +3.13 | −1.78 | −0.80 | +0.01 | −2.33 |
| `add` α = 2.0 | **+4.32** | [3.56, 5.12] | +4.45 | **−4.18** | −2.70 | −1.89 | −3.64 |
| `add_own` (α_i) | **+3.42** | [2.81, 4.08] | — | −2.08 | −0.87 | −0.21 | −2.44 |
| `random` α = 0.5 / 1 / 2 | −0.09 / −0.03 / −0.13 | all span 0 | — | +0.39 / +0.39 / −0.10 | — | — | — |

**The random arm is flat at all three doses**, which is what licenses reading the rest as
species-specific rather than non-specific perturbation. Weighted meta-regression (`stage5_gain.py`,
overdispersion **φ = 2.16**): `add_a1.0` logit gain **+0.1236 ± 0.0068 (z = +18.1)**, `add_own`
**+0.1465 ± 0.0068 (z = +21.5)**.

**Dose–response is monotone and saturating, and so is its cost.** Paired within-gene *(computed for this
write-up)*: α=1 − α=0.5 = **+1.60 pp** [1.14, 2.07]; α=2 − α=1 = **+1.21 pp** [0.80, 1.65], both
Wilcoxon p < 1e-6. Against that, aa identity falls −0.17 → −1.78 → −4.18 pp and indels appear only at
α = 2 — the last doubling buys 39 % more private-bp for 2.3× the protein cost.

**The two dose conventions agree**, the pre-registered discriminator between "the direction transfers
differently" and "the dose was unequal": `add_own` beats α = 1 by **+0.31 pp** [−0.07, +0.69], p = 0.049.

**Gate 5 did not fire**; α = 1's aa cost (−1.78) is well inside the −4 to −37 pp collapse band. Note
α = 2 sits at that band's edge at −4.18.

### Confounds

| arm | what changes | Δ vs unsteered | paired vs `add` α = 1 | Wilcoxon |
|---|---|---|---|---|
| `add_cone_removed` | anisotropy axis μ̂ projected out, norm restored | +3.09 | **−0.02** [−0.33, +0.30] | 0.63 |
| `add_gc_removed` | ΔGC3 axis projected out, norm restored | **+0.81** | **−2.30** [−2.82, −1.77] | **1.2e-14** |
| `cross_gene` | another gene's LOO vector, norm-matched | +3.21 | +0.11 [−0.23, +0.43] | 0.38 |

*(paired column computed for this write-up; `analysis_summary.csv` compares each arm to unsteered.)*

**`gc_removed` is a 74 % kill.** Same norm, layer and α — only the ΔGC3 component is gone. What remains
still beats unsteered (+0.81, p = 0.0029) but is a quarter of the effect, and the residual arm is worse
behaved, not merely weaker: aa cost unchanged (−1.87 vs −1.78) while indels (+1.76 bp, p = 0.0027) and
premature stops (+0.35, p = 0.021) go **up**, where every other steered arm reduces stops by 1.4–3.6.

Two readings this run cannot separate: either the effect is largely a composition shift, or
`pct_private_correct` — scored where platypus differs from human — is itself substantially satisfiable
by a GC shift. Separating them needs the diagnostic sites split into GC-changing and GC-neutral and
every arm re-scored on each half.

**`cone_removed` costs nothing** — the low-rank anisotropy that dominates the H1c null design is
causally irrelevant to steering. Note this is a *different* axis from the PC1 of the difference-vector
cloud that the geometry section removes, which is ~89 % aligned with the GC axis (§ H1a/H1b).

**`cross_gene` is vacuous as implemented.** `stage4_steer.py:314-317` uses gene *j = (i+51) mod n*'s
**leave-one-out** vector, and all 400 LOO vectors agree at cosine 0.999995 (§ Stage 3), so it re-injects
the direction it is meant to contrast with. +0.11 pp is what a control that changes nothing should give.
The design's test is gene *j*'s own `D_j`, already in scope as `DALL[li][j]`.

### H2c — cluster-matched steering (exploratory, gate not passed)

Two k = 5 panel directions per gene, both excluding it, both norm-matched to ‖v_-i‖, differing only in
whether the donors come from the gene's own H1c cluster. No gene was skipped for cluster size.

| arm | Δ private-bp | CI |
|---|---|---|
| `panel_same_k5` | **+3.33** | [2.74, 3.93] |
| `panel_other_k5` | +2.19 | [1.65, 2.75] |

Paired: **+1.13 pp** [+0.67, +1.62], Wilcoxon **p = 1.3e-6** *(computed here)*. But `panel_same_k5` is
indistinguishable from full-panel `add` α = 1 (+0.22 pp, p = 0.098), so the contrast is the
*degradation* of the cross-cluster arm, not a gain from matching.

**Not readable as validating H1c**, for three reasons: the stability gate failed, so the partition is not
reproducible; the clusters are conservation bands, so a same-cluster panel is also conservation-matched
and no same-stratum arm exists to separate them; and donor clusters differ in `delta_norm` (5.9 → 9.0),
so the pools differ in more than membership despite norm-matching the final vector.

### Gain by stratum

*Computed for this write-up; the pipeline reports only the Spearman (`analysis_gradient.csv`) and the
mid-vs-extreme contrast (`stage5/gain_mid_vs_extreme.csv`).* Mean per-gene Δ private-bp, pp:

| condition | s0 fastest | s1 | s2 | s3 | s4 conserved | ρ vs `perc_id_hp` |
|---|---|---|---|---|---|---|
| `add` α = 0.5 | +1.25 | +1.39 | +1.35 | +1.62 | +1.91 | +0.011 (p = 0.83) |
| `add` α = 1.0 | +2.15 | +2.60 | +2.36 | +3.26 | **+5.18** | **+0.115 (p = 0.022)** |
| `add` α = 2.0 | +2.97 | +4.05 | +3.29 | +4.75 | **+6.57** | +0.088 (p = 0.081) |
| `add_own` | +2.40 | +3.03 | +3.44 | +3.64 | +4.59 | +0.095 (p = 0.058) |
| `gc_removed` | −0.32 | +0.82 | +1.10 | +0.57 | +1.91 | +0.124 (p = 0.013) |
| `panel_same_k5` | +2.79 | +3.66 | +3.39 | +3.74 | +3.04 | −0.020 (p = 0.69) |
| `random` α = 1.0 | −0.05 | −0.25 | +0.30 | −0.17 | +0.04 | +0.006 (p = 0.90) |

Gain rises weakly toward the conserved end, and **it is headroom**: the unsteered baseline falls with
conservation (42.2 / 40.3 / 39.5 / 40.3 / 36.4 pp; ρ = −0.199, p = 6.5e-5), and partialling it out
removes the gradient — `add_a1.0` +0.115 → **+0.032 (p = 0.52)** (logit scale +0.123 → +0.038),
`add_own` +0.095 → +0.006 *(computed here)*.

The aa cost runs the other way: `aa_id_to_target` vs `perc_id_hp` at ρ = −0.13 to −0.15 (p = 0.002–0.011)
for every genuinely steered arm, ≈ 0 for random. Conserved genes pay more protein damage per unit gain.

*→ the causal outcome for **H2a**, **H2b**, **H2c** and all confound arms:
`analysis_summary.csv` (per-arm effects + CIs), `analysis_gradient.csv` (gradient), `stage4_scores.csv`
(per gene × condition × sample); figures `figures/6_stage4_arms` (all conditions with CIs, the
dose–response and its cost, gain by stratum against the baseline) and
`figures/7_conservation_scatter` row 2 (per-gene gain, protein cost, and the baseline-removed
version of the same scatter).*

## Stage 5 — evolutionary rate on a fixed topology

`stage5/` (`strat/stage5_{orthologs,trees,dnds}.py`), CPU, in parallel with stage 4, 5 h 21 min.

| file | what |
|---|---|
| `ortholog_resolution.csv`, `presence_matrix.csv` | 1:1 orthologs per gene across the mammal set |
| `core_candidates.csv` | candidate common cores (a menu, 23 rows): all 400 genes have human + platypus, 348 a complete 6-taxon core, 314 a 9-taxon core |
| `cds_per_gene.csv`, `seqs/`, `trees/` | per-gene CDS, MAFFT → trimAl → IQ-TREE `-te` |
| `tree_stats.csv` | **399/400 ok**, 1 `skip` |
| `dnds/`, `dnds.csv`, `dnds_branches.csv` | **386/400 ok**, 13 `timeout(3600 s)`, 1 yn00 failure |

Fixed VertLife MamPhy topology pruned per gene, lengths re-estimated under LG+G4. Taxa per tree
6 / 23 / 24 (153 genes have all 24); trimmed alignment 59 / 474 / 4,855 codons. Medians:
`omega_m0` 0.126, `dN_hp_yn` 0.117, `dS_hp_yn` **1.288**, `dN_background_yn` 0.062,
`dS_background_yn` 0.486, `tree_dS_m0` 4.932.

**The common core was not enforced** — `core_candidates.csv` is unconsumed, there is no `core_complete`
column, and trees use whatever taxa each gene has. `tree_len` / `diameter` / `treeness` therefore remain
taxon-set dependent, protected only by `n_taxa_tree` entering the nuisance partials.

**The free-ratio codeml arm was not requested** (`--models yn m0 m2`), so no `*_fr` predictors exist —
i.e. no per-branch dS that is not a rescaled branch length, which is exactly what the failing dS control
most needs.

### Geometry vs rate

`stage5/rate_vs_direction.csv` (`stage5_merge.py`), L27, Spearman with nuisance partials on
`aln_len_trimmed`, `n_taxa_tree`, `log_cds_len`, `retained_frac`. 54 tests, Bonferroni 9.3e-4
(`**` = survives).

| predictor | role | ‖D_i‖ | `loo_cos` | `loo_cos` PC1-removed |
|---|---|---|---|---|
| `focal_residual` | primary (PC3) | **+0.367** \*\* | +0.112 | −0.164 |
| `dN_hp_yn` | primary (PC1) | **+0.262** \*\* | −0.036 | **−0.210** \*\* |
| `dN_background_yn` | primary | +0.032 | **−0.184** \*\* | **−0.225** \*\* |
| every ω | primary | ≤ +0.118 | ≤ 0.10 | ≤ 0.10 |
| `dS_hp_yn` | **negative control** | **+0.275** \*\* | −0.121 | **−0.297** \*\* |
| `dS_background_yn` | **negative control** | −0.118 | **−0.293** \*\* | **−0.204** \*\* |
| `tree_dS_m0` | **negative control** | +0.085 | **−0.265** \*\* | **−0.258** \*\* |
| `diameter` | legacy | +0.177 | **−0.259** \*\* | **−0.394** \*\* |
| `tree_len` | legacy | +0.110 | −0.194 | **−0.304** \*\* |
| `treeness` | legacy | −0.135 | +0.097 | +0.091 |
| `human_residual` | symmetric control | −0.005 | +0.079 | +0.093 |

**G3 is answered: rate predicts geometry across the full conservation range** — `diameter` vs
PC1-removed `loo_cos` −0.394, `focal_residual` vs ‖D_i‖ +0.367, both surviving partials, with
`human_residual` null everywhere (|ρ| ≤ 0.09, p > 0.06). The earlier six-family null was range
restriction.

**But the dS control fails on every outcome**: ‖D_i‖ takes `dS_hp_yn` +0.275 over `dN_hp_yn` +0.262; raw
`loo_cos` takes `dS_background_yn` −0.293 over `dN_background_yn` −0.184; PC1-removed takes `dS_hp_yn`
−0.297 over `dN_hp_yn` −0.210. Read literally, the geometry tracks how much *sequence* has changed at
least as well as what the *protein* has become.

**A saturation caveat the pipeline's own flag misses.** `dS_saturated` flags the **platypus-terminal
branch** (0/386, and both logs print the reassuring line), but the confirmatory control is the
**pairwise** `dS_hp_yn`, median **1.288** with **149/398 above 1.5**. That is the likely source of the
sign disagreement between `dS_hp_yn` (+0.275 on ‖D_i‖) and `dS_background_yn` (−0.118), and it means
neither dS column is currently a clean neutral-divergence control.

*→ the four confirmatory predictors and controls for **H1a/H1b/H2a/H2b**, and **G3**:
`tree_stats.csv`, `dnds.csv`, `dnds_branches.csv`, `rate_vs_direction.csv`; `logs/06a_merge.log` has the
per-predictor summary; figures `figures/5_rate_outcome_matrix` (predictors × outcomes, rows grouped by
what each statistic measures — protein change, neutral change, selection, total divergence,
lineage-specific rate, shape, frame; the four PAML codon-model flavours are dropped from the display
as collinear with the yn00 estimates, though their tests still count toward the Bonferroni threshold)
and `figures/9_legacy_tree_stats` (the legacy five as scatters with fits).*

## H1a and H1b — the pre-registered shape statistics

`stage5/rate_vs_direction_shape.csv`. Four confirmatory predictors, one per independent rate axis, each
fitted linear + quadratic + mid-vs-extreme, on `loo_cos` raw and PC1-removed (H1a) and ‖D_i‖ (H1b).
Holm within each family.

**H1b — magnitude. All four axes survive Holm; the relationship is linear.**

| predictor | linear | p | quadratic | p | mid−ext | Holm |
|---|---|---|---|---|---|---|
| `focal_residual` | **+0.923** | 2.5e-14 | **−0.082** | 1.3e-4 | −0.46 | **1.0e-13** |
| `dS_hp_yn` *(control)* | **+0.593** | 9.4e-6 | −0.062 | 0.30 | −0.48 | **2.2e-5** |
| `dN_hp_yn` | **+0.485** | 7.2e-6 | −0.001 | 0.99 | −0.48 | **2.2e-5** |
| `treeness` | **−0.251** | 0.0061 | −0.014 | 0.80 | −0.46 | **0.0061** |

**Units, because these are slopes and not correlations.** `linear` and `quadratic` are regression
coefficients in **outcome units per SD of predictor**, from `y ~ 1 + z + z²`. ‖D_i‖ has sd 1.83 across
genes, so dN's +0.485 is ρ ≈ 0.27 — and the Spearman grid of § Stage 5 indeed reports +0.262. Dividing
each slope by the outcome's sd reproduces its ρ to within the rank-vs-linear difference
(dN +0.266 vs +0.262; `treeness` −0.137 vs −0.135; `focal_residual` +0.506 vs +0.367, the gap there
being its long right tail). The two tables are the same result on two scales, not two results.

Faster genes carry proportionally larger difference vectors. **The negative control outranks the primary
predictor** (dS +0.593 vs dN +0.485 on the same genes and outcome; ρ +0.275 vs +0.262). The design's dose consequence
follows: since ‖D_i‖ rises with rate, fixed α = 1 is a relatively smaller dose for fast genes — which is
why both dose conventions are primary, and § Stage 4 shows they agree.

**H1a — orientation. All four axes survive Holm, with non-uniform shape.**

| predictor | best statistic | value | p | Holm |
|---|---|---|---|---|
| `treeness` | quadratic on `loo_cos` | **−0.0254** | **2.7e-10** | **2.2e-9** |
| `dS_hp_yn` *(control)* | linear on PC1-removed `loo_cos` | **−0.0504** | **1.4e-5** | **8.2e-5** |
| `focal_residual` | linear on `loo_cos` | +0.0250 | 0.0071 | 0.031 |
| `dN_hp_yn` | quadratic on `loo_cos` | −0.0117 | 0.042 | 0.046 |

Orientation is the curved phenomenon and magnitude the clean linear one, so splitting them was right — a
combined H1 would have averaged the two into mush. Mid-vs-extreme is significant on the PC1-removed
outcome (+0.042, p = 0.023), not the raw one (+0.019, p = 0.23). Again the strongest effect is
`treeness` and the second strongest the **negative control**.

**What "PC1-removed" actually removes, and why it is not the anisotropy correction.** The two are
routinely conflated, and on this run they are different axes. The anisotropy correction is the
*cone*: μ̂, the mean direction of the raw pooled activations. It is negligible at L27 by every
measure — `frac_on_cone` 0.012, |cos(v, μ̂)| = 0.108, median per-gene |cos(D_i, μ̂)| = **0.074** — and
the `cone_removed` steering arm confirms it causally at 0.02 pp. PC1 is something else: the first
principal component of the mean-centred **difference-vector cloud** (`loo_pc1_cos` in
`stage5_merge.py`), fit on the other genes. At L27 it explains 21 % of the cloud's variance and

> **|cos(PC1, GC axis)| = 0.892**, against |cos(PC1, μ̂)| = 0.072 and |cos(PC1, v)| = 0.443.

PC1 *is* the composition axis, near enough. So the PC1-removed row is the observational twin of the
`gc_removed` steering arm, not a redundant anisotropy control — and it behaves like one: removing it
**strengthens** the rate structure rather than leaving it unchanged (`dS_hp_yn` linear −0.028 → −0.050;
`perc_id_hp` vs orientation +0.092 → +0.289). Composition is masking rate signal in the raw
orientation, which is the same relationship the causal arm reports from the other side.
*(Computed for this write-up from `stage2/pooled_representations.npz` and
`stage3_cds_mean/loo_vectors.npz`; not a pipeline output.)*

**Coverage gap:** only `loo_cos`, PC1-removed `loo_cos` and ‖D_i‖ were carried into
`rate_vs_direction*.csv`. The design also asks for `q_loo` and `alpha_to_match_own_shift` (H1b) and
`frac_on_cone` (H1a); those columns exist per gene in `stage3_cds_mean/loo_diagnostics.csv` and
`geom_cds_mean/per_gene_by_layer.csv` but were never regressed on rate.

*→ **H1a**, **H1b**: `rate_vs_direction_shape.csv` (confirmatory), `rate_vs_direction.csv` (all 19);
figures `figures/2_h1ab_geometry_vs_rate` (per gene, all three outcomes × four predictors, fits and
mid-vs-extreme marked), `figures/5_rate_outcome_matrix`, and `figures/7_conservation_scatter` row 1
(the same outcomes against conservation, with a line and a ρ per panel).*

## H1d — depth

**Not tested.** No per-gene argmax-layer regression on rate was run, and the only depth output is gate
4's histogram — a licensing check, not the hypothesis. With 355/400 genes peaking at the same layer
there is no variance to regress, so H1d is **unanswerable on this panel** rather than null.

*→ `stage3_gates/gate4_summary.json` → `tier_d.layer_histogram` (the only evidence that exists); no
figure, because there is nothing to plot.*

## H2a and H2b — steerability vs rate

`strat/stage5_gain.py` → `stage5/{rate_vs_gain,rate_vs_gain_confirmatory,gain_mid_vs_extreme}.csv`;
log `logs/06b_gain.log`.

The design's beta-binomial GLMM was implemented as a **meta-regression**, and the substitution is sound:
every steered cell is compared to the same gene's unsteered cell at the same sites, so the gene random
effect differences out exactly, leaving a per-gene effect size and its sampling variance.

```
y_i   = logit(p_steered,i) − logit(p_unsteered,i)
var_i = phi · [ 1/(n_s p_s q_s) + 1/(n_u p_u q_u) ]      delta method, overdispersed
y_i   = b0 + b1·z(rate_i) + b2·z(rate_i)² + e_i,  weights 1/var_i
```

**φ = 2.16**, from between-sample scatter within each (gene, condition) cell against its binomial
expectation — nominal trials are ~2× less precise than they look, which absorbs both dependence problems
(sites within a generation are correlated; samples share a prompt).

**H2b fails.** 36 tests (`perc_id_hp` excluded as the sampling frame), Bonferroni 0.0014. **15 linear
terms are nominally p < 0.05 against 1.8 expected; 5 survive Bonferroni:**

| predictor | role | `add_a1.0` β/SD | p | `add_own` β/SD | p |
|---|---|---|---|---|---|
| `dS_background_yn` | **negative control** | **−0.0925** | **5e-6** | **−0.1071** | **3e-6** |
| `background_rate` | primary | **−0.0713** | **2.5e-4** | **−0.0716** | **0.0012** |
| `tree_len` | legacy | **−0.0614** | **0.0012** | −0.0556 | 0.0096 |
| `treeness` | legacy | −0.0425 | 0.0016 | −0.0389 | 0.011 |
| `tree_dS_m0` | **negative control** | −0.0574 | 0.0037 | −0.0569 | 0.011 |
| `omega_hp_yn` | primary | −0.0500 | 0.0048 | −0.0371 | 0.064 |
| `dN_background_yn` | primary | −0.0402 | 0.014 | −0.0331 | 0.074 |
| `dS_hp_yn` | **negative control** | +0.0287 | 0.13 | +0.0506 | 0.018 |
| `focal_residual` | primary | +0.0091 | 0.60 | +0.0379 | 0.052 |
| `dN_hp_yn` | primary | −0.0160 | 0.33 | +0.0007 | 0.97 |
| `human_residual` | symmetric control | −0.0072 | 0.81 | +0.0013 | 0.97 |

Signs are consistent — faster genes steer less well — and `human_residual` is null, so the symmetric
control behaves. But the prediction was that dN, `focal_residual` and `treeness` predict gain **while dS
does not**, and the strongest predictor in the run is the negative control `dS_background_yn` (~1.3× the
best primary predictor) while `dN_hp_yn` is dead in both conditions.

Not a headroom artifact — partialling out the unsteered baseline leaves `dS_background_yn` at
ρ = −0.281 (p = 1.2e-8) and −0.252 (p = 3.8e-7) *(computed here)* — and not the sampling frame in
disguise, since it correlates only −0.171 with `perc_id_hp` against `focal_target_dist` −0.93 and
`dN_hp_yn` −0.88. Note the two dS columns take **opposite signs on gain**, which is the saturation
problem above showing up causally.

**H2a is null under all three pre-registered statistics.** Holm within each (arm, condition):

| condition | predictor | linear | p | quadratic | p | Holm |
|---|---|---|---|---|---|---|
| `add_a1.0` | `treeness` | **−0.0425** | 0.0016 | −0.0004 | 0.96 | **0.0063** |
| `add_own` | `treeness` | **−0.0389** | 0.011 | −0.0039 | 0.69 | **0.045** |
| `add_own` | `dS_hp_yn` *(control)* | +0.0506 | 0.018 | +0.0030 | 0.75 | 0.054 |
| `add_own` | `focal_residual` | +0.0379 | 0.052 | −0.0025 | 0.56 | 0.104 |
| `add_a1.0` | `focal_residual` | +0.0091 | 0.60 | +0.0016 | 0.67 | 0.66 |
| `add_a1.0` | `dN_hp_yn` | −0.0160 | 0.33 | −0.0020 | 0.84 | 0.66 |
| `add_own` | `dN_hp_yn` | +0.0007 | 0.97 | −0.0035 | 0.76 | 0.76 |
| `add_a1.0` | `dS_hp_yn` *(control)* | +0.0287 | 0.13 | +0.0128 | 0.12 | 0.35 |

Mid-vs-extreme (strata 2–3 vs 0–4), the third statistic:

| condition | mid | ext | diff | MWU p | weighted |
|---|---|---|---|---|---|
| `add_a1.0` | +2.82 | +3.66 | **−0.84** | 0.26 | +2.92 vs +2.94 |
| `add_own` | +3.54 | +3.49 | **+0.05** | 0.83 | +3.87 vs +3.23 |

Every quadratic is flat (p ≥ 0.56), both mid-vs-extreme contrasts are null, one pointing the wrong way.
`treeness` — the one confirmatory predictor not loading on PC1 — carries a small monotone effect and is
all that survives Holm. With the baseline-partial result of § Stage 4, **steering gain is essentially
rate-independent once headroom is accounted for** — the design's "interesting either way" outcome, not a
failed measurement.

*→ **H2a**, **H2b**: `rate_vs_gain_confirmatory.csv` (confirmatory four),
`gain_mid_vs_extreme.csv` (third statistic), `rate_vs_gain.csv` (all 19), `logs/06b_gain.log`;
figures `figures/3_h2a_gain_vs_rate` (same predictor axes as figure 2) and
`figures/5_rate_outcome_matrix`.*

## Gate status

| gate | when | status |
|---|---|---|
| 1 | after stage 0 | **passed** — 1,392+ blocks/stratum vs a floor of 200 |
| 2 | after stage 1 | **passed** — 400 post-QC vs 250; fastest stratum 80 vs 40 |
| 3 | after stage 2 | **passed, both halves** — split-half 0.996 vs 0.3; H1c existence p = 0.0020 |
| 4 | after stage 3 | **both tiers declined.** Tier C: contrast clears but H1c stability failed (gate is an AND). Tier D: 88.75 % peak at the frozen layer |
| 5 | stage 4 leading genes | **did not fire** — dose-response monotone, aa −1.78 pp at α = 1 vs a −4 to −37 pp collapse band (α = 2 sits at its edge) |

## Deviations from the pre-registered design

Each one narrows what the run can claim.

- **Common core not enforced** — trees use per-gene taxon sets (6–24), so `tree_len` / `diameter` /
  `treeness` stay taxon-set dependent (§ Stage 5).
- **Legacy-recipe arm not run** — `stage5_trees.py` implements only `-te`, not the free-topology
  2026-08-04 recipe, so G3 answers "range restriction?" but not "or method?".
- **Free-ratio codeml not requested** — no `*_fr` per-branch predictors.
- **Codon-0 sensitivity arm not run** — a free paired within-panel comparison; the offsets sit unused in
  `stage1/pairs.csv`.
- **H1a/H1b coverage partial** — `q_loo`, `alpha_to_match_own_shift`, `frac_on_cone` never regressed.
- **`aligned_mean` embedded but not analysed** — no robustness pass on positional non-correspondence.
- **The gain model carries no covariates** — `wls_slope` fits `y ~ 1 + z + z²` only; the design's list
  (log CDS length, GC, retained fraction, window length, ‖D_i‖, prompt offsets, unsteered baseline) is
  absent, and the baseline is the one that demonstrably matters. `spearman_rho_partial` adjusts only for
  `aln_len_trimmed`, `n_taxa_tree`, `log_cds_len`, `retained_frac`.
- **No pool-reweighted estimates** — everything here is a design estimate on a uniform-in-stratum panel,
  not a genome-wide effect size.
- **No opossum wrong-species arm** (`stage3_config.json`: `n_with_opossum_wrong_species_control: 0`).

## Bugs and traps

- **`cross_gene` tests nothing** (`stage4_steer.py:314-317`) — swaps another gene's *leave-one-out*
  vector, which agree at cosine 0.999995. Use `DALL[li][j]`.
- **The m2 convergence count in `logs/05_stage5.log` prints `-771/386`** — `stage5_dnds.py:503` does
  `(~ok.m2_converged.fillna(False)).sum()` on an **object-dtype** column, so `~` is bitwise on bools
  (`~True == -2`). Truth from `dnds.csv`: **1 non-convergent of 386**, plus 13 genes with no m2. The
  conclusion is unaffected; the printed number is unusable.
- **`dS_saturated` does not cover the statistic under test** — it flags the platypus terminal branch
  (0/386) while `dS_hp_yn` is itself saturating (median 1.288, 149/398 > 1.5).
- **Unrestricted argmax-layer search is spurious** (§ Gate 4) — search the candidate band only.
- **The mismatched-pair null is degenerate for `loo`** by construction; read sign-flip and mismatch-pool.

## Code

`strat/run_400.sh` drives the chain; `strat/queue_h2c.sh` ran the exploratory H2c arm. Modules:
`stage0_pool.py`, `stage1_qc.py`, `stage2_embed.py`, `h1c_clusters.py`, `stage3_gates.py`,
`stage4_steer.py`, `stage4_analysis.py`, `stage5_{orthologs,trees,dnds,merge,gain}.py`,
`layer_table.py`; reusing `../delta_stats.py` and `../stage3_select.py`.

Numbers marked *(computed for this write-up)* — the paired arm contrasts, the per-stratum gain table and
the baseline partials — came directly from `stage4_scores.csv` and `stage5/dnds.csv` and are not
reproduced by any script in the repo.

---

## Hypothesis index

Paths relative to `results/2026-08-08_platypus-strat-400/`.

| # | hypothesis | § | evidence | figure | outcome |
|---|---|---|---|---|---|
| — | what each stratum contains | Stage 0–1 | `stage1/pairs.csv`, `stage5/*` | `8` | conservation, dN and tree length track the strata; **CDS length does not** (flat across all five), and `retained_frac` rises 0.81 → 0.94 |
| **G1** | geometry depends on rate | Stages 2–3, 5 | `geom_cds_mean/per_gene_by_layer.csv`, `stage5/rate_vs_direction*.csv` | `2`, `5`, `7`, `geom_cds_mean/figures/1–8` | **supported** — orientation and magnitude both rate-dependent |
| H1a | orientation varies with rate | H1a/H1b | `stage5/rate_vs_direction_shape.csv` | `2` rows 1–2, `5`, `7` A–B | **supported**, all 4 axes survive Holm; curved, strongest `treeness` (quad p = 2.7e-10), control second. `q_loo` / `frac_on_cone` untested |
| H1b | ‖D_i‖ varies with rate | H1a/H1b | `stage5/rate_vs_direction_shape.csv` | `2` row 3, `5`, `7` C | **supported**, all 4 axes, linear. dS control (+0.593) outranks dN (+0.485). `q_loo` / `alpha_to_match_own_shift` untested |
| H1c-1 | clusters exist | H1c | `h1c_clusters/existence.csv` | `4` panel A | **PASS** — excess +0.057 at k = 7, p = 0.0020 |
| H1c-2 | clusters are stable | H1c | `h1c_clusters/h1c_summary.json` → `stability` | `4` panel B | **FAIL** — bootstrap ARI 0.485 [q05 0.327], half 0.456 |
| H1c-3 | membership (exploratory) | H1c | `h1c_clusters/{membership,assignments}.csv` | `4` panel C | clusters ≈ conservation bands (p = 4.8e-30); not a claim while H2c is unlicensed |
| H1d | peak layer varies with rate | H1d | `stage3_gates/gate4_summary.json` → `tier_d` | — | **unanswerable** — 355/400 peak at L27 |
| **G2** | steerability depends on rate | Stage 4, H2a/H2b | `stage4_.../analysis_summary.csv`, `stage5/rate_vs_gain*.csv` | `6`, `3`, `7` | steering works strongly; rate-dependence essentially null |
| H2a | gain depends on rate (3 statistics) | H2a/H2b | `stage5/rate_vs_gain_confirmatory.csv`, `gain_mid_vs_extreme.csv` | `3`, `6` lower row, `7` row 2 | **null on all three**, bar a small monotone `treeness` term (Holm 0.0063 / 0.045); the raw gradient is baseline headroom |
| H2b | dN/`focal_residual`/`treeness` predict gain, dS does not | H2a/H2b | `stage5/rate_vs_gain.csv`, `logs/06b_gain.log` | `5` gain columns, `3` | **FAILS** — `dS_background_yn` strongest (Bonferroni, baseline-robust); `dN_hp_yn` dead; `human_residual` correctly null |
| H2c | cluster-matched beats cross-cluster | Stage 4 | `analysis_summary.csv` rows `panel_{same,other}_k5`; licensing in `gate4_summary.json` | `6` panel A | **+1.13 pp, p = 1.3e-6, but unlicensed and uninterpretable** |
| **G3** | was the earlier null range restriction? | Stage 5 | `stage5/rate_vs_direction.csv`, `logs/06a_merge.log` | `5` | **yes** — rate predicts geometry (\|ρ\| up to 0.39), symmetric control null. Legacy recipe not re-run, so "or method?" is not fully excluded |
| — | QC is not conservation-neutral | Stage 1 | `stage1/attrition.csv`, `stage1_summary.json` | `1`, `8` | 1.43× pass-rate gradient; all inference conditional on QC |
| — | anisotropy confound | Stage 4 | `analysis_summary.csv` → `add_cone_removed_a1.0` | `6` panel A | **not a confound** — costs 0.02 pp |
| — | composition / GC confound | Stage 4 | `analysis_summary.csv` → `add_gc_removed_a1.0` | `6` panel A | **major** — removes 74 % of the effect |
| — | gene-vs-species identity | Stage 4 | `analysis_summary.csv` → `cross_gene_a1.0` | `6` panel A | **untested** — arm is vacuous as implemented |
| — | dose confounded with rate | Stages 3–4 | `layer_candidates.csv`, `analysis_summary.csv` | `6` dose panel | real (H1b) but not driving it — both dose conventions agree |
| — | legacy five tree statistics (exploratory) | Stage 5 | `stage5/rate_vs_direction.csv`, `rate_vs_gain.csv` | `9`, `5` | `diameter` is the strongest single geometry predictor in the run (−0.398 on PC1-removed orientation); four of the five load on PC1, so ~2–3 independent tests, and only the fixed-topology recipe exists |
| — | non-specific perturbation | Stage 4 | `analysis_summary.csv` → `random_a{0.5,1.0,2.0}` | `6` panels A, dose | **controlled** — flat at every dose |

**Figures.** `figures/` in the run directory, PNG + PDF: `1_qc_attrition`,
`2_h1ab_geometry_vs_rate`, `3_h2a_gain_vs_rate`, `4_h1c_clusters`, `5_rate_outcome_matrix`,
`6_stage4_arms`, `7_conservation_scatter`, `8_strata_composition`, `9_legacy_tree_stats`,
from `strat/hypothesis_figures.py` (`--only N`
rebuilds one; only 2 and 7 need `loo_cos_pc1`, cached after the first run as
`geom_cds_mean/loo_cos_pc1_L27.csv`). Per-layer stage-2 statistics and their nulls are
in `geom_cds_mean/figures/` (8 panels) from `../figures.py`. In figures 2 and 3 every point is one gene and the drawn
curve is the model reported in the corresponding CSV, not a fresh fit; figure 7 plots the same
outcomes against `perc_id_hp`, whose lines are ordinary least-squares fits shown for readability
since the sampling frame is not a confirmatory predictor.

## Still outstanding

1. **Separate `gc_removed` from the readout** — partition diagnostic sites into GC-changing and
   GC-neutral and re-score every arm. Until then, "toward platypus" and "toward platypus *composition*"
   are not distinguished, and the 74 % result points at the second.
2. **A real `cross_gene` control** using `D_j` — one line, and the only arm testing gene-specificity.
3. **A same-stratum vs other-stratum panel arm** beside H2c.
4. **Decide what H1c is** — existence passes, stability fails. Either drop the partition or describe the
   residual cloud continuously; the low-rank structure is solid, the 7 clusters are not.
5. **Clean per-branch dS** (free-ratio arm, or an unsaturated pairwise estimator).
6. **Re-fit H2a with the design's covariates**, unsteered baseline first.
7. **The unrun design arms** — codon-0 subset, legacy-recipe trees, common-core enforcement,
   `aligned_mean` pass, pool-reweighted estimates.
