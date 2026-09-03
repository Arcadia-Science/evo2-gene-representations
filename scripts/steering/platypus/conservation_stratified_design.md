# Conservation-stratified species-direction geometry and steerability in Evo2 (human → platypus)

**Document status:** production design record for Experiment 3. The canonical executable workflow
is `experiments/exp3_platypus_steering.sh`; completed results are summarized in
`results_n400.md` and `results_site_directionality.md`. This record retains pre-run decisions
and rationale, including exploratory arms that were not promoted to the publication.

Experimental proposal · 2026-08-05, revised 2026-08-07 · extends
`results/2026-07-28_evo2-platypus-paired` (103 genes) and `results/2026-08-04_platypus103-treestats`.

**This document is method only.** Every measurement from the 100-gene pilot — results, effect sizes,
gate outcomes, run contents — lives in `pilot_results.md`. Where a choice here was made *because* of a
pilot number, the choice is stated and the evidence is cited there.

---

## 1. Motivation

We can read a **species direction** out of Evo2's residual stream: for a human/platypus 1:1
ortholog pair, `D_i = h_i,platypus − h_i,human` (mean-pooled over the CDS). Averaged over 103 genes
this direction is real (split-half stable, off the anisotropic axis at blocks 24–27), and injecting
the leave-one-out mean `v_-i` into block 27 during autoregressive generation moves the output
toward platypus at diagnostic sites (+3.8 pp over unsteered at α = 1, +6.5 pp at α = 2, with the
random-direction arm at the unsteered level).

Two questions are unanswered, and one prior result is suspect:

1. **Is there one species direction, or one per evolutionary regime?** Individual genes sit at mean
   |cos| = 0.70 to the panel mean — a strong shared component plus a substantial per-gene residual.
   We do not know whether that residual is noise or is *organised by how fast the gene evolves*, nor
   whether slow- and fast-evolving genes carry their platypus-ness at the **same depth** in the
   stack.
2. **Does evolutionary rate predict steerability?** On 2026-08-04 we tested the five tree statistics
   of `possible-tree-stats-for-steering-project (1).md` against every stage-2 and stage-4 outcome on
   the 103-gene panel and got a **clean null** (largest |ρ| anywhere = 0.18, n = 94).
3. **That null is not safe.** The 103 genes are six paralog families (HOX, RAB/RAS/ARF/RHO GTPases,
   CYP450) chosen for having enough platypus-covered members to support leave-one-out. Nine of them
   have *protein-identical* mammalian orthologs. This is a **range-restricted, family-clustered**
   sample at the conserved end of the genome. Restriction of range attenuates correlations, and at
   n = 94 we could only have detected ρ ≥ 0.29 at 80 % power — so ρ = 0.18 is exactly the ambiguous
   zone. The observed null is equally consistent with "rate does not matter" and with "we sampled
   almost no variation in rate."

**This experiment removes the range restriction.** It replaces the six-family panel with a
genome-wide, conservation-stratified panel of human↔platypus orthologs spanning the full
conservation spectrum, and re-runs stages 1 → 2 → 3 → 4 with enough genes to resolve ρ ≈ 0.14.

**Pooling is `cds_mean`** — the mean residual-stream vector over the full CDS. Stage 2 also saves an
aligned-column mode (`aligned_mean`) for free in the same forward pass, but no pre-registered test
depends on it: on the pilot its random-direction arm was contaminated where `cds_mean`'s sat at zero.
`prefix_last` is not analysed.

Every direction in the experiment is estimated with the target gene's **entire homology block** held
out, and the panel is built so that leave-one-out *is* leave-family-out. That is what supports the
claim "gene-generic" rather than "generalises across the six families we happened to pick."

---

## 2. Goals and hypotheses

**G1 — Geometry (descriptive).** Does the human→platypus difference vector depend on the gene's
evolutionary rate — in *orientation*, in *magnitude*, or in *cluster membership* — and does it sit at
the same *depth* in the stack?

Every G1 test is **per gene**, with the target gene's entire homology block held out. Nothing bins
genes or averages within a stratum: a stratum mean only detects rate-organised geometry if
similarly-conserved genes deviate from the consensus in a *shared* direction, which is an assumption,
not a finding. Strata are the sampling frame (§ Stage 0) and a descriptive covariate, nothing more.

Each of H1a, H1b and H1d is fitted with **three pre-registered statistics — linear, quadratic, and a
mid-vs-extreme contrast** — because the pilot's rate profiles are *humps*, and a monotone test is
structurally blind to a hump. All are run raw **and** PC1-removed, with PCs fit on training blocks only
so every PC1-removed number is held out. Predictors are the four confirmatory rate statistics of
§ Stage 5, Holm-corrected across 4 predictors × 2 outcomes — not `perc_id_hp`, which is the sampling
frame and a saturating, weaker instrument.

- **H1a *Orientation*.** Block-held-out `loo_cos` (and `q_loo`) varies with rate: fast genes are less
  well described by the gene-generic consensus direction.
- **H1b *Magnitude*.** ‖D_i‖ varies with rate: conserved genes sit closer to their platypus ortholog
  and so need a *smaller* intervention to move. Fitted **separately** from H1a — a gene can carry a
  large difference vector pointing the consensus way, or a small one that does not, and the two have
  opposite implications for steering. Because "needs less steering" is a *dose* statement, `q_loo` and
  `alpha_to_match_own_shift` are analysed alongside ‖D_i‖, and H1b predicts *which* dose convention
  H2a's effect should appear under.
- **H1c *Directional clusters*.** With the shared component removed, do the per-gene residuals `r_i`
  fall into groups that carry the species direction in distinct ways, and who is in them? Three parts,
  each gating the next:
  **(1) existence** — the **max-over-k silhouette excess** against a **spectrum-matched Gaussian
  null**: a cloud whose covariance eigenvalues match the observed residuals but whose orientation is
  random, so rank, anisotropy and participation ratio are preserved and only clustering is destroyed.
  Two design points, both load-bearing: the null must be spectrum-matched rather than isotropic,
  because a low-rank *unclustered* cloud yields a large positive silhouette at every k; and the
  statistic maximises over k ∈ 2…8 while the null replicates take the same maximum, which makes
  searching over k free and removes any need to commit to a k in advance. An isotropic null is
  reported alongside to show how much apparent structure is anisotropy alone.
  **(2) stability** — block-bootstrap and disjoint-half ARI;
  **(3) membership** — tested against rate and conservation; reported descriptively for CDS length,
  GC3 divergence, ‖D_i‖, retained-column fraction, Pfam clan and GO class.
  Parts 1–2 fix algorithm, distance and k range in advance (average linkage on angular distance,
  k ∈ 2…8, k reported as the argmax of the silhouette excess so selection and testing use one
  statistic); part 3 is **exploratory by declaration**. A surviving cluster is a hypothesis, not a
  result — it must be validated causally by H2c before any claim rests on it.
- **H1d *Depth*.** The layer at which a gene's LOO alignment peaks varies with rate. Null: no rate
  dependence in the per-gene argmax layer.

**G2 — Steerability (causal).** Does the intervention work better or worse on fast-evolving genes,
and does evolutionary rate — including the five tree statistics that came out null — predict
private-bp recovery or amino-acid recovery once the range is opened up?

- **H2a**: steering gain (steered − unsteered private-bp recovery) depends on rate. Same three
  pre-registered shape statistics as G1, for the same reason: on the pilot the gain profile was a hump
  on which a monotone test returned ρ ≈ 0 while a mid-vs-extreme contrast was strongly significant.
- **H2b**: `dN_hp_yn`, `focal_residual` and `treeness` predict gain and/or `aa_id_to_target`, while
  **`dS_hp_yn` does not** — the protein-level prediction, with dS as its built-in negative control and
  `human_residual` as the symmetric control. **This control is currently failing on the pilot**, so
  H2b is a live risk to the protein-level account rather than a formality.
- **H2c** *(conditional on H1c)*: a **cluster-matched** LOO direction steers gene *i* better than one
  built from genes in other clusters, at matched norm and matched panel size *k*. Not run unless H1c's
  existence and stability gates pass and the stage-3 cosine pre-check clears. H2c is the only causal
  validation of H1c, and so the only thing that can turn a cluster into a claim.

**G3 — Methodological.** Establish whether the 2026-08-04 null was a range-restriction artifact,
and produce a reusable conservation-stratified platypus panel for future steering work.

**Falsifiable framing.** The interesting outcome is *either* sign. A clean null on n ≈ 400 with the
full conservation range and adequate power is a genuine result — it would say the species direction
is a rate-independent, gene-generic feature of the representation, which is a strong claim about
what Evo2 encodes. The current null cannot say that.

---

## 3. Design

### Stage 0 — Pool construction and stratification (new; CPU, no Evo2)

**Pool.** All human protein-coding genes with a Compara `ortholog_one2one`, high-confidence platypus
ortholog. One BioMart query (`oanatinus_homolog_ensembl_gene`, `_perc_id`, `_orthology_type`,
`_orthology_confidence`) pinned to the **same Ensembl release as the existing ortholog calls
(release 116)** — cross-release gene-model drift silently corrupts loci.

Note the fast end is floored at **50 % protein identity**: Compara cannot call high-confidence 1:1
orthology below that, so genes too diverged to identify confidently are excluded by construction. That
bound is inherent to any ortholog-based design, not a choice, and it caps how much range this
experiment can open.

**Stratification variable** — `perc_id_hp`, Compara human↔platypus protein % identity. Chosen
because it is (a) available pool-wide for free from the same record that establishes 1:1 orthology,
(b) computed from sequence only, so it is independent of Evo2, of the embeddings, and of the
tree-statistic predictors we will later test, and (c) the most direct proxy for "how different is
platypus's copy of this gene."

`perc_id_hp` is the **sampling frame only**. Primary inference uses predictors that do not share
branches with the human↔platypus comparison (§ Stage 5), so no analysis regresses an outcome on
(essentially) the variable that defined the strata.

**Homology blocks, and why the panel is block-disjoint.** The pool is clustered by
**sequence similarity** — single-linkage at ≥ 30 % protein identity over an all-vs-all of the pool,
merged with shared-Pfam-clan membership — not by curated family labels, because what leaks between a
steering direction and the gene it steers is sequence similarity, which crosses curated family
boundaries freely. The panel then takes **one gene per block**. This makes ordinary leave-one-out
*identical to leave-family-out*: excluding gene *i* excludes every relative of *i* in the panel, so
`v_-i` is estimated entirely from non-homologous genes. That is what licenses the phrase
**gene-generic**, and it costs nothing — no new machinery, and blocks are the inferential unit for the
bootstrap and permutation tests in §5 without any loss of effective n, because blocks and genes are
one-to-one.

The cost of full disjointness is that the panel cannot measure *how much* family leakage would have
inflated a non-disjoint design. That number is recoverable for free from the existing 103-gene panel,
which is six families with cached embeddings: LOO vs leave-family-out `loo_cos` and `q_loo` there,
CPU only, reported as a retro-check rather than paid for with new panel slots.

**Sampling.** Five equal-n strata at pool quintiles of `perc_id_hp`, sampled **uniformly across
strata rather than proportionally**. A uniform-in-predictor design maximises power per gene for the
continuous analyses. Consequence to declare up front: the resulting correlations are *design*
correlations, not population correlations — we report both the design estimate and a
pool-reweighted estimate, and never present the design estimate as a genome-wide effect size.

**The strata carry no inferential weight.** They are a sampling frame and nothing else: they spread
the panel evenly over the conservation range so the continuous G1/G2 tests have leverage at both ends.
No hypothesis is tested by comparing strata, no direction is estimated by averaging within a stratum,
and `perc_id_hp` is not a confirmatory predictor. Strata survive in the analysis in exactly two
places — as a descriptive covariate, and as the grouping for the shape-free **mid-vs-extreme
contrast** (strata 2–3 vs 0–4), which is a test of *curvature in rate*, not a test of strata.

**Target n and a fully predetermined draw.** 400 genes post-QC (80 blocks per stratum). At n = 400 we
detect ρ = 0.14 at 80 % power (α = 0.05, two-sided) versus ρ = 0.29 at n = 94 — the point of the
redesign. To make replacement decisions impossible rather than merely unbiased: **every block in a
stratum is randomly permuted once, under a recorded seed, and the ordering is written to disk before
any QC runs; the panel is the first 80 passers in that frozen order.** There is no top-up step to
inform, so no decision can be contaminated by annotation quality or by intermediate results, and a
stratum that cannot fill does so visibly (gate 1, §6) instead of through a judgment call.

**Pre-registered before any embedding:** stratum edges, target n, the frozen block ordering and seed,
the clustering threshold, primary predictors, primary steering readout, the analysis models in §5,
and the gates in §6. The layer freeze is deliberately *not* pre-registered as a formula — see
Stage 3.

### Stage 1 — Paired dataset and QC (CPU)

Reuses `dataset.py`, generalised from a hard-coded `FAMILIES` list to an arbitrary gene list. Per
gene, one canonical transcript per species (identical rule both species), both CDS complete and in
frame, and the 90 bp prompt must be 30 **homologous, indel-free** codons aligned 1:1 in both species,
so the prompt is positionally comparable and the generation window starts at a known boundary.

**The shifted-window rule is primary at n = 400** (`--max-start-codon 30`): the prompt is the *first*
indel-free 30-codon homologous window whose **start** is at or before human codon 30 — so the window
lies inside the first 60 codons — rather than being required to start at codon 0. Codon-0
(`--max-start-codon 0`) delegates to `qc_pair`, so the sensitivity arm is bit-identical to the pilot's
production gate.

**The sensitivity arm is a nested subset, not a second panel.** The shifted rule passes a *superset* of
what codon-0 passes (verified: it rejects none of the codon-0 passers), so the sensitivity analysis is
simply the primary panel restricted to genes whose window starts at codon 0 — a paired, within-panel
comparison at zero extra GPU cost. It is deliberately **not** a separate frozen draw: filling 80
blocks/stratum under the codon-0 rule would select a partly different gene set, so the comparison would
confound the prefix rule with the gene sample, and it would cost a second full embedding and
generation run. The consequence to state when reporting it: the subset is stratum-imbalanced, since
codon-0 pass rates rise with conservation, so it is powered unevenly across the range.

**Why the shifted rule.** The indel-free-prefix gate is not conservation-neutral: fast genes are
likelier to carry an indel in their first 30 codons, so requiring codon 0 re-imposes exactly the range
restriction this experiment exists to remove. On the pilot it cost a 2.6× pass-rate gradient across
strata. The shifted rule recovers two thirds of those position-failures at a rate that is *flat* across
strata, which shows the gradient is mostly a **position** artifact rather than a sequence-quality one;
the recovered panel also steers at least as well as the codon-0 panel. (`pilot_results.md`, stage 1.)

Consequences to carry:

- QC pass rate and rejection reason **by stratum** are reported under both rules.
- All inference is *conditional on QC*: the conclusion is about genes with an indel-free homologous
  30-codon window inside their first 60 codons.
- **Open caveat:** `--max-start-codon` bounds the **human** window start only. Where a large insertion
  intervenes, the platypus counterpart can begin much later — by hundreds of bp in the pilot. The 30
  prompt codons are the same alignment columns in both species, but not at a comparable *absolute*
  position in platypus. Window-start offsets in both species are per-gene covariates; stage 4's
  per-gene prompt offset handles the scoring side.
- **Replication set, distinct from the sensitivity arm:** `--recovered-only` keeps *only* the genes
  codon-0 rejects, so the resulting panel is disjoint from a codon-0 panel rather than a
  re-measurement of it. That is what the pilot's shifted run is, and it is the stronger test — an
  effect that reproduces on genes the old gate threw away cannot be an artifact of the old gate.

### Stage 2 — Direction geometry, mean-pooled (GPU, then CPU)

One forward per (gene, species) with the hook-based pooler, tapping all 32 blocks and pooling inside
the hook, so full-CDS pooling costs the same memory as a prefix pass.

**Pooling is `cds_mean` — the mean over the full CDS — and it is the only mode any test depends on.**
`aligned_mean` (mean over *only* the 1:1 codon-alignment columns, the same columns in both species,
restricted to columns with ≥ 30 bp of indel-free upstream alignment) is still computed and saved,
because it is one extra array in the same forward pass, but it is not propagated to stages 3–5.
`prefix_last` is dropped from the analysis entirely.

**Accepted limitation:** in `cds_mean`, position *i* in human and position *i* in platypus are not the
same site, so per-gene residual structure could in principle be alignment slippage rather than biology.
Retained-column fraction is carried per gene as a covariate, and `aligned_mean` — which is unbiased in
this respect but biased the opposite way, since aligned-column coverage falls as divergence rises — is
available for one post-hoc robustness pass on frozen results.

Then `delta_stats.py` computes, at every layer:

| statistic | null | what it answers |
|---|---|---|
| directional coherence `C = ‖mean_i u_i‖` | 1/√N | is there a shared direction |
| leave-one-out alignment `a_i = cos(D_i, v_-i)` | 0 | does the consensus describe gene *i* |
| split-half stability `cos(v_A, v_B)` | 0 | the decisive test — disjoint (and homology-disjoint) halves |
| delta spectrum / participation ratio | — | how many dimensions the cloud occupies |
| nulls: sign-flip, mismatched-pair (pooled) | — | calibration |

Never the non-held-out mean: self-inclusion alone gives cos ≈ 1/√N, indistinguishable from the
coherence floor.

**New in this experiment, all run at every layer, all per gene:**

- **H1a / H1b** — regressions of `loo_cos`, `q_loo`, `frac_on_cone` (orientation) and ‖D_i‖
  (magnitude) on rate, linear + quadratic + mid-vs-extreme, **raw and PC1-removed**. PCs are estimated
  on training blocks only: fitting PC1 on all genes then evaluating a held-out gene against it leaks
  that gene into the axis being removed.
- **H1c** — residual structure `D_i = (shared) + r_i`, then existence → stability → membership.
  Existence uses the **norm-preserving direction-shuffle null** (§2): it isolates *shared* deviation
  from merely *larger* deviation, which a label permutation cannot do, because permuting labels
  equalises residual variance across groups.
- **H1d** — per-gene argmax layer of `a_i` regressed on rate.
- Per-stratum summary tables, descriptive only, no test attached.

Blocks 30–31 are analysed but flagged: activations reach ~1e12 there, block 31 is byte-identical to
block 30 in the cached mammal embeddings, and coherence peaks spuriously where the norm explodes.
All delta math in float64.

Two implementation facts worth recording. **Long CDS are tiled**: a single forward over a 10.7 kb CDS
OOMs a 23 GB A10G, so any CDS beyond 6 kb is embedded in 6 kb windows with 1 kb of leading context
discarded per window — windows tile contiguously, no position is counted twice, every pooled position
keeps ≥ 1 kb of upstream context, and genes under 6 kb take the bit-identical single-forward path
(6/100 genes affected; `n_windows` recorded, with an assert that pooled positions equal CDS length).
**The within-family mismatched-pair null is degenerate here by construction** — one gene per homology
block means permuting platypus assignments within a block is the identity map — so only the pooled
mismatch null is meaningful on this panel.

### Stage 3 — LOO vectors and layer freeze (CPU)

`stage3_select.py` per gene and candidate layer: `v_-i = mean_{j≠i} D_j` — which on a block-disjoint
panel is a **leave-family-out** vector — plus `loo_cos`, `q_loo` (the honest projection),
`frac_on_cone`, `|cos(v_-i, GC axis)|`, `rel_norm`, and `alpha_to_match_own_shift`, on `cds_mean`.

**Layer freeze is a human decision, not a formula.** Stage 2 reports the full statistic × layer grid
with its nulls and cone diagnostics, and the layer for stage 4 is chosen by the PI from that output.
The safeguard that matters is not automation but ordering: the choice is **recorded in the stage-3
config before stage 4 runs, and is therefore blind to every stage-4 outcome**, so it cannot be tuned
to the result. The diagnostics that inform it are the same ones that governed the previous run —
coherence clearing its isotropic floor with sign-flip p < 0.05, panel-level split-half > 0.3, low cone
fraction — reported rather than thresholded.

Also built here: **cluster-restricted LOO vectors** `v_-i^(same-cluster)` and `v_-i^(other-clusters)`
at matched *k* and matched norm, from the H1c partition (for H2c); and **per-gene peak layers**
(for Tier D).

**Free pre-checks that gate the expensive arms** (the RF-panel lesson: measure direction contrast
before spending GPU hours):

- `cos(v^same-cluster_k, v^other-clusters_k)` at matched *k*. Panel-mean vectors converge mechanically
  with *k* (0.81 at k = 2 → 0.97 at k = 16 on the 103-gene panel), so if the cross-cluster contrast
  at the *k* we can afford is no larger than the random-vs-random contrast at the same *k*, H2c is
  near-mechanically null and we do not run it. Contrast at small *k* (3–8) is where it can bite.
- **`corr(‖D_i‖, rate)` — H1b in descriptive form.** If fast genes have systematically larger ‖D_i‖,
  a fixed α = 1 is a *relatively smaller* dose for them and a larger one for conserved genes, so dose
  is confounded with the independent variable. The pilot confirms this correlation is real, so stage
  4's two dose conventions are load-bearing rather than a formality.
- **Per-gene argmax-layer spread**, for Tier D: if per-gene peak layers are effectively constant, the
  per-gene-layer arm has no contrast to test and is not run.

### Stage 4 — Generation and scoring (GPU, the dominant cost)

Additive operator (not clamp: pos_sd/species_sd ≈ 67–75, so clamp overwrites ~70× more
within-sequence structure than the species difference it imposes), per-gene LOO vector, 1000 bp
generated past the 90 bp prompt, temperature 0.7, top-k 4.

**Readouts** (always reported together — on Panel B they moved in opposite directions):
`pct_private_correct` at codon-aligned diagnostic sites where platypus differs from human;
`aa_id_to_target`; and the coherence set `nt_id_to_target`, `indel_bp`, `indel_events`,
`premature_stop`.

**Two dose conventions, both pre-registered**, because neither is neutral:

- **Fixed α = 1** — identical intervention magnitude for every gene; comparable *intervention*, but
  unequal relative to each gene's own shift.
- **Own-shift-matched α_i** — α_i set to reproduce gene *i*'s teacher-forced projection `q_loo`;
  comparable *effect size*, but unequal intervention.

If the conservation effect appears under one and not the other, that discriminates "the direction
transfers differently" from "the dose was unequal." This is reported as a primary contrast, not a
robustness check.

**Adaptive sampling for equal power.** Generation is stochastic, so each (gene, condition) cell is
generated `n_samples` times (5 today). Precision of the site proportion is set by sites × samples, and
sites vary 19-fold (29 / 105 / 544 = min / median / max per 1000 bp) — SE(p) at 5 samples runs 4.2 pp
for a 29-site gene down to 1.0 pp for a 544-site one, against a ~4 pp effect. Since site count *is* a
function of divergence, flat sampling would make power vary along the x-axis. So
`n_samples_i = clip(ceil(400/sites_i), 5, 16)`. **This is nearly free**: a cell's samples are batched
into ONE `model.generate` call, so wall-clock is set by the number of *cells* (~28.4 s each, measured),
not by samples — a wider batch costs the same sequential decode steps. The 400-trial target is a budget
knob (SE ≈ 2.5 pp) recorded in the config, not a principled constant; the §5 GLMM already weights by
trials, and the trials are nominal (sites within a generation are correlated), which is what the
beta-binomial overdispersion and gene random effect absorb.

**Minimum-site rule.** Below ~20 diagnostic sites one base moves the proportion >5 pp (the
denominator-noise trap from the earlier seed selection). 0/103 existing genes fall below 20, but the
new panel's conserved stratum will. Rather than drop them — re-imposing the range restriction this
experiment exists to remove — the window is extended past 1000 bp until 20 sites accumulate, capped at
the CDS end, with window length carried as a covariate.

**Conditions — every arm runs on every gene.** There is no 120-gene subset and no tiering by gene
count: the panel is 400 genes and each condition below is generated for all of them. Grouping is kept
only to say what licenses each block.

| block | conditions | purpose |
|---|---|---|
| primary | `unsteered`, `add α=1`, `add α_i`, `random α=1` | the effect and its null |
| dose | `add α=0.5`, `add α=2`, **`random α=0.5`, `random α=2`** | dose-response, each dose against its OWN norm-matched null |
| confounds | `cone_removed`, `gc_removed`, `cross_gene` | anisotropy, composition, gene-vs-species identity |
| H2c | `panel_same k`, `panel_other k` (norm-matched) | cluster-matched steering, **only if H1c and the stage-3 cosine gate pass** |

**The random arm is repeated at every dose, not just α = 1**, and that is not redundancy.
`pct_private_correct` is scored at sites where platypus differs from human, and unsteered generation is
human-like — so *any* non-specific push away from human has roughly a 1-in-3 chance of landing on the
platypus base by accident. A larger perturbation therefore raises the metric with no species
specificity at all. The pilot measured this directly: `aligned_mean` delivers a 33 % larger
residual-relative dose than `cds_mean` (`rel_norm` 0.45 vs 0.34) and its *random* arm sat +0.8 pp above
unsteered while `cds_mean`'s sat at zero, with twice the amino-acid damage. A dose-response curve read
against an α = 1 null would attribute that non-specific inflation to the direction.

**`k` in Tier C** = how many genes are averaged into the direction. Today `v_-i` averages *all* other
genes (k = 102, one global direction); Tier C instead uses k from gene i's own H1c cluster vs k from
the other clusters, norm-matched. k must be small because panel means converge mechanically on the
global mean (cosine to it: 0.87 at k = 3, 0.92 at k = 5, 0.97 at k = 16), so at k = 16 both arms sit
≳ 0.94 from *each other* and there is no contrast to test — how the earlier RF-panel arm came out
near-mechanically null. Hence **k = 3–5**, and the gate: `cos(same-cluster k, other-cluster k)` must
be below `cos(random k, random k)` at the same k. Contrast, not cluster size, is the constraint —
which also means a small H1c cluster is not disqualifying so long as it exceeds k.

**Baselines and gene selection.** No block selects genes on an outcome, so the `unsteered` cells are
generated once for the whole panel and shared by every condition — the comparison is always within
gene, at the same sites. The rule this replaces still applies to any *future* arm that does subset on
a noisy ranking: regenerate its baseline rather than reading the stored one, because genes picked at an
extreme of a noisy statistic have baselines that are conditionally low or high, and the regression to
the mean on re-measurement mimics "the treatment rescues bad genes and hurts good ones" exactly.

**Not run at n = 400:** the per-gene peak-layer arm (H1d's causal test). Its stage-3 gate requires
per-gene argmax layers to spread across the candidate band, and on the pilot 88 % of genes peaked at
the frozen layer, so the arm would re-run the frozen-layer condition at full price. Steering is
**`blocks.27` only**.

### Stage 5 — Evolutionary rate on a fixed topology (CPU, parallel)

For each panel gene: 1:1 ortholog CDS → protein → MAFFT → trimAl v1.5.rev0 → IQ-TREE 2.3.6, fixed
model (LG+G4; leaving `-m` unset re-runs ModelFinder per gene and stalls the sweep). Two changes from
the 2026-08-04 recipe, both of which separate *rate* from *tree estimation error*:

- **Fixed species topology.** Branch lengths are estimated with `-te <species tree pruned to the
  core>`, using the accepted mammalian topology (`tree/species_tree.nwk`, VertLife MamPhy) for
  topology only — its lengths are in My, so lengths are re-estimated in subs/site. An unconstrained
  per-gene tree injects gene-tree noise exactly where alignments are shortest and most conserved,
  which is the conserved end of our stratification. Fixing the topology makes `tree_len`, `diameter`
  and `treeness` pure rate measures rather than rate ⊕ topology error ⊕ ILS ⊕ hidden paralogy.
- **Common core taxon set.** With 11–24 taxa per gene these statistics are not comparable between
  genes, and `n_taxa` as a covariate does not fix it because *which* taxa are missing matters as much
  as how many. `gene_trees.py` already supports fixed cores (`--core`, default `broad12`; the previous
  run used `--core per-gene`); the core here is chosen at stage 0 from the genome-wide presence matrix
  to maximise genes × taxa, with human and platypus mandatory. The variable-taxon analysis becomes
  secondary, and gene-tree **discordance** (RF, sCF) — the only quantities that require a free
  topology — moves to its own secondary analysis rather than contaminating the rate summaries.

**Terms.** **dN** = nonsynonymous substitution rate, the amino-acid-*changing* changes per site where
such a change is possible — how much the protein has changed. **dS** = synonymous rate, changes that
leave the amino acid unchanged — roughly neutral, so it measures how much *sequence* has changed with
no protein consequence. **ω = dN/dS** normalises protein change by neutral change: < 1 purifying
selection, ≈ 1 neutral, > 1 positive selection. Suffixes name the estimator and the branches
(`_hp_yn` = human–platypus pairwise via yn00; `_background` = the rest of the tree; `_m0` / `_m2` =
PAML codon models).

**Four confirmatory rate predictors, one per independent rate axis.** Stage 5 produces 19 statistics,
but not 19 independent ones, and testing all of them is what made the pilot uninterpretable — 144 tests,
nothing surviving correction, arguable either way. Their correlation matrix has an **effective dimension
of about 3** (`pilot_results.md`), resolving into interpretable axes:

| axis | loads on | reads as |
|---|---|---|
| PC1 | `dN`, `ω`, `mean`/`focal_target_dist` | amino-acid divergence |
| PC2 | `tree_dS_m0`, `dS_background_yn`, `dS_hp_yn` (ω opposite sign) | synonymous divergence |
| PC3 | `focal_residual`, `platypus_branch` | platypus-lineage acceleration |
| — | — | `treeness`, independent of all three | rate heterogeneity / tree shape |

The confirmatory set is **`dN_hp_yn`** (PC1), **`dS_hp_yn`** (PC2, the negative control),
**`focal_residual`** (PC3 — platypus terminal branch net of the gene's general tendency,
`resid(platypus branch ~ background rate)`), and **`treeness`** (PC4). Holm-corrected across
4 predictors × 2 outcomes for G1, and separately for G2. Everything else — `background_rate`,
`human_residual`, the M0/M2 flavours, the five legacy statistics — is **exploratory**, reported with
effect sizes and no significance claims. `human_residual` is the **symmetric negative control**: it
should predict nothing if the effect is platypus-specific, and does not enter the correction.

**dN and dS are not collapsed into a composite**, because their contrast *is* the scientific question.
The project has shown Evo2's species signal is largely **protein-level** (synonymous-recode ρ +0.315
vs codon-shuffle +0.234 vs GC floor +0.118), so the prediction is that **dN and ω predict steerability
and dS does not** — a dS effect would mean we are tracking neutral divergence, i.e. composition.
Keeping them separate costs almost nothing: on the pilot they are near-orthogonal.

Caveat: human–platypus dS ≈ 1.0–1.6 is near saturation, so *platypus-terminal* dS is poorly estimated;
whole-tree background dS is the more trustworthy quantity, and both are reported.

**Legacy replication arm — kept, because G3 depends on it.** The five statistics of the write-up
(**`tree_len`**, **`diameter`**, **`focal_target_dist`**, **`mean_target_dist`**, **`treeness`**) are
recomputed *both* on the improved fixed-topology/common-core trees and on **exactly the 2026-08-04
recipe** (per-gene taxa, free topology, trimAl). Without the second, we cannot say whether the earlier
null was range restriction or method, which was the whole point of G3. Trees are CPU-cheap; both run.

Known and confirmed: stats 1–4 intercorrelate +0.51…+0.96 (all co-scale with overall rate); only
`treeness` is independent (|ρ| ≤ 0.13). So the legacy arm is ≈ **2–3 independent tests, not 5**, and
multiplicity is corrected on the effective dimension (eigenvalues of the statistic correlation
matrix), with per-statistic values also reported.

Genes whose mammalian orthologs are protein-identical (9/103 previously — ARF1 had one unique
sequence across 10 species) admit no protein tree and are recorded as `skip_conserved`. Note the
direction of this loss: it removes genes from the *most conserved* stratum, partially re-imposing the
range restriction we are trying to escape. Reported per stratum; a nucleotide-level tree is the
fallback for these genes and is analysed as a separate arm, never pooled with the protein trees.

---

## 4. Data and compute

| item | cost |
|---|---|
| BioMart pool + stratification | minutes, CPU |
| All-vs-all similarity clustering of the pool → homology blocks | ~1 h CPU (DIAMOND/BLAST) |
| Ensembl **CDS FASTA + GTF** for 24 mammals, release 116 | ~2–3 GB download → `/opt/dlami/nvme` |
| Stage 1 QC on ~1000 candidates | ~1 h CPU |
| Stage 2 embedding, ~400 genes × 2 species, all 32 blocks | ~1–2 h A10G |
| Stage 2/3 statistics (Gram-matrix formulation, 1000+ replicates) | minutes CPU |
| Stage 4, all arms on all 400 genes, adaptive sampling | ~41 h A10G (see below) |
| Stage 5 alignments + IQ-TREE `-te`, ~400 genes × 2 tree recipes | ~4–6 h CPU, parallel |
| PAML `codeml` 2-ratio + free-ratio, ~400 genes | build from source; ~6–10 h CPU, parallel |

Stage 4 costs **~28.4 s per (gene, condition) cell** (measured; samples are batched into one
`generate`, so cell count — not sample count — sets wall-clock). With every arm on all 400 genes:
primary 4 + dose 4 + confounds 3 = **11 conditions x 400 = 4,400 cells ~ 35 h**, plus H2c's 2
conditions = 800 cells ~ 6 h if licensed. **~41 h A10G total.** Running every arm on the full panel
rather than a 120-gene subset is what costs this: it buys the confound and dose arms the same n = 400
power as the primary effect, so a null in those arms is interpretable instead of ambiguous.

Only CDS FASTA and GTF are needed — not whole genomes — because this experiment works on spliced CDS
throughout. That is ~2–3 GB rather than the ~120 GB of `download_bulk.py`'s genomic route. Note
`/opt/dlami/nvme` is **ephemeral**: the previous mammal genomes were lost to it, so provenance
records exactly which release/assembly files were used and the download is scripted and resumable.

---

## 5. Analysis plan

**Primary test for G2.** Beta-binomial GLMM on diagnostic-site hits, not a Spearman on per-gene
percentages — the percentages have wildly unequal denominators across exactly the axis under study:

```
hits_i ~ Binomial(n_scorable_sites_i × n_samples_i, p_i)
logit(p_i) = β0 + β1·condition + β2·rate + β3·(condition × rate)
                                         + β3q·(condition × rate²)
             + β4·unsteered_baseline_i + covariates + (1 | gene/block)
```

**H2a is tested by β3 and β3q jointly, plus the mid-vs-extreme contrast — three declared statistics,
not one.** The quadratic is not a robustness check bolted on afterwards; it is pre-registered
alongside the linear term because the pilot's gain profile is a hump, on which a monotone test returns
ρ ≈ 0 while the mid-vs-extreme contrast is strongly significant. Fitted once per confirmatory rate
predictor (`dN_hp_yn`, `focal_residual`, `treeness`; `dS_hp_yn` as the negative
control; `human_residual` as the symmetric control). Covariates: log CDS length, GC content,
retained-column fraction, scoring-window length, `aln_len`, ‖D_i‖, `rel_norm`, prompt-window start
offset in both species, and the unsteered baseline (conserved genes' diagnostic sites may be
intrinsically easier or harder independent of steering). Overdispersion via the beta-binomial. Because
the panel is block-disjoint, gene and block are the same unit — no separate family random effect is
needed, and **all bootstraps and permutations resample blocks**, so nothing borrows strength across
homologs.

**Primary tests for G1.** All per gene, at the frozen layer, `cds_mean`, raw and PC1-removed (PCs
refit inside each replicate on training blocks only), blocks as the resampling unit:

- **H1a orientation / H1b magnitude.** Inverse-variance-weighted regression of `loo_cos` (and `q_loo`)
  and, separately, of ‖D_i‖ on z(rate): linear + quadratic, plus the mid-vs-extreme contrast. Four
  confirmatory predictors × two outcomes, Holm-corrected across those eight; unweighted Spearman
  reported alongside for continuity with the 2026-08-04 null.
- **H1c clusters.** Existence: max-over-k silhouette excess against the **spectrum-matched Gaussian
  null**, 500 replicates, with an isotropic null reported as an anisotropy reference. Stability:
  block-bootstrap ARI and disjoint-half ARI. Membership: rate and conservation tested; CDS length,
  GC3 divergence, ‖D_i‖, retained-column fraction, Pfam clan and GO class reported descriptively,
  exploratory by declaration.
- **H1d depth.** Per-gene argmax layer of `a_i` regressed on rate, same three shape statistics.

**Exploratory — the five legacy tree statistics.** Spearman ρ of each against `loo_cos`, ‖D_i‖,
steering gain at α = 1, gain at α_i and `Δaa_id_to_target`, on both the fixed-topology and the exact
2026-08-04-recipe trees, so the comparison to the earlier null is apples-to-apples. They are
exploratory because they sit *inside* the confirmatory axes — `diameter`, `tree_len`,
`focal_target_dist` and `mean_target_dist` all load on PC1; only `treeness` is independent, which is
why it is promoted. Note the consequence: the pilot's single best geometry predictor, `diameter`, lives
in this tier, because the confirmatory set is declared on the axis structure rather than on the pilot's
rankings. Reported with effect sizes, no significance claims, and partials on
`aln_len` and `n_taxa` — on the 138-gene broad panel `aln_len`, a pure null control, earned nearly as
much as any tree statistic.

**Reported regardless of significance:** effect sizes with CIs, the per-stratum descriptive tables,
the QC-attrition-by-stratum table, and the stratum composition by functional class.

**Confounds and how each is handled**

| confound | handling |
|---|---|
| diagnostic-site count scales with divergence (power ∝ rate) | sites-as-trials GLMM + adaptive sampling + ≥ 20-site floor |
| fixed α is a different relative dose per gene | two dose conventions, both primary; `corr(‖D_i‖, rate)` measured at stage 3 — and this *is* H1b, so the confound and the hypothesis are the same measurement |
| indel-free-prefix QC gate removes fast genes preferentially | shifted-window rule is **primary** at n = 400; codon-0 becomes the sensitivity arm; pass rate reported per stratum under both |
| shifted-window prompt is not at a comparable absolute position in platypus | window start offset in *both* species carried per gene as a covariate; stage 4's per-gene prompt offset re-keys the diagnostic sites |
| protein-identical orthologs remove the most conserved genes from stage 5 | reported per stratum; nucleotide-tree fallback as a separate arm |
| conservation ↔ functional class (fast: immune/olfactory/testis; slow: core machinery) | block-disjoint genome-wide sampling; GO-class composition reported per stratum; within-largest-class replication |
| homologs leaking between the direction and the gene it steers | one gene per similarity block, so LOO ≡ leave-family-out; blocks are the resampling unit; leakage magnitude quantified retrospectively on the 6-family 103-gene panel |
| gene-tree estimation error masquerading as rate | fixed species topology (`-te`) + common core taxon set; discordance analysed separately |
| variable taxon sets make `tree_len`/`diameter`/`treeness` incomparable | common core for primary statistics; variable-taxon version secondary |
| positional non-correspondence inside full-CDS pooling | **accepted limitation**; retained-column fraction as a covariate, `aligned_mean` available for one post-hoc robustness pass |
| neutral divergence vs protein-level change | dS as an explicit negative control against dN/ω — **currently failing on the pilot**, and not rescued by co-scaling with total divergence |
| binning genes by conservation presumes they deviate *together* | all G1 tests are per gene; no stratum-mean geometry anywhere |
| multiplicity across 19 correlated rate statistics | four confirmatory predictors on the four independent axes, Holm-corrected; everything else exploratory and labelled |
| shared anisotropy axis masks phylogenetic signal | every geometry test run raw **and** PC1-removed |
| GC / composition | `gc_removed` arm + `abs_cos_gc` per gene |
| gene-level vs species-level identity | `cross_gene` arm; wrong-species (opossum) arm reusable from the existing pipeline |
| norm explosion at blocks 30–31 | flagged, excluded from layer selection |
| CDS length ↔ `cds_mean` pooling and ↔ `tree_len` | covariate in every model; partials reported |
| Evo2 pretraining exposure could correlate with conservation | unsteered per-gene reconstruction quality as a covariate |

---

## 6. Decision gates

Fixed before running, each cheap and each able to stop the expensive stage:

1. **After stage 0.** If any stratum has < 200 candidate *blocks*, merge strata to four and restate the
   power target. Do not silently proceed with unequal n. The core taxon set is fixed here, from the
   presence matrix, before any tree is built.
2. **After stage 1**, under the shifted-window rule. If post-QC n < 250 blocks or the fastest stratum
   retains < 40, stop; the options are then widening `--max-start-codon` beyond 60 or restating the
   power target, either of which is a recorded design change.
3. **After stage 2.** If **panel** split-half stability never exceeds 0.3 at any layer ≤ 27, there is
   no shared direction to study and the run stops. Separately, the **H1c existence gate**: if the
   max-over-k silhouette excess does not clear the spectrum-matched null at p < 0.05, H1c is null and
   Tier C does not run. Neither gate excludes genes — H1a/H1b proceed regardless.
4. **After stage 3 (the free gates).** Tier C runs only if the same-cluster vs other-cluster cosine
   contrast at feasible *k* exceeds random-vs-random at the same *k*, **and** H1c's stability gate
   passed. Tier D runs only if per-gene argmax layers actually spread.
5. **Stage 4 leading genes.** Genes are ordered so one high-power gene per stratum runs first and
   every cell is appended as it completes. If α = 1 shows a flat dose-response or the protein
   collapses (the Panel-B failure: −4 to −37 pp aa identity), the run is killed after ~1 h with
   nothing recomputed if it is not.

---

## 7. Deliverables

1. `results/<date>_platypus-conservation-strat/` — stage 0–5 outputs, one config JSON per stage,
   full provenance (Ensembl release, assemblies, tool versions, seeds).
2. A reusable **block-disjoint, conservation-stratified** human↔platypus panel (~400 QC-passing genes,
   one per homology block, with CDS, fixed-topology trees, rate statistics, dN/dS, and cached
   full-CDS activations at all 32 blocks, with aligned-column activations saved alongside) — the
   substrate for any future steering work, replacing the six-family panel.
3. Figures: `loo_cos` vs rate and ‖D_i‖ vs rate, per gene, with the fitted linear and quadratic terms
   and the mid-vs-extreme contrast marked (raw and PC1-removed); steering gain vs rate on the *same*
   axes, so H1a/H1b and H2a can be read against each other; H1c cluster diagnostics (silhouette excess
   against the spectrum-matched null, ARI stability) and the cluster membership profile;
   coherence/stability by layer; the rate-predictor × outcome matrix with dS shown as the negative
   control and the confirmatory four separated from the exploratory rest; the legacy five-statistic
   matrix on both tree recipes; QC attrition by stratum under both prefix rules.
4. A written answer to G3: with 4× the genes, the full conservation range, and a
   properly-powered site-level model, is the 2026-08-04 null real or was it range restriction?

---

## 8. What would make this wrong, and why the null is still worth having

The result most likely to mislead us is a **spurious** conservation effect driven by site-count
power, unequal effective dose, or the QC gate — which is why all three are measured explicitly and
two of them (dose, gate) get their own arms rather than a footnote.

The result most likely to be *under*-claimed is H2c. The 2026-08-04 panel work established that high
mutual cosine between genes' difference vectors does **not** predict steering transfer — the gene's
own vector beat every multi-gene panel, and no panel beat random selection. Averaging removes
gene-specific structure the intervention needs. So a cluster-matched panel is not expected to win
merely by being more homogeneous; the stage-3 cosine gate exists to stop us paying for that lesson
twice. **Whatever H1c says about cluster geometry must be validated against the causal outcome before
it is believed** — that is the standing lesson of this project, and it is why H1c's own p-values are
explicitly not sufficient to make a cluster into a claim.

A second thing could make this wrong, and it is unresolved rather than controlled: the **dS negative
control fails on geometry**, and it is not rescued by co-scaling with total divergence. If synonymous
divergence genuinely predicts ‖D_i‖ better than nonsynonymous does, then what we call a species
direction is tracking how much sequence has changed rather than what the protein has become, and the
protein-level account does not survive contact with this panel. n = 400 does not fix this by being
larger; it sharpens the answer whichever way it falls.

And if the answer is a clean, well-powered null across the whole conservation spectrum, that is
publishable in its own right: it says Evo2's human→platypus direction is a **gene-generic,
rate-independent** property of the representation, not an aggregate of lineage-specific rate
signals — a substantive claim about what the model has learned, which the current underpowered null
cannot support. "Gene-generic" is earned here by construction rather than asserted: every direction is
estimated with the target's whole homology block held out, on a panel built so that no two genes are
relatives, with blocks as the unit of every bootstrap and permutation.
