# Adversarial scientific audit

**Target:** `pub/Beyond sequence statistics_ Testing gene family organization in DNA language model
representations (2).pdf` (working title in the file: *"Gene family representation in Evo2 entangles
biology with sequence composition"*) and the code and result artifacts in this repository.

**Scope of this audit:** scientific validity, hidden confounders, leakage, experimental units,
implementation choices that make the desired answer easier, and mismatch between the stated and the
implemented methodology. Software correctness was checked only where a bug could change a
conclusion.

**What I did:** reconstructed each experiment from executable code, recomputed published statistics
from the saved artifacts, and ran seven cheap audit analyses (labelled **AUDIT** throughout; all
scratch code lives outside the repo and nothing under `results/` or `pub/` was modified).

---

## Overall verdict

**If this were submitted today, I would trust two results, ask for major revision on a third, and
demand that the headline of Experiment 3 be withdrawn or re-derived.**

| | Trust as written? |
|---|---|
| **Exp 1 — between-family geometry tracks a between-family baseline, layer-dependently** | **Yes, mechanically.** The numbers reproduce exactly from code. But the *interpretation* ("homology axis beats compositional controls") does not survive a better compositional baseline. Revise. |
| **Exp 1 — within-family results are too heterogeneous to support a claim** | **Yes.** The paper's own restraint here is correct, and if anything understated: nucleotide composition beats the species tree at *every* layer. |
| **Exp 2 — synonymous recoding preserves between-family geometry (ρ ≈ 0.96–0.99) far better than k-mer-preserving shuffles (ρ ≈ 0.68–0.85)** | **Yes, the observation is real and reproducible, and it is the paper's strongest result.** The *conclusion* drawn from it is over-reached, and the control leaks the family label. Revise, and promote Figure 12. |
| **Exp 3 — "steering moves generation toward platypus"; "the vector carries genuine platypus signal"; "private base-pair recovery improves in every stratum"** | **No. Re-run required before publication.** The repository's own composition-matched null (`site_directionality/summary.csv`, column `d_excess`) shows the entire private-base gain is explained by the change in generated nucleotide composition, with **p = 0.15 at the primary dose**. That analysis exists in the repo and is absent from the manuscript. |
| **Exp 3 — "steering pulls toward platypus without driving Evo2 into frameshifts or premature truncation"** | **No. Contradicted by the repository's own data** (premature stops fall 5.4 → 0.7 per generation; amino-acid identity falls 35.7% → 18.6% in the most conserved stratum). |

---

## The five most important findings

### 1. CRITICAL — Experiment 3's headline effect is entirely compositional by the repo's own null.

`scripts/steering/platypus/strat/site_directionality.py:118` implements `analytic_null()`: the
expected private-base recovery when the generated base calls are permuted within
(human base × codon position) strata. The difference `excess = A_cov − null` is the
composition-adjusted recovery. Recomputed per gene from `site_directionality/per_record.csv.gz`
(**AUDIT**, n = 398 genes, Wilcoxon signed-rank over genes — the same unit the paper uses):

| condition | raw Δ private bp (pp) | p | **Δ composition-adjusted (`excess`)** | **p** |
|---|---|---|---|---|
| add α=0.5 | +1.21 | 6e-4 | +0.02 | 0.61 |
| **add α=1 (primary arm)** | **+2.68** | **<1e-9** | **+0.26** | **0.15** |
| add α=2 | +4.71 | <1e-9 | +0.43 | 0.094 |
| add α=3 | +5.68 | <1e-9 | +0.53 | 0.099 |
| add α=4 | +6.02 | <1e-9 | +0.32 | 0.53 |
| random (norm-matched) | −0.26 | 0.70 | −0.12 | 0.69 |

The composition-predicted recovery (`d_A_null`) rises +1.20 / +2.41 / +4.28 / +5.14 / +5.70 pp
across the dose ladder — i.e. **90–95 % of the reported gain is predicted by composition alone**,
and none of the residual is significant at any dose. `frac_improved` for `excess` is 0.47–0.55,
i.e. a coin flip.

**Affected:** the "Steering results" and "More steering results" sections, Figures 7, 8, 9a, and the
Key Takeaway "the vector carries genuine platypus signal."

### 2. CRITICAL — The paper's non-private control does not do what the paper says it does.

Manuscript: *"As a control, we repeated the analysis at non-private mismatch sites … There,
increasing α doesn't strengthen the platypus signal, which suggests our 'private' definition
captures the platypus-specific information."*

`site_directionality/summary.csv`, `shared_not_private` rows: at α = 1 the **primary outcome is
larger at the control sites than at the private sites** (Δ = +3.30 pp vs +2.68 pp), and the GC-class
decomposition is essentially identical (gc_up +9.41/+9.24, gc_down −7.75/−7.17, gc_neutral
+2.31/+2.79). Only the secondary metric *C* (platypus choice among departures) behaves differently
(+0.84 → −0.87 at the control sites vs +2.35 → +2.97 at private sites). The control therefore fails
for the metric the paper leads with and passes only for a metric worth 2–3 pp on a 51 % base. The
sentence as written is not supported by the file it summarises.

### 3. CRITICAL — The manuscript's quoted steering percentages come from a metric the codebase
explicitly forbids reporting.

Manuscript: *"In the most diverged stratum (stratum zero), private base pair recovery rises from
~42 % unsteered to ~46 % at α = 4 … In the most conserved stratum (stratum four) … from ~36 % to
~44 %."*

The repository's own figure-source table `results/2026-08-08_platypus-strat-400/figures/6e_levels_by_stratum.csv`
(the strict `private` = autapomorphy metric that Figures 7 and 8 plot) gives **39.81 → 44.03** and
**31.69 → 41.64**. The manuscript's four numbers instead match the *legacy* column
`pct_private_correct` in `stage4_scores.csv` (42.19 → 45.70; 36.36 → 44.03), which
`scripts/steering/platypus/strat/strat_metric.py:42-47` documents as the *"PRE-FIX pairwise
human-vs-platypus difference, scored through the protein alignment whose bias runs along the
conservation axis — historical only … **Do not report it as a new result**."* The amino-acid numbers
in the same sentences match the current tables exactly, so this is a partial transcription from a
superseded table, not a wholesale error. Either way the text and the figure do not describe the same
quantity.

### 4. HIGH — Experiment 1's "homology" axis is an amino-acid-composition axis, and a better
compositional baseline beats it at every layer.

`scripts/baselines/pfam_hmm_jsd.py:31` reduces each family's Pfam HMM to `mat.mean(axis=0)` — the
mean match-emission profile, i.e. a 20-dimensional amino-acid composition vector with all positional
information averaged away. The manuscript describes this construction honestly, then labels it the
"evolutionary baseline" and contrasts it with "compositional baselines (GC and k-mer equivalents)".

**AUDIT:** I built the JSD between families' *observed* mean amino-acid composition, computed
directly from the panel's own 11,288 CDS (no Pfam, no HMM, no homology model) and correlated it with
the published Wasserstein matrices:

| layer | Pfam-HMM JSD (published) | **observed aa-composition JSD** | k-mer | GC | Pfam ρ *partialling out* aa comp. |
|---|---|---|---|---|---|
| 6 | 0.484 | **0.794** | 0.501 | 0.349 | 0.212 |
| 12 | 0.570 | **0.664** | 0.435 | 0.169 | 0.394 |
| 20 | 0.442 | **0.586** | 0.278 | 0.232 | 0.235 |
| 27 | 0.359 | **0.709** | 0.470 | 0.449 | 0.046 |

Plain amino-acid composition beats the Pfam baseline **at all 32 layers** (0.41–0.79 vs 0.11–0.57)
and beats both nucleotide controls at layers 3–27. Controlling for it, the Pfam correlation collapses
to 0.05–0.39; controlling for Pfam, the aa-composition correlation stays at 0.46–0.74.
Pfam-HMM JSD itself correlates with observed aa composition at ρ = 0.47.

The between-family geometry is therefore best described as **family amino-acid composition**, which
is a compositional statistic — exactly the class of explanation the paper set out to exclude. This
also reframes Experiment 2: synonymous recoding preserves amino-acid composition *exactly*, so the
ρ ≈ 0.99 preservation and the low shuffle preservation are two views of one fact.

### 5. HIGH — The Experiment-2 headline control is built from the grouping variable it is used to test.

`synonymous_recode` (`scripts/controls/make_control_sequences.py:195`) resamples each codon from a
**family-pooled** codon-usage table (`FAMILY_USAGE_CONTROLS`, line 17;
`build_family_codon_usage`, line 179). The control therefore preserves the protein *and* imposes
family-level codon composition — the family label is an input to the control whose output is then
scored on how well it reproduces the between-**family** geometry.

**AUDIT** (regenerated controls from the same generators and seeds, 1,195 loci, 48 families;
family-level codon-frequency distance matrices, no model involved):

| encoder | synonymous_recode (family table) | **recode with a GLOBAL codon table** | 6-mer shuffle |
|---|---|---|---|
| in-frame codon frequency | 0.992 | **0.823** | 0.915 |
| amino-acid frequency | 1.000 | 1.000 | 0.738 |

A protein-identical recode that does *not* consult the family label preserves family-level codon
composition at 0.82, not 0.99. The published control's extra 0.17 of nucleotide-level family
structure is label leakage. The clean version (global codon table) was never run.

---

# Per-experiment audit

## Study map

| | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| **Hypothesis** | Evo2's gene-family geometry aligns with evolutionary relationships beyond composition | Compositional controls cannot reproduce that geometry | The species direction is causally usable for generation |
| **Dataset** | 48 HGNC families → 1,061 human paralog groups → 11,288 ortholog loci across 24 mammals (`data/mammalian_orthologs/complete_manifest.csv`) | same loci, coding positions replaced in place | 400 human/platypus 1:1 pairs, one per MMseqs2 block, 5 identity quintiles (+ a separate 103-gene panel for layer choice) |
| **Representation** | residual stream, all 32 `blocks.i`, CDS positions only, mean of the *second half of CDS positions*, L2-normalised | same | CDS-mean pooled, blocks 0–31 |
| **Comparison** | between: exact W2 over angular² cost between family distributions; within: pairwise angular within an ortholog group | control vs natural, same statistic | additive injection of `α·v₋ᵢ` at every position of `blocks.27` |
| **Baselines** | Pfam-HMM JSD, MamPhy species tree, MAFFT/FastTree patristic, 6-mer cosine, |ΔGC| | natural geometry | unsteered, norm-matched random, cross-gene, cone-removed, GC-removed |
| **Statistic** | Spearman ρ over family pairs / over species pairs | Spearman ρ over family pairs | per-gene Δ vs unsteered |
| **Inference** | **none in the publication driver** (`exp1…sh:76` passes `--n-perms 0`); bootstrap/Wilcoxon over ortholog groups for within-family only | none | Wilcoxon over genes (in the figures), permutation null in `site_directionality` |
| **Result** | Pfam ρ 0.21→0.57→0.19 across depth | recode 0.96–0.99, shuffles 0.68–0.85, GC 0.65–0.92 | +2.7 pp private-bp at α=1, +6.0 pp at α=4 |

---

## Experiment 1 — between-family geometry

**Claim.** *"In the middle of the model, around layers 10–26, the Pfam baseline correlation climbs
to ρ ≈ 0.5–0.58 and sits above both compositional controls."*

**What the code actually tests.** `scripts/baselines/ot_between_family_sweep.py` computes, per layer,
the exact 2-Wasserstein distance between each pair of the 48 families treated as uniform empirical
distributions of L2-normalised locus vectors (`scripts/baselines/ot_between_family.py`), then
Spearman-correlates the 1,128 upper-triangle entries against three family-level matrices.

**Evidence supporting the claim.** The values reproduce exactly. I re-downloaded all 48 Pfam HMMs and
recomputed the JSD matrix: recomputed ρ matches the published ρ to 4 decimal places at every layer I
checked (L0 +0.2098, L10 +0.5588, L12 +0.5698, L15 +0.5097, L20 +0.4417, L27 +0.3589, L31 +0.1926).
Provenance for Figure 1 is **traceable**.

**Accuracy of the stated range.** Within layers 10–26 the Pfam ρ is 0.409–0.570; the quoted
"0.5–0.58" holds only at layers 10–15 (6 of 17 layers). Pfam is *below* the k-mer control at layers
25 and 26, so "sits above both compositional controls" fails at 2 of the 17 layers named.

**Strongest alternative explanation.** The geometry is family amino-acid composition. See headline
finding 4: observed aa-composition JSD beats Pfam-HMM JSD at all 32 layers and largely absorbs it in
partial correlation.

**Do existing controls rule it out?** No. The two controls (6-mer cosine, |ΔGC|) are *nucleotide*
composition. No protein-level compositional control is present anywhere in Experiment 1.

**Confounders I tested and cleared (AUDIT).**
- *Family size.* Family sizes span 20–2,578 embedded loci, and W2 between empirical distributions is
  n-biased. The size term `1/√n_a + 1/√n_b` correlates with W2 at only −0.12…+0.26, and partialling
  it out moves the Pfam ρ by < 0.02 at every layer. **Not a confounder.**
- *Locus length.* |Δ log mean locus length| correlates with W2 at 0.06–0.44 (peaks at the two layers
  where every signal is weak). Small, but see the implementation concern below.

**Statistical validity.** Weak. There is **no significance test and no confidence interval on
Figure 1** — `experiments/exp1_gene_family_geometry.sh:76` passes `--n-perms 0`, so `p_mantel` is
blank in every `between_family_ot_scores.csv`. The 1,128 family pairs are not independent
observations (48 families generate them), so a naive Spearman p would have been wrong anyway; the
Mantel machinery exists (`--n-perms 9999` default) and was deliberately switched off for the
publication run. Differences between adjacent baselines (e.g. Pfam 0.442 vs k-mer 0.278 at L20) are
reported with no uncertainty at all.

**Implementation concerns.**
- **`blocks.30` and `blocks.31` are byte-identical for every one of the 11,288 loci** (verified on
  random samples: `max|L30 − L31| = 0.0`). Evo2 is being tapped twice at the same point. The paper
  says "all 32 layers"; there are 31 distinct states. The team knows
  (`scripts/mammalian_orthologs/paired_p3_figure.py:33`, "Evo2 taps the last block twice") but the
  duplicate is plotted as two points in Figures 1, 2 and 4 and is not flagged in the manuscript.
- **Blocks 28–31 are numerically degenerate.** Mean pooled vector norms per layer: 10.8 at L27, 4.1e3
  at L28, 6.3e6 at L29, **1.5e12 at L30/31**, with a single coordinate carrying 73 % of the L30 norm.
  Any geometry read off those blocks is dominated by one activation outlier. Figure 12's generator
  excludes them; Figures 1, 2 and 4 do not.
- **Paper/code mismatch on input length.** The manuscript says gene inputs are *"centred and clipped
  at 100 kb"*. `scripts/mammalian_orthologs/extract_loci_bulk.py` applies no clip, and
  `embed_cds_masked_mammal.py:59` caps windows at `MAX_WINDOWS = 24 × 8,000 bp = 192 kb`, after which
  `_window_bounds` switches to evenly spaced windows **with gaps** (line 69). 925 loci exceed 100 kb;
  459 exceed 192 kb. **AUDIT:** 350 loci lose coding positions entirely (mean coverage 64 %, minimum
  11 %), concentrated in specific families (glutamate_metabotropic 75 % mean coverage,
  phosphodiesterase 92 %). Not fatal at 3 % of loci, but the methods statement is wrong and the loss
  is family-structured.
- **Pooling rationale does not match the implementation.** The paper justifies second-half pooling by
  "the autoregressive nature of the model gives early tokens low context". The code
  (`embed_cds_masked_mammal.py:96`) takes the second half of *CDS positions concatenated across
  windows*, and every window restarts context at position 0. For a 12-window locus, positions early
  in window 8 have almost no context yet sit in the "second half". Context depth is therefore a
  function of locus length, which is family-structured (median locus length: histone_h4 312 bp,
  glutamate_metabotropic 224 kb).
- **The Pfam-JSD matrix behind Figure 1 is never written to disk** by the publication driver — it is
  re-fetched live from InterPro on each run. The archived artifact is only the ρ.

**Verdict.** The measurement is **SUPPORTED** and reproducible. The interpretation —
"homology/evolutionary axis above compositional controls" — is **OVERSTATED**: the baseline labelled
homology is an amino-acid composition vector, and a straightforward protein-composition control beats
it everywhere.

**Minimum additional test.** Add the observed amino-acid-composition JSD as a fourth series in
Figure 1, and report Pfam ρ partialled on it. Restore the Mantel test (`--n-perms 9999`) so the
figure carries p-values. Drop or merge `blocks.31`.

---

## Experiment 1 — within-family geometry

**Claim.** *"Overall, our takeaway from the within-family analysis is simply that this approach is too
generic to support a general claim."*

**What the code actually tests.** Despite the manuscript's equation for `d_wf` (a scalar mean pairwise
distance per family), `scripts/mammalian_orthologs/mammal_score.py:124-142` computes, for each
**ortholog group** (one human paralog across ≥ 10 mammals, `MIN_SP = 10`, line 25), the Spearman ρ
between the group's pairwise angular distance matrix and each baseline, then averages those ρ within
a family. Figures 2 and 3 plot that per-family mean. The manuscript's formula is not the published
quantity. 611 of 1,034 groups pass the ≥10-species filter — a number the paper does not state.

**Terminology.** "Within-family" in the figures means *within an ortholog group* (a species tree
across 24 mammals), not within a paralog family. The manuscript uses one term for both. A reader
would reasonably conclude Figure 2 is about paralog relationships; it is about species relationships.

**What the data actually show (and the paper does not report).** Mean ρ across 48 families:

| layer | GC | 6-mer | gene tree | **species tree** |
|---|---|---|---|---|
| 0 | 0.684 | 0.719 | 0.618 | 0.441 |
| 15 | 0.324 | 0.462 | 0.470 | 0.305 |
| 27 | 0.469 | 0.637 | 0.583 | 0.442 |

Nucleotide composition (6-mer) exceeds the independent species tree at **every one of the 32 layers**,
GC alone matches or beats it at every layer, and every baseline peaks at layer 0/1 rather than
mid-stack. The honest summary of Figure 2 is "within an ortholog group, Evo2's geometry tracks
nucleotide composition better than phylogeny, at every depth" — a stronger and more useful negative
result than "too generic to support a claim".

**Statistical validity.** The bootstrap in `within_family_uncertainty.py` resamples **ortholog
groups** within a family — the correct unit for the plotted per-family mean. But each group-level ρ is
a Spearman over up to 276 pairwise distances treated as independent, and all 611 groups are scored
against the *same* 24-species tree matrix, so the per-family values are not independent of each other
either. Since the paper draws no positive conclusion here, this is a secondary issue.

**Provenance — a real discrepancy.** The manuscript quotes *"the median k-mer and species tree
correlation is 0.913, but ranges from 0.173 to 0.988. And the median k-mer and gene tree correlation
is 0.862, but ranges from −0.256 to 0.993."* Recomputing from the file that backs the published
figures
(`results/layer_sweep_summaries/mammalian-orthologs-cdsmask-48fam/within_family_vs_layer_wasserstein_angular.csv`)
gives **0.916 (0.001–0.991)** and **0.891 (−0.249–0.993)**. The manuscript's numbers match
`within_family_vs_layer_metric_pairs.csv` / `..._v2_metric_pairs.csv` exactly — the **geodesic**
tables, which `REPRODUCING.md` states are archived and not part of the publication pipeline. The text
and the figure are computed with different distance metrics.

**Verdict.** Claim as stated: **SUPPORTED WITH CAVEATS** (the restraint is right). The reported
metric-pair statistics: **untraceable to the published pipeline**. The terminology and the stated
formula: **paper/code mismatch**.

**Minimum additional test.** Recompute the two metric-pair medians from the angular tables, correct
the `d_wf` definition in Methods, and state that composition outranks the species tree at every layer.

---

## Experiment 2 — composition controls

**Claim.** *"The synonymous recode reproduces the natural geometry between gene families almost
exactly, with ρ ranging from 0.97–0.99 between layers two and 27 … This is very strong evidence that
Evo2 represents protein translation in its latent organization of gene families."* And in Key
Takeaways: *"difficult to explain unless the model represents protein-level information it was never
trained to predict."*

**What the code actually tests.** `controls_score_graphfree.py` recomputes the exact W2 family matrix
for each control condition and Spearman-correlates it with the natural W2 matrix at the same layer.
Both scoring paths (`mammal_controls_score.py` per-block, and the graph-free path) agree.

**Evidence supporting the claim.** Reproduced from `results/_ot_control_preservation_transcript_cdsmask.csv`:
synonymous_recode 0.960–0.997 over layers 2–27; the shuffles 0.649–0.934; gc_match lowest at every
layer. The stated range is right except at layer 27 (0.960, marginally below "0.97"). **This is a
genuinely informative result**: a manipulation that changes 22.8 % of coding bases (81.8 % of them at
codon position 3) preserves the geometry, while a manipulation that preserves the exact 6-mer spectrum
does not. A pure k-mer counter would give ρ = 1.0 for the shuffle and < 1 for the recode — the
observed ordering is the **opposite**, which is real evidence that the geometry is not a k-mer
statistic. Credit where due.

**Accuracy of the secondary claim.** *"consistently rank 6-mer > 4-mer > dinucleotide > GC match
across layers"* — **AUDIT:** the full ordering holds in only **16 of 32 layers**; 6-mer > 4-mer holds
in 22/32; only "dinucleotide > GC match" is truly consistent (32/32). The three shuffle rungs differ
by 0.01–0.03 with no error bars.

**Strongest alternative explanations.**
1. **The geometry is family amino-acid composition**, which the recode preserves exactly and every
   shuffle destroys (audit finding 4). "Represents protein translation" over-reads a bag-of-amino-acids
   statistic that follows from applying a codon table in frame.
2. **Label leakage.** The recode draws codons from a *family-pooled* table, so it reconstructs
   family-level nucleotide composition as well as the protein (audit finding 5: 0.99 with the family
   table vs 0.82 with a global table).
3. **ORF integrity, not protein identity.** A 6-mer shuffle destroys the reading frame, scatters stop
   codons through the CDS and removes all positional order. The model may be distinguishing
   "plausible coding sequence" from "junk", not reading the protein. The recode is the only ordinary
   rung that remains a valid ORF.

**Do existing controls rule these out?** Partly, and better than the manuscript admits — but only via
an analysis the manuscript never mentions.

**Figure 12 is the right experiment and it is missing from the paper.** `paired_p3_syn` and
`paired_p3_missense` edit the *same* eligible position-3 sites at the same rate, never introduce
stops, and both keep the ORF; only one keeps the protein. Between-family preservation
(`blocks*/controls/control_between_scores.csv`): over layers 3–27, syn 0.934–0.987 and missense
0.845–0.974; missense is lower in 24 of the 28 non-degenerate layers, mean gap −0.052 (mid-stack
layers 10–27: syn ≈ 0.98, missense ≈ 0.89). Within-family
(`paired_p3_protein_vs_nucleotide_paired_stats.csv`): mean gap −0.06 to −0.085 in mid layers,
missense lower in 75–88 % of the 48 families, Wilcoxon p down to 6e-9. **This is the paper's real
protein result, and it is modest** — roughly 90 % of the between-family preservation is achieved
*without* the protein.

I attempted to falsify it and failed, which strengthens it: **AUDIT** shows the *synonymous* arm is
compositionally *further* from natural than the missense arm (mean |ΔGC| 0.050 vs 0.021; mean |ΔGC3|
0.149 vs 0.064), so the syn-arm advantage cannot be a "less perturbed nucleotide composition"
artifact. The two arms are nonetheless not composition-matched (the syn arm is family-usage weighted,
the missense arm uniform), which should be stated.

**Statistical validity.** No CIs, no permutation tests, on Figure 4. The 1,128 family pairs are not
independent. Figure 12's per-family Wilcoxon (n = 48 families) is appropriate, but
`paired_p3_figure.py:342` defaults `--peak-layer 18`, described in its own help text as "the deepest
gap" — a post-hoc-selected layer whose quoted p-value is not a confirmatory statistic.

**Implementation concerns.**
- **The embedded control sequences for the four nucleotide rungs cannot be regenerated.** The
  repository's own artifact `results/…transcript_cdsmask/control_sequence_identity.md` states: *"the
  embedded draw is PYTHONHASHSEED-salted and cannot be regenerated"*. Git history confirms the shuffle
  branch previously seeded with `random.Random(f"{control}:{key}".__hash__() & 0xFFFFFFFF)`; those
  caches date from 2026-08-03…08-16, before the `zlib.crc32` fix now in
  `embed_cds_masked_mammal.py`. Re-running `exp2 --run` draws different control sequences. This is a
  reproducibility, not a validity, problem (the draws come from the same constrained ensemble), but
  `REPRODUCING.md` does not mention it.
- The paired-p3 caches (2026-08-26/27) postdate the fix and *are* reproducible.
- The identity audit in the same document is reassuring and should be cited in the paper: excess
  identity to source over the constraint's own null is 0.0000 ± 0.0003 for every shuffle rung and
  +0.0001 for the recode, so no rung is smuggling back source nucleotides.

**Verdict.** Observation: **STRONGLY SUPPORTED**. "Evo2 represents protein translation" /
"protein-level information": **OVERSTATED** — supportable only as "family amino-acid composition",
and only after the family-table leakage is removed. The ranking of the shuffle rungs:
**NOT SUPPORTED** as "consistent".

**Minimum additional tests.**
1. Re-run `synonymous_recode` with a **global** codon-usage table (removes family-label leakage).
   ~16–27 GPU-hours, one rung.
2. Promote Figure 12 into the manuscript and let it carry the protein claim, with the caveat that the
   arms are not composition-matched.
3. Add family amino-acid-composition JSD to Figure 1 so the two experiments are read together.

---

## Experiment 3 — platypus steering

### 3a. Layer selection

**Claim.** *"On a ~100-gene panel of human–platypus pairs, the LOO-cos is near zero through layers
9–20, rises from layer ~21, and saturates near 1.0 at layers 28–31 … The magnitude spread … drops to
a global minimum at layer 27."*

**Verified** from `results/2026-07-28_evo2-platypus-paired/stage2_cds_mean/layer_stats.csv`:
`loo_median` −0.08…+0.01 at layers 9–20, 0.73 at L27, 0.98 at L30/31; `delta_norm_cv` global minimum
0.3226 at L27. The criterion is outcome-independent, so **no outcome-based layer selection**. Good.

**Undisclosed panel composition.** The "~100-gene panel" is 103 genes drawn from **six** families and
is **71 % small GTPases** (rab 35, arf 16, ras 14, rho 8) plus 26 HOX genes. LOO cosine among such
close paralogs is not the same quantity as LOO cosine among independent genes. The panel overlaps the
n=400 steering panel in only 2 genes, so there is no leakage into the steering result — but the
composition should be stated.

**"Saturates near 1.0 at 28–31"** is carried by layers 30 and 31, which are the **same layer** (see
Exp 1). At 28–29 the median is 0.77–0.80.

**Undisclosed dose.** `stage3_cds_mean/layer_candidates.csv` records `rel_norm_v_over_pooled` = **0.364
at layer 27** — α = 1 perturbs the residual stream by 36 % of its own norm at every position, α = 4 by
146 %. Layer 24 sits at 0.10. The published L24-vs-L27 contrast is therefore largely a dose contrast,
not a layer contrast. Also `abs_cos_gc_median` = 0.40 at L27: the direction is 40 % aligned (by
cosine) with the ΔGC3 regression axis before any steering happens.

**Verdict.** **SUPPORTED WITH CAVEATS.** Selection is clean; the panel and the realised dose are
undisclosed.

### 3b. Panel construction

**Strong.** `stage0_pool.py` keeps only 1:1 high-confidence Compara orthologs with a unique platypus
partner, clusters the human proteome with MMseqs2 (`--min-seq-id 0.30 -c 0.50`), takes **one gene per
homology block** (the member nearest the block median, so no block contributes its extreme member),
cuts identity quintiles on the representatives, and freezes a seeded random order before any
outcome is observed. This is a genuinely well-designed sampling frame with real between-gene
independence — the best-engineered part of the paper. 398 of 400 genes survive to scoring.

One caveat: `strat_metric.py:83` defaults `--min-voters 1`, so a base can be called "private to
platypus" on the evidence of a single other mammal. In practice the median is 19 voting species and
only 2 genes have < 5, so the exposure is small — but the paper's "relative to the other 24 mammalian
species" is not literally what was required.

### 3c. Steering outcome — the headline

**Claim.** *"Private base-pair recovery improves over the unsteered baseline in every stratum, and
that improvement mostly grows with conservation … steering pulls the generated sequence toward
private platypus sequences without driving Evo2 into frameshifts or premature truncation."* And:
*"the vector carries genuine platypus signal, but is also partly a generic 'not-human' direction."*

**What the code actually tests.** `stage4_steer.py` adds `α·v₋ᵢ` to `blocks.27` at every position,
generates 1,000 bp (temperature 0.7, top-k 4, 5–16 samples per gene) from a 90 bp human prompt, and
`stage4_rescore_nt.py` aligns each generation to the platypus CDS and counts matches at sites where
platypus differs from human and no other sampled mammal carries the platypus base.

**Evidence supporting the claim.** Real and reproducible: per-gene Δ private bp = +2.65 pp at α = 1
(Wilcoxon p < 1e-9, n = 398), rising to +6.06 pp at α = 4 (from `stage4_scores_nt.csv`; the
`site_directionality` recomputation over covered sites gives +2.68 and +6.02); the norm-matched
random arm is flat (−0.20 pp, p = 0.86). Figures 7 and 8 are traceable to `6d_dose_by_stratum.csv` and
`6e_levels_by_stratum.csv`, which I reproduced from the raw score table.

**Strongest alternative explanation: a generic GC-increasing push meeting a GC-richer target.**
**AUDIT** — GC by codon position computed directly from `generations.jsonl.gz`:

| | GC1 | GC2 | GC3 | GC total |
|---|---|---|---|---|
| human CDS | 56.7 | 42.0 | 56.7 | 51.8 |
| **platypus CDS** | **58.8** | **44.0** | **66.4** | **56.4** |
| unsteered generations | 58.1 | 47.9 | 60.1 | 55.4 |
| **add α = 1 (primary arm)** | 66.8 | 55.1 | **74.8** | 65.6 |
| add α = 4 | 83.3 | 74.1 | 90.7 | 82.7 |

At the *primary* dose the generations are already 8 points more GC-rich at third positions than any
real platypus CDS, and 14 points above human at GC2 — a position where a change necessarily changes
the amino acid. The manuscript's *"Median GC3 crossed the platypus CDS level (66.6 %) somewhere
between α = 1 and α = 2"* understates this; pooled GC3 is 74.8 % at α = 1.

**Do existing controls rule it out? No — and the repo's own controls say the opposite.**

1. **Composition-matched permutation null** (finding 1): Δ`excess` = +0.26 pp, **p = 0.15**, at α = 1.
2. **GC-class decomposition** (`site_directionality/summary.csv`): at α = 1, private sites split
   +9.24 pp (gc_up) / −7.17 pp (gc_down) / +2.79 pp (gc_neutral). The net positive exists only
   because GC-increasing sites outnumber GC-decreasing ones — a direct consequence of platypus being
   the GC-richer genome.
3. **Non-private control fails** (finding 2).
4. **`cross_gene` is not a control.** `stage4_steer.py:333` uses gene *j = (i+51) mod n*'s
   leave-one-out vector. Because every `v₋ᵢ` is a mean over 399 of the same 400 deltas, `v₋ᵢ` and
   `v₋ⱼ` are nearly collinear by construction. Empirically the "cross-gene" arm gives **+2.94 pp**,
   *larger* than the matched arm's +2.68 pp. The intervention has **zero gene specificity**;
   `add_cone_removed` (+3.03 pp) and `panel_same_k5` (+3.40 pp) behave the same way. No arm in the
   run demonstrates that the direction is specific to anything.
5. **`gc_removed` is a weak GC control.** `stage3_select.py:94` removes a *single* linear direction
   (the univariate ΔGC3 regression axis). The resulting arm still shifts composition — GC3 falls to
   56.4 %, *below* unsteered — so its +1.80 pp gain is itself compositional, just in the opposite
   direction (gc_up −0.70, gc_down +4.52). Its Δ`excess` (+0.45, p = 0.008) is the only nominally
   composition-independent signal in the whole run.

**AUDIT — the conservation trend does not survive removing the GC axis.** Per-gene Spearman:
GC-bias of a gene's private sites vs its steering gain **ρ = 0.425, p = 6e-19** (18 % of rank
variance in the headline outcome is explained purely by whether a gene's private sites happen to be
GC-increasing). Human–platypus identity vs gain: ρ = 0.128, p = 0.011 (weak), and for the GC-removed
arm ρ = 0.039, **p = 0.44** — the conservation gradient disappears. The manuscript's *"that
improvement mostly grows with conservation"* is a property of the GC-carrying direction.

**AUDIT — the widening spread with conservation is partly mechanical.** Median scorable private sites
per gene falls monotonically 76.5 → 34 across strata 0 → 4, so the expected pure-sampling SD of the
per-gene Δ rises 3.97 → 5.33 pp on its own; the observed SD rises 4.98 → 9.50 pp. Some of the reported
widening is a shrinking evidence base, and the paper offers it as biology.

**The protein-coherence claim is contradicted by the run's own scores.** Per-gene means from
`stage4_scores_nt.csv`:

| stratum | premature stops/generation, unsteered → α=4 | aa identity to platypus, unsteered → α=4 | nt identity, unsteered → α=4 |
|---|---|---|---|
| 0 | 5.61 → 0.81 | 17.2 → 14.8 | 45.2 → 43.1 |
| 4 | 5.40 → 0.68 | 35.7 → 18.6 | 54.2 → 43.3 |

Stop codons fall by ~87 % — the arithmetic consequence of driving GC to 83 %, since all three stop
codons are AT-rich. The manuscript reads this as *"holding indel and premature-stop rates roughly
steady"* and *"the model bends toward the target species without abandoning the constraints of a
protein-coding gene."* The rates are not steady, and the direction of change is a composition
artifact, not evidence of preserved coding constraint. Meanwhile amino-acid identity to the target
falls by half in the most conserved stratum and nucleotide identity to the target falls in every
stratum — the generations get *further* from the platypus gene as they get more "platypus-like" at
private sites.

*"Concentrating substitutions at the third codon position"* also overstates: ΔGC1 = +25.2,
ΔGC2 = +26.2, ΔGC3 = +30.6 points. All three positions move by roughly the same amount.

**A framing problem that predates all of this.** The unsteered generations already differ from the
human base at **70.3 %** of private sites (`base_L`), have 17–36 % amino-acid identity to the target,
and carry ~5 premature stops per 1,000 bp. Evo2 given a 90 bp prompt does not reconstruct the gene.
"Steering moves generation away from human" is measured against a baseline that is already
overwhelmingly not-human, which makes the L/C decomposition much harder to interpret than the paper
suggests.

**Statistical validity.** The unit (per-gene mean over 5–16 samples, Wilcoxon over 398 genes) is
correct, and genes are genuinely independent thanks to the block-disjoint sampling. `n_samples` is
inversely tied to `n_diag`, which changes per-gene precision but not the unit. No multiplicity
correction across the ~23 conditions × several metrics reported; not critical given the effect sizes,
but the `excess` p-values (0.09–0.53) would not survive any correction.

**Implementation concerns.**
- `generations.jsonl.gz` was truncated by an instance shutdown and is read to the truncation point
  (`stage4_rescore_nt.py:139-152`); 48,481 of 48,484 records survive. Immaterial.
- The arm directory also holds retained `_L24` conditions, which `REPRODUCING.md` flags; analyses that
  read "every saved condition" will pick them up.

**Verdict.**
- *"Private base-pair recovery improves over the unsteered baseline in every stratum"* —
  **SUPPORTED as a raw observation, NOT SUPPORTED as evidence of platypus-directed generation.**
- *"the vector carries genuine platypus signal"* — **NOT SUPPORTED** at the stated dose (p = 0.15
  under the run's own composition-matched null).
- *"improvement mostly grows with conservation"* — **NOT SUPPORTED** once the GC axis is removed.
- *"our 'private' definition captures the platypus-specific information … rather than a shift shared
  across species"* — **NOT SUPPORTED**; the effect is at least as large at the shared control sites.
- *"without driving Evo2 into frameshifts or premature truncation" / "keeps playing by the rules of a
  coding sequence"* — **NOT SUPPORTED**; contradicted by the run's own stop-codon and identity tables.
- *"steering moves generation both away from human and toward platypus, where the away-from-human
  effect is the larger"* — **SUPPORTED** descriptively (d_L +3.5 → +9.2 pp; d_C +2.4 → +3.0 pp), with
  the caveat that neither quantity is composition-adjusted.
- Layer selection, panel construction, random null — **STRONGLY SUPPORTED** as methodology.

**Minimum additional tests.**
1. **Report `d_excess` in the manuscript.** It already exists. Nothing else in Experiment 3 should be
   claimed until it is on the page.
2. **A GC-matched steering control**: a direction constructed to produce the *same* GC3 shift as
   `add_a1.0` while carrying no species contrast (e.g. the ΔGC3 regression axis itself, norm-matched).
   If that arm reproduces the +2.7 pp, the experiment is null. ~13 GPU-hours, one arm.
3. **A reverse-direction arm** (`−v` at α = 1). If private recovery *also* rises, the effect is
   perturbation magnitude, not direction.
4. Replace `cross_gene` with a control that is actually orthogonal (e.g. a human→mouse contrast
   direction of matched norm).

---

## Cross-experiment systemic risks

1. **Nucleotide composition is the untested common cause across all three experiments**, and in all
   three the protein-level version of the confounder is the one that matters. Exp 1 controls for GC
   and 6-mers but not amino-acid composition, which dominates. Exp 3 controls for GC only through a
   single linear direction. A single added baseline — family/gene amino-acid composition — changes
   the reading of Exp 1 and Exp 2 together.
2. **The last two blocks are the same block, and blocks 28–31 are numerically degenerate** (norms up
   to 1.5e12). This affects the layer axis of Figures 1, 2, 4, 6a and 6b simultaneously.
3. **No significance testing on the two headline between-family figures.** The permutation machinery
   exists in the code and is switched off in the publication drivers. Every between-family ρ in the
   paper is a point estimate over 1,128 non-independent pairs.
4. **Post-hoc layer/metric choice is pervasive but mostly benign.** Layer 27 was chosen on geometry,
   not outcome; Figure 12's peak layer 18 was chosen on the outcome and its p-value should be treated
   as descriptive. The bigger risk is the reverse: several *unfavourable* analyses that exist in the
   repository (`d_excess`, `gc_class.csv`, `shared_not_private`, `19_gc_class_recovery`,
   `13_gc_in_steering_vector`, Figure 12) are absent from the manuscript, while the favourable
   summary statistics are present. I make no claim about intent — the draft is visibly unfinished
   (placeholder text, "TBD", "blah blah") — but as it stands the manuscript reports the subset of the
   evidence that supports the conclusion.
5. **Terminology collapses distinct concepts.** "Within-family" means within an ortholog group;
   "gene tree / patristic" is a MAFFT+FastTree tree of the *same* sequences being embedded (a
   sequence-similarity baseline, not an independent phylogeny — only the MamPhy species tree is
   independent); "homology" labels an amino-acid composition vector; "private" is defined against a
   24-mammal panel but tolerates single-species evidence.

---

## Provenance concerns

| Result | Status |
|---|---|
| Figure 1 ρ values | **Traceable.** Pfam JSD re-fetched and recomputed; matches published ρ to 4 dp at every layer checked. |
| Figures 2/3 per-family ρ | **Traceable** to `blocks*/within_family_*_angular.csv` via `layer_sweep_summary.py`. |
| Manuscript's within-family metric-pair medians (0.913 / 0.862) | **Inconsistent.** They come from the archived *geodesic* tables; the published angular figures give 0.916 / 0.891. |
| Figure 4 | **Traceable** to `results/_ot_control_preservation_*.csv` and `_angular_control_preservation_*.csv`; two independent scoring paths agree. |
| Control *sequences* embedded for gc_match / dinuc / 4-mer / 6-mer | **Not reproducible.** The repo's own `control_sequence_identity.md` states the embedded draw was PYTHONHASHSEED-salted; git history confirms the earlier `str.__hash__` seed. |
| Pfam-JSD matrix behind Figure 1 | **Not archived.** Re-fetched live from InterPro at run time; only the ρ is stored. |
| Figure 12 | **Traceable** to `control_rho_by_layer.csv` and `blocks*/controls/control_between_scores.csv`, but `pub/figures/MANIFEST.md` notes the angular paired-p3 summary "was staged outside `results/`", so part of the input path is outside the tracked result tree. |
| Figures 5–11 | **Traceable.** All regenerated 2026-08-31 10:40 from artifacts I independently reproduced from `stage4_scores_nt.csv`, `6d/6e_*.csv`, `site_directionality/summary.csv`, `generations.jsonl.gz`. |
| Manuscript's private-bp stratum percentages (42/46, 36/44) | **Demonstrably not from the published metric.** They match the legacy `pct_private_correct` column, which the code documents as "do not report". |
| Figures 2, 11, 12 | Present in `pub/figures/` with **no corresponding text or claim** in the manuscript draft. |
| Hard-coded final statistics in figure scripts | **None found.** Every numeric literal I inspected is layout or style. |

---

## Paper/code discrepancies (exact references)

| # | Manuscript says | Code does | Reference |
|---|---|---|---|
| 1 | "centred and clipped at 100 kb" | no clipping; window cap 24 × 8,000 bp, then evenly spaced windows **with gaps** | `scripts/mammalian_orthologs/embed_cds_masked_mammal.py:40,59,69`; 925 loci > 100 kb, 350 loci lose CDS positions |
| 2 | "mean-pooling the second half of token positions" | second half of **CDS positions concatenated across windows** | `embed_cds_masked_mammal.py:96` |
| 3 | "all 32 layers of Evo2" | `blocks.30` and `blocks.31` are byte-identical for every locus | `scripts/evo2/evo2_embedding.py:12`; verified on the cache |
| 4 | `d_wf` = mean pairwise angular distance within a family (a scalar) | per-**ortholog-group** Spearman ρ of the pairwise angular matrix vs a baseline, averaged within family | `scripts/mammalian_orthologs/mammal_score.py:118-142` |
| 5 | "Pfam JSD … evolutionary baseline" vs "compositional baselines" | Pfam HMM reduced to a 20-d mean amino-acid composition vector | `scripts/baselines/pfam_hmm_jsd.py:31` |
| 6 | "48 … comprising 1,144 paralogs"; "families with fewer than four paralogs were excluded" | manifest carries 1,061 groups (1,034 embedded); 2 families have < 4 groups | `data/mammalian_orthologs/complete_manifest.csv` |
| 7 | median k-mer/species-tree 0.913 (0.173–0.988) | published angular tables give 0.916 (0.001–0.991) | `results/layer_sweep_summaries/…/within_family_vs_layer_wasserstein_angular.csv` |
| 8 | "6-mer > 4-mer > dinucleotide > GC match across layers" | full ordering holds in 16/32 layers | `results/_ot_control_preservation_transcript_cdsmask.csv` |
| 9 | "private base pair recovery rises from ~42 % … to ~46 %" | strict private metric gives 39.8 → 44.0 | `results/2026-08-08_platypus-strat-400/figures/6e_levels_by_stratum.csv`; legacy column warned against at `strat_metric.py:42-47` |
| 10 | "indel burden and premature stops don't shift significantly" | stops fall 5.4 → 0.7 per generation across the dose ladder | `stage4_cds_mean_blocks27/stage4_scores_nt.csv` |
| 11 | "Median GC3 crossed the platypus CDS level between α = 1 and α = 2" | pooled GC3 is already 74.8 % at α = 1 (platypus 66.4 %) | computed from `generations.jsonl.gz` |
| 12 | non-private sites used as a control that "doesn't strengthen the platypus signal" | the primary metric improves *more* at those sites (+3.30 vs +2.68 pp) | `site_directionality/summary.csv` |
| 13 | "~100-gene panel of human–platypus pairs" | 103 genes from 6 families, 71 % small GTPases | `results/2026-07-28_evo2-platypus-paired/stage1/pairs.csv` |
| 14 | Figure 1 correlations reported without inference | `--n-perms 0` in the publication driver disables the Mantel test | `experiments/exp1_gene_family_geometry.sh:76` vs `ot_between_family_sweep.py:79-82` |

---

## Cheap falsification tests, ranked

Ranked by *P(uncovers a real problem) × scientific importance ÷ cost*.

| # | Test | Cost | Why |
|---|---|---|---|
| 1 | **Report `d_excess` from `site_directionality/summary.csv` in the manuscript.** | Zero — already computed | Highest-probability, highest-importance: it overturns the Experiment-3 headline. |
| 2 | **Add family amino-acid-composition JSD as a baseline in Figure 1**, plus the partial correlation against Pfam. | ~10 CPU-minutes (I ran it) | Overturns the "homology above composition" reading of Experiment 1. |
| 3 | **Report the `shared_not_private` primary-metric deltas alongside the private ones.** | Zero — already computed | Falsifies the paper's specificity control. |
| 4 | **Restore `--n-perms 9999` for the between-family sweep.** | ~20 CPU-min | Figures 1 and 4 currently carry no inference at all. |
| 5 | **Reverse-direction steering arm (`−v`, α = 1).** | ~13 GPU-h | If private recovery also rises, Experiment 3 is a perturbation-magnitude effect. |
| 6 | **GC-matched sham steering direction** (norm-matched ΔGC3 axis, no species contrast). | ~13 GPU-h | The single most decisive positive control for Experiment 3. |
| 7 | **Recode with a global codon-usage table.** | ~16–27 GPU-h | Removes the family-label leakage from the Experiment-2 headline control. |
| 8 | **Drop `blocks.31` (or fix the tap) and mark 28–31 as degenerate** on every layer axis. | Minutes | Removes a duplicated point and a numerically meaningless region from five figures. |
| 9 | **Re-run within-family scoring restricted to genes in a narrow GC band**, or partial out GC. | ~3 CPU-h | Tests whether any within-group species-tree signal survives composition. |
| 10 | **Equal-size family subsampling for the between-family W2** (`subsample_sensitivity` already exists). | ~1 CPU-h | I found family size is *not* a confounder, so this is a low-yield confirmation. |

---

## Severity table

| Severity | Finding | Affected figure / claim |
|---|---|---|
| **CRITICAL** | The private-base-recovery gain is not significant under the run's own composition-matched null (Δ`excess` = +0.26 pp, p = 0.15 at α = 1; +0.32 pp, p = 0.53 at α = 4) | Figures 7, 8, 9a; "Steering results", "More steering results", and the Key Takeaway "the vector carries genuine platypus signal" |
| **CRITICAL** | The non-private control fails for the primary metric (+3.30 pp at control sites vs +2.68 pp at private sites) | The control sentence in "More steering results"; Figure 9b |
| **CRITICAL** | Manuscript's quoted private-bp percentages come from the legacy column the codebase forbids reporting | "Steering results" paragraph 2; inconsistent with Figures 7–8 |
| **CRITICAL** | "Holds indel and premature-stop rates roughly steady" is contradicted by the run (stops 5.4 → 0.7; aa identity 35.7 → 18.6) | Key Takeaways paragraph 2; Figure 7 discussion |
| **HIGH** | Observed amino-acid composition beats the Pfam "homology" baseline at all 32 layers and absorbs it in partial correlation | Figure 1; "Gene family representations correlate with evolutionary and compositional baselines" |
| **HIGH** | `synonymous_recode` uses a family-pooled codon table — the grouping variable leaks into the control (0.99 vs 0.82 with a global table) | Figure 4; "very strong evidence that Evo2 represents protein translation" |
| **HIGH** | Conservation gradient in the steering gain disappears when the GC axis is removed (ρ 0.128 → 0.039, p = 0.44); per-gene GC-bias predicts the gain at ρ = 0.425 | "that improvement mostly grows with conservation"; Figure 8 |
| **HIGH** | `cross_gene` and `cone_removed` arms give gains ≥ the matched arm — the intervention has no gene specificity, and no arm in the run establishes any specificity | The premise of Experiment 3 as a "steering vector" rather than a global perturbation |
| **HIGH** | No significance test or CI on Figures 1 and 4 (`--n-perms 0`) | Figures 1, 4 and every between-family ρ quoted in the text |
| **MEDIUM** | `blocks.30` ≡ `blocks.31`; blocks 28–31 numerically degenerate (norms to 1.5e12) | Layer axes of Figures 1, 2, 4, 6a, 6b; "saturates near 1.0 at layers 28–31" |
| **MEDIUM** | No 100 kb clip; 350 loci lose 36 % of coding positions on average, family-structured | Methods; Figures 1–4 |
| **MEDIUM** | Widening spread across conservation strata is partly a shrinking evidence base (76.5 → 34 scorable sites) | "the most conserved genes show … the most variable [response]"; Figure 7 |
| **MEDIUM** | Manuscript's within-family metric-pair medians come from the archived geodesic tables, not the published angular ones | "Patterns within gene families" paragraph |
| **MEDIUM** | "6-mer > 4-mer > dinucleotide > GC" holds in 16/32 layers, reported as consistent | Figure 4 discussion |
| **MEDIUM** | Embedded shuffle-control sequences are not regenerable (PYTHONHASHSEED-salted seed at embed time) | Figure 4 reproducibility; `REPRODUCING.md` |
| **MEDIUM** | Steering dose is undisclosed: `‖v‖ / ‖h‖` = 0.36 at α = 1, 1.46 at α = 4; direction is 40 % cosine-aligned with the GC3 axis | Experiment 3 methods; the L24-vs-L27 comparison is a dose comparison |
| **MEDIUM** | "Within-family" denotes within-ortholog-group; the manuscript's `d_wf` formula is not the plotted quantity | Methods; Figures 2, 3 |
| **LOW** | Pfam-JSD matrix re-fetched live and never archived | Figure 1 reproducibility |
| **LOW** | 103-gene layer-selection panel is 71 % small GTPases from 6 families | Figures 6a, 6b |
| **LOW** | `--min-voters 1` allows a "private" call on one voting species (affects 2/398 genes) | Private-site definition |
| **LOW** | Figure 12's `--peak-layer 18` is outcome-selected; its p-value is descriptive | Figure 12 (currently unreferenced) |
| **LOW** | Family size is *not* a confounder for the between-family W2 (checked, cleared) | — |
| **LOW** | Control rungs retain no source nucleotides beyond their own constraint (excess identity ≈ 0; checked, cleared) | — |

---

## What I would keep

To be clear about what survives: the mammalian-ortholog panel construction, the platypus panel's
block-disjoint stratified sampling, the norm-matched random null, the outcome-independent layer
selection, the exact-OT between-family metric, the control-identity audit, and the paired
synonymous/missense arms are all sound, careful work — several of them better than what the
manuscript actually leans on. The recurring problem is not the engineering. It is that the analyses
that would qualify the conclusions have already been run, live in this repository, and did not make
it onto the page.

---

*Audit analyses were run outside the repository; no file under `results/`, `data/`, or `pub/` was
modified. Reproduction scripts are in this session's scratchpad,
`/opt/dlami/nvme/uv/tmp/claude-1000/-home-ubuntu-Development-glm-latent-mapping/b38711f4-cb44-476f-8858-e82c86aee722/scratchpad/`:
`audit_controls.py` (control regeneration from the shipped generators and seeds, compositional
encoders, global-codon-table recode), `audit_pfam2.py` (Pfam re-fetch and verification of the
published Figure-1 rho), `audit_pfam3.py` (amino-acid-composition baseline, partial correlations,
family-size and locus-length confounder checks). The steering recomputations quoted above were run
inline against `results/2026-08-08_platypus-strat-400/` and are reproducible from
`stage4_scores_nt.csv`, `site_directionality/per_record.csv.gz`, `site_directionality/summary.csv`,
`site_directionality/sites_per_gene.csv` and `stage4_cds_mean_blocks27/generations.jsonl.gz`.*
