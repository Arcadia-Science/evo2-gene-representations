# Results — is the recovery gain platypus-directed, or a substitution bias?

**Document status:** production result record supporting Figures 9a–9b. The active runner generates
at block 27. This record also reports descriptive block-24 conditions retained in the existing
generation artifact.

Run `results/2026-08-08_platypus-strat-400`, arm `stage4_cds_mean_blocks27`, Evo2-7B, 398 human/platypus
gene pairs, 22 conditions across steering layers blocks.27 and blocks.24. **No sequences were
generated for this analysis** — everything below is scored from the saved `generations.jsonl.gz`
(39 848 records readable; the file is truncated at the tail by the instance shutdown that ended the
original sweep, and the truncation point is reported by every script that reads it).

**No resampling and no significance tests anywhere in this document.** Bootstrap CIs and Wilcoxon
tests were dropped by request; every number is the observed result over the samples actually
generated (5 per gene per condition for most cells, up to 16 for a few). Where spread matters it is
reported as the observed per-gene SD, IQR and fraction of genes improved.

---

## 1. What was measured

At every predefined platypus-private site, with **H** = human base, **P** = platypus base,
**Y** = generated base:

| | definition | reads as |
|---|---|---|
| **A** | Y = P, over covered sites | platypus recovery — the run's headline |
| **L** | Y ≠ H, over covered sites | leave-human rate |
| **C** | Y = P given Y ≠ H | platypus choice among departures |
| **K** | covered sites / all predefined sites | alignment coverage |

A = L × C identically, which is what makes the split readable. Two site sets are scored in the same
pass from the same alignment:

| site set | definition | sites/gene (mean, median) |
|---|---|---|
| **private** (headline) | platypus base differs from human **and** no other sampled mammal carries it | 56.1, 52.5 |
| platy, not human | platypus base differs from human (pairwise) | 196.4, 190.5 |

Evidence base for the private set: median **19** voting ortholog species per gene (mean 18.3).
All 398 genes have positive ortholog evidence and a non-empty private set, so no gene silently drops
out of any panel.

Two recovery denominators are reported, because the brief's definition and the published one differ:
**A_cov** counts only sites the alignment reaches (the brief), **A_all** keeps gap-aligned sites in
the denominator as misses (what `stage4_rescore_nt.py` publishes).

---

## 2. Controls

Four controls, run on all 398 genes before any result was read.

| control | A | L | C | K | expected |
|---|---|---|---|---|---|
| human window fed in as the generation | 0.000 | 0.000 | — | 100.000 | A = L = 0 exactly |
| platypus target fed in | 100.000 | 100.000 | 100.000 | 100.000 | 100 exactly |
| shuffled human (composition kept, sequence destroyed) | 44.13 | 80.46 | 54.79 | 95.58 | the chance floor |

The two structural controls pass **exactly** (max deviation < 1e-9), on both site sets. The script
hard-fails rather than reporting anything if they do not.

The shuffled-human floor reads **44.1 %** on the private set, against an unsteered arm of **37.5 %**.
This reproduces the floor already documented in `strat/strat_metric.py` and carries the same warning:
**the absolute level of A is not interpretable against zero.** A sequence carrying no platypus
information at all scores above the unsteered arm, because a human-prompted generation actively emits
the human base where platypus differs. Only the paired within-gene change is readable.

A fourth check: the 1000-permutation shuffle null agrees with its closed-form expectation across all
79 695 records — mean absolute difference **0.095 pp**, max |z| 4.81 over 79 695 comparisons.

*Data:* `site_directionality/controls.csv`, `controls_per_gene.csv`, `sites_per_gene.csv`.

---

## 3. Reconciliation against the published number

`A_all` reproduces `pct_autapomorphy_correct` exactly, to the reported digit, on every arm the
published table covers:

| arm | published | A_all here |
|---|---|---|
| L27 add, α=1 | +2.65 | **+2.65** |
| L27 add_own | +3.40 | **+3.40** |
| L27 random, α=1 | −0.20 | **−0.20** |
| L24 add, α=1 | +0.51 | **+0.51** |
| L24 add_own | +0.89 | **+0.89** |

So what follows is a decomposition of the published result, not a competing measurement.

The L24 dose rungs α = 0.5, 2, 3, 4 have never been nt-rescored and are absent from
`stage4_scores_nt.csv`; they are scored here for the first time, and they cover only **87–88 of 398
genes**. Every table and figure flags them.

---

## 4. The decomposition — A, L, C, K

Mean per-gene change against the same gene's unsteered generation, private site set, layer 27
(pp; 398 genes unless noted).

| arm | ΔA_cov | ΔL | ΔC | ΔK | Δexcess |
|---|---|---|---|---|---|
| add α=0.5 | +1.21 | +0.89 | +1.56 | +0.04 | +0.02 |
| add α=1 | +2.68 | +3.51 | +2.35 | +0.08 | +0.26 |
| add α=2 | +4.71 | +6.32 | +3.32 | +0.28 | +0.43 |
| add α=3 | +5.68 | +8.21 | +3.34 | +0.21 | +0.53 |
| add α=4 | +6.02 | +9.18 | +2.97 | +0.37 | +0.32 |
| add_own | +3.36 | +3.75 | +2.89 | +0.22 | +0.55 |
| add_gc_removed α=1 | +1.80 | +2.98 | +1.30 | +0.05 | +0.45 |
| add_cone_removed α=1 | +3.03 | +3.33 | +2.95 | +0.05 | +0.41 |
| cross_gene α=1 | +2.94 | +3.58 | +2.51 | +0.32 | +0.37 |
| panel_same_k5 | +3.44 | +3.60 | +2.88 | +0.08 | +0.51 |
| panel_other_k5 | +2.01 | +2.94 | +1.64 | +0.18 | +0.09 |
| random α=1 | −0.26 | −0.39 | −0.19 | +0.16 | −0.12 |

Layer 24, α=1: ΔA_cov +0.47, ΔL +0.73, ΔC +0.38, ΔK +0.19, Δexcess −0.20.

Three readings follow directly.

**Coverage is not the story.** ΔK never exceeds +0.37 pp on any arm, against a per-gene SD of ~1.9 pp.
The alignment denominator is stable, so the recovery comparison is like-for-like. That disposes of the
fourth failure mode in the brief.

**Both L and C rise, so on the raw decomposition this is not purely "less human".** At α=1, L rises
+3.51 pp and C rises +2.35 pp: departures from the human base become more frequent *and*, among those
departures, the platypus base is picked more often. On the private set C stays positive across the
whole ladder.

**But C stops rising and the site set matters.** ΔC peaks at α=2–3 (+3.3) and falls at α=4 (+2.97)
while L keeps climbing to +9.18 — the classic "A ↑, L ↑, C ↓" signature appearing at the top of the
ladder. On the **looser** site set the signature arrives much earlier: ΔC is +0.95 at α=1, +0.16 at
α=3 and **−0.29 at α=4**, with ΔL +7.57. So on the pairwise site set, high-dose steering raises
recovery purely by mutating away from human.

**Per-gene spread swamps the means.** ΔA_cov at α=1 is +2.68 pp with a per-gene SD of **8.08 pp** and
60.3 % of genes improved; at L24 α=1 the median gene change is exactly **0.00** and only 49.5 % of
genes improve. These are small mean shifts inside a very wide distribution.

*Data:* `site_directionality/summary.csv` (all 21 arms × both site sets, with median/SD/IQR/fraction
improved for every quantity), `per_gene.csv`, `per_record.csv.gz`.
*Figure:* `figures_27/18_leave_human_vs_platypus_choice.png` (and `_platy_not_human`), `figures_24/…`.

---

## 5. The composition-matched null — how much of the gain survives

Within each gene × condition the observed base calls are permuted among covered private sites,
preserving **the human starting base and the codon position** (12 strata), 1000 times. That holds the
substitution spectrum fixed and destroys only the site-to-base pairing.

**excess recovery = observed A_cov − expected A_cov under the shuffle**

| arm | raw ΔA_cov | Δ null | Δexcess | excess as share of the raw gain |
|---|---|---|---|---|
| add α=0.5 | +1.21 | +1.20 | +0.02 | **1 %** |
| add α=1 | +2.68 | +2.41 | +0.26 | **10 %** |
| add α=2 | +4.71 | +4.28 | +0.43 | **9 %** |
| add α=3 | +5.68 | +5.14 | +0.53 | **9 %** |
| add α=4 | +6.02 | +5.70 | +0.32 | **5 %** |
| add_own | +3.36 | +2.81 | +0.55 | **16 %** |
| add_gc_removed α=1 | +1.80 | +1.35 | +0.45 | **25 %** |
| panel_same_k5 | +3.44 | +2.93 | +0.51 | **15 %** |
| L24 add α=1 | +0.47 | +0.67 | **−0.20** | negative |

**This is the central result. Between 84 % and 99 % of the headline recovery gain is reproduced by a
null that knows only the steered substitution spectrum and nothing about which site is which.** At
α=1 the +2.68 pp gain is +2.41 pp null and +0.26 pp excess. Excess does not scale with dose: it peaks
at +0.53 pp (α=3) and falls at α=4, while the raw gain keeps climbing. At layer 24 α=1 the excess is
*negative* — the raw +0.47 pp gain is smaller than its own composition null predicts.

Excess is also not zero for a sequence with no platypus information: the shuffled-human control reads
**+12.2 pp** excess, because the pairwise aligner itself selects for matches at every site. So excess,
like A, is only interpretable as a paired change — which is what the table reports.

Per-gene, Δexcess at α=1 is +0.26 pp with SD **3.67 pp** and 53.3 % of genes improved. At the gene
level that is indistinguishable from a coin flip.

*Data:* `site_directionality/summary.csv` (`d_excess`, `d_A_null` and their spread),
`figures_27/20_excess_vs_raw_recovery.csv`.
*Figure:* `figures_27/20_excess_vs_raw_recovery.png`.

---

## 6. Where in the site set the gain lives — the GC diagnostic

Private sites split by what the human → platypus substitution does to GC. Class sizes per gene:
GC-increasing 23.4, GC-decreasing 15.8, GC-neutral 16.9.

ΔA (pp) by class, private set, layer 27:

| arm | GC-increasing (A/T→G/C) | GC-decreasing (G/C→A/T) | GC-neutral (A↔T, G↔C) |
|---|---|---|---|
| add α=0.5 | +4.12 | −3.88 | +1.92 |
| add α=1 | **+9.24** | **−7.17** | +2.79 |
| add α=2 | +16.03 | −12.48 | +3.87 |
| add α=3 | +19.29 | −15.08 | +5.18 |
| add α=4 | **+21.10** | **−17.86** | +5.73 |
| add_own | +9.84 | −7.20 | +3.85 |
| **add_gc_removed α=1** | **−0.70** | **+4.52** | +1.37 |
| random α=1 | −0.99 | −0.40 | +0.68 |
| L24 add α=1 | +1.55 | −1.77 | +1.14 |

Fraction of genes improved at α=1: 82 % on GC-increasing sites, **22 %** on GC-decreasing sites.

The steering arms gain enormously wherever the platypus base is the GC-richer one and lose almost as
much wherever it is the AT-richer one. The net +9.24 / −7.17 / +2.79 combination, weighted by the class
sizes, is the whole +2.68 pp. **The mechanism is a GC push, not a platypus push.**

The exception is informative. `add_gc_removed`, which projects the GC axis out of the steering vector,
**flips the sign**: it gains +4.52 pp on GC-decreasing sites and loses nothing on GC-increasing ones —
the only arm whose gain is symmetric in GC. It also carries the highest excess share (25 %). Its total
gain is smaller (+1.80 pp), but a larger fraction of it is the kind that cannot be explained by
composition.

GC-neutral sites gain a consistent +1.4 to +5.7 pp. This cannot be a GC effect by construction, and
it is the second piece of evidence, alongside `add_gc_removed`, that something beyond the GC axis is
present — small, but not zero.

*Data:* `site_directionality/gc_class.csv`, `figures_27/19_gc_class_recovery.csv`.
*Figure:* `figures_27/19_gc_class_recovery.png`.

---

## 7. Composition of the generations against the real CDS windows

Every statistic computed on the same window per gene, for the human reference, the platypus reference
and every generation. Fractions, means over 398 genes.

| | human ref | unsteered | L24 α=1 | L27 α=1 | L27 α=4 | platypus ref |
|---|---|---|---|---|---|---|
| overall GC | 0.5160 | 0.5562 | 0.5803 | 0.6550 | 0.8221 | **0.5571** |
| GC1 | 0.5652 | 0.5824 | 0.5989 | 0.6667 | 0.8279 | 0.5804 |
| GC2 | 0.4105 | 0.4792 | 0.4966 | 0.5476 | 0.7325 | 0.4246 |
| GC3 | 0.5721 | 0.6071 | 0.6455 | 0.7507 | 0.9057 | 0.6663 |
| CpG O/E | 0.4228 | 0.4913 | 0.5688 | **0.6895** | 0.9590 | **0.6405** |
| Ts/Tv vs human | — | 0.7089 | 0.6797 | 0.6065 | 0.5612 | **1.3633** |
| substitution rate vs human | 0 | 0.4699 | 0.4783 | 0.4946 | 0.5630 | **0.2280** |
| f(A) | 0.2608 | 0.2344 | 0.2225 | 0.1829 | 0.0901 | 0.2431 |
| f(C) | 0.2520 | 0.2579 | 0.2733 | 0.3085 | 0.3601 | 0.2758 |
| f(G) | 0.2640 | 0.2983 | 0.3070 | 0.3466 | 0.4620 | 0.2814 |
| f(T) | 0.2232 | 0.2094 | 0.1972 | 0.1621 | 0.0879 | 0.1998 |

GC / GC3 / GC1 / GC2 and the codon-usage divergences all reproduce the published §3 table to the
reported digit (unsteered 55.6/60.7, L27 α=1 65.5/75.1, L27 α=4 82.2/90.6, platypus 55.7/66.6, human
51.6/57.2).

Four things stand out.

**The unsteered model already matches platypus GC.** Unsteered overall GC is 0.5562; the platypus
reference is 0.5571. The two species are only 4.1 GC points apart (51.6 vs 55.7), and the unsteered
generations sit *on* the platypus value. Steering to α=1 moves GC to 0.6550 — **past** platypus by
more than twice the entire human–platypus gap, and to the 92nd percentile of the human window
distribution. On GC, steering does not close a gap; it opens one.

**CpG O/E is the one statistic that lands.** Human 0.423, platypus 0.641, unsteered 0.491, L27 α=1
**0.690** — close to platypus and only slightly past it. α=4 (0.959) overshoots badly. This is a real
directional hit, and the only clean one in the table.

**Ts/Tv moves the wrong way.** The real platypus-vs-human ratio is 1.363, the strong transition bias
every mammalian lineage shows. Unsteered generations sit at 0.709, and steering pushes them *further
down* — 0.607 at α=1, 0.561 at α=4. Steering makes the substitution process less like real evolution
monotonically with dose.

**The generations are not between the two species.** The unsteered generations already differ from
the human window at a substitution rate of 0.470, against a real human–platypus divergence of 0.228 —
**twice** the actual evolutionary distance, before any steering. Steering raises it to 0.563.

*Data:* `composition_profile/scalars_by_condition.csv` (every scalar × every condition, with the
percentile of each condition mean inside both species' per-gene distributions),
`features_per_gene.csv.gz`, `reference_windows.csv`, `dinucleotide_relative_abundance.csv`,
`codon_usage_rscu.csv`.
*Figures:* `figures_27/21_composition_vs_reference_distributions.png`,
`figures_27/22_dinucleotide_relative_abundance.png`.

---

## 8. Distance to each species — both distances, always

Jensen–Shannon divergence in bits, against **that gene's own** human and platypus windows.

| | d_H dinuc | d_P dinuc | d_H codon | d_P codon | d_H syn-usage | d_P syn-usage | d_H 4-mer | d_P 4-mer |
|---|---|---|---|---|---|---|---|---|
| JSD(human, platypus) | — | 0.0179 | — | 0.0979 | — | 0.1034 | — | 0.1257 |
| unsteered | 0.0675 | 0.0761 | 0.2201 | 0.2423 | 0.1464 | 0.1667 | 0.2527 | 0.2762 |
| L24 α=1 | 0.0762 | 0.0784 | 0.2345 | 0.2458 | 0.1548 | 0.1652 | 0.2691 | 0.2789 |
| L27 α=1 | 0.0882 | 0.0784 | 0.2598 | **0.2447** | 0.1824 | **0.1658** | 0.2909 | 0.2749 |
| L27 α=2 | 0.1357 | 0.1102 | 0.3279 | 0.2851 | 0.2269 | 0.1843 | 0.3662 | 0.3209 |
| L27 α=4 | 0.2074 | 0.1689 | 0.4072 | 0.3524 | 0.2572 | 0.2053 | 0.4703 | 0.4052 |

**The scale bar reframes everything.** The two species are 0.098 bits apart in codon usage. The
unsteered generation is **0.220 bits from human** — 2.2× the entire between-species distance — and
0.242 from platypus. In dinucleotide space it is 3.8× the species separation from human. The
generations do not live between human and platypus; they live far outside the segment joining them,
in a direction neither species occupies.

**Within that, steering moves away from human much faster than toward platypus.** From unsteered to
L27 α=1: d_H rises 0.220 → 0.260 (+0.040 codon) while d_P falls only 0.2423 → 0.2447 — in fact d_P
*rises* marginally in codon space and falls by 0.0009 in synonymous usage. The condition does cross
the "closer to platypus than to human" diagonal (d_P < d_H at α=1 and above), but it does so by
receding from human, not by approaching platypus. By α=4 both distances are 2–4× the species
separation.

The one exception is synonymous codon usage, where d_P falls slightly (0.1667 → 0.1658 at α=1) while
d_H rises 0.1464 → 0.1824 — the largest genuinely-toward-platypus movement anywhere in the table,
and it is 0.0009 bits against a 0.1034-bit species separation, i.e. **0.9 % of the way**.

*Data:* `composition_profile/jsd_by_condition.csv`, `figures_27/23_jsd_to_each_species.csv`.
*Figure:* `figures_27/23_jsd_to_each_species.png`.

---

## 9. The substitution spectrum against the real one

The 12 classes counted human → generated, with the real platypus-vs-human spectrum on the same genes
as the yardstick.

| class | unsteered | L27 α=1 | L27 α=4 | **real platypus** |
|---|---|---|---|---|
| A>G *(ts)* | 0.114 | 0.138 | 0.187 | **0.157** |
| T>C *(ts)* | 0.092 | 0.112 | 0.125 | **0.181** |
| C>T *(ts)* | 0.078 | 0.049 | 0.023 | **0.106** |
| G>A *(ts)* | 0.081 | 0.052 | 0.022 | **0.111** |
| A>C | 0.085 | 0.114 | 0.143 | 0.079 |
| C>G | 0.099 | 0.112 | 0.138 | 0.075 |
| T>G | 0.084 | 0.108 | 0.157 | 0.054 |
| G>C | 0.085 | 0.100 | 0.100 | 0.077 |

Resemblance to the real spectrum, all arms ranked:

| condition | JSD to real spectrum | Pearson r | transition share |
|---|---|---|---|
| real platypus | 0 | 1.000 | **0.555** |
| add α=0.5 | 0.0314 | 0.619 | 0.362 |
| random α=1 | 0.0316 | 0.752 | 0.368 |
| **unsteered** | **0.0332** | **0.712** | 0.366 |
| add α=1 | 0.0384 | 0.531 | 0.353 |
| add_own | 0.0392 | 0.526 | 0.353 |
| add_gc_removed α=1 | 0.0430 | 0.457 | 0.350 |
| add α=2 | 0.0595 | 0.484 | 0.351 |
| add α=4 | **0.0901** | **0.457** | 0.355 |

**Every steered arm at α ≥ 1 is further from the real platypus substitution spectrum than the
unsteered baseline, and the random arm is closer to it than any steered arm.** Pearson r against the
real spectrum falls monotonically with dose, 0.712 → 0.531 → 0.457. The reason is visible class by
class: real platypus evolution is transition-dominated (55.5 % transitions), and steering *suppresses*
two of the four transition classes — C>T falls 0.078 → 0.023 and G>A falls 0.081 → 0.022 — while
inflating the three GC-increasing transversions A>C, C>G and T>G. The transition share stays flat at
~0.35 across every arm and never moves toward 0.555.

*Data:* `composition_profile/substitution_classes.csv`, `substitution_spectrum_agreement.csv`.
*Figure:* `figures_27/24_substitution_spectrum.png`.

---

## 10. Verdict

Against the four readings the brief set out, on the private site set at layer 27:

- **ΔK ≈ 0 on every arm** — the recovery improvement is not an artefact of a changing alignment
  denominator. That control passes cleanly.
- **A ↑, L ↑, C ↑** at α ≤ 3, so on the raw decomposition the platypus base *is* preferentially
  selected among departures. Taken alone this reads as the brief's "genuine platypus-directed
  steering" case.
- **The composition null overturns it.** 84–99 % of the raw gain is reproduced by permuting the same
  base calls among sites within (human base, codon position). Excess recovery is +0.26 pp of a
  +2.68 pp gain at α=1, does not scale with dose, is negative at layer 24, and has a per-gene SD 14×
  its own mean. The rise in C is what a GC-biased substitution process produces at a site set where
  60 % of the GC-directional sites (23.4 of 39.2 per gene) happen to be GC-increasing.
- **At high dose on the looser site set the failure mode is explicit:** ΔC goes negative at α=4 while
  ΔL reaches +7.6 — recovery rising only because departures got more frequent.

The composition profile says the same thing three more ways: the unsteered model already sits at the
platypus GC value and steering pushes it 2× the species gap past it; the generations are 2–4× the
entire human–platypus distance away from both species in every frequency space measured, so they are
off the axis rather than on it; and the steered substitution spectrum moves monotonically *away* from
the real platypus-vs-human spectrum, suppressing exactly the transitions that dominate real mammalian
divergence.

Two findings point the other way and are worth following:

1. **`add_gc_removed` inverts the GC signature.** Projecting the GC axis out of the steering vector
   flips the sign of the class asymmetry (+4.52 pp on GC-decreasing sites, −0.70 on GC-increasing) and
   yields the highest excess share of any arm (25 %). The residual, GC-free component of the steering
   direction behaves differently in kind from the bulk of it.
2. **GC-neutral sites gain consistently** (+2.79 pp at α=1, +5.73 at α=4), which no GC mechanism can
   explain, and **CpG O/E lands near the platypus value** at α=1 (0.690 vs 0.641) — the one scalar
   where the steered arm arrives rather than overshoots.

Both are small. Both survive the GC explanation. The productive next step is to characterise the
steering direction with GC projected out — the arm that already exists — rather than to push dose on
the current vector, which buys raw recovery and loses everything else.

Per the framing of the request, amino-acid and exact-nucleotide identity to the platypus ortholog were
not used as success criteria anywhere above.

---

## 11. Hypothesis → results-file index

| question | answer | file |
|---|---|---|
| Does the metric measure what it claims (structural controls)? | yes, exactly | `site_directionality/controls.csv` |
| Does this reproduce the published +2.65 / +3.40 pp? | yes, to the digit | `site_directionality/summary.csv` (`d_A_all`) |
| Is the gain an artefact of alignment coverage? | no, ΔK ≤ +0.37 pp | `site_directionality/summary.csv` (`d_K`) |
| Is the output merely less human (A↑ L↑ C flat)? | no — C rises too, at α ≤ 3 | `figures_27/18_leave_human_vs_platypus_choice.png` |
| Does recovery rise only via more departures (A↑ L↑ C↓)? | yes at α=4 on the looser site set | `summary.csv`, site_set = `platy_not_human` |
| Is the gain explained by the substitution/composition bias? | **yes, 84–99 % of it** | `figures_27/20_excess_vs_raw_recovery.csv` |
| Is any of the gain beyond that bias? | +0.26 pp of +2.68 pp; per-gene SD 3.67 | `summary.csv` (`d_excess*`) |
| Is the gain a GC effect? | yes — +9.2 pp GC-up vs −7.2 pp GC-down | `site_directionality/gc_class.csv` |
| Does anything survive GC removal? | yes — sign flips, 25 % excess share | `gc_class.csv`, row `add_gc_removed_a1.0` |
| Where does each condition sit between the species (scalars)? | see percentiles | `composition_profile/scalars_by_condition.csv` |
| Does composition move toward platypus (frequency vectors)? | away from human, ~1 % toward platypus | `composition_profile/jsd_by_condition.csv` |
| Does the substitution spectrum look like real divergence? | no, and worse with dose | `composition_profile/substitution_spectrum_agreement.csv` |
| Full per-record / per-gene numbers | — | `site_directionality/per_record.csv.gz`, `per_gene.csv`, `composition_profile/features_per_gene.csv.gz` |

**Code.** `strat/site_directionality.py`, `strat/site_directionality_figures.py`,
`strat/composition_stats.py`, `strat/composition_profile.py`,
`strat/composition_profile_figures.py`; site definition extended in `strat/sites_nt.py`
(`site_pairs_nt`); 40 unit tests in `strat/test_composition_stats.py`.
