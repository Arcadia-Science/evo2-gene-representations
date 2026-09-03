# Quantitative audit of the current manuscript

**Audited artifact:** `pub/Beyond sequence statistics_ Testing gene family organization in DNA language model representations (9).pdf`  
**SHA-256:** `f2a7dd56c83026e360f1bff69a53bddbc20766254d758a94cd47ebffd7d68a50`  
**Audit date:** 2026-09-02  
**Scope:** every numerical, statistical, sample-size, layer, parameter, trend, comparative, and figure-linked claim in the current PDF. Historical audits in the repository were treated only as leads and were not accepted as evidence without checking the current artifacts. Source files, results, figures, and manuscript were not changed.

PDF locations below are line numbers in a layout-preserving extraction made with:

```bash
pdftotext -layout "pub/Beyond sequence statistics_ Testing gene family organization in DNA language model representations (9).pdf" /tmp/manuscript9.txt
```

## Executive assessment

The current analysis products are substantially more internally consistent than the older repository state: all current Experiment 2 figure tables use the same **400 genes**, and the current Experiment 1 figures use **angular distance** and true **2-Wasserstein distance**. The old 103-gene, 398-gene, geodesic, and squared-Wasserstein-era artifacts are not feeding the current frozen figure tables.

The manuscript is nevertheless **not quantitatively publication-ready**. The most important defects are:

1. **Two exact stratum-4 steering values are from the superseded 398-gene analysis.** Current 400-gene values are amino-acid identity **36.43% to 19.03%**, not **35.66% to 18.59%**. The current unsteered private-site value is **31.45%** (rounds to 31.5), not 31.7. The old values reproduce the manuscript numbers.
2. **The shared representation methods do not describe Experiment 2.** Experiment 2 pools the whole CDS from 6,000-bp overlapping windows and uses raw mean vectors; it does not use the Experiment 1 second-half pooling, 8,000-bp threshold, 24-window cap, or L2-normalized vectors.
3. **The claimed Ensembl ortholog confidence filters are absent from the implemented pipeline.** The code requires `ortholog_one2one`, but the reported `>25% identity`, confidence, synteny/WGA, and tree-compliance criteria are neither stored nor applied.
4. **The GC3 reference interpretation is reversed.** Unsteered generated GC3 has mean **60.68%** and median **61.57%**, both below the platypus CDS reference of **66.67%**; the manuscript says the median starts above it.
5. **“Indel burden and premature stops don't shift significantly” is false as written.** At alpha=1, premature stops decrease in every stratum with gene-bootstrap intervals excluding zero; stratum-4 indel burden increases modestly with an interval excluding zero. The safer conclusion, supported by the data, is that steering did not substantially *increase premature stops* and had limited effects on indels.
6. **Several Experiment 1 summary bounds are stale or imprecise.** The current k-mer/species-tree curve range is **0.0009–0.9912**, not 0.001–0.988. The k-mer/gene-tree calculation has **47**, not 48, evaluable families and ranges **−0.2488–0.9934**, not −0.256–0.993.
7. **Exact reruns are not fully reproducible.** Generation sampling has no recorded PyTorch/model-generation seed, the Stage 4 configuration file was overwritten and hand-corrected, and several external inputs are not frozen by content hash/release.

The table below contains **93 atomic claim checks**: **48 VERIFIED**, **8 VERIFIED — ROUNDING**, **10 INCORRECT**, **2 WRONG EXPERIMENT VERSION**, **9 METHOD/TEXT MISMATCH**, **8 AMBIGUOUS PROVENANCE**, and **8 UNVERIFIED**. Severity assignments are editorial judgments: “critical” changes the stated experiment or primary evidence; “major” can change interpretation or reproducibility; “minor” is a bounded wording, rounding, caption, or bookkeeping defect.

## Evidence map and audit conventions

| Code | Evidence used |
|---|---|
| `M` | Current PDF and `/tmp/manuscript9.txt` extraction |
| `E1-panel` | `scripts/families_data.json`, `data/mammalian_orthologs/family_sizes.csv`, `complete_manifest.csv`, embedding cache, and final per-group score tables |
| `E1-code` | `scripts/mammalian_orthologs/{resolve_orthologs,assemble_datasets,embed_cds_masked_mammal,mammal_score}.py` and `scripts/baselines/ot_between_family.py` |
| `E1-fig` | `figure_data/exp1_between_family_by_layer.csv`, `figure_data/exp1_within_family_by_layer.csv`, and matching publication PDFs |
| `E2-panel` | frozen sampling order/configuration, `figure_data/exp3_panel.csv`, and Stage 1/2/3/4 gene identifiers |
| `E2-code` | `scripts/steering/platypus/strat/stage{0,1,2,3,4}_*.py` and figure builders |
| `E2-fig` | current `figure_data/exp3_*` tables and matching publication PDFs |
| `R` | independent regrouping/recalculation from the row-level or per-gene frozen data |

“Verified” means the claim follows from the checked current artifact at the stated analysis grain. “Verified — rounding” means the unrounded result supports the printed value. “Wrong experiment version” is reserved for a number that matches a superseded cohort. “Ambiguous provenance” means a plausible artifact exists but the exact manuscript-to-artifact link or external input cannot be proven. “Unverified” is used when the repository does not contain adequate evidence.

## Master per-claim audit table

| ID | PDF line(s) | Claim | Audit evidence / recomputation | Status | Severity | Required correction |
|---|---:|---|---|---|---|---|
| Q001 | 59 | Evo2 has 40B parameters | No model card/source is resolved by the manuscript's `:ref` placeholder or frozen locally. | UNVERIFIED | minor | Resolve the citation and state the precise model variant. |
| Q002 | 62 | NT work found >100 annotation-associated features | Referenced external study is not identified in the rendered PDF. | UNVERIFIED | minor | Repair citation and verify wording against the primary source. |
| Q003 | 63 | A1408G steering result | External result; no quantitative source is recoverable from the rendered citation. | UNVERIFIED | minor | Repair citation. |
| Q004 | 65 | 5-mer-preserving decoys recover roughly 85% | External result; no source/result is frozen locally. | UNVERIFIED | minor | Repair citation and define the measured quantity. |
| Q005 | 77 | Goodfire post was August 2025 | Bibliography includes a 2025 entry, but the rendered reference link is unresolved. | UNVERIFIED | minor | Add a resolvable citation/date. |
| Q006 | 82–83 | 1–4-mer XGBoost predicts activations at ~0.9 correlation | External result and correlation definition are not available locally. | UNVERIFIED | minor | Cite the primary result and name the correlation/statistical unit. |
| Q007 | 109 | Two main experiments | Current manuscript contains Experiment 1 and Experiment 2; the former composition-control experiment is no longer presented. | VERIFIED | minor | None. |
| Q008 | 116 | Evo2-7B base model | Experiment scripts/configuration consistently target the 7B model. | VERIFIED | minor | None. |
| Q009 | 116 | All experiments used one NVIDIA A10G | No machine log or immutable run metadata demonstrates this for every artifact. | AMBIGUOUS PROVENANCE | minor | Cite run logs or qualify as the intended hardware. |
| Q010 | 117–118 | Inputs contain UTRs, introns, exons on gene strand; coding states retained | Experiment 1 manifest/extraction and CDS masks support this workflow. Experiment 2 instead embeds CDS sequence. | METHOD/TEXT MISMATCH | major | Scope this paragraph to Experiment 1 and separately describe Experiment 2. |
| Q011 | 119–120 | Inputs >8,000 bp tiled, max 24 windows | This is the Experiment 1 embedding policy. Experiment 2 uses 6,000-bp windows with 1,000-bp overlap and no 24-window policy. | METHOD/TEXT MISMATCH | major | Scope to Experiment 1; report the Experiment 2 window policy. |
| Q012 | 121–122 | Second-half mean pooling | Implemented for Experiment 1. Experiment 2 uses whole-CDS mean pooling. | METHOD/TEXT MISMATCH | critical | Separate the experiment-specific representation definitions. |
| Q013 | 123 | Vectors L2-normalized | Experiment 1 distance functions normalize vectors; Experiment 2's steering differences are computed from raw means. | METHOD/TEXT MISMATCH | critical | Do not state this as shared preprocessing. |
| Q014 | 144,158 | 48 HGNC families/groups | `families_data.json` and final figure tables contain exactly 48 families. | VERIFIED | minor | None. |
| Q015 | 159–161 | 1,144 intended human paralogs | `family_sizes.csv` sums to 1,144 HGNC members across the 48 families. | VERIFIED | minor | Clarify that this is the intended query panel, not the analyzed human N. |
| Q016 | 146–147,162–166 | 24 mammals, Ensembl release 116; 6 primates, 4 glires, 10 laurasiatherians, 4 others | Species list has 24 entries with the printed partition; download/config code records release 116. | VERIFIED | minor | None. |
| Q017 | 161–163 | Every one of the 1,144 genes was queried by Ensembl ID | Resolver iterates all configured family members, but a complete immutable query-response ledger for all 1,144 is absent. | AMBIGUOUS PROVENANCE | minor | Preserve request status for every intended gene. |
| Q018 | 167–170 | Only “high-confidence” orthologs with >25% identity plus synteny/WGA/tree compliance were retained | Resolver only tests `type == ortholog_one2one`; final tables lack identity, confidence, synteny, WGA, and tree-compliance fields. Extraction does not add them. | METHOD/TEXT MISMATCH | critical | Implement and freeze these filters or delete the claim. |
| Q019 | 170 | At most one ortholog per species | Resolver stores species records by species key; current manifest has no duplicate group/species pairs. | VERIFIED | minor | None. |
| Q020 | 170–172 | Protein-coding, valid CDS, no excessive Ns/gaps | QC code checks protein-coding status, frame/start/stop, N fraction ≤0.02, and maximum gap run <50, with documented soft length bounds. | VERIFIED | minor | Report the actual thresholds. |
| Q021 | 172 | Groups with <10 species excluded | Final score input has 611 groups with at least 10 represented species. | VERIFIED | minor | Report 611 analyzed groups and attrition. |
| Q022 | 158–173 | Dataset implied to comprise 1,144 human paralogs plus orthologs | Actual matched embedding/manifest intersection is 11,288 loci, 1,034 groups, and 815 human genes; the scored ≥10-species set is 9,262 loci in 611 groups. | METHOD/TEXT MISMATCH | major | Add a sampling/attrition table and distinguish intended, resolved, embedded, and analyzed N. |
| Q023 | 148–150 | All 32 layers; two geometry scales | Frozen between-family table is 32×3; within-family table covers 32 layers and 48 families. | VERIFIED | minor | None. |
| Q024 | 150–151,184–197 | Between-family metric is 2-Wasserstein on empirical family distributions | Code uses uniform empirical weights, squared angular ground costs, `ot.emd2`, then square root. Metadata records POT 0.9.7, 11,288 genes, and `cost=theta^2`. | VERIFIED | critical | None; retain metadata with release artifact. |
| Q025 | 189–190 | Displayed Wasserstein equation | The rendered equation has a comma between `pi_ij` and `d_ang^2`, not multiplication, making the expression malformed. | INCORRECT | major | Replace comma with multiplication/adjacency and define uniform masses. |
| Q026 | 200–206 | Angular distance is arccos(dot)/pi in [0,1] for normalized vectors | Implementation clips cosine, applies arccos/pi, and normalizes inputs where used. | VERIFIED | minor | None. |
| Q027 | 209–218 | Within-family score is per-ortholog-group Spearman, then family mean | Current angular score pipeline implements this hierarchy. | VERIFIED | major | State how undefined group correlations are omitted. |
| Q028 | 228 | Pfam baseline is a 20-dimensional amino-acid emission profile | Current baseline construction produces 20 amino-acid dimensions. | VERIFIED | minor | None. |
| Q029 | 228–230 | Pfam profile is reproducibly defined | The active provenance does not pin an immutable Pfam/InterPro content hash/release, while local and live-query paths coexist. | AMBIGUOUS PROVENANCE | major | Freeze the exact Pfam input and content hash. |
| Q030 | 232–234 | Species tree averages patristic distance over 100 posterior trees | Tree builder loads/prunes the posterior tree collection and averages pairwise distances; the configured collection has 100 trees. | VERIFIED | minor | Record the input-tree hash. |
| Q031 | 237–238 | Gene trees use MAFFT v7 and FastTree v2 | Pipeline invokes those major versions, but exact binaries/minor versions are not frozen per group. | AMBIGUOUS PROVENANCE | minor | Record command lines and exact versions. |
| Q032 | 239–242 | GC fraction and overlapping 6-mers at 1-bp step | Baseline implementations match these definitions. | VERIFIED | minor | None. |
| Q033 | 261–265 | Figure 1 has three baselines for all 32 layers | Frozen table has 96 rows: 32 each for Pfam, GC, and 6-mer. Publication PDF matches the current source PDF by hash. | VERIFIED | minor | None. |
| Q034 | 269–270 | Edge-layer Pfam rho ranges ~0.11–0.26 | Across the visually low edge set, current minimum is 0.1139 and maximum is 0.2689; the upper value rounds to 0.27, not 0.26. | INCORRECT | minor | Print ~0.11–0.27 or define exactly which layers count as edges. |
| Q035 | 270–272 | Layers 10–24 show stronger Pfam than compositional baselines | For every layer 10–24, Pfam rho (0.4198–0.5698) exceeds both GC and 6-mer rho. | VERIFIED | major | None. |
| Q036 | 273–278 | Family ordering varies; adrenoceptor favors gene tree, GPx favors 6-mer, peroxidase baselines are similar | Across-layer means support the first two comparisons. Peroxidase means are close (0.415–0.465), though “consistently ... across all layers” is stronger than the data warrant. | VERIFIED | major | Change “consistently ... across all layers” to “similar on average.” |
| Q037 | 281–283 | K-mer/species-tree curve correlation median 0.916 over 48 families | Independent Spearman calculation: n=48, median 0.916040. | VERIFIED — ROUNDING | minor | None. |
| Q038 | 282–283 | K-mer/species-tree range 0.001–0.988 | Current range is 0.0009166–0.9912007. Lower bound rounds to 0.001; upper bound does not match. | INCORRECT | major | Replace with 0.001–0.991. |
| Q039 | 281,283–284 | K-mer/gene-tree curve statistic is over all 48 families | One family lacks sufficient gene-tree values; only 47 correlations are defined. | INCORRECT | major | State n=47 and identify missingness handling. |
| Q040 | 283–284 | K-mer/gene-tree median 0.891 | Independent calculation over the 47 defined families gives 0.890742. | VERIFIED — ROUNDING | minor | None. |
| Q041 | 283–284 | K-mer/gene-tree range −0.256–0.993 | Current range is −0.248763–0.993401. | INCORRECT | major | Replace with −0.249–0.993. |
| Q042 | 296–300 | Figure 3 contains sequence identity plus four named baselines | Current frozen table and publication PDF show four baselines: GC, 6-mer, species tree, and gene tree. No sequence-identity series is plotted. | INCORRECT | major | Remove “sequence-identity” from the caption or add the series. |
| Q043 | 304–305 | Data include representations/distances for all 48 families | Current score/figure data include 48 families, but exact externally shared files cannot be identified from the rendered callout. | AMBIGUOUS PROVENANCE | minor | Link an immutable release manifest. |
| Q044 | 363–370 | Platypus is the sister lineage/outgroup and private sites are defined relative to the mammal panel | Species set/tree topology supports platypus as the monotreme outgroup in this panel. | VERIFIED | minor | Prefer “phylogenetically most distant from human in this panel.” |
| Q045 | 378–380 | Experiment 2 uses Ensembl 116 and MMseqs2 release 15; at most one gene/block | Frozen Stage 0 config records release 116, clustering parameters, 6,961 blocks, and representative selection; panel has 400 distinct blocks. | VERIFIED | major | Record exact MMseqs command/version hash. |
| Q046 | 381–385 | Five equal bins, boundaries 64.68/72.89/80.48/88.09, 80 sampled per quintile | Frozen frame boundaries are 64.6778, 72.8907, 80.4829, 88.0866; final panel has 80 genes in each stratum. | VERIFIED — ROUNDING | major | None. |
| Q047 | 385–386 | All 400 have complete in-frame canonical CDS and a 90-bp homologous indel-free prompt near the start | Stage 1 gates retain 400; prompts are 90 bp with no alignment gaps and human start codon ≤30. The exact biological “canonical/complete” status depends on upstream Ensembl exports but is recorded. | VERIFIED | major | None. |
| Q048 | 394–397 | Earlier 1,500–2,000-bp prompts are >11× longer than 90 bp | 1,500/90=16.7 and 2,000/90=22.2; “>11×” is mathematically true but oddly loose. | VERIFIED | minor | Prefer 17–22×. |
| Q049 | 407–410,418 | Steering panel has n=400, five strata | Every current panel, geometry, steering, directionality, and codon table has the same 400-gene identifier set; 80 per stratum. | VERIFIED | critical | None. |
| Q050 | 418–430 | Per-gene whole-CDS contrastive mean; leave-one-out direction | Stage 2 whole-CDS mean tensors and Stage 3 formulas reproduce all 12,800 per-gene/layer values; maximum LOO discrepancy is 1.33e−15. | VERIFIED | critical | None. |
| Q051 | 435 | alpha=1 is one full mean shift; alpha=0.5 is half | Intervention code scales the held-out vector by alpha. | VERIFIED | minor | None. |
| Q052 | 437–444 | Geometry uses LOO cosine and magnitude spread | Frozen layer table contains per-gene LOO cosine and CV of difference norms, matching the definitions. | VERIFIED | major | Define “spread” explicitly as CV in the text. |
| Q053 | 447–448 | LOO cosine is near zero through layers 9–20 | Medians range from 0.300 at L9 and 0.169 at L10 down to ~0.098–0.132 for L11–20. “Near zero” is defensible for most, not all, of the interval. | INCORRECT | minor | Say “falls to ~0.10–0.17 across most of L10–20; L9 is 0.30.” |
| Q054 | 448 | LOO cosine rises around L21 and saturates near 1 at L28–31 | It rises from 0.142 (L21) to 0.248, 0.377, 0.605...; L28–29 are ~0.87 and L30–31 are ~0.991. | VERIFIED | minor | “Approaches saturation by L28–29 and reaches ~0.99 at L30–31” is more exact. |
| Q055 | 453 | Magnitude spread is high at L9–23 and minimum at L27 | CV declines from 2.045 (L9) to 1.505 (L23), then 0.585, 0.478, 0.337, and 0.241 at L27; L27 is the minimum among L0–27. | VERIFIED | major | None. |
| Q056 | 454 | L27 has the highest LOO cosine before final layers | L0 median is 0.78446, slightly above L27's 0.77493. L27 is the highest in the late-middle candidate band, not literally all pre-final layers. | INCORRECT | major | Define candidate band/exclusion rule; say “highest late-middle-layer value.” |
| Q057 | 467–475 | L27 combines high alignment and lowest pre-final spread | Supported given the intended late-middle selection and exclusion of L28–31; the exclusion is a scientific judgment, not a data-derived gate. | VERIFIED | major | State the pre-specified or post hoc selection rule. |
| Q058 | 482–489 | L27 injection at every position; 90-bp first gap-free 30-codon window; 1,000-bp generation; T=0.7; alpha={0.5,1,2,3,4}; unsteered control | Stage 4 row metadata/config and conditions support these values. Decoder also uses `top_k=4`, omitted from the manuscript. | VERIFIED | critical | Add `top_k=4` and number of generations per gene. |
| Q059 | 486–489 | Exact generation procedure is reproducible | Temperature/top-k are stored, but no model/PyTorch sampling seed or generator is passed to `model.generate`; config was overwritten and later hand-corrected. | AMBIGUOUS PROVENANCE | critical | Freeze full run configs, code commit, per-batch seeds, and generated-record hashes. |
| Q060 | 504–521 | Four outcomes and their definitions | Frozen score columns implement private recovery, global translated AA identity, generated-bp indel percentage, and premature stops per 100 codons. | VERIFIED | major | State gap/coverage and stop-count eligibility rules. |
| Q061 | 517–518,583 | Private bases are unique to platypus across the 24-mammal panel / other 23 mammals | Definition is correct in principle, but per-gene available voting species range 2–22 other non-focal mammals (median 19); missing species are ignored. | METHOD/TEXT MISMATCH | major | Say “unique among available aligned bases” and report voting-depth distribution. |
| Q062 | 506–509,552–553 | At alpha=1, private recovery increases in every stratum | Gene-level mean deltas are +1.482, +2.229, +1.950, +2.692, +5.005 percentage points. | VERIFIED | major | None. |
| Q063 | 509,554–556 | AA response becomes more negative with conservation except stratum 3 | Alpha=1 deltas are +0.059, −0.956, −2.357, −0.614, −5.135 points; this matches the stated exception. | VERIFIED | major | None. |
| Q064 | 556–557 | Response spread widens and most-conserved genes are most variable | SD-of-delta values are largest in stratum 4 for both private recovery and AA identity; stratum 3 interrupts a strictly monotone trend. | VERIFIED | minor | Say “generally widens,” not monotone. |
| Q065 | 557–558 | Indels and premature stops do not shift significantly across strata | Gene bootstrap at alpha=1 finds stop deltas below zero in all five strata with 95% intervals excluding zero; stratum-4 indel delta is +0.349 points with interval approximately +0.04 to +0.69. | INCORRECT | major | Report direction and intervals; avoid a blanket nonsignificance claim. |
| Q066 | 560–561,697–698 | Steering does not substantially increase frameshifts/stops | Stops decrease; indel changes are small except a modest stratum-4 increase. The narrower “does not substantially increase” conclusion is supported. | VERIFIED | major | Use this narrower formulation consistently. |
| Q067 | 541–543 | Figure 8 alpha ranges from 0.5 to 4% | Alpha is a dimensionless vector multiplier, not a percentage. | METHOD/TEXT MISMATCH | minor | Delete the `%` sign after 4. |
| Q068 | 543 | Stronger steering progressively increases private recovery | At alpha=4, strata 1 and 2 dip slightly from their alpha=3 maxima; trend is broadly dose-increasing but not strictly progressive in every stratum. | METHOD/TEXT MISMATCH | minor | Say “generally increases, with plateaus/slight reversals at the highest dose.” |
| Q069 | 552–555 | Private gain is positive in every stratum and mostly grows with conservation | Alpha=4 deltas are +4.221, +5.143, +4.272, +6.860, +10.176 points; all positive, broadly larger at high conservation. | VERIFIED | major | None. |
| Q070 | 566–567 | Stratum 0: private 39.8→44.0; AA 17.17→14.77 | Current 400-gene means are 39.807→44.028 and 17.169→14.771. | VERIFIED — ROUNDING | major | None. |
| Q071 | 568 | Stratum 4 private 31.7→41.6 | Current means are 31.451→41.627. The printed 31.7 is the old 398-gene value (31.691); 41.6 rounds correctly in both versions. | WRONG EXPERIMENT VERSION | critical | Replace with 31.5→41.6 and regenerate prose directly from current table. |
| Q072 | 568–569 | Stratum 4 AA 35.66→18.59 | Current means are 36.432→19.025. The printed pair exactly matches the old 398-gene analysis. | WRONG EXPERIMENT VERSION | critical | Replace with 36.43→19.03. |
| Q073 | 569 | More-conserved genes get more private-base-like while losing more AA identity at high dose | Alpha=4 effects are largest in stratum 4 for both private gain and AA loss. | VERIFIED | major | None. |
| Q074 | 583–595 | L and C are defined on covered private sites as leave-human and choose-platypus-among-departures | Per-gene directionality table and analysis code implement these conditional rates. | VERIFIED | major | State handling of zero-denominator genes. |
| Q075 | 614–615,627–630 | Private-site L rises monotonically; C rises to alpha 2–3 then declines at 4 | Mean L deltas: 0.910, 3.574, 6.378, 8.298, 9.266. Mean C deltas: 1.585, 2.333, 3.345, 3.400, 3.004. | VERIFIED — ROUNDING | major | None. |
| Q076 | 617–620 | Random control is per-gene Gaussian, norm-matched, same draw reused across dose | Intervention constructs one random direction per gene, rescales to target norm, and multiplies that vector across doses. | VERIFIED | major | Record random seeds in immutable run metadata. |
| Q077 | 620,635–640 | Random controls show no comparable dose response | Private-site random deltas remain near zero through available matched doses (L roughly −0.57 to −0.30; C −0.22 to +0.09). | VERIFIED | major | None. |
| Q078 | 642–644 | Less specific platypus-not-human sites do not show the same C response | C deltas across alpha 0.5–4 are 0.959, 0.932, 0.506, 0.156, −0.330, unlike private sites. | VERIFIED — ROUNDING | major | None. |
| Q079 | 661–664 | GC1/2/3 endpoints: 58.21→82.71, 47.88→73.08, 60.68→90.46 | Current 400-gene gene-weighted means are 58.2057→82.7118, 47.8772→73.0801, and 60.6771→90.4593. | VERIFIED — ROUNDING | major | None. |
| Q080 | 661–664 | GC rises at every codon position and fastest at GC3 | Endpoint increases are +24.51, +25.20, and +29.78 points for GC1, GC2, and GC3. | VERIFIED | major | None. |
| Q081 | 663–664 | Median unsteered GC3 starts above platypus CDS level 66.67% | Unsteered generated GC3 median is 61.57% (mean 60.68%); platypus reference mean is 66.668% and median 67.186%. Generated GC3 starts below either reference summary. | INCORRECT | critical | Reverse the statement and specify whether reference line is mean or median. |
| Q082 | 664 | Overall GC scales monotonically with alpha | Overall means are 55.585, 60.619, 65.430, 74.129, 78.874, 82.100% from unsteered through alpha=4. | VERIFIED — ROUNDING | major | None. |
| Q083 | 692–700 | Steering preferentially increases private bases and induces a broad compositional shift without substantially increasing stops/indels | Private-vs-loose site analysis and GC dose response support preferential/base-compositional effects; safety wording is supported only in the narrower “not substantially increase” sense. | VERIFIED | major | Retain narrow language and add uncertainty. |
| Q084 | 702–705 | Both main questions receive “strongly supported yeses” | Experiment 1 within-family evidence is explicitly heterogeneous; Experiment 2 loses AA identity and is confounded by a strong GC shift. This is an interpretation stronger than the quantitative evidence. | UNVERIFIED | major | Temper to a qualified, experiment-specific conclusion. |
| Q085 | Fig. 1 | Current visual derives from current angular/Wasserstein data | Publication PDF hash matches the current angular/Wasserstein source figure; frozen rows use `approach=wasserstein`. | VERIFIED | critical | None. |
| Q086 | Fig. 3 | Current visual derives from current angular within-family data | Publication PDF hash matches the current angular source figure; builder reads files with `_angular` names. | VERIFIED | critical | Fix caption per Q042. |
| Q087 | Figs. 5–10 | Current steering visuals use one 400-gene cohort | Gene-set digest is identical across panel, geometry, Stage 3 diagnostics, raw scoring plan/scores, steering, site-directionality, and codon tables. | VERIFIED | critical | None. |
| Q088 | Embedded media | Drive assets in the PDF map uniquely to local publication files | The PDF exposes Google Drive IDs, while local manifest/scripts identify plausible matching outputs; Drive-ID-to-local-file identity is not frozen in a manifest. | AMBIGUOUS PROVENANCE | major | Add source path and SHA-256 for every embedded figure/diagram. |
| Q089 | Figure captions | Diagram figures are quantitatively auditable | Experiment overview diagrams have no identified local source asset or data-bearing manifest. | UNVERIFIED | minor | Store diagram sources and hashes. |
| Q090 | Current release | No 103-gene artifacts feed current results | The 103-gene digest occurs only under `figure_data/old_paired103` and the historical Stage 1 branch; no current frozen figure table shares it. | VERIFIED | critical | Move historical data under an explicit archive directory. |
| Q091 | Current release | No 398-gene artifacts feed current figures | All current primary tables use the same 400 genes. Pre-400/398 products remain elsewhere and explain Q071–Q072, but are not current figure inputs. | VERIFIED | critical | Archive or clearly label every 398-gene artifact. |
| Q092 | Current release | No old geodesic/squared-Wasserstein artifacts feed current figures | Current builder selects angular within-family files and `wasserstein`; old geodesic/centroid outputs and `(OLD)` figures remain but are not referenced. Direct between-table recomputation agrees to floating precision. | VERIFIED | critical | Archive old products and enforce manifest validation. |
| Q093 | Frozen data | Row-level current steering outcomes are complete and traceable | Raw score table has 49,220 rows, 400 genes, 23 conditions, no duplicate gene-condition-sample keys. Generation JSONL has 49,217 rows: three `ENSG00000153391/add_own_L24` samples are missing, outside current L27/dose/GC figures. | AMBIGUOUS PROVENANCE | minor | Repair/record the three missing historical records; keep a completeness check. |

## A. Sample-size and cohort audit

### Experiment 1 attrition

The manuscript currently collapses several distinct counts into “1,144 paralogs.” The auditable lineage is:

| Stage | Families | Human genes/groups | All loci | Audit interpretation |
|---|---:|---:|---:|---|
| Curated HGNC target list | 48 | 1,144 intended human genes | — | Supports the printed 1,144 only as the starting list |
| Resolver output with at least one ortholog | 48 | 1,062 groups | — | Query/resolution attrition begins here |
| Current complete manifest | 48 | 1,061 groups / 881 human rows | 12,294 | Manifest contains entries without matching current embeddings |
| Manifest ∩ current embedding cache | 48 | 1,034 groups / 815 human rows | 11,288 | Actual population used by the current between-family OT metadata |
| Groups passing ≥10-species analysis gate | 48 | 611 groups | 9,262 | Actual within-family analysis population |
| Groups with defined gene-tree correlations | 47 families represented | 608 groups | — | Explains n=47 for the curve-pair gene-tree statistic |

There are four stale embedding-cache keys not present in the active manifest. This does not change the current tables because the analysis intersection is explicit, but it is a preventable provenance hazard.

### Experiment 2 panel and row counts

The current panel lineage is coherent:

| Artifact/stage | Genes | Rows / structure | Gene-set result |
|---|---:|---:|---|
| Frozen Stage 0 sampling frame | 6,961 blocks | 6,961 representatives | Source of quintile edges |
| Stage 1 accepted paired panel | 400 | 80 per stratum | 400 unique homology blocks |
| Stage 2 embeddings | 400 | human and platypus, 32 layers | Exact panel match |
| Per-gene direction geometry | 400 | 12,800 = 400×32 | Exact panel match |
| Raw Stage 4 score table | 400 | 49,220 rows, 23 conditions | Exact panel match |
| Frozen steering outcomes | 400 | 9,200 = 400×23 | Exact panel match |
| Site directionality | 400 | 27,600 = 400×23×3 site sets | Exact panel match |
| Current codon/composition table | 400 | 6,800 = 400×17 conditions | Exact panel match |

The current **400-gene digest is `0546fcae925c8e5b`** (first 16 hexadecimal characters of the audit's sorted-ID digest). Historical 103-gene data have digest `4267…`; historical 398-gene products have digest `a0d…`. Neither historical set appears in the current primary figure tables.

## B. Distance-definition and aggregation audit

### Between-family distance

The implemented current metric is:

\[
W_2(A,B)=\left(\min_{\pi\in\Pi(a,b)}\sum_{ij}\pi_{ij}\,d_{ang}(x_i,y_j)^2\right)^{1/2},
\qquad
d_{ang}(x,y)=\arccos(\hat x^T\hat y)/\pi.
\]

Each family member has uniform mass within its family; family sizes need not be equal. The implementation calls `ot.emd2` on the **squared** angular cost matrix and then takes the square root. Therefore the stored current value is W2, not W2-squared. Independent reconstruction agrees with the frozen between-family table to numerical precision (maximum absolute discrepancy on the order of 1e−16).

The manuscript equation must be repaired because the rendered comma after `pi_ij` changes the mathematical expression.

### Within-family distance and aggregation

For each ortholog group and layer, pairwise latent angular distances are correlated by Spearman rho with each baseline distance vector. Family-level curves are means of defined group-level rhos. Across-layer curve-pair summaries then correlate the 32 layer values between two family curves. This multi-stage averaging/correlation is implemented, but the manuscript needs to state omission rules for undefined correlations and the resulting n=47 for gene-tree comparisons.

### Old-vs-current metric isolation

| Risk | Historical artifacts found | Current selection | Audit conclusion |
|---|---|---|---|
| Geodesic vs angular within-family distance | Old geodesic score files and figures remain | Builder reads `_angular` scores | Current figures are angular |
| Centroid/squared-Wasserstein vs W2 | Old centroid/legacy outputs remain | Builder selects `approach == wasserstein`; metadata says squared angular cost + square root | Current figure is W2 |
| 103 vs 400 steering geometry | `figure_data/old_paired103/` remains | Current geometry table has 400 genes×32 layers | Current geometry is 400 |
| 398 vs 400 generation outcomes | Pre-400 products remain | Current score/figure tables share exact 400 set | Figures are 400; prose Q071–Q072 is stale |

## C. Independent recomputation results

All values below were recomputed from the current frozen per-layer, per-gene, or row-level tables rather than copied from figure annotations.

### Experiment 1 summaries

| Quantity | Recomputed current value |
|---|---:|
| Pfam rho, layers 10–24 | 0.419823 to 0.569831 |
| K-mer vs species-tree curve rho, n | 48 |
| K-mer vs species-tree curve rho, median/range | 0.916040 / 0.000917–0.991201 |
| K-mer vs gene-tree curve rho, n | 47 |
| K-mer vs gene-tree curve rho, median/range | 0.890742 / −0.248763–0.993401 |
| Adrenoceptor across-layer mean rho: gene tree / 6-mer / GC / species | 0.77394 / 0.71532 / 0.62870 / 0.59991 |
| Glutathione peroxidase: 6-mer / GC / gene tree / species | 0.68399 / 0.51185 / 0.47349 / 0.36115 |
| Peroxidase: 6-mer / gene tree / GC / species | 0.46505 / 0.43407 / 0.42699 / 0.41545 |

### Experiment 2 geometry

Selected layer medians/CVs illustrate the actual selection landscape:

| Layer | Median LOO cosine | CV of direction magnitude |
|---:|---:|---:|
| 9 | 0.30044 | 2.0448 |
| 10 | 0.16927 | — |
| 12 | 0.11850 | — |
| 20 | 0.09838 | — |
| 21 | 0.14186 | — |
| 23 | 0.37710 | 1.5056 |
| 24 | 0.60467 | 0.5847 |
| 25 | 0.64842 | 0.4776 |
| 26 | 0.71945 | 0.3374 |
| 27 | 0.77493 | 0.2408 |
| 28 | 0.87612 | 0.4288 |
| 29 | 0.87136 | 0.4171 |
| 30–31 | 0.99097 | 0.6162 |

Layer 27 is a defensible **late-middle** choice because it maximizes alignment in that band while minimizing magnitude CV before the late anisotropic regime. It is not literally the highest LOO value among every layer below 28 because layer 0 is 0.78446.

### Steering endpoints by stratum

Gene-weighted current means:

| Stratum | Private unsteered | Private alpha=4 | Delta | AA unsteered | AA alpha=4 | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 39.807 | 44.028 | +4.221 | 17.169 | 14.771 | −2.398 |
| 1 | 36.659 | 41.802 | +5.143 | 19.884 | 14.862 | −5.021 |
| 2 | 35.903 | 40.176 | +4.272 | 22.473 | 15.751 | −6.722 |
| 3 | 37.202 | 44.062 | +6.860 | 23.426 | 14.849 | −8.577 |
| 4 | 31.451 | 41.627 | +10.176 | 36.432 | 19.025 | −17.407 |

This table directly exposes the manuscript's 398-to-400 prose contamination in stratum 4.

### Site-direction and GC dose response

| alpha | Private-site Delta L | Private-site Delta C | Loose-site Delta C |
|---:|---:|---:|---:|
| 0.5 | +0.910 | +1.585 | +0.959 |
| 1 | +3.574 | +2.333 | +0.932 |
| 2 | +6.378 | +3.345 | +0.506 |
| 3 | +8.298 | +3.400 | +0.156 |
| 4 | +9.266 | +3.004 | −0.330 |

| Condition | GC1 | GC2 | GC3 | Overall GC |
|---|---:|---:|---:|---:|
| Unsteered | 58.206 | 47.877 | 60.677 | 55.585 |
| alpha=0.5 | 62.434 | 51.363 | 68.062 | 60.619 |
| alpha=1 | 66.596 | 54.682 | 74.986 | 65.430 |
| alpha=2 | 74.837 | 61.954 | 85.558 | 74.129 |
| alpha=3 | 79.638 | 67.724 | 89.215 | 78.874 |
| alpha=4 | 82.712 | 73.080 | 90.459 | 82.100 |

## D. Figure-to-data provenance

Current local publication figures 1, 3, and 5–10 were compared with the current figure-source outputs. Their file hashes match the corresponding current source PDFs. Figures 2 and 4 are overview diagrams; their editable/local sources were not identified. The rendered manuscript stores Drive URLs rather than a content-addressed mapping, so exact PDF-embedded-image-to-local-source provenance remains ambiguous even where visual/hash-consistent local publication files exist.

| Manuscript figure | Current data/source | Cohort / metric | Audit |
|---|---|---|---|
| Fig. 1 | `exp1_between_family_by_layer.csv` | 32 layers; W2/angular; 3 baselines | Current and internally consistent |
| Fig. 2 | Overview diagram | No data table | Source asset not identified |
| Fig. 3 | `exp1_within_family_by_layer.csv` | Angular; 48 families, 4 baselines | Current; caption incorrectly adds sequence identity |
| Fig. 4 | Overview diagram | No data table | Source asset not identified |
| Fig. 5 | `exp3_panel*`, rate/tree summaries | 400 genes, 80/stratum | Current |
| Fig. 6 | `exp3_direction_*` | 400×32 | Current, not old 103 |
| Fig. 7 | `exp3_steering_outcomes.csv` | 400×23 | Current, not old 398 |
| Fig. 8 | same outcomes, dose subset | 400 genes | Current; alpha caption typo |
| Fig. 9 | `exp3_site_directionality_*` | 400 genes | Current |
| Fig. 10 | `exp3_codon_substitutions.csv`, reference windows | 400 genes | Current; prose misreads reference level |

## E. Reproducibility, ambiguity, and data-hygiene findings

### Critical/major reproducibility gaps

- **Unseeded stochastic decoding.** The panel/random controls have seeds, but token sampling in `model.generate` does not receive a generator and there is no recorded torch RNG state. Existing sequences can be audited; an exact regeneration cannot be guaranteed.
- **Mutable Stage 4 configuration.** `stage4_config.json` represents the last invocation rather than an immutable ledger of all 23 conditions and includes a note that gene/drop counts were corrected by hand after a two-gene addendum. Per-row metadata salvages much of the audit trail, but this is not release-grade provenance.
- **Ortholog filtering mismatch.** Claimed confidence/evolutionary filters are not executable from or evidenced in the frozen tables.
- **External biological inputs not fully pinned.** Ensembl release is recorded, but Pfam/InterPro content, posterior species-tree file, exact MAFFT/FastTree builds, and every remote figure/data release need hashes.
- **Analysis denominators are implicit.** Private-site voting depth, undefined Spearman correlations, bootstrap resampling unit, and metric eligibility filters should be in Methods and figure-data metadata.

### Minor hygiene findings

- Four natural-embedding cache files are stale relative to the active manifest.
- Three historical L24 generation records are absent from JSONL despite score rows; they do not feed the current focal figures.
- Current and obsolete files coexist in high-visibility directories. Labels such as `(OLD)`, `old_paired103`, and `pre400_bak` help, but automated builders should reject non-manifest inputs and releases should physically separate archives.
- The experiment runner still contains a historical 103-gene branch even though current figures use the 400-gene branch. This is confusing but did not contaminate the present outputs.

## Prioritized correction list

### Before scientific review

1. Replace stratum-4 endpoints with **31.5→41.6% private recovery** and **36.43→19.03% AA identity**.
2. Rewrite shared representation Methods into separate Experiment 1 and Experiment 2 pipelines.
3. Either implement/document the asserted Compara identity/confidence/synteny/tree filters or remove them.
4. Correct GC3 interpretation: unsteered generated GC3 starts **below** the platypus CDS reference.
5. Replace the nonsignificance claim for indels/stops with effect estimates and gene-level bootstrap intervals.
6. Correct Experiment 1 ranges and n: **0.001–0.991 (n=48)** and **−0.249–0.993 (n=47)**.
7. Fix Figure 3 caption and the malformed W2 equation.

### Before reproducible release

8. Freeze one machine-readable run manifest per condition containing commit, model revision, all decoder settings, RNG seeds/states, source-data hashes, and output hashes.
9. Publish explicit attrition tables for both experiments and denominator metadata for every figure statistic.
10. Add an embedded-figure manifest mapping manuscript figure/Drive ID to local source path, data table, and SHA-256.
11. Move all 103-, 398-, geodesic-, centroid-, and other superseded artifacts into a read-only historical archive excluded by default builders.
12. Add release assertions that every current Experiment 2 figure shares the frozen 400-gene set and every Experiment 1 figure declares angular/W2 metric semantics.

## Bottom line

The current figures are based on the intended modern pipelines: **400 genes**, **angular within-family distances**, and **true 2-Wasserstein between-family distances**. The main contamination is textual, not graphical: stale 398-gene endpoints, inaccurate current ranges, and methods language inherited from a different representation pipeline. Correcting those items, tightening statistical wording, and freezing the missing run/input provenance would make the quantitative story internally defensible.
