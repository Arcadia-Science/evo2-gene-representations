# Forensic reproducibility and correctness audit

**Audit date:** 2026-09-01 UTC  
**Audited working tree:** branch `pipeline-refactor`, `HEAD` `5089d47bd50aed758ce896af036d8cc968608bf4`  
**Primary publication material:** `pub/` (named `pubs/` in the request)  
**Overall verdict:** **MOSTLY**  
**Confidence:** moderate

## Executive conclusion

The principal descriptive results can be traced to stored artifacts and, in several important cases, independently recomputed. All 14 current publication figure stems in `pub/figures/` are byte-for-byte identical (PNG and PDF) to the source renders identified by `pub/figures/MANIFEST.md`. The Experiment 1 baseline correlations, representative optimal-transport matrices, Experiment 2 control-preservation curves, Experiment 3 leave-one-out directions, steering deltas, dose responses, site-directionality summaries, and GC-by-codon-position values agree with independent calculations from their immediate upstream artifacts. One valid 348-bp Evo2 locus was also re-embedded from the current raw sequence and current checkpoint; all 32 x 4096 cached values reproduced bit-for-bit.

That is not a full end-to-end reproduction. The working tree is substantially dirty, `pub/`, `results/`, and much of the relevant documentation are untracked or ignored, and the stochastic generation stage is not exactly reproducible: `stage4_steer.py` records a seed but does not seed model sampling, appends heterogeneous resume runs into one table, and overwrites the only configuration file on every invocation. Several manuscript statements are stale or do not match the implemented method. Most seriously, the manuscript says pooled representations are L2-normalized, while the steering direction is constructed from unnormalized means, and the synonymous-recoding control uses family-specific codon distributions, which can itself encode the family label. These issues weaken the strongest causal interpretation (“protein-level information”) and prevent exact regeneration of the steering outputs, but they do not erase the descriptive patterns present in the stored data.

The appropriate rating is therefore **MOSTLY**, not YES: most central numerical patterns are real in the supplied artifacts and major deterministic calculations check out, but exact end-to-end reproducibility and some stated interpretations require material correction.

## Scope and evidentiary limits

- This audit examined the present working tree, not a clean immutable release. `git status` shows many modified/deleted tracked files and untracked `pub/`, `tests/`, `REPRODUCING.md`, and result notes. Consequently, commit `5089d47…` does not identify the audited state.
- Raw/intermediate storage is large (`data/` about 155 GB, `results/` about 25 GB, `deprecated/` about 46 GB). I checked schemas, counts, identities, selected raw-to-derived paths, and all publication render identities; I did not rerun every multi-day embedding or generation job.
- The publication PDF audited was `pub/Beyond sequence statistics_ Testing gene family organization in DNA language model representations (2).pdf`. It is visibly a draft: it includes placeholders such as “X way,” “80-something-percent,” empty media source fields, and unfinished alt text.
- The request referred to `pubs/`; this repository has `pub/`.
- Stored artifacts are evidence of internal consistency, not proof that the original raw acquisition was unbiased or that stochastic generations can be recreated.

## Reproduction and verification performed

### Environment and tests

- Python 3.12.10 in `/home/ubuntu/uv/venv`.
- Key installed versions: Evo2 0.5.5, PyTorch 2.11.0+cu128, flash-attn 2.8.3, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, POT 0.9.7, Biopython 1.87.
- `uv.lock` provides hashes for Python dependencies. External bioinformatics programs are not uniformly locked; stage-5 metadata does record IQ-TREE 2.3.6, while other stages rely on tools such as MAFFT, FastTree, MMseqs2, trimAl, and PAML.
- `uv run --no-sync pytest -q -p no:cacheprovider`: **66 passed**, two non-failing POT warnings.
- `bash -n` passed for the experiment runners.
- Test coverage is concentrated on control generation and optimal transport. There are no direct tests for raw locus extraction, masking, Evo2 pooling, steering hooks, stochastic generation, figure-data joins, or manuscript consistency.

### Publication figure provenance

For each of Figure 1, 2, 3, 4, 5, 6a, 6b, 7, 8, 9a, 9b, 10, 11, and 12, both publication PNG and PDF have the same SHA-256 digest as the source render described in `pub/figures/MANIFEST.md`. This proves that the publication-facing copies came from those source renders without a hidden image-editing step. It does not prove that each source render was built from the claimed raw inputs.

The manifest mapping is coherent with `REPRODUCING.md` and the three experiment runners. Figure 12 has a special provenance note: its paired-P3 angular summary was originally staged outside `results/`; a corresponding source render now exists in the run output. Former geodesic versions of Figures 1 and 12 remain beside the current versions as `(OLD)` files, reducing the risk of silent replacement but increasing the need to identify the exact manuscript copy.

### Experiment 1: family geometry

- `complete_manifest.csv`: 12,294 rows, 48 families, 1,061 ortholog groups, 24 species, no duplicate group/species rows; 10,293 rows marked analyzable and all of those pass QC. Species counts are unbalanced.
- CDS-mask report: 15,055 `ok`, 1,679 `unmappable`, one `no_seq`; 11,288 target keys are used.
- Natural embedding cache: 11,292 arrays, including four stale keys. Each control cache contains exactly 11,288 arrays.
- Independent recomputation of raw CDS k-mer and GC distance-matrix correlations against all 32 stored W2 matrices agreed to maximum absolute error `8.33e-17`.
- Independent recomputation of W2 matrices from cached embeddings for Hox (720 loci), globins (98), and Ras GTPases (489) at layer 12 agreed to maximum absolute error `1.67e-16`.
- A current raw locus, `OR5AC2__felis_catus` (348 CDS bases), was passed through the current Evo2 7B checkpoint (`bda0089f92582d5baabf0f22d9fc85f3588f6b58`) and current pooling code. The resulting `(32, 4096)` array was bit-for-bit equal to the cache (`max_abs=0`). The run used the loader's BF16 fallback because Transformer Engine is absent.
- Figure 1 summary has 96 rows (32 layers x 3 baselines). Pfam correlation ranges 0.114–0.570 and peaks at layer 12; k-mer ranges 0.161–0.644 and peaks at layer 1; GC ranges 0.163–0.686 and peaks at layer 0. This supports the broad depth-localization claim.
- The publication figure was built with zero Mantel permutations, so all Mantel p-values are NaN. The manuscript does not appear to make a p-value claim for that panel, but the absence of inference should be explicit.

### Experiment 2: composition controls

- Independent recomputation of every stored between-family control-preservation value from natural/control OT matrices across all layers agreed to maximum absolute error `1.11e-16`.
- Between-family preservation ranges: synonymous 0.883–0.997 (mean 0.972), 6-mer 0.539–0.988, 4-mer 0.522–0.988, dinucleotide 0.528–0.985, GC 0.520–0.983.
- The manuscript's strict ordering “6-mer > 4-mer > dinucleotide > GC” is not consistent at every layer. The complete ordering occurs at 16/32 layers; 6-mer exceeds 4-mer at 22/32, 4-mer exceeds dinucleotide at 25/32, and dinucleotide exceeds GC at 32/32. It is a broad tendency, not a universal ranking.
- Within-family middle-layer behavior is broadly as stated: synonymous preservation is about 0.60–0.72 over the main middle band, while ordinary controls are mostly around 0.4–0.6.
- The control identity table contains all 11,288 sequences per rung. Synonymous recoding changes 22.819% of bases, preserves amino acids exactly, and puts 81.839% of changes at codon position 3, agreeing with the manuscript's 22.9% and 81.7% rounded claims.
- Paired-P3 analysis uses matched eligible sites and a matched change rate. At layer 18 the mean missense-minus-synonymous preservation gap is -0.0849; missense is lower in 42/48 families and the paired Wilcoxon p-value is `6.35e-9`.

### Experiment 3: platypus steering

- Stage 1 has exactly 400 genes, 400 unique homology blocks, 80 per stratum, with no duplicates. Actual identity ranges are 50.17–64.50, 64.71–72.83, 72.91–80.47, 80.67–87.96, and 88.10–100%, consistent with the rounded bins in the manuscript.
- No selected row has the stated opossum wrong-species control (`has_opossum` is false throughout).
- Stage 2 records 400 genes, 12 windowed genes, 6,000-bp windows with 1,000-bp overlap, and retained-coverage quantiles 0.485/0.900/0.996.
- Stage 3 has 12,800 per-gene/layer diagnostic rows (400 x 32). Independently recomputed leave-one-out vectors at layers 0, 12, 24, and 27 differ from saved float32 vectors by at most `2.98e-8`. The implementation correctly excludes the focal gene (`scripts/steering/platypus/stage3_select.py:67-89`).
- Blocks 30–31 are numerically degenerate (pooled and direction norms around `1e12`); the primary layer 27 choice avoids these blocks. This exclusion is encoded in plotting/selection code but should be stated in the manuscript.
- Stage 4's scoring plan starts with 400 genes but only 398 are usable. All principal plotted conditions cover these 398. Figure 7 therefore has stratum counts 80, 80, 79, 80, 79, not 80 each.
- `stage4_scores_nt.csv`: 48,484 rows, 398 genes, 23 conditions, no duplicate gene/condition/sample keys. It mixes layer-24 and layer-27 interventions.
- `generations.jsonl.gz`: 48,481 records. Three scored records have no matching generation record. This is small but means generation-derived and score-derived analyses are not based on precisely identical rows.
- Independent reconstruction of Figure 8 agrees with its CSV. At alpha 4, private-site improvement by stratum is +4.221, +5.143, +4.157, +6.860, and +9.949 percentage points; amino-acid identity changes by -2.398, -5.021, -6.702, -8.577, and -17.074 points.
- Current absolute private-site levels differ from prose: stratum 0 is 39.81 to 44.03%, not about 42 to 46%; stratum 4 is 31.69 to 41.64%, not about 36 to 44%. Amino-acid prose is close to current data.
- Figure 7's alpha-1 gene-bootstrap summaries reproduce: private-site correctness +2.652 pp (95% bootstrap CI 1.874–3.423), amino-acid identity -1.777 pp (-2.710 to -0.863), indel fraction -0.002 pp (-0.123 to 0.121), and stop density -0.782 per 100 codons (-0.952 to -0.634).
- Site directionality reproduces the qualitative claim. Across alpha 0.5, 1, 2, 3, 4, private-site `d_L` is 0.890, 3.510, 6.322, 8.214, 9.175, while `d_C` is 1.560, 2.353, 3.321, 3.338, 2.968. The loose non-human site set does not strengthen at high dose. These are descriptive summaries without confidence intervals or formal inference.
- GC claims reproduce from the cached codon table: unsteered GC1/GC2/GC3 are 58.24/47.92/60.71%; at alpha 4 they are 82.79/73.25/90.57%. Platypus reference GC3 is 66.63%.
- Stage 5 is incomplete but transparently records attrition: 399/400 trees are `ok`; dN/dS has 383/399 `ok` and 16 one-hour timeouts. Figure 11 must therefore not be described as an inference over all 400 genes for every predictor.

## Findings ranked by severity

### HIGH 1 — Stochastic steering generations cannot be exactly reproduced

**Evidence.** `scripts/steering/platypus/strat/stage4_steer.py:125` exposes `--seed`, but the seed is used only for constructed random directions and panel sampling (`:327-365`). The call to `model.generate` at `:416-422` supplies temperature and top-k but no generator/seed, and no `torch.manual_seed`, CUDA seed, NumPy global seed, or Python seed is set before generation. The saved config nevertheless records the seed at `:498`.

The same output directory is deliberately reused with `--resume` for multiple arms. Existing rows are read and appended, then `stage4_config.json` is unconditionally overwritten with only the most recent invocation's layers, arms, alphas, decoding parameters, and seed (`:468-502`). The current directory named `stage4_cds_mean_blocks27` contains both layer-24 and layer-27 conditions, while its current config says only layer 24, `gc_removed`, alpha 1. The table has missing `temperature`/`top_k` for 27,404/48,484 historical rows.

**Impact.** Exact regeneration of the causal steering figures is impossible from the recorded provenance, even with the same code and checkpoint. Different sampling RNG state can change every continuation. The current config cannot establish the parameters used for most previously appended conditions.

**Recommended correction.** Seed Python, NumPy, PyTorch CPU, and each CUDA device immediately before every condition/sample, preferably with a deterministic per-key seed derived from `(run, gene, condition, sample)`. Record checkpoint digest, RNG seed, package/hardware path, prompt, and decoding parameters in every generation record. Use immutable per-invocation config files or a run ledger; never overwrite provenance when resuming into a mixed table.

### HIGH 2 — Manuscript normalization does not match the steering method

**Evidence.** The manuscript says pooled representations are L2-normalized after mean pooling. Experiment 1 scoring normalizes before angular/OT calculations (`scripts/mammalian_orthologs/mammal_score.py:120-122` and the OT code), so its reported correlations are protected. Steering is not: `Pooler.pooled()` returns raw means (`scripts/steering/platypus/strat/stage2_embed.py:119-130`), and `stage3_select.py:67-89` subtracts those raw human and platypus means to form `D` and `v_-i`. There is no L2 normalization between these operations.

**Impact.** This is not cosmetic. Subtracting normalized vectors and subtracting raw vectors produce different directions and magnitudes, hence potentially different generated sequences and dose-response curves. Either the manuscript method or the implemented intervention is wrong.

**Recommended correction.** Decide which representation is scientifically intended, rerun steering if normalization was intended, and describe each experiment separately if Experiment 1 uses normalized vectors while Experiment 3 uses raw residual means. Add a unit test that asserts the exact representation transformation used to build `D`.

### HIGH 3 — Synonymous recoding leaks family-specific composition into the family-control task

**Evidence.** Codon usage is estimated separately for each labeled family from all target sequences (`scripts/controls/make_control_sequences.py:179-192`; `embed_cds_masked_mammal.py:216-222`). Each sequence is then recoded by sampling synonymous codons from its own family's distribution (`make_control_sequences.py:195-208`). Thus the transformation is given the target family label and writes a family-specific nucleotide signature into every recoded sequence.

**Impact.** Near-perfect reconstruction of *between-family* geometry is not uniquely attributable to preserved protein sequence. It can also arise because the control explicitly preserves/reinjects family-level codon composition. The paired-P3 missense comparison is useful corroboration, but its synonymous and missense alternatives are not matched for nucleotide identity/composition and therefore does not fully remove this confound. The manuscript's claim that the result is “difficult to explain unless the model represents protein-level information” is too strong.

**Recommended correction.** Add recodes drawn from a global codon distribution, leave-one-family-out distributions, species-only distributions, and per-sequence synonymous permutations; compare them with the family-conditioned control. State that the current result supports protein-preserving and family-composition-preserving information jointly, not protein information uniquely.

### MEDIUM 1 — The within-family quantity in the manuscript is not the implemented statistic

**Evidence.** The manuscript defines a scalar within-family mean pairwise distance. The code instead computes a Spearman matrix correlation separately within each ortholog group, then averages those correlations within family (`scripts/mammalian_orthologs/mammal_score.py:106-143`). The 6,112-row summary reflects 48 families for sequence/species baselines and 47 for patristic distance.

**Impact.** Readers cannot reproduce Figure 2 from the stated formula, and the statistic has a different interpretation and weighting.

**Recommended correction.** Replace the manuscript equation with the per-group upper-triangle Spearman calculation and family-level mean, including minimum-pair filters and missing-family handling, or change the analysis to match the stated scalar.

### MEDIUM 2 — Claims that indels and premature stops remain steady are false for the stored data

**Evidence.** The manuscript says indel burden and premature stops do not shift significantly and later says they are held “roughly steady across steering strengths.” At alpha 1, stop density decreases by 0.782 per 100 codons with a gene-bootstrap 95% CI of -0.952 to -0.634; every stratum's interval is negative. Overall indel fraction is steady at alpha 1, although stratum 4 increases by 0.353 pp (CI 0.038–0.694), and higher-dose raw comparisons show additional shifts.

**Impact.** The coding-coherence narrative is misdescribed. Reduced stops may be favorable, but it is not “no shift”; high-dose indel behavior also needs explicit reporting.

**Recommended correction.** Report stop density as a significant decrease, report indel estimates by dose and stratum, and avoid equating a bootstrap interval excluding zero with a multiple-testing-adjusted hypothesis test.

### MEDIUM 3 — Several manuscript numbers and population descriptions are stale

**Evidence.** Current Experiment 1 curve correlations are: k-mer vs species-tree median 0.916, range 0.0009–0.991 (n=48), rather than 0.913 and 0.173–0.988; k-mer vs gene-tree median 0.891, range -0.249–0.993 (n=47), rather than 0.862 and -0.256–0.993. Experiment 3 prose repeatedly says 400 analyzed genes, but principal generation figures use 398. Private-site absolute percentages differ by about 2–4.3 points from prose. “Other 24 mammals” is also incorrect: the panel has 24 mammals total, including human and platypus, and per-site voting depth is often below the theoretical maximum.

**Impact.** Qualitative trends remain, but precise claims and denominators are unreliable.

**Recommended correction.** Generate manuscript tables directly from frozen figure-data CSVs and fail CI on mismatched n/ranges. State attrition and effective species voting depth in each caption.

### MEDIUM 4 — Pfam baseline is not self-contained or release-pinned

**Evidence.** `scripts/baselines/pfam_hmm_jsd.py` fetches 48 HMMs live from InterPro and records neither release, response digest, nor a frozen HMM bundle. The active run retains final correlations but not the exact Pfam baseline matrix beside the W2 results. A local Pfam database exists, but the active code does not use it. A live reconstruction attempt reached only 34/48 requests during the audit window.

**Impact.** The Pfam curve cannot presently be regenerated exactly and can drift when InterPro/Pfam records change.

**Recommended correction.** Vendor or content-address the 48 HMM files, record accession versions and response hashes, and save the exact baseline matrix used by the figure.

### MEDIUM 5 — Paired-P3 layer scope and inference are underdocumented

**Evidence.** `paired_p3_figure.py` excludes blocks 28–31 as degenerate and reports 28 layers, although the manuscript generally describes all layers. Of 28 nominal Wilcoxon tests, 26 have uncorrected p<0.05 but only 19 survive simple Bonferroni correction. The paired arms use separate deterministic RNG streams and distinct alternative-base distributions.

**Impact.** The main protein-vs-nucleotide result remains strong at layer 18, but “only protein differs” and all-layer language are overstated.

**Recommended correction.** Disclose excluded layers, report adjusted intervals/p-values, and add nucleotide-composition-matched paired arms.

### MEDIUM 6 — External acquisition and some failures can collapse to absence

**Evidence.** Ensembl requests retry broad exceptions and eventually return `None` (`scripts/mammalian_orthologs/resolve_orthologs.py:56-71`), making persistent network/server failure observationally similar to a definitive missing ortholog unless downstream logs are retained. Several pipeline stages catch broad exceptions and continue; stage 4 does at least write failures to `failures.log` (`stage4_steer.py:461`).

**Impact.** Dataset attrition can depend on transient services. Current manifests make final counts inspectable, but raw acquisition is not independently replayable without frozen responses.

**Recommended correction.** Distinguish HTTP-not-found, exhausted retry, parse error, and biological absence in manifests; cache raw API responses with release and digest; make unexpected attrition fail a completeness gate.

### LOW 1 — “W2” storage is the squared transport objective

**Evidence.** The OT code uses squared angular ground cost and stores POT's `emd2` objective without taking a square root. That is W2-squared, while manuscript notation mixes angular cost and W2 terminology.

**Impact.** A square root is monotone, so the reported Spearman correlations are unchanged. Numeric distances and method labeling are nonetheless inconsistent.

**Recommended correction.** Call the stored quantity `W2_squared` or take the square root and update formulas/metadata.

### LOW 2 — Cache and record hygiene is imperfect

**Evidence.** The natural embedding cache contains four keys no longer in the active manifest. Three score rows lack corresponding generation JSONL records. Figure codon statistics reuse `figures/13b_codon_substitution_stats.csv`; it covers all 398 genes for the principal conditions but includes a few partial layer-24 dose conditions (2, 1, 1, 1 genes), which downstream condition filters must continue to exclude.

**Impact.** Current primary summaries appear unaffected, but naive globbing or aggregation can silently include stale/partial data.

**Recommended correction.** Store manifest digests in each cache, reject extra keys by default, and add one-to-one generation/score and condition-completeness assertions.

### LOW 3 — The publication draft is not release-ready

**Evidence.** The PDF contains placeholders, incomplete media markup, missing alt text, imprecise wording, and numeric placeholders. Figure 11 and stage-5 analyses are present in the publication bundle but barely integrated into the narrative.

**Impact.** Readers cannot reliably distinguish final claims from drafting notes.

**Recommended correction.** Freeze a manuscript source linked to the exact figure/data manifest, remove placeholders, and archive the resulting PDF digest.

## Scientific-claim assessment

| Claim | Assessment | Basis |
|---|---|---|
| Evo2 family geometry tracks Pfam/sequence/species structure with depth dependence | **Supported descriptively** | Stored matrices and independently recomputed correlations agree; Pfam source itself is not frozen. |
| Synonymous recoding preserves between-family geometry far better than ordinary composition controls | **Supported as measured** | Recomputed preservation is about 0.96–1.00 over much of the middle model. |
| The synonymous result uniquely demonstrates protein-level representation | **Not established** | Family-conditioned codon usage leaks the family label/composition; paired-P3 helps but does not fully match nucleotide distributions. |
| A leave-one-gene-out platypus direction emerges around layer 27 without focal-gene leakage | **Supported for stored embeddings** | LOO construction is correct and independently recomputed; raw-vs-normalized method mismatch remains. |
| Steering increases platypus-private recovery and reduces amino-acid identity | **Supported for stored generations** | Gene-level deltas, CIs, and dose/stratum patterns reproduce. Exact generation cannot be replayed because sampling is unseeded. |
| Steering moves toward platypus rather than merely away from human | **Partially supported** | Private-site directionality shows the claimed dose shape; looser sites do not. Summaries are descriptive and composition is strongly entangled. |
| Coding validity is maintained with steady indels/stops | **Mixed/worded incorrectly** | Alpha-1 overall indels are stable, but stop density clearly decreases and some strata/doses shift. |
| GC rises fastest at the third codon position | **Supported** | Exact cached means reproduce 61→91% GC3, 58→83% GC1, 48→73% GC2. |

## Reproducibility checklist

| Component | Status |
|---|---|
| Python dependency lock | Good (`uv.lock` with hashes) |
| Audited source snapshot | Poor (large dirty/untracked working tree) |
| Raw biological input provenance | Partial (release metadata exists, live API failure modes remain) |
| Model checkpoint identity | Partial (resolved snapshot observed during audit; not consistently stored per artifact) |
| Deterministic embedding | Good in targeted check (bit-for-bit one-locus reproduction) |
| Deterministic controls | Good (stable per-key RNG design and complete caches) |
| Deterministic steering generation | **Failed** (sampling RNG not seeded) |
| Configuration provenance | Partial/failed for mixed stage 4; better for stages 0–3 and 5 |
| Figure-to-source-render traceability | Excellent (all publication files byte-identical to mapped renders) |
| Source-render-to-upstream-table traceability | Good for tested figures; not universally enforced by hashes |
| Automated tests | Passing but narrow |
| Manuscript/data synchronization | Weak (stale values, denominator and method mismatches) |

## Prioritized remediation

1. Freeze the audited source, manuscript, raw manifests, model snapshot, and artifact manifest in one clean tagged release.
2. Repair stage-4 RNG and provenance, then rerun the main generation grid; this is required for exact reproducibility.
3. Resolve the raw-vs-L2-normalized steering discrepancy and rerun if the paper's stated method is intended.
4. Add global/leave-one-family-out synonymous controls before retaining the strongest protein-level interpretation.
5. Regenerate all manuscript numbers and denominators directly from frozen CSVs; correct stop/indel wording.
6. Freeze Pfam HMM inputs and Ensembl responses with versions and hashes.
7. Add integration tests for extraction orientation/masks, pooling, LOO exclusion, generation seeding, score/generation joins, condition completeness, and manuscript figure-data contracts.

## Final rating

**MOSTLY.** The repository contains substantial, internally coherent evidence for its major descriptive patterns, and deterministic parts of the pipeline survive independent numerical checks—including an exact raw-sequence-to-embedding spot reproduction. However, the current materials do not support a YES because the core stochastic generation stage lacks effective seeding and immutable configuration provenance, the manuscript misstates normalization and within-family calculations, the synonymous control confounds the strongest protein-specific conclusion, and several published numbers/denominators are stale. The likely scientific direction is credible; exact end-to-end reproduction and the strongest mechanistic interpretation remain unresolved pending the corrections above.
