Experiment 1: mammalian gene-family geometry
HGNC families
  → Ensembl ortholog resolution/download
  → locus extraction and dataset assembly
  → cap families + construct CDS masks
  → Evo2 embeddings at 32 layers
  ├─ within-family angular/geodesic scoring
  ├─ between-family centroid-geodesic scoring
  ├─ between-family Wasserstein scoring
  └─ phylogenetic / Pfam / k-mer / GC comparisons
       → Figures 1–3

Experiment 2: sequence controls
Experiment 1 loci
  → synonymous, GC, dinucleotide, k-mer and paired-p3 controls
  → control embeddings
  ├─ graph/geodesic preservation score
  └─ “graph-free” angular/Wasserstein score
       → Figures 4 and 12

Experiment 3: platypus steering
Human/platypus ortholog acquisition
  → pooling + sequence QC
  → embeddings and direction geometry
  → gene/cluster/gate selection
  → activation steering and generation
  → nucleotide rescoring + site directionality
  → evolutionary predictors: trees, dN/dS, gain models
       → Figures 5–11
The authoritative drivers are [Experiment 1](/home/ubuntu/Development/glm-latent-mapping/experiments/exp1_gene_family_geometry.sh), [Experiment 2](/home/ubuntu/Development/glm-latent-mapping/experiments/exp2_controls.sh), and [Experiment 3](/home/ubuntu/Development/glm-latent-mapping/experiments/exp3_steering.sh).
Important data-flow discrepancies
1. Figure 1 does not appear to use Wasserstein distance
The paper and reproduction narrative describe the between-family analysis as Wasserstein/W2. The experiment does calculate W2 in [ot_between_family_sweep.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/ot_between_family_sweep.py), but the Figure 1 call in [exp1_gene_family_geometry.sh](/home/ubuntu/Development/glm-latent-mapping/experiments/exp1_gene_family_geometry.sh) does not pass that output to the figure generator.
Instead, [layer_sweep_summary.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/layer_sweep_summary.py) defaults to between_family_baseline_scores.csv, produced by [mammal_between.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/mammal_between.py) using centroid geodesic distance.
The _graphfree output suffix changes the output name, not the default between-family input.
This needs scientific resolution before geometry code is consolidated.
2. Figure 2 and its inferential analysis use different within-family metrics
The Figure 2 path explicitly selects angular files. However, [within_family_uncertainty.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/within_family_uncertainty.py) reads unsuffixed per_group_scores_{arm}.csv, which comes from the geodesic branch.
Thus the visualization and inferential claims appear to operate on different metrics.
3. The graph-free control scorer still depends on graph-derived artifacts
[controls_score_graphfree.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/controls_score_graphfree.py) imports graph utilities, calculates a connected-neighbor value, and reads the geodesic cache created by [mammal_controls_score.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/mammal_controls_score.py).
Figure 4 only consumes the Wasserstein result, so this dependency and much of the computation are unnecessary.
4. One control-generation path is not reproducible across Python processes
[embed_cds_masked_mammal.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/embed_cds_masked_mammal.py) derives some random seeds from Python’s string __hash__(). Python randomizes string hashes unless PYTHONHASHSEED is fixed, so identical commands may generate different controls in different processes.
This is a correctness issue, not a cosmetic refactor.
Consolidation recommendations
1. Remove definite dead and dormant code
Current files
- [control_ladder.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/control_ladder.py)
- [within_family_permutation.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/within_family_permutation.py)
- [extract_loci.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/extract_loci.py)
- Dead functions in [gene_families.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/gene_families.py), [arcadia_style.py](/home/ubuntu/Development/glm-latent-mapping/scripts/arcadia_style.py), [ot_between_family.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/ot_between_family.py), and [strat_metric.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/strat_metric.py)
What they do
- control_ladder.py defines an unused alternate control ladder.
- within_family_permutation.py is an older generic permutation analysis superseded by the mammalian uncertainty analysis.
- Most of extract_loci.py is an unused REST extraction CLI. The active bulk extractor imports only constants and manifest/QC helpers from it.
- family_members, cmap_with_missing, subsample_sensitivity, and stratum_counts have no callers.
Why problematic
They create false alternative entry points and increase the apparent number of supported analyses. extract_loci.py is particularly misleading because its main workflow is dormant while another file depends on incidental pieces of it.
Proposed destination
Move the small manifest/QC helpers needed by extract_loci_bulk.py into dataset assembly. Remove the uncalled functions.
Files deletable
- control_ladder.py
- within_family_permutation.py
- extract_loci.py, after moving its live helpers
Risk: Low.
Protecting tests
- Import/CLI smoke tests for all experiment-driver commands.
- A tiny locus fixture verifying that manifest and QC output remain identical.
- Static check that every Python entry point is referenced by a driver or documented manual workflow.
2. Centralize control definitions and generation
Current files
- [make_control_sequences.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/make_control_sequences.py)
- [embed_cds_masked_mammal.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/embed_cds_masked_mammal.py)
- [mammal_controls_score.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/mammal_controls_score.py)
- [controls_score_graphfree.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/controls_score_graphfree.py)
- [control_tables.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/control_tables.py)
- [plot_control_wasserstein.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/plot_control_wasserstein.py)
- [paired_p3_figure.py](/home/ubuntu/Development/glm-latent-mapping/scripts/controls/paired_p3_figure.py)
What they do
They independently encode control names, order, labels, colors, eligible sites, generation rules, and scoring behavior.
Why problematic
The same conceptual control has several representations. control_tables.py is a one-consumer indirection. The unstable hash seed also makes one path non-reproducible.
Proposed destination
A focused controls.py containing:
- One CONTROL_SPECS definition.
- Stable seed derivation using a cryptographic digest or CRC.
- Sequence transforms and invariants.
- Shared display labels/order.
- A single scoring function supporting natural, controlled, and paired-p3 comparisons.
Create one experiment2_figures.py for Figures 4 and 12.
Files deletable after migration
- control_tables.py
- controls_score_graphfree.py
- plot_control_wasserstein.py
- paired_p3_figure.py
- Potentially mammal_controls_score.py if its useful scoring is absorbed into the shared scorer
Risk: Low to medium.
Protecting tests
- Exact sequence length and terminal-base preservation.
- Exact k-mer-count preservation where promised.
- Synonymous controls preserve translated protein.
- Paired-p3 controls alter only eligible sites at the intended rate.
- The same input and seed produce identical output in separate Python processes.
- Golden Figure 4/12 source tables.
3. Collapse the plotting framework to the subset actually used
Current files
- [arcadia_pub.py](/home/ubuntu/Development/glm-latent-mapping/scripts/arcadia_pub.py)
- [arcadia_style.py](/home/ubuntu/Development/glm-latent-mapping/scripts/arcadia_style.py)
- [plot_utils.py](/home/ubuntu/Development/glm-latent-mapping/scripts/plot_utils.py)
- Entire [arcadia_plots](/home/ubuntu/Development/glm-latent-mapping/scripts/arcadia_plots) package
- Ignored .claude/skills/arcadia-plots/arcadia_style/ copy
What they do
A vendored plotting framework provides sizes, palettes, typography, subplot layout, legends, and styling. The publication code uses only a small subset. plot_utils.py is an 18-line wrapper around the style initializer.
Why problematic
There are two diverging copies of much of the style implementation: the tracked scripts/arcadia_plots tree and an ignored tooling copy under .claude. Approximately 1,700 tracked lines are maintained for a narrow publication API.
Proposed destination
One small figure_style.py exposing only the constants and helpers used by the paper:
- Figure dimensions
- Typography and palette
- apply
- mono_ticks
- sync_keys
- Required Arcadia glyph handling
Alternatively, depend directly on the upstream style package if reproducibility permits pinning it.
The .claude copy should be treated as generated tooling, never as a second source of truth.
Files deletable after migration
- plot_utils.py
- arcadia_pub.py
- arcadia_style.py
- scripts/arcadia_plots/
Risk: Medium because figure appearance can drift.
Protecting tests
- Exact 500/1000-point output dimensions.
- Font and required-glyph audit.
- Source-table snapshots.
- Selected rendered-image snapshots or perceptual comparisons for publication panels.
4. Consolidate mammalian dataset construction
Current files
- [resolve_orthologs.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/resolve_orthologs.py)
- [download_bulk.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/download_bulk.py)
- [extract_loci_bulk.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/extract_loci_bulk.py)
- [assemble_datasets.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/assemble_datasets.py)
- [build_capped_manifest.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/build_capped_manifest.py)
- [build_cds_masks_mammal.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/build_cds_masks_mammal.py)
What they do
They resolve orthologs, download genomic data, extract transcript loci, create several dataset tracks, cap families, and construct CDS masks.
Why problematic
Several files are deterministic consecutive transformations that do not represent reusable subsystems. assemble_datasets.py creates balanced_core and or_track datasets that the active experiments do not consume; the experiment instead constructs another capped manifest.
Proposed destination
A mammal_dataset.py module with explicit subcommands for:
1. Resolve/download external data.
2. Extract and validate loci.
3. Materialize the single publication manifest, cap, and masks.
Keep downloading as an operational checkpoint, but share configuration and validation.
Files deletable after migration
- extract_loci.py
- build_capped_manifest.py
- build_cds_masks_mammal.py
- Potentially assemble_datasets.py and extract_loci_bulk.py once replaced by the consolidated entry point
Remove generation of inactive dataset tracks unless another documented workflow consumes them.
Risk: Medium.
Protecting tests
A tiny GTF/FASTA fixture covering:
- Both strands.
- Transcript/CDS bounds.
- Start and stop codons.
- Missing or malformed annotations.
- QC flags.
- Family caps.
- Exact CDS masks and final manifest rows.
5. Make gene-family metadata single-source
Current files
- [gene_families.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/gene_families.py)
- families_data.json
What they do
Both encode family order and specifications. The Python module also contains a live HGNC builder that writes the JSON. Runtime resolution reads the JSON, while plots import selected Python definitions.
Why problematic
Generated data is also duplicated as hand-maintained Python constants. It is unclear which representation is authoritative.
Proposed destination
Treat families_data.json as the frozen publication input. Keep a small loader module for family order, Pfam metadata, and colors. Move the live HGNC refresh operation into a clearly separate maintenance command.
Files deletable
The large current gene_families.py implementation, replaced with a compact loader; the filename may remain.
Risk: Low.
Protecting tests
- JSON schema validation.
- Unique genes and family identifiers.
- Stable publication family order.
- Expected family count and representative members.
6. Choose one family-geometry implementation and remove the general OT framework
Current files
- [mammal_score.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/mammal_score.py)
- [mammal_between.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/mammal_between.py)
- [geodesic_utils.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/geodesic_utils.py)
- [ot_between_family.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/ot_between_family.py)
- [ot_between_family_sweep.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/ot_between_family_sweep.py)
- [within_family_uncertainty.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/within_family_uncertainty.py)
- Both control scorers
What they do
They provide at least four related distance routes:
- Within-family angular distance.
- Within-family full-panel kNN geodesic submatrices.
- Between-family centroid geodesic.
- Between-family Wasserstein/W2.
The OT module additionally implements FGW, entropic/exact solvers, random couplings, multi-initialization and sensitivity utilities.
Why problematic
Every active caller invokes OT with no FGW alphas; the publication uses W2. Roughly several hundred lines implement unused research generality. More importantly, multiple metrics are silently mixed between figures and inference.
Proposed destination
After resolving the Figure 1/2 discrepancies, create a small family_geometry.py with one published contract:
- Direct angular within-family scoring.
- W2 between-family scoring.
- Shared family loading and ordering.
- Shared bootstrap/permutation inference.
- Natural and controlled embeddings handled through the same API.
Retain graph/geodesic methods only in an explicitly labeled diagnostic module if they remain scientifically valuable.
Files deletable after migration
- mammal_between.py
- ot_between_family_sweep.py
- Most or all of ot_between_family.py
- geodesic_utils.py if graph diagnostics are retired
- The separate graph and graph-free control scorers
Risk: High because this touches primary scientific claims.
Protecting tests
- Small exact angular and W2 matrices with analytically known values.
- Stable family ordering and missing-data behavior.
- Exact Spearman/Mantel results on a fixed fixture.
- Golden source tables from Figures 1, 2, and 4.
- A characterization test comparing current centroid-geodesic, W2, angular, and geodesic results before selecting the canonical method.
7. Prune publication figure generators to publication paths
Current files
- [layer_sweep_summary.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/layer_sweep_summary.py)
- [within_family_per_family_grid.py](/home/ubuntu/Development/glm-latent-mapping/scripts/baselines/within_family_per_family_grid.py)
- [figures.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/figures.py)
- [hypothesis_figures.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/hypothesis_figures.py)
- [gc_codon_figures.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/gc_codon_figures.py)
- [site_directionality_figures.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/site_directionality_figures.py)
- [dose_and_alpha_figures.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/dose_and_alpha_figures.py)
- [steering_delta_strip.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/steering_delta_strip.py)
What they do
These modules contain numerous diagnostic panels, render modes, and alternative analyses. The experiment driver selects only narrow paths:
- hypothesis_figures: figures 5 and 8.
- gc_codon_figures: figure 14.
- steering_delta_strip: violin-strata.
- figures --pub: only two outputs are published, although other panels are still calculated.
- site_directionality --pub: still calculates several nonpublication figures.
within_family_per_family_grid.py produces only Figure 3 and imports a private labeling helper from layer_sweep_summary.py.
Why problematic
Thousands of lines of dormant diagnostic branches are coupled to the publication route. “Publication mode” often suppresses writes but not computation. Private cross-module imports signal poor ownership.
Proposed destination
- One experiment1_figures.py containing Figures 1–3.
- One experiment2_figures.py containing Figures 4 and 12.
- One paper-focused experiment3_figures.py, or two modules divided by geometry versus evolutionary analysis.
- Move nonpublication diagnostics to notebooks or a clearly optional exploratory area only if still needed.
Files deletable after migration
All listed specialized figure files can be replaced. At minimum, delete within_family_per_family_grid.py and remove nonpublication branches from the larger modules.
Risk: Medium to high for visual and statistical fidelity.
Protecting tests
- Golden input/source tables for every published panel.
- Figure dimension, panel-count, label, and glyph checks.
- Deterministic plotting with fixed seeds.
- Smoke tests ensuring publication mode computes only requested panels.
8. Consolidate steering-site annotation and rescoring
Current files
- [diagnostic_sites.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/diagnostic_sites.py)
- [sites_nt.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/sites_nt.py)
- [stage4_rescore_nt.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage4_rescore_nt.py)
- [site_directionality.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/site_directionality.py)
What they do
They define codon or nucleotide target sites, load the same FASTAs and generated sequences, perform alignments, and calculate overlapping private/loose site metrics.
sites_nt.py contains two routines that perform effectively the same alignment and loop, differing mainly in whether the human base is retained. The rescoring and directionality stages duplicate much of their I/O and multiprocessing setup.
Why problematic
The same generated sequence can be realigned and rescored multiple times, while the scientific definition of a “target site” is distributed across four modules. Dynamic file-path imports are used because the directory is not a proper package.
Proposed destination
A single score_generations.py with:
- One site-definition representation.
- One alignment per sequence.
- Base L/C/A metrics and directionality derived from the same annotated alignment.
- Explicit private/loose policies.
- A normal package import path.
Files deletable after migration
All four current modules can be replaced by the shared scorer and a small figure consumer.
Risk: High.
Protecting tests
Tiny hand-verified human/platypus/generated/outgroup alignments covering:
- Insertions and deletions.
- Reverse or ambiguous bases.
- Private versus loose target sites.
- Human, platypus and alternative outcomes.
- Exact L/C/A and directionality scores.
- Equivalence with current output on a frozen sample.
9. Consolidate direction geometry and evolutionary predictor tables
Current files
- [delta_stats.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/delta_stats.py)
- [stage3_select.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage3_select.py)
- [h1c_clusters.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/h1c_clusters.py)
- [stage3_gates.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage3_gates.py)
- [stage4_analysis.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage4_analysis.py)
- [stage5_merge.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage5_merge.py)
- [stage5_gain.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage5_gain.py)
- Predictor-table construction inside hypothesis_figures.py
What they do
- delta_stats and stage3_select both load pooled representations, compute human–platypus deltas, mean/leave-one-out directions, and cosine statistics.
- h1c_clusters and stage3_gates produce one-off selection/gating reports.
- stage5_merge, stage5_gain, and the figure code independently merge substantially the same predictor tables.
Why problematic
Core numerical concepts are recomputed in multiple stages. The predictor population used for modeling can diverge from the population shown in figures. Identical helpers such as unit, gc3, rank correlation, and bootstrap intervals recur in different files.
Proposed destination
- direction_geometry.py: calculate and persist deltas, LOO directions, GC/cone adjustments, cluster annotations, and gates once.
- evolution_analysis.py: one build_predictor_table and gain/model analysis used by both statistics and plotting.
- Fold the one-off stage reports into these modules as named outputs rather than separate scripts.
Files deletable after migration
The listed stage-specific files can mostly be replaced, particularly delta_stats.py, stage3_select.py, stage5_merge.py, and stage5_gain.py.
Risk: Medium to high.
Protecting tests
- Exact small-vector tests for normalization, mean/LOO direction, cosine, and GC/cone removal.
- Predictor-table fixture checking joins, missingness, gene population and column values.
- Regression tests that analysis and figures consume the exact same predictor table.
- Golden result-table schemas.
10. Centralize low-level FASTA and run configuration utilities
Current duplication
FASTA readers occur in at least:
- stage2_embed.py
- stage3_select.py
- stage4_steer.py
- autapomorphy_orthologs.py
- h1c_clusters.py
- stage5_orthologs.py
- stage5_trees.py
- stage5_dnds.py
- gc_codon_figures.py
- Nested variants in rescoring and directionality modules
Configuration duplication includes:
- Run dates such as 2026-07-16, 2026-07-28, and 2026-08-08.
- Layer 27 and 32-block assumptions.
- Species lists and Ensembl release 116 endpoints.
- MIN_SP=10.
- Control names and order.
- Hard-coded human and platypus identifiers.
Why problematic
These copies are small individually but encode important scientific assumptions. Run IDs embedded inside modules make reusable code depend on a particular historical artifact directory.
Proposed destination
Use a few domain-specific modules rather than a generic utility framework:
- fasta.py for one streaming FASTA reader/writer.
- mammal_panel.json for species/release/panel definitions.
- experiment_config.py for stable layer and threshold defaults.
- Per-run JSON manifests written by drivers and passed explicitly to analysis code.
Files deletable
Mostly duplicated functions and hard-coded blocks rather than entire files. control_tables.py becomes unnecessary.
Risk: Low to medium.
Protecting tests
- Multiline, blank-line, duplicate-ID, and lowercase FASTA fixtures.
- Configuration schema validation.
- CLI tests proving alternate run directories and layers do not require source edits.
11. Unify ortholog acquisition only after transcript-policy characterization
Current files
- [resolve_orthologs.py](/home/ubuntu/Development/glm-latent-mapping/scripts/mammalian_orthologs/resolve_orthologs.py)
- [autapomorphy_orthologs.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/autapomorphy_orthologs.py)
- [stage5_orthologs.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/strat/stage5_orthologs.py)
What they do
They contain overlapping Ensembl REST access, species mapping, one-to-one selection, caching and CDS retrieval.
Why problematic
Each path makes subtly different biological choices. Some use canonical CDS, while another chooses the longest CDS from bulk data. Consolidating without first making that policy explicit could silently alter the gene panel.
Proposed destination
A shared ensembl.py client plus explicit policies such as:
- canonical_transcript
- longest_complete_cds
- Required one-to-one orthology
- Release and species panel
Ideally, materialize the 24-mammal ortholog/CDS dataset once and reuse it in Experiments 1 and 3.
Files deletable after migration
The three existing scripts could become one dataset command with subcommands, but only after validating transcript equivalence.
Risk: High.
Protecting tests
- Recorded Ensembl JSON fixtures.
- One-to-one and ambiguous ortholog selection.
- Explicit canonical-versus-longest transcript cases.
- Stable species normalization.
- Comparison of current gene/CDS selections across all active panels.
12. Reconsider the separate 100-gene Figure 6 panel
Current files/data flow
- [dataset.py](/home/ubuntu/Development/glm-latent-mapping/scripts/steering/platypus/dataset.py)
- Separate embedding and delta_stats run associated with the older paired panel
- The larger 400-gene Experiment 3 route independently calculates similar geometry
What they do
Figure 6 uses a separate approximately 100-gene paired panel, while the steering selection and subsequent results use the larger 400-gene panel.
Why problematic
This duplicates dataset and geometry machinery and makes the explanatory geometry figure describe a different population from the downstream experiment.
Proposed destination
Regenerate Figure 6 from the canonical 400-gene geometry artifact, if scientifically acceptable, and remove the older paired-panel branch.
Files deletable
- scripts/steering/platypus/dataset.py
- The paired-panel branch in the Experiment 3 driver
- Associated specialized intermediate-generation code
Risk: Very high because it changes a publication figure’s source population.
Protecting tests
Tests can protect schema and geometry calculations, but cannot decide scientific equivalence. This requires explicit scientific review and a before/after comparison of distributions and conclusions.
Thin wrappers and experimental/one-off classification
High-confidence thin wrappers:
- [plot_utils.py](/home/ubuntu/Development/glm-latent-mapping/scripts/plot_utils.py): wrapper around styling initialization.
- steer_lib.load_model: wrapper around [evo2_embedding.py](/home/ubuntu/Development/glm-latent-mapping/scripts/evo2/evo2_embedding.py); callers should import the adapter directly.
- control_tables.py: one-consumer presentation indirection.
One-off but publication-reachable stage scripts:
- Mammalian extraction, assembly, cap, masks, score, between-family score, uncertainty, and figure scripts.
- Almost every stage0_* through stage5_* Experiment 3 file.
- h1c_clusters.py and stage3_gates.py are preregistration/reporting gates rather than reusable components.
- stage4_analysis.py is an intermediate reporting analysis; published plots primarily consume later nucleotide rescoring and evolutionary outputs.
These are not all dead. The issue is that many adjacent stages share data loading and numerical logic yet are maintained as isolated programs.
Generated or vendored code that should not be hand-maintained:
- families_data.json versus duplicated Python family constants.
- scripts/arcadia_plots/ versus the ignored .claude style copy.
- Historical run IDs and artifact paths embedded in source instead of run manifests.
Duplicate functions and concepts
Notable exact or near-exact duplication includes:
- unit in stage3_gates.py and h1c_clusters.py.
- The same log helper in approximately six Experiment 3 stages.
- Multiple identical FASTA readers.
- _load dynamic-import helpers in plotting/scoring modules.
- Rank-correlation helpers in both within-family inference implementations.
- Bootstrap interval implementations in analysis and figure code.
- gc3 in selection and clustering.
- Mammal embedding/CDS loaders in mammal_score.py and mammal_between.py.
- Predictor-table joins in stage5_merge.py, stage5_gain.py, and hypothesis_figures.py.
- Site alignment and annotation in sites_nt.py, rescoring, and directionality.
- Ortholog REST/caching logic across three acquisition pipelines.
Tests and CI
There are no active repository tests outside deprecated/.
The closest items are:
- A manual OT self-test embedded in the implementation, primarily covering unused FGW functionality.
- Runtime assertions that execute only after expensive data preparation.
- Structural checks embedded in generation/scoring scripts.
CI only runs Ruff. The Makefile’s lint command uses ruff check --exit-zero, so it cannot fail the build. pytest is listed as a dependency but is not run.
A minimum protective suite should include:
1. Dataset extraction and mask fixture.
2. Ortholog-policy fixtures.
3. Exact angular/W2 geometry fixture.
4. Control preservation and stable-seed tests.
5. Steering-site alignment/scoring fixture.
6. Direction-vector and predictor-table fixtures.
7. Golden publication source tables.
8. Publication figure dimension/glyph smoke tests.
9. CLI smoke tests for all three experiment drivers.
10. CI that runs pytest and uses failing Ruff exit codes.
Dependency reduction
The source appears not to directly use several declared dependencies, including transformers, datasets, huggingface_hub, umap, seaborn, networkx, and edlib. Some may be transitive requirements of Evo2, so they should be removed only after a clean-environment installation test.
If graph/geodesic analysis is retired, scikit-learn may also become removable. If the vendored plotting framework is removed, direct fontTools usage may disappear.