# Reproducing the publication

The publication is organized around two shell runners in [experiments/](experiments/). Run them
from the repository root. With no argument they print every stage and execute nothing:

```bash
bash experiments/exp1_gene_family_geometry.sh
bash experiments/exp2_platypus_steering.sh
```

A third runner, [`analyses/controls/run_composition_controls.sh`](analyses/controls/), drives the
composition-controls analysis. It behaves identically but is not part of the publication pipeline;
[analyses/controls/README.md](analyses/controls/README.md) covers it.

Use `--figures` to render from existing artifacts or `--run` for an end-to-end run. `--figures`
needs nothing but the tracked `figure_data/`. Full runs are resumable, and additionally need
Evo2-7B, many GPU-hours, the reference downloads and the external tools listed under
[Required inputs](#required-inputs) — each runner checks the tools before it starts work.

## Run order

`--figures` reads only tracked tables, so the runners are independent in that mode and can be run in
any order.

**`--run` is not.** Experiment 2 consumes experiment 1's outputs, so experiment 1 must complete
first:

| Consumer | Reads | Written by |
|---|---|---|
| exp 2, stages E1–E2 | `data/mammalian_orthologs/tree/species_tree.nwk` | exp 1, A6 `build_species_tree.py` |
| controls, stage C1 | `data/cache/mammal_cds_positions.json` | exp 1, A7 `build_cds_masks_mammal.py` |
| controls, stage D2 | `blocks15/betweenfam_ot_metadata.json` | exp 1, B3 `ot_between_family_sweep.py` |

Experiment 2 and the composition controls do not depend on each other and can run in either order
once experiment 1 is done. Each of these reads is guarded: a missing input stops the stage naming
the file *and the stage that writes it*, rather than raising a bare `FileNotFoundError`.

## Publication scope

| Experiment | Analysis | Figures |
|---|---|---|
| 1 | Between- and within-family geometry across 48 HGNC families and 24 mammals | 1–3 |
| 2 | Conservation-stratified human-to-platypus steering at block 27 | 5–11 |

Both experiments embed all 32 residual-stream blocks. Transcript spans are split into windows;
hidden states are selected at CDS positions and mean-pooled over the second half of those positions.

Figures 4, 12 and 13 belong to the composition-controls analysis in
[analyses/controls/](analyses/controls/) and are rendered by its own runner; the publication figure
numbers of experiments 1 and 2 are unaffected by whether it is run.

## Required inputs

`--figures` reads only [figure_data/](figure_data/), which is tracked, so it runs on a fresh clone:

| Table prefix | Figures |
|---|---|
| `figure_data/exp1_*` | 1–3 |
| `figure_data/exp2_*` | 5–11 |

`figure_data/README.md` lists every table with its grain and size. `--run` additionally needs
`data/` and writes the dated run directories under `results/`; the last stage of each runner calls
`scripts/build_figure_data.py`, which collapses those into `figure_data/`. The composition-controls
tables are built by the same script into `analyses/controls/figure_data/`.

Each publication runner copies its completed panels into `pub/figures/`; the
composition-controls runner writes its own into `analyses/controls/figures/`.

### Reference downloads and where they live

A full `--run` also needs large reference files that no part of the repository tracks and that are
too big to ship. Every path is resolved by [`scripts/paths.py`](scripts/paths.py) and defaults to a
location inside the checkout, under the git-ignored `data/external/`, so a fresh clone resolves
somewhere writable without configuration. Set an environment variable to point anywhere else — for
example at a copy that already exists on a fast local disk:

| Variable | Default | Holds | Needed by |
|---|---|---|---|
| `GLM_SCRATCH` | `data/external` | root for all of the below | — |
| `GLM_MAMMAL_GENOMES` | `$GLM_SCRATCH/mammal_genomes` | Ensembl genome FASTA + GTF, 24 species | exp 1, A2–A3 |
| `GLM_STRAT_SEQS` | `$GLM_SCRATCH/strat_seqs` | human + platypus CDS, peptide and GTF | exp 2, A1–A2 |
| `GLM_STRAT_MAMMAL_CDS` | `$GLM_STRAT_SEQS/mammals` | per-species CDS FASTA, 24 mammals | exp 2, E1 |
| `GLM_TMPDIR` | system temp (`TMPDIR`) | MAFFT/FastTree scratch during alignment | exp 1, B2 |

```bash
export GLM_SCRATCH=/mnt/fast/glm        # move all of them at once
export GLM_MAMMAL_GENOMES=/mnt/genomes  # or just one
```

`mammal_genomes` is fetched by `scripts/mammalian_orthologs/download_bulk.py` (experiment 1, stage
A2). The `strat_seqs` files are **not fetched by any script** — download them from
<https://ftp.ensembl.org/pub/release-116/>, matching the release the Compara orthology used:

| File | Path |
|---|---|
| `Homo_sapiens.GRCh38.cds.all.fa.gz` | `fasta/homo_sapiens/cds/` |
| `Homo_sapiens.GRCh38.pep.all.fa.gz` | `fasta/homo_sapiens/pep/` |
| `Homo_sapiens.GRCh38.116.gtf.gz` | `gtf/homo_sapiens/` |
| `Ornithorhynchus_anatinus.mOrnAna1.p.v1.cds.all.fa.gz` | `fasta/ornithorhynchus_anatinus/cds/` |
| `Ornithorhynchus_anatinus.mOrnAna1.p.v1.116.gtf.gz` | `gtf/ornithorhynchus_anatinus/` |

and one `<species>.cds.fa.gz` per mammal in the panel under `$GLM_STRAT_MAMMAL_CDS/`.

The mammalian species tree is a third download: the pruned VertLife posterior set at
`data/mammalian_orthologs/tree/vertlife_pruned/output.nex`, exported from
<https://vertlife.org/data/mammals/> for the 24 species in `build_species_tree.py`'s `V2E` map
(experiment 1, stage A6).

A stage that needs one of these and cannot find it stops immediately, naming the file, the path it
looked in, and the variable that moves it — rather than failing partway through.

### External tools

Beyond the Python dependencies, `--run` shells out to these. None is pip-installable, and
`--figures` needs none of them. Each runner checks its own before doing any work; to check by hand:

```bash
uv run python scripts/check_tools.py            # all of them, with versions
uv run python scripts/check_tools.py --exp exp2 # just one experiment's
```

| Tool | Version used | Found via | Needed by | Install |
|---|---|---|---|---|
| `mafft` | v7.505 | `PATH` | exp 1 B2, exp 2 E2 | `apt install mafft` or `conda install -c bioconda mafft` |
| `FastTree` | 2.1.11 | `PATH` | exp 1 B2 | `apt install fasttree` or `conda install -c bioconda fasttree` |
| `mmseqs` | 15-6f452 | `PATH` | exp 2 A1 | `apt install mmseqs2` or `conda install -c bioconda mmseqs2` |
| `iqtree2` | 2.3.6 | `data/tools/` | exp 2 E2 | a release from <https://github.com/iqtree/iqtree2/releases> |
| `trimal` | 1.5.rev0 | `data/tools/` | exp 2 E2, E3 | build from <https://github.com/inab/trimal> |
| `codeml` | PAML 4.10.10 | `data/tools/` | exp 2 E3 | build PAML, copy `src/codeml` |
| `yn00` | PAML 4.10.10 | `data/tools/` | exp 2 E3 | build PAML, copy `src/yn00` |

`data/tools/` is git-ignored, so the four binaries there do not survive a clone and must be
rebuilt or re-downloaded. PAML is at <http://abacus.gene.ucl.ac.uk/software/paml.html>.
`scripts/check_tools.py` is the single registry behind both the table and the runtime errors, so a
missing tool reports the same install command wherever it is discovered.

## Experiment 1: gene-family geometry

```bash
bash experiments/exp1_gene_family_geometry.sh --run
bash experiments/exp1_gene_family_geometry.sh --figures
```

The runner performs the following stages:

1. Resolve one-to-one orthologs with Ensembl Compara release 116.
2. Download Ensembl annotations and genomes, extract transcript loci and CDS sequences, and assemble
   the complete and 400-cap manifests.
3. Build CDS-position masks and the independent mammalian species-tree distance matrix.
4. Embed the CDS positions of each transcript locus at all 32 blocks.
5. Score within-family geometry against the species tree, MAFFT/FastTree patristic distances,
   6-mer distances, and GC differences.
6. Score between-family exact Wasserstein distances against Pfam-HMM JSD,
   6-mer, and GC baselines.
7. Run the all-layer angular bootstrap and Wilcoxon uncertainty analyses.

Important settings:

- `mammal_score.py --distance angular` generates the direct-angular tables used by Figures 2–3.
- `ot_between_family_sweep.py --n-perms 9999` retains optional W2 Mantel significance tests; the
  publication driver passes `--n-perms 0` and skips them. The composition-controls analysis reuses
  the natural W2 matrices this stage writes.
- `within_family_uncertainty.py` runs every available layer with 10,000 bootstrap replicates by
  default and Wilcoxon signed-rank inference. Geodesic-only within-family Mantel paths are archived.

Key active files:

| Responsibility | Code |
|---|---|
| Ortholog resolution and extraction | `scripts/mammalian_orthologs/{resolve_orthologs,download_bulk,extract_loci_bulk,assemble_datasets}.py` |
| Manifests, species tree, and CDS masks | `scripts/mammalian_orthologs/{build_capped_manifest,build_species_tree,build_cds_masks_mammal}.py` |
| Embedding | `scripts/mammalian_orthologs/embed_cds_masked_mammal.py` |
| Within-family scoring and inference | `scripts/mammalian_orthologs/{mammal_score,within_family_uncertainty}.py` |
| Between-family scoring | `scripts/mammalian_orthologs/mammal_between.py`, `scripts/baselines/{ot_between_family,ot_between_family_sweep}.py` |
| Figures 1–3 | `scripts/layer_sweep_summary.py`, `scripts/within_family_per_family_grid.py` |

## Experiment 2: platypus steering

```bash
bash experiments/exp2_platypus_steering.sh --run
bash experiments/exp2_platypus_steering.sh --figures
```

Experiment 2 uses one panel: a conservation-stratified set of 400 human–platypus ortholog pairs,
used for generation, site-directionality, evolutionary-rate analyses, and every published figure it
produces (5–11).

A second 103-gene paired panel supplied Figures 6a–6b until 2026-09-01, when they moved to the
n=400 panel so the direction geometry and the steering results it explains come from the same gene
set. The stages that built it were removed from the pipeline on 2026-09-03; they cost ~2.5 h per
run (2 h of it GPU) producing output nothing read.

The production steering stages are:

1. Build a block-disjoint, five-stratum ortholog panel and apply shifted-window QC.
2. Embed human and platypus CDS representations across all 32 blocks.
3. Estimate leave-one-gene-out steering directions and run the pre-specified gates.
4. Generate at `blocks.27` for the primary, dose, and confound arms.
5. Score amino-acid and nucleotide outcomes, including private-site directionality.
6. Estimate fixed-topology branch lengths, dN, dS, and omega, then merge them with steering
   outcomes.

The runner generates at block 27. The stage-4 directory may also hold exploratory conditions from
other layers (named with an `_L<n>` suffix) and arms that were generated but not reported;
`build_figure_data.py` admits only rows whose recorded layer is `blocks.27`, plus the hook-free
`unsteered` baseline, and names any condition it excludes.

### Known gap: the tracked steering table

`figure_data/exp2_steering_outcomes.csv` predates the driver's C3 fix. Its dose ladder runs to
alpha 4 while the norm-matched random null stops at alpha 2, because C3 used to request
`--arms add` alone. C3 now requests `--arms add random`, and `build_figure_data.STEER_CONDITIONS`
requires `random_a3.0` and `random_a4.0`.

So a full `--run` produces a table with **two more conditions** than the one figures 7 and 8 were
rendered from. Neither figure plots the random ladder — figure 8 derives its doses from the `add_a*`
arms — so the panels are not expected to change, but the artifact and a fresh run differ until the
table is regenerated. `build_figure_data.py --experiments exp2` fails against a stage-4 directory
that lacks the two conditions, naming them. No missing measurement has been synthesized.

### Reproducibility of the generations

Sampling is seeded per cell: `torch.manual_seed(seed + crc32("<gene>:<condition>"))` runs before
each `generate` call, so a cell's draw does not depend on which cells that invocation happened to
run — which matters because `--resume` skips a different set each time. Random steering directions
are seeded per gene and layer.

Two limits, stated rather than implied:

- **CUDA kernel non-determinism is not controlled.** `torch.use_deterministic_algorithms` is not
  enabled, so bitwise-identical output across machines or driver versions is not claimed; what is
  fixed is the sampling stream for a given cell.
- **The archived generations predate this seeding and cannot be reproduced.** Stage 4 ran with an
  unseeded sampler, so re-running it yields statistically comparable but different sequences. The
  archived per-sample scores and generated sequences are the record of that run; seeding makes
  future runs reproducible, not past ones.

| Figure | Generator |
|---|---|
| 5, 11 | `scripts/steering/platypus/strat/hypothesis_figures.py` |
| 6a, 6b | `scripts/steering/platypus/figures.py` |
| 7 | `scripts/steering/platypus/strat/steering_delta_strip.py` |
| 8 | `scripts/steering/platypus/strat/dose_and_alpha_figures.py` |
| 9a, 9b | `scripts/steering/platypus/strat/site_directionality_figures.py` |
| 10 | `scripts/steering/platypus/strat/gc_codon_figures.py` |

## Composition controls

An additional analysis over experiment 1's panel, with its own runner, tracked tables and figures:

```bash
bash analyses/controls/run_composition_controls.sh --run
bash analyses/controls/run_composition_controls.sh --figures
```

It rewrites the coding positions of each natural transcript span under matched constraints — GC
matching, dinucleotide and k-mer shuffles, family-conditioned synonymous recoding, a nested
missense subset, and the paired third-codon-position arms — re-embeds the panel, and scores how
much of the natural geometry each rung preserves. It needs experiment 1's CDS-position cache and W2
family order, and no external tools.
[analyses/controls/README.md](analyses/controls/README.md) covers what it measures, what it can be
reused for, and which stages are optional.

## Validate a checkout

```bash
uv run ruff format --check scripts
uv run ruff check scripts
uv run python -m compileall -q scripts
bash -n experiments/*.sh analyses/controls/*.sh

bash experiments/exp1_gene_family_geometry.sh
bash experiments/exp2_platypus_steering.sh
bash analyses/controls/run_composition_controls.sh
```

To validate the publication artifacts, run both experiment scripts with `--figures` and confirm that
each reports zero failed panels and that every panel measures exactly 1,000 or 500 points wide. The
composition-controls runner reports the same way for its own three panels.
