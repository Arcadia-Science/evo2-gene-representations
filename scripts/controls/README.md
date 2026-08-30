# scripts/controls/ — composition controls

All composition-control code lives here: sequence construction, scoring, tables and
figures. Scoring for a specific panel lives with that panel
(`mammalian_orthologs/controls_score_graphfree.py`, `evo2/cdspool_controls.py`).


One-off, finished analyses that are **not** part of the standard from-scratch pipelines
(`scripts/evo2/run_*_pipeline.sh`, `scripts/gpnstar/run_paralog_human_gene_pipeline_gpnstar.sh`). They are run by
hand against an existing natural run and produce paper figures/tables. Kept separate so the
model `scripts/` dirs stay limited to the live pipeline.

All scripts run from the repo root (`uv run python scripts/controls/<script>.py ...`) and import
shared helpers from `scripts/` via `sys.path`.

## Compositional controls (Section 3)

Do the within-family signals survive composition-preserving scrambles, or are they an
artifact of nucleotide composition?

- `make_control_sequences.py` — build composition-control sequence sets (dinuc/codon/
  synonymous/GC/k-mer shuffles) mirroring `data/evo2_gene_families/`. Also `missense_subset`,
  the **nonsynonymous** counterpart of `synonymous_recode`, editing only bases the recode itself
  edited — a strict subset, never a base it left alone. It therefore moves *fewer* nucleotides than
  the recode (identity 0.877 vs 0.769) while damaging the protein the recode kept (aa identity
  0.717), so a ρ below the recode's cannot be blamed on nucleotide loss.
- `embed_and_score_controls.py` — embed each Evo2 control through the same tap as the
  natural run and score (a) within-family geodesic-vs-taxonomy ρ (RECOVERY) and (b)
  control-geodesic-vs-natural-geodesic ρ (RECONSTRUCTION / PRESERVATION), within and
  between. `--panel ortholog` re-embeds the cross-kingdom KEGG controls; `--panel human`
  reads the matched-human control run dirs (preservation only). Writes
  `<natural-run>/controls/control_within_scores.csv` + `control_between_scores.csv`.
- `embed_and_score_msa_controls.py` — the GPN-Star analog: MSA column / conservation
  ablation controls. `--axis between` → `<run>/msa_controls/msa_control_between_scores.csv`
  (centroid preservation); `--axis within` → `<run>/controls/control_within_scores.csv`
  (per-family recovery vs patristic + `rho_geodesic_vs_natural` preservation).
- `control_comparison_figure.py` — Figure 3: natural vs. each control, per family.
- `control_sequence_identity.py` — the ladder's own confound check, sequence-level and GPU-free:
  how much of the SOURCE sequence does each rung actually retain, and is its ρ just tracking that?
  Reports per-rung positional / edit-distance / amino-acid / per-codon-position identity to source,
  each against the **self-pair null** (the same identity between two independent draws of the same
  control = the identity its constraint forces with no source information), then plots ρ against
  identity at every layer and tests monotonicity with and without the matched pair. Writes at the
  run's top level (identity is layer-independent, so it is **not** written per-`blocks<L>`):
  `control_sequence_identity.{csv,md}`, `control_identity_table.{png,pdf}` (the headline figure —
  the per-rung numbers as a table beside the ρ-vs-identity scatter), `control_rho_vs_identity.{png,
  pdf}` (the six-panel version), `control_rho_vs_identity_stats.csv`,
  `control_identity_within_rung.{png,pdf}`. Panels: `xkingdom` (control FASTAs on disk → identity is
  exact), `human_cds`, `mammal_cdsmask`.

The shared preservation metric lives in `scripts/geodesic_utils.py`
(`within_preservation_rho`, `between_preservation_rho`) and is wired into all three
pipeline runners (`scripts/{evo2,gpnstar}/run_*_pipeline*.sh`), so every run writes its
control-reconstruction CSVs into its own `results/<run>/controls/` folder.

See **`control_metrics_guide.md`** for how to read these CSVs — what each column means
(reconstruction/preservation vs recovery), the within/between axes, and which file/column
holds which number per model.
