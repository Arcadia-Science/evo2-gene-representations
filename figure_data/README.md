# Figure data

The tidy tables every publication figure reads. Each generator reads one or two of these and
nothing else, so a fresh clone can render the complete figure set with no downloads and no GPU:

```bash
bash experiments/exp1_gene_family_geometry.sh --figures   # figures 1-3
bash experiments/exp2_composition_controls.sh --figures   # figures 4, 12
bash experiments/exp3_platypus_steering.sh --figures      # figures 5-11
```

Panels are written to `pub/figures/`.

## Where these come from

`scripts/build_figure_data.py` writes this directory from a completed `--run`. It is the only step
that reads the dated run directories under `results/`, so those stay local build artifacts and this
directory is the tracked form of the numbers. Rebuild after a full run:

```bash
uv run python scripts/build_figure_data.py
```

Nothing in `--figures` writes here, so rendering a figure never modifies one of its own inputs.

## The tables

| File | Rows x cols | Size | Contents |
|---|---|---|---|
| `exp1_between_family_by_layer.csv` | 96 x 5 | 4 KB | Figure 1 — W2 family geometry vs baselines |
| `exp1_within_family_by_layer.csv` | 6,112 x 4 | 370 KB | Figures 2, 3 — within-group rho vs baselines |
| `exp2_control_preservation.csv` | 14,112 x 5 | 905 KB | Figures 4, 12 — control vs natural geometry |
| `exp3_direction_layer_stats.csv` | 32 x 31 | 18 KB | Figure 6a — paired-panel direction statistics |
| `exp3_direction_per_gene_by_layer.csv` | 16,096 x 7 | 1.5 MB | Figures 6b, 11 — per-gene direction geometry |
| `exp3_panel.csv` | 400 x 28 | 149 KB | Figure 5 — the 400 human/platypus pairs |
| `exp3_panel_attrition.csv` | 558 x 19 | 88 KB | Figure 5 — candidates dropped by QC |
| `exp3_panel_coverage.csv` | 400 x 5 | 19 KB | Figure 5 — aligned CDS coverage per pair |
| `exp3_steering_outcomes.csv` | 9,154 x 26 | 2.0 MB | Figures 7, 8 — steering outcomes per gene and condition |
| `exp3_site_directionality_summary.csv` | 66 x 122 | 130 KB | Figures 9a, 9b — leave-human and platypus-choice rates |
| `exp3_site_directionality_gc_class.csv` | 198 x 13 | 40 KB | Figures 9a, 9b — the same, split by GC class |
| `exp3_site_directionality_per_gene.csv` | 27,462 x 22 | 7.7 MB | Figures 9a, 9b — per-gene spread |
| `exp3_generation_composition.csv` | 4,821 x 6 | 447 KB | Figure 10 — GC of the generations |
| `exp3_generation_reference_windows.csv` | 398 x 5 | 28 KB | Figure 10 — human and platypus reference GC |
| `exp3_codon_substitutions.csv` | 4,781 x 14 | 1.0 MB | Figure 10 — GC by codon position |
| `exp3_rates_tree_stats.csv` | 400 x 15 | 66 KB | Figure 11 — fixed-topology branch lengths |
| `exp3_rates_dnds.csv` | 399 x 51 | 162 KB | Figure 11 — dN, dS and omega |
| `exp3_rates_vs_direction.csv` | 57 x 10 | 6 KB | Figure 11 — rate vs direction geometry |
| `exp3_rates_vs_direction_shape.csv` | 12 x 15 | 3 KB | Figure 11 — the same, shape tests |
| `exp3_rates_vs_gain.csv` | 38 x 16 | 7 KB | Figure 11 — rate vs steering gain |
| `exp3_direction_nulls.npz` | binary | 1.4 MB | Figures 6a, 6b — permutation nulls |

`MANIFEST.csv` repeats this table in machine-readable form.

## Grain

Most tables are copied through unchanged from the analysis that wrote them. Three are reshaped:

* `exp1_between_family_by_layer` and `exp1_within_family_by_layer` are long-format roll-ups of the
  per-block score tables — one row per (layer, baseline) and per (layer, metric, family) rather than
  one file per block.
* `exp2_control_preservation` merges the within-family and between-family control scores into one
  long table. `family` is set on within-family rows and empty on between-family rows, whose geometry
  is a single family x family matrix per block. Figure 4 plots the mean over families; figure 12
  uses the per-family values.
* `exp3_steering_outcomes` is one row per (gene, condition). The analysis writes one row per
  generated sample; figures 7 and 8 average over samples within a gene before plotting anything, so
  the aggregation is lossless for them.

## Not here

The raw per-sample scores, the generated sequences, the Evo2 embedding caches and the sequence
datasets are not tracked. They are archived separately; see the data-availability statement in the
manuscript.
