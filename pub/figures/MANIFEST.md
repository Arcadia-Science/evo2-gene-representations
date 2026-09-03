# Publication figures

These are the publication-ready PNG and PDF panels for the manuscript in `pub/`. PDF text remains
editable and the publication fonts are embedded.

Rebuild the complete set from existing result artifacts with:

```bash
bash experiments/exp1_gene_family_geometry.sh --figures
bash experiments/exp2_composition_controls.sh --figures
bash experiments/exp3_platypus_steering.sh --figures
```

Each runner carries the exact commands for its figures and checks that every completed panel is
1,000 or 500 points wide. These files are publication-facing copies; their source renders remain
beside the corresponding analysis artifacts in the ignored result directories.

## What "publication geometry" changes

`scripts/arcadia_pub.py` is a second mode over `scripts/arcadia_style.py`, opted into per
generator with `--pub`. The everyday figures are dense diagnostic grids and keep the compact
scale; these do not.

| | Diagnostic (default) | Publication (`--pub`) |
|---|---|---|
| Panel width | whatever `bbox_inches="tight"` crops to — 969 / 1092 / 429 / 508 pt across the old set | exactly **1,000 or 500 pt**, tight-bbox off |
| Body / axis titles | 8 pt | **15 pt** |
| Numerals | 6 pt, in the body font | **14.5 pt, Atkinson Hyperlegible Mono** |
| Key title | 7 pt | 17 pt SemiBold over a Chateau rule |
| Chart title | `suptitle` on the artwork | none — the caption carries it (guide, rule 1) |
| Wide figures | run wider | **stack**, or re-lay out (see below) |

**Greek and arrows are spelled in Latin** — "Spearman rho", "alpha", "higher is better".
Atkinson Hyperlegible has no lowercase Greek and no arrow glyphs at all. The old figures only
looked correct because matplotlib silently substituted DejaVu Sans for those characters, so ρ was
set in a different typeface from the words around it. `arcadia_pub.audit_glyphs` now reads the
font's own cmap and **fails the render** rather than letting a tofu box ship. Δ, μ, π, ±, −, ×, ·
and ≥ are present and are used as-is.

## The figures

| Draft | File | Size (pt) | Source (before copying) |
|---|---|---|---|
| 1 | `fig01_between_family_rho_by_layer` | 1000 × 560 | `…/mammalian-orthologs-cdsmask-48fam/pub/between_axis_vs_layer_wasserstein_angular` (uncapped W2) |
| 2 | `fig02_within_family_rho_by_layer` | 1000 × 992 | same folder, `within_family_vs_layer_wasserstein_angular` (angular) |
| 3 | `fig03_within_family_three_families` | 1000 × 423 | same folder, `within_family_vs_layer_wasserstein_angular_zoom3` (angular) |
| 4 | `fig04_controls_vs_natural_by_layer` | 1000 × 723 | `results/layer_sweep_summaries/pub/controls_layer_summary_…-wasserstein` |
| 5 | `fig05_stratum_composition` | 1000 × 1000 | `…platypus-strat-400/figures/pub/8b_strata_composition_frame` |
| 6a | `fig06a_leave_one_out_cosine_by_block` | 500 × 360 | `…evo2-platypus-paired/stage2_cds_mean/figures/pub/1_loo_median_by_layer` |
| 6b | `fig06b_delta_magnitude_spread_by_block` | 500 × 360 | same folder, `8b_magnitude_spread` |
| 7 | `fig07_steering_delta_by_stratum` | 1000 × 640 | `…strat-400/figures/pub/10_steering_delta_violin_strata` |
| 8 | `fig08_dose_response_by_stratum` | 1000 × 728 | `…strat-400/figures/pub/6d_dose_by_stratum` |
| 9a | `fig09a_leave_human_vs_platypus_choice_private` | 500 × 500 | `…strat-400/figures_27/pub/18_leave_human_vs_platypus_choice` |
| 9b | `fig09b_leave_human_vs_platypus_choice_loose` | 500 × 500 | same folder, `…_platy_not_human` |
| 10 | `fig10_gc_by_codon_position` | 1000 × 480 | `…strat-400/figures_27/pub/14_gc_codon_position` |
| 11 | `fig11_rate_predictor_outcome_matrix` | 1000 × 800 | `…strat-400/figures/pub/5_rate_outcome_matrix` |
| 12 | `fig12_paired_p3_protein_vs_nucleotide` | 1000 × 724 | existing W2 control tables plus angular paired-p3 preservation recomputed from the complete embedding caches |
| 13 | `fig13_control_identity_vs_rho` | 1000 × 393 | `…transcript_cdsmask/pub/control_identity_vs_rho` — sequence identity tables joined to figure 4's own rho |

6a/6b and 9a/9b are 500 pt so each pair sits side by side, as in the draft. Everything else is a
full 1,000 pt panel.

Figures 1 and 12 also retain their former geodesic-derived versions with `(OLD)` in the filename
for direct comparison. The Figure 12 paired-p3 angular summary was staged outside `results/` for the
Phase 1 correction, so the existing result directory was not modified.

## Layout changes made to fit 1,000 pt at 15 pt type

These are the places where the figure had to be re-laid out rather than just re-sized. Each is a
judgment call, recorded here so it can be reversed.

| Figure | Change | Why |
|---|---|---|
| 1 | **Overview panel only**; the three per-baseline panels dropped | Four panels across 1,000 pt leave 235 pt each. The per-baseline panels re-plot the same three series one at a time; every comparison the figure makes is already in the panel that overlays them. |
| 2 | 48-family key moved **below** the grid, 4 columns | A right-hand key at 15 pt would be ~960 pt tall against a 660 pt grid. Column count is measured against the panel width, not guessed. |
| 4 | Two panels **stacked**, sharing the layer axis | Side by side they get ~470 pt each; stacked they get the full 940 pt, and the duplicate x-axis goes away. |
| 7 | Stratum table replaced by an ordinary **colour key**; per-band n dropped | A 3-column table of 6.6 pt cells does not survive being set at 15 pt. The strata are equal-sized quintiles, so the counts carry nothing. |
| 8 | Five strata **wrapped to 3 columns** | Five across 1,000 pt is 180 pt each — narrower than one panel title. |
| 9a/9b | Dose points labelled with the bare α; the three random arms became **one key entry** | At 15 pt "alpha=2" and "alpha=3" overlapped, and the random labels ran off the panel. |
| 10 | CDS reference labels made **single-line**, one above its rule and one below | The two references are 4 pp apart; a two-line label at 15 pt is taller than that gap. |
| 11 | Row-group titles shortened; the two one-row groups lost theirs | A rotated title is bounded by the height of the group it labels. |
| 13 | Reduced to the **two ρ-vs-identity scatters**; the identity/self-pair-null bars, the excess bars, the per-block monotonicity panels and the in-panel statistics box all dropped | The figure asks one question and the reader answers it off the x-axis. The identity decomposition that proves the answer (identity equals each rung's own self-pair null, so excess is 0 ± 0.001) is a number, not a shape, and reads better in the caption and `control_sequence_identity.md`. All four dropped panels survive in the diagnostic render. |

## Labels rewritten for a reader

The generators label series with the CSV column names, which is right for a diagnostic and wrong
on a page. Pub mode substitutes reader-facing names, defined next to the figure that uses them:

* `layer_sweep_summary.PUB_BASELINE_LABELS` — `pfam_jsd` → "Pfam-domain JSD"
* `layer_sweep_summary.PUB_FAMILY_LABELS` — `cytochrome_p450` → "Cytochrome P450", `ras_gtpases` → "Ras GTPases"
* `plot_control_wasserstein.PUB_RUNG_LABELS` — `kmer4_shuffle` → "4-mer shuffle"
* `control_sequence_identity.LABEL` — the same map, plus `missense_subset`, with the "(protein kept)" / "(protein damaged)" parentheticals dropped: figure 4 draws that distinction, figure 13 does not. Overridden locally so figure 4's key is unaffected
* `hypothesis_figures.PUB_PREDICTOR_LABELS` / `PUB_OUTCOME_LABELS` — `dN_hp_yn` → "dN, human–platypus"
* `site_directionality_figures` — the axes say "Δ leave-human rate" / "Δ platypus-choice rate" rather than the CSV's `d_L` / `d_C`

## Figures drawn in two versions

Three figures keep a fuller diagnostic version in the run dir. In each case the extra panels
answer a different question from the one the publication figure makes.

| Figure | Publication version | Diagnostic version, kept in the run dir |
|---|---|---|
| 5 | `8b_strata_composition_frame` — the eight sampling-frame panels | `8_strata_composition` adds a row on the READOUT (scorable sites, strictness, voting depth, unsteered floor) |
| 6b | `8b_magnitude_spread` — the magnitude panel alone | `8_delta_norm_by_gene_layer` pairs it with the per-gene ‖Δ‖ trajectories |
| 11 | `5_rate_outcome_matrix` | same figure; the draft's copy was a 1103 × 1254 slide screenshot |
