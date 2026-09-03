# Composition controls

An additional analysis over experiment 1's mammal panel. It asks a different question from the two
publication experiments: **which sequence constraints are sufficient to reproduce the gene-family
geometry Evo2 assigns to natural sequences?**

Each control rewrites the coding positions of a natural transcript span in place — the span, the
CDS mask, and every non-coding position are left alone — and the panel is re-embedded and re-scored
against the natural geometry. A rung that preserves the geometry shows that whatever it kept is
enough to produce it; a rung that destroys the geometry shows that whatever it discarded was
carrying it.

The ladder, weakest constraint first:

| Rung | Keeps | Destroys |
|---|---|---|
| `kmer6_shuffle`, `kmer4_shuffle`, `dinuc_shuffle` | k-mer composition to that order | codon structure, protein |
| `gc_match` | GC content only | everything above it |
| `synonymous_recode` | the protein, exactly | coding nucleotide composition |
| `missense_subset` | the number of edited bases, matched to the recode | the protein |
| `paired_p3_syn` / `paired_p3_missense` | third-codon-position edits at the same rate | one arm keeps the protein, the other does not |

The last three rungs are the informative ones: `synonymous_recode` and `missense_subset` edit the
same bases, and the `paired_p3` arms edit codon position 3 at the same rate, so within each pair
the only thing that differs is whether the protein survives.

## What you can use it for

- **A sufficiency test for any geometry result on this panel.** Re-score a new metric or layer
  against the control ladder and see how much of it survives when composition is matched.
- **A composition null that is matched per locus**, rather than a global shuffle: span length, CDS
  mask and non-coding context are identical to the natural sequence, so a difference cannot be
  attributed to any of them.
- **Separating protein-level from nucleotide-level signal**, via the two matched pairs, without a
  residual-mismatch caveat — both arms of a pair move the same number of bases.
- **Sequence operators for other work.** `scripts/controls/make_control_sequences.py` is importable
  on its own: GC matching, k-mer shuffles, family-conditioned synonymous recoding, the nested
  missense subset, and the paired third-position arms, all deterministic per locus.
- **A check that a preservation correlation is not just retained nucleotides**, using the sequence
  identity of each control to the gene it was built from (panel 13).

## Running it

```bash
bash analyses/controls/run_composition_controls.sh            # print the plan, run nothing
bash analyses/controls/run_composition_controls.sh --figures  # re-render the panels (~3 min, CPU)
bash analyses/controls/run_composition_controls.sh --run      # full pipeline        (~160 h, GPU)
```

`--figures` reads only the tracked tables in [`figure_data/`](figure_data/) and renders into
`analyses/controls/figures/`, so it works on a fresh clone with no downloads and no GPU.

`--run` re-embeds eight control rungs through Evo2-7B and needs **experiment 1's outputs**: stage C1
reads `data/cache/mammal_cds_positions.json` (experiment 1, stage A7) and stage D2 reads the W2
family order from `blocks15/betweenfam_ot_metadata.json` (experiment 1, stage B3). Run
[experiment 1](../../experiments/exp1_gene_family_geometry.sh) first. Each rung is resumable, and
scoring upserts, so it can be run against a partial chain. It needs no external tools beyond the
Python dependencies.

## Panels

| Panel | Shows | Generator |
|---|---|---|
| `fig04_controls_vs_natural_by_layer` | preservation of the natural geometry per rung, per block | `scripts/controls/plot_control_wasserstein.py --pub` |
| `fig12_paired_p3_protein_vs_nucleotide` | the matched third-position pair, protein vs nucleotide | `scripts/mammalian_orthologs/paired_p3_figure.py --pub` |
| `fig13_control_identity_vs_rho` | each rung's nucleotide identity to its source against the rho it produces | `scripts/controls/control_sequence_identity.py --stage figure --pub` |

The `fig04`/`fig12`/`fig13` stems are the panel numbers these figures carry in the repository's
figure set; nothing renumbers when this analysis is run or skipped.

## Code and data

| Path | Purpose |
|---|---|
| [`figure_data/`](figure_data/) | the three tracked tables the panels read |
| `scripts/controls/` | sequence operators, identity analysis, and the preservation plot |
| `scripts/mammalian_orthologs/mammal_controls_score.py` | per-block W2 and angular preservation scoring |
| `scripts/mammalian_orthologs/controls_score_graphfree.py` | independent graph-free recomputation, kept as a cross-check |
| `scripts/mammalian_orthologs/paired_p3_figure.py` | the matched-pair panel |
| `scripts/mammalian_orthologs/embed_cds_masked_mammal.py` | shared with experiment 1; `--control <rung>` embeds a control |
| [`scripts/controls/control_metrics_guide.md`](../../scripts/controls/control_metrics_guide.md) | what each metric means and which comparisons are valid |

## Which stage writes what

`mammal_controls_score.py` (stage D1) writes the per-block
`controls/control_{between,within}_scores.csv` that `build_figure_data.control_preservation()`
collapses into `figure_data/control_preservation.csv` — **this is what panel 04 reads**.

`controls_score_graphfree.py` (stage D2, ~2 h CPU) writes
`results/_{ot,angular}_control_preservation_*.csv`. Nothing in the repository reads those files:
they are an independent graph-free recomputation of the same quantities, kept as a cross-check.
**Stage D2 is optional** — skipping it changes no panel. Do not skip D1 for it: D1 is the stage the
panels actually depend on.
