# Interpreting the control baselines

The composition / ablation controls answer **two distinct questions**, scored along **two
axes**. The CSVs put several of these side by side, so the same file can contain columns
that look unrelated. This note explains what each column means and how to read it, so a low
value in one column and a high value in another are not mistaken for a contradiction.

---

## The two questions

### 1. Reconstruction (a.k.a. **preservation**)
> *Does the control's latent geometry look like the **natural** run's latent geometry?*

This is a control-vs-**model** comparison: take the natural embeddings' geodesic and the
control embeddings' geodesic and correlate them (Spearman ρ). It asks whether the ablated
property of the sequence was **load-bearing for the model's geometry**.

- **ρ near 1** → the control reconstructed the natural geometry, i.e. the model did **not**
  rely on the aspect that control destroyed (or that aspect was trivially recoverable).
- **ρ falling toward 0** → the ablated aspect carried real structure in the model's
  representation; removing it moved the geometry.

This is the metric used in the cross-model comparison table (the "easy reconstruction"
question). It has no external ground truth — it only compares model to model.

### 2. Recovery
> *Does the control's geometry still track an **external biological ground truth**?*

This is a control-vs-**ground-truth** comparison: correlate the control geodesic against a
fixed reference signal — **phylogeny** (patristic tree distance, GPN-Star) or **host
taxonomy** (Evo2 cross-kingdom). It asks whether the biological signal the natural run
recovered **survives** the ablation.

- **ρ near the natural run's ρ** → the signal survived; whatever the control preserved is
  enough to explain that biological signal (i.e. the signal may be a composition artifact).
- **ρ collapsing toward 0** → the ablation destroyed the biological signal.

**Crucial caveat for reading recovery:** recovery is only interpretable when the **natural**
run itself has signal to lose. If the natural run's recovery ρ is already near zero (as
GPN-Star is on the *within-family* axis at its deep layers — it has little within-family
phylogenetic signal there to begin with), then the controls' recovery ρ will also be near
zero, and that says **nothing** about the controls. Always read the `natural` row first; if
it is near zero, ignore the recovery column for that run and rely on **reconstruction**.

These two questions are independent. A control can fully reconstruct the natural geometry
(high preservation) while that geometry has no biological recovery signal at all — the two
columns are answering different things and should not be compared to each other.

---

## The two axes

- **WITHIN-family** — compare the **gene × gene** geodesic *inside each family*, then average
  the per-family ρ with equal weight. This is the sensitive, per-gene view (no averaging
  across genes). Families below the member floor are dropped (blank rows).
- **BETWEEN-family** — compare the **family-centroid** geodesic (one node per family).
  Because each family is collapsed to its centroid, per-gene perturbations average out, so
  the between axis is intrinsically more robust (preservation tends to read higher here than
  within for the same control).

---

## Where each number lives

### GPN-Star (human panel)

| Axis | File | Column | Question |
|---|---|---|---|
| WITHIN | `results/<run>/controls/control_within_scores.csv` | `rho_geodesic_vs_natural` | **reconstruction** (preservation) |
| WITHIN | `results/<run>/controls/control_within_scores.csv` | `rho_geodesic_patristic` | **recovery** vs phylogeny (read the `natural` row first) |
| BETWEEN | `results/<run>/msa_controls/msa_control_between_scores.csv` | `rho_vs_natural_geodesic` | **reconstruction** (preservation) |
| BETWEEN | `results/<run>/msa_controls/msa_control_between_scores.csv` | `rho_geodesic_vs_pfamjsd` | supplementary: centroid geometry vs Pfam-JSD homology |

`control_within_scores.csv` is **per-family rows** (with a `natural` block where preservation
is 1.0 by definition); take the **mean over families per condition** to get the headline
number. The GPN between-control keeps its original `msa_controls/` location.

### Evo2 (human panel and cross-kingdom / ortholog panel)

| Axis | File | Column | Question |
|---|---|---|---|
| WITHIN | `results/<run>/controls/control_within_scores.csv` | `rho_geodesic_vs_natural` | **reconstruction** (preservation) |
| WITHIN | `results/<run>/controls/control_within_scores.csv` | `spearman_geodesic_taxonomy` | **recovery** vs host taxonomy *(ortholog panel only; human paralogs have no taxonomy)* |
| BETWEEN | `results/<run>/controls/control_between_scores.csv` | `rho_vs_natural_centroid` | **reconstruction** (preservation) |
| BETWEEN | `results/<run>/controls/control_between_scores.csv` | `rho_centroid_vs_pfamjsd` | supplementary: centroid geometry vs Pfam-JSD homology |

The **human** Evo2 panel reports preservation only (no taxonomy column), since human paralogs
have no cross-organism taxonomy ground truth. The **ortholog** panel reports both.

---

## A third column, when the question is "is the ladder just sequence retention?"

`scripts/controls/control_sequence_identity.py` writes `<run>/control_sequence_identity.csv` — no model
involved, one row per control:

| column | meaning |
|---|---|
| `pos_identity` / `edit_identity` / `aa_identity` | identity of the control to **its own source sequence** (positional, global-edit-distance, frame-0 protein) |
| `null_pos_identity` / `null_edit_identity` | the same identity between **two independent draws of the same control** — the part its constraint forces with no information from the specific source |
| `pos_excess` / `edit_excess` | identity − null, i.e. the identity attributable to **retained source sequence** |
| `pos_identity_chance` | Σ_b p_src(b)·p_ctl(b), the composition-only floor |
| `p1/p2/p3_identity`, `frac_bases_changed`, `frac_changes_at_p3` | where in the codon a rung's changes fall |

**Read `pos_excess`, not `pos_identity`.** "77% identical" (synonymous recode) and "28% identical"
(6-mer shuffle) are not comparable quantities — the first is mostly the genetic code, the second
mostly base composition. Only the excess over each rung's own null is retained source sequence, and
measured 2026-08-25 it is **≈0 (±0.001) for every rung on every panel**. A rung's `pos_identity` is
therefore a measure of how tight its constraint is — which is the rung's *definition* — and not an
independent explanatory variable for its ρ.

**What this does and does not settle.** It rules out "recode's ρ is high because the recoded CDS
still contains most of the original nucleotides" — the nucleotides it shares are the ones the
protein forces, shared equally with an unrelated recode of the same protein. It does **not**
separate "the model reads the protein" from "the model reads nucleotides at protein-determined
positions", because those two are collinear by construction in a synonymous recode.

## The rung that separates them: `missense_subset`

`missense_subset` is `synonymous_recode`'s nonsynonymous counterpart, drawn *from* that locus's own
recode and editing **only bases the recode itself edited** — a strict subset of its changes, never a
base it left alone (verified on every sequence of both panels). Where no amino-acid change is
reachable inside the recode's sites for a codon — the common case, since a 4-fold degenerate codon's
wobble base cannot be changed to a different amino acid — that codon is left **entirely alone**
rather than edited somewhere the recode did not touch.

Measured on the cross-kingdom panel:

| | `synonymous_recode` | `missense_subset` |
|---|---|---|
| coding bases changed | 23.1% | **12.3%** (53% as many) |
| nt identity to source | 0.769 | **0.877** |
| amino-acid identity | 1.000 | **0.717** |
| p1 / p2 / p3 identity | 0.916 / 0.972 / 0.413 | 0.918 / 0.973 / 0.739 |

Note the positional profile: positions 1 and 2 track the recode almost exactly, and the whole
difference is at the wobble base — which is the point. This rung perturbs the nucleotide sequence
**strictly less** than the recode does, at a subset of the same sites, while damaging the protein
the recode preserved.

That asymmetry is what makes it readable, and the reading is **one-sided**:

| observation | conclusion |
|---|---|
| `missense_subset` ρ falls below `synonymous_recode` | the geometry tracks the **protein** — fewer nucleotides were moved, so nucleotide loss cannot explain the drop |
| `missense_subset` ρ holds at `synonymous_recode` | suggestive of nucleotide identity, but **weak** — see below |

**Why the null direction is weak.** Skipping unsatisfiable codons buys the subset guarantee at the
price of a gentler perturbation on *both* axes at once: only ~28% of residues change. A rung that
leaves 72% of the protein intact cannot support "ρ did not fall, therefore protein does not matter".
Always report its achieved `aa_identity` next to its ρ. The informative direction is ρ falling while
nucleotide identity rises.

## Quick reading recipe

1. Pick the **axis** you care about (within = sensitive per-gene; between = robust, family-level).
2. For the "did the model rely on this sequence property" question, read the **reconstruction**
   column (`rho_*_vs_natural*`): high = not load-bearing / easily reconstructed, low = load-bearing.
3. For the "did the biological signal survive" question, read the **recovery** column — but
   only after checking the `natural` row has signal; if natural ≈ 0, skip it.
4. Don't compare a reconstruction column to a recovery column — they answer different questions.
