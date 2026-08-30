# Between-family ground-truth baselines — design & rationale

**Status:** design, for critical review before implementation. Companion to
`gene_family_design.md` (which covers the pipeline as a whole) and to the Mod
suggestions in `project_overview.txt`.

**What this fixes.** The current between-family (Axis A) result has exactly one
continuous ground truth — Pfam JSD — and it is null (ρ = +0.096, p = 0.33). We can
say families are *separable* (within/between ratio 1.74) but not say *what kind of
relatedness* the geometry encodes. This document specifies a **multi-axis** ground
truth so we can ask the question the Mod is really after:

> Does the latent geometry track **evolutionary homology**, **biochemical mechanism**,
> or **biological role** — and can we tell those apart?

The panel was built for this. It contains deliberately **convergent** (non-homologous
but functionally/chemically related) families — the three carbonic anhydrases, globins
vs hemerythrin, type-1 vs type-2 opsins — which only become informative once homology
and analogy are *separate* axes.

---

## 0. Result on the 15-family run (2026-06-21)

Implemented in `scripts/baselines/between_family_baselines.py`; scores in
`between_family_baseline_scores.csv`; figure `between_axis_scores.png`. Spearman ρ of the
family-centroid geodesic vs each baseline (105 family pairs, F=15):

| Axis | Baseline | ρ | Mantel p |
|------|----------|----|----------|
| Homology | homology_tier | +0.035 | 0.36 |
| Homology | pfam_clan | +0.034 | 0.36 |
| Mechanism | cofactor | −0.076 | 0.66 |
| Mechanism | ec_number | +0.102 | 0.16 |
| Context | gas_process | +0.040 | 0.31 |
| Context | kegg_module | — | (constant; dropped) |
| Control | gc_content | −0.275 | — |
| Control | **kmer** | **−0.781** | — |

**Read:** a clean null for biology between families — no homology / mechanism / context
axis tracks the between-family geometry (all Mantel p > 0.16). The only strong signal is
the **composition control, and it is *negative*** (k-mer ρ = −0.78): family pairs that are
geodesically far apart are, if anything, *more* compositionally similar. So between
families, Evo2's geometry is neither a biology detector nor a naïve k-mer counter on this
panel. This sharpens the paper's framing: the gene-family contribution is **Axis B
(within-family)**, where the per-family ρ is strong; Axis A separability (ratio 1.74)
stands, but *what* organizes the families is not any single biological axis we can name.
Two structural reasons, both worth stating: (a) the panel is by design a set of mutually
**non-homologous** families, so the homology axis has almost no between-family gradation
to detect; (b) the convergent-analogy contrast (the three CAs; globins↔hemerythrin) is a
*handful* of pairs against 105 — even if the geometry honoured it, it cannot move a
rank correlation. The convergent pairs are better examined *directly* (their geodesic
rank among all pairs) than via a global ρ — a targeted follow-up, noted in §7.

---

## 1. The three axes (+ one control)

The Mod states the same idea twice — a comparison table (Orthology / Domain-structure /
Function-cofactor) and a "3 baselines" list (Homology / Molecular-mechanism /
Biological-context). They collapse to **three orthogonal axes**. The table's single
"function/cofactor" column splits: **cofactor → mechanism**, **role/process → context**.

| Axis | Asks | Convergent families should be… |
|------|------|-------------------------------|
| **1. Homology** | shared *ancestry* | **FAR** (that's what makes them convergent) |
| **2. Mechanism / chemistry** | shared *machinery* (cofactor, catalysed reaction) | **CLOSE** if chemistry is shared |
| **3. Context / process** | shared *biological role* | **CLOSE** if role is shared |
| **(control) Composition** | shared *nucleotide statistics* | the null any real signal must beat |

The whole experiment is the **contrast**: a family pair that is far on Axis 1 but close
on Axis 2/3, which the geometry nonetheless places close, is evidence the model encodes
**biochemical analogy beyond homology**.

---

## 2. Baseline → axis map

- **Axis 1 — Homology** *(3 baselines)*
  - **Curated superfamily / fold tier** — ordinal, hand-assigned, Pfam-clan cross-checked
  - **Pfam HMM JSD** — continuous, data-derived (graded homology; see §4 for the caveat)
  - **Pfam clan** *(implemented in place of eggNOG — see §6)* — data-derived categorical
    homology from InterPro clan membership; the lightweight cross-check on the curated tier
- **Axis 2 — Mechanism / chemistry** *(2 baselines)*
  - **Cofactor distance** — curated metal/cofactor set
  - **EC-number distance** — curated, KEGG-sourced enzyme-class hierarchy
- **Axis 3 — Context / process** *(2 baselines)*
  - **Gas-substrate + biological-process distance** — curated
  - **KEGG module / pathway overlap** — data-derived (free from our KO source)
- **Control — Composition** *(2 baselines)*
  - **k-mer cosine distance** — already built
  - **GC-content distance** — trivial companion null

Scoring: every baseline becomes an F×F (F=15) family distance matrix; we Spearman-
correlate its upper triangle against the latent **family-centroid geodesic** matrix and
report a Mantel p-value (permutation over family labels). The deliverable is a bar chart
of ρ per baseline, grouped by axis — "which axis does the geometry track."

---

## 3. Per-family curated annotations (the input to vet)

These hand annotations feed Axes 1–3. **This table is the thing to examine critically** —
the data-derived baselines (Pfam JSD, eggNOG, KEGG) are checks on it.

| Family | Gas | Cofactor | Fold / superfamily | EC | Biological process |
|--------|-----|----------|--------------------|----|--------------------|
| globins | O₂ | heme b | globin fold | — | O₂ transport / storage / sensing |
| hemerythrin | O₂ | **non-heme di-iron** | hemerythrin 4-helix | — | O₂ transport / storage |
| heme_copper_oxidase | O₂ | heme a/a₃ + **Cu_B** | HCO superfamily | 7.1.1.9 | aerobic respiration (terminal oxidase) |
| cytochrome_p450 | O₂ | heme b (thiolate) | P450 fold | 1.14.– | monooxygenation (xenobiotic/biosynth) |
| nitric_oxide_synthase | O₂→**NO** | heme b (thiolate) + BH₄ | NOS oxygenase | 1.14.13.39 | NO synthesis / signalling |
| heme_oxygenase | O₂→**CO** | heme (substrate) | HO all-α | 1.14.14.18 | heme catabolism → CO + biliverdin |
| carbonic_anhydrase_alpha | CO₂ | **Zn** | α-CA | 4.2.1.1 | CO₂ hydration |
| carbonic_anhydrase_beta | CO₂ | **Zn** | β-CA | 4.2.1.1 | CO₂ hydration |
| carbonic_anhydrase_gamma | CO₂ | **Zn / Fe** | γ-CA (LβH) | 4.2.1.1 | CO₂ hydration |
| methane_monooxygenase | CH₄ | di-iron (sMMO) / **Cu** (pMMO) | mixed | 1.14.13.25 / 1.14.18.3 | methane oxidation |
| methyl_coenzyme_m_reductase | CH₄ | **Ni–F430** | MCR | 2.8.4.1 | methanogenesis |
| nitrogenase | N₂ | **FeMo-co / [4Fe-4S]** | P-loop NTPase (NifH) | 1.18.6.1 | N₂ fixation |
| ras_gtpases | — | Mg²⁺ / GTP | P-loop GTPase | 3.6.5.2 | signal-transduction switch |
| olfactory_receptors | — | — | class-A GPCR (7TM) | — | olfactory signal transduction |
| opsins | light | retinal | **type-2 = class-A GPCR; type-1 = microbial rhodopsin** | — | phototransduction / ion transport |

Key convergence / homology calls (drive the Axis-1 tier matrix):
- **The three CAs are *not* homologous to one another** — α, β, γ are three independent
  inventions of a Zn-CO₂ hydratase. Same mechanism, same context, zero shared ancestry.
- **globins vs hemerythrin** — both reversible O₂ carriers, *different* metal centre
  (heme vs non-heme di-iron), no common ancestor.
- **olfactory_receptors vs type-2 opsins** — both class-A GPCRs → genuinely homologous
  (Pfam clan CL0192). Type-1 opsins are *not* (microbial rhodopsins, convergent 7-TM).
- **ras_gtpases vs nitrogenase (NifH)** — both P-loop (Walker-A) NTPases → deep, distant
  homology; everything else about them differs.
- **NOS vs P450** — both heme-thiolate monooxygenase chemistry (cofactor analogy) but
  the oxygenase folds are treated as non-homologous here (conservative).

---

## 4. Method-by-method: theory, math, what it can and cannot see

Notation: F families; for family *a*, let S_a be its set of member CDS. A baseline
produces D ∈ ℝ^{F×F}, scored by Spearman ρ of `upper(D)` vs `upper(D_geodesic_centroid)`.

### Axis 1 — Homology

**(1a) Curated superfamily / fold tier** — *ordinal, interpretable*
Each family is assigned a fold/superfamily label (col 4 of §3). The pair distance is an
ordinal tier (the Mod's tiers, inverted to a distance):

```
D_tier(a,b) = 0   if a == b
            = 1   same superfamily / Pfam clan (homologous)        e.g. OLFR–opsins(t2)
            = 2   distant shared fold division (e.g. both P-loop)   e.g. Ras–NifH
            = 3   unrelated
```
- *Sees:* curated, panel-specific homology — the cleanest statement of "are these
  descended from one ancestral gene/domain."
- *Cannot see:* graded distance within a tier; it is coarse by construction.
- *Cross-check:* Pfam **clan** membership pulled from InterPro (we already fetch Pfam
  entries for JSD) validates the tier-1 calls without hand-assertion.

**(1b) Pfam HMM JSD** — *continuous, data-derived* — **the one you flagged**
Implementation (`scripts/baselines/pfam_hmm_jsd.py`): fetch the family's canonical Pfam HMM; take its
match-state emission matrix E_a ∈ ℝ^{M_a×20}; **average over the M_a columns** to one
amino-acid vector p_a = mean_c E_a[c] ∈ Δ^{20}; then

```
D_JSD(a,b) = JSD(p_a, p_b)        (base-2 Jensen–Shannon divergence, squared)
JSD(p,q)   = ½ KL(p‖m) + ½ KL(q‖m),   m = ½(p+q),   KL(p‖m)=Σ p_i log₂(p_i/m_i)
```
- **Why it is a *homology* baseline, not mechanism:** the HMM is built from a deep MSA of
  the family's homologs, so the profile is an evolutionary object. JSD between two
  profiles is small when their conserved cores share residue composition — i.e. graded
  domain homology. It is the **continuous companion** to the categorical tier (1a).
- **The caveat that explains ρ≈0:** collapsing E_a to a *mean* 20-vector throws away all
  positional and structural information — it reduces a profile to "average conserved-
  residue composition." Two unrelated families with similar overall amino-acid usage look
  close; two homologs with compositionally distinct cores look far. So it is a *weak*
  homology signal, which is consistent with the null result. A stronger drop-in would be
  **HMM–HMM comparison** (HHsearch/`hhalign` probability) which keeps positional profile
  structure; flagged as a possible upgrade, not in scope now.
- *Sees:* conserved-core composition similarity. *Cannot see:* positional homology, fold,
  anything for families without a clean single Pfam (γ-CA, MMO use approximate reps).

**(1c) Pfam clan — implemented; eggNOG deferred** — *data-derived categorical homology*
Pfam clans are Pfam's expert-curated homology groupings (families in the same clan are
homologous by descent), fetched from InterPro (`entry/pfam/<acc>` → `metadata.set_info`)
in the same loop that already pulls Pfam HMMs for JSD. `D_clan(a,b) = 0` if a and b share
a clan, else 1; no-clan families are singletons (distance 1 to all). On this panel the
*only* shared clan is **OLFR ↔ opsins (CL0192, GPCR_A)** — which exactly reproduces the
hand-coded tier-1 edge, validating the curated tier from data. It is near-degenerate as a
correlation baseline (one informative pair) precisely because the panel is, by design, a
set of mutually non-homologous families: **there is almost no between-family homology
*gradation* to detect** — the homology structure lives *within* families (Axis B).

> **Why Pfam clan instead of eggNOG.** eggNOG-mapper is not installed and needs a ~50 GB
> DB + DIAMOND + a multi-hour run; and on a 15-family cross-kingdom panel the OG sets
> would be near-disjoint — exactly the degeneracy KEGG modules hit below (only 4/15
> families carry any module, none shared → constant matrix → dropped). Pfam clan is the
> lighter expert-curated homology signal that reuses the InterPro plumbing already here.
> eggNOG remains a documented option if a graded orthogroup distance is wanted later.

**(eggNOG, original spec) orthologous-group distance** — *data-derived orthogroup*
Map each member CDS (protein) to eggNOG OGs (via eggNOG-mapper, or the per-family
dominant OG). Represent family *a* by its multiset of OGs across taxonomic levels; then

```
D_egg(a,b) = 1 − Jaccard(OG_a, OG_b)      OG_a = set of eggNOG OGs hit by S_a
```
or, finer, 1 − (depth of the most-specific eggNOG taxonomic level at which a and b share
an OG) / (max depth). Two families sharing an OG at a deep (ancestral) level are
homologous; sharing only at a shallow level (or not at all) are not.
- *Sees:* curated phylogenomic orthology across bacteria/archaea/eukaryotes — the most
  "objective" homology signal, and the data-derived cross-check on (1a).
- *Cost / caveat:* eggNOG-mapper needs the ~50 GB eggNOG DB + DIAMOND; heavier than the
  rest. Between-family orthogroup distance is only well-defined for families that *do*
  land in shared deep OGs — convergent families correctly share none (D = 1), which is
  the point, but it makes the metric coarse (mostly 1s off-diagonal). Treated as a
  cross-check on the tier matrix, not the primary Axis-1 number.

### Axis 2 — Mechanism / chemistry

**(2a) Cofactor distance** — *curated set*
Each family → a cofactor set C_a (col "Cofactor" of §3), e.g. {heme_b}, {non-heme_di-iron},
{Zn}, {FeMo-co}. A graded scheme rather than pure Jaccard so chemically-near cofactors
score near (heme_a vs heme_b closer than heme vs Zn):

```
D_cof(a,b) = 1 − max_{x∈C_a, y∈C_b} sim(x,y)
sim(x,y) ∈ [0,1] from a small curated cofactor-similarity table
            (e.g. all hemes 0.8–1.0 to each other; heme↔non-heme-Fe 0.4; metal↔none 0)
```
- *Sees:* shared catalytic chemistry independent of ancestry — exactly the analogy
  signal. globins/HCO/P450/NOS/HO cluster (all heme); the CAs cluster (all Zn).
- *Cannot see:* anything beyond the curated cofactor call; mixed families (MMO sMMO/pMMO)
  are approximate.

**(2b) EC-number distance** — *curated / KEGG hierarchy*
EC numbers are a 4-level hierarchy `c1.c2.c3.c4`. Distance = how early they diverge:

```
D_EC(a,b) = 1 − (shared leading-level count) / 4
            e.g. 4.2.1.1 vs 4.2.1.1 → 0;  1.14.13.39 vs 1.14.14.18 → 1−2/4 = 0.5
```
Non-enzymes (GPCRs, opsins) get a sentinel "non-enzyme" class, max distance to enzymes,
0 to each other.
- *Sees:* the reaction catalysed — a function axis orthogonal to ancestry. The **three
  CAs share 4.2.1.1 exactly** → D_EC = 0, the sharpest analogy test in the panel.
- *Cannot see:* substrate identity beyond reaction class; regulation; non-catalytic role.

### Axis 3 — Context / process

**(3a) Gas-substrate + biological-process distance** — *curated*
Two categorical labels per family: the gas it acts on (O₂/CO₂/N₂/CH₄/NO/CO/none) and a
coarse process label (col "Biological process"). Distance combines them:

```
D_ctx(a,b) = ½·[gas_a ≠ gas_b] + ½·(1 − process_similarity(a,b))
```
with process_similarity from a small curated process ontology (e.g. "O₂ transport" vs
"aerobic respiration" both under aerobic-O₂-handling → partial credit).
- *Sees:* shared biological role regardless of mechanism. globins (O₂ transport) and HCO
  (O₂ respiration) share the gas but differ in process; the three CAs share both.
- *Cannot see:* anything we did not curate; coarse by design.

**(3b) KEGG module / pathway overlap** — *data-derived*
We already pull each family from KEGG KOs, so each family inherits the KEGG modules /
pathways its KOs belong to. Family *a* → module set M_a:

```
D_kegg(a,b) = 1 − Jaccard(M_a, M_b)
```
- *Sees:* curated pathway co-membership — a data-derived Axis-3 signal at no extra fetch.
- *Cannot see:* relatedness for families whose KOs carry sparse module annotation
  (expect many disjoint sets → coarse, like eggNOG).

### Control — Composition (the null biology must beat)

**(4a) k-mer cosine distance** — built (`scripts/baselines/kmer_sequence_divergence.py`).
Per CDS, an L1-normalised 4^k-dim nucleotide k-mer frequency vector f; pairwise
`1 − cos(f_i, f_j)`; family distance = mean over cross-family member pairs.
- *Sees:* raw nucleotide composition / codon-ish statistics. If the geometry tracks this
  as well as it tracks Axes 1–3, the "biological understanding" claim collapses to
  "compositional statistics" — the calibration the Mod insists on.

**(4b) GC-content distance** — trivial companion.
`D_GC(a,b) = |meanGC(S_a) − meanGC(S_b)|`. The crudest composition null; included because
it is free and a useful floor.

---

## 5. The contrasts that make the result (read this as the hypothesis grid)

| Pair | Axis 1 Homology | Axis 2 Mechanism | Axis 3 Context | Geometry-close ⇒ |
|------|-----------------|------------------|----------------|------------------|
| α-CA vs β-CA vs γ-CA | **far** (3 origins) | **identical** (Zn, EC 4.2.1.1) | **identical** (CO₂) | model encodes mechanism/role over homology — *cleanest analogy test* |
| globins vs hemerythrin | far | differs (heme vs di-Fe) | shared (O₂ transport) | model encodes *role* (process) analogy |
| globins vs HCO/P450/NOS/HO | far | shared (heme) | partial (O₂) | model encodes *cofactor* chemistry cluster |
| OLFR vs type-2 opsins | **close** (GPCR clan) | shared (7TM) | differ (odorant vs light) | homology recovered |
| type-1 vs type-2 opsins | far | shared (7TM, retinal) | shared (photo) | convergence test (our planted control) |
| Ras vs NifH | distant (P-loop) | differ | differ | deep-fold homology surfaces only if Axis 1 dominates |
| anything vs Ras / OLFR | far on all | far | far | negative-control floor |

If the geometry's ρ is highest against **Axis 1** → it is a homology detector (the
expected, less interesting result). If it is comparably high or higher against **Axis 2/3
on the convergent rows specifically** → the headline: *Evo2 places non-homologous
heme/O₂/CO₂ proteins closer than homology predicts, capturing biochemical analogy.*

---

## 6. Feasibility / cost summary

| Baseline | Source | Cost | New dependency |
|----------|--------|------|----------------|
| tier (1a) | curated + InterPro clan | trivial | — (reuse InterPro fetch) |
| Pfam JSD (1b) | InterPro HMM | built | — |
| eggNOG (1c) | eggNOG-mapper + DB | **heavy** (~50 GB DB, DIAMOND) | eggNOG-mapper |
| cofactor (2a) | curated | trivial | — |
| EC (2b) | curated / KEGG | trivial | — |
| gas/process (3a) | curated | trivial | — |
| KEGG module (3b) | KEGG REST (have KOs) | light | — (reuse KEGG client) |
| k-mer (4a) | sequences | built | — |
| GC (4b) | sequences | trivial | — |

Recommended build order: curated axes + KEGG module + GC first (one module, one figure,
no heavy deps) → wire scoring & figure → add the data-derived homology cross-check.
**Status (2026-06-21):** all curated axes + k-mer/GC controls + Pfam-clan + KEGG-module
built, scored, figured, and wired into `run_ortholog_gene_pipeline_evo2.sh` (Step 3b). KEGG
module is dropped as constant (§4); eggNOG superseded by Pfam clan (§4 1c).

---

## 6b. Convergent-pair rank test + GPN-Star comparison (2026-06-21)

Both built. The rank test is in `between_family_baselines.py` (`convergent_pair_ranks.csv`);
GPN-Star runs via `--seq-source gpn` on its run dir (shares the curated annotations; adds
`hox`, `carbonic_anhydrase`). Per-family patristic gold standard:
`scripts/baselines/protein_alignment_patristic_seqid.py` (MAFFT→FastTree→patristic, model-agnostic).

**Convergent-pair rank test (Evo2)** — percentile of each pair's geodesic closeness among
all 105 pairs (0 = closest). The global null *was* hiding structure, but not clean analogy:

| Pair | type | geodesic %ile | homology %ile |
|------|------|---------------|---------------|
| globins↔hemerythrin | convergent O₂ | **0.0 (closest of all)** | 51 |
| globins↔ras_gtpases | **negative control** | **1.0** | 51 |
| globins↔heme_oxygenase | heme cluster | 2.9 | 51 |
| 3× carbonic anhydrases | convergent CO₂ | 14–26 | 51 |
| globins↔NOS | heme cluster | 88 | 51 |
| OLFR↔opsins | homologous GPCR | 38 | 0 |

The closest pairs are **globin-centric**, and a *negative control* (globins↔Ras) ranks
2nd-closest — so the proximity is **globins sitting hub-like in the manifold**, not a clean
biochemical-analogy signal. The CA trio is moderately close (mechanism/context put them at
the 2–6th %ile, geometry at 14–26) — a weak nod to analogy, not a clean hit. Honest read:
no clean homology-vs-analogy story between families; the structure is a globin hub.

**GPN-Star vs Evo2 (the architecture contrast, plan takeaway #3).** Same baselines, 9
human-paralog families (36 pairs). The decisive baseline is the *continuous* homology one,
**Pfam JSD** (the categorical tier/clan are near-degenerate on this panel — only OLFR↔opsins
share a clan — so they read weak regardless):

| Axis (baseline) | Evo2 ρ | GPN-Star ρ |
|------|--------|-----------|
| homology — **Pfam JSD** (graded) | +0.096 | **+0.396** (rank-p .017, Mantel .10) |
| homology — tier/clan (categorical) | +0.035 | +0.155 |
| k-mer (composition) | **−0.781** | +0.310 |
| GC | −0.275 | +0.247 |

**GPN-Star's between-family geometry tracks graded domain homology (Pfam JSD +0.40) — its
strongest signal, ahead of composition.** It is *suggestive*, not conclusive: the
conservative family-label Mantel is marginal (p≈0.10) at N=9 families, but it is the only
baseline approaching significance and it beats k-mer. Evo2 shows no homology signal
(+0.10 ns) and composition *anti*-correlates (−0.78). So the models are closer to **mirror
images** than a simple sign-flip:

| | within-family (patristic) | between-family (best biological axis) |
|---|---|---|
| **Evo2** | **strong** (ρ .15–.78, all p≈0) | null (Pfam JSD +0.10 ns; k-mer −0.78) |
| **GPN-Star** | ≈0 (paralog tree not recovered) | **graded domain homology** (Pfam JSD +0.40, marginal) |

Mechanistically sensible: GPN-Star is a genome-anchored MLM trained on cross-species
**aligned** windows → it encodes conserved-domain divergence (what Pfam JSD measures)
*between* families; Evo2's autoregressive manifold instead captures the fine evolutionary
gradient *within* each family. (Earlier drafts of this doc mis-stated GPN-Star as
composition-tracking — that conflated the weak categorical clan/tier with the omitted
graded Pfam JSD, which is the real signal.) GPN-Star's negative controls also behave sanely
(globins↔hox 86th %ile = far; Evo2 had globins close to everything).

**Within-family patristic gold standard (the Axis-B headline).** Per-family
MAFFT→FastTree patristic distance vs the within-family geodesic.
- **Evo2: strong and significant for every family** — ρ +0.15 → +0.78, all p≈0
  (opsins +0.71, MMO +0.78, γ-CA +0.70, NOS +0.67, globins +0.64, nitrogenase +0.57,
  mcrA +0.52, Ras +0.52; lowest HCO +0.15). Evo2's within-family geodesic genuinely
  recapitulates the gene tree. This is the project's contribution.
- **GPN-Star: ρ ≈ 0 for every family** (±0.06, mostly ns) — its geodesic does *not*
  recover the gene tree among human paralogs (known paralog-signal weakness + last-layer
  tap). `within_family_patristic.csv` in each run; figure `within_correlations.png`.

The model contrast is the cleanest result of all: **Evo2 recovers within-family
evolutionary structure where GPN-Star does not**, while *neither* recovers a between-family
biological axis. The within-family signal is the gene-family contribution; the
between-family axis is a calibrated null.

**GO baselines (added 2026-06-21).** GO molecular-function (`go_mf`, mechanism axis) and
GO biological-process (`go_bp`, context axis), from GO's curated pfam2go mapping (namespace
via QuickGO; canonical curated fill for the 4 families pfam2go does not map at Pfam level:
α/γ-CA, hemerythrin, sMMO). Both null between families (Evo2 go_mf +0.044, go_bp +0.039;
GPN go_mf −0.024, go_bp +0.155 — all ns), consistent with the other six biological axes.
**Eight independent biological signals now return null between families; only composition
(k-mer/GC) correlates.** That is the calibration the proposal asked for.

**Why not eggNOG/OMA/OrthoDB orthogroup distance?** Three reasons, in order: (1) *cost* —
eggNOG-mapper + its ~50 GB DB + DIAMOND are not installed and would be a multi-hour run on
5,276 proteins (OMA/OrthoDB similar: large bulk files + a protein→CDS join). (2) *It would
be degenerate between these families.* Orthogroup distance is only defined between families
that share a deep OG; this panel is *by design* mutually non-homologous, so the families
land in disjoint OGs → a near-constant "no shared OG" matrix — exactly the degeneracy KEGG
modules hit here (4/15 families mapped, none shared → constant → dropped). High cost for a
predictably uninformative matrix. (3) *Orthogroups are a within-family signal, not a
between-family one* — "which members are 1:1 orthologs" is fine-grained inside a family and
collapses between unrelated families; our within-family structure is already covered by the
patristic tree + k-mer + taxonomy. **Pfam clan is the right-sized substitute**: clans are
expert-curated homology groupings, reuse the InterPro plumbing, and recovered the one real
between-family homology edge (OLFR↔opsins, CL0192) from data. eggNOG remains a documented
option (§7) if a graded orthogroup distance is ever wanted.

---

## 7. Follow-ups this run surfaces

- **Targeted convergent-pair test (high value).** A global ρ over 105 pairs cannot
  reflect a handful of analogy pairs. Instead, for each convergent pair (α/β/γ-CA;
  globins↔hemerythrin; the heme cluster), report its **geodesic-centroid rank among all
  105 pairs** and compare to where homology/mechanism/context place it. "The three CAs
  rank in the closest decile despite zero homology" is a sharper, directly-testable
  statement than a null ρ. Small addition to `between_family_baselines.py`.
- **Within-family gold standard (Axis B), per the PANTHER-for-within-family decision.**
  Replace the coarse taxonomic-rank ground truth with a per-family **patristic** distance:
  align each family (mafft/muscle) → tree (FastTree/IQ-TREE) → patristic, for the families
  with enough clean coverage. Needs mafft + FastTree installed (neither present yet).
  Strengthens the result that is already strong rather than the null between-family one.
- **eggNOG (if a graded orthogroup distance is later wanted):** install eggNOG-mapper +
  DB (~50 GB) + DIAMOND, map member proteins → OGs, `1 − Jaccard(OG sets)`. Expect
  near-disjoint OGs on this panel (coarse), as KEGG modules showed.
- **HMM–HMM JSD upgrade:** replace mean-emission JSD with HHsearch/`hhalign` probability
  to keep positional profile structure (§4 1b caveat).
