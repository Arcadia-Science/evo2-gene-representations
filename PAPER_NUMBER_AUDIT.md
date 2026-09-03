# Numerical provenance audit — *Beyond sequence statistics: Testing gene family organization in DNA language model representations*

**Audit date:** 2026-09-01 · **Scope:** every numerical claim in the manuscript, traced to the exact
stored artifact it came from · **Mode:** read-only. No analysis code, manuscript, result, cached
output, figure or table was modified. The only files created are `PAPER_NUMBER_AUDIT.md`,
`PAPER_NUMBER_AUDIT.csv` and `audit_scripts/`.

---

## 0. What was audited

### Manuscript source

| | |
|---|---|
| **File audited** | `pub/Beyond sequence statistics_ Testing gene family organization in DNA language model representations (3).pdf` |
| PDF title | "Beyond sequence statistics: Testing gene family organization in DNA language model representations" |
| Producer | Skia/PDF m154 **Google Docs Renderer** — there is no `.tex`, `.md` or `.qmd` source in this repository |
| Pages | 23 |
| Modified | 2026-09-01 15:41 |
| Git status | **`pub/` is untracked** (`?? pub/` in `git status`) |

**Line numbers throughout this report refer to `pdftotext -layout` output**, reproducible with:

```bash
pdftotext -layout "pub/Beyond sequence statistics_ Testing gene family organization in DNA language model representations (3).pdf" /tmp/manuscript.txt
sed -n '240,260p' /tmp/manuscript.txt      # e.g. the Experiment 1 result paragraph
```

There is only one manuscript in the repository, so there is no risk of having audited an obsolete
draft. But note this is a **working draft**, not a submission candidate: it still contains
`(CITE HGNC)`, `shared xxx`, `TO DO: add fig`, `(to be replaced:)`, `[give the axis ranges here]`,
`(check if via an amino acid [aa] alignment or nucleotide alignment)`, `Title.`, `blah blah`,
`organizes genetic information in an X way`, and every citation rendered as `:ref`.

### Result artifacts

The tracked numeric surface is `figure_data/` (21 tables + `MANIFEST.csv`, written by
`scripts/build_figure_data.py`). The three publication runs it is built from are **not tracked**:

| Run directory | Supplies |
|---|---|
| `results/2026-07-16_mammalian-orthologs-transcript_cdsmask/` | Experiments 1 and 2 (32 `blocks*/` sub-dirs) |
| `results/2026-07-28_evo2-platypus-paired/stage2_cds_mean/` | Figures 6a/6b (the 103-gene paired panel) |
| `results/2026-08-08_platypus-strat-400/` | Experiment 3 (the 400-pair stratified panel) |

Both are present on this machine, so I could audit at both levels.

### Verification depth achieved

| Level | What was done | Coverage |
|---|---|---|
| **L1 — artifact match** | Manuscript value vs the stored result | all 99 claims |
| **L2 — recomputation from upstream data** | Recomputed the statistic from the immediate upstream observations | Experiment 1 between-family (all 32 layers, all 3 baselines), Experiment 1 within-family (3 families × 2 baselines at layer 15), Experiment 2 control preservation (7 of 8 rungs at layer 15), the synonymous-recode and paired-p3 sequence statistics, and every Experiment 3 aggregate that `figure_data/` supports |

---

## 1. Numbers you should personally check — a 30-minute list

Ranked by how much a mistake would cost. Every command runs from the repository root and prints the
underlying value.

### 🔴 1. Experiment 3's headline steering percentages come from a metric the repo labels *historical only*

**Claim (lines 596–599):** stratum 0 private-bp recovery "~42% unsteered to ~46% at α = 4";
stratum 4 "from ~36% to ~44%".

**File:** `results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/stage4_scores.csv`
**Column:** `pct_private_correct` — **not** in `figure_data/` at all.

```bash
uv run python - <<'PY'
import pandas as pd
R='results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/'
d=pd.read_csv(R+'stage4_scores.csv'); n=pd.read_csv(R+'stage4_scores_nt.csv')
c=['unsteered','add_a4.0']
for src,col in ((d,'pct_private_correct'),(n,'pct_diagnostic_correct'),(n,'pct_autapomorphy_correct')):
    g=src[src.condition.isin(c)].groupby(['gene','stratum','condition'])[col].mean().reset_index()
    print(col); print(g.pivot_table(index='stratum',columns='condition',values=col,aggfunc='mean')[c].round(3)); print()
PY
```

You will see three different answers for the same sentence:

| stratum | metric | unsteered | α = 4 |
|---|---|---|---|
| 0 | `pct_private_correct` (legacy) | **42.19** | **45.70** ← the paper |
| 0 | `pct_autapomorphy_correct` (figures 7–8) | 39.81 | 44.03 |
| 4 | `pct_private_correct` (legacy) | **36.36** | **44.04** ← the paper |
| 4 | `pct_autapomorphy_correct` (figures 7–8) | 31.69 | 41.64 |

**Where to find it, and is it frozen? Yes.** `.frozen_freeze.sh` snapshotted the whole
Experiment 3 run on 2026-08-31, and that snapshot carries the legacy table with a checksum:

```bash
# the frozen copy, and the manifest line that pins it
ls -la .frozen/2026-08-31/results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/
grep 'stage4_cds_mean_blocks27/stage4_scores.csv' .frozen/2026-08-31/MANIFEST.sha256
# verify the live file still matches the freeze
sha256sum results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/stage4_scores.csv
```

Live and frozen are byte-identical:
`c3a5e8112c9c879b0cb92d18d4f3b4ecdae2754a6c08d6e952f1abe64d17c54f`, and that hash is the one
recorded in `.frozen/2026-08-31/MANIFEST.sha256:19130`. So the number is auditable today and pinned
against silent drift.

Three caveats on that freeze:

* **`.frozen/` is gitignored** (`.gitignore:116`), so it lives only on this machine and in whatever
  backup covers it. It is not part of the tracked repository, and a fresh clone has neither the
  frozen copy nor the run directory.
* The *later* freeze, `.frozen/figure_data_pre_refactor_2026-09-01/`, contains **only**
  `stage4_scores_nt.csv` — the legacy table was not carried forward. `.frozen/2026-08-31/` is the
  single surviving snapshot of it.
* The legacy column is nevertheless **regenerable**: the generated sequences themselves are retained
  (`generations.jsonl.gz`, 9.5 MB, fields `gene`/`condition`/`sample`/`seq`, present both live and in
  the 2026-08-31 freeze), so any site metric can be re-scored from them without re-running the GPU
  generation.

**Why this deserves manual checking.** `scripts/steering/platypus/strat/strat_metric.py:37–48`
describes `pct_private_correct` as: *"PRE-FIX pairwise human-vs-platypus difference, scored through
the protein alignment whose bias runs along the conservation axis — historical only. Its column is
named `pct_private_correct` but it is NOT the `private` metric."* The manuscript's own definition of
"private" (line 609: *"the platypus base is unique relative to the other 24 mammals"*) describes
`pct_autapomorphy_correct`, which is what figures 7 and 8 plot. So the prose numbers, the prose
definition, and the figures are three different things. And because the legacy column was dropped
from `figure_data/`, these four numbers cannot be checked from the **tracked** tables at all — only
from the untracked run directory or the gitignored freeze above.

---

### 🔴 2. "the other 24 mammalian species" — there are at most 22, and typically 19

**Claim (lines 558–559 and 609).**

```bash
uv run python -c "import pandas as pd; d=pd.read_csv('figure_data/exp3_steering_outcomes.csv'); print(d.n_voting_species.describe()); print(d.n_voting_species.value_counts().sort_index())"
uv run python -c "import sys; sys.path.insert(0,'scripts/mammalian_orthologs'); from resolve_orthologs import SPECIES; print(len(SPECIES),'species total, including human and platypus')"
```

The Experiments 1–2 panel is **24 mammals in total**, human and platypus included. The outgroup for
a private site is therefore at most 22, and the observed per-gene voting depth is median 19, max 22,
min 2. The related phrasings at line 655 and line 729 ("our 24-mammal panel", "our 24-mammal set")
are correct; only "the **other** 24" is wrong, and it appears twice.

---

### 🟠 3. The within-family metric-pair medians come from a different distance metric than figures 2–3

**Claim (lines 252–254):** median k-mer/species-tree 0.913 (0.173–0.988); median k-mer/gene-tree
0.862 (−0.256–0.993).

```bash
uv run python - <<'PY'
import pandas as pd
B='results/layer_sweep_summaries/mammalian-orthologs-cdsmask-48fam/'
g=pd.read_csv(B+'within_family_vs_layer_metric_pairs.csv')            # geodesic
A=pd.read_csv(B+'within_family_vs_layer_graphfree_metric_pairs.csv')  # angular -> figures 2-3
for nm,df,ka in (("geodesic (matches the paper)",g,'k-mer composition (CDS)'),
                 ("angular (backs figures 2-3)",A,'k-mer composition (CDS, k=6)')):
    for mb in ('species tree (mammal)','patristic tree'):
        s=df[(df.metric_a==ka)&(df.metric_b==mb)].spearman_across_layers
        print('%-28s %-22s n=%d median=%.4f min=%.4f max=%.4f'%(nm,mb,len(s),s.median(),s.min(),s.max()))
PY
```

The manuscript's six numbers reproduce the **geodesic** roll-up to the digit. The **angular** roll-up
— which is what `figure_data/exp1_within_family_by_layer.csv` and figures 2–3 are built from
(`pub/figures/MANIFEST.md`, and `build_figure_data.py:39-49` reading the `*_angular.csv` files) —
gives **0.916 (0.001–0.991)** and **0.891 (−0.249–0.993)**. The minimum is the loudest difference:
0.173 vs 0.001.

---

### 🔴 4. "Median GC3 crossed the platypus CDS level (66.6%) somewhere between α = 1 and α = 2"

**Claim (lines 670–671).**

```bash
uv run python -c "import pandas as pd; d=pd.read_csv('figure_data/exp3_generation_composition.csv'); c=['unsteered','add_a0.5','add_a1.0','add_a2.0','add_a3.0','add_a4.0']; print(d[d.condition.isin(c)].groupby('condition')[['gc','gc3']].agg(['mean','median']).reindex(c).round(2))"
uv run python -c "import pandas as pd; print(pd.read_csv('figure_data/exp3_generation_reference_windows.csv')[['gc_human','gc3_human','gc_platypus','gc3_platypus']].agg(['mean','median']).round(3))"
```

Median GC3 is 61.56 unsteered and **70.26 already at α = 0.5**, against a platypus reference of
66.63. The crossing is between unsteered and α = 0.5, not between α = 1 and α = 2. (The 66.6% itself
is exactly right — it is the mean of `gc3_platypus`, which is how `gc_codon_figures.py:378` draws
the reference line.)

---

### 🔴 5. "with the exception of stratum four" — the exception is stratum three

**Claim (lines 584–585).**

```bash
uv run python - <<'PY'
import pandas as pd
d=pd.read_csv('figure_data/exp3_steering_outcomes.csv')
u=d[d.condition=='unsteered'].set_index('gene'); a=d[d.condition=='add_a1.0'].set_index('gene')
k=u.index.intersection(a.index)
z=(a.loc[k,['pct_autapomorphy_correct','aa_id_to_target']]-u.loc[k,['pct_autapomorphy_correct','aa_id_to_target']])
z['stratum']=u.loc[k,'stratum']; print(z.groupby('stratum').mean().round(3))
PY
```

Mean Δ amino-acid identity by stratum: **+0.06, −0.96, −2.33, −0.61, −5.10**. Stratum 4 is the most
negative — the strongest case for the trend, not the exception. Stratum **3** is where monotonicity
breaks. This reads like an off-by-one in the stratum index.

---

### 🔴 6. "Families that contained fewer than four protein-coding paralogs were excluded"

**Claim (lines 144–145).**

```bash
uv run python -c "import json; g=json.load(open('scripts/families_data.json'))['gene_families']; print(sorted((len(v),k) for k,v in g.items())[:4])"
uv run python -c "import json; m=json.load(open('results/2026-07-16_mammalian-orthologs-transcript_cdsmask/blocks15/betweenfam_ot_metadata.json')); print([f for f in m['fam_order'] if f in ('heme_oxygenase','nitric_oxide_synthase')]); print(len(m['fam_order']),'families in the W2 matrix')"
```

`heme_oxygenase` has **2** paralogs and `nitric_oxide_synthase` has **3**, and both are in the 48×48
between-family Wasserstein matrix and in every within-family table. No ≥4-paralog filter exists
anywhere in the publication code path; what does exist is `MIN_MEMBERS = 8` embedded loci
(`mammal_between.py:28`) and `MIN_SP = 10` species per ortholog group (`mammal_score.py:25`).

---

### 🔴 7. "centered and clipped at 100 kb" describes a retired pipeline

**Claim (lines 104–105).**

```bash
uv run python -c "
import json,glob,numpy as np
L=[]
for f in glob.glob('data/mammalian_orthologs/loci/*.json'):
    d=json.load(open(f))
    if 'locus_seq' in d: L.append(len(d['locus_seq']))
L=np.array(L); print('n',len(L),'median',int(np.median(L)),'max',L.max())
print('over 100 kb:',(L>100000).sum()); print('over 192 kb:',(L>192000).sum())"
grep -rn '100_000' deprecated/publication-obsolete-shared/files/scripts/sample_human_genes.py
sed -n '40p;56,70p' scripts/mammalian_orthologs/embed_cds_masked_mammal.py
```

**1,487 of 16,734 loci (8.9%) exceed 100 kb**, up to 1,522,218 bp. The only `EVO2_MAX_LEN = 100_000`
in the repository is in `deprecated/publication-obsolete-shared/`, i.e. the retired human-paralog
panel. What the mammalian pipeline actually does is tile in 8,000 bp windows capped at
`MAX_WINDOWS = 24` (≈192 kb); beyond that the windows are **evenly spaced rather than contiguous**,
so 712 loci (4.25%) are sampled with gaps — which also qualifies "re-stitched before aggregation".

---

### 🔴 8. "Compara flagged as high confidence (those with >25% identity)" — no such filter ran in Experiments 1–2

**Claim (lines 152–155).**

```bash
uv run python -c "import json,glob; d=json.load(open(sorted(glob.glob('data/cache/mammal_homology/*.json'))[0])); print(sorted(d[0].keys()))"
grep -n 'format=condensed\|ortholog_one2one' scripts/mammalian_orthologs/resolve_orthologs.py
```

The cached homology records carry only `{protein_id, id, species, method_link_type, taxonomy_level,
type}`. The REST call uses `format=condensed`, which returns neither `is_high_confidence` nor percent
identity, and the only filter applied is `type == "ortholog_one2one"`
(`resolve_orthologs.py:154`). Experiment 3 **is** built on a high-confidence set
(`stage0_config.json`: `n_one2one_hiconf: 11135`, via BioMart), so the sentence is right there and
wrong in the Experiments 1–2 Data section.

---

### 🔴 9. "consistently rank 6-mer > 4-mer > dinucleotide > GC match across layers"

**Claim (lines 345–346).**

```bash
uv run python - <<'PY'
import pandas as pd
d=pd.read_csv('figure_data/exp2_control_preservation.csv')
p=d[d.axis=='between_family'].pivot(index='layer',columns='condition',values='rho')
ok=(p.kmer6_shuffle>p.kmer4_shuffle)&(p.kmer4_shuffle>p.dinuc_shuffle)&(p.dinuc_shuffle>p.gc_match)
print('holds at layers:',list(ok[ok].index)); print('fails at:',list(ok[~ok].index))
print('6mer>4mer %d/32  4mer>dinuc %d/32  dinuc>gc %d/32'%((p.kmer6_shuffle>p.kmer4_shuffle).sum(),(p.kmer4_shuffle>p.dinuc_shuffle).sum(),(p.dinuc_shuffle>p.gc_match).sum()))
print(p[['kmer6_shuffle','kmer4_shuffle','dinuc_shuffle','gc_match']].mean().round(4))
PY
```

The strict ordering holds at **16 of 32 layers**. Only "GC-matched is lowest" is consistent (32/32).
The claim *is* true of the mean-over-layers ordering (0.799 / 0.795 / 0.789 / 0.763), which is
probably what was meant.

---

### 🟡 10. The Experiment 2 recode statistics live only in the manuscript

**Claim (lines 320–321):** 59.2% of codons, 22.9% of bases, 81.7% at position 3.

```bash
uv run python audit_scripts/recode_stats.py            # ~3 min; regenerates the recode and counts
uv run python -c "import pandas as pd; d=pd.read_csv('figure_data/exp2_control_identity.csv'); print(d.loc[d.condition=='synonymous_recode',['frac_bases_changed','frac_changes_at_p3','p1_identity','p2_identity','p3_identity']].T)"
```

| | paper | my recomputation (pooled) | stored table (per-sequence mean) |
|---|---|---|---|
| codons changed | 59.2% | **59.142%** (2,862,579 / 4,840,183) | *not stored anywhere* |
| bases changed | 22.9% | 22.852% (3,318,272 / 14,520,549) | 22.819% |
| of those, at p3 | 81.7% | 81.754% (2,712,818 / 3,318,272) | 81.839% |

Two of the three are right; **59.2% rounds from 59.14% only if you round up**, and the codon-change
fraction is in no artifact at all — it exists solely in the manuscript.

---

### 🟠 11. The matched-control triple 14.8% / 44.4% / 55.6% does not reproduce, and is not stored

**Claim (lines 394–396).**

```bash
uv run python audit_scripts/paired_p3_stats.py   # ~5 min
uv run python audit_scripts/p3_denoms.py         # eligibility under four candidate denominators
```

| | paper | recomputed |
|---|---|---|
| eligible sites, as a share of codons | 44.4% | **44.168%** (2,137,832 / 4,840,183) |
| base-change rate | 14.8% | 14.723% (= 44.168 / 3) |
| nonsynonymous-arm aa identity | ~55.6% | 55.728% |

Eligibility is **deterministic** given the sequence set, so this is not sampling noise. The paper's
three numbers are internally consistent with each other (44.4/3 = 14.8; 100−44.4 = 55.6), which
suggests they were derived from one assumed 44.4% rather than measured. No denominator I tried
(all codons, sense codons, sense non-stop codons, per-sequence mean) reaches 44.4%.

---

### 🟡 12. Experiment 3's real N is 398, never 400

```bash
uv run python -c "import json; c=json.load(open('results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/stage4_config.json')); print({k:c[k] for k in ('n_genes','dropped','min_sites','gen_bp','temperature','top_k','decoding','seed')})"
uv run python -c "import pandas as pd; print(pd.read_csv('figure_data/exp3_steering_outcomes.csv').gene.nunique(),'genes'); print(pd.read_csv('figure_data/exp3_site_directionality_summary.csv').n_genes_total.unique())"
```

Two genes (`ENSG00000183036`, `ENSG00000138495`) were dropped for having fewer than 20 diagnostic
sites. Every steering result number is over 398 genes; the manuscript only ever says 400. The same
command also shows **`top_k: 4`** — the manuscript says only "decoding at temperature 0.7" and omits
the top-4 truncation.

---

### 🟠 13. Figure 12's missense arm is being regenerated right now

```bash
ls data/cache/mammal_embed/ | grep paired_p3
ls data/cache/mammal_embed/transcript_cdsmask_paired_p3_missense | wc -l          # partial
ls data/cache/mammal_embed/transcript_cdsmask_paired_p3_missense.uniform_pre_2026-09-01 | wc -l   # 11288
ps aux | grep -c '[e]mbed_cds_masked_mammal.*paired_p3_missense'
```

At the time of this audit an `embed_cds_masked_mammal.py --control paired_p3_missense` job was
running (started 2026-09-01 15:23, i.e. *after* the manuscript PDF was written at 15:41 — the two
overlap). The complete pre-existing cache has been renamed `…uniform_pre_2026-09-01` and the live
directory held only 181 of 11,288 embeddings. **Every `paired_p3_missense` number now in
`figure_data/exp2_control_preservation.csv` and in `pub/figures/fig12_*` is from the superseded
draw and will change when this finishes.**

---

### 🟡 14. Blocks 30 and 31 of Evo2 are byte-identical

```bash
uv run python -c "
import numpy as np, glob
fs=sorted(glob.glob('data/cache/mammal_embed/transcript_cdsmask/*__*.npy'))[:200]
print(sum(np.array_equal(np.load(f)[30], np.load(f)[31]) for f in fs),'/',len(fs),'loci have blocks30 == blocks31')"
uv run python -c "import pandas as pd; d=pd.read_csv('figure_data/exp1_between_family_by_layer.csv'); print(d[d.layer>=30].pivot(index='layer',columns='baseline',values='rho'))"
```

200/200 sampled loci. Every "all 32 layers" figure therefore plots 31 distinct layers with the last
point duplicated — visible as identical rows for layers 30 and 31 in
`exp1_between_family_by_layer.csv`, `exp2_control_preservation.csv` and
`exp3_direction_layer_stats.csv`. Worth a sentence in the methods.

---

### 🟡 15. Ten of the 48 families share a Pfam accession, so 32 family pairs have JSD ≡ 0

```bash
uv run python -c "
import sys, collections; sys.path.insert(0,'scripts')
from gene_families import PFAM_ACCESSIONS
c=collections.Counter(v.split('.')[0] for v in PFAM_ACCESSIONS.values())
print('distinct accessions:',len(c),'for',len(PFAM_ACCESSIONS),'families')
for acc,n in c.items():
    if n>1: print(' ',acc,'x',n,'->',[k for k,v in PFAM_ACCESSIONS.items() if v.split('.')[0]==acc])"
uv run python audit_scripts/recompute_between.py   # prints the tied-pair sensitivity
```

Eight GPCR families all map to `PF00001`, three GTPase families to `PF00071`, two ADAM families to
`PF01421`. Their Pfam-JSD distances are exactly 0 by construction — 32 of the 1,128 family pairs.
**The good news:** removing those pairs lowers the mid-stack Pfam ρ by only about 0.02
(0.5698 → 0.5470 at layer 12), so Experiment 1's headline survives. Still worth stating, because a
reader will notice that eight of the 48 "families" share one domain model.

### 🟡 16. The Pfam-JSD baseline matrix is not persisted anywhere

`pfam_hmm_jsd.py:25` fetches HMMs live from InterPro at score time, and no
`pfam_jsd_distances.csv` exists under any `mammalian-orthologs*` run directory. Experiment 1's
central baseline therefore cannot be reproduced from tracked artifacts without a network fetch (or
the local `data/tools/pfam/Pfam-A.hmm`, which is what `audit_scripts/recompute_between.py` uses).

---

## 2. Master table

Full machine-readable form: **`PAPER_NUMBER_AUDIT.csv`** (99 rows, one per claim, with the
`verification_command` column giving a runnable lookup for each).

Legend — ✅ verified by independent recomputation · 🟢 verified against the stored artifact ·
🟡 probably correct, one verification step missing or a minor rounding gap · 🟠 ambiguous provenance ·
🔴 disagrees with the underlying result · ⚫ unverifiable.

### 2.1 Experiment 1 — between-family geometry

| ID | Lines | Claim | Reported | Underlying | Artifact · location | L1 | L2 | Status |
|---|---|---|---|---|---|---|---|---|
| N001 | 242–244 | Pfam ρ at earliest/latest layers | ~0.1–0.2 | L0 0.20984, L1 0.26324, L2 0.26890, L28 0.16496, L29 0.11390, L30/31 0.19265 | `figure_data/exp1_between_family_by_layer.csv` · `baseline=pfam_jsd`, those layers, col `rho` | partial | ✔ 8.3e-17 | 🔴 upper bound is 0.27, not 0.2 |
| N002 | 244–245 | Pfam ρ, layers 10–26 | ~0.5–0.58 | min 0.40883 (L26), max 0.56977 (L12) | same · layers 10–26 | partial | ✔ | 🔴 only L10–15 reach 0.50; max is 0.570 |
| N003 | 245 | Pfam sits above both compositional controls, layers 10–26 | qualitative | holds L10–24; fails L25 (0.4189 < kmer 0.4261) and L26 (0.4088 < kmer 0.4477, < gc 0.4140) | same · pivot layer × baseline | partial | ✔ | 🔴 true for L10–24 |
| N004 | 242–243 | Earliest/latest layers track composition, particularly GC | qualitative | GC: L0 0.686, L1 0.572, L29 0.595, L30/31 0.604; but k-mer 0.644 > GC at L1 | same | ✔ | ✔ | 🟡 |

> **Level-2 note.** `audit_scripts/recompute_between.py` rebuilds the Pfam-JSD, 6-mer and GC baseline
> matrices from scratch and re-scores all 32 stored W2 matrices. **Max |difference| vs the published
> table: 8.3 × 10⁻¹⁷ for Pfam, 5.6 × 10⁻¹⁷ for k-mer, 8.3 × 10⁻¹⁷ for GC.** The *artifact* is exactly
> right; only the prose describing it is loose.

### 2.2 Experiment 1 — within-family geometry

| ID | Lines | Claim | Reported | Geodesic roll-up (matches) | Angular roll-up (backs figs 2–3) | Status |
|---|---|---|---|---|---|---|
| N005 | 252–253 | median k-mer × species-tree | 0.913 | **0.912723** (n=48) | 0.916040 | 🟠 |
| N006 | 253 | its minimum | 0.173 | **0.172901** | 0.000917 | 🟠 |
| N007 | 253 | its maximum | 0.988 | **0.988310** | 0.991201 | 🟠 |
| N008 | 253–254 | median k-mer × gene-tree | 0.862 | **0.862144** (n=47) | 0.890742 | 🟠 |
| N009 | 254 | its minimum | −0.256 | **−0.256113** | −0.248763 | 🟠 |
| N010 | 254 | its maximum | 0.993 | **0.993022** | 0.993401 | 🟠 |
| N011 | 252 | "computed over all 48 families" | 48 | 48 for the species-tree pair; **47** for the gene-tree pair | — | 🔴 for N008 |

Artifact: `results/layer_sweep_summaries/mammalian-orthologs-cdsmask-48fam/within_family_vs_layer_metric_pairs.csv`,
rows `metric_a='k-mer composition (CDS)'`, column `spearman_across_layers`.
Competing artifact: `…/within_family_vs_layer_graphfree_metric_pairs.csv` (identical to
`figure_data/exp1_within_family_by_layer.csv`, which is what the figures read).

> **Level-2 note.** `audit_scripts/recompute_within.py` rebuilds the per-family angular ρ for
> `globins`, `adrenoceptor` and `peroxidase` at layer 15, straight from the embedding cache, against
> both the species-tree and 6-mer baselines. **All six values reproduce the published table exactly
> (difference 0.0).** The angular artifact is correct; the manuscript is quoting the geodesic one.
>
> Also worth knowing: each per-family ρ is a mean over only 5–8 ortholog groups for those families,
> and the analysis N (611 analyzable groups over 11,288 embedded loci) never appears in the paper.
>
> **The within-group grain is confirmed at every cell, not just the ones I spot-checked.**
> `within_family_uncertainty.py:100-124` recomputes each published family value as the mean of its
> per-group ρ and hard-fails on any deviation above 1e-9. The 2026-08-31 validation run logged
> `[check] 6,112 published family values reproduced exactly from the per-group scores` —
> and 6,112 is exactly the row count of `figure_data/exp1_within_family_by_layer.csv`, i.e. all 32
> layers × 48 families × 4 baselines (less the three patristic gaps). So the aggregation step
> "per-ortholog-group Spearman, then mean within family" is verified for the whole table:
>
> ```bash
> grep -n 'published family values reproduced' .frozen/validation-2026-08-31/logs/a3_exp1_B4_within_uncertainty.log
> sed -n '100,124p' scripts/mammalian_orthologs/within_family_uncertainty.py
> uv run python audit_scripts/trace_within_group.py --family globins --layer 15
> ```

### 2.3 Experiment 2 — control construction

| ID | Lines | Claim | Reported | Underlying | Artifact | Status |
|---|---|---|---|---|---|---|
| N012 | 320–321 | recode changed this share of codons | 59.2% | **59.142%** (2,862,579 / 4,840,183) | *none — no artifact stores it* | 🟡 rounds to 59.1 |
| N013 | 321 | …altering this share of coding bases | 22.9% | 22.852% pooled (3,318,272 / 14,520,549); 22.819% as the stored per-sequence mean | `figure_data/exp2_control_identity.csv` · row `synonymous_recode`, col `frac_bases_changed` | ✅ |
| N014 | 321 | …of which this share at p3 | 81.7% | 81.754% pooled (2,712,818 / 3,318,272); 81.839% stored | same · col `frac_changes_at_p3` | ✅ |
| N022 | 394 | paired-p3 base-change rate | 14.8% | **14.723%** | *none* | 🟡 |
| N023 | 395 | paired-p3 eligible sites, share of codons | 44.4% | **44.168%** (2,137,832 / 4,840,183) — deterministic | *none* | 🟠 |
| N024 | 396 | nonsynonymous-arm aa identity | ~55.6% | **55.728%** | *none* | 🟠 |
| N090 | 313–316 | shuffle orders | k = 2, 4, 6 | `ORDINARY_CONTROLS` lists exactly `dinuc`, `kmer4`, `kmer6` | `scripts/controls/make_control_sequences.py:8-14` | ✅ |

### 2.4 Experiment 2 — control preservation

| ID | Lines | Claim | Reported | Underlying | Status |
|---|---|---|---|---|---|
| N015 | 342–343 | recode between-family ρ, layers 2–27 | 0.97–0.99 | 0.960086 (L27) – 0.997136 (L2); L8 = 0.9625 also below 0.97 | 🟡 |
| N016 | 343–344 | …in the final blocks | 0.88–0.90 | 0.883076 / 0.892622 / 0.904152 / 0.904152 | ✅ |
| N017 | 345–346 | 6-mer > 4-mer > dinuc > GC across layers | qualitative | strict ordering at **16/32** layers; dinuc>GC at 32/32; means do follow the order | 🔴 |
| N018 | 350 | k-mer-only shuffles reach | ρ = 0.7–0.85 | all layers 0.5195–0.9885; layers 10–27 0.6499–0.8259 | 🟠 no layer range given |
| N019 | 355–356 | within-family recode ρ, middle layers | ~0.6–0.72 | L10–26: 0.623247 – 0.721595 | 🟢 |
| N020 | 356–357 | within-family shuffles | 0.4–0.5 | L10–26: 0.4117–0.6477 (band holds L10–23; L24–26 are 0.58–0.65) | 🟡 |
| N021 | 355–356 | recode is still the highest | qualitative | highest at 27/32 layers among figure 4's rungs; beaten at L0, L3, L29, L30, L31 | 🟡 |
| N025 | 381–386 | protein disruption: little effect early, modest mid, much larger late | qualitative | missense − recode gap: L0–2 **+0.0103**, L3–9 +0.0029, L10–23 **−0.0261**, L24–27 −0.0307, L28–31 **−0.1603** | ✅ |

Artifact for all of these: `figure_data/exp2_control_preservation.csv`, rows `axis=between_family`
(or `within_family`, averaged over `family`), column `rho`. Generating code
`scripts/mammalian_orthologs/mammal_controls_score.py:354,373-382`; figure 4 is
`scripts/controls/plot_control_wasserstein.py --pub`.

> **Level-2 note.** `audit_scripts/recompute_controls.py` rebuilds the Wasserstein matrices for
> natural and for each control rung at layer 15, directly from the Evo2 embedding caches, and
> re-scores them:
>
> ```
> natural W2 vs stored matrix: max |diff| = 1.110e-16
> gc_match           0.726563 = 0.726563   (diff 0.00e+00)
> dinuc_shuffle      0.758585 = 0.758585   (diff 0.00e+00)
> kmer4_shuffle      0.769656 = 0.769656   (diff 0.00e+00)
> kmer6_shuffle      0.759910 = 0.759910   (diff 0.00e+00)
> synonymous_recode  0.986308 = 0.986308   (diff 0.00e+00)
> missense_subset    0.952650 = 0.952650   (diff 0.00e+00)
> paired_p3_syn      0.976905 = 0.976905   (diff 0.00e+00)
> paired_p3_missense  n/a  (published 0.894417 — cache mid-regeneration, see §4)
> ```
>
> Experiment 2's headline artifact is exact. Seven of eight rungs recompute bit-for-bit; the eighth
> is the one currently being re-embedded.

### 2.5 Experiment 3 — direction geometry (figures 6a/6b, the 103-gene paired panel)

| ID | Lines | Claim | Reported | Underlying | Status |
|---|---|---|---|---|---|
| N026 | 508–509 | LOO cosine near zero, layers 9–20 | ~0 | median −0.0826 … −0.0090 | 🟢 |
| N027 | 509 | rises from layer ~21 | ~21 | 0.0126 → 0.0705 → 0.2201 → 0.4917 → 0.5527 → 0.6811 → 0.7299 (L21→27) | 🟢 |
| N028 | 509–510 | saturates near 1.0 at layers 28–31 | near 1.0 | 0.7988 / 0.7723 / 0.9764 / 0.9764 | 🟡 generous for L28–29 |
| N029 | 513–514 | magnitude spread higher through 9–23 | qualitative | `delta_norm_cv` 2.3685 → 1.7555 | 🟢 |
| N030 | 514–515 | global minimum at layer 27 | layer 27 | `delta_norm_cv` min **0.3226** at L27 (next 0.4043 at L26) | 🟢 |
| N071 | 508 | panel size | ~100 genes | **103** | ✅ |

Artifact: `figure_data/exp3_direction_layer_stats.csv` (`panel=paired103`), columns `loo_median`,
`delta_norm_cv`. L2 recomputation not attempted — the paired-panel embeddings are not tracked.

### 2.6 Experiment 3 — steering outcomes

| ID | Lines | Claim | Reported | Legacy `pct_private_correct` | Headline `pct_autapomorphy_correct` | Status |
|---|---|---|---|---|---|---|
| N031 | 596 | stratum 0 private bp, unsteered | ~42% | **42.187** | 39.807 | 🟠 |
| N032 | 596–597 | stratum 0 at α = 4 | ~46% | **45.696** | 44.028 | 🟠 |
| N035 | 598 | stratum 4, unsteered | ~36% | **36.362** | 31.691 | 🟠 |
| N036 | 598 | stratum 4 at α = 4 | ~44% | **44.035** | 41.640 | 🟠 |

| ID | Lines | Claim | Reported | Underlying (`aa_id_to_target`, mean over genes) | Status |
|---|---|---|---|---|---|
| N033 | 597 | stratum 0 aa identity, unsteered | ~17% | 17.169 | ✅ |
| N034 | 597 | stratum 0 at α = 4 | ~15% | 14.771 | ✅ |
| N037 | 598–599 | stratum 4, unsteered | ~36% | 35.664 | ✅ |
| N038 | 599 | stratum 4 at α = 4 | ~19% | 18.591 | ✅ |

**The amino-acid numbers do not have the metric problem.** `aa_id_to_target` is **byte-identical in
both score tables** — legacy `stage4_scores.csv` and current `stage4_scores_nt.csv` agree on all
48,484 (gene, condition, sample) rows, max |diff| = 0.000e+00 — and it is carried through into
`figure_data/exp3_steering_outcomes.csv`. So all four aa numbers are checkable from the tracked
table, and they reproduce exactly. Only the private-bp metric was redefined.

```bash
# the four numbers, plus every cell of the dose x stratum grid
uv run python - <<'PY'
import pandas as pd
d = pd.read_csv('figure_data/exp3_steering_outcomes.csv')
c = ['unsteered','add_a0.5','add_a1.0','add_a2.0','add_a3.0','add_a4.0']
s = d[d.condition.isin(c)]
t = s.pivot_table(index='stratum', columns='condition', values='aa_id_to_target', aggfunc='mean')[c]
t.loc['all'] = s.groupby('condition')['aa_id_to_target'].mean().reindex(c)
print(t.round(2))
PY
# and the same table in the legacy file, to confirm the column did not change
uv run python -c "
import pandas as pd, numpy as np
R='results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/'
k=['gene','condition','sample']
m=pd.read_csv(R+'stage4_scores.csv')[k+['aa_id_to_target']].merge(
  pd.read_csv(R+'stage4_scores_nt.csv')[k+['aa_id_to_target']], on=k, suffixes=('_legacy','_nt'))
print(len(m),'rows; max |diff| =', np.abs(m.aa_id_to_target_legacy-m.aa_id_to_target_nt).max())"
```

**Amino-acid identity to the platypus protein (%), mean over genes:**

| stratum | unsteered | α = 0.5 | α = 1 | α = 2 | α = 3 | α = 4 |
|---|---|---|---|---|---|---|
| 0 (most diverged) | **17.17** | 17.94 | 17.23 | 16.16 | 15.43 | **14.77** |
| 1 | 19.88 | 19.35 | 18.93 | 17.56 | 15.82 | 14.86 |
| 2 | 22.42 | 21.36 | 20.09 | 17.57 | 15.94 | 15.72 |
| 3 | 23.43 | 23.50 | 22.81 | 19.75 | 16.65 | 14.85 |
| 4 (most conserved) | **35.66** | 35.59 | 30.57 | 26.56 | 21.59 | **18.59** |
| all 398 genes | 23.69 | 23.52 | 21.91 | 19.51 | 17.08 | 15.75 |

The bolded cells are the manuscript's ~17% → ~15% and ~36% → ~19% (lines 597–599). Both round
correctly.

**Paired Δ vs unsteered (pp), mean over the same genes** — this is the quantity figures 7 and 8
plot, and it is where the stratum-3/stratum-4 wording problem (N041) shows up:

| stratum | α = 0.5 | α = 1 | α = 2 | α = 3 | α = 4 |
|---|---|---|---|---|---|
| 0 | +0.77 | +0.06 | −1.01 | −1.74 | −2.40 |
| 1 | −0.54 | −0.96 | −2.32 | −4.06 | −5.02 |
| 2 | −1.06 | −2.33 | −4.84 | −6.48 | −6.70 |
| 3 | +0.08 | **−0.61** | −3.68 | −6.77 | −8.58 |
| 4 | −0.08 | **−5.10** | −9.10 | −14.08 | −17.07 |
| all | −0.16 | −1.78 | −4.18 | −6.61 | −7.93 |

n = 80/80/79/80/79 genes per stratum (398 total). At every α from 1 upward, stratum 4 carries the
largest protein cost and stratum 3 is the one that breaks the monotone trend — at α = 1 the break is
stark (−0.61 vs −2.33 in stratum 2). The manuscript's "with the exception of stratum four" is
therefore wrong at every dose, not just at α = 1.

| ID | Lines | Claim | Underlying | Status |
|---|---|---|---|---|
| N039 | 582–583 | improves in every stratum | Δ at α=1: +1.482 / +2.229 / +1.930 / +2.692 / +4.949 pp | ✅ |
| N040 | 583–584 | improvement grows with conservation | monotone apart from stratum 2; `results_n400.md` records ρ = +0.115 (p = 0.022), collapsing to +0.032 (p = 0.52) once the unsteered baseline is partialled out — **a caveat the manuscript omits** | ✅ (incomplete) |
| N041 | 584–585 | aa change increasingly negative, "exception of stratum four" | +0.059 / −0.956 / −2.325 / **−0.614** / −5.096 — the exception is stratum **three** | 🔴 |
| N042 | 587–588 | indel and premature stops do not shift | Δ indel % by stratum −0.050 / −0.129 / −0.183 / +0.001 / +0.354; Δ premature stops −0.017 … −0.092. No test is reported anywhere | 🟡 |

Artifacts: `figure_data/exp3_steering_outcomes.csv` for everything except the four legacy
percentages, which need
`results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/stage4_scores.csv`.

### 2.7 Experiment 3 — site directionality (figures 9a/9b)

| ID | Lines | Claim | Underlying (`site_set=private`, n = 398) | Status |
|---|---|---|---|---|
| N043 | 641–642 | C rises with α, saturating at α = 2–3 | `lvl_C` 52.966 / 53.759 / 54.727 / **54.744** / 54.375 from `base_C` 51.406 | 🟢 |
| N044 | 642–643 | C dips at α = 4 | `d_C` 3.338 → 2.968 | 🟢 |
| N045 | 645–646 | L increases monotonically | `lvl_L` 71.144 / 73.765 / 76.576 / 78.468 / 79.430 from `base_L` 70.255 | 🟢 |
| N046 | 646 | L moves over a substantially wider range | `d_L` 0.890 → 9.175 pp vs `d_C` 1.560 → 3.338 pp | 🟢 — but the manuscript still contains the placeholder **"[give the axis ranges here]"** |
| N047 | 657–661 | at non-private sites α does not strengthen the platypus signal | `shared_not_private` `d_C`: +0.905 / +0.838 / +0.085 / −0.374 / −0.865 | 🟢 |

Artifact: `figure_data/exp3_site_directionality_summary.csv`, columns `base_L`/`lvl_L`/`d_L` and
`base_C`/`lvl_C`/`d_C`. L2 recomputation not attempted — the generated sequences are not tracked.

### 2.8 Experiment 3 — generated composition (figure 10)

| ID | Lines | Claim | Reported | Underlying (mean over 4,781 rows) | Status |
|---|---|---|---|---|---|
| N048/N049 | 668–669 | GC3, unsteered → α = 4 | 61% → 91% | 60.73 → 90.56 (median 61.57 → 91.02) | ✅ |
| N050/N051 | 669–670 | GC1 | 58% → 83% | 58.24 → 82.79 | ✅ |
| N052/N053 | 670 | GC2 | 48% → 73% | 47.91 → 73.23 | ✅ |
| N054 | 670 | platypus CDS GC3 reference | 66.6% | **66.6319** = mean of `gc3_platypus` over 398 genes (median 67.186) | ✅ |
| N055 | 670–671 | median GC3 crosses it between α = 1 and α = 2 | — | crosses between **unsteered (61.56) and α = 0.5 (70.26)** | 🔴 |
| N056 | 671–673 | overall GC monotone, 80-something % at α = 4 | — | 55.62 → 60.67 → 65.50 → 74.23 → 78.98 → **82.21** (median 83.06) | ✅ |

Artifacts: `figure_data/exp3_codon_substitutions.csv` (`gc1`,`gc2`,`gc3`),
`figure_data/exp3_generation_composition.csv` (`gc`,`gc3`),
`figure_data/exp3_generation_reference_windows.csv` (`gc3_platypus`).

### 2.9 Dataset counts

| ID | Lines | Claim | Reported | Independent count | Status |
|---|---|---|---|---|---|
| N057 | 141–142 | HGNC gene groups | 48 | 48 (`families_data.json`; 48×48 W2 matrix; 48 families in every within-family table) | ✅ |
| N058 | 142, 146 | human paralogs | 1,144 | 1,144 entries but **1,143 distinct symbols** — `STEAP1` is listed in both `steap_metalloreductase` and `serine_protease` | ✅ (with a duplicate) |
| N059 | 144–145 | families with <4 paralogs excluded | threshold 4 | **violated**: `heme_oxygenase` (2), `nitric_oxide_synthase` (3), both retained | 🔴 |
| N060 | 148 | mammalian panel | 24 | 24 | ✅ |
| N061 | 148–151 | clade breakdown | 6/4/10 + 4 | primate 6, glires 4, laurasiatheria 10, afrotheria/xenarthra/marsupial/monotreme 1 each | ✅ |
| N062 | 151–152 | orthologs from the remaining species | 23 | 23 | ✅ |
| N063 | 152–155 | Compara high-confidence, >25% identity | filter | **no such filter in the Exp 1–2 path** | 🔴 |
| N064 | 157–158 | groups with <10 species excluded | 10 | `MIN_SP = 10`; **611** analyzable groups survive (of 1,034 embedded) | ✅ |
| N065 | 210–213 | MamPhy posterior trees | 100 | 100 tree blocks; `sample_size: 100` | ✅ |
| N066 | 205–208 | Pfam profile dimensionality | 20 | 20 | ✅ (see §1.15 on shared accessions) |
| N067 | 452–454 | five identity strata | 50–64/65–73/73–80/81–88/88–100 | 50.17–64.50 / 64.71–72.83 / 72.91–80.47 / 80.67–87.96 / 88.10–100.00 | ✅ |
| N068 | 454 | pairs per stratum | 80 | 80 / 80 / 80 / 80 / 80, from 558 examined | ✅ |
| N069 | 481–482 | paired human–platypus set | n = 400 | 400 sampled; **398 analyzed** everywhere downstream | 🟡 |
| N070 | 456–457 | minimum protein identity | ≥50% | min 50.1736 | ✅ |
| N072 | 558–559, 609 | outgroup for "private" | the other 24 | at most 22; median voting depth 19 | 🔴 |
| N073 | 239 | "all 32 layers of Evo2" | 32 | 32 rows, but blocks 30 and 31 are byte-identical (200/200 loci) | 🟡 |

Also derivable and never stated: 1,144 panel paralogs → 1,062 human genes with resolved orthologs →
1,061 manifest groups → 1,034 embedded groups → **11,288 embedded loci** → **611 analyzable
(≥10-species) groups**.

---

## 3. `METHOD_PARAMETER_AUDIT`

Numeric methodological parameters, separated from empirical results.

| ID | Lines | Parameter | Paper value | Code/config value | File:line | Status |
|---|---|---|---|---|---|---|
| N074 | 103 | base model | Evo2-7B | `MODEL_NAME = "evo2_7b"` | `scripts/evo2/evo2_embedding.py:8` | ✅ |
| N075 | 103–104 | hardware | one NVIDIA A10G | AWS g5.8xlarge, one A10G | `README.md` Requirements | 🟢 |
| N076 | 104–105 | input clip | **100 kb** | **no clip**; 8.9% of loci exceed 100 kb (max 1.52 Mb). The only `100_000` is in `deprecated/publication-obsolete-shared/files/scripts/sample_human_genes.py:36` | `scripts/mammalian_orthologs/embed_cds_masked_mammal.py` | 🔴 |
| N077 | 106 | tiling threshold | >8,000 bp | `EVO2_WINDOW = 8000` | `embed_cds_masked_mammal.py:40` | ✅ |
| — | 106 | *(unstated)* window cap | — | `MAX_WINDOWS = 24` ⇒ ≈192 kb, and beyond that the windows are **evenly spaced, not contiguous** (712 loci, 4.25%) | `embed_cds_masked_mammal.py:57,61-69` | ⚠️ omitted from the paper |
| N078 | 107–109 | pooling | second half of token positions | `half = len(allpos)//2; allpos[half:].mean(0)` | `embed_cds_masked_mammal.py:95-98` | ✅ |
| N079 | 109–110 | normalization | L2 | `{l2_normalize: true, standardize: false}` | `…/blocks15/betweenfam_ot_metadata.json` | ✅ |
| N080 | 164–183 | between-family metric | 2-Wasserstein over angular distance | `cost = theta**2`, `theta = arccos(clip(cos,-1,1))/pi`, exact `ot.emd2`, uniform within-family weights, POT 0.9.7 | `scripts/baselines/ot_between_family.py:49-80` | ✅ formula matches exactly |
| N081 | 220–222 | k-mer order | 6 | `k = 6` in both the within- and between-family paths | `kmer_sequence_divergence.py:43`, `mammal_score.py:93`, `mammal_between.py:47` | ✅ |
| N082 | 215–217 | aligner / tree builder | MAFFT v7, FastTree v2 | mafft **v7.505**, FastTree **2.1.11** | `protein_alignment_patristic.py:109,126` | ✅ (versions not pinned in-repo) |
| N083 | 146, 449 | Ensembl Compara release | 116 | `REL = "release-116"`; `ensembl_release: 116` | `download_bulk.py:16`; `stage0_config.json` | ✅ |
| N084 | 450 | MMseqs2 release | 15 | `15-6f452+ds-2` | `stage0_pool.py:113-125` | ✅ |
| — | 450 | *(unstated)* clustering params | — | `mmseqs_min_seq_id: 0.3`, `mmseqs_cov: 0.5` → 6,961 blocks, 5,027 singletons | `stage0_config.json` | ⚠️ omitted |
| N085 | 455, 459, 537 | prompt length | 90 bp | all 400 prefixes are exactly 90 chars | `figure_data/exp3_panel.csv` | ✅ |
| N086 | 538–539 | prompt window | first 30-codon indel-free window | `max_start_codon: 30`; 108/558 candidates rejected as `window_starts_too_late` | `stage1/stage1_summary.json` | ✅ |
| N087 | 539 | generation length | the next 1,000 bp | `gen_bp: 1000` (a cap; observed 126–1,000 as generation stops at the CDS end) | `stage4_config.json` | ✅ |
| N088 | 539 | decoding | temperature 0.7 | `temperature: 0.7`, **`top_k: 4`**, `"decoding": "top-4 truncation then temperature 0.7"` | `stage4_config.json` | 🟡 the top-4 truncation is omitted from the paper |
| N089 | 541 | steering strengths | α ∈ {0.5,1,2,3,4} | all five `add_a*` conditions present over 398 genes | `figure_data/exp3_steering_outcomes.csv` | ✅ |
| N090 | 313–316 | shuffle orders | k = 2, 4, 6 | `ORDINARY_CONTROLS` | `make_control_sequences.py:8-14` | ✅ |
| N091 | 561 | premature stops | per 100 codons | column `premature_stop`, 0.5609 unsteered → 0.3958 at α = 4 | `figure_data/exp3_steering_outcomes.csv` | 🟢 |
| N092 | 462 | prior work prompt | 1,000–3,000 bp, ≥11× ours | 1000/90 = 11.1 — arithmetic checks out; the source value is external | — | external |

---

## 4. Stale results, provenance ambiguity, and figure-vs-text divergence

### 4.1 An embedding job is running *right now* that will change figure 12

```
$ ps aux | grep embed_cds_masked_mammal
ubuntu 724583 ... scripts/mammalian_orthologs/embed_cds_masked_mammal.py --families <all 48> --control paired_p3_missense   # started 15:23
$ ls data/cache/mammal_embed/ | grep paired_p3
transcript_cdsmask_paired_p3_missense                          # 181 of 11,288 .npy as of this audit
transcript_cdsmask_paired_p3_missense.uniform_pre_2026-09-01   # 11,288 .npy — the draw every current number came from
transcript_cdsmask_paired_p3_syn                               # 11,288 .npy
```

The manuscript PDF is timestamped 15:41; the re-embed started at 15:23. Every `paired_p3_missense`
value in `figure_data/exp2_control_preservation.csv` (written 15:40) and in
`pub/figures/fig12_paired_p3_protein_vs_nucleotide.*` (written 03:40) comes from the
`…uniform_pre_2026-09-01` draw and **will move** when this finishes. The manuscript itself says
"we're also running a stricter matched control", so this is expected — but N022–N024 are stated as
settled facts in a paragraph about a control that is mid-flight.

### 4.2 Text and figures built from different distance metrics (Experiment 1)

`results/layer_sweep_summaries/mammalian-orthologs-cdsmask-48fam/` holds **four** roll-ups over the
same per-layer run: `SOURCE.md` (v1, geodesic), `SOURCE_v2.md`, `SOURCE_graphfree.md` and
`SOURCE_wasserstein_angular.md`. `pub/figures/MANIFEST.md` records that figures 1–3 come from the
`wasserstein_angular` set, and `build_figure_data.py:39-49` confirms `figure_data/` reads the
`*_angular.csv` files. The manuscript's within-family metric-pair sentence quotes the **geodesic**
set (N005–N010). The same folder still contains the geodesic figures, so it is easy to read the
wrong one.

### 4.3 `(OLD)` figures still sitting in `pub/figures/`

```
fig01_between_family_rho_by_layer (OLD).pdf / .png     2026-08-30 23:35
fig12_paired_p3_protein_vs_nucleotide (OLD).pdf / .png 2026-08-30 23:36
```

`MANIFEST.md` documents these as the former geodesic-derived versions, kept deliberately. That is
fine as long as nothing downstream picks them up by filename glob — but they are the exact kind of
file that ends up in a submission by accident.

### 4.4 `stage4_config.json` in the block-27 arm describes a block-24 run

```bash
uv run python -c "import json; print(json.load(open('results/2026-08-08_platypus-strat-400/stage4_cds_mean_blocks27/stage4_config.json')))"
# -> {"layers": ["blocks.24"], "layer_label": "blocks.24", "arms": ["gc_removed"], "alphas": [1.0], ...}
```

The directory is named `…_blocks27` and holds the primary block-27 generations, but its config file
was overwritten by a later layer-24 `gc_removed` pass. `REPRODUCING.md` flags the same hazard
("an existing directory may also contribute retained `_L24` conditions"). The per-row `layer` column
in the score tables is the reliable field; the config file is not.

### 4.5 Untracked artifacts

* `pub/` — the manuscript and every published figure — is untracked.
* `figure_data/exp2_control_identity.csv` and `…_by_family.csv` (figure 13, the "controls for the
  controls" section) are untracked and were written at 15:40 today.
* `figure_data/MANIFEST.csv` is modified relative to HEAD.
* The legacy `pct_private_correct` column that four manuscript numbers depend on is in no tracked
  file.
* The Pfam-JSD baseline matrix is in no file at all (fetched live from InterPro at score time).

---

## 5. Numerical integrity verdict

| | |
|---|---|
| **Total manuscript numerical claims audited** | **99** |
| Empirical results | 53 |
| Dataset counts | 14 |
| Method/parameter values | 24 |
| External citation numbers (not repo-verifiable) | 8 |
| | |
| ✅ Fully verified — independently recomputed from upstream data | **44** |
| 🟢 Verified against the stored result artifact only | **14** |
| 🟡 Probably correct — one verification step missing or a minor rounding gap | **9** |
| 🟠 Ambiguous provenance | **13** |
| 🔴 Disagrees with the underlying result | **11** |
| ⚫ Unverifiable | **0** |
| — external, out of scope | **8** |

Per-ID rosters (the `id` column of `PAPER_NUMBER_AUDIT.csv`):

* 🔴 N001, N002, N003, N011, N017, N041, N055, N059, N063, N072, N076
* 🟠 N005–N010, N018, N023, N024, N031, N032, N035, N036
* 🟡 N004, N012, N015, N020, N021, N022, N028, N042, N073

The **artifacts are in excellent shape.** Every Experiment 1 between-family value reproduces to
8 × 10⁻¹⁷ from rebuilt baselines; every within-family value I recomputed reproduces to 0.0; seven of
eight Experiment 2 control ρ values reproduce bit-for-bit from the raw embedding caches. Essentially
all of the trouble is in the **prose layer**: stated ranges that are narrower than the data, one
superseded metric, one stale roll-up, and a handful of method sentences describing a retired
pipeline.

### Did I find any numerical statement not supported by the repository results?

Yes — **11 outright disagreements** (🔴) plus **13 claims whose provenance is ambiguous** (🟠), several of which are substantively wrong depending on which source you take as authoritative. In descending order of consequence:

1. **N031/N032/N035/N036 — the four headline steering percentages** (lines 596–599) come from
   `pct_private_correct`, which `strat_metric.py` documents as a superseded protein-alignment metric
   whose bias runs along the conservation axis. Figures 7–8 and the manuscript's own definition of
   "private" both refer to `pct_autapomorphy_correct`, which gives 39.8→44.0 and 31.7→41.6, not
   42→46 and 36→44.
2. **N072 — "the other 24 mammalian species"** (twice). The panel is 24 including human and platypus;
   the outgroup is at most 22, median 19.
3. **N059 — "families with fewer than four paralogs were excluded".** Two such families
   (`heme_oxygenase` n=2, `nitric_oxide_synthase` n=3) are in the panel and in every analysis.
4. **N063 — the ">25% identity" Compara high-confidence filter** was not applied in Experiments 1–2;
   the condensed REST format does not even return the field.
5. **N076 — "centered and clipped at 100 kb".** No clip; 8.9% of loci exceed 100 kb, and the real
   behaviour is a 24-window (≈192 kb) cap with *non-contiguous* sampling beyond it.
6. **N041 — "with the exception of stratum four".** Stratum 4 is the strongest case for the trend;
   the exception is stratum 3.
7. **N055 — "median GC3 crossed 66.6% between α = 1 and α = 2".** It crosses between unsteered and
   α = 0.5.
8. **N017 — "consistently rank 6-mer > 4-mer > dinucleotide > GC across layers".** The strict order
   holds at 16 of 32 layers (though the mean-over-layers ordering does follow it).
9. **N002 — Pfam ρ "climbs to 0.5–0.58" over layers 10–26.** Actual 0.409–0.570; only layers 10–15
   exceed 0.50.
10. **N003 — "sits above both compositional controls" over layers 10–26.** Fails at layers 25 and 26.
11. **N001 — Pfam ρ "0.1–0.2" at the earliest and latest layers.** True for layers 28–31 and layer 0;
    layers 1–2 are 0.26–0.27 and layers 3–8 are 0.47–0.52.
12. **N011 — "computed over all 48 families".** The gene-tree median is over 47.
13. **N005–N010 — the six metric-pair statistics** match the geodesic roll-up exactly but not the
    angular tables the published figures are drawn from.
14. **N023/N024 — 44.4% of codons and ~55.6% aa identity.** Deterministic quantities; the measured
    values are 44.168% and 55.728%.
15. **N015 — recode ρ "0.97–0.99" over layers 2–27.** Actual floor 0.960 (layers 8 and 27).
16. **N012 — "59.2% of codons".** Recomputes to 59.142%, and is stored nowhere.
17. **N069/N087/N088 — Experiment 3's stated N and decoding.** Every result is over 398 genes, not
    400; and `top_k = 4` truncation is omitted from "decoding at temperature 0.7".

Not numerical errors, but they change how a number should be read, so I am listing them:

* **N040** — the conservation gradient in steering gain (ρ = +0.115, p = 0.022) collapses to
  +0.032 (p = 0.52) once the unsteered baseline is partialled out. `results_n400.md:314-317` records
  this; the manuscript presents the gradient without it.
* **N066** — 10 of 48 families share a Pfam accession, so 32 of 1,128 family pairs have JSD ≡ 0. The
  effect on the headline is small (ρ drops ~0.02) but the reader should be told.
* **N073** — blocks 30 and 31 are byte-identical, so "all 32 layers" is 31 distinct layers.
* **§4.1** — the paired-p3 missense arm is being re-embedded as of this audit.
* The manuscript's definition of `d_wf` (lines 189–195, "average pairwise angular distance between
  all genes in a family") does not describe what the code computes. `mammal_score.py:78-104` builds
  distances **within an ortholog group** — one gene across up to 24 species — and then averages the
  per-group ρ up to the family. That is consistent with the species-tree and gene-tree baselines, but
  the stated formula would give a paralog-level quantity that is never computed.
* Line 497–498, "this is why we sampled each *i* ∈ N from a different clade", does not describe the
  sampling: pairs were sampled one per MMseqs2 homology block, not one per clade.

### Which numbers should you personally inspect before submission?

Prioritized checklist — the commands are in §1.

- [ ] **1.** The four steering percentages (§1.1) — decide which metric the paper is reporting, then
      make the prose, the definition and figures 7–8 agree.
- [ ] **2.** "the other 24 mammalian species" → 22 at most, median 19 (§1.2). Two occurrences.
- [ ] **3.** The within-family metric-pair sentence (§1.3) — regenerate from the angular tables the
      figures use, or say explicitly that it is a geodesic result.
- [ ] **4.** The GC3 crossing point (§1.4).
- [ ] **5.** "exception of stratum four" (§1.5).
- [ ] **6.** The ≥4-paralog exclusion claim (§1.6).
- [ ] **7.** The 100 kb clip and the 24-window cap (§1.7).
- [ ] **8.** The Compara >25%-identity filter (§1.8).
- [ ] **9.** The 6-mer > 4-mer > dinucleotide ranking (§1.9).
- [ ] **10.** The Pfam ρ ranges, 0.1–0.2 and 0.5–0.58 (master table §2.1) — widen them or narrow the
      layer ranges they are attached to.
- [ ] **11.** 59.2% / 22.9% / 81.7% (§1.10) and 14.8% / 44.4% / 55.6% (§1.11) — and persist them, so
      they stop being manuscript-only numbers.
- [ ] **12.** N = 398 vs 400, and `top_k = 4` (§1.12).
- [ ] **13.** Wait for the `paired_p3_missense` re-embed, then re-derive every paired-p3 number and
      re-render figure 12 (§1.13, §4.1).
- [ ] **14.** Decide whether to say that blocks 30 and 31 are identical (§1.14).
- [ ] **15.** Decide whether to disclose the shared Pfam accessions (§1.15).
- [ ] **16.** Add the analysis sample sizes the paper never states: 11,288 embedded loci, 1,034
      embedded ortholog groups, 611 analyzable (≥10-species) groups.
- [ ] **17.** Fix the `d_wf` definition (lines 189–195) to describe within-ortholog-group distances.

---

## Appendix A — audit scripts

Six read-only recomputation scripts are in `audit_scripts/` with a README. None writes into
`results/`, `figure_data/`, `data/` or `pub/`.

| Script | Recomputes | Runtime |
|---|---|---|
| `recompute_between.py` | all 32 layers × 3 baselines of Experiment 1's between-family ρ, from the stored W2 matrices and rebuilt Pfam/6-mer/GC baselines, plus the Pfam-tie sensitivity | ~2 min |
| `recompute_within.py` | within-family ρ for 3 families × 2 baselines at layer 15, from the embedding cache | ~1 min |
| `trace_within_group.py` | prints the groups, members, distance matrices, upper-triangle vectors, per-group ρ and family mean for one family/layer, then diffs against `figure_data/` | ~1 min |
| `recompute_controls.py` | figure 4's between-family control ρ at layer 15, from the embedding caches | ~10 min, ~13 GB RAM |
| `recode_stats.py` | synonymous-recode codon / base / p3 change rates | ~3 min |
| `paired_p3_stats.py` | paired-p3 eligibility, base-change rate, arm-wise aa identity | ~5 min |
| `p3_denoms.py` | the eligibility fraction under four candidate denominators | ~3 min |

`recompute_between.py` reads Pfam profiles from the local `data/tools/pfam/Pfam-A.hmm` (release
38.2) rather than fetching from InterPro, so it needs no network.
