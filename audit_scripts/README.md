# Audit scripts

Read-only recomputation scripts written for `PAPER_NUMBER_AUDIT.md`. None of them writes into
`results/`, `figure_data/`, `data/` or `pub/`; they load artifacts and print numbers.

Run from the repository root:

| Script | Recomputes | Runtime |
|---|---|---|
| `recode_stats.py` | synonymous-recode codon/base/p3 change rates (N012-N014) | ~3 min, CPU |
| `paired_p3_stats.py` | paired-p3 eligibility, base-change rate, aa identity (N022-N024) | ~5 min, CPU |
| `p3_denoms.py` | the same eligibility fraction under four candidate denominators | ~3 min, CPU |
| `recompute_between.py` | every Experiment 1 between-family rho, from the stored W2 matrices and rebuilt baselines, plus a Pfam-tie sensitivity (N001-N004) | ~2 min, CPU |
| `recompute_within.py` | within-family rho for three families at layer 15, from the embedding cache | ~1 min, CPU |
| `recompute_controls.py` | figure 4's between-family control-preservation rho at layer 15, from the embedding caches (N015, N016, N025) | ~10 min, CPU, ~13 GB RAM |

`recompute_between.py` reads Pfam profiles from the local `data/tools/pfam/Pfam-A.hmm` instead of
fetching them from InterPro, so it does not need the network.
