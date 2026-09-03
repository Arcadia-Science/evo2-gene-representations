"""AUDIT (read-only): recompute between-family Spearman rho from the stored W2 matrices and
freshly rebuilt Pfam-JSD / 6-mer / GC baselines. Also reports the Pfam rho with the family pairs
that SHARE a Pfam accession (JSD == 0 by construction) removed."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
for p in ('scripts','scripts/baselines','scripts/mammalian_orthologs'):
    sys.path.insert(0, str(REPO/p))
from gene_families import PFAM_ACCESSIONS
import mammal_between as mb
from mammal_score import load_cds
import pyhmmer

RUN = REPO/'results/2026-07-16_mammalian-orthologs-transcript_cdsmask'
fams = list(pd.read_csv(RUN/'blocks15/betweenfam_ot_wasserstein_distances.csv', index_col=0).index)
stack, meta = mb.load_embedded('transcript_cdsmask', 'complete_manifest.csv')
meta = meta[meta.family.isin(fams)].reset_index(drop=True)
cds = load_cds()
print(f"{len(fams)} families, {len(meta)} loci")

acc_of = {f: PFAM_ACCESSIONS[f].split('.')[0] for f in fams}
by_acc = {}
with pyhmmer.plan7.HMMFile(str(REPO/'data/tools/pfam/Pfam-A.hmm')) as fh:
    for hmm in fh:
        a = str(hmm.accession).split('.')[0]
        if a in set(acc_of.values()):
            by_acc[a] = np.array(hmm.match_emissions)[1:].mean(axis=0)
print(f"Pfam profiles loaded: {len(by_acc)} distinct accessions for {len(fams)} families")
vec = {f: by_acc[acc_of[f]] for f in fams}
n=len(fams); P=np.zeros((n,n))
for i,a in enumerate(fams):
    for j in range(i+1,n):
        P[i,j]=P[j,i]=jensenshannon(vec[a],vec[fams[j]],base=2)**2
K = mb.kmer_between(meta, cds, fams)
G = mb.gc_between(meta, cds, fams)
iu = np.triu_indices(n,1)
tied = np.array([acc_of[fams[i]]==acc_of[fams[j]] for i,j in zip(*iu)])
print(f"family pairs with an identical Pfam accession (JSD == 0): {tied.sum()} of {len(tied)}")
rows=[]
for L in range(32):
    w = pd.read_csv(RUN/f'blocks{L}/betweenfam_ot_wasserstein_distances.csv', index_col=0).values[iu]
    rows.append({'layer':L,
        'pfam_jsd':spearmanr(w,P[iu]).statistic,
        'pfam_no_tied':spearmanr(w[~tied],P[iu][~tied]).statistic,
        'kmer':spearmanr(w,K[iu]).statistic,
        'gc_content':spearmanr(w,G[iu]).statistic})
R=pd.DataFrame(rows).set_index('layer')
pub=pd.read_csv(REPO/'figure_data/exp1_between_family_by_layer.csv').pivot(index='layer',columns='baseline',values='rho')
print("\nRECOMPUTED vs PUBLISHED, max |diff|:")
for c in ['pfam_jsd','kmer','gc_content']:
    print("  %-11s %.3e" % (c, np.abs(R[c]-pub[c]).max()))
print(); print(R.round(4).to_string())
print("\npfam layers 10-26: as published %.4f-%.4f ; tied pairs removed %.4f-%.4f"
      %(R.loc[10:26,'pfam_jsd'].min(),R.loc[10:26,'pfam_jsd'].max(),
        R.loc[10:26,'pfam_no_tied'].min(),R.loc[10:26,'pfam_no_tied'].max()))
