"""AUDIT (read-only): recompute the within-family (angular vs species-tree and vs 6-mer) rho for
three families at one layer, straight from the embedding cache. Writes nothing into the repo."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
for p in ('scripts','scripts/baselines','scripts/mammalian_orthologs'):
    sys.path.insert(0, str(REPO/p))
import mammal_score as ms
from kmer_sequence_divergence import kmer_distance_matrix

L = 15
FAMS = ['globins','adrenoceptor','peroxidase']
MIN_SP = ms.MIN_SP
stack, meta = ms.load_embedded('transcript_cdsmask')
pat = pd.read_csv(REPO/'data/mammalian_orthologs/tree/species_patristic.csv', index_col=0)
cds = ms.load_cds()
kpos = {k:i for i,k in enumerate(meta['key'])}
out = []
for fam in FAMS:
    sub = meta[meta.family==fam]
    per_group_st, per_group_km = [], []
    for group, g in sub.groupby('group'):
        members = g['key'].tolist(); species = g['species'].tolist()
        if len(members) < MIN_SP: continue
        V = stack[L][[kpos[m] for m in members]]
        U = V/np.clip(np.linalg.norm(V,axis=1,keepdims=True),1e-12,None)
        G = np.arccos(np.clip(U@U.T,-1.0,1.0))/np.pi
        ST = pat.loc[species,species].values
        KM = kmer_distance_matrix([cds[m] for m in members], k=6)
        iu = np.triu_indices(len(members),1)
        for name,B,acc in (('speciestree',ST,per_group_st),('kmer',KM,per_group_km)):
            a,b = G[iu], B[iu]
            ok = np.isfinite(a)&np.isfinite(b)
            if ok.sum()>=6 and np.ptp(a[ok])>0 and np.ptp(b[ok])>0:
                acc.append(spearmanr(a[ok],b[ok]).statistic)
    out.append({'family':fam,'n_groups':len(per_group_st),
                'speciestree_recomputed':np.mean(per_group_st),
                'kmer_recomputed':np.mean(per_group_km)})
R = pd.DataFrame(out).set_index('family')
pub = pd.read_csv(REPO/'figure_data/exp1_within_family_by_layer.csv')
pub = pub[(pub.layer==L)&(pub.family.isin(FAMS))].pivot(index='family',columns='metric',values='rho')
J = R.join(pub)
J['diff_speciestree'] = J.speciestree_recomputed - J['species tree (mammal)']
J['diff_kmer'] = J.kmer_recomputed - J['k-mer composition (CDS, k=6)']
pd.set_option('display.width',250)
print(f"layer {L}, MIN_SP={MIN_SP}")
print(J.round(6).to_string())
