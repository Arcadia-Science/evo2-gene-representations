"""AUDIT (read-only): recompute figure 4's between-family control-preservation rho at blocks15
straight from the Evo2 embedding caches. Reads only; writes nothing into the repo."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
for p in ('scripts','scripts/baselines','scripts/mammalian_orthologs'):
    sys.path.insert(0, str(REPO/p))
import mammal_between as mb
from ot_between_family import compute_ot_matrices

L = 15
CACHE = REPO/'data/cache/mammal_embed'
stack, meta = mb.load_embedded('transcript_cdsmask', 'complete_manifest.csv')
fams = list(pd.read_csv(REPO/'results/2026-07-16_mammalian-orthologs-transcript_cdsmask/blocks15/betweenfam_ot_wasserstein_distances.csv', index_col=0).index)
meta = meta[meta.family.isin(fams)].reset_index(drop=True)
keys = meta['key'].tolist(); fam_arr = meta['family'].to_numpy()
iu = np.triu_indices(len(fams),1)

def w2_ut(arm):
    d = CACHE/arm
    E = np.stack([np.load(d/f'{k}.npy')[L] for k in keys])
    return compute_ot_matrices(E, fam_arr, fams, alphas=()).matrices['wasserstein'][iu]

nat = w2_ut('transcript_cdsmask')
stored = pd.read_csv(REPO/'results/2026-07-16_mammalian-orthologs-transcript_cdsmask/blocks15/betweenfam_ot_wasserstein_distances.csv', index_col=0).values[iu]
print("natural W2 vs stored matrix: max |diff| = %.3e" % np.abs(nat-stored).max())

pub = pd.read_csv(REPO/'figure_data/exp2_control_preservation.csv')
pub = pub[(pub.layer==L)&(pub.axis=='between_family')].set_index('condition')['rho']
print("\n%-20s %12s %12s %10s" % ("condition","recomputed","published","diff"))
for c in ['gc_match','dinuc_shuffle','kmer4_shuffle','kmer6_shuffle','synonymous_recode',
          'missense_subset','paired_p3_syn','paired_p3_missense']:
    d = CACHE/f'transcript_cdsmask_{c}'
    n_have = len(list(d.glob('*__*.npy')))
    if n_have < len(keys):
        print("%-20s %12s %12.6f %10s  (cache has %d/%d .npy — RE-EMBED IN PROGRESS)"
              % (c,"n/a",pub.get(c,np.nan),"",n_have,len(keys))); continue
    r = spearmanr(nat, w2_ut(f'transcript_cdsmask_{c}')).statistic
    print("%-20s %12.6f %12.6f %10.2e" % (c, r, pub.get(c,np.nan), r-pub.get(c,np.nan)))
