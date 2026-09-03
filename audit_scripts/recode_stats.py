"""AUDIT (read-only): recompute the synonymous-recode change statistics the manuscript quotes."""
import sys, random, zlib
from pathlib import Path
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
sys.path.insert(0, str(REPO/'scripts'))
sys.path.insert(0, str(REPO/'scripts'/'controls'))
import numpy as np
import control_sequence_identity as csi

nat, fam_of = csi.load_sources()
print(f"n seqs={len(nat)} n families={len(set(fam_of.values()))}")
A = csi.draw_control('synonymous_recode', nat, fam_of, 'drawA')

tot_cod = ch_cod = 0
tot_b = ch_b = 0
p_ch = [0,0,0]
p_tot = [0,0,0]
for k, s in nat.items():
    a = A[k]
    n = min(len(s), len(a))
    ncod = n//3
    for c in range(ncod):
        i = c*3
        src, new = s[i:i+3], a[i:i+3]
        tot_cod += 1
        if src != new: ch_cod += 1
        for j in range(3):
            p_tot[j]+=1; tot_b+=1
            if src[j]!=new[j]:
                p_ch[j]+=1; ch_b+=1
print("codons: %d changed / %d = %.4f%%" % (ch_cod, tot_cod, 100*ch_cod/tot_cod))
print("bases : %d changed / %d = %.4f%%" % (ch_b, tot_b, 100*ch_b/tot_b))
tot_changes = sum(p_ch)
print("of changed bases, at p3: %d / %d = %.4f%%" % (p_ch[2], tot_changes, 100*p_ch[2]/tot_changes))
for j in range(3):
    print("  p%d changed %.4f%% (identity %.6f)" % (j+1, 100*p_ch[j]/p_tot[j], 1-p_ch[j]/p_tot[j]))
