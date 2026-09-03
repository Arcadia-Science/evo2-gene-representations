"""AUDIT: eligibility fraction under several plausible denominators."""
import sys
from pathlib import Path
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
sys.path.insert(0, str(REPO/'scripts')); sys.path.insert(0, str(REPO/'scripts'/'controls'))
import control_sequence_identity as csi
from controls import make_control_sequences as mcs
nat, fam_of = csi.load_sources()
tot=0; sense=0; elig=0; nonstop=0
per_seq=[]
for k,s in nat.items():
    s=s.upper(); n=len(s)//3; e=0; se=0
    for i in range(n):
        cod=s[i*3:i*3+3]; tot+=1
        aa=mcs._CODON.get(cod)
        if aa is not None: sense+=1; se+=1
        if aa is not None and aa!='*': nonstop+=1
        syn,mis=mcs._p3_alternatives(cod)
        if syn and mis: elig+=1; e+=1
    per_seq.append(e/max(n,1))
import numpy as np
print("all codons          %d -> elig %.4f%%"%(tot,100*elig/tot))
print("sense codons        %d -> elig %.4f%%"%(sense,100*elig/sense))
print("sense non-stop      %d -> elig %.4f%%"%(nonstop,100*elig/nonstop))
print("mean per-sequence eligibility: %.4f%%  (median %.4f%%)"%(100*np.mean(per_seq),100*np.median(per_seq)))
