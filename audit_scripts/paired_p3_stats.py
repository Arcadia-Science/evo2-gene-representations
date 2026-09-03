"""AUDIT (read-only): recompute the paired-p3 matched-control statistics the manuscript quotes."""
import sys, random, zlib
from collections import defaultdict
from pathlib import Path
REPO = Path('/home/ubuntu/Development/glm-latent-mapping')
sys.path.insert(0, str(REPO/'scripts')); sys.path.insert(0, str(REPO/'scripts'/'controls'))
import control_sequence_identity as csi
from controls import make_control_sequences as mcs

nat, fam_of = csi.load_sources()
by_fam = defaultdict(list)
for g,s in nat.items(): by_fam[fam_of[g]].append(s)
usage = mcs.build_family_codon_usage(by_fam)

tot_cod=0; elig=0
tot_b=0; ch_syn=0; ch_mis=0
p3_syn=0; p3_mis=0
aa_same_syn=0; aa_same_mis=0; aa_tot=0
for k,s in nat.items():
    s=s.upper()
    rs = random.Random(zlib.crc32(f"audit:paired_p3_syn:{k}".encode()))
    rm = random.Random(zlib.crc32(f"audit:paired_p3_missense:{k}".encode()))
    a_syn = mcs.paired_p3(s, usage[fam_of[k]], rs, 'synonymous')
    a_mis = mcs.paired_p3(s, usage[fam_of[k]], rm, 'missense')
    ncod = len(s)//3
    for i in range(ncod):
        cod = s[i*3:i*3+3]
        syn, mis = mcs._p3_alternatives(cod)
        tot_cod += 1
        if syn and mis: elig += 1
        cs, cm = a_syn[i*3:i*3+3], a_mis[i*3:i*3+3]
        for j in range(3):
            tot_b += 1
            if cod[j]!=cs[j]:
                ch_syn+=1
                if j==2: p3_syn+=1
            if cod[j]!=cm[j]:
                ch_mis+=1
                if j==2: p3_mis+=1
        aa = mcs._CODON.get(cod)
        if aa is not None and aa!='*':
            aa_tot+=1
            if mcs._CODON.get(cs)==aa: aa_same_syn+=1
            if mcs._CODON.get(cm)==aa: aa_same_mis+=1
print("codons total       %d" % tot_cod)
print("eligible sites     %d = %.4f%% of codons" % (elig, 100*elig/tot_cod))
print("syn arm  bases changed %.4f%%   (of changes, at p3: %.2f%%)" % (100*ch_syn/tot_b, 100*p3_syn/max(ch_syn,1)))
print("mis arm  bases changed %.4f%%   (of changes, at p3: %.2f%%)" % (100*ch_mis/tot_b, 100*p3_mis/max(ch_mis,1)))
print("syn arm  aa identity   %.4f%%" % (100*aa_same_syn/aa_tot))
print("mis arm  aa identity   %.4f%%" % (100*aa_same_mis/aa_tot))
