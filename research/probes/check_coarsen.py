import sys, time, numpy as np
sys.path[:0] = [".", "tests", "research/probes"]
from emdata import load
from corvid.amge.whitney import fine_level
from corvid.amge.coarsen import coarsen, check

m = sys.argv[1]; nlev = int(sys.argv[2]) if len(sys.argv) > 2 else 3; passes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
d = load(m)
L = fine_level(d["coords"], d["tets"], d["edges"], d["free"])
for l in range(nlev):
    t0 = time.time()
    Lc, P, C = coarsen(L, target=passes)
    ck = check(L, Lc, P)
    print(f"L{l}->L{l+1}: elem {L.n_elem}->{Lc.n_elem}  face {L.n_face}->{Lc.n_face}  "
          f"edge {L.n_edge}->{Lc.n_edge} (free {L.free_edge.sum()}->{Lc.free_edge.sum()})  "
          f"vert {L.n_vert}->{Lc.n_vert}  {time.time()-t0:.1f}s")
    print("   " + "  ".join(f"{k}: {v:.1e}" for k, v in ck.items()))
    L = Lc
    if L.n_elem < 4: break
