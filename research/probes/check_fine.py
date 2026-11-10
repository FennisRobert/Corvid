import sys, numpy as np
sys.path.insert(0, "."); sys.path[:0] = ["tests", "research/probes"]
from emdata import load
from corvid.amge.whitney import fine_level

for m in sys.argv[1:]:
    d = load(m)
    L = fine_level(d["coords"], d["tets"], d["edges"], d["free"])
    nE = L.n_edge
    print(f"{m}: V={L.n_vert} E={nE} F={L.n_face} T={L.n_elem}")
    print("  |D1 D0| =", abs(L.D1 @ L.D0).max(), " |D2 D1| =", abs(L.D2 @ L.D1).max())
    CC = (L.D1.T @ L.assemble_M2() @ L.D1).toarray()
    M1 = L.assemble_M1().toarray()
    E = d["E"][:nE, :nE].toarray(); B = d["B"][:nE, :nE].real.toarray()
    fr = d["free"]
    # EMerge E = curl curl / mu, B = eps mass (+ folded boundary terms on boundary rows)
    rE = E.real[np.ix_(fr, fr)]; rB = B[np.ix_(fr, fr)]
    print("  curl-curl rel err:", np.linalg.norm(rE - CC[np.ix_(fr, fr)]) / np.linalg.norm(rE))
    print("  mass rel err     :", np.linalg.norm(rB - M1[np.ix_(fr, fr)]) / np.linalg.norm(rB))
    r = np.abs(rE - CC[np.ix_(fr, fr)]).max(1) / abs(rE).max()
    q = np.abs(rB - M1[np.ix_(fr, fr)]).max(1) / abs(rB).max()
    print("  rows off >1e-8: curl", (r > 1e-8).sum(), " mass", (q > 1e-8).sum(), "of", fr.sum())
