"""Localisation of the slow modes of E: participation ratio, and support in vertex blocks."""
import sys
import numpy as np, scipy.sparse.linalg as sla
from common import Problem
from proto import TwoLevel, mesh_vertex_patches
from exp_keff import keff_h
p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney
pats = [x for x in mesh_vertex_patches(p) if len(x)]
pc = TwoLevel(K, pats, c)
E = sla.LinearOperator(K.shape, matvec=lambda v: v - pc.apply(K @ v), dtype=complex)
mu, V = sla.eigs(E, k=40, which="LM", tol=1e-5, ncv=120)
o = np.argsort(-abs(mu)); V = V[:, o]
kh, _ = keff_h(p)
pr = [(np.sum(abs(v) ** 2) ** 2) / (p.n * np.sum(abs(v) ** 4)) for v in V.T]
# number of vertex blocks holding 90% of the energy
nb = []
for v in V.T:
    e = np.array([np.sum(abs(v[x]) ** 2) for x in pats]); e = np.sort(e)[::-1]
    nb.append(np.searchsorted(np.cumsum(e) / e.sum(), 0.9) + 1)
print(f"{sys.argv[1]}: {len(pats)} vertex blocks")
print(f"  participation ratio of the 40 slowest modes: median {np.median(pr):.3f}, min {np.min(pr):.3f}")
print(f"  vertex blocks holding 90% of the energy: median {np.median(nb):.0f}, min {np.min(nb)}, max {np.max(nb)}")
big = kh > np.quantile(kh, 0.8)
print(f"  share of energy on the 20% largest-kh DOFs: {np.mean([np.sum(abs(v[big])**2)/np.sum(abs(v)**2) for v in V.T]):.2f}")
