"""Where do the slow error modes of corvid 0.2.1 live on the model4 horn?
Shares of |v|^2 for the dominant eigenvectors of E = I - M K: feed waveguide (x < 0), horn interior
(inside the flare), next to PEC (DOFs touching a vertex that carries an eliminated DOF), port / ABC rows
(complex entries, not PML), PML rows, Whitney vs complement; mean local kh."""
import numpy as np, scipy.sparse.linalg as sla
import corvid
from corvid import _kernels
from common import Problem

p = Problem("model4", 1); K = p.K.tocsr()
s = corvid.Solver(p.top)
m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(K)
coarse = np.flatnonzero(m).astype(np.int64)
pc = _kernels.TwoLevelSolver(p.kernels_csr(), p.top.block_ptr, p.top.block_idx, coarse, order=1)
E = sla.LinearOperator(K.shape, matvec=lambda v: v - pc.apply(np.ascontiguousarray(K @ v, dtype=complex)), dtype=complex)
mu, V = sla.eigs(E, k=24, which="LM", tol=1e-4, ncv=70)
o = np.argsort(-abs(mu)); mu, V = mu[o], V[:, o]
print("largest |mu|:", np.round(abs(mu[:12]), 3))

X = p.nodes; x = p.loc
# vertices that carry an eliminated (PEC) DOF
ne, nt = p.edges.shape[1], p.tris.shape[1]
allg = np.arange(2 * ne + 2 * nt)
elim = np.setdiff1d(allg, p.sid)
kind_g = np.where(elim < ne, 0, np.where(elim < 2 * ne, 1, 2))
ent_g = np.where(elim < ne, elim, np.where(elim < 2 * ne, elim - ne, np.where(elim < 2 * ne + nt, elim - 2 * ne, elim - 2 * ne - nt)))
pec_v = np.zeros(X.shape[1], bool)
pec_v[p.edges[:, ent_g[kind_g < 2]].ravel()] = True
pec_v[p.tris[:, ent_g[kind_g == 2]].ravel()] = True
en = p.edges.T[np.minimum(p.ent, ne - 1)]; fn = p.tris.T[np.minimum(p.ent, nt - 1)]
dv = np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)
near_pec = pec_v[dv].any(1)
rows = np.repeat(np.arange(p.n), np.diff(K.indptr))
cplx = np.bincount(rows, (K.data.imag != 0).astype(float), p.n) > 0
port_abc = cplx & ~p.pml
h = np.linalg.norm(X[:, p.edges[0]] - X[:, p.edges[1]], axis=0)
kh = p.k0 * np.where(p.kind < 2, h[np.minimum(p.ent, ne - 1)], np.median(h))
regions = {"feed (x<0)": x[0] < 0, "horn region (0<x<19mm)": (x[0] >= 0) & (x[0] < 0.019), "beyond (x>19mm)": x[0] >= 0.019,
           "next to PEC": near_pec, "port/ABC rows": port_abc, "PML rows": p.pml, "exact level": m, "Whitney": p.kind == 0}
print("share of all DOFs:  " + "  ".join(f"{k} {v.mean():.0%}" for k, v in regions.items()))
w = abs(V[:, :12]) ** 2; w /= w.sum(0)
for i in range(12):
    print(f" |mu| {abs(mu[i]):.3f}: " + "  ".join(f"{k} {w[r, i].sum():.0%}" for k, r in regions.items()) + f"  kh {np.sum(w[:, i] * kh):.2f}")
