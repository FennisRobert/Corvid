"""Column errors of localized correctors vs the exact X = K_ff^{-1} K_fc, and spectrum of K_ff near 0."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from common import Problem
from proto import direct
from exp_lod import vertex_graph, dof_vertex_incidence
p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney; f = np.setdiff1d(np.arange(p.n), c)
floc = -np.ones(p.n, int); floc[f] = np.arange(len(f))
Kff, Kfc = K[f][:, f].tocsc(), K[f][:, c].tocsc()
G = vertex_graph(p); IncT = dof_vertex_incidence(p).T.tocsr()
lu = direct(Kff)
# smallest |eig| of K_ff relative to its diagonal scale
d = np.abs(Kff.diagonal()).mean()
ev = sla.eigs(Kff, k=6, sigma=0, return_eigenvectors=False)
print(f"K_ff: smallest |lambda| / mean|diag| = {np.sort(abs(ev))[:3] / d}")
for ell in (1, 2, 3):
    errs, mins = [], []
    for j in np.random.default_rng(2).choice(len(c), 40, replace=False):
        x = lu.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex)))
        vs = p.edges[:, p.ent[c[j]]].copy()
        for _ in range(ell - 1): vs = np.unique(np.r_[vs, G[vs].indices])
        loc = floc[np.unique(IncT[vs].indices)]; loc = loc[loc >= 0]
        A = Kff[loc][:, loc].toarray()
        xs = np.zeros_like(x); xs[loc] = np.linalg.solve(A, Kfc[loc, j].toarray().ravel())
        errs.append(np.linalg.norm(xs - x) / np.linalg.norm(x))
        mins.append(np.min(abs(np.linalg.eigvals(A))) / d)
    errs = np.array(errs)
    print(f"ell={ell}: rel. column error median {np.median(errs):.2e} max {errs.max():.2e}; min local |eig|/diag median {np.median(mins):.2e} min {np.min(mins):.2e}")
