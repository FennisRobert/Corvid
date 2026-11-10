"""B3: which ingredient must be exact -- the prolongation P or the coarse operator Kc?
Coarse correction e = P Kc^{-1} P^T r with P in {injection, LOD ell=1, exact ideal}
and Kc in {K_cc, LOD ell=1 Galerkin, exact Schur S}."""
import sys
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct
from exp_lod import vertex_graph, dof_vertex_incidence

p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney; f = np.setdiff1d(np.arange(p.n), c); nc = len(c)
floc = -np.ones(p.n, int); floc[f] = np.arange(len(f))
Kff, Kfc = K[f][:, f].tocsc(), K[f][:, c].tocsc()
pats = [x for x in mesh_vertex_patches(p) if len(x)]
lu = direct(Kff)
X = np.column_stack([lu.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex))) for j in range(nc)])
G = vertex_graph(p); IncT = dof_vertex_incidence(p).T.tocsr()
X1 = np.zeros_like(X)
for j in range(nc):
    loc = floc[np.unique(IncT[p.edges[:, p.ent[c[j]]]].indices)]; loc = loc[loc >= 0]
    X1[loc, j] = np.linalg.solve(Kff[loc][:, loc].toarray(), Kfc[loc, j].toarray().ravel())
def P_of(Xm):
    P = np.zeros((p.n, nc), complex); P[c] = np.eye(nc); P[f] = -Xm; return P
Ps = {"injection": P_of(np.zeros_like(X)), "LOD1": P_of(X1), "ideal": P_of(X)}
Kcs = {"K_cc": K[c][:, c].toarray(), "LOD1 Galerkin": Ps["LOD1"].T @ (K @ Ps["LOD1"]), "exact S": Ps["ideal"].T @ (K @ Ps["ideal"])}
print(f"{sys.argv[1]}: n={p.n}; ||Kc - S||/||S||:", {k: f"{np.linalg.norm(v - Kcs['exact S']) / np.linalg.norm(Kcs['exact S']):.3f}" for k, v in Kcs.items()})
for pn, P in Ps.items():
    for kn, Kc in Kcs.items():
        F = sla.lu_factor(Kc)
        pc = TwoLevel(K, pats, c, coarse_fn=lambda r, P=P, F=F: P @ sla.lu_solve(F, P.T @ r))
        x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
        print(f"  P={pn:10s} Kc={kn:14s} its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
