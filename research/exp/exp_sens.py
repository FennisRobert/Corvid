"""B5: tolerance of the (Q = injection, P, Kc = P^T K P) coarse level to errors in P.
P_eps = ideal + truncation: X entries below eps * max|X(:,j)| dropped (a sparse, *local* P);
also LOD ell-patch correctors."""
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
def P_of(Xm):
    P = np.zeros((p.n, nc), complex); P[c] = np.eye(nc); P[f] = -Xm; return P
def run(nm, Xm, galerkin=True):
    P = P_of(Xm)
    Kc = P.T @ (K @ P) if galerkin else P_of(X).T @ (K @ P_of(X))
    F = sla.lu_factor(Kc)
    pc = TwoLevel(K, pats, c, coarse_fn=lambda r: P @ sla.lu_solve(F, r[c]))
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
    err = np.linalg.norm(Xm - X) / np.linalg.norm(X)
    print(f"  {nm:28s} ||X~-X||/||X|| {err:.1e} nnz/col {np.count_nonzero(Xm)/nc:6.0f} its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
mx = np.abs(X).max(0)
for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4):
    run(f"drop < {eps:g} max, Galerkin", np.where(np.abs(X) >= eps * mx, X, 0))
run("drop < 1e-2 max, exact S", np.where(np.abs(X) >= 1e-2 * mx, X, 0), galerkin=False)
for ell in (1, 2, 3):
    Xl = np.zeros_like(X)
    for j in range(nc):
        vs = p.edges[:, p.ent[c[j]]].copy()
        for _ in range(ell - 1): vs = np.unique(np.r_[vs, G[vs].indices])
        loc = floc[np.unique(IncT[vs].indices)]; loc = loc[loc >= 0]
        Xl[loc, j] = np.linalg.solve(Kff[loc][:, loc].toarray(), Kfc[loc, j].toarray().ravel())
    run(f"LOD ell={ell}, Galerkin", Xl)
