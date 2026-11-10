"""B4: separate restriction Q^T, coarse operator Kc and prolongation P:  e = P Kc^{-1} Q^T r."""
import sys
import numpy as np, scipy.linalg as sla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct
from exp_lod import vertex_graph, dof_vertex_incidence

p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney; f = np.setdiff1d(np.arange(p.n), c); nc = len(c)
floc = -np.ones(p.n, int); floc[f] = np.arange(len(f))
Kff, Kfc = K[f][:, f].tocsc(), K[f][:, c].tocsc()
pats = [x for x in mesh_vertex_patches(p) if len(x)]
lu = direct(Kff)
X = np.column_stack([lu.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex))) for j in range(nc)])
IncT = dof_vertex_incidence(p).T.tocsr()
X1 = np.zeros_like(X)
for j in range(nc):
    loc = floc[np.unique(IncT[p.edges[:, p.ent[c[j]]]].indices)]; loc = loc[loc >= 0]
    X1[loc, j] = np.linalg.solve(Kff[loc][:, loc].toarray(), Kfc[loc, j].toarray().ravel())
def P_of(Xm):
    P = np.zeros((p.n, nc), complex); P[c] = np.eye(nc); P[f] = -Xm; return P
Inj, P1, Pid = P_of(0 * X), P_of(X1), P_of(X)
S = Pid.T @ (K @ Pid); S1 = P1.T @ (K @ P1); Kcc = K[c][:, c].toarray()
cases = [("Q=ideal Kc=S  P=inj ", Pid, S, Inj), ("Q=ideal Kc=S1 P=inj ", Pid, S1, Inj),
         ("Q=LOD1  Kc=S1 P=inj ", P1, S1, Inj), ("Q=LOD1  Kc=S  P=inj ", P1, S, Inj),
         ("Q=inj   Kc=S  P=ideal", Inj, S, Pid), ("Q=ideal Kc=Kcc P=inj", Pid, Kcc, Inj)]
for nm, Q, Kc, P in cases:
    F = sla.lu_factor(Kc)
    pc = TwoLevel(K, pats, c, coarse_fn=lambda r, P=P, F=F, Q=Q: P @ sla.lu_solve(F, Q.T @ r))
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
    print(f"  {nm} its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
