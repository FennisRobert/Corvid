"""E1: energy-minimising-AMG-style prolongation for the p-two-level method.
P~ has the LOD1 / thresholded sparsity of X and is fitted row by row so that P~ B_c = B exactly
(least squares) for m near-kernel vectors B of K (smallest |lambda|, by shift-invert: concept test only).
Coarse: Q = injection, Kc = P~^T K P~."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as dla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct

name = sys.argv[1]
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney; f = np.setdiff1d(np.arange(p.n), c); nc = len(c)
pats = [x for x in mesh_vertex_patches(p) if len(x)]
lu = direct(K[f][:, f])
Kfc = K[f][:, c].tocsc()
X = np.column_stack([lu.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex))) for j in range(nc)])
mx = np.abs(X).max(0)
Klu = direct(K)
Kinv = sla.LinearOperator(K.shape, matvec=lambda v: Klu.solve(np.ascontiguousarray(v, dtype=complex)), dtype=complex)
ev, V = sla.eigs(K, k=200, sigma=0, OPinv=Kinv, which="LM")
order = np.argsort(abs(ev)); V = V[:, order]; ev = ev[order]
print(f"{name}: smallest |lambda(K)| {abs(ev[0]):.2e} ... 200th {abs(ev[-1]):.2e}")

def run(label, pattern, m):
    """pattern: boolean (nf x nc) allowed entries of X~."""
    B = V[:, :m]; Bc = B[c]; Bf = B[f]
    Xt = np.zeros_like(X)
    X0 = np.where(pattern, X, 0)                         # start: truncated exact X
    for i in range(len(f)):
        J = np.flatnonzero(pattern[i])
        if len(J) == 0: continue
        Cm = Bc[J].T                                      # m x |J|:   -X~[i,J] @ Bc[J] = Bf[i]
        x0 = X0[i, J]
        t = -Bf[i] - x0 @ Bc[J]                             # constraint residual: X~[i,J] Bc[J] = -Bf[i]
        d, *_ = np.linalg.lstsq(Cm, t, rcond=1e-10)        # min-norm correction
        Xt[i, J] = x0 + d
    P = np.zeros((p.n, nc), complex); P[c] = np.eye(nc); P[f] = -Xt
    cons = np.linalg.norm(P @ Bc - B) / max(np.linalg.norm(B), 1e-300)
    F = dla.lu_factor(P.T @ (K @ P))
    pc = TwoLevel(K, pats, c, coarse_fn=lambda r: P @ dla.lu_solve(F, r[c]))
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
    print(f"  {label:26s} m={m:3d} nnz/col {pattern.sum()/nc:5.0f} constraint err {cons:.1e} its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)

for thr in (1e-1, 1e-2):
    pat = np.abs(X) >= thr * mx
    for m in (0, 30, 100, 200):
        run(f"drop<{thr:g} pattern", pat, m)
