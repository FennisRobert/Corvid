"""H1: are the slow modes the near-kernel of K_ff (the complement block), concentrated in large elements?"""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as dla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct
from exp_keff import keff_h

p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney; nc = len(c)
f = np.setdiff1d(np.arange(p.n), c)
pats = [x for x in mesh_vertex_patches(p) if len(x)]
kh, _ = keff_h(p)
Kff = K[f][:, f].tocsc()
lu = direct(Kff)
Op = sla.LinearOperator(Kff.shape, matvec=lambda v: lu.solve(np.ascontiguousarray(v, dtype=complex)), dtype=complex)
ev, U = sla.eigs(Kff, k=150, sigma=0, OPinv=Op)
o = np.argsort(abs(ev)); ev, U = ev[o], U[:, o]
d = np.abs(Kff.diagonal()).mean()
print(f"{sys.argv[1]}: |eig(K_ff)|/mean diag: {abs(ev[0])/d:.1e} (1st) {abs(ev[19])/d:.1e} (20th) {abs(ev[-1])/d:.1e} (150th)")
w = abs(U[:, :20]) ** 2
print(f"  kh of DOFs: median {np.median(kh):.2f}; energy-weighted kh of the 20 lowest K_ff modes: {np.sum(w * kh[f][:, None], 0).mean():.2f}")
def run(Z, label):
    Pinj = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
    P = sp.hstack([Pinj, sp.csr_matrix(Z)]).tocsr()
    F = dla.lu_factor((P.T @ K @ P).toarray())
    pc = TwoLevel(K, pats, c, coarse_fn=lambda r: P @ dla.lu_solve(F, P.T @ r))
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
    print(f"  {label:36s} its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
run(np.zeros((p.n, 0)), "Whitney only")
for m in (10, 30, 60, 100, 150):
    Z = np.zeros((p.n, m), complex); Z[f] = U[:, :m]
    run(Z, f"+ {m} lowest eigenvectors of K_ff")
