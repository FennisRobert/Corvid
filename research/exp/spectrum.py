"""Spectrum of the CORVID-preconditioned operator K M (right preconditioning) and the
location of its outlying eigenvectors.  usage: spectrum.py dataset [nev] [exact]"""
import sys
import numpy as np
import scipy.sparse.linalg as sla
from common import Problem, _kernels

name = sys.argv[1]
nev = int(sys.argv[2]) if len(sys.argv) > 2 else 200
exact = sys.argv[3] if len(sys.argv) > 3 else "none"
p = Problem(name)
coarse = p.top.whitney
if exact == "auto":
    m = p.pml.copy(); m[p.top.whitney] = True; coarse = np.flatnonzero(m)
pc = _kernels.TwoLevelSolver(p.kernels_csr(), p.top.block_ptr, p.top.block_idx, coarse.astype(np.int64), order=1)
# eigenvalues of T = I - M K (error propagator); outliers of M K = eigenvalues of T near or outside |mu| = 1
T = sla.LinearOperator((p.n, p.n), matvec=lambda v: v - pc.apply(np.ascontiguousarray(p.K @ v, dtype=complex)), dtype=complex)
mu, V = sla.eigs(T, k=nev, which="LM", tol=1e-6, ncv=max(2 * nev + 1, 60))
lam = 1 - mu
o = np.argsort(-abs(mu)); mu, lam, V = mu[o], lam[o], V[:, o]
B = 3e-3 if "b5" not in name else 5e-3
thick = 2e-3
depth = np.clip(np.max(p.loc - B, axis=0) / thick, 0, 1)
print(f"{name}: n={p.n}, PML rows {p.pml.mean():.0%}, coarse rows {len(coarse)}")
for thr in (0.9, 0.7, 0.5, 0.3):
    print(f"  #eig of I-MK with |mu|>{thr}: {np.sum(abs(mu) > thr)}" + ("  (all computed!)" if np.sum(abs(mu) > thr) == nev else ""))
w = abs(V) ** 2
w /= w.sum(0)
print("  rank |mu|   lambda(MK)          PML   W    E2   F    mean depth")
for i in list(range(10)) + list(range(10, nev, max(1, nev // 10))):
    wi = w[:, i]
    print(f"  {i:4d} {abs(mu[i]):.3f}  {lam[i].real:+.3f}{lam[i].imag:+.3f}i  {wi[p.pml].sum():.2f} {wi[p.kind==0].sum():.2f} "
          f"{wi[p.kind==1].sum():.2f} {wi[p.kind==2].sum():.2f}  {np.sum(wi*depth)/max(wi[p.pml].sum(),1e-12):.2f}")
np.savez(f"/tmp/spec_{name}_{exact}.npz", mu=mu, V=V[:, :min(nev, 60)])
