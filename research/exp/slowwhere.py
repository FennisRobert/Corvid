"""Where do the slow modes of E live? Share on rows touched by complex (ABC / port) entries,
distance to the outer boundary, local kh; and does adding them to the coarse space fix it?"""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as dla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches
from exp_keff import keff_h

p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney; nc = len(c)
pats = [x for x in mesh_vertex_patches(p) if len(x)]
pc = TwoLevel(K, pats, c)
E = sla.LinearOperator(K.shape, matvec=lambda v: v - pc.apply(K @ v), dtype=complex)
mu, V = sla.eigs(E, k=120, which="LM", tol=1e-5, ncv=260)
o = np.argsort(-abs(mu)); mu, V = mu[o], V[:, o]
rows = np.repeat(np.arange(p.n), np.diff(K.indptr))
cplx = np.bincount(rows, (K.data.imag != 0).astype(float), p.n) > 0
kh, _ = keff_h(p)
lo, hi = p.loc.min(1), p.loc.max(1)
# distance to the outer (absorbing) faces x=max, y=max, z=max
dist = np.min(np.stack([hi[0] - p.loc[0], hi[1] - p.loc[1], hi[2] - p.loc[2]]), 0) / (2 * np.pi / p.k0)
w = abs(V[:, :30]) ** 2; w /= w.sum(0)
print(f"{sys.argv[1]}: rows with complex entries {cplx.mean():.1%} of n; mean dist to ABC faces {dist.mean():.2f} wavelengths")
print(f"  30 slowest modes: share on complex rows {np.mean(w[cplx].sum(0)):.2f}, mean dist to ABC {np.mean((w * dist[:, None]).sum(0)):.2f} wl,"
      f" energy-weighted kh {np.mean((w * kh[:, None]).sum(0)):.2f} (median kh {np.median(kh):.2f})")
for m in (20, 60, 120):
    Pinj = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
    P = sp.hstack([Pinj, sp.csr_matrix(V[:, :m])]).tocsr()
    F = dla.lu_factor((P.T @ K @ P).toarray())
    pc2 = TwoLevel(K, pats, c, coarse_fn=lambda r: P @ dla.lu_solve(F, P.T @ r))
    x, its, hist = gmres(lambda v: K @ v, pc2.apply, p.b, maxiter=100)
    print(f"  + {m:3d} slow modes of E in the coarse space: its {its:3d} [{fmt_hist(hist)}]", flush=True)
