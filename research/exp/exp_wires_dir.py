"""N4: wire direction on the horn (additive, w=3, Chebyshev order 2)."""
import time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, pack, kcsr, lambda_max
from corvid import _kernels
from exp_wires import pencils
p = Problem("model4", 1); K = p.K.tocsr()
s = corvid.Solver(p.top)
m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(K)
coarse = np.flatnonzero(m)
pats_all = mesh_vertex_patches(p); pats = [x for x in pats_all if len(x)]
tl = TwoLevel(K, pats, coarse); kk = kcsr(K)
X = p.nodes
print("horn extent (mm):", [(round(X[a].min()*1e3, 1), round(X[a].max()*1e3, 1)) for a in range(3)], flush=True)
for axis in (1, 2):
    sw = _kernels.RxSweep(kk, *pack(pencils(p, pats_all, axis, 3.0)))
    B = lambda v: sw.additive(np.ascontiguousarray(v, dtype=complex))
    lm = 1.1 * lambda_max(K, B, p.n)
    def smooth(b, y):
        r = b - K @ y; d = 4 / (3 * lm) * B(r)
        y = y + d; r = r - K @ d
        d = d * (1 / 5) + (12 / (5 * lm)) * B(r)
        return y + d
    def apply(r):
        r = np.asarray(r, complex); y = smooth(r, np.zeros_like(r))
        res = r - K @ y; y[coarse] += tl.clu.solve(np.ascontiguousarray(res[coarse]))
        return smooth(r, y)
    t = time.time()
    x, its, hist = gmres(lambda v: K @ v, apply, p.b, maxiter=150)
    print(f"  wires w=3 along {'xyz'[axis]}: its {its}, res {np.linalg.norm(K @ x - p.b)/np.linalg.norm(p.b):.0e}, {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
