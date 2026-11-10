"""M5: Krylov (minimal-residual) smoother: each smoothing step minimises ||b - K (y + Z c)|| over
Z = [B r, B K B r, ...] (m vectors, B = vertex-block additive Schwarz or one multiplicative sweep),
outer flexible GMRES (common.gmres stores the preconditioned vectors). Work counted in block sweeps.
usage: exp_krylov_smoother.py dataset ..."""
import sys, time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist
from proto import mesh_vertex_patches
from exp_smoothers import Smoother

for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr()
    m = np.zeros(p.n, bool); m[p.top.whitney] = True
    ex = corvid.Solver(p.top)._exact_rows(K)
    if ex is not None: m |= ex
    coarse = np.flatnonzero(m)
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    sm = Smoother(K, pats, coarse, "AS1")
    print(f"{name}: n={p.n}", flush=True)

    def mr_smooth(b, y, msteps, inner):
        r = b - K @ y
        Z, W, v = [], [], r
        for _ in range(msteps):
            z = inner(v); Z.append(z); W.append(K @ z); v = W[-1]
        Z = np.array(Z).T; W = np.array(W).T
        cvec, *_ = np.linalg.lstsq(W, r, rcond=None)
        return y + Z @ cvec

    for label, msteps, kind in (("MR(1) additive", 1, "as"), ("MR(2) additive", 2, "as"), ("MR(3) additive", 3, "as"),
                                ("MR(1) multiplicative", 1, "gs"), ("MR(2) multiplicative", 2, "gs")):
        inner = sm.B if kind == "as" else (lambda v: sm.gs(v, np.zeros_like(v), True))

        def apply(r, msteps=msteps, inner=inner):
            r = np.asarray(r, complex)
            y = mr_smooth(r, np.zeros_like(r), msteps, inner)
            res = r - K @ y
            y[coarse] += sm.tl.clu.solve(np.ascontiguousarray(res[coarse]))
            return mr_smooth(r, y, msteps, inner)

        t = time.time()
        x, its, hist = gmres(lambda v: K @ v, apply, p.b, maxiter=150)
        print(f"  {label:22s} its {its:3d}  block sweeps {its * 2 * msteps:4d}  res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e}"
              f"  {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
