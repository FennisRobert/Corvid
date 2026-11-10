"""N3: additive wire blocks (+ CORVID's lambda_max damping, order 1 or 2), optionally together with the
vertex blocks, on the horn and the under-resolved box."""
import sys, time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, pack, kcsr, lambda_max
from corvid import _kernels
from exp_wires import pencils

def run(name, pid):
    p = Problem(name, pid); K = p.K.tocsr()
    s = corvid.Solver(p.top); s.factorize(K); s.solve(p.b)
    m = np.zeros(p.n, bool); m[p.top.whitney] = True
    ex = s._exact_rows(K)
    if ex is not None: m |= ex
    coarse = np.flatnonzero(m)
    pats_all = mesh_vertex_patches(p); pats = [x for x in pats_all if len(x)]
    tl = TwoLevel(K, pats, coarse)
    print(f"{name} p{pid}: n={p.n}; corvid 0.2.1: its {s.info[0]['iterations']}", flush=True)
    kk = kcsr(K)
    for label, blocks in (("wires w=2 x", pencils(p, pats_all, 0, 2.0)), ("wires w=2 x + vertex blocks", pencils(p, pats_all, 0, 2.0) + pats),
                          ("wires w=3 x", pencils(p, pats_all, 0, 3.0))):
        sw = _kernels.RxSweep(kk, *pack(blocks))
        B = lambda v: sw.additive(np.ascontiguousarray(v, dtype=complex))
        lm = 1.1 * lambda_max(K, B, p.n)
        for order in (1, 2):
            def smooth(b, y):
                r = b - K @ y
                d = 4 / (3 * lm) * B(r)
                for k in range(1, order):
                    y = y + d; r = r - K @ d
                    c1, c2 = (2 * k - 1) / (2 * k + 3), (8 * k + 4) / ((2 * k + 3) * lm)
                    d = d * c1 + c2 * B(r)
                return y + d
            def apply(r):
                r = np.asarray(r, complex)
                y = smooth(r, np.zeros_like(r))
                res = r - K @ y
                y[coarse] += tl.clu.solve(np.ascontiguousarray(res[coarse]))
                return smooth(r, y)
            t = time.time()
            x, its, hist = gmres(lambda v: K @ v, apply, p.b, maxiter=150)
            print(f"  additive {label:28s} order {order}: its {its:3d}, factor {sw.factor_nnz*16/1e6:4.0f} MB, "
                  f"res {np.linalg.norm(K @ x - p.b)/np.linalg.norm(p.b):.0e} {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)

run("box_abc_r30", 1)
run("model4", 1)
