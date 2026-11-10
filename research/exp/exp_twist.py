"""N5: 'mode conversion' between block arrangements (the antenna designer's diagonal post).
Each arrangement B_j (vertex blocks, x-wires, y-wires, z-wires) is applied additively with its own
lambda_max damping (order 1 Chebyshev, as CORVID). Variants:
  pre/post twist  pre-smooth with arrangement A, post-smooth with arrangement B (fixed preconditioner)
  rotate          preconditioner of outer iteration i uses arrangement i mod m (flexible GMRES)
usage: exp_twist.py  (runs the under-resolved box and the model4 horn p1)"""
import time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, pack, kcsr, lambda_max
from corvid import _kernels
from exp_wires import pencils


def run(name, pid):
    p = Problem(name, pid); K = p.K.tocsr()
    s = corvid.Solver(p.top)
    m = np.zeros(p.n, bool); m[p.top.whitney] = True
    ex = s._exact_rows(K)
    if ex is not None: m |= ex
    coarse = np.flatnonzero(m)
    pats_all = mesh_vertex_patches(p); pats = [x for x in pats_all if len(x)]
    tl = TwoLevel(K, pats, coarse); kk = kcsr(K)
    arr = {"V": pats, "x": pencils(p, pats_all, 0, 3.0), "y": pencils(p, pats_all, 1, 3.0), "z": pencils(p, pats_all, 2, 3.0)}
    sm = {}
    for key, blocks in arr.items():
        sw = _kernels.RxSweep(kk, *pack(blocks))
        B = (lambda sw: (lambda v: sw.additive(np.ascontiguousarray(v, dtype=complex))))(sw)
        sm[key] = (B, 4 / (3 * 1.1 * lambda_max(K, B, p.n)))

    def smooth(key, b, y):
        B, om = sm[key]
        return y + om * B(b - K @ y)

    def two_level(pre, post):
        def apply(r):
            r = np.asarray(r, complex)
            y = smooth(pre, r, np.zeros_like(r))
            res = r - K @ y
            y[coarse] += tl.clu.solve(np.ascontiguousarray(res[coarse]))
            return smooth(post, r, y)
        return apply

    print(f"{name} p{pid}: n={p.n}", flush=True)
    cases = [("V/V (CORVID)", two_level("V", "V")), ("x/x", two_level("x", "x")),
             ("x/V twist", two_level("x", "V")), ("V/x twist", two_level("V", "x")),
             ("x/y twist", two_level("x", "y")), ("y/z twist", two_level("y", "z"))]
    for label, apply in cases:
        t = time.time()
        x, its, hist = gmres(lambda v: K @ v, apply, p.b, maxiter=150)
        print(f"  {label:14s} its {its:3d} res {np.linalg.norm(K @ x - p.b)/np.linalg.norm(p.b):.0e} {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
    for label, seq in (("rotate x,y,z", "xyz"), ("rotate V,x", "Vx"), ("rotate V,x,y,z", "Vxyz")):
        state = {"i": 0}
        def apply(r, seq=seq):
            key = seq[state["i"] % len(seq)]; state["i"] += 1
            return two_level(key, key)(r)
        t = time.time()
        x, its, hist = gmres(lambda v: K @ v, apply, p.b, maxiter=150)
        print(f"  {label:14s} its {its:3d} res {np.linalg.norm(K @ x - p.b)/np.linalg.norm(p.b):.0e} {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)


run("box_abc_r30", 1)
run("model4", 1)
