"""N2: wire smoothers on the model4 horn (p1, p4), coarse = Whitney + deep PML (corvid 0.2.1)."""
import sys, time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches
from exp_wires import WireSmoother, pencils
for pid in (1,):
    p = Problem("model4", pid); K = p.K.tocsr()
    s = corvid.Solver(p.top); t = time.time(); s.factorize(K); s.solve(p.b)
    m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(K)
    coarse = np.flatnonzero(m)
    print(f"model4 p{pid}: n={p.n}; corvid 0.2.1: its {s.info[0]['iterations']} ({time.time()-t:.1f}s), vertex blocks {s.memory_bytes/1e6:.0f} MB", flush=True)
    pats_all = mesh_vertex_patches(p); pats = [x for x in pats_all if len(x)]
    wx = [pencils(p, pats_all, 0, 2.0)]
    for label, sets, om in (("mult. vertex om 0.6", [pats], 0.6), ("mult. vertex om 0.8", [pats], 0.8),
                            ("wires w=2 x om 0.6", wx, 0.6), ("wires w=2 x om 0.8", wx, 0.8), ("wires w=2 x om 0.4", wx, 0.4)):
        t = time.time()
        ws = WireSmoother(K, coarse, sets, pats, omega=om)
        tset = time.time() - t; t = time.time()
        x, its, hist = gmres(lambda v: K @ v, ws.apply, p.b, maxiter=150)
        print(f"  {label:22s}: its {its:3d}, factor {ws.nnz/1e6:6.2f}M ({ws.nnz*16/1e6:.0f} MB), res {np.linalg.norm(K @ x - p.b)/np.linalg.norm(p.b):.0e}, "
              f"setup {tset:.1f}s solve {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
