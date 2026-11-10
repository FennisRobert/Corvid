"""M1: compatible relaxation (Brannick-Falgout). Relax K_ff e_f = 0 with the vertex-block smoother
restricted to the complement rows (additive with CORVID's damping, and multiplicative), report the
asymptotic convergence factor. Small factor: the Whitney coarse space is adequate, the smoother limits.
usage: exp_cr.py dataset ..."""
import sys
import numpy as np
from common import Problem
from proto import mesh_vertex_patches, lambda_max

for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
    f = np.setdiff1d(np.arange(p.n), c)
    floc = -np.ones(p.n, int); floc[f] = np.arange(len(f))
    Kff = K[f][:, f].tocsr()
    pats = []
    for x in mesh_vertex_patches(p):
        y = floc[x]; y = y[y >= 0]
        if len(y): pats.append(y)
    inv = [np.linalg.inv(Kff[x][:, x].toarray()) for x in pats]
    def B(r):
        z = np.zeros_like(r)
        for x, A in zip(pats, inv): z[x] += A @ r[x]
        return z
    om = 4 / (3 * 1.1 * lambda_max(Kff, B, len(f)))
    rng = np.random.default_rng(0)
    for mode in ("additive", "multiplicative"):
        e = rng.standard_normal(len(f)) + 0j
        rates = []
        for it in range(40):
            n0 = np.linalg.norm(e)
            if mode == "additive":
                e = e - om * B(Kff @ e)
            else:
                for x, A in zip(pats, inv):
                    e[x] -= A @ (Kff[x] @ e)
                for x, A in zip(reversed(pats), reversed(inv)):
                    e[x] -= A @ (Kff[x] @ e)
            rates.append(np.linalg.norm(e) / n0)
        print(f"{name}: compatible relaxation ({mode}{', omega %.2f' % om if mode == 'additive' else ', sym. GS sweep'}): "
              f"factor per sweep {np.mean(rates[-10:]):.3f}", flush=True)
