"""C1: complex-shifted coarse operator  Kc = K_cc + i beta D,  D = |diag(K_cc)|  (injection P)."""
import sys
import numpy as np, scipy.sparse as sp
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches
for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    Kcc = K[c][:, c].tocsr(); D = sp.diags(np.abs(Kcc.diagonal()))
    print(name)
    for beta in (0, 0.01, 0.03, 0.1, 0.3, -0.03, -0.1):
        pc = TwoLevel(K, pats, c, coarse_op=(Kcc + 1j * beta * D).tocsr())
        x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
        print(f"  beta {beta:+.2f}: its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
