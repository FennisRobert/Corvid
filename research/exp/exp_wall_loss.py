"""P1: artificial loss on metal walls, in the preconditioner only. K_pc = K + i beta |diag K| on the
unknowns touching a metal vertex (all such vertices, or those with k0 h > 1). The two-level
preconditioner (vertex blocks + exact level of 0.2.1) is built from K_pc; GMRES runs on the true K.
Compared with corvid 0.2.1 and 0.3.0 (exact wall rows)."""
import sys, time
import numpy as np, scipy.sparse as sp, corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches
from check_wall import AP
from check_wall2 import wall_vertex_rows


def wall_rows_all(p, kappa):
    r = wall_vertex_rows(p, kappa)                    # complement rows around selected wall vertices
    # include the Whitney rows there too
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    return r


cases = [("model4", 7), ("box_abc_r30", 1), ("model1", 1)]
for name, pid in cases:
    p = Problem(name, pid) if (name.startswith("model4") or name.startswith("box")) else AP(name, pid)
    K = p.K.tocsr()
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    res = {}
    for label, k0 in (("0.2.1", None), ("0.3.0", p.k0)):
        t = time.time(); s = corvid.Solver(top, maxiter=200, restart=200).factorize(K, k0=k0); s.solve(p.b)
        res[label] = f"{s.info[0]['iterations']} its {time.time()-t:.1f}s"
    base = corvid.Solver(top)._exact_rows(K)
    m = np.zeros(p.n, bool); m[p.top.whitney] = True
    if base is not None: m |= base
    coarse = np.flatnonzero(m)
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    out = []
    d = np.abs(K.diagonal())
    for kappa in (1.0,):
        rows = wall_vertex_rows(p, kappa)
        for beta in (0.001, 0.003, 0.01):
            Kpc = (K + sp.diags(1j * beta * d * rows)).tocsr()
            t = time.time()
            pc = TwoLevel(Kpc, pats, coarse)
            x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
            ok = np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b) < 1e-5
            out.append(f"walls kh>{kappa:.0f} beta {beta}: {its if ok else 'stall'}")
    print(f"{name:13s} p{pid}: 0.2.1 {res['0.2.1']} | 0.3.0 {res['0.3.0']} | " + " | ".join(out), flush=True)
