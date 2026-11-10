"""D2: exact level = Whitney + PML complement rows at BFS depth >= L (graph distance from the
non-PML vertices; topology only). usage: exp_depth.py dataset ..."""
import sys, time
import numpy as np
import corvid
from common import Problem, sparta_nnz
from proto import mesh_vertex_patches, bfs_depth

for name in sys.argv[1:]:
    p = Problem(name)
    pats = mesh_vertex_patches(p)
    vpml = np.array([len(x) > 0 and p.pml[x].mean() > 0.5 for x in pats])
    depth = bfs_depth(len(pats), p.edges, vpml)
    # DOF depth: max over its vertices
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    en = p.edges.T[np.minimum(p.ent, ne - 1)]; fn = p.tris.T[np.minimum(p.ent, nt - 1)]
    nodes = np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)
    ddof = depth[nodes].max(1)
    D = depth.max()
    W = np.zeros(p.n, bool); W[p.top.whitney] = True
    full = sparta_nnz(p.K)[0]
    print(f"{name}: n={p.n}, PML rows {p.pml.mean():.0%}, BFS depth 1..{D}, direct LDL {full/1e6:.1f}M")
    for L in [None] + list(range(1, min(D, 5) + 1)):
        m = (p.pml & ~W) if L is None else (p.pml & ~W & (ddof >= L))
        t = time.time()
        s = corvid.Solver(p.top, exact=m, maxiter=150, restart=150).factorize(p.K); s.solve(p.b)
        mm = m | W
        nz = sparta_nnz(p.K[mm][:, mm])[0]
        print(f"  {'auto' if L is None else 'depth >= %d' % L:10s} +{m.sum():6d} rows  coarse LDL {nz/1e6:6.2f}M ({nz/full:4.0%} of direct)"
              f"  its {s.info[0]['iterations']:3d}  {time.time()-t:.1f}s", flush=True)
