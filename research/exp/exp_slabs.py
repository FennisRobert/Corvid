"""S1: depth-layer (slab) block Gauss-Seidel sweeps through the PML, Whitney exact coarse.
Compared with CORVID 0.2.0 'auto' (Whitney + all PML rows exact). usage: exp_slabs.py dataset ..."""
import sys, time
import numpy as np
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, bfs_depth


def run(name):
    p = Problem(name)
    pats = mesh_vertex_patches(p)
    nv = len(pats)
    nonempty = [x for x in pats if len(x)]
    vpml = np.array([len(x) > 0 and p.pml[x].mean() > 0.5 for x in pats])
    depth = bfs_depth(nv, p.edges, vpml)
    D = depth.max()
    print(f"{name}: n={p.n}, PML rows {p.pml.mean():.0%}, BFS layers {D}")

    def slabs(w, stride, d0=1):
        out = []
        for j in range(d0, D + 1, stride):
            vs = np.flatnonzero((depth >= j) & (depth < j + w) & vpml)
            if len(vs):
                out.append(np.unique(np.concatenate([pats[v] for v in vs])))
        return out

    W = p.top.whitney
    allpml = np.flatnonzero(p.pml | np.isin(np.arange(p.n), W))
    configs = [("baseline: Whitney exact", W, None),
               ("0.2.0 auto: Whitney+PML exact", allpml, None),
               ("slab GS w=1", W, slabs(1, 1)),
               ("slab GS w=2 stride 1", W, slabs(2, 1)),
               ("slab GS w=2 stride 2", W, slabs(2, 2)),
               ("slab GS w=1 from depth 2", W, slabs(1, 1, 2))]
    for label, coarse, blocks in configs:
        t = time.time()
        pc = TwoLevel(p.K, nonempty, coarse, sweep_blocks=blocks)
        ts = time.time() - t
        t = time.time()
        x, its, hist = gmres(lambda v: p.K @ v, pc.apply, p.b, maxiter=100)
        rr = np.linalg.norm(p.K @ x - p.b) / np.linalg.norm(p.b)
        print(f"  {label:32s} its {its:3d} res {rr:.0e} | LDL nnz coarse {pc.coarse_nnz/1e6:5.2f}M + sweep {pc.sweep_nnz/1e6:5.2f}M"
              f" | setup {ts:.1f}s solve {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)


for name in sys.argv[1:]:
    run(name)
