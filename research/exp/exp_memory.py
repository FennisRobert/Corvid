"""Q1: memory of corvid 0.3.0 on the horns vs a direct LDL^T, and three reductions:
  (a) drop vertex blocks whose DOFs are all in the exact level
  (b) pml_depth = 3
  (c) vertex blocks as sparse SPARTA factors instead of dense packed inverses (storage only here)
All numbers in MB of complex128 entries (16 B); SPARTA factor nonzeros for LDL^T."""
import sys, time
import numpy as np, corvid
from common import Problem, gmres, fmt_hist, sparta_nnz
from proto import TwoLevel, mesh_vertex_patches, pack, kcsr
from corvid import _kernels

MB = 16 / 1e6
for name, pid in (("model4", 1), ("model4", 7), ("model4_d8_r15", 1)):
    p = Problem(name, pid); K = p.K.tocsr()
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    direct = sparta_nnz(K)[0] * MB
    print(f"{name} p{pid}: n={p.n}, direct LDL {direct:.0f} MB", flush=True)
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    for depth in (2, 3):
        s = corvid.Solver(top, pml_depth=depth)
        s.k0 = p.k0
        ex = s._exact_rows(K)
        m = ex.copy(); m[p.top.whitney] = True
        coarse = np.flatnonzero(m)
        exact_mb = sparta_nnz(K[m][:, m])[0] * MB
        for drop in (False, True):
            keep = [x for x in pats if not (drop and m[x].all())]
            dense_mb = sum(len(x) * (len(x) + 1) // 2 for x in keep) * MB
            sw = _kernels.RxSweep(kcsr(K), *pack(keep))
            sparse_mb = sw.factor_nnz * MB
            t = time.time()
            pc = TwoLevel(K, keep, coarse)
            x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
            ok = np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b) < 1e-5
            print(f"  pml_depth {depth} {'drop blocks in exact region' if drop else 'all vertex blocks          '}: exact {exact_mb:4.0f} MB + blocks "
                  f"{dense_mb:4.0f} MB dense ({sparse_mb:4.0f} MB as sparse factors) = {exact_mb + dense_mb:4.0f} MB "
                  f"({(exact_mb + dense_mb) / direct:4.0%} of direct; {(exact_mb + sparse_mb) / direct:4.0%} with sparse blocks) | its {its if ok else 'stall'} "
                  f"({len(keep)}/{len(pats)} blocks) {time.time()-t:.1f}s", flush=True)
