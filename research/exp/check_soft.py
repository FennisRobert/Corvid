"""Soft PML (EMerge deltamax 2) vs default (8): corvid with the automatic switch; memory vs direct."""
import time, numpy as np, corvid
from common import Problem, sparta_nnz
for name in ("box_d2", "box_d8", "model4_d2", "model4_d8", "model4_d2_r15", "model4_d8_r15"):
    p = Problem(name, 1)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    direct = sparta_nnz(p.K)[0] * 16 / 1e6
    t = time.time(); s = corvid.Solver(top, maxiter=300, restart=300).factorize(p.K, k0=p.k0); s.solve(p.b); tt = time.time() - t
    m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(p.K)
    ex = sparta_nnz(p.K[m][:, m])[0] * 16 / 1e6
    print(f"{name:14s} n={p.n:6d} |s|99 {s.pml_stretch:4.1f}: its {s.info[0]['iterations']:3d} {tt:4.1f}s | exact level {ex:4.0f} MB + blocks {s.memory_bytes/1e6:4.0f} MB "
          f"= {ex + s.memory_bytes/1e6:4.0f} MB vs direct {direct:4.0f} MB ({(ex + s.memory_bytes/1e6)/direct:.0%})", flush=True)
