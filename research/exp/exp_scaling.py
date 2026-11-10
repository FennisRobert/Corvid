"""R1: memory scaling of corvid 0.4.0 vs a direct LDL^T (SPARTA), soft PML (deltamax 2).
Rows: n, direct MB, corvid exact-level MB, block MB, total MB, ratio, iterations, times."""
import sys, time
import numpy as np, corvid
from common import Problem, sparta_nnz

rows = []
for name in sys.argv[1:]:
    p = Problem(name)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    t = time.time(); dn, dt = sparta_nnz(p.K); direct = dn * 16 / 1e6
    t = time.time(); s = corvid.Solver(top).factorize(p.K, k0=p.k0); tf = time.time() - t
    t = time.time(); s.solve(p.b); ts = time.time() - t
    m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(p.K)
    ex = sparta_nnz(p.K[m][:, m])[0] * 16 / 1e6
    bl = s.memory_bytes / 1e6
    epl = p.edges_per_wavelength()[0]
    print(f"{name:8s} n={p.n:8d} epl {epl:4.1f} | direct {direct:7.0f} MB ({dt:5.1f}s) | corvid exact {ex:6.0f} + blocks {bl:6.0f} = {ex + bl:6.0f} MB "
          f"({(ex + bl) / direct:4.0%}) | its {s.info[0]['iterations']:3d} factorize {tf:5.1f}s solve {ts:5.1f}s", flush=True)
    rows.append((p.n, direct, ex, bl))
if len(rows) > 1:
    a = np.array(rows, float)
    for j, label in ((1, "direct"), (2, "corvid exact level"), (3, "corvid blocks"), (None, "corvid total")):
        y = a[:, 2] + a[:, 3] if j is None else a[:, j]
        e = np.polyfit(np.log(a[:, 0]), np.log(y), 1)[0]
        print(f"  fitted exponent, memory ~ n^p: {label:20s} p = {e:.2f}")
