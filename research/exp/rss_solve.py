"""Peak process memory of one solve. usage: rss_solve.py dataset {corvid|direct} [k0|nok0]
Prints RSS after loading K (baseline) and the peak RSS after the solve (macOS ru_maxrss is in bytes)."""
import sys, gc, resource, time
import numpy as np
from common import Problem


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def current_mb():
    import subprocess, os
    return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()) / 1e3


name, mode = sys.argv[1], sys.argv[2]
p = Problem(name)
K, b = p.K.tocsr(), np.ascontiguousarray(p.b, dtype=complex)
gc.collect()
base = current_mb()
peak0 = peak_mb()
t = time.time()
if mode == "corvid":
    import corvid
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    s = corvid.Solver(top, method="cocg").factorize(K, k0=p.k0)
    x = s.solve(b)
    info = f"its {s.info[0]['iterations']}, pml exact {s.n_pml_exact}, wall exact {s.n_wall_exact}, blocks {s.memory_bytes/1e6:.0f} MB"
else:
    from corvid import _kernels
    from proto import kcsr
    d = _kernels.RxDirect(kcsr(K))
    x = d.solve(b)
    info = f"factor nnz {d.factor_nnz/1e6:.1f}M = {d.factor_nnz*16/1e6:.0f} MB"
res = np.linalg.norm(K @ x - b) / np.linalg.norm(b)
print(f"{name} {mode}: n={p.n}, nnz(K)={K.nnz/1e6:.1f}M | RSS after loading {base:.0f} MB, peak {peak_mb():.0f} MB "
      f"(load peak {peak0:.0f}) -> solver adds {peak_mb() - base:.0f} MB | {time.time()-t:.1f}s res {res:.0e} | {info}")
