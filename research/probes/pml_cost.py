import sys, time, resource
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
exec(open("research/probes/pml_fix.py").read().split("def run(")[0])
from corvid._kernels import bench_direct
coarse = np.flatnonzero(pml | np.isin(np.arange(n), top.whitney))
Kc = K8[coarse][:, coarse].tocsr(); Kc.sort_indices()
for name, M in (("full K (direct solver)", K8), ("CORVID coarse: Whitney + PML", Kc)):
    nnzL, tf, ts = bench_direct(tup(M), 3)
    print(f"  {name:30s}: n={M.shape[0]:6d}  nnz(L)={nnzL/1e6:5.1f}M ({nnzL*16/2**20:4.0f} MB)  factor {tf:.2f}s  solve {ts:.1f} ms")
