"""Package check: Solver(pml_depth=1) (= CORVID 0.2.0 auto) vs 2 vs 3 on all PML datasets."""
import sys, time
import numpy as np, corvid
from common import Problem, sparta_nnz
cases = [("sm_d8", 1), ("box_d8", 1), ("box_d8_b5", 1), ("model4", 1), ("model4", 4), ("model4", 7), ("model4_d8_r15", 1), ("box_abc", 1)]
for name, pid in cases:
    p = Problem(name, pid)
    out = []
    for d in (1, 2, 3):
        t = time.time()
        s = corvid.Solver(p.top, pml_depth=d, maxiter=300, restart=300).factorize(p.K)
        tf = time.time() - t; t = time.time(); s.solve(p.b); ts = time.time() - t
        m = np.zeros(p.n, bool); m[p.top.whitney] = True
        ex = s._exact_rows(p.K)
        if ex is not None: m |= ex
        out.append(f"depth {d}: +{s.n_exact:6d} rows, LDL {sparta_nnz(p.K[m][:, m])[0]/1e6:5.1f}M, its {s.info[0]['iterations']:3d}, {tf + ts:5.1f}s")
    print(f"{name:14s} p{pid} n={p.n:6d} | " + " | ".join(out), flush=True)
