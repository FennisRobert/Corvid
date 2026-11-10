import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.argv = ["x", "model2_abc2", "1"]
src = open("research/probes/coarse_gs.py").read().split("rows = np.repeat")[0]
exec(src)
P1c = P1.tocsc()
for name, opts in [("SuperLU COLAMD", dict(permc_spec="COLAMD")), ("SuperLU MMD_AT_PLUS_A sym", dict(permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)))]:
    t0 = time.time(); lu = spla.splu(P1c, **opts); tf = time.time() - t0
    r = np.random.default_rng(0).standard_normal(P1.shape[0]) + 0j
    lu.solve(r); t0 = time.perf_counter()
    for _ in range(50): lu.solve(r)
    ts = (time.perf_counter() - t0) / 50 * 1e3
    print(f"{name:28s}: n={P1.shape[0]} nnz(L+U)={lu.L.nnz + lu.U.nnz}  factor {tf:.2f}s  solve {ts:.2f} ms")

from corvid._kernels import bench_direct
A = sp.csr_matrix(P1); A.sort_indices()
tt = (A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data.astype(np.complex128), A.shape[0], A.shape[1])
nnzL, tf, ts = bench_direct(tt, 50)
print(f"SPARTA CPU LDLT              : n={A.shape[0]} nnz(L)={nnzL}  factor {tf:.2f}s  solve {ts:.2f} ms")
