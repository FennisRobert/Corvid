"""CORVID vs PML strength (deltamax) on the regenerated horn. usage: pml_strength.py dir ..."""
import sys, time
import numpy as np, scipy.sparse as sp
import corvid
from corvid import _kernels

def load(d):
    f = np.load(f"{d}/problem_1.npz")
    A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
    sid = f["solve_ids"]
    return f["mesh_edges"], f["mesh_tris"], sid, A[sid][:, sid].tocsr(), f["b"][sid], f["solution"][sid]

def lam(K, top, its=40):
    n = K.shape[0]
    k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), n, n)
    s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney[:1], order=1)   # ~ one-level M
    v = np.random.default_rng(0).standard_normal(n) + 0j
    for _ in range(its):
        w = s.apply(K @ v); l = np.linalg.norm(w); v = w / l
    return l

for d in sys.argv[1:]:
    edges, tris, sid, K, b, ref = load(d)
    K.sort_indices()
    top = corvid.Topology.nedelec2(edges, tris, sid)
    print(f"{d.split('/')[-1]}: n={K.shape[0]}, |K x_ref - b|/|b| = {np.linalg.norm(K @ ref - b) / np.linalg.norm(b):.1e}, "
          f"|lambda_max(S K)| ~ {lam(K, top):.2f} (order-1 smoother scale incl.)", flush=True)
    for order in (1, 2):
        s = corvid.Solver(top, order=order, maxiter=300, restart=300).factorize(K)
        t0 = time.time(); x = s.solve(b); ts = time.time() - t0
        print(f"   order {order}: GMRES {s.info[0]['iterations']:3d} its, residual {s.info[0]['residual']:.1e}, "
              f"error vs EMerge {np.linalg.norm(x - ref) / np.linalg.norm(ref):.1e}, solve {ts:.2f}s", flush=True)
