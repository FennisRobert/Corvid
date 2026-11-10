"""Two PML fixes on the res-0.15 horn (delta = 8):
  A: coarse level = Whitney rows + every PML DOF, solved exactly
  B: CORVID built on the soft-PML matrix (delta = 1, same mesh) as preconditioner for delta = 8"""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["reference/helix"]
from helix import palace_pc as pp
import corvid
from corvid import _kernels
S = "research/data"
def load(d):
    f = np.load(f"{S}/{d}/problem_1.npz")
    A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
    sid = f["solve_ids"]; K = A[sid][:, sid].tocsr(); K.sort_indices()
    return f["mesh_edges"], f["mesh_tris"], sid, K, np.ascontiguousarray(f["b"][sid][:, 0], dtype=np.complex128)
tup = lambda K: (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), K.shape[0], K.shape[1])
E, F, sid, K8, b = load("model4_d8_r15")
_, _, sid1, K1, _ = load("model4_d1_r15")
assert np.array_equal(sid, sid1) and K1.shape == K8.shape
top = corvid.Topology.nedelec2(E, F, sid); n = K8.shape[0]
rows = np.repeat(np.arange(n), np.diff(K8.indptr))
frac = np.bincount(rows, weights=(K8.data.imag != 0).astype(float), minlength=n) / np.diff(K8.indptr)
pml = frac > 0.5
print(f"n={n}, PML DOFs {pml.sum()} ({pml.mean():.0%}), Whitney {len(top.whitney)}")

def run(name, Kpc, coarse, solver="gmres"):
    t0 = time.time(); s = _kernels.TwoLevelSolver(tup(Kpc), top.block_ptr, top.block_idx, coarse, order=1); tf = time.time() - t0
    t0 = time.time()
    if Kpc is K8:
        x, its, rr = s.solve(b, solver, 1e-6, 300, 300)
    else:
        x, its, rr = pp.gmres(K8, b, s.apply, rtol=1e-6, restart=300, maxiter=300)
    print(f"  {name:52s}: {its:3d} its, residual {rr:.1e}, factorize {tf:.2f}s, solve {time.time() - t0:.2f}s", flush=True)

#run("current (Whitney coarse)", K8, top.whitney)
run("A: coarse = Whitney + all PML DOFs", K8, np.flatnonzero(pml | np.isin(np.arange(n), top.whitney)).astype(np.int64))
#run("B: CORVID on the delta=1 matrix, GMRES on delta=8", K1, top.whitney)
d = K8.diagonal()
ratio = abs(d.imag) / abs(d)
wh = np.isin(np.arange(n), top.whitney)
print(f"|Im K_ii| / |K_ii| in PML rows: median {np.median(ratio[pml]):.2f}, p10 {np.percentile(ratio[pml], 10):.2f}; outside PML: max {ratio[~pml].max():.3f}")
for tau in (0.3, 0.5, 0.7, 0.85, 0.95):
    sel = ratio >= tau
    run(f"A(tau={tau}): Whitney + {sel.sum():5d} rows with |Im|/|K_ii| >= {tau}", K8, np.flatnonzero(sel | wh).astype(np.int64))
