"""Stationary contraction of e <- (I - M K) e and the spectrum of M K (Arnoldi
Ritz values), for the CORVID two-level M. usage: contraction.py model pid | dir"""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes"]
import corvid
from corvid import _kernels
a = sys.argv[1]
if "/" in a:
    f = np.load(f"{a}/problem_1.npz")
    A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
    sid = f["solve_ids"]; K = A[sid][:, sid].tocsr(); E, F = f["mesh_edges"], f["mesh_tris"]; name = a.split("/")[-1]
else:
    from test_api import emerge_inputs
    E, F, sid, K, b = emerge_inputs(a, int(sys.argv[2])); name = f"{a} p{sys.argv[2]}"
K = sp.csr_matrix(K); K.sort_indices(); n = K.shape[0]
top = corvid.Topology.nedelec2(E, F, sid)
k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), n, n)
s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney, order=1)
op = spla.LinearOperator((n, n), matvec=lambda v: s.apply(K @ np.ascontiguousarray(v, dtype=np.complex128)), dtype=complex)
ev = spla.eigs(op, k=12, which="LM", return_eigenvectors=False, maxiter=3000, tol=1e-3)
ev2 = spla.eigs(op, k=12, which="SR", return_eigenvectors=False, maxiter=3000, tol=1e-3)
far = lambda z: np.abs(1 - z)
print(f"{name}: largest |lambda(MK)|: " + " ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in ev[np.argsort(-abs(ev))][:5]))
print(f"   most negative Re lambda(MK): " + " ".join(f"{z.real:+.2f}{z.imag:+.2f}i" for z in ev2[np.argsort(ev2.real)][:5]))
print(f"   max |1 - lambda| among these: {max(far(ev).max(), far(ev2).max()):.2f}")
