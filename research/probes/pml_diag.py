"""model4 p1: where does CORVID lose it on the PML horn?"""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from test_api import emerge_inputs
from helix import palace_pc as pp
import corvid
from corvid import _kernels
edges, tris, sid, K, b = emerge_inputs("model4", 1)
b = np.ascontiguousarray(b[:, 0], dtype=np.complex128)
top = corvid.Topology.nedelec2(edges, tris, sid)
K = sp.csr_matrix(K); K.sort_indices()
k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), K.shape[0], K.shape[1])

def hist(name, M, maxit=300):
    rel = []
    x, its, rr = pp.gmres(K, b, M, rtol=1e-6, restart=maxit, maxiter=maxit, callback=lambda i, r: rel.append(r))
    pick = [rel[i] for i in (9, 29, 99, 199, len(rel) - 1) if i < len(rel)]
    print(f"  {name:44s}: {its:3d} its, residual at 10/30/100/200/end: " + " ".join(f"{r:.1e}" for r in pick), flush=True)

for order in (1, 2):
    s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney, order=order)
    hist(f"two-level, order {order}, unrestarted GMRES", s.apply)
s1 = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney[:1], order=1)
hist("one-level Schwarz (no coarse), order 1", s1.apply)
W = K[top.whitney][:, top.whitney].tocsc()
import scipy.sparse.linalg as spla
lu = spla.splu(W)
def coarse_only(r):
    z = np.zeros_like(r); z[top.whitney] = lu.solve(r[top.whitney]); return z
hist("coarse only (Whitney LU, nothing else)", coarse_only, 100)
hist("no preconditioner", lambda r: r, 100)
