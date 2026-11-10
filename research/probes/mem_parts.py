import sys, resource
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs
import corvid
from corvid import _kernels
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
part = sys.argv[1]
edges, tris, sid, K, b = emerge_inputs("model2_abc2", 1)
top = corvid.Topology.nedelec2(edges, tris, sid)
K = sp.csr_matrix(K); K.sort_indices()
p0 = peak()
if part == "whitney":
    W = K[top.whitney][:, top.whitney].tocsr(); W.sort_indices()
    t = (W.indptr.astype(np.int32), W.indices.astype(np.int32), W.data.astype(np.complex128), W.shape[0], W.shape[1])
    p0 = peak(); nnzL, tf, ts = _kernels.bench_direct(t, 1); p1 = peak()
    print(f"SPARTA Whitney LDL^T: nnz(L)={nnzL/1e6:.2f}M ({nnzL*16/2**20:.0f} MB of factor values) -> RSS +{p1-p0:.0f} MB")
elif part == "blocks":
    # blocks only: a solver whose coarse set is a single row (negligible)
    k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), K.shape[0], K.shape[1])
    p0 = peak(); s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney[:1], order=2); p1 = peak()
    print(f"split K + vertex blocks ({s.schwarz_bytes/2**20:.0f} MB) + 1-row coarse -> RSS +{p1-p0:.0f} MB")
if part == "direct":
    t = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), K.shape[0], K.shape[1])
    p0 = peak(); nnzL, tf, ts = _kernels.bench_direct(t, 1); p1 = peak()
    print(f"SPARTA full K LDL^T: nnz(L)={nnzL/1e6:.1f}M ({nnzL*16/2**20:.0f} MB of factor values) -> RSS +{p1-p0:.0f} MB")
