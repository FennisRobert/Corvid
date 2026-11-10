"""lambda_max of (additive vertex-block Schwarz) * K, and where its eigenvector lives."""
import sys
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs, DATA
import corvid
for m, pid in (("model2_abc2", 1), ("model3", 9), ("model4", 1)):
    edges, tris, sid, K, b = emerge_inputs(m, pid)
    f = np.load(f"{DATA}/{m}/problem_{pid}.npz")
    E = sp.csr_matrix((f["E_data"], f["E_indices"], f["E_indptr"]), shape=tuple(f["E_shape"]))[sid][:, sid].tocsr()
    rows = np.repeat(np.arange(E.shape[0]), np.diff(E.indptr))
    pml = np.zeros(E.shape[0], bool); pml[np.unique(rows[E.data.imag != 0])] = True
    top = corvid.Topology.nedelec2(edges, tris, sid); K = sp.csr_matrix(K); n = K.shape[0]
    bp, bi = top.block_ptr, top.block_idx
    blocks = [bi[bp[v]:bp[v + 1]] for v in range(len(bp) - 1)]
    inv = [np.linalg.inv(K[d][:, d].toarray()) for d in blocks]
    def S(r):
        z = np.zeros(n, complex)
        for d, A in zip(blocks, inv):
            z[d] += A @ r[d]
        return z
    v = np.random.default_rng(0).standard_normal(n) + 0j
    for _ in range(60):
        w = S(K @ v); l = np.linalg.norm(w); v = w / l
    mult = np.bincount(bi, minlength=n)
    share = np.sum(abs(v[pml]) ** 2) / np.sum(abs(v) ** 2)
    i = np.argsort(-abs(v))[:20]
    print(f"{m} p{pid}: |lambda_max| {l:6.2f}  (block multiplicity mean {mult.mean():.2f}, max {mult.max()})  "
          f"PML DOFs {pml.mean():.0%}, eigenvector weight in PML {share:.0%}, top-20 entries in PML {pml[i].mean():.0%}")
