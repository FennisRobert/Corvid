"""Vertex blocks: density, and storage of a sparse factor vs the dense inverse."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs
import corvid
model, pid = sys.argv[1], int(sys.argv[2])
edges, tris, sid, K, b = emerge_inputs(model, pid)
top = corvid.Topology.nedelec2(edges, tris, sid)
K = sp.csr_matrix(K)
bp, bi = top.block_ptr, top.block_idx
rng = np.random.default_rng(0)
dense = nnzb = fill_amd = 0
idx = rng.choice(len(bp) - 1, size=min(400, len(bp) - 1), replace=False)
for v in idx:
    d = bi[bp[v]:bp[v + 1]]; m = len(d)
    A = K[d][:, d].tocsc()
    dense += m * (m + 1) / 2
    nnzb += (sp.triu(A).nnz)
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    fill_amd += lu.L.nnz                        # L of an LDL^T-like factor (symmetric mode)
s = len(bp) - 1
print(f"{model} p{pid}: per vertex block (sample {len(idx)} of {s}): dense upper {dense/len(idx):.0f}, "
      f"block upper nnz {nnzb/len(idx):.0f} ({nnzb/dense:.0%} full), sparse factor L {fill_amd/len(idx):.0f} entries "
      f"-> {fill_amd/dense:.0%} of the dense inverse")
