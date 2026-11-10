"""Per-block amplification: spectral norm of A_v^-1 K[v, :] (the block's local action),
air vs PML, and where the worst ones are."""
import sys
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs, DATA
import corvid
m, pid = sys.argv[1], int(sys.argv[2])
edges, tris, sid, K, b = emerge_inputs(m, pid)
f = np.load(f"{DATA}/{m}/problem_{pid}.npz")
E = sp.csr_matrix((f["E_data"], f["E_indices"], f["E_indptr"]), shape=tuple(f["E_shape"]))[sid][:, sid].tocsr()
rows = np.repeat(np.arange(E.shape[0]), np.diff(E.indptr))
pml = np.zeros(E.shape[0], bool); pml[np.unique(rows[E.data.imag != 0])] = True
xyz = f["dof_coords"][:, sid]
top = corvid.Topology.nedelec2(edges, tris, sid); K = sp.csr_matrix(K)
bp, bi = top.block_ptr, top.block_idx
amp, inpml, cond, where = [], [], [], []
for v in range(len(bp) - 1):
    d = bi[bp[v]:bp[v + 1]]
    A = K[d][:, d].toarray()
    Kv = K[d].toarray() if len(d) < 0 else K[d]
    M = np.linalg.solve(A, K[d].toarray())          # m x n, local action
    amp.append(np.linalg.norm(M, 2)); inpml.append(pml[d].mean()); cond.append(np.linalg.cond(A))
    where.append(xyz[:, d].mean(1))
amp, inpml, cond, where = map(np.array, (amp, inpml, cond, where))
P = inpml > 0.5
q = lambda a: f"median {np.median(a):6.2f}  p95 {np.percentile(a, 95):7.2f}  max {a.max():8.2f}"
print(f"{m} p{pid}: {len(amp)} blocks ({P.sum()} in PML)")
print(f"  ||A_v^-1 K_v,:||  air: {q(amp[~P])}")
if P.any():
    print(f"  ||A_v^-1 K_v,:||  PML: {q(amp[P])}")
print(f"  cond(A_v)         air: median {np.median(cond[~P]):.1e}  max {cond[~P].max():.1e}" +
      (f"   PML: median {np.median(cond[P]):.1e}  max {cond[P].max():.1e}" if P.any() else ""))
top10 = np.argsort(-amp)[:8]
for i in top10:
    print(f"    worst block {i}: amp {amp[i]:7.2f}  cond {cond[i]:.1e}  PML share {inpml[i]:.0%}  at "
          f"({where[i][0]*1e3:6.2f}, {where[i][1]*1e3:6.2f}, {where[i][2]*1e3:6.2f}) mm")
