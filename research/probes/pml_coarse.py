"""model4: is the Whitney coarse level the problem in the PML? Coarse set = Whitney rows
plus every DOF in the PML (rows where the curl matrix E itself is complex)."""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs, DATA
import corvid
from corvid import _kernels
pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
edges, tris, sid, K, b = emerge_inputs("model4", pid)
f = np.load(f"{DATA}/model4/problem_{pid}.npz")
E = sp.csr_matrix((f["E_data"], f["E_indices"], f["E_indptr"]), shape=tuple(f["E_shape"]))[sid][:, sid].tocsr()
rows = np.repeat(np.arange(E.shape[0]), np.diff(E.indptr))
pml = np.zeros(E.shape[0], bool); pml[np.unique(rows[E.data.imag != 0])] = True
top = corvid.Topology.nedelec2(edges, tris, sid)
K = sp.csr_matrix(K); K.sort_indices(); n = K.shape[0]
k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), n, n)
bb = np.ascontiguousarray(b[:, 0], dtype=np.complex128)
whit = np.zeros(n, bool); whit[top.whitney] = True
print(f"model4 p{pid}: n={n}, PML DOFs (complex curl rows) {pml.sum()} ({pml.mean():.0%}), Whitney {whit.sum()}, "
      f"Whitney in PML {np.sum(whit & pml)}")
for name, cs in (("Whitney (current)", whit), ("Whitney + all PML DOFs", whit | pml), ("Whitney + PML Whitney only", whit)):
    idx = np.flatnonzero(cs).astype(np.int64)
    t0 = time.time(); s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, idx, order=1); tf = time.time() - t0
    t0 = time.time(); x, its, rr = s.solve(bb, "gmres", 1e-6, 300, 300); ts = time.time() - t0
    print(f"  coarse = {name:28s} ({len(idx):6d} rows): GMRES {its:3d} its, residual {rr:.1e}, factorize {tf:.2f}s solve {ts:.2f}s", flush=True)
# smoother without the PML: restrict every vertex block to non-PML DOFs
bp, bi = top.block_ptr, top.block_idx
keep = ~pml[bi]
cnt = np.add.reduceat(keep.astype(np.int64), bp[:-1]) if len(bi) else np.zeros(0, np.int64)
cnt[np.diff(bp) == 0] = 0
bp2 = np.r_[0, np.cumsum(cnt)].astype(np.int64)
bi2 = bi[keep].astype(np.int32)
nz = np.diff(bp2) > 0
bp2 = np.r_[0, np.cumsum(np.diff(bp2)[nz])].astype(np.int64)
for name, cs in (("Whitney + all PML DOFs", whit | pml), ("Whitney only", whit)):
    idx = np.flatnonzero(cs).astype(np.int64)
    s = _kernels.TwoLevelSolver(k, bp2, bi2, idx, order=1)
    t0 = time.time(); x, its, rr = s.solve(bb, "gmres", 1e-6, 300, 300); ts = time.time() - t0
    print(f"  smoother WITHOUT PML, coarse = {name:22s}: GMRES {its:3d} its, residual {rr:.1e}, solve {ts:.2f}s", flush=True)
