"""Slowest error mode of e <- (I - M K) e for the CORVID two-level M. usage: pml_errmode.py dir"""
import sys
import numpy as np, scipy.sparse as sp
import corvid
from corvid import _kernels
d = sys.argv[1]
f = np.load(f"{d}/problem_1.npz")
A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
sid = f["solve_ids"]; K = A[sid][:, sid].tocsr(); K.sort_indices(); n = K.shape[0]
X = f["mesh_nodes"]; E = f["mesh_edges"]; F = f["mesh_tris"]; ne, nf = E.shape[1], F.shape[1]
top = corvid.Topology.nedelec2(E, F, sid)
# location and kind of every solved DOF
g = sid
mid_e = X[:, E].mean(1); mid_f = X[:, F].mean(1)
loc = np.where(g < 2 * ne, mid_e[:, np.where(g < ne, g, g - ne) % ne], mid_f[:, np.where(g < 2 * ne + nf, g - 2 * ne, g - 2 * ne - nf) % nf])
kind = np.where(g < ne, "W", np.where(g < 2 * ne, "E2", "F"))
# PML rows: most of the row is complex (volume stretching), surface terms touch only part of a row
rows = np.repeat(np.arange(n), np.diff(K.indptr))
frac = np.bincount(rows, weights=(K.data.imag != 0).astype(float), minlength=n) / np.diff(K.indptr)
pml = frac > 0.5
k = (K.indptr.astype(np.int32), K.indices.astype(np.int32), K.data.astype(np.complex128), n, n)
s = _kernels.TwoLevelSolver(k, top.block_ptr, top.block_idx, top.whitney, order=1)
e = np.random.default_rng(0).standard_normal(n) + 0j
rho = []
for it in range(80):
    e2 = e - s.apply(K @ e)
    rho.append(np.linalg.norm(e2) / np.linalg.norm(e)); e = e2 / np.linalg.norm(e2)
share = lambda m: np.sum(abs(e[m]) ** 2)
print(f"{d.split('/')[-1]}: n={n}, PML DOFs {pml.mean():.0%}; stationary contraction per step (last 10): {np.mean(rho[-10:]):.3f}")
print(f"  slowest error mode: in PML {share(pml):.0%} | Whitney {share(kind == 'W'):.0%}, 2nd edge DOF {share(kind == 'E2'):.0%}, face {share(kind == 'F'):.0%}")
for ax, nm in enumerate("xyz"):
    w = abs(e) ** 2
    print(f"  {nm}: weighted mean {np.sum(w * loc[ax]) / w.sum() * 1e3:6.2f} mm, spread {np.sqrt(np.sum(w * (loc[ax] - np.sum(w * loc[ax]) / w.sum()) ** 2) / w.sum()) * 1e3:5.2f} mm "
          f"(domain {loc[ax].min()*1e3:6.2f} .. {loc[ax].max()*1e3:6.2f}; PML from {loc[ax, pml].min()*1e3:6.2f})")
# smoothness: correlation of the mode between neighbouring DOFs (Whitney edges sharing a node)
