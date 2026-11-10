"""model4 p1: where is the error after 100 CORVID iterations? (PML rows vs the rest)"""
import sys
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from test_api import emerge_inputs, DATA
from helix import palace_pc as pp
import corvid
edges, tris, sid, K, b = emerge_inputs("model4", 1)
f = np.load(f"{DATA}/model4/problem_1.npz")
ref = f["solution"][sid, 0]; xyz = f["dof_coords"][:, sid]
b = np.ascontiguousarray(b[:, 0], dtype=np.complex128)
top = corvid.Topology.nedelec2(edges, tris, sid)
s = corvid.Solver(top, order=1, maxiter=100, restart=100).factorize(K)
x = s.solve(b)
err = x - ref
K = sp.csr_matrix(K)
rows = np.repeat(np.arange(K.shape[0]), np.diff(K.indptr))
pml = np.zeros(K.shape[0], bool); pml[np.unique(rows[K.data.imag != 0])] = True
share = lambda v, m: np.sum(abs(v[m]) ** 2) / np.sum(abs(v) ** 2)
print(f"after {s.info[0]['iterations']} its, residual {s.info[0]['residual']:.1e}, error {np.linalg.norm(err)/np.linalg.norm(ref):.1e}")
print(f"rows with imaginary part (PML + port): {pml.mean():.0%} of DOFs")
print(f"  share of |solution|^2 there: {share(ref, pml):.0%}   share of |error|^2 there: {share(err, pml):.0%}")
print(f"  relative error inside: {np.linalg.norm(err[pml])/np.linalg.norm(ref[pml]):.1e}   outside: {np.linalg.norm(err[~pml])/np.linalg.norm(ref[~pml]):.1e}")
for ax, nm in enumerate("xyz"):
    print(f"  {nm}-range of PML rows {xyz[ax, pml].min()*1e3:6.2f} .. {xyz[ax, pml].max()*1e3:6.2f} mm; others {xyz[ax, ~pml].min()*1e3:6.2f} .. {xyz[ax, ~pml].max()*1e3:6.2f} mm")
