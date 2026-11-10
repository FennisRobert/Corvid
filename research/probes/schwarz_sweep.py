"""Unshifted K: vertex-patch Schwarz (+ Hiptmair gradient step) fine smoother,
exact SPARTA LU of R^T K R, all Rust. Sweep over problems.
usage: schwarz_sweep.py model id1 id2 ..."""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from emdata import DATA
from helix.mldata import MLPreconData
from helix import palace_pc as pp
from corvid._kernels import PSolver

def t(A, dtype):
    A = sp.csr_matrix(A); A.sort_indices()
    return (A.indptr.astype(np.int32), A.indices.astype(np.int32), np.ascontiguousarray(A.data, dtype=dtype), A.shape[0], A.shape[1])
C, R_ = np.complex128, np.float64
model = sys.argv[1]
for pid in map(int, sys.argv[2:]):
    rec = MLPreconData.load(f"{DATA}/{model}/problem_{pid}.npz")
    K, P, b = pp.restricted_system(rec, "keep")
    sid = rec.solve_ids; nE = rec.mesh_edges.shape[1]; nF = rec.mesh_tris.shape[1]
    Gf = rec.grad.tocsr(); cols = pp.auxiliary_columns(Gf, sid, mode="filtered"); Gf = Gf[sid][:, cols].tocsr()
    R, kept = pp.whitney_injection(sid, nE, nF); R = R.tocsr()
    inj = R.tocsc().indices.astype(np.int64)
    pos = -np.ones(2 * nE + 2 * nF, np.int64); pos[sid] = np.arange(len(sid))
    nV = rec.mesh_nodes.shape[1]; Em, Fm = rec.mesh_edges, rec.mesh_tris
    ve = sp.csr_matrix((np.ones(2 * nE), (Em.ravel(order="F"), np.repeat(np.arange(nE), 2))), shape=(nV, nE))
    vf = sp.csr_matrix((np.ones(3 * nF), (Fm.ravel(order="F"), np.repeat(np.arange(nF), 3))), shape=(nV, nF))
    patches = []
    for v in range(nV):
        e = ve.indices[ve.indptr[v]:ve.indptr[v + 1]]; f = vf.indices[vf.indptr[v]:vf.indptr[v + 1]]
        d = pos[np.r_[e, nE + e, 2 * nE + f, 2 * nE + nF + f]]
        d = d[d >= 0]
        if len(d):
            patches.append(d)
    pptr = np.r_[0, np.cumsum([len(p) for p in patches])].astype(np.int64)
    pidx = np.concatenate(patches).astype(np.int32)
    Kc = (R.T @ K @ R).tocsr()
    AG = pp.fix_zero_rows((Gf.T @ K @ Gf).tocsr())
    out = []
    for order, grad in ((2, True), (1, True), (2, False)):
        t0 = time.time()
        rs = PSolver(t(K, C), t(K, C), t(Gf, R_), t(Gf.T, R_), t(AG, C), np.zeros(0, np.int64), inj, [], t(Kc, C),
                     order=order, cycles=1, coarse_its=1, patch_ptr=pptr, patch_idx=pidx, use_grad=grad)
        tb = time.time() - t0
        for m in ("gmres", "cocg"):
            t0 = time.time(); x, its, rr = rs.solve(b.astype(C), m, 1e-6, 300, 300); ts = time.time() - t0
            out.append(f"{'o%d%s' % (order, 'g' if grad else '')} {m[:2]} {its:3d} ({ts:.2f}s{'' if rr < 1e-6 else ' rr=%.0e' % rr})")
        del rs
    print(f"{model} p{pid:<2d} n={K.shape[0]:6d} k0={rec.k0:6.1f} setup~{tb:.1f}s | " + "  ".join(out), flush=True)
