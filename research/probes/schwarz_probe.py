"""Vertex-patch Schwarz vs Hiptmair as the fine (N2) smoother, all in Rust,
coarse = exact SPARTA LU of the Whitney level. usage: schwarz_probe.py model pid"""
import sys, time
import numpy as np, scipy.sparse as sp
sys.argv = ["x", sys.argv[1], sys.argv[2]]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
from corvid._kernels import PSolver, timers

def t(A, dtype):
    A = sp.csr_matrix(A); A.sort_indices()
    return (A.indptr.astype(np.int32), A.indices.astype(np.int32), np.ascontiguousarray(A.data, dtype=dtype), A.shape[0], A.shape[1])
C, R_ = np.complex128, np.float64

K, b, pc = pp.build(rec); sm = pc.smoother
inj = pc.R.tocsc().indices.astype(np.int64)
S = pp.SurfaceBlockJacobi(pc.A, "lu").S.astype(np.int64)
P = pc.A
# vertex patches in solve numbering: DOFs on incident edges (2 each) and faces (2 each)
nV = rec.mesh_nodes.shape[1]; E = rec.mesh_edges; F = rec.mesh_tris; nE, nF = E.shape[1], F.shape[1]
pos = -np.ones(2 * nE + 2 * nF, np.int64); pos[rec.solve_ids] = np.arange(len(rec.solve_ids))
vd = [[] for _ in range(nV)]
for e in range(nE):
    for v in E[:, e]:
        vd[v] += [e, nE + e]
for f in range(nF):
    for v in F[:, f]:
        vd[v] += [2 * nE + f, 2 * nE + nF + f]
patches = [p[p >= 0] for p in (pos[np.array(d)] for d in vd)]
patches = [p for p in patches if len(p)]
pptr = np.r_[0, np.cumsum([len(p) for p in patches])].astype(np.int64)
pidx = np.concatenate(patches).astype(np.int32)
sz = np.diff(pptr)
print(f"n={K.shape[0]}  {len(patches)} vertex patches, size mean {sz.mean():.0f} max {sz.max()}  "
      f"(sum m^2 = {np.sum(sz.astype(float)**2)/1e6:.1f}M entries)")
G_ = (t(sm.G, R_), t(sm.Gt, R_))
Kc = (R.T @ K @ R).tocsr()

def run(name, op, coarse, use_patches, use_grad, order, cycles, methods=("gmres", "cocg")):
    AG = pp.fix_zero_rows((sm.Gt @ op @ sm.G).tocsr())
    t0 = time.time()
    kw = dict(order=order, cycles=cycles, coarse_its=1, use_grad=use_grad)
    if use_patches:
        kw.update(patch_ptr=pptr, patch_idx=pidx)
    rs = PSolver(t(K, C), t(op, C), *G_, t(AG, C), S, inj, [], t(coarse, C), **kw)
    tb = time.time() - t0
    for m in methods:
        timers(); t0 = time.time(); x, its, rr = rs.solve(b.astype(C), m, 1e-6, 300, 400); ts = time.time() - t0
        tm = dict(timers())
        print(f"  {name:44s} {m:5s}: {its:4d} its  relres {rr:.1e}  solve {ts:5.2f}s  (setup {tb:.1f}s, "
              f"patch/surface {tm['surface LU']:.2f}s, SpMV {tm['fine Chebyshev SpMV']:.2f}s, coarse {tm['coarsest LU']:.2f}s)", flush=True)

run("Hiptmair+surfLU, P (current)", P, P1, False, True, 4, 2)
for order, cycles in ((4, 2), (2, 1), (1, 1)):
    run(f"Schwarz+grad, P, cheb {order}, cycles {cycles}", P, P1, True, True, order, cycles)
for order, cycles in ((2, 1), (1, 1)):
    run(f"Schwarz no grad, P, cheb {order}, cycles {cycles}", P, P1, True, False, order, cycles)
for order, cycles in ((2, 1), (1, 1)):
    run(f"Schwarz+grad, K (unshifted), cheb {order}, cyc {cycles}", K, Kc, True, True, order, cycles)
