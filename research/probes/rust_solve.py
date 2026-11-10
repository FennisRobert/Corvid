"""Whole solve phase in Rust (corvid._kernels.PSolver) vs the Python prototype
(helix palace_pc + CORVID GS coarse), same inputs. usage: rust_solve.py model pid"""
import sys, time
import numpy as np, scipy.sparse as sp
sys.argv = ["x", sys.argv[1], sys.argv[2]]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
from corvid._kernels import PSolver, timers


def t(A, dtype):
    A = sp.csr_matrix(A); A.sort_indices()
    return (A.indptr.astype(np.int32), A.indices.astype(np.int32),
            np.ascontiguousarray(A.data, dtype=dtype), A.shape[0], A.shape[1])

C, R_ = np.complex128, np.float64
t0 = time.time()
K, b, pc = pp.build(rec)
sm = pc.smoother
a = AMGe(P1, Ps, Gs)                                      # Galerkin chain for the CORVID levels
levels = []
for l, (Al, Pl, Gl) in enumerate(zip(a.As[:-1], Ps, Gs[:-1])):
    AG = (Gl.T @ Al @ Gl).tocsr()
    levels.append((t(Al, C), t(Pl, R_), t(Pl.T, R_), t(Gl, R_), t(Gl.T, R_), t(AG, C)))
inj = pc.R.tocsc().indices.astype(np.int64)
S = pp.SurfaceBlockJacobi(pc.A, "lu").S.astype(np.int64)
import os
COLORED = os.environ.get("COLORED", "1") == "1"
NLEV = int(os.environ.get("NLEV", len(levels)))
levels = levels[:NLEV]
print("CORVID smoothed levels", NLEV, "coarsest rows", a.As[NLEV].shape[0], "nnz", a.As[NLEV].nnz)
rs = PSolver(t(K, C), t(pc.A, C), t(sm.G, R_), t(sm.Gt, R_), t(sm.SG.A, C), S, inj,
             levels, t(a.As[NLEV], C), order=4, cycles=2, coarse_its=2, colored=COLORED)
print("multicolour GS:", COLORED)
print(f"rust setup {time.time() - t0:.2f}s (incl. python build)")

# python reference preconditioner
ar = AMGe(P1, Ps, Gs)
ar.sm = [TalosSmoother(Al, G, np.array([], int)) for Al, G in zip(ar.As[:-1], Gs[:-1])]
pc.coarse = stat(ar, P1, 2)
r = np.random.default_rng(0).standard_normal(K.shape[0]) * (1 + 0.3j)
zp, zr = pc(r), rs.apply(r)
print(f"|M_rust r - M_py r| / |M_py r| = {np.linalg.norm(zr - zp) / np.linalg.norm(zp):.2e}")

for method in ("gmres", "cocg"):
    t0 = time.time()
    x, its, rr = (pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=600) if method == "gmres"
                  else pp.cocg(K, b, pc, rtol=1e-6, maxiter=600))
    tp = time.time() - t0
    timers(); t0 = time.time(); x2, its2, rr2 = rs.solve(b.astype(C), method, 1e-6, 200, 600); tr = time.time() - t0
    tm = timers()
    print(f"  {method:5s}  python {its:4d} its {rr:.1e} {tp:5.2f}s   rust {its2:4d} its {rr2:.1e} {tr:5.2f}s")
    print("     rust breakdown: " + "  ".join(f"{n} {v:.2f}s" for n, v in tm))
