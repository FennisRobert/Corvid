"""AMGe vs helix's AMS as the Whitney-level solver of the Palace-style
preconditioner (helix/palace_pc.build), on the same matrix A1 = Re+Im of
R^T P_flip R.

  inner: PCG on A1 x = b, preconditioned by AMS(2 its) or AMGe(1 or 2 cycles)
  outer: GMRES(200) on the full N2 system with palace_pc's preconditioner and
         its coarse solve swapped
usage: compare_coarse.py model [pid] [target]
"""
import sys, time
import numpy as np
import scipy.sparse as sp

sys.path[:0] = [".", "tests", "research/probes",
                "reference/helix"]
from emdata import load, DATA
from corvid.amge.mg import hierarchy, AMGe
from helix.mldata import MLPreconData
from helix import palace_pc as pp

model = sys.argv[1]; pid = int(sys.argv[2]) if len(sys.argv) > 2 else 1
target = int(sys.argv[3]) if len(sys.argv) > 3 else 32
d = load(model, pid)
rec = MLPreconData.load(f"{DATA}/{model}/problem_{pid}.npz")

t0 = time.time()
K, b, pc = pp.build(rec)                                   # AMS coarse (current best)
t_build = time.time() - t0
sid = rec.solve_ids; nE = d["n_edges"]
R, kept = pp.whitney_injection(sid, nE, d["n_tris"])
assert np.array_equal(kept, np.flatnonzero(d["free"]))
Pc = (R.T @ pp.restricted_system(rec, "flip")[1] @ R).tocsr()
A1 = sp.csr_matrix(Pc.real + Pc.imag)

t0 = time.time()
Ps, Gs = hierarchy(d["coords"], d["tets"], d["edges"], d["free"], target=target)
t_h = time.time() - t0
amge = AMGe(A1, Ps, Gs)
t_amge = time.time() - t0
sizes = [A1.shape[0]] + [P.shape[1] for P in Ps]
print(f"{model} p{pid}: N2 n={K.shape[0]}  Whitney n={A1.shape[0]}  levels {sizes}  "
      f"hierarchy {t_h:.1f}s (+galerkin/smoothers {t_amge - t_h:.1f}s)")


def pcg(A, b, M, tol=1e-8, maxit=500):
    x = np.zeros_like(b); r = b.copy(); z = M(r); p = z.copy(); rz = r @ z; bn = np.linalg.norm(b)
    for k in range(1, maxit + 1):
        Ap = A @ p; a = rz / (p @ Ap); x += a * p; r -= a * Ap
        if np.linalg.norm(r) < tol * bn:
            return k
        z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
    return maxit


def stationary(M, A, n):
    def f(r):
        x = M(r)
        for _ in range(n - 1):
            x = x + M(r - A @ x)
        return x
    return f


rhs = np.random.default_rng(0).standard_normal(A1.shape[0])
ams_r = pc.coarse.f                                          # helix ams_n (2 stationary AMS its)
for name, M in [("AMS x2", ams_r), ("AMGe x1", amge), ("AMGe x2", stationary(amge, A1, 2))]:
    t0 = time.time(); k = pcg(A1, rhs, M); t = time.time() - t0
    print(f"  inner PCG 1e-8  {name:8s}: {k:4d} its  {t:6.2f}s")

for name, cs in [("AMS x2", pc.coarse), ("AMGe x1", pp.RealOnComplex(amge)),
                 ("AMGe x2", pp.RealOnComplex(stationary(amge, A1, 2)))]:
    pc.coarse = cs
    t0 = time.time()
    x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
    print(f"  outer GMRES 1e-6 {name:8s}: {its:4d} its  relres {rr:.1e}  {time.time() - t0:6.1f}s")
