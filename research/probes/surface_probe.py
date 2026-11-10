"""TALOS-style surface block in the fine Hiptmair smoother of helix's
Palace-style preconditioner (model2: second-order ABC).

fine smoother variants (pre order; post is the reverse, so M stays symmetric):
  cur      Chebyshev scaled by SurfaceBlockJacobi(lu), gradient    (helix default)
  gs-vol   Chebyshev Jacobi on VOLUME rows only, surface solve, gradient
  gs-sbj   Chebyshev scaled by SurfaceBlockJacobi(lu), surface solve, gradient
coarse: exact complex LU of R^T P R (isolates the fine level), or AMGe.
usage: surface_probe.py model pid [coarse]
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path[:0] = [".", "tests", "research/probes",
                "reference/helix"]
from emdata import DATA, load
from helix.mldata import MLPreconData
from helix import palace_pc as pp
from helix.hierarchy import fix_zero_rows
from corvid.amge.mg import hierarchy, AMGe

m, pid = sys.argv[1], int(sys.argv[2])
coarse = sys.argv[3] if len(sys.argv) > 3 else "exact"
rec = MLPreconData.load(f"{DATA}/{m}/problem_{pid}.npz")
K, P, b = pp.restricted_system(rec, "keep")
sid = rec.solve_ids; nE = rec.mesh_edges.shape[1]; nT = (rec.E.shape[0] - 2 * nE) // 2
Gf = rec.grad.tocsr(); cols = pp.auxiliary_columns(Gf, sid, mode="filtered"); Gf = Gf[sid][:, cols].tocsr()
R, kept = pp.whitney_injection(sid, nE, nT); R = R.tocsr()
P1 = (R.T @ P @ R).tocsr()
sbj = pp.SurfaceBlockJacobi(P, "lu")
S = sbj.S
lu_S = spla.splu(P[S][:, S].tocsc(), **pp._LU)
print(f"{m} p{pid}: n={P.shape[0]} surface rows {len(S)}  coarse={coarse}")


class SurfGS:
    def __init__(self, A, G, minv, order=4):
        self.A, self.G, self.Gt = A, G, G.T.tocsr()
        self.Sm = pp.Chebyshev4(A, order, minv=minv)
        self.SG = pp.Chebyshev4(fix_zero_rows((self.Gt @ A @ G).tocsr()), order)

    def surf(self, b, y):
        y = y.copy(); y[S] += lu_S.solve(np.ascontiguousarray((b - self.A @ y)[S])); return y

    def grad(self, b, y):
        return y + self.G @ self.SG(self.Gt @ (b - self.A @ y))

    def pre(self, b):
        return self.grad(b, self.surf(b, self.Sm(b)))

    def post(self, b, y):
        return self.Sm(b, self.surf(b, self.grad(b, y)))


d_vol = 1.0 / P.diagonal(); d_vol[S] = 0.0
smoothers = {
    "cur": pp.DistRelax(P, Gf, 4, minv=sbj),
    "gs-vol": SurfGS(P, Gf, lambda r: d_vol * r),
    "gs-sbj": SurfGS(P, Gf, sbj),
}

if coarse == "exact":
    csolve = spla.splu(P1.tocsc(), **pp._LU).solve
else:                                      # AMGe on the flipped SPD matrix (current best)
    d = load(m, pid)
    Ps, Gs = hierarchy(d["coords"], d["tets"], d["edges"], d["free"], target=32)
    Pc = (R.T @ pp.restricted_system(rec, "flip")[1] @ R).tocsr()
    A1 = sp.csr_matrix(Pc.real + Pc.imag); amge = AMGe(A1, Ps, Gs)
    csolve = pp.RealOnComplex(lambda r: (lambda x: x + amge(r - A1 @ x))(amge(r)))

for name, sm in smoothers.items():
    pc = pp.PalacePC(P, sm, R, csolve, 2)
    for solver in ("gmres", "cocg"):
        t0 = time.time()
        if solver == "gmres":
            x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
        else:
            x, its, rr = pp.cocg(K, b, pc, rtol=1e-6, maxiter=600)
        print(f"  {name:7s} {solver:5s}: {its:4d} its  relres {rr:.1e}  {time.time() - t0:5.1f}s")
