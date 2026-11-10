"""Ceiling of p-multigrid on the TRUE operator K (no shift): fine GS Hiptmair on K,
exact LU of R^T K R on the Whitney level. Compare with the shifted route (P)."""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from emdata import DATA
from helix.mldata import MLPreconData
from helix import palace_pc as pp
from pyamg.relaxation.relaxation import gauss_seidel

m, pid = sys.argv[1], int(sys.argv[2])
rec = MLPreconData.load(f"{DATA}/{m}/problem_{pid}.npz")
K, P, b = pp.restricted_system(rec, "keep")
sid = rec.solve_ids; nE = rec.mesh_edges.shape[1]; nT = (rec.E.shape[0] - 2 * nE) // 2
Gf = rec.grad.tocsr(); cols = pp.auxiliary_columns(Gf, sid, mode="filtered"); Gf = Gf[sid][:, cols].tocsr()
R, kept = pp.whitney_injection(sid, nE, nT); R = R.tocsr()
X = rec.mesh_nodes; e = rec.mesh_edges
h = np.linalg.norm(X[:, e[0]] - X[:, e[1]], axis=0)
lam = 2 * np.pi / rec.k0
print(f"{m} p{pid}: n={K.shape[0]}  k0={rec.k0:.1f}  wavelength {lam*1e3:.1f} mm  edge h median {np.median(h)*1e3:.2f} max {h.max()*1e3:.2f} mm"
      f"  -> {lam/np.median(h):.0f} (median) / {lam/h.max():.0f} (max) edges per wavelength")


class GSHip:
    def __init__(self, A, G, sweeps, S=None):
        self.A, self.G, self.Gt, self.s = A.tocsr(), G, G.T.tocsr(), sweeps
        self.S = S
        self.AG = pp.fix_zero_rows((self.Gt @ self.A @ G).tocsr())
        if S is not None:
            self.lu = spla.splu(self.A[S][:, S].tocsc(), **pp._LU)
    def _surf(self, b, y):
        if self.S is not None:
            y[self.S] += self.lu.solve(np.ascontiguousarray((b - self.A @ y)[self.S]))
        return y
    def _grad(self, b, y, sw):
        r = self.Gt @ (b - self.A @ y); z = np.zeros_like(r)
        gauss_seidel(self.AG, z, r, iterations=self.s, sweep=sw); return y + self.G @ z
    def pre(self, b):
        y = np.zeros_like(b); gauss_seidel(self.A, y, b, iterations=self.s, sweep="forward")
        return self._grad(b, self._surf(b, y), "forward")
    def post(self, b, y):
        y = self._surf(b, self._grad(b, y, "backward")); gauss_seidel(self.A, y, b, iterations=self.s, sweep="backward"); return y


def run(name, A, sm, csolve, cycles):
    pc = pp.PalacePC(A, sm, R, csolve, cycles)
    t0 = time.time(); x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=300)
    print(f"  {name:44s}: {its:4d} its  relres {rr:.1e}  {time.time()-t0:5.1f}s", flush=True)

luK = spla.splu((R.T @ K @ R).tocsc(), **pp._LU).solve
luP = spla.splu((R.T @ P @ R).tocsc(), **pp._LU).solve
print("  exact P^-1 (shifted floor): see helix notes, model2 p1 = 40")
rows = np.repeat(np.arange(P.shape[0]), np.diff(P.tocsr().indptr))
S = np.unique(rows[P.tocsr().data.imag != 0])
for sw in (1, 2):
    for cyc in (1, 2):
        run(f"K: GS+surf x{sw}, exact R'KR, cycles {cyc}", K, GSHip(K, Gf, sw, S), luK, cyc)
for sw in (1, 2):
    run(f"P: GS+surf x{sw}, exact R'PR, cycles 2", P, GSHip(P, Gf, sw, S), luP, 2)
