"""TALOS recipe on the AMGe levels: GS edge sweep -> exact surface block ->
GS gradient correction (pre), reverse with backward sweeps (post)."""
import sys, time
import numpy as np, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes"]
exec(open("research/probes/coarse_keep.py").read().split("variants = {")[0])
from pyamg.relaxation.relaxation import gauss_seidel

class TalosSmoother:
    def __init__(self, A, G, rows, sweeps=1):
        self.A, self.G, self.Gt = A.tocsr(), G.tocsr(), G.T.tocsr()
        self.AG = (self.Gt @ self.A @ self.G).tocsr()
        self.rows = rows; self.sweeps = sweeps
        self.lu = spla.splu(self.A[rows][:, rows].tocsc()) if len(rows) else None
    def _gs(self, A, x, b, sweep):
        gauss_seidel(A, x, b, iterations=self.sweeps, sweep=sweep)
    def _surf(self, b, y):
        if self.lu is not None:
            y[self.rows] += self.lu.solve(np.ascontiguousarray((b - self.A @ y)[self.rows]))
    def _grad(self, b, y, sweep):
        r = self.Gt @ (b - self.A @ y); z = np.zeros_like(r); self._gs(self.AG, z, r, sweep); y += self.G @ z
    def pre(self, b):
        y = np.zeros_like(b); self._gs(self.A, y, b, "forward"); self._surf(b, y); self._grad(b, y, "forward"); return y
    def post(self, b, y):
        y = y.copy(); self._grad(b, y, "backward"); self._surf(b, y); self._gs(self.A, y, b, "backward"); return y

variants = {}
for surfon in (True, False):
    a = AMGe(P1, Ps, Gs, surface=surf)
    masks = [surf]
    for Pl in Ps:
        masks.append(np.asarray(abs(Pl).T @ masks[-1].astype(float)).ravel() > 0)
    a.sm = [TalosSmoother(Al, G, np.flatnonzero(mk) if surfon else np.array([], int))
            for Al, G, mk in zip(a.As[:-1], Gs[:-1], masks)]
    variants[f"AMGe keep GS {'+surf' if surfon else 'nosurf'} x1"] = a
    variants[f"AMGe keep GS {'+surf' if surfon else 'nosurf'} x2"] = stat(a, P1, 2)
for name, cs in variants.items():
    pc.coarse = cs
    t0 = time.time()
    x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
    print(f"  {name:26s}: {its:4d} its  relres {rr:.1e}  {time.time() - t0:5.1f}s")
