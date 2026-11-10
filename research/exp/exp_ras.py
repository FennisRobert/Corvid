"""G1: how the vertex-block solves are combined: additive (CORVID), restricted additive (partition of
unity in the overlap), multiplicative (block Gauss-Seidel over vertex blocks, forward pre /
backward post). Whitney exact coarse in all cases. usage: exp_ras.py dataset ..."""
import sys, time
import numpy as np
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, lambda_max


class Smoothed:
    def __init__(self, K, pats, c, mode, omega=None):
        self.K, self.c, self.mode = K, c, mode
        self.pats = pats
        self.inv = [np.linalg.inv(K[x][:, x].toarray()) for x in pats]
        cnt = np.zeros(K.shape[0])
        for x in pats: cnt[x] += 1
        self.w = [1.0 / cnt[x] for x in pats]
        self.tl = TwoLevel(K, pats, c)                    # for the coarse factor and the AS omega
        self.omega = omega if omega is not None else (self.tl.c0 if mode == "as" else 1.0)
        if mode == "ras" and omega == "auto":
            self.omega = 4 / (3 * 1.1 * lambda_max(K, self.ras, K.shape[0]))

    def ras(self, r):
        z = np.zeros_like(r)
        for x, A, w in zip(self.pats, self.inv, self.w):
            z[x] += w * (A @ r[x])
        return z

    def as_(self, r):
        z = np.zeros_like(r)
        for x, A in zip(self.pats, self.inv):
            z[x] += A @ r[x]
        return z

    def smooth(self, b, y, forward=True):
        K = self.K
        if self.mode == "gs":
            order = range(len(self.pats)) if forward else reversed(range(len(self.pats)))
            for j in order:
                x = self.pats[j]
                y[x] += self.omega * (self.inv[j] @ (b[x] - K[x] @ y))
            return y
        f = self.ras if self.mode == "ras" else self.as_
        return y + self.omega * f(b - K @ y)

    def apply(self, r):
        r = np.asarray(r, complex)
        y = self.smooth(r, np.zeros_like(r), True)
        res = r - self.K @ y
        y[self.c] += self.tl.clu.solve(np.ascontiguousarray(res[self.c]))
        return self.smooth(r, y, False)


for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    print(f"{name}: n={p.n}, epl {p.edges_per_wavelength()[0]:.1f}, PML rows {p.pml.mean():.0%}")
    for mode, om in (("as", None), ("ras", 1.0), ("ras", "auto"), ("gs", 1.0), ("gs", 0.7)):
        t = time.time()
        pc = Smoothed(K, pats, c, mode, om)
        x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
        print(f"  {mode:4s} omega {pc.omega:.3f}: its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e}"
              f"  {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
