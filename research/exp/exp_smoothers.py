"""M2/M3: smoother variants inside CORVID's two-level method (coarse = Whitney, plus the deep PML
rows of corvid 0.2.1 when a PML is present). Work = block sweeps per preconditioner application.
  AS1     damped additive, order 1 (CORVID)               2 sweeps
  AS2     additive, Chebyshev order 2                     4 sweeps
  ASw     additive order 1, separate damping on Whitney and complement rows
  GS1     multiplicative: forward pre, backward post     2 sweeps
  GS2     two symmetric sweeps pre and post               4 sweeps
usage: exp_smoothers.py dataset ..."""
import sys, time
import numpy as np
import corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, lambda_max


class Smoother:
    def __init__(self, K, pats, coarse, mode, wc=1.0, wf=1.0):
        self.K, self.mode, self.pats = K, mode, pats
        self.inv = [np.linalg.inv(K[x][:, x].toarray()) for x in pats]
        self.rows = [K[x] for x in pats]
        self.tl = TwoLevel(K, pats, coarse)
        self.c = coarse
        self.om = self.tl.c0
        self.lmax = 4 / (3 * self.om)
        self.wvec = np.full(K.shape[0], wf); self.wvec[coarse] = wc

    def B(self, r):
        z = np.zeros_like(r)
        for x, A in zip(self.pats, self.inv): z[x] += A @ r[x]
        return z

    def gs(self, b, y, forward):
        order = range(len(self.pats)) if forward else reversed(range(len(self.pats)))
        for j in order:
            x = self.pats[j]
            y[x] += self.inv[j] @ (b[x] - self.rows[j] @ y)
        return y

    def cheb(self, b, y, order):
        lm = self.lmax
        r = b - self.K @ y
        d = (4 / (3 * lm)) * self.B(r)
        for k in range(1, order):
            y = y + d
            r = r - self.K @ d
            c1, c2 = (2 * k - 1) / (2 * k + 3), (8 * k + 4) / ((2 * k + 3) * lm)
            d = d * c1 + c2 * self.B(r)
        return y + d

    def smooth(self, b, y, pre):
        m = self.mode
        if m == "AS1": return self.cheb(b, y, 1)
        if m == "AS2": return self.cheb(b, y, 2)
        if m == "ASw": return y + self.om * self.wvec * self.B(b - self.K @ y)
        if m == "GS1": return self.gs(b, y, pre)
        if m == "GS2":
            y = self.gs(b, y, pre); return self.gs(b, y, not pre)

    def apply(self, r):
        r = np.asarray(r, complex)
        y = self.smooth(r, np.zeros_like(r), True)
        res = r - self.K @ y
        y[self.c] += self.tl.clu.solve(np.ascontiguousarray(res[self.c]))
        return self.smooth(r, y, False)


if __name__ == "__main__":
    work = {"AS1": 2, "AS2": 4, "ASw": 2, "GS1": 2, "GS2": 4}
    for name in sys.argv[1:]:
        p = Problem(name); K = p.K.tocsr()
        s = corvid.Solver(p.top)
        m = np.zeros(p.n, bool); m[p.top.whitney] = True
        ex = s._exact_rows(K)
        if ex is not None: m |= ex
        coarse = np.flatnonzero(m)
        pats = [x for x in mesh_vertex_patches(p) if len(x)]
        print(f"{name}: n={p.n}, coarse rows {len(coarse)}", flush=True)
        configs = [("AS1", {}), ("AS2", {}), ("GS1", {}), ("GS2", {}),
                   ("ASw", dict(wc=1.0, wf=1.5)), ("ASw", dict(wc=0.7, wf=1.3)), ("ASw", dict(wc=1.3, wf=1.0))]
        for mode, kw in configs:
            t = time.time()
            sm = Smoother(K, pats, coarse, mode, **kw)
            x, its, hist = gmres(lambda v: K @ v, sm.apply, p.b, maxiter=150)
            print(f"  {mode} {str(kw):24s} its {its:3d}  block sweeps {its * work[mode]:4d}  "
                  f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e}  {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
