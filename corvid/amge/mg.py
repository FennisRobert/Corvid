"""The multigrid that uses the hierarchy: a V-cycle for an H(curl) matrix on the
free Whitney edges, Hiptmair smoothing on every level, a direct coarsest solve.

The smoother needs only matrix-vector products: 4th-kind Chebyshev, Jacobi
scaled, on A and on G^T A G with G the level's exact discrete gradient (the
coarse incidence D0c, which commutes with prolongation by construction).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .coarsen import coarsen
from .whitney import fine_level


def hierarchy(coords, tets, edges, free, eps=None, inv_mu=None, target=32,
              max_coarse=500, max_levels=10):
    """Prolongations between FREE edges and the gradient of every level.
    Returns (Ps, Gs): Ps[l] maps level l+1 to level l, Gs[l] is level l's
    gradient restricted to free edges and to vertices that touch one."""
    L = fine_level(coords, tets, edges, free, eps, inv_mu)
    Ps, Gs = [], [_grad(L)]
    while L.free_edge.sum() > max_coarse and len(Ps) < max_levels - 1:
        Lc, (P1, _, _), _ = coarsen(L, target=target)
        if Lc.free_edge.sum() > 0.8 * L.free_edge.sum():
            break                                    # coarsening has stalled
        Ps.append(P1[L.free_edge][:, Lc.free_edge].tocsr())
        Gs.append(_grad(Lc))
        L = Lc
    return Ps, Gs


def _grad(L):
    G = L.D0.tocsr()[L.free_edge]
    keep = np.flatnonzero(np.diff(G.tocsc().indptr))
    return G[:, keep].tocsr()


def _lambda_max(A, minv, its=30, seed=0):
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    lam = 0.0
    for _ in range(its):
        w = minv(A @ v)
        lam = np.linalg.norm(w)
        v = w / lam
    return lam


class Chebyshev:
    """4th-kind Chebyshev (Phillips & Fischer 2022), Jacobi scaled."""

    def __init__(self, A, order=2, sf=1.1, minv=None):
        self.A = A
        if minv is None:
            d = np.abs(A.diagonal())
            d[d == 0] = 1.0
            d_inv = 1.0 / d
            minv = lambda r: d_inv * r
        self.minv = minv
        self.order = order
        self.lmax = sf * _lambda_max(A, minv)

    def __call__(self, b, y=None):
        A, mi, lm = self.A, self.minv, self.lmax
        if y is None:
            y = np.zeros_like(b); r = b.copy()
        else:
            y = y.copy(); r = b - A @ y
        d = (4.0 / (3.0 * lm)) * mi(r)
        for k in range(1, self.order):
            y += d
            r -= A @ d
            d = ((2 * k - 1) / (2 * k + 3)) * d + ((8 * k + 4) / ((2 * k + 3) * lm)) * mi(r)
        return y + d


class SurfaceJacobi:
    """Chebyshev scaling: exact LU on the surface rows (second-order ABC, whose
    surface derivative terms pointwise scaling cannot handle), Jacobi elsewhere."""

    def __init__(self, A, rows, direct):
        d = np.abs(A.diagonal()); d[d == 0] = 1.0
        self.d_inv, self.rows = 1.0 / d, rows
        self.solve = direct(A[rows][:, rows])

    def __call__(self, r):
        z = self.d_inv * r
        z[self.rows] = self.solve(np.ascontiguousarray(r[self.rows]))
        return z


class _Hiptmair:
    def __init__(self, A, G, order, minv=None):
        self.A, self.G, self.Gt = A, G, G.T.tocsr()
        self.S = Chebyshev(A, order, minv=minv)
        self.SG = Chebyshev((self.Gt @ A @ G).tocsr(), order)

    def pre(self, b):
        y = self.S(b)
        return y + self.G @ self.SG(self.Gt @ (b - self.A @ y))

    def post(self, b, y):
        y = y + self.G @ self.SG(self.Gt @ (b - self.A @ y))
        return self.S(b, y)


def _direct(A):
    try:
        import sparta
    except ImportError:
        from scipy.sparse.linalg import splu
        return splu(A.tocsc()).solve
    return sparta.analyze(A.tocsc()).factorize().solve


class AMGe:
    """V-cycle preconditioner for A on the free fine edges. A may be complex;
    the hierarchy is real, so every level's operator is the Galerkin P^T A P."""

    def __init__(self, A, Ps, Gs, order=2, surface=None, direct=_direct):
        """surface: bool mask of fine rows that carry surface derivative terms;
        a coarse row is a surface row when its prolongation touches one."""
        self.As = [A.tocsr()]
        masks = [None if surface is None else np.asarray(surface, bool)]
        for P in Ps:
            self.As.append((P.T @ self.As[-1] @ P).tocsr())
            if masks[-1] is not None:
                masks.append(np.asarray(abs(P).T @ masks[-1].astype(float)).ravel() > 0)
        self.Ps = Ps
        self.sm = []
        for l, (Al, G) in enumerate(zip(self.As[:-1], Gs[:-1])):
            mv = None
            if surface is not None and masks[l].any():
                mv = SurfaceJacobi(Al, np.flatnonzero(masks[l]), direct)
            self.sm.append(_Hiptmair(Al, G, order, mv))
        self.coarse = direct(self.As[-1])

    def _v(self, l, b):
        if l == len(self.Ps):
            return self.coarse(b)
        A, P, S = self.As[l], self.Ps[l], self.sm[l]
        y = S.pre(b)
        y = y + P @ self._v(l + 1, P.T @ (b - A @ y))
        return S.post(b, y)

    def __call__(self, b):
        return self._v(0, np.asarray(b))
