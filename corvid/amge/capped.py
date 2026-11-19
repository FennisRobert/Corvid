"""Wavelength-capped agglomeration: the AMGe hierarchy of the Whitney level, coarsened only while every
aggregate stays small against the local wavelength.

An aggregate a may grow while  k_loc(a) * H(a) <= kappa,  with H(a) the diagonal of its bounding box and
k_loc(a) = k0 * max over its tets of the refractive index sqrt(er ur). Tets that are already at the limit
stay single (the fine Whitney space is reproduced there exactly), so only regions meshed finer than the
wave needs coarsen. Everything else (ball-shaped aggregates that never cross PEC, the exact de Rham
coarse spaces) is `coarsen()` unchanged.

Used by Solver(coarsen_kh=kappa); see that docstring.
"""
from __future__ import annotations

import time
import warnings
from collections import defaultdict

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .agglomerate import _Topo, _cell, _glue_ok, _rows
from .coarsen import coarsen
from .mg import _grad
from .whitney import fine_level

TARGET = 32          # elements per aggregate at most (the prototype's default)
MAX_LEVELS = 10
STALL = 0.85         # stop when a level keeps more than this share of the free edges


def _grow(T: _Topo, bmin, bmax, kl, kappa, target):
    """Region growing as agglomerate.grow, refusing any element that would push the aggregate's
    k_loc * bounding-box diagonal past kappa."""
    nel = T.L.n_elem
    cells = [_cell(T, t) for t in range(nel)]
    part = -np.ones(nel, int)
    k = 0
    for seed in range(nel):
        if part[seed] >= 0:
            continue
        part[seed] = k
        S = tuple(set(x) for x in cells[seed])
        lo, hi, km, m = bmin[seed].copy(), bmax[seed].copy(), kl[seed], 1
        while m < target:
            cand = defaultdict(int)
            for f in S[2]:
                for t in _rows(T.D2T, f)[0]:
                    if part[t] < 0:
                        cand[t] += 1
            added = False
            for t, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
                lo2, hi2, km2 = np.minimum(lo, bmin[t]), np.maximum(hi, bmax[t]), max(km, kl[t])
                if km2 * np.linalg.norm(hi2 - lo2) > kappa:
                    continue
                if _glue_ok(T, cells[t], S):
                    part[t] = k
                    m += 1
                    added = True
                    lo, hi, km = lo2, hi2, km2
                    for a, c in zip(S, cells[t]):
                        a |= c
                    break
            if not added:
                break
        k += 1
    return part


def build_hierarchy(coords, tets, edges, free, k_tet, kappa, target=TARGET, max_levels=MAX_LEVELS,
                    backend="rust", threads=None):
    """Prolongations and gradients of the wavelength-capped hierarchy.

    coords (V, 3), tets (T, 4), edges (E, 2) (Whitney orientation edges[e, 0] -> edges[e, 1]),
    free (E,) bool: the solved Whitney edges, k_tet (T,): the local wavenumber per tet.
    Returns (Ps, Gs, sizes, seconds): Ps[l] maps level l+1 to level l on free edges (level 0 in
    increasing edge order), Gs[l] is level l's gradient on its free edges (one per smoothed level),
    sizes the free edges per level. Ps is empty when nothing coarsens.

    backend "rust" (corvid._kernels.amge_hierarchy, threaded) or "python" (this module, the reference)."""
    if backend == "rust":
        return _build_rust(coords, tets, edges, free, k_tet, kappa, target, max_levels, threads)
    if backend != "python":
        raise ValueError("backend must be 'rust' or 'python'")
    t0 = time.perf_counter()
    X = np.asarray(coords, float)
    tets = np.asarray(tets, np.int64)
    with warnings.catch_warnings():
        # thin, flat aggregates (e.g. in a PCB substrate) give badly scaled local saddle points; the
        # de Rham identities still hold, so SciPy's conditioning warnings are expected here
        warnings.simplefilter("ignore", sla.LinAlgWarning)
        L = fine_level(X, tets, edges, free)
        bmin, bmax, kl = X[tets].min(1), X[tets].max(1), np.asarray(k_tet, float).copy()
        Ps, Gs, sizes = [], [_grad(L)], [int(L.free_edge.sum())]
        while len(Ps) < max_levels - 1:
            part = _grow(_Topo(L), bmin, bmax, kl, kappa, target)
            if part.max() + 1 > 0.9 * L.n_elem:
                break
            Lc, P, C = coarsen(L, part=part)
            if Lc.free_edge.sum() == 0 or Lc.free_edge.sum() > STALL * L.free_edge.sum():
                break
            Ps.append(P[0][L.free_edge][:, Lc.free_edge].tocsr())
            Gs.append(_grad(Lc))
            sizes.append(int(Lc.free_edge.sum()))
            bmin = np.array([bmin[g].min(0) for g in C.ae_elems])
            bmax = np.array([bmax[g].max(0) for g in C.ae_elems])
            kl = np.array([kl[g].max() for g in C.ae_elems])
            L = Lc
    return Ps, Gs[:len(Ps)], sizes, time.perf_counter() - t0


def _build_rust(coords, tets, edges, free, k_tet, kappa, target, max_levels, threads, check=False):
    from .. import _kernels
    t0 = time.perf_counter()
    ps, gs, sizes, checks = _kernels.amge_hierarchy(
        np.ascontiguousarray(coords, dtype=np.float64), np.ascontiguousarray(tets, dtype=np.int64),
        np.ascontiguousarray(edges, dtype=np.int64), np.ascontiguousarray(free, dtype=bool),
        np.ascontiguousarray(k_tet, dtype=np.float64), float(kappa), target=int(target), max_levels=int(max_levels),
        check=check, threads=threads)
    to = lambda m: sp.csr_matrix((m[2], m[1], m[0]), shape=(m[3], m[4]))
    out = [to(m) for m in ps], [to(m) for m in gs], list(sizes), time.perf_counter() - t0
    return (*out, checks) if check else out


def galerkin_levels(A, Ps, Gs):
    """Per smoothed level (A_l, P_l, P_l^T, G_l, G_l^T, G_l^T A_l G_l) and the coarsest operator,
    A_{l+1} = P_l^T A_l P_l, all CSR with sorted indices."""
    def csr(M):
        M = sp.csr_matrix(M)
        M.sort_indices()
        return M
    levels, Al = [], csr(A)
    for P, G in zip(Ps, Gs):
        P, G = csr(P), csr(G)
        levels.append((Al, P, csr(P.T), G, csr(G.T), csr(G.T @ Al @ G)))
        Al = csr(P.T @ Al @ P)
    return levels, Al
