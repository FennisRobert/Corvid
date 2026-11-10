"""Two-level CORVID-style preconditioner assembled from the Rust research kernels
(corvid._kernels.Rx*: SPARTA CPU LDL^T in double precision, dense-block Schwarz, block sweeps).

    M r:  y  = c0 B r                         (order-1 Chebyshev = damped additive vertex Schwarz)
          y  = sweep_forward(r, y)            (optional block Gauss-Seidel over extra blocks)
          y += R Kc^{-1} R^T (r - K y)        (exact coarse solve)
          y  = sweep_backward(r, y)
          y += c0 B (r - K y)
"""
from collections import deque
import numpy as np
import scipy.sparse as sp
from corvid import _kernels


def kcsr(K):
    K = sp.csr_matrix(K); K.sort_indices()
    return (K.indptr.astype(np.int32), K.indices.astype(np.int32), np.ascontiguousarray(K.data, dtype=np.complex128),
            K.shape[0], K.shape[1])


def pack(blocks):
    ptr = np.zeros(len(blocks) + 1, np.int64)
    ptr[1:] = np.cumsum([len(b) for b in blocks])
    idx = np.concatenate(blocks).astype(np.int64) if blocks else np.zeros(0, np.int64)
    return ptr, idx


def direct(M):
    return _kernels.RxDirect(kcsr(M))


def lambda_max(K, B, n, its=300, tol=1e-4):
    v = np.random.default_rng(0).standard_normal(n) + 0j
    v /= np.linalg.norm(v)
    lam = 0
    for _ in range(its):
        w = B(K @ v)
        l = np.linalg.norm(w)
        v = w / l
        if lam and abs(l - lam) < tol * l:
            break
        lam = l
    return l


class TwoLevel:
    def __init__(self, K, patches, coarse, sweep_blocks=None, coarse_op=None, sf=1.1, coarse_fn=None):
        self.K = sp.csr_matrix(K)
        n = self.K.shape[0]
        kk = kcsr(self.K)
        self.sch = _kernels.RxSchwarz(kk, *pack([np.asarray(p, np.int64) for p in patches]))
        self.c0 = 4.0 / (3.0 * sf * lambda_max(self.K, lambda v: self.sch.apply(np.ascontiguousarray(v)), n))
        self.coarse = np.asarray(coarse)
        self.coarse_fn = coarse_fn
        self.clu = None if coarse_fn else direct(self.K[self.coarse][:, self.coarse] if coarse_op is None else coarse_op)
        self.sw = _kernels.RxSweep(kk, *pack(sweep_blocks)) if sweep_blocks else None
        self.coarse_nnz = self.clu.factor_nnz if self.clu else 0
        self.sweep_nnz = self.sw.factor_nnz if self.sw else 0
        self.schwarz_mb = self.sch.bytes / 1e6

    def apply(self, r):
        r = np.ascontiguousarray(r, dtype=complex)
        K = self.K
        y = self.c0 * self.sch.apply(r)
        if self.sw:
            y = self.sw.sweep(r, y, True)
        res = r - K @ y
        if self.coarse_fn:
            y += self.coarse_fn(res)
        else:
            y[self.coarse] += self.clu.solve(np.ascontiguousarray(res[self.coarse]))
        if self.sw:
            y = self.sw.sweep(r, y, False)
        return y + self.c0 * self.sch.apply(np.ascontiguousarray(r - K @ y))


def mesh_vertex_patches(p):
    """DOFs touching each mesh vertex (corvid's vertex blocks, indexed by mesh vertex)."""
    ne = p.edges.shape[1]
    en = p.edges.T[np.minimum(p.ent, ne - 1)]
    fn = p.tris.T[np.minimum(p.ent, p.tris.shape[1] - 1)]
    nodes = np.where(p.kind[:, None] < 2, np.c_[en, -np.ones(p.n, int)], fn)
    rows = np.repeat(np.arange(p.n), 3)
    cols = nodes.ravel()
    ok = cols >= 0
    nv = p.nodes.shape[1]
    inc = sp.csc_matrix((np.ones(ok.sum()), (rows[ok], cols[ok])), shape=(p.n, nv))
    inc.sum_duplicates(); inc.sort_indices()
    return [inc.indices[inc.indptr[v]:inc.indptr[v + 1]].astype(np.int64) for v in range(nv)]


def bfs_depth(nv, edges, is_pml):
    """Graph distance of each vertex from the non-PML vertices (0 outside the PML)."""
    A = sp.csr_matrix((np.ones(edges.shape[1] * 2), (np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]])), shape=(nv, nv))
    d = np.full(nv, -1)
    q = deque(np.flatnonzero(~is_pml))
    d[~is_pml] = 0
    while q:
        v = q.popleft()
        for w in A.indices[A.indptr[v]:A.indptr[v + 1]]:
            if d[w] < 0:
                d[w] = d[v] + 1; q.append(w)
    d[d < 0] = 0
    return d
