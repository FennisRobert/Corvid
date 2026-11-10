"""L1: local spectral coarse space (GenEO / local-Hodge style) on top of the Whitney level.

  1. vertex aggregates (k-means on vertex positions, ~`agg` vertices each)
  2. local domain = DOFs touching the aggregate's vertices or their 1-ring
  3. local problem: smallest |lambda| of D^-1/2 K_a D^-1/2, D = |diag K| (locally algebraically smooth)
  4. smooth partition of unity (vertex weights, 1 inside, 1/2 on the ring, normalised), DOF weight = mean
  5. basis reduction: zero the Whitney rows, pivoted QR, drop |R_ii| < tol |R_11|
  6. coarse = Whitney (injection) + span, Galerkin, bordered sparse solve

usage: exp_spectral.py dataset [dataset ...]"""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.linalg as dla
from scipy.cluster.vq import kmeans2
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches


def dof_vertices(p):
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    en = p.edges.T[np.minimum(p.ent, ne - 1)]
    fn = p.tris.T[np.minimum(p.ent, nt - 1)]
    return np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)          # (n, 3), edges repeat a node


def local_spectral(p, K, agg, nvec, seed=1):
    pats = mesh_vertex_patches(p)
    nv = len(pats)
    used = np.flatnonzero([len(x) > 0 for x in pats])
    nagg = max(1, len(used) // agg)
    _, lab = kmeans2(p.nodes[:, used].T, nagg, seed=seed, minit="++")
    vlab = -np.ones(nv, int); vlab[used] = lab
    E = p.edges
    A = sp.csr_matrix((np.ones(2 * E.shape[1]), (np.r_[E[0], E[1]], np.r_[E[1], E[0]])), shape=(nv, nv))
    dv = dof_vertices(p)
    Kd = np.abs(K.diagonal())
    # vertex partition of unity
    W = np.zeros((nv, nagg))
    for a in range(nagg):
        inside = np.flatnonzero(vlab == a)
        ring = np.setdiff1d(np.unique(A[inside].indices), inside)
        W[inside, a] = 1.0; W[ring, a] += 0.5
    W /= np.maximum(W.sum(1, keepdims=True), 1e-300)
    wdof = W[dv].mean(1)                                                   # (n, nagg)
    cols, nloc = [], []
    for a in range(nagg):
        dom = np.flatnonzero(wdof[:, a] > 0)
        if len(dom) == 0:
            continue
        s = 1 / np.sqrt(Kd[dom])
        Ka = (K[dom][:, dom].toarray() * s[:, None]) * s[None, :]
        lam, V = np.linalg.eig(Ka)
        o = np.argsort(np.abs(lam))[:nvec]
        Va = V[:, o] * s[:, None]
        Z = np.zeros((p.n, len(o)), complex)
        Z[dom] = Va * wdof[dom, a][:, None]
        cols.append(Z); nloc.append(len(dom))
    return np.concatenate(cols, axis=1), nagg, int(np.mean(nloc))


def bordered(K, pats, c, U, tol=1e-6):
    base = TwoLevel(K, pats, c)
    Uo = U.copy(); Uo[c] = 0
    Q, Rr, piv = dla.qr(Uo, mode="economic", pivoting=True)
    d = np.abs(np.diag(Rr)); keep = d > tol * d[0]
    Uo = Q[:, keep]
    KU = K @ Uo
    Bm = KU[c]
    Wm = np.column_stack([base.clu.solve(np.ascontiguousarray(Bm[:, j])) for j in range(Bm.shape[1])])
    Sf = dla.lu_factor(Uo.T @ KU - Bm.T @ Wm)
    def cfn(r):
        y = base.clu.solve(np.ascontiguousarray(r[c]))
        z = dla.lu_solve(Sf, Uo.T @ r - Bm.T @ y)
        e = Uo @ z; e[c] += y - Wm @ z
        return e
    return TwoLevel(K, pats, c, coarse_fn=cfn), Uo.shape[1], base


if __name__ == "__main__":
    for name in sys.argv[1:]:
        p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
        if p.pml.any():
            m = p.pml.copy(); m[c] = True; c = np.flatnonzero(m)   # keep 0.2.0's exact PML level
        pats = [x for x in mesh_vertex_patches(p) if len(x)]
        base = TwoLevel(K, pats, c)
        x, its0, _ = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
        print(f"{name}: n={p.n}, coarse rows {len(c)}, CORVID 0.2.0 {its0} its", flush=True)
        for agg, nvec in ((8, 4), (8, 8), (16, 8), (16, 16), (32, 16), (32, 32)):
            t = time.time()
            U, nagg, nloc = local_spectral(p, K, agg, nvec)
            pc, m, _ = bordered(K, pats, c, U)
            ts = time.time() - t
            x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
            print(f"  aggregates of ~{agg:2d} vertices ({nagg:4d}, local size ~{nloc:4d}), {nvec:2d} vectors each -> +{m:5d} after QR"
                  f" ({m / p.n:.1%} of n): its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} setup {ts:.1f}s [{fmt_hist(hist)}]", flush=True)
