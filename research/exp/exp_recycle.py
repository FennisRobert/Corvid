"""J1: recycling across a frequency sweep. After each solve, harmonic Ritz vectors of K M with the
smallest |theta| are mapped to error space (z = M v = Z g) and added to the coarse space of the
next frequency (Whitney + span U, Galerkin). No extra preconditioner applications are spent.
usage: exp_recycle.py sweep_dataset nfreq kkeep"""
import sys
import numpy as np, scipy.sparse as sp, scipy.linalg as dla
from common import Problem
from proto import TwoLevel, mesh_vertex_patches


def gmres_basis(matvec, prec, b, rtol=1e-6, maxiter=150):
    n = len(b); m = maxiter; bn = np.linalg.norm(b)
    V = np.zeros((m + 1, n), complex); Z = np.zeros((m, n), complex); H = np.zeros((m + 1, m), complex)
    V[0] = b / bn; e1 = np.zeros(m + 1, complex); e1[0] = bn
    for j in range(m):
        Z[j] = prec(V[j]); w = matvec(Z[j])
        for _ in range(2):
            cc = V[:j + 1].conj() @ w; H[:j + 1, j] += cc; w = w - V[:j + 1].T @ cc
        H[j + 1, j] = np.linalg.norm(w); V[j + 1] = w / H[j + 1, j]
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1[:j + 2], rcond=None)
        if np.linalg.norm(e1[:j + 2] - H[:j + 2, :j + 1] @ y) / bn < rtol:
            break
    k = j + 1
    return Z[:k].T @ y, k, H[:k + 1, :k], Z[:k]


def harmonic_ritz(H, Z, nkeep):
    k = H.shape[1]
    Hk = H[:k, :k]
    f = np.linalg.solve(Hk.conj().T, np.eye(k)[:, -1]) * abs(H[k, k - 1]) ** 2
    th, G = np.linalg.eig(Hk + np.outer(f, np.eye(k)[-1]))
    o = np.argsort(abs(th))[:nkeep]
    return Z.T @ G[:, o]


name, nf, kkeep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 2 * kkeep
exact = len(sys.argv) > 5 and sys.argv[5] == 'auto'
U = None
print(f"{name}: recycle {kkeep} harmonic Ritz vectors per frequency (cap {cap})")
for i in range(1, nf + 1):
    p = Problem(name, pid=i); K = p.K.tocsr(); c = p.top.whitney
    if exact:
        import corvid
        top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
        sv = corvid.Solver(top); sv.k0 = p.k0
        m = sv._exact_rows(K).copy(); m[c] = True; c = np.flatnonzero(m)
    nc = len(c)
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    if exact:
        mm = np.zeros(p.n, bool); mm[c] = True
        pats = [x for x in pats if not mm[x].all()]
    base = TwoLevel(K, pats, c)
    _, its0, _, _ = gmres_basis(lambda v: K @ v, base.apply, p.b)
    if U is None:
        pc = base
    else:
        # bordered coarse level [R, U]: sparse factor of R^T K R (base.clu) + m x m Schur complement
        Uo = U.copy(); Uo[c] = 0                       # complement of range(R)
        Uo, _ = np.linalg.qr(Uo)
        KU = np.column_stack([K @ Uo[:, j] for j in range(Uo.shape[1])])
        Bm = KU[c]                                     # R^T K U
        Wm = np.column_stack([base.clu.solve(np.ascontiguousarray(Bm[:, j])) for j in range(Bm.shape[1])])
        Sm = Uo.T @ KU - Bm.T @ Wm
        Sf = dla.lu_factor(Sm)
        def cfn(r, Uo=Uo, Wm=Wm, Bm=Bm, Sf=Sf, clu=base.clu, c=c):
            y = clu.solve(np.ascontiguousarray(r[c]))
            z = dla.lu_solve(Sf, Uo.T @ r - Bm.T @ y)
            e = Uo @ z
            e[c] += y - Wm @ z
            return e
        pc = TwoLevel(K, pats, c, coarse_fn=cfn)
    x, its, H, Z = gmres_basis(lambda v: K @ v, pc.apply, p.b)
    rr = np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b)
    Unew = harmonic_ritz(H, Z, kkeep)
    U = Unew if U is None else np.column_stack([Unew, U])[:, :cap]
    print(f"  f{i} (k0 = {p.k0:.1f}): CORVID 0.2.0 {its0:3d} its | recycled coarse (+{0 if i == 1 else U.shape[1]}) {its:3d} its, res {rr:.0e}", flush=True)
