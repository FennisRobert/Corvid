"""B1: Schur-complement / ideal-prolongation coarse levels against CORVID's injected Whitney level.
usage: exp_schur.py dataset ..."""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct


def entity_blockdiag_inv(Kff, kind_f, ent_f):
    """Inverse of the entity-block diagonal of K_ff: one block per edge (2nd edge DOF) and per face (2 DOFs)."""
    key = kind_f.astype(np.int64) * (10 ** 9) + ent_f
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    rows, cols, vals = [], [], []
    Kd = Kff.tocsr()
    for a, b in zip(starts, ends):
        idx = order[a:b]
        blk = Kd[idx][:, idx].toarray()
        inv = np.linalg.inv(blk)
        rr, cc = np.meshgrid(idx, idx, indexing="ij")
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(inv.ravel())
    m = Kff.shape[0]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))


def run(name):
    p = Problem(name)
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    K = p.K.tocsr()
    c = p.top.whitney
    f = np.setdiff1d(np.arange(p.n), c)
    Kcc, Kcf, Kfc, Kff = K[c][:, c], K[c][:, f], K[f][:, c], K[f][:, f]
    e = p.edges_per_wavelength()
    print(f"{name}: n={p.n} (Whitney {len(c)}, complement {len(f)}), edges/lambda median {e[0]:.1f} max-edge {e[1]:.1f}")

    def prolong(X):           # P = [I; -X] as an n x nc sparse matrix
        nc = len(c)
        Pc = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
        Xs = sp.csr_matrix(X)
        Xc = Xs.tocoo()
        Pf = sp.csr_matrix((-Xc.data, (f[Xc.row], Xc.col)), shape=(p.n, nc))
        return (Pc + Pf).tocsr()

    def galerkin_fn(P):
        Kc = (P.T @ K @ P).tocsr()
        lu = direct(Kc)
        fn = lambda r: P @ lu.solve(np.ascontiguousarray(P.T @ r))
        return fn, lu.factor_nnz, Kc.nnz

    configs = []
    configs.append(("injection, K_cc (CORVID 0.2.0)", None, None))
    # entity-block-diagonal K_ff^{-1}
    D = entity_blockdiag_inv(Kff, p.kind[f], p.ent[f])
    P1 = prolong(D @ Kfc)
    configs.append(("P = [I; -D_ent^-1 K_fc], Galerkin", *galerkin_fn(P1)[:1], ))
    # diagonal K_ff
    Dd = sp.diags(1.0 / Kff.diagonal())
    P0 = prolong(Dd @ Kfc)
    configs.append(("P = [I; -diag^-1 K_fc], Galerkin", galerkin_fn(P0)[0]))
    # exact ideal (ceiling), only for small problems
    if len(f) <= 12000:
        lu_ff = direct(Kff)
        X = np.column_stack([lu_ff.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex))) for j in range(len(c))])
        S = Kcc.toarray() - Kcf @ X
        Slu = sla.lu_factor(S)
        Pd = np.zeros((p.n, len(c)), complex); Pd[c] = np.eye(len(c)); Pd[f] = -X
        configs.append(("ideal P, exact Schur (ceiling)", lambda r: Pd @ sla.lu_solve(Slu, Pd.T @ r)))
        configs.append(("injection, exact Schur S", lambda r: _inj(r, c, Slu, p.n)))
    for item in configs:
        label, fn = item[0], item[1]
        t = time.time()
        pc = TwoLevel(K, pats, c, coarse_fn=fn)
        ts = time.time() - t
        t = time.time()
        x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
        rr = np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b)
        print(f"  {label:36s} its {its:3d} res {rr:.0e} | setup {ts:.1f}s solve {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
    for label, P in (("D_ent", P1), ("diag", P0)):
        Kc = (P.T @ K @ P).tocsr()
        print(f"  coarse operator {label}: nnz {Kc.nnz} vs K_cc {Kcc.nnz} (x{Kc.nnz / Kcc.nnz:.2f}), LDL nnz {direct(Kc).factor_nnz/1e6:.2f}M vs {direct(Kcc).factor_nnz/1e6:.2f}M")


def _inj(r, c, Slu, n):
    y = np.zeros(n, complex)
    y[c] = sla.lu_solve(Slu, r[c])
    return y


for name in sys.argv[1:]:
    run(name)
