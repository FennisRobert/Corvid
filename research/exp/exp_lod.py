"""B2: decay of the ideal prolongation X = K_ff^{-1} K_fc and LOD-type localized correctors.
For each Whitney edge e, the corrector column is solved on the f-unknowns within `ell` vertex
layers of e (homogeneous Dirichlet outside), and the coarse operator is P^T K P.
usage: exp_lod.py dataset ..."""
import sys, time
import numpy as np
import scipy.sparse as sp
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct, bfs_depth


def vertex_graph(p):
    nv = p.nodes.shape[1]
    E = p.edges
    return sp.csr_matrix((np.ones(2 * E.shape[1]), (np.r_[E[0], E[1]], np.r_[E[1], E[0]])), shape=(nv, nv))


def dof_vertex_incidence(p):
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    en = p.edges.T[np.minimum(p.ent, ne - 1)]
    fn = p.tris.T[np.minimum(p.ent, nt - 1)]
    nodes = np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)
    rows = np.repeat(np.arange(p.n), 3)
    return sp.csr_matrix((np.ones(3 * p.n), (rows, nodes.ravel())), shape=(p.n, p.nodes.shape[1])).astype(bool).astype(float)


def run(name, ells=(1, 2)):
    p = Problem(name)
    K = p.K.tocsr()
    c = p.top.whitney
    f = np.setdiff1d(np.arange(p.n), c)
    floc = -np.ones(p.n, int); floc[f] = np.arange(len(f))
    Kff, Kfc = K[f][:, f].tocsc(), K[f][:, c].tocsc()
    G = vertex_graph(p)
    Inc = dof_vertex_incidence(p)          # DOF x vertex
    IncT = Inc.T.tocsr()
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    print(f"{name}: n={p.n}, Whitney {len(c)}, epl {p.edges_per_wavelength()[0]:.1f}")

    # decay of exact columns
    lu = direct(Kff)
    rng = np.random.default_rng(1)
    sample = rng.choice(len(c), 6, replace=False)
    dist_all = []
    for j in sample:
        x = lu.solve(np.ascontiguousarray(Kfc[:, j].toarray().ravel().astype(complex)))
        e = p.ent[c[j]]
        d = np.full(p.nodes.shape[1], 99); d[p.edges[:, e]] = 0
        frontier = p.edges[:, e]
        for lev in range(1, 12):
            nb = np.unique(G[frontier].indices)
            nb = nb[d[nb] == 99]; d[nb] = lev; frontier = nb
            if len(nb) == 0: break
        dd = (Inc[f].multiply(1)).tocsr()
        dof_d = np.array([d[dd.indices[dd.indptr[i]:dd.indptr[i + 1]]].min() for i in range(len(f))])
        prof = [np.abs(x[dof_d == L]).max() if np.any(dof_d == L) else 0 for L in range(0, 8)]
        dist_all.append(np.array(prof) / prof[0])
    print("  decay max|X(:,e)| by vertex distance 0..7 (mean over 6 edges):", " ".join(f"{v:.1e}" for v in np.mean(dist_all, 0)))

    def localized_P(ell):
        nc = len(c)
        rows, cols, vals = [], [], []
        for j in range(nc):
            e = p.ent[c[j]]
            vs = p.edges[:, e].copy()
            for _ in range(ell - 1):
                vs = np.unique(np.r_[vs, G[vs].indices])
            dofs = np.unique(IncT[vs].indices)            # DOFs touching the vertex set
            loc = floc[dofs]; loc = loc[loc >= 0]
            A = Kff[loc][:, loc].toarray()
            bvec = Kfc[loc, j].toarray().ravel()
            xs = np.linalg.solve(A, bvec)
            rows.append(f[loc]); cols.append(np.full(len(loc), j)); vals.append(-xs)
        Pf = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(p.n, nc))
        Pc = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
        return (Pc + Pf).tocsr()

    for ell in ells:
        t = time.time()
        P = localized_P(ell)
        Kc = (P.T @ K @ P).tocsr()
        clu = direct(Kc)
        fn = lambda r, P=P, clu=clu: P @ clu.solve(np.ascontiguousarray(P.T @ r))
        pc = TwoLevel(K, pats, c, coarse_fn=fn)
        ts = time.time() - t
        x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
        rr = np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b)
        print(f"  LOD ell={ell}: its {its:3d} res {rr:.0e} | P nnz/col {P.nnz/len(c):.0f}, Kc nnz {Kc.nnz} "
              f"(K_cc {K[c][:, c].nnz}), LDL {clu.factor_nnz/1e6:.2f}M | setup {ts:.1f}s [{fmt_hist(hist)}]", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
