"""M4: optimized restricted additive Schwarz (ORAS) on vertex stars, from EMerge element matrices.
Local problem on the star of vertex v (all solved DOFs of the tets containing v):
    A_v = sum_{T ni v} (E_T - k^2 B_T) + (ABC surface terms of K on the star)
          + i eta k sum_{T ni v} B_T|outer / h_T           (absorbing proxy on the star boundary)
where 'outer' = DOFs whose entity does not contain v. Restricted gluing: the correction is kept on
the DOFs incident to v only, weighted by 1 / multiplicity. eta = 0: Neumann-extended RAS.
Two-level with the Whitney coarse level; damping from lambda_max as in CORVID; also undamped.
usage: exp_oras.py dataset_with_elements"""
import sys, time
import numpy as np, scipy.sparse as sp
from common import Problem, gmres, fmt_hist, DATA
from proto import TwoLevel, mesh_vertex_patches, lambda_max

name = sys.argv[1]
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
f = np.load(f"{DATA}/{name}/problem_1.npz")
T2F = f["tet_to_field"]; nt = T2F.shape[1]; ndof = int(f["A_shape"][0])
E_T = f["elem_E"].reshape(nt, 20, 20); B_T = f["elem_B"].reshape(nt, 20, 20)
R_T = f["elem_rows"].reshape(nt, 20, 20)[:, :, 0]          # DOF id of each local row
loc = -np.ones(ndof, int); loc[p.sid] = np.arange(p.n)
k = p.k0
tets = f["mesh_tets"]; X = p.nodes
hT = np.mean([np.linalg.norm(X[:, tets[a]] - X[:, tets[b]], axis=0) for a in range(4) for b in range(a + 1, 4)], axis=0)
Kvol = sp.csr_matrix(((E_T - k * k * B_T).ravel(), (np.repeat(R_T, 20, axis=1).ravel(), np.tile(R_T, (1, 20)).ravel())), shape=(ndof, ndof))
Kvol = Kvol[p.sid][:, p.sid]
Ksurf = (K - Kvol).tocsr()
pats_v = mesh_vertex_patches(p)                               # incident DOFs per mesh vertex
mult = np.zeros(p.n)
for x in pats_v: mult[x] += 1
v2t = sp.csr_matrix((np.ones(tets.size), (tets.ravel(), np.tile(np.arange(nt), 4))), shape=(X.shape[1], nt))


def build(eta, alpha=0.0):
    blocks = []
    for v in range(X.shape[1]):
        inc = pats_v[v]
        if len(inc) == 0: continue
        ts = v2t[v].indices
        dofs = loc[R_T[ts]]                                    # (nts, 20) solved ids or -1
        dom = np.unique(dofs[dofs >= 0])
        pos = -np.ones(p.n, int); pos[dom] = np.arange(len(dom))
        A = np.zeros((len(dom), len(dom)), complex)
        for t_i, t in enumerate(ts):
            d = dofs[t_i]; ok = d >= 0
            ii = pos[d[ok]]
            Kt = (E_T[t] - k * k * B_T[t])[np.ix_(ok, ok)]
            A[np.ix_(ii, ii)] += Kt
            if eta or alpha:
                outer = ~np.isin(d[ok], inc)
                Bo = B_T[t][np.ix_(ok, ok)] * np.outer(outer, outer) / hT[t]
                A[np.ix_(ii, ii)] += (1j * eta * k + alpha / hT[t]) * Bo
        A += Ksurf[dom][:, dom].toarray()
        keep = pos[inc]                                       # restricted: incident DOFs only
        blocks.append((dom, keep, inc, np.linalg.inv(A)))
    return blocks


def make_B(blocks, restricted=True):
    def B(r):
        z = np.zeros_like(r)
        for dom, keep, inc, Ainv in blocks:
            zl = Ainv @ r[dom]
            if restricted:
                z[inc] += zl[keep] / mult[inc]
            else:
                z[dom] += zl
        return z
    return B


pats = [x for x in pats_v if len(x)]
base = TwoLevel(K, pats, c)
def two_level(B, om):
    def apply(r):
        r = np.asarray(r, complex)
        y = om * B(r)
        res = r - K @ y
        y[c] += base.clu.solve(np.ascontiguousarray(res[c]))
        return y + om * B(r - K @ y)
    return apply

x, its, hist = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
print(f"{name}: n={p.n}; CORVID (additive vertex blocks, Dirichlet): its {its}", flush=True)
for eta, alpha in ((0, 1.0), (0, 4.0), (0, 16.0), (0, 64.0), (1.0, 16.0)):
    t = time.time()
    bl = build(eta, alpha)
    B = make_B(bl)
    lm = lambda_max(K, B, p.n)
    om = 4 / (3 * 1.1 * lm)
    x, its, hist = gmres(lambda v: K @ v, two_level(B, om), p.b, maxiter=150)
    print(f"  ORAS real Robin alpha {alpha:5.1f}/h, eta {eta}: omega {om:.2f}: its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e}  {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
