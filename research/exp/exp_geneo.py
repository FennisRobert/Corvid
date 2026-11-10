"""L2: GenEO-type coarse space with true Neumann local matrices (element matrices exported by
gen_box.py elem=1; air-only models).

  aggregate a = a few mesh vertices; elements T_a = tets touching them; DOFs Omega_a = their DOFs
  Neumann matrix   K^N_a = sum_{T in T_a} (E_T - k^2 B_T)   (+ surface terms of K restricted to Omega_a)
  eigenproblem     K^N_a v = lambda D_a A+_a D_a v,  A+ = E + k^2 B assembled (Dirichlet restriction)
  D_a              partition of unity by multiplicity;  coarse vectors D_a v for the smallest |lambda|
  coarse           Whitney (injection) + span, pivoted QR, Galerkin, bordered solve
usage: exp_geneo.py dataset"""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.linalg as dla
from scipy.cluster.vq import kmeans2
from common import Problem, gmres, fmt_hist, DATA
from proto import TwoLevel, mesh_vertex_patches
from exp_spectral import bordered

name = sys.argv[1]
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
f = np.load(f"{DATA}/{name}/problem_1.npz")
T2F = f["tet_to_field"]; nt = T2F.shape[1]; ndof = int(f["A_shape"][0])
E_T = f["elem_E"].reshape(nt, 20, 20) if f["elem_E"].size == nt * 400 else None
B_T = f["elem_B"].reshape(nt, 20, 20)
rows = f["elem_rows"].reshape(nt, 20, 20); cols = f["elem_cols"].reshape(nt, 20, 20)
k2 = p.k0 ** 2
loc = -np.ones(ndof, int); loc[p.sid] = np.arange(p.n)          # global -> solved index
# assembled volume operator vs K: surface terms = K - volume
def assemble(Te, scale=1.0):
    r, cc = loc[rows.ravel()], loc[cols.ravel()]
    ok = (r >= 0) & (cc >= 0)
    return sp.csr_matrix((Te.ravel()[ok] * scale, (r[ok], cc[ok])), shape=(p.n, p.n))
Evol, Bvol = assemble(E_T), assemble(B_T)
Kvol = Evol - k2 * Bvol
Ksurf = (K - Kvol).tocsr(); Ksurf.eliminate_zeros()
rel = abs(Ksurf).sum() / abs(K).sum()
print(f"{name}: n={p.n}; |K - sum_T K_T| / |K| = {rel:.2e}; surface rows {np.mean(np.diff(Ksurf.indptr) > 0):.1%}", flush=True)
Aplus = (Evol + k2 * Bvol).tocsr()
pats = [x for x in mesh_vertex_patches(p) if len(x)]
base = TwoLevel(K, pats, c)
x, its0, _ = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
print(f"  CORVID 0.2.0: {its0} its", flush=True)

tets = f["mesh_tets"]; nv = p.nodes.shape[1]
used = np.unique(tets)
v2t = sp.csr_matrix((np.ones(tets.size), (tets.ravel(), np.tile(np.arange(nt), 4))), shape=(nv, nt))


def geneo(agg, nvec, tau=None):
    nagg = max(1, len(used) // agg)
    _, lab = kmeans2(p.nodes[:, used].T, nagg, seed=1, minit="++")
    doms, tsets = [], []
    for a in range(nagg):
        vs = used[lab == a]
        if len(vs) == 0: continue
        ts = np.unique(v2t[vs].indices)
        d = np.unique(loc[T2F[:, ts].ravel()]); d = d[d >= 0]
        doms.append(d); tsets.append(ts)
    mult = np.zeros(p.n)
    for d in doms: mult[d] += 1
    Z, sizes = [], []
    for d, ts in zip(doms, tsets):
        pos = -np.ones(p.n, int); pos[d] = np.arange(len(d))
        KN = np.zeros((len(d), len(d)), complex)
        r, cc = loc[rows[ts].ravel()], loc[cols[ts].ravel()]
        ok = (r >= 0) & (cc >= 0)
        np.add.at(KN, (pos[r[ok]], pos[cc[ok]]), (E_T[ts] - k2 * B_T[ts]).ravel()[ok])
        KN += Ksurf[d][:, d].toarray()
        D = 1.0 / mult[d]
        Rhs = (Aplus[d][:, d].toarray() * D[:, None]) * D[None, :]
        lam, V = dla.eig(KN, Rhs)
        o = np.argsort(np.abs(lam))
        o = o[:nvec] if tau is None else o[np.abs(lam[o]) < tau][:4 * nvec]
        z = np.zeros((p.n, len(o)), complex); z[d] = V[:, o] * D[:, None]
        Z.append(z); sizes.append(len(d))
    return np.concatenate(Z, axis=1), len(doms), int(np.mean(sizes))


for agg, nvec in ((2, 2), (2, 4), (4, 4), (4, 8), (8, 8), (8, 16)):
    t = time.time()
    U, na, sz = geneo(agg, nvec)
    pc, m, _ = bordered(K, pats, c, U)
    ts_ = time.time() - t
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
    print(f"  aggregates of ~{agg} vertices ({na}, local size ~{sz}), {nvec} each -> +{m} ({m / p.n:.1%} of n): its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} setup {ts_:.1f}s [{fmt_hist(hist)}]", flush=True)
