"""K1: character-twisted (wave-ray / Fermi-surface patch) coarse space.
Coarse functions g_{a,d,p}(x) = chi_a(x) p exp(i k d.x): chi_a a smooth partition of unity over
aggregates of size ~ lambda/2, d on a Fibonacci sphere, p two polarisations orthogonal to d.
N2 coefficients by least squares on EMerge's own point evaluator Phi (exported by gen_box.py phi=1).
Coarse space = Whitney (injection) + span{g}, Galerkin, bordered solve.
usage: exp_waveray.py dataset"""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.linalg as dla
from scipy.cluster.vq import kmeans2
from common import Problem, gmres, fmt_hist, DATA
from proto import TwoLevel, mesh_vertex_patches, direct

name = sys.argv[1]
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney; nc = len(c)
f = np.load(f"{DATA}/{name}/problem_1.npz")
Phi = sp.csr_matrix((f["Phi_data"], f["Phi_indices"], f["Phi_indptr"]), shape=tuple(f["Phi_shape"]))
pts = f["Phi_pts"]
Phi_s = Phi[:, p.sid].tocsc()                      # only solved DOFs (PEC DOFs are zero anyway)
N = (Phi_s.T @ Phi_s).tocsr()
Nlu = direct(N + 1e-12 * sp.diags(N.diagonal()))
def fit(field):                                     # field: (3, npts) complex -> coefficients on solved DOFs
    return Nlu.solve(np.ascontiguousarray(Phi_s.T @ field.T.ravel()))
k = p.k0
L = 2 * np.pi / k
pats = [x for x in mesh_vertex_patches(p) if len(x)]

# 1) sanity: a fitted plane wave is a discrete near-solution in the interior
d0 = np.array([1, 0.3, 0.2]); d0 /= np.linalg.norm(d0); pol = np.cross(d0, [0, 0, 1]); pol /= np.linalg.norm(pol)
u = fit(pol[:, None] * np.exp(1j * k * d0 @ pts)[None, :])
r = K @ u
lo, hi = p.loc.min(1), p.loc.max(1)
inner = np.all((p.loc > lo[:, None] + 0.3 * L) & (p.loc < hi[:, None] - 0.3 * L), axis=0)
rnd = (np.random.default_rng(0).standard_normal(p.n) + 0j)
fitres = np.linalg.norm(Phi_s @ u - (pol[:, None] * np.exp(1j * k * d0 @ pts)[None, :]).T.ravel()) / np.linalg.norm(np.exp(1j * k * d0 @ pts)) / np.sqrt(1)
print(f"{name}: n={p.n}, k h median {k * np.median(np.linalg.norm(p.nodes[:, p.edges[0]] - p.nodes[:, p.edges[1]], axis=0)):.2f}")
print(f"  plane-wave fit: relative point misfit {fitres:.1e}; interior |Ku|/(|K||u|) {np.linalg.norm(r[inner]) / np.linalg.norm(abs(K)[inner] @ abs(u)):.1e}"
      f"  (random vector {np.linalg.norm((K @ rnd)[inner]) / np.linalg.norm(abs(K)[inner] @ abs(rnd)):.1e})")

def fib_sphere(m):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)

def coarse_space(agg_size, ndir):
    span = (hi - lo) / L
    nagg = max(1, int(np.prod(np.maximum(1, np.round(span / agg_size)))))
    _, lab = kmeans2(p.loc.T / L, nagg, seed=1, minit="++")
    cent = np.array([p.loc[:, lab == a].mean(1) for a in range(nagg) if np.any(lab == a)])
    # smooth partition of unity on the sample points: Gaussian bumps normalised
    dist2 = ((pts.T[:, None, :] - cent[None, :, :]) ** 2).sum(-1) / (agg_size * L) ** 2
    w = np.exp(-2.0 * dist2); w /= w.sum(1, keepdims=True)
    G = []
    for d in fib_sphere(ndir):
        a1 = np.cross(d, [0.3, 0.5, 0.8]); a1 /= np.linalg.norm(a1); a2 = np.cross(d, a1)
        ph = np.exp(1j * k * d @ pts)
        for pv in (a1, a2):
            for a in range(len(cent)):
                G.append(fit(pv[:, None] * (w[:, a] * ph)[None, :]))
    return np.column_stack(G), len(cent)

def bordered(U, base):
    Uo = U.copy(); Uo[c] = 0
    Uo, sv, _ = np.linalg.svd(Uo, full_matrices=False)
    Uo = Uo[:, sv > 1e-8 * sv[0]]
    KU = K @ Uo
    Bm = KU[c]
    Wm = np.column_stack([base.clu.solve(np.ascontiguousarray(Bm[:, j])) for j in range(Bm.shape[1])])
    Sf = dla.lu_factor(Uo.T @ KU - Bm.T @ Wm)
    def cfn(r):
        y = base.clu.solve(np.ascontiguousarray(r[c]))
        z = dla.lu_solve(Sf, Uo.T @ r - Bm.T @ y)
        e = Uo @ z; e[c] += y - Wm @ z
        return e
    return TwoLevel(K, pats, c, coarse_fn=cfn), Uo.shape[1]

base = TwoLevel(K, pats, c)
x, its, hist = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
print(f"  CORVID 0.2.0: its {its}")
for agg_size, ndir in ((1.0, 6), (1.0, 12), (0.5, 6), (0.5, 12), (0.5, 20)):
    t = time.time()
    G, na = coarse_space(agg_size, ndir)
    pc, m = bordered(G, base)
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
    print(f"  wave-ray: aggregates ~{agg_size} wl ({na}), {ndir} directions x 2 pol -> +{m} coarse vectors: its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e}  setup {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
