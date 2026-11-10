"""K2: tangentially twisted PML coarse space (Fermi-surface patches where scale separation exists).
In each PML face slab (normal axis nu, depth t in [0,1]):
    g(x) = chi_a(x_t) h_j(t) p exp(i k d . x_t),   d unit tangential, p in {nu, d_perp, d}
chi_a: smooth partition of unity over tangential cells of ~lambda/2;  h_j: hat functions in depth.
N2 coefficients by least squares on EMerge's evaluator Phi. Coarse = Whitney + span (bordered, QR).
Compared with CORVID 0.2.0 (full PML exact) and 0.2.1 (PML depth >= 2 exact).
usage: exp_pml_twist.py dataset box_mm thick_mm"""
import sys, time
import numpy as np, scipy.sparse as sp
import corvid
from common import Problem, gmres, fmt_hist, DATA, sparta_nnz
from proto import TwoLevel, mesh_vertex_patches, direct
from exp_spectral import bordered

name, B, T = sys.argv[1], float(sys.argv[2]) * 1e-3, float(sys.argv[3]) * 1e-3
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
f = np.load(f"{DATA}/{name}/problem_1.npz")
Phi = sp.csr_matrix((f["Phi_data"], f["Phi_indices"], f["Phi_indptr"]), shape=tuple(f["Phi_shape"]))
pts = f["Phi_pts"]
Phi_s = Phi[:, p.sid].tocsc()
Nlu = direct((Phi_s.T @ Phi_s).tocsr() + 1e-12 * sp.eye(p.n))
fit = lambda fld: Nlu.solve(np.ascontiguousarray(Phi_s.T @ fld.T.ravel()))
k = p.k0; L = 2 * np.pi / k
pats = [x for x in mesh_vertex_patches(p) if len(x)]

for pml_depth in (1, 2):
    s = corvid.Solver(p.top, pml_depth=pml_depth, maxiter=150, restart=150).factorize(K); s.solve(p.b)
    m = np.zeros(p.n, bool); m[c] = True; m |= s._exact_rows(K)
    print(f"{name}: corvid pml_depth={pml_depth}: exact rows {m.sum()} ({m.mean():.0%}), LDL {sparta_nnz(K[m][:, m])[0]/1e6:.2f}M, its {s.info[0]['iterations']}", flush=True)
base = TwoLevel(K, pats, c)
x, its0, _ = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
print(f"  Whitney only: its {its0}  (Whitney LDL {base.coarse_nnz/1e6:.2f}M)", flush=True)

exc = pts - B                                          # excess beyond the air box per axis
nu = np.argmax(exc, 0); depth = np.clip(exc.max(0) / T, 0, 1); inpml = exc.max(0) > 0


def build(ncell, ndir, nhat, kfac=(1.0,)):
    G = []
    hats = np.linspace(0, 1, nhat)
    for ax in range(3):
        sel = inpml & (nu == ax)
        tax = [a for a in range(3) if a != ax]
        xt = pts[tax][:, sel]
        cells = np.linspace(0, B + T, ncell + 1); cc = 0.5 * (cells[1:] + cells[:-1])
        cen = np.array([(u, v) for u in cc for v in cc])
        w = np.exp(-((xt.T[:, None, :] - cen[None]) ** 2).sum(-1) / (0.6 * (B + T) / ncell) ** 2)
        w /= w.sum(1, keepdims=True)
        hw = np.maximum(0, 1 - np.abs(depth[sel][:, None] - hats[None]) * (nhat - 1))
        for kf in kfac:
            for th in np.arange(ndir) * 2 * np.pi / ndir:
                dt = np.array([np.cos(th), np.sin(th)])
                ph = np.exp(1j * kf * k * dt @ xt)
                d3 = np.zeros(3); d3[tax] = dt
                dp = np.zeros(3); dp[tax] = [-dt[1], dt[0]]
                nv = np.zeros(3); nv[ax] = 1
                for pol in (nv, dp, d3):
                    for a in range(len(cen)):
                        for j in range(nhat):
                            fld = np.zeros((3, pts.shape[1]), complex)
                            fld[:, sel] = pol[:, None] * (w[:, a] * hw[:, j] * ph)[None, :]
                            G.append(fit(fld))
    return np.column_stack(G)


for ncell, ndir, nhat in ((1, 6, 4), (2, 6, 4), (2, 8, 5), (3, 8, 5)):
    t = time.time()
    G = build(ncell, ndir, nhat)
    pc, mvec, _ = bordered(K, pats, c, G)
    ts = time.time() - t
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
    print(f"  twisted PML space: {ncell}x{ncell} cells/face, {ndir} dirs, 3 pols, {nhat} depth hats -> +{mvec} vectors: its {its:3d} "
          f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} setup {ts:.1f}s [{fmt_hist(hist)}]", flush=True)
