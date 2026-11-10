"""model4 p1: two-level with separate smoother damping in the PML and outside."""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from test_api import emerge_inputs, DATA
from helix import palace_pc as pp
import corvid
edges, tris, sid, K, b = emerge_inputs("model4", 1)
f = np.load(f"{DATA}/model4/problem_1.npz")
E = sp.csr_matrix((f["E_data"], f["E_indices"], f["E_indptr"]), shape=tuple(f["E_shape"]))[sid][:, sid].tocsr()
rows = np.repeat(np.arange(E.shape[0]), np.diff(E.indptr))
pml = np.zeros(E.shape[0], bool); pml[np.unique(rows[E.data.imag != 0])] = True
b = np.ascontiguousarray(b[:, 0], dtype=np.complex128)
top = corvid.Topology.nedelec2(edges, tris, sid); K = sp.csr_matrix(K); n = K.shape[0]
bp, bi = top.block_ptr, top.block_idx
blocks = [bi[bp[v]:bp[v + 1]] for v in range(len(bp) - 1)]
inv = [np.linalg.inv(K[d][:, d].toarray()) for d in blocks]
lu = spla.splu(K[top.whitney][:, top.whitney].tocsc())
def S(r):
    z = np.zeros(n, complex)
    for d, A in zip(blocks, inv):
        z[d] += A @ r[d]
    return z
def C(r):
    z = np.zeros(n, complex); z[top.whitney] = lu.solve(r[top.whitney]); return z
def lam(mask):
    v = np.random.default_rng(0).standard_normal(n) * mask + 0j
    for _ in range(40):
        w = S(K @ v) * mask; l = np.linalg.norm(w); v = w / l
    return l
la, lp = lam(~pml), lam(pml)
print(f"lambda_max: air {la:.2f}, PML {lp:.2f}")
for name, om in (("global 4/(3*13)", np.full(n, 4 / (3 * max(la, lp)))),
                 ("per region 4/(3*lambda_region)", np.where(pml, 4 / (3 * lp), 4 / (3 * la))),
                 ("air 4/(3*lambda_air), PML 0.5x that", np.where(pml, 2 / (3 * lp), 4 / (3 * la)))):
    def M(r, om=om):
        y = om * S(r); y = y + C(r - K @ y); return y + om * S(r - K @ y)
    rel = []
    t0 = time.time()
    x, its, rr = pp.gmres(K, b, M, rtol=1e-6, restart=300, maxiter=200, callback=lambda i, r: rel.append(r))
    pick = [rel[i] for i in (9, 29, 99, len(rel) - 1) if i < len(rel)]
    print(f"  damping {name:36s}: {its:3d} its, residual at 10/30/100/end " + " ".join(f"{r:.1e}" for r in pick), flush=True)
