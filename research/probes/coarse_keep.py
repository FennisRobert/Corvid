"""Coarse (Whitney-level) solver variants inside helix's preconditioner, fine
smoother fixed at helix's default. 'keep' = the complex unflipped R^T P R."""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = [".", "tests", "research/probes",
                "reference/helix"]
from emdata import DATA, load
from helix.mldata import MLPreconData
from helix import palace_pc as pp
from corvid.amge.mg import hierarchy, AMGe

m, pid = sys.argv[1], int(sys.argv[2])
rec = MLPreconData.load(f"{DATA}/{m}/problem_{pid}.npz")
K, b, pc = pp.build(rec)
sid = rec.solve_ids; nE = rec.mesh_edges.shape[1]; nT = (rec.E.shape[0] - 2 * nE) // 2
R, kept = pp.whitney_injection(sid, nE, nT)
P = pp.restricted_system(rec, "keep")[1]
P1 = (R.T @ P @ R).tocsr()
Pf = (R.T @ pp.restricted_system(rec, "flip")[1] @ R).tocsr(); A1 = sp.csr_matrix(Pf.real + Pf.imag)
rows = np.repeat(np.arange(P1.shape[0]), np.diff(P1.indptr))
surf = np.zeros(P1.shape[0], bool); surf[rows[P1.data.imag != 0]] = True
d = load(m, pid)
t0 = time.time(); Ps, Gs = hierarchy(d["coords"], d["tets"], d["edges"], d["free"], target=32)
print(f"{m} p{pid}: Whitney n={P1.shape[0]} surface rows {surf.sum()} levels {[P1.shape[0]] + [p.shape[1] for p in Ps]} ({time.time()-t0:.1f}s)")

def stat(M, A, n):
    def f(r):
        x = M(r)
        for _ in range(n - 1):
            x = x + M(r - A @ x)
        return x
    return f

variants = {
    "exact keep": spla.splu(P1.tocsc(), **pp._LU).solve,
    "AMGe flip x2": pp.RealOnComplex(stat(AMGe(A1, Ps, Gs), A1, 2)),
}
for order in (2, 4):
    a = AMGe(P1, Ps, Gs, order=order, surface=surf)
    variants[f"AMGe keep+surf o{order} x1"] = a
    variants[f"AMGe keep+surf o{order} x2"] = stat(a, P1, 2)
variants["AMGe keep nosurf o2 x2"] = stat(AMGe(P1, Ps, Gs), P1, 2)
for name, cs in variants.items():
    pc.coarse = cs
    t0 = time.time()
    try:
        x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
        print(f"  {name:24s}: {its:4d} its  relres {rr:.1e}  {time.time() - t0:5.1f}s")
    except Exception as e:
        print(f"  {name:24s}: failed {e}")
