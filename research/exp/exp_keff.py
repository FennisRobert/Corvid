"""D1: exact level = Whitney + complement rows with k_eff h > kappa, where
k_eff = k |s| and |s| = 1/sqrt(1 - rho^2), rho = sum|Im K_ij| / sum|K_ij| (algebraic PML strength)."""
import sys, time
import numpy as np, scipy.sparse as sp
import corvid
from common import Problem, sparta_nnz

def keff_h(p):
    X = p.nodes; E = p.edges; T = p.tris
    le = np.linalg.norm(X[:, E[0]] - X[:, E[1]], axis=0)
    lt = (np.linalg.norm(X[:, T[0]] - X[:, T[1]], axis=0) + np.linalg.norm(X[:, T[1]] - X[:, T[2]], axis=0) + np.linalg.norm(X[:, T[0]] - X[:, T[2]], axis=0)) / 3
    h = np.where(p.kind < 2, le[np.minimum(p.ent, len(le) - 1)], lt[np.minimum(p.ent, len(lt) - 1)])
    K = p.K.tocsr(); rows = np.repeat(np.arange(p.n), np.diff(K.indptr))
    rho = np.bincount(rows, np.abs(K.data.imag), p.n) / np.bincount(rows, np.abs(K.data), p.n)
    s = 1 / np.sqrt(np.clip(1 - rho ** 2, 1e-6, 1))
    return p.k0 * h * np.where(p.pml, s, 1.0), s

for name in sys.argv[1:]:
    p = Problem(name)
    kh, s = keff_h(p)
    full = sparta_nnz(p.K)[0]
    print(f"{name}: n={p.n}, k0 h median {np.median(p.k0 * kh / kh * kh / np.where(p.pml, s, 1)):.2f}; direct LDL {full/1e6:.2f}M")
    W = np.zeros(p.n, bool); W[p.top.whitney] = True
    for kappa in (np.inf, 2.5, 2.0, 1.6, 1.3, 1.0):
        m = (kh > kappa) & ~W
        t = time.time()
        sv = corvid.Solver(p.top, exact=m, maxiter=150, restart=150).factorize(p.K); sv.solve(p.b)
        mm = m | W
        print(f"  kappa {kappa:4}: +{m.sum():6d} rows ({m.mean():5.1%}), coarse LDL {sparta_nnz(p.K[mm][:, mm])[0]/1e6:6.2f}M"
              f" ({sparta_nnz(p.K[mm][:, mm])[0]/full:4.0%} of direct)  its {sv.info[0]['iterations']:3d}  {time.time()-t:.1f}s", flush=True)
    sv = corvid.Solver(p.top, exact="auto", maxiter=150, restart=150).factorize(p.K); sv.solve(p.b)
    print(f"  0.2.0 auto: +{sv.n_exact} rows, its {sv.info[0]['iterations']}")
