"""I1: bootstrap enrichment of the Whitney level. Test vectors T = E^q R (R random, E the CORVID
two-level error propagator), optionally localised by a partition of unity over aggregates, added to
the coarse space (Galerkin). Setup cost counted in preconditioner applications (m q).
usage: exp_bootstrap.py dataset ..."""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.linalg as dla
from scipy.cluster.vq import kmeans2
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches


def enriched(p, K, pats, c, Z):
    nc = len(c)
    Pinj = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
    P = sp.hstack([Pinj, sp.csr_matrix(Z)]).tocsr()
    F = dla.lu_factor((P.T @ K @ P).toarray())
    return TwoLevel(K, pats, c, coarse_fn=lambda r: P @ dla.lu_solve(F, P.T @ r))


for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr(); c = p.top.whitney
    pats = [x for x in mesh_vertex_patches(p) if len(x)]
    base = TwoLevel(K, pats, c)
    x, its0, _ = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
    print(f"{name}: n={p.n}, CORVID 0.2.0: {its0} its")
    L = 2 * np.pi / p.k0
    rng = np.random.default_rng(0)
    for m, q in ((8, 4), (8, 8), (16, 4), (16, 8), (32, 6)):
        T = rng.standard_normal((p.n, m)) + 0j
        for _ in range(q):
            T = T - np.column_stack([base.apply(K @ T[:, j]) for j in range(m)])
            T, _ = np.linalg.qr(T)
        for nagg in (1, 8, 16):
            if nagg == 1:
                Z = T
            else:
                _, lab = kmeans2(p.loc.T / L, nagg, seed=1, minit="++")
                Z = np.concatenate([np.where((lab == a)[:, None], T, 0) for a in range(nagg)], axis=1)
            pc = enriched(p, K, pats, c, Z)
            x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=150)
            print(f"  m={m:2d} q={q}  ({m*q:3d} setup applications)  PU {nagg:2d}: coarse +{Z.shape[1]:4d}  its {its:3d} "
                  f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)
