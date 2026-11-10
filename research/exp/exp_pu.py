"""F1 (concept test): Whitney level enriched by partition-of-unity-localised near-resonant waves.
Coarse space V_c = range(injection) + span{chi_a * v_j}: chi_a = indicator of aggregate a (k-means
on DOF positions), v_j = eigenvectors of K nearest 0 (stand-ins for local plane waves).
Galerkin coarse operator, Q = P. Compared with global deflation (span{v_j} without localisation)."""
import sys
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as dla
from scipy.cluster.vq import kmeans2
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, direct

name = sys.argv[1]
p = Problem(name); K = p.K.tocsr(); c = p.top.whitney; nc = len(c)
pats = [x for x in mesh_vertex_patches(p) if len(x)]
Klu = direct(K)
Kinv = sla.LinearOperator(K.shape, matvec=lambda v: Klu.solve(np.ascontiguousarray(v, dtype=complex)), dtype=complex)
ev, V = sla.eigs(K, k=60, sigma=0, OPinv=Kinv, which="LM")
V = V[:, np.argsort(abs(ev))]
L = 2 * np.pi / p.k0
ext = p.loc.max(1) - p.loc.min(1)
print(f"{name}: n={p.n}, Whitney {nc}, domain {ext / L} wavelengths")

def coarse_from(Z, label):
    Pinj = sp.csr_matrix((np.ones(nc), (c, np.arange(nc))), shape=(p.n, nc))
    P = sp.hstack([Pinj, sp.csr_matrix(Z)]).tocsr()
    Kc = (P.T @ K @ P).toarray()
    F = dla.lu_factor(Kc)
    pc = TwoLevel(K, pats, c, coarse_fn=lambda r: P @ dla.lu_solve(F, P.T @ r))
    x, its, hist = gmres(lambda v: K @ v, pc.apply, p.b, maxiter=100)
    print(f"  {label:34s} coarse dim {P.shape[1]:5d} its {its:3d} res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} [{fmt_hist(hist)}]", flush=True)

coarse_from(np.zeros((p.n, 0)), "Whitney only (CORVID 0.2.0)")
for m in (10, 20, 40, 60):
    coarse_from(V[:, :m], f"+ {m} global eigenvectors")
for nagg in (4, 8, 16, 32):
    _, lab = kmeans2(p.loc.T / L, nagg, seed=1, minit="++")
    for m in (5, 10, 20):
        Z = np.concatenate([np.where((lab == a)[:, None], V[:, :m], 0) for a in range(nagg)], axis=1)
        coarse_from(Z, f"+ PU({nagg} aggregates) x {m} waves")
