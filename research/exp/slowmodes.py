"""Anatomy of the slow modes of E = S (I - Pi) S on a problem: for the dominant eigenvectors v of E,
report |mu|, the K-Rayleigh quotient relative to |K| (near-kernel or not), the Whitney share,
the smoother factor ||S v|| / ||v||, the coarse factor ||(I - Pi) v|| / ||v||, and the eigenvalue of
omega B K along v (v^* omega B K v / v^* v)."""
import sys
import numpy as np, scipy.sparse.linalg as sla
from common import Problem
from proto import TwoLevel, mesh_vertex_patches, direct

p = Problem(sys.argv[1]); K = p.K.tocsr(); c = p.top.whitney
pats = [x for x in mesh_vertex_patches(p) if len(x)]
pc = TwoLevel(K, pats, c)
w = pc.c0
B = lambda v: pc.sch.apply(np.ascontiguousarray(v, dtype=complex))
Smooth = lambda v: v - w * B(K @ v)
def coarse(v):
    r = K @ v
    e = np.zeros_like(v); e[c] = pc.clu.solve(np.ascontiguousarray(r[c]))
    return v - e
E = sla.LinearOperator(K.shape, matvec=lambda v: Smooth(coarse(Smooth(v))), dtype=complex)
mu, V = sla.eigs(E, k=40, which="LM", tol=1e-6, ncv=100)
o = np.argsort(-abs(mu)); mu, V = mu[o], V[:, o]
Kabs = abs(K)
print(f"{sys.argv[1]}: n={p.n}, omega={w:.3e}")
print(" |mu|   |v'Kv|/v'|K|v  Whitney  ||Sv||  ||(I-Pi)v||  wBK Rayleigh")
for i in range(0, 40, 3):
    v = V[:, i] / np.linalg.norm(V[:, i])
    rq = abs(np.vdot(v, K @ v)) / np.real(np.vdot(abs(v), Kabs @ abs(v)))
    ws = np.sum(abs(v[c]) ** 2)
    sb = np.vdot(v, w * B(K @ v))
    print(f" {abs(mu[i]):.3f}  {rq:10.2e}   {ws:5.2f}   {np.linalg.norm(Smooth(v)):.3f}   {np.linalg.norm(coarse(v)):9.3f}   {sb.real:+.3f}{sb.imag:+.3f}i")
# reference: same quantities for the 5 eigenvectors of K nearest 0
Klu = direct(K)
Kinv = sla.LinearOperator(K.shape, matvec=lambda v: Klu.solve(np.ascontiguousarray(v, dtype=complex)), dtype=complex)
ev, U = sla.eigs(K, k=5, sigma=0, OPinv=Kinv)
print(" near-kernel eigenvectors of K:")
for i in range(5):
    v = U[:, i] / np.linalg.norm(U[:, i])
    rq = abs(np.vdot(v, K @ v)) / np.real(np.vdot(abs(v), Kabs @ abs(v)))
    print(f"   rq {rq:.2e} Whitney {np.sum(abs(v[c])**2):.2f} ||Sv|| {np.linalg.norm(Smooth(v)):.3f} ||(I-Pi)v|| {np.linalg.norm(coarse(v)):.3f} ||Ev|| {np.linalg.norm(E @ v):.3f}")
