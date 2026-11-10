"""Schwarz combinations on model4 (PML horn) and model2 (no PML), Python prototype.
Two-level multiplicative: smoother, exact Whitney correction, smoother."""
import sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
sys.path[:0] = ["tests", "research/probes", "reference/helix"]
from test_api import emerge_inputs
from helix import palace_pc as pp
import corvid

model, pid = sys.argv[1], int(sys.argv[2])
edges, tris, sid, K, b = emerge_inputs(model, pid)
b = np.ascontiguousarray(b[:, 0], dtype=np.complex128)
top = corvid.Topology.nedelec2(edges, tris, sid)
K = sp.csr_matrix(K); K.sort_indices(); n = K.shape[0]
bp, bi = top.block_ptr, top.block_idx
blocks = [bi[bp[v]:bp[v + 1]] for v in range(len(bp) - 1)]
t0 = time.time()
inv = [np.linalg.inv(K[d][:, d].toarray()) for d in blocks]
mult = np.bincount(bi, minlength=n).astype(float)
# owner for RAS: the block with the lowest index containing the DOF
owner = np.full(n, -1); 
for v, d in enumerate(blocks):
    free = d[owner[d] < 0]; owner[free] = v
own = [d[owner[d] == v] for v, d in enumerate(blocks)]
ownmask = [owner[d] == v for v, d in enumerate(blocks)]
print(f"{model} p{pid}: n={n}, {len(blocks)} blocks, inverses {time.time() - t0:.1f}s, multiplicity mean {mult.mean():.2f}")
W = K[top.whitney][:, top.whitney].tocsc(); lu = spla.splu(W)

def S_as(r):
    z = np.zeros(n, complex)
    for d, A in zip(blocks, inv):
        z[d] += A @ r[d]
    return z
def S_pou(r):                                  # symmetric partition-of-unity weighting
    w = 1 / np.sqrt(mult); z = np.zeros(n, complex); rw = w * r
    for d, A in zip(blocks, inv):
        z[d] += A @ rw[d]
    return w * z
def S_ras(r):                                  # restricted: each DOF from its owning block
    z = np.zeros(n, complex)
    for d, A, m in zip(blocks, inv, ownmask):
        z[d[m]] = (A @ r[d])[m]
    return z
def C(r):
    z = np.zeros(n, complex); z[top.whitney] = lu.solve(r[top.whitney]); return z
def lam(S):
    v = np.random.default_rng(0).standard_normal(n) + 0j
    for _ in range(30):
        w = S(K @ v); l = np.linalg.norm(w); v = w / l
    return l
def two_level(S, omega):
    def M(r):
        y = omega * S(r)
        y = y + C(r - K @ y)
        return y + omega * S(r - K @ y)
    return M

for name, S in (("additive (current)", S_as), ("partition-of-unity", S_pou), ("restricted (RAS)", S_ras)):
    l = lam(S)
    for om_name, om in (("4/(3 lmax)", 4 / (3 * l)), ("1", 1.0)):
        rel = []
        t0 = time.time()
        x, its, rr = pp.gmres(K, b, two_level(S, om), rtol=1e-6, restart=300, maxiter=200, callback=lambda i, r: rel.append(r))
        pick = [rel[i] for i in (9, 29, 99, len(rel) - 1) if i < len(rel)]
        print(f"  {name:22s} omega {om_name:11s} (lmax {l:6.2f}): {its:3d} its, residual at 10/30/100/end "
              + " ".join(f"{r:.1e}" for r in pick) + f"  ({time.time() - t0:.0f}s)", flush=True)
