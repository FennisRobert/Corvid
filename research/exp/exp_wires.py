"""N1: 'wire' (line) blocks, the antenna designer's idea. Vertices are binned into thin pencils along x,
y and z (cross-section ~ `w` median edge lengths); each pencil's block = union of the vertex blocks
of its vertices, factored with SPARTA (LDL^T, double). Smoother = multiplicative sweep over the pencils
of one or all three orientations (forward pre, backward post), Whitney (+ deep PML) exact coarse.
Compared with CORVID's additive vertex blocks and the multiplicative vertex sweep.
usage: exp_wires.py dataset ..."""
import sys, time
import numpy as np
import corvid
from common import Problem, gmres, fmt_hist
from proto import TwoLevel, mesh_vertex_patches, pack
from corvid import _kernels
from proto import kcsr


def pencils(p, pats, axis, w):
    X = p.nodes
    h = np.median(np.linalg.norm(X[:, p.edges[0]] - X[:, p.edges[1]], axis=0))
    used = np.flatnonzero([len(x) > 0 for x in pats])
    other = [a for a in range(3) if a != axis]
    key = np.floor(X[other][:, used] / (w * h)).astype(int)
    _, lab = np.unique(key.T, axis=0, return_inverse=True)
    blocks = []
    for g in range(lab.max() + 1):
        vs = used[lab.ravel() == g]
        vs = vs[np.argsort(X[axis, vs])]
        blocks.append(np.unique(np.concatenate([pats[v] for v in vs])))
    return blocks


class WireSmoother:
    def __init__(self, K, coarse, block_sets, vertex_pats, omega=1.0):
        self.K = K
        self.tl = TwoLevel(K, vertex_pats, coarse)
        self.c = coarse
        kk = kcsr(K)
        self.sweeps = [_kernels.RxSweep(kk, *pack(bs)) for bs in block_sets]
        self.nnz = sum(s.factor_nnz for s in self.sweeps)
        self.omega = omega

    def apply(self, r):
        r = np.ascontiguousarray(r, dtype=complex)
        y = np.zeros_like(r)
        for s in self.sweeps:
            y = s.sweep(r, y, True, self.omega)
        res = r - self.K @ y
        y[self.c] += self.tl.clu.solve(np.ascontiguousarray(res[self.c]))
        for s in reversed(self.sweeps):
            y = s.sweep(r, y, False, self.omega)
        return y


for name in sys.argv[1:]:
    p = Problem(name); K = p.K.tocsr()
    m = np.zeros(p.n, bool); m[p.top.whitney] = True
    ex = corvid.Solver(p.top)._exact_rows(K)
    if ex is not None: m |= ex
    coarse = np.flatnonzero(m)
    pats_all = mesh_vertex_patches(p)
    pats = [x for x in pats_all if len(x)]
    base = TwoLevel(K, pats, coarse)
    x, its, hist = gmres(lambda v: K @ v, base.apply, p.b, maxiter=150)
    vb = sum(len(x) * (len(x) + 1) // 2 for x in pats)
    print(f"{name}: n={p.n}; CORVID additive vertex blocks: its {its:3d}, block storage {vb/1e6:.2f}M entries", flush=True)
    vs = WireSmoother(K, coarse, [pats], pats)
    x, its, hist = gmres(lambda v: K @ v, vs.apply, p.b, maxiter=150)
    print(f"  multiplicative vertex blocks (SPARTA):        its {its:3d}, factor {vs.nnz/1e6:.2f}M [{fmt_hist(hist)}]", flush=True)
    for w in (1.0, 2.0):
        for axes in ((0,), (2,), (0, 1, 2)):
            t = time.time()
            sets = [pencils(p, pats_all, a, w) for a in axes]
            ws = WireSmoother(K, coarse, sets, pats)
            x, its, hist = gmres(lambda v: K @ v, ws.apply, p.b, maxiter=150)
            nb = sum(len(s) for s in sets)
            print(f"  wires w={w} axes {'xyz'[axes[0]] if len(axes) == 1 else 'xyz'}: {nb:4d} wires, its {its:3d}, factor {ws.nnz/1e6:.2f}M, "
                  f"res {np.linalg.norm(K @ x - p.b) / np.linalg.norm(p.b):.0e} {time.time()-t:.1f}s [{fmt_hist(hist)}]", flush=True)
