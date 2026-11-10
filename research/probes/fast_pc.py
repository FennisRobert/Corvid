"""model2: helix preconditioner + CORVID GS coarse, with the fine-level sparse
products moved to CORVID's threaded Rust kernel. Same arithmetic, same its."""
import sys, time, cProfile, pstats, io
import numpy as np
sys.argv = ["x", sys.argv[1], sys.argv[2]]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
from corvid.fast import FastCsr

a = AMGe(P1, Ps, Gs)
a.sm = [TalosSmoother(Al, G, np.array([], int)) for Al, G in zip(a.As[:-1], Gs[:-1])]
coarse = stat(a, P1, 2)


class _Prod:
    """scipy CSR for pyamg's GS, threaded FastCsr for every product."""
    def __init__(self, A):
        self.csr, self.f = A, FastCsr(A)
        self.shape, self.dtype, self.indptr = A.shape, A.dtype, A.indptr
    def __matmul__(self, x):
        return self.f @ x


def fast_coarse(a):
    for sm in a.sm:
        Af, AGf, Gf_ = FastCsr(sm.A), FastCsr(sm.AG), FastCsr(sm.G)
        csrA, csrAG = sm.A, sm.AG
        if GS == "pyamg":
            sm._gs = (lambda A, x, b, sweep, sm=sm, cA=csrA, cAG=csrAG:
                      gauss_seidel(cA if A is sm.A else cAG, x, b, iterations=sm.sweeps, sweep=sweep))
        else:
            sm._gs = (lambda A, x, b, sweep, sm=sm:
                      A.gauss_seidel(x, b, forward=(sweep == "forward"), sweeps=sm.sweeps, nblocks=NB))
        sm.A, sm.AG, sm.G, sm.Gt = Af, AGf, Gf_, Gf_.T
    a.As = [FastCsr(A) for A in a.As]
    a.Ps = [FastCsr(P) for P in a.Ps]

def solve(K, pc, solver):
    t0 = time.time()
    if solver == "gmres":
        x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
    else:
        x, its, rr = pp.cocg(K, b, pc, rtol=1e-6, maxiter=600)
    return its, rr, time.time() - t0

def fast(pc, K):
    sm = pc.smoother
    Pf = FastCsr(pc.A)
    pc.A = sm.A = sm.S.A = Pf
    sm.SG.A = FastCsr(sm.SG.A)
    sm.G = FastCsr(sm.G); sm.Gt = sm.G.T
    pc.R = FastCsr(pc.R)
    return FastCsr(K)


import copy, os
GS = os.environ.get('GS', 'rust'); NB = int(os.environ.get('NB', '1'))
print('coarse GS:', GS, 'nblocks', NB)
a_fast = copy.deepcopy(a); fast_coarse(a_fast)
coarse_fast = stat(a_fast, FastCsr(P1), 2)

for mode in ("rust",):
    K0, b, pc = pp.build(rec)
    pc.coarse = coarse if mode == "scipy" else coarse_fast
    K1 = fast(pc, K0) if mode == "rust" else K0
    for solver in ("gmres", "cocg"):
        its, rr, t = solve(K1, pc, solver)
        print(f"  {mode:5s} {solver:5s}: {its:4d} its  relres {rr:.1e}  {t:5.2f}s", flush=True)
PROFILE = os.environ.get('PROF'); K0, b, pc = pp.build(rec); pc.coarse = coarse_fast; K1 = fast(pc, K0)
if not PROFILE: sys.exit()
pr = cProfile.Profile(); pr.enable(); solve(K1, pc, "cocg"); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10); print(s.getvalue()[:3000])
