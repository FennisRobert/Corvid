import sys, time, cProfile, pstats, io
import numpy as np, scipy.sparse as sp
sys.argv = ["x", "model2_abc2", "1"]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
a = AMGe(P1, Ps, Gs, surface=surf)
masks = [surf]
for Pl in Ps:
    masks.append(np.asarray(abs(Pl).T @ masks[-1].astype(float)).ravel() > 0)
a.sm = [TalosSmoother(Al, G, np.array([], int)) for Al, G in zip(a.As[:-1], Gs[:-1])]
pc.coarse = stat(a, P1, 2)
print("nnz: K", K.nnz, " P(fine)", pc.A.nnz, " Gf", pc.smoother.G.nnz, " GtAG", pc.smoother.SG.A.nnz,
      " coarse levels", [A.nnz for A in a.As])
# time one preconditioner application, split fine vs coarse
r = b.copy()
def timeit(f, n=10):
    t0 = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t0) / n * 1e3
print(f"K matvec {timeit(lambda: K @ r):.2f} ms   P matvec {timeit(lambda: pc.A @ r):.2f} ms")
print(f"full PC apply {timeit(lambda: pc(r)):.1f} ms")
print(f"  fine pre-smooth {timeit(lambda: pc.smoother.pre(r)):.1f} ms")
rc = pc.R.T @ r
print(f"  coarse solve (CORVID GS x2) {timeit(lambda: pc.coarse(rc)):.1f} ms")
pr = cProfile.Profile(); pr.enable()
x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14); print(s.getvalue()[:4000])
print("its", its)
