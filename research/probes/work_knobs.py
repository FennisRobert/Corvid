import sys, time
import numpy as np
sys.argv = ["x", sys.argv[1], sys.argv[2]]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
a = AMGe(P1, Ps, Gs)
a.sm = [TalosSmoother(Al, G, np.array([], int)) for Al, G in zip(a.As[:-1], Gs[:-1])]
for order in (4, 2, 1):
    for cycles in (2, 1):
        for cx in (2, 1):
            K, b, pc = pp.build(rec, order=order, cycles=cycles)
            pc.coarse = stat(a, P1, cx)
            for solver in ("gmres", "cocg"):
                t0 = time.time()
                if solver == "gmres":
                    x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=600)
                else:
                    x, its, rr = pp.cocg(K, b, pc, rtol=1e-6, maxiter=600)
                print(f"  cheb order {order} cycles {cycles} coarse x{cx} {solver:5s}: {its:4d} its  relres {rr:.1e}  {time.time() - t0:5.1f}s", flush=True)
