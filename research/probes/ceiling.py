import sys, time
sys.path[:0] = ["reference/helix", "tests", "research/probes"]
from emdata import DATA
from helix.mldata import MLPreconData
from helix import palace_pc as pp
m, pid = sys.argv[1], int(sys.argv[2])
rec = MLPreconData.load(f"{DATA}/{m}/problem_{pid}.npz")
for coarse in ("ams", "exact"):
    K, b, pc = pp.build(rec, coarse=coarse)
    t0 = time.time(); x, its, rr = pp.gmres(K, b, pc, rtol=1e-6, restart=200, maxiter=400)
    print(f"{m} p{pid} coarse={coarse}: {its} its relres {rr:.1e} {time.time()-t0:.1f}s")
