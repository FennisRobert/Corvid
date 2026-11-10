"""Peak memory per stage of a CORVID solve (fresh process). usage: mem_stages.py model pid"""
import sys, resource, time
import numpy as np
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs
import corvid

def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
model, pid = sys.argv[1], int(sys.argv[2])
edges, tris, sid, K, b = emerge_inputs(model, pid)
del edges, tris
f = np.load(f"data/aegis/{model}/problem_{pid}.npz")
edges, tris = f["mesh_edges"], f["mesh_tris"]
p0 = peak()
kb = (K.data.nbytes + K.indices.nbytes + K.indptr.nbytes) / 2**20
top = corvid.Topology.nedelec2(edges, tris, sid); p1 = peak()
sol = corvid.Solver(top).factorize(K); p2 = peak()
x = sol.solve(b); p3 = peak()
print(f"{model} p{pid}: n={K.shape[0]} nnz={K.nnz}  K itself {kb:.0f} MB, vertex blocks {sol.memory_bytes/2**20:.0f} MB")
print(f"peak RSS: after loading {p0:.0f} MB | topology +{p1-p0:.0f} | factorize +{p2-p1:.0f} | solve +{p3-p2:.0f} MB "
      f"| CORVID total over the loaded problem: {p3-p0:.0f} MB = {(p3-p0)/kb:.1f}x K")
