"""Default Solver (exact="auto") on everything: PML rows detected and solved exactly."""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs
import corvid
S = "research/data"
def scratch(d):
    f = np.load(f"{S}/{d}/problem_1.npz")
    A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
    sid = f["solve_ids"]; return f["mesh_edges"], f["mesh_tris"], sid, A[sid][:, sid].tocsr(), f["b"][sid]
cases = [("model1", 1), ("model2_abc2", 1), ("model3", 9)] + [("model4", p) for p in (1, 4, 7)] + \
        [(d, None) for d in ("model4_abc_r15", "model4_d8_r15", "model4_d4_r15")]
for m, pid in cases:
    E, F, sid, K, b = emerge_inputs(m, pid) if pid else scratch(m)
    top = corvid.Topology.nedelec2(E, F, sid)
    for exact in (None, "auto"):
        t0 = time.time(); s = corvid.Solver(top, exact=exact, maxiter=300, restart=300).factorize(K); tf = time.time() - t0
        t0 = time.time(); x = s.solve(b); ts = time.time() - t0
        rr = np.linalg.norm(K @ x - b) / np.linalg.norm(b)
        print(f"{m:16s} {'' if pid is None else 'p%d' % pid:3s} n={K.shape[0]:6d} exact={str(exact):4s} (+{s.n_exact:6d} rows): "
              f"its {[i['iterations'] for i in s.info]} residual {rr:.1e} factorize {tf:.2f}s solve {ts:.2f}s", flush=True)
