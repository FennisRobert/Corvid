"""model4 (horn antenna with PML): does CORVID converge? usage: pml_probe.py pid ..."""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs, DATA
import corvid
for pid in map(int, sys.argv[1:]):
    edges, tris, sid, K, b = emerge_inputs("model4", pid)
    ref = np.load(f"{DATA}/model4/problem_{pid}.npz")["solution"][sid]
    sym = abs(K - K.T).max() / abs(K).max()
    rows = np.repeat(np.arange(K.shape[0]), np.diff(K.indptr))
    imrows = len(np.unique(rows[K.data.imag != 0]))
    print(f"model4 p{pid}: n={K.shape[0]}, |K-K^T|/|K| = {sym:.1e}, rows with imaginary part {imrows} ({imrows / K.shape[0]:.0%})")
    top = corvid.Topology.nedelec2(edges, tris, sid)
    for order in (1, 2):
        for method in ("gmres", "cocg"):
            try:
                t0 = time.time(); s = corvid.Solver(top, order=order, method=method, maxiter=300).factorize(K); tf = time.time() - t0
                t0 = time.time(); x = s.solve(b); ts = time.time() - t0
                err = np.linalg.norm(x - ref) / np.linalg.norm(ref)
                print(f"  order {order} {method:5s}: its {[i['iterations'] for i in s.info]} residual "
                      f"{['%.1e' % i['residual'] for i in s.info]}  error vs EMerge {err:.1e}  factorize {tf:.2f}s solve {ts:.2f}s", flush=True)
            except Exception as e:
                print(f"  order {order} {method:5s}: FAILED {type(e).__name__}: {e}", flush=True)
