import sys, time, os
import numpy as np
sys.path[:0] = [".", "tests", "research/probes", "reference/helix"]
from emdata import DATA
from helix.mldata import MLPreconData
from helix import palace_pc as pp
from corvid.fast import FastCsr
rec = MLPreconData.load(f"{DATA}/model2_abc2/problem_1.npz")
K, P, b = pp.restricted_system(rec, "keep")
F = FastCsr(K)
x = np.random.default_rng(0).standard_normal(K.shape[0]) + 1j
def t(f, n=50):
    f(); t0 = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t0) / n * 1e3
print("max err", abs(F @ x - K @ x).max() / abs(K @ x).max())
ts, tf = t(lambda: K @ x), t(lambda: F @ x)
mb = (K.nnz * 20 + K.shape[0] * 48) / 1e6
print(f"n={K.shape[0]} nnz={K.nnz}: scipy {ts:.2f} ms  rust {tf:.3f} ms  ({ts/tf:.1f}x, {mb/tf:.0f} GB/s)  threads={os.environ.get('RAYON_NUM_THREADS','all')}")
