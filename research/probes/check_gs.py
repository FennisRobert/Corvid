import sys, numpy as np, scipy.sparse as sp
sys.path[:0] = ["."]
from pyamg.relaxation.relaxation import gauss_seidel
from corvid.fast import FastCsr
rng = np.random.default_rng(1)
n = 3000
A = sp.random(n, n, 5e-3, random_state=1) + sp.random(n, n, 5e-3, random_state=2) * 1j
A = (A + A.T + sp.eye(n) * 8).tocsr()
b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
for sweep in ("forward", "backward"):
    x1 = np.zeros(n, complex); x2 = np.zeros(n, complex)
    gauss_seidel(A, x1, b, iterations=3, sweep=sweep)
    FastCsr(A).gauss_seidel(x2, b, forward=(sweep == "forward"), sweeps=3)
    print(sweep, "max |rust - pyamg| / |x| =", abs(x1 - x2).max() / abs(x1).max())
