"""Corvid on the Weng filter inputs dumped by weng_model.py: wall time and core usage per phase.

    python bench/weng/bench_corvid.py [freq ...]      e.g. 0.50 2.50 4.50 (default: all dumps)

Core usage = process CPU time / wall time, so 8.0 means all 8 threads busy. Runs at 8 threads.
"""
import glob
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import corvid  # noqa: E402
from corvid import _kernels  # noqa: E402

NTHREADS = 8
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Phase:
    def __init__(self, name, rows):
        self.name, self.rows = name, rows

    def __enter__(self):
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.c0, self.t0 = r.ru_utime + r.ru_stime, time.perf_counter()
        return self

    def __exit__(self, *a):
        r = resource.getrusage(resource.RUSAGE_SELF)
        wall = time.perf_counter() - self.t0
        cpu = r.ru_utime + r.ru_stime - self.c0
        self.rows.append((self.name, wall, cpu / max(wall, 1e-9)))


def load(path):
    f = np.load(path)
    K = sp.csr_matrix((f["K_data"], f["K_indices"], f["K_indptr"]), shape=tuple(f["K_shape"]))
    return f, K, f["b"]


def run(path):
    f, K, b = load(path)
    rows = []
    with Phase("topology", rows):
        top = corvid.Topology(f["dof_entity"], f["dof_kind"], f["edges"], f["tris"],
                              wall_vertices=f["wall_vertices"], coords=f["nodes"])
    sol = corvid.Solver(top, method="cocg", threads=NTHREADS)
    _kernels.timers()
    with Phase("factorize", rows):
        sol.factorize(K, k0=float(f["k0"]))
    tf = dict(_kernels.timers())
    with Phase("solve", rows):
        x = sol.solve(b)
    ts = dict(_kernels.timers())
    its = [i["iterations"] for i in sol.info]
    res = np.linalg.norm(K @ x - b.reshape(x.shape), axis=0) / np.linalg.norm(b.reshape(x.shape), axis=0)
    print(f"\n{os.path.basename(path)}  n={K.shape[0]}  nnz={K.nnz}  iterations {its}  residual {np.max(res):.1e}"
          f"  exact rows {sol.n_exact}  blocks {sol.memory_bytes / 1e6:.0f} MB")
    for name, wall, use in rows:
        print(f"  {name:<12} {wall:7.2f} s   cores {use:4.1f}")
    for label, t in (("factorize", tf), ("solve", ts)):
        parts = "  ".join(f"{k} {v:.2f}" for k, v in t.items() if v > 0.005)
        print(f"  {label} timers [s]: {parts}")
    return rows, its


if __name__ == "__main__":
    want = sys.argv[1:]
    paths = sorted(glob.glob(os.path.join(DATA, "weng_*GHz.npz")))
    if want:
        paths = [p for p in paths if any(f"weng_{w}GHz" in p for w in want)]
    print(f"corvid {corvid.__version__}  threads {NTHREADS}")
    for p in paths:
        run(p)
