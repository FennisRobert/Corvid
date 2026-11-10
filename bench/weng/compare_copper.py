"""Weng filter with copper (ThinConductor, split DoFs) traces: S-parameters from the direct solver and
from CORVID over 5 points of the sweep. Run with EMerge's Python: .../python bench/weng/compare_copper.py"""
import os, sys
os.environ["WENG_TRACE"] = "copper"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import weng_model as wm
from emerge_config import config
config.set_acc_threads(8)
res = {}
for method in ("direct", "iterative"):
    sim = wm.build(71, name=f"cu_{method}")
    step = 4.0e9 / 70
    sim.mw.set_frequency_range(0.5e9 + 30 * step, 0.5e9 + 34 * step, 5)
    sim.mw.set_method(method)
    d = sim.mw.run_sweep(nthreads=8)
    g = d.scalar.grid
    res[method] = (np.asarray(g.S(1, 1)), np.asarray(g.S(2, 1)), np.asarray(g.freq))
f = res["direct"][2] / 1e9
for name, k in (("S11", 0), ("S21", 1)):
    a, b = res["direct"][k], res["iterative"][k]
    print(f"### {name}: max |direct - corvid| = {np.max(np.abs(a - b)):.2e}")
    for fi, x, y in zip(f, a, b):
        print(f"###   {fi:.3f} GHz  direct {20*np.log10(abs(x)):8.3f} dB  corvid {20*np.log10(abs(y)):8.3f} dB")
