"""Time breakdown along a horn sweep (deltamax 2): edges/lambda, iterations, factorize / solve time,
exact rows (walls), with and without the wall rule, and a direct solve for reference."""
import time, numpy as np, corvid
from common import Problem, sparta_nnz
from proto import direct
for pid in range(1, 7):
    p = Problem("horn_d2_sweep", pid)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    out = []
    for label, k0 in (("walls", p.k0), ("no walls", None)):
        t = time.time(); s = corvid.Solver(top, method="cocg", maxiter=500).factorize(p.K, k0=k0); tf = time.time() - t
        t = time.time(); s.solve(p.b); ts = time.time() - t
        out.append(f"{label}: {s.info[0]['iterations']:3d} its, factorize {tf:4.2f}s + solve {ts:4.2f}s = {tf+ts:4.2f}s (+{s.n_wall_exact} wall rows)")
    t = time.time(); d = direct(p.K); x = d.solve(np.ascontiguousarray(p.b, dtype=complex)); td = time.time() - t
    print(f"f={p.k0*299792458/2/np.pi/1e9:6.1f} GHz n={p.n} epl {p.edges_per_wavelength()[0]:.1f} | " + " | ".join(out) + f" | direct {td:4.2f}s", flush=True)
