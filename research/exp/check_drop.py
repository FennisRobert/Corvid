"""Package check: corvid with blocks inside the exact level dropped, all datasets."""
import time, numpy as np, corvid
from common import Problem
from check_wall import AP
cases = [("model1", 1), ("model2_abc2", 1), ("model3", 9), ("box_abc_r30", 1), ("sm_abc", 1), ("box_d8", 1), ("sm_d8", 1),
         ("box_d8_b5", 1), ("model4", 1), ("model4", 4), ("model4", 7), ("model4_d8_r15", 1)]
for name, pid in cases:
    p = Problem(name, pid) if (name.startswith(("model4", "box", "sm"))) else AP(name, pid)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    t = time.time()
    s = corvid.Solver(top, maxiter=300, restart=300).factorize(p.K, k0=p.k0); s.solve(p.b)
    print(f"{name:13s} p{pid} n={p.n:6d}: its {s.info[0]['iterations']:3d} res {s.info[0]['residual']:.0e} blocks {s.n_blocks:5d} "
          f"({s.memory_bytes/1e6:5.0f} MB) exact +{s.n_exact:6d}  {time.time()-t:.1f}s", flush=True)
