"""Package check of corvid 0.3 wall rule: Topology.nedelec2(..., nodes) + factorize(K, k0)."""
import sys, time
import numpy as np, corvid
from common import Problem
from check_wall import AP
cases = [("model1", 1), ("model2_abc2", 1), ("model3", 9), ("box_abc_r30", 1), ("sm_abc", 1), ("box_d8", 1), ("sm_d8", 1),
         ("model4", 1), ("model4", 4), ("model4", 7), ("model4_d8_r15", 1)]
for name, pid in cases:
    p = Problem(name, pid) if (name.startswith("model4") or name.startswith(("box", "sm"))) else AP(name, pid)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    out = []
    for label, k0 in (("without k0 (= 0.2.1)", None), ("with k0", p.k0)):
        t = time.time()
        s = corvid.Solver(top, maxiter=300, restart=300).factorize(p.K, k0=k0); s.solve(p.b)
        out.append(f"{label}: {s.info[0]['iterations']:3d} its +{s.n_exact:6d} rows {time.time()-t:4.1f}s res {s.info[0]['residual']:.0e}")
    print(f"{name:13s} p{pid} n={p.n:6d} | " + " | ".join(out), flush=True)
