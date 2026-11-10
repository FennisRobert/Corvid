"""corvid sparse vs dense vertex blocks: same preconditioner, memory, time; total vs direct."""
import time, numpy as np, corvid
from common import Problem, sparta_nnz
from check_wall import AP
for name, pid in (("model4_d2", 1), ("model4_d2_r15", 1), ("model4", 1), ("model4", 7), ("model2_abc2", 1), ("model3", 9), ("box_abc_r30", 1)):
    p = Problem(name, pid) if name.startswith(("model4", "box")) else AP(name, pid)
    top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
    direct = sparta_nnz(p.K)[0] * 16 / 1e6
    out, ys = [], []
    for sparse in (False, True):
        t = time.time(); s = corvid.Solver(top, sparse_blocks=sparse).factorize(p.K, k0=p.k0); tf = time.time() - t
        t = time.time(); s.solve(p.b); ts = time.time() - t
        ys.append(s._pc.apply(np.ascontiguousarray(p.b, dtype=complex)))
        m = np.zeros(p.n, bool); m[p.top.whitney] = True; m |= s._exact_rows(p.K)
        ex = sparta_nnz(p.K[m][:, m])[0] * 16 / 1e6
        tot = ex + s.memory_bytes / 1e6
        out.append(f"{'sparse' if sparse else 'dense '}: {s.info[0]['iterations']:3d} its, blocks {s.memory_bytes/1e6:4.0f} MB, total {tot:4.0f} MB "
                   f"({tot/direct:4.0%} of direct), factorize {tf:.1f}s solve {ts:.1f}s" + (f", {s._pc.dense_fallbacks} dense fallbacks" if sparse else ""))
    diff = np.linalg.norm(ys[0] - ys[1]) / np.linalg.norm(ys[0])
    print(f"{name:14s} p{pid} n={p.n:6d} direct {direct:4.0f} MB | preconditioner difference {diff:.0e}\n    " + "\n    ".join(out), flush=True)
