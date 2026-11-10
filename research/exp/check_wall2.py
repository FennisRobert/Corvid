"""O3: wall rule at vertex granularity: wall vertices (carrying an eliminated DOF) whose longest incident
edge has k0 h > kappa; exact level += all complement DOFs touching those vertices."""
import sys, time
import numpy as np, corvid
from common import Problem, sparta_nnz
from check_wall import AP

def wall_vertex_rows(p, kappa):
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    elim = np.setdiff1d(np.arange(2 * ne + 2 * nt), p.sid)
    kind_g = np.where(elim < ne, 0, np.where(elim < 2 * ne, 1, 2))
    ent_g = np.where(elim < ne, elim, np.where(elim < 2 * ne, elim - ne, np.where(elim < 2 * ne + nt, elim - 2 * ne, elim - 2 * ne - nt)))
    nv = p.nodes.shape[1]
    pec_v = np.zeros(nv, bool)
    pec_v[p.edges[:, ent_g[kind_g < 2]].ravel()] = True
    pec_v[p.tris[:, ent_g[kind_g == 2]].ravel()] = True
    X = p.nodes; h = np.linalg.norm(X[:, p.edges[0]] - X[:, p.edges[1]], axis=0)
    vh = np.zeros(nv); np.maximum.at(vh, p.edges[0], h); np.maximum.at(vh, p.edges[1], h)
    sel = pec_v & (p.k0 * vh > kappa)
    en = p.edges.T[np.minimum(p.ent, ne - 1)]; fn = p.tris.T[np.minimum(p.ent, nt - 1)]
    dv = np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)
    return sel[dv].any(1) & (p.kind > 0)

if __name__ == "__main__":
    cases = [("model4", 1), ("model4", 7), ("model4_d8_r15", 1), ("box_abc_r30", 1), ("box_d8", 1), ("model2_abc2", 1), ("model3", 9), ("model1", 1)]
    for name, pid in cases:
        p = Problem(name, pid) if (name.startswith("model4") or name.startswith("box")) else AP(name, pid)
        K = p.K.tocsr()
        base = corvid.Solver(p.top)._exact_rows(K); base = np.zeros(p.n, bool) if base is None else base
        out = []
        for kappa in (None, 0.0, 1.0, 1.3):
            ex = base if kappa is None else base | wall_vertex_rows(p, kappa)
            t = time.time()
            s = corvid.Solver(p.top, exact=ex, maxiter=300, restart=300).factorize(K); s.solve(p.b)
            mm = ex.copy(); mm[p.top.whitney] = True
            out.append(f"{'0.2.1' if kappa is None else 'kh>%.1f' % kappa}: {s.info[0]['iterations']:3d} its {sparta_nnz(K[mm][:, mm])[0]/1e6:5.1f}M {time.time()-t:4.1f}s")
        print(f"{name:13s} p{pid} n={p.n:6d} | " + " | ".join(out), flush=True)
