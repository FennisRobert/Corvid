"""O1: exact level = Whitney + deep PML (0.2.1) + complement rows next to metal (PEC), optionally only
inside the horn region. model4 p1, p4, p7."""
import time
import numpy as np, corvid
from common import Problem, sparta_nnz

def near_pec_rows(p):
    ne, nt = p.edges.shape[1], p.tris.shape[1]
    elim = np.setdiff1d(np.arange(2 * ne + 2 * nt), p.sid)
    kind_g = np.where(elim < ne, 0, np.where(elim < 2 * ne, 1, 2))
    ent_g = np.where(elim < ne, elim, np.where(elim < 2 * ne, elim - ne, np.where(elim < 2 * ne + nt, elim - 2 * ne, elim - 2 * ne - nt)))
    pec_v = np.zeros(p.nodes.shape[1], bool)
    pec_v[p.edges[:, ent_g[kind_g < 2]].ravel()] = True
    pec_v[p.tris[:, ent_g[kind_g == 2]].ravel()] = True
    en = p.edges.T[np.minimum(p.ent, ne - 1)]; fn = p.tris.T[np.minimum(p.ent, nt - 1)]
    dv = np.where(p.kind[:, None] < 2, np.c_[en, en[:, :1]], fn)
    return pec_v[dv].any(1)

if __name__ == "__main__":
    for pid in (1, 4, 7):
        p = Problem("model4", pid); K = p.K.tocsr()
        base = corvid.Solver(p.top)._exact_rows(K)
        W = p.kind > 0
        npec = near_pec_rows(p)
        horn = (p.loc[0] >= 0) & (p.loc[0] < 0.019)
        full = sparta_nnz(K)[0]
        print(f"model4 p{pid}: n={p.n}, direct LDL {full/1e6:.1f}M", flush=True)
        for label, extra in (("0.2.1 (deep PML)", np.zeros(p.n, bool)),
                             ("+ W next to PEC in horn", npec & horn & W),
                             ("+ W next to PEC everywhere", npec & W),
                             ("+ all W in horn region", horn & W)):
            ex = base | extra
            t = time.time()
            s = corvid.Solver(p.top, exact=ex, maxiter=200, restart=200).factorize(K); s.solve(p.b)
            mm = ex.copy(); mm[p.top.whitney] = True
            print(f"  {label:28s} +{extra.sum():6d} rows  exact LDL {sparta_nnz(K[mm][:, mm])[0]/1e6:5.1f}M  its {s.info[0]['iterations']:3d}  {time.time()-t:.1f}s", flush=True)
