"""O2: wall rows (complement unknowns touching a vertex with an eliminated DOF) on all datasets."""
import sys, time
import numpy as np, corvid
from common import Problem, sparta_nnz
from exp_wall_exact import near_pec_rows
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs

class AP:  # aegis problem with the fields near_pec_rows needs
    def __init__(self, m, pid):
        import numpy as np
        f = np.load(f"data/aegis/{m}/problem_{pid}.npz")
        self.edges, self.tris, self.sid, self.K, self.b = emerge_inputs(m, pid)
        self.nodes = f["mesh_nodes"]; self.n = self.K.shape[0]
        ne, nt = self.edges.shape[1], self.tris.shape[1]; g = self.sid
        self.kind = np.where(g < ne, 0, np.where(g < 2 * ne, 1, 2))
        self.ent = np.where(g < ne, g, np.where(g < 2 * ne, g - ne, np.where(g < 2 * ne + nt, g - 2 * ne, g - 2 * ne - nt)))
        self.top = corvid.Topology.nedelec2(self.edges, self.tris, self.sid)
        self.k0 = float(f["k0"])
        self.b = self.b if self.b.ndim == 1 else self.b[:, 0]

if __name__ == "__main__":
    cases = [("model4_d8_r15", 1), ("box_d8", 1)]
    for name, pid in cases:
        p = Problem(name, pid) if (name.startswith("model4") or name.startswith("box")) else AP(name, pid)
        K = p.K.tocsr()
        base = corvid.Solver(p.top)._exact_rows(K)
        base = np.zeros(p.n, bool) if base is None else base
        wall = near_pec_rows(p) & (p.kind > 0)
        X = p.nodes; h = np.linalg.norm(X[:, p.edges[0]] - X[:, p.edges[1]], axis=0)
        ne = p.edges.shape[1]
        # local edge length per DOF: the edge itself, or the mean edge of the triangle
        T = p.tris
        ht = (np.linalg.norm(X[:, T[0]] - X[:, T[1]], axis=0) + np.linalg.norm(X[:, T[1]] - X[:, T[2]], axis=0) + np.linalg.norm(X[:, T[0]] - X[:, T[2]], axis=0)) / 3
        k0 = 2 * np.pi * 90e9 / 299792458.0 if not hasattr(p, "k0") else p.k0
        kh = k0 * np.where(p.kind < 2, h[np.minimum(p.ent, ne - 1)], ht[np.minimum(p.ent, T.shape[1] - 1)])
        out = []
        for label, ex in (("0.2.1", base), ("+wall kh>1.0", base | (wall & (kh > 1.0))), ("+wall kh>1.2", base | (wall & (kh > 1.2)))):
            t = time.time()
            s = corvid.Solver(p.top, exact=ex, maxiter=300, restart=300).factorize(K); s.solve(p.b)
            mm = ex.copy(); mm[p.top.whitney] = True
            out.append(f"{label}: its {s.info[0]['iterations']:3d} exact LDL {sparta_nnz(K[mm][:, mm])[0]/1e6:5.1f}M {time.time()-t:5.1f}s")
        print(f"{name:14s} p{pid} n={p.n:6d} wall {wall.mean():4.0%} kh>1: {(wall & (kh > 1.0)).mean():4.0%} kh>1.2: {(wall & (kh > 1.2)).mean():4.0%} | " + " | ".join(out), flush=True)
