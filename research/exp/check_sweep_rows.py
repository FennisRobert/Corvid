"""Bug report: 'extra exact rows' climb along a frequency sweep. Per call: PML rows (strength switch),
wall rows (k0 h > wall_kh), and iterations with / without the wall rule; also the smallest relative
imaginary part among detected complex entries (noise check)."""
import sys, numpy as np, corvid
from common import Problem
for name, nf in (("model4", 11), ("sweep_abc_r30", 5), ("sweep_sm_abc", 5), ("sweep_sm_d8", 5)):
    print(name, flush=True)
    for pid in range(1, nf + 1):
        p = Problem(name, pid); K = p.K.tocsr()
        top = corvid.Topology.nedelec2(p.edges, p.tris, p.sid, nodes=p.nodes)
        s = corvid.Solver(top, method="cocg"); s.k0 = p.k0
        pml = corvid.solver.pml_rows(K)
        auto = s._exact_rows(K); walls = s._wall_rows()
        nw = 0 if walls is None else int(walls.sum())
        im = np.abs(K.data.imag); rel = im[im > 0] / np.abs(K.data[im > 0])
        res = []
        for k0 in (p.k0, None):
            sv = corvid.Solver(top, method="cocg", maxiter=300).factorize(K, k0=k0); sv.solve(p.b)
            res.append(f"{sv.info[0]['iterations']:3d} its (+{sv.n_exact})")
        print(f"  p{pid:2d} f={p.k0*299792458/2/np.pi/1e9:6.2f} GHz n={p.n} | pml rows {pml.sum():6d} |s|99 {s.pml_stretch:4.1f} | wall rows {nw:5d} | "
              f"min rel Im {rel.min() if rel.size else 0:.1e} | with walls {res[0]} | without walls {res[1]}", flush=True)
