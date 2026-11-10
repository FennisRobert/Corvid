"""Headless model4 (horn + PML) at one frequency with a chosen PML strength.
usage: gen_model4_pml.py deltamax exponent outdir"""
import sys, os
import emerge as em
import numpy as np
import scipy.sparse as sp
from emerge._emerge import solver as _solver

_captured = []
_orig = _solver.SolveRoutine.solve
def _capture(self, A, b, solve_ids, *args, **kwargs):
    sol, rep = _orig(self, A, b, solve_ids, *args, **kwargs)
    _captured.append((sp.csr_matrix(A), np.asarray(b), np.asarray(solve_ids), np.asarray(sol)))
    return sol, rep
_solver.SolveRoutine.solve = _capture
deltamax, exponent, outdir = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
res = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
profile = sys.argv[5] if len(sys.argv) > 5 else "standard"
nfreq = int(sys.argv[6]) if len(sys.argv) > 6 else 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pml_profile_patch as _pp
_pmlbox_mod = sys.modules["emerge._emerge.geo.pmlbox"]
PROFILES = {
    "standard": lambda t, e, d: 1 - 1j * t ** e * d,                              # EMerge's
    "phase":    lambda t, e, d: np.exp(-1j * np.arctan(d) * t ** e),              # |s| = 1, same max phase
    "magnitude": lambda t, e, d: (1 + (np.hypot(1, d) - 1) * t ** e) + 0j,        # real, same max |s|
}
_pp.PROFILE = PROFILES[profile]
_pmlbox_mod._add_pml_layer = _pp._add_pml_layer
os.makedirs(outdir, exist_ok=True)
mm = 0.001
wga, wgb, WH, HH, Lhorn = 2.01 * mm, 1.01 * mm, 10 * mm, 7 * mm, 21 * mm
Lfeed, th, dx = 2 * mm, 1 * mm, 2 * mm
model = em.Simulation("HornPML")
hornCS = em.CS(em.YAX, em.ZAX, em.XAX)
feed = em.geo.Box(Lfeed, wga / 2, wgb / 2, position=(-Lfeed, 0, 0))
horn_in = em.geo.Horn((wga, wgb), (WH, HH), Lhorn, hornCS)
horn_out = em.geo.Horn((wga + 2 * th, wgb + 2 * th), (WH + 2 * th, HH + 2 * th), Lhorn, hornCS)
ibox = em.geo.Box(30 * mm, 30 * mm, 30 * mm)
horn_in = em.geo.intersect(horn_in, ibox, remove_tool=False)
horn_out = em.geo.intersect(horn_out, ibox)
rat = 1.6
if deltamax > 0:
    air, *pmls = em.geo.pmlbox(4 * mm, rat * WH / 2, rat * HH / 2, (Lhorn - dx, 0, 0), thickness=2 * mm,
                                N_mesh_layers=4, sides="TRA", exponent=exponent, deltamax=deltamax)
else:   # deltamax <= 0: the same outer box, plain air, absorbing boundary on the outer faces
    air = em.geo.Box(4 * mm + 2 * mm, rat * WH / 2 + 2 * mm, rat * HH / 2 + 2 * mm, position=(Lhorn - dx, 0, 0))
    air.max_meshsize = 0.5 * mm
air2 = em.geo.subtract(air, horn_out)
model.mw.set_frequency_range(90e9, 110e9 if nfreq > 1 else 90e9, nfreq)
model.mw.set_resolution(res)
model.generate_mesh()
p1 = model.mw.bc.ModalPort(feed.left, 1)
PMC = model.mw.bc.PMC(model.select.face.inplane(0, 0, 0, plane=em.XZPLANE))
abc = model.mw.bc.AbsorbingBoundary(model.select.face.inplane(Lhorn - dx, 0, 0, plane=em.YZPLANE))
if deltamax <= 0:
    outer = model.mw.bc.AbsorbingBoundary(air2.faces("back", "top", "right", tool=air))
data = model.mw.run_sweep()
mesh = model.mesh
for i, (A, b, sid, sol) in enumerate(_captured, 1):
    np.savez(os.path.join(outdir, f"problem_{i}.npz"), A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
             A_shape=np.array(A.shape), b=b, solve_ids=sid, solution=sol,
             mesh_edges=np.asarray(mesh.edges), mesh_tris=np.asarray(mesh.tris), mesh_nodes=np.asarray(mesh.nodes), k0=2 * np.pi * (90e9 + (i - 1) * (20e9 / max(nfreq - 1, 1))) / 299792458.0)
print("done", outdir)
