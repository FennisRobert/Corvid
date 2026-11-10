"""Small, fast test problems: an open-ended WR-10 waveguide radiating into an air box
(quarter model: PMC at y = 0, PEC at z = 0), truncated by a PML or by a first-order ABC.

usage: gen_box.py outdir [key=value ...]
  res=0.15        EMerge resolution (fraction of the wavelength)
  delta=8         PML deltamax (<= 0: no PML, ABC on the outer faces)
  exp=1.5         PML profile exponent
  thick=2         PML thickness [mm]
  nl=4            PML mesh layers
  box=3           air box edge [mm] (x length; y, z half widths)
  f=90            frequency [GHz] (f2=, nf= for a sweep f..f2 in nf points)
  profile=standard|phase|magnitude
Runs with EMerge's own Python (<EMerge's python>).
"""
import sys, os
import numpy as np
import scipy.sparse as sp
import emerge as em
from emerge._emerge import solver as _solver

outdir = sys.argv[1]
P = dict(res=0.15, delta=8.0, exp=1.5, thick=2.0, nl=4, box=3.0, f=90.0, f2=0.0, nf=1, phi=0, elem=0, profile="standard")
for a in sys.argv[2:]:
    k, v = a.split("=")
    P[k] = v if k == "profile" else float(v)

_captured = []
_orig = _solver.SolveRoutine.solve
def _capture(self, A, b, solve_ids, *args, **kwargs):
    sol, rep = _orig(self, A, b, solve_ids, *args, **kwargs)
    _captured.append((sp.csr_matrix(A), np.asarray(b), np.asarray(solve_ids), np.asarray(sol)))
    return sol, rep
_solver.SolveRoutine.solve = _capture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pml_profile_patch as _pp
PROFILES = {
    "standard": lambda t, e, d: 1 - 1j * t ** e * d,
    "phase": lambda t, e, d: np.exp(-1j * np.arctan(d) * t ** e),
    "magnitude": lambda t, e, d: (1 + (np.hypot(1, d) - 1) * t ** e) + 0j,
}
_pp.PROFILE = PROFILES[P["profile"]]
sys.modules["emerge._emerge.geo.pmlbox"]._add_pml_layer = _pp._add_pml_layer

os.makedirs(outdir, exist_ok=True)
mm = 1e-3
a, b, Lfeed, B = 2.54 * mm, 1.27 * mm, 2 * mm, P["box"] * mm
model = em.Simulation("BoxPML")
feed = em.geo.Box(Lfeed, a / 2, b / 2, position=(-Lfeed, 0, 0))
if P["delta"] > 0:
    air, *pmls = em.geo.pmlbox(B, B, B, (0, 0, 0), thickness=P["thick"] * mm, N_mesh_layers=int(P["nl"]),
                                sides="TRA", exponent=P["exp"], deltamax=P["delta"])
else:
    air = em.geo.Box(B + P["thick"] * mm, B + P["thick"] * mm, B + P["thick"] * mm, position=(0, 0, 0))
model.mw.set_frequency_range(P["f"] * 1e9, (P["f2"] or P["f"]) * 1e9, int(P["nf"]))
model.mw.set_resolution(P["res"])
model.generate_mesh()
port = model.mw.bc.ModalPort(feed.left, 1)
model.mw.bc.PMC(model.select.face.inplane(0, 0, 0, plane=em.XZPLANE))
if P["delta"] <= 0:
    model.mw.bc.AbsorbingBoundary(air.faces("back", "top", "right"))
model.mw.run_sweep()
mesh = model.mesh
extra = {}
if P["phi"]:
    # Phi: N2 coefficients -> field values at interior sample points (probe the evaluator with
    # unit vectors of a colouring in which no two DOFs of a colour share a tet)
    basis = model.mw.basis
    T2F = np.asarray(basis.tet_to_field)                     # (20, ntets) global DOF ids
    ndof, nt = T2F.max() + 1, T2F.shape[1]
    X = np.asarray(mesh.nodes); tets = np.asarray(mesh.tets)
    bary = np.array([[.25, .25, .25, .25]] + [[.55 if i == j else .15 for i in range(4)] for j in range(4)]
                    + [[.4 if i in (a, b) else .1 for i in range(4)] for a in range(4) for b in range(a + 1, 4)])
    pts = np.einsum("pk,dkt->dtp", bary, X[:, tets]).reshape(3, -1)  # (3, nt * npt), tet-major
    npt = len(bary)
    tmap = np.repeat(np.arange(nt), npt)
    tm = basis.interpolate_index(pts[0], pts[1], pts[2])
    print("mapping agrees:", np.mean(np.asarray(tm).ravel() == tmap))
    # greedy colouring of the DOF "shares a tet" graph
    D2T = sp.csr_matrix((np.ones(T2F.size), (T2F.T.ravel(), np.repeat(np.arange(nt), T2F.shape[0]))), shape=(ndof, nt))
    G = (D2T @ D2T.T).tocsr()
    col = -np.ones(ndof, int)
    for d in range(ndof):
        used = col[G.indices[G.indptr[d]:G.indptr[d + 1]]]
        cnum = 0
        used = set(used[used >= 0].tolist())
        while cnum in used: cnum += 1
        col[d] = cnum
    rows, cols, vals = [], [], []
    for cc in range(col.max() + 1):
        fld = (col == cc).astype(complex)
        v = np.asarray(basis.interpolate(fld, pts[0], pts[1], pts[2], tm)).reshape(3, -1)
        act = np.where(col[T2F] == cc, T2F, -1).max(0)        # active DOF per tet (or -1)
        dof_pt = np.repeat(act, npt)
        ok = dof_pt >= 0
        for comp in range(3):
            rows.append(3 * np.flatnonzero(ok) + comp); cols.append(dof_pt[ok]); vals.append(v[comp, ok])
    Phi = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * pts.shape[1], ndof))
    print("Phi", Phi.shape, Phi.nnz, "colours", col.max() + 1)
    extra = dict(Phi_data=Phi.data, Phi_indices=Phi.indices, Phi_indptr=Phi.indptr, Phi_shape=np.array(Phi.shape), Phi_pts=pts)
if P["elem"]:
    # element matrices E_T, B_T (vacuum tensors: valid for air-only models) in EMerge's own COO order
    import emerge._emerge.physics.microwave.assembly.emr_microwave as emw
    basis = model.mw.basis
    nt = mesh.tets.shape[1]
    I3 = np.repeat(np.eye(3, dtype=np.complex128)[:, :, None], nt, axis=2)
    dE, dB, rr, cc = emw.curlcurl._matrix_builder(mesh.nodes, mesh.tets, mesh.tris, mesh.edges, basis.tet_to_field,
                                                  mesh.tet_to_edge, mesh.tet_to_tri, I3, I3, np.arange(nt), basis.dofcodes3d)
    print("element COO", np.asarray(dE).shape, "per tet", np.asarray(dE).size / nt)
    extra.update(elem_E=np.asarray(dE), elem_B=np.asarray(dB), elem_rows=np.asarray(rr), elem_cols=np.asarray(cc),
                 tet_to_field=np.asarray(basis.tet_to_field))
for i, (A, rhs, sid, sol) in enumerate(_captured, 1):
    np.savez(os.path.join(outdir, f"problem_{i}.npz"), A_data=A.data, A_indices=A.indices, A_indptr=A.indptr,
             A_shape=np.array(A.shape), b=rhs, solve_ids=sid, solution=sol,
             mesh_edges=np.asarray(mesh.edges), mesh_tris=np.asarray(mesh.tris), mesh_nodes=np.asarray(mesh.nodes),
             mesh_tets=np.asarray(mesh.tets), params=np.array(repr(P)), **extra, k0=2 * np.pi * (P["f"] + (i - 1) * ((P["f2"] or P["f"]) - P["f"]) / max(int(P["nf"]) - 1, 1)) * 1e9 / 299792458.0)
print("done", outdir, {k: v for k, v in P.items()})
