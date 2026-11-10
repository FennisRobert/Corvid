"""Wei-Chung Weng coupled-line filter (from EMerge_Numerics_local/Comparisons/wei_chung_weng), on a
coarser mesh for Corvid development. Runs with EMerge's Python:

    <EMerge's python> bench/weng/weng_model.py dump      # save the solver inputs
    <EMerge's python> bench/weng/weng_model.py sweep     # time a short sweep
    WENG_DUMP_DIR=lo .../python bench/weng/weng_model.py dumpsweep 0 6     # points 0-5 of the 71-point sweep

`dump` wraps SolverCorvid.solve so that the exact inputs EMerge gives Corvid (K, b, topology, k0) are
written to bench/weng/data/weng_<f>GHz.npz, for bench_corvid.py. All runs use at most 8 threads.
"""
import os
import sys
import time

import numpy as np
import emerge as em
from emerge_config import config

NTHREADS = 8
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", os.environ.get("WENG_DUMP_DIR", ""))
mm = 0.001

# Mesh density knobs (the original script: resolution 0.1, trace quality 3, boundary size w50/5)
RESOLUTION = 0.15
TRACE_QUALITY = 1.5
BOUNDARY_DIV = 2.5


def build(npoints, name="weng_bench"):
    L1, L2, L3, L4, L5, L6 = 18.39 * mm, 4.45 * mm, 13.67 * mm, 1.51 * mm, 10.30 * mm, 1.99 * mm
    w2, a, g, w50, w1, w3, L50 = 0.40 * mm, 0.32 * mm, 0.12 * mm, 1.77 * mm, 0.2 * mm, 0.635 * mm, 6 * mm
    h = 0.813 * mm
    mat = em.Material(3.55, tand=0.0021, color="#e7d3a7")

    sim = em.Simulation(name, loglevel="INFO")
    # WENG_TRACE=copper: ThinConductor traces (EMerge splits the DoFs on both sides of the sheet)
    trace = em.lib.COPPER if os.environ.get("WENG_TRACE", "pec").lower() == "copper" else em.lib.PEC
    pcb = em.geo.PCB(h, 1.0, material=mat, trace_material=trace)
    pcb.new(0, 0, w50, (0, -1), trace_layer=1)[1].straight(L50).taper(L6, width=w3).jump(
        -w3 / 2, -w1 / 2, width=w1, direction=(1, 0)
    ).straight(L1 + w3).jump(side="right", gap=g, reverse=L1).straight(L1)["p1"].straight(a + w2).jump(
        -w2 / 2, w1 / 2, width=w2, direction=(0, 1)
    ).straight(L2).turn(-90).straight(L3)
    pcb.new(pcb["p1"].xy[0], pcb["p1"].xy[1] - g - w1, w1, (-1, 0), trace_layer=1).straight(L1 - w1).turn(
        -90
    ).straight(L4 + w1)["p2"].jump(w1 / 2, w1 / 2, w1, (1, 0)).straight(L5)
    mx, my = pcb["p2"].xy
    my = my + w1 / 2
    traces = pcb.compile_paths(merge=True)
    lp1 = pcb.modal_port(1, 1, 3 * mm)
    traces_mirror = em.geo.mirror(traces, (mx, my, 0), (0, 1, 0), make_copy=True)
    lp2 = em.geo.mirror(lp1, (mx, my, 0), (0, 1, 0), make_copy=True)
    pcb.determine_bounds(5 * mm, 0, 5 * mm, 0)
    ox, oy = pcb.origin[0], pcb.origin[1]
    w, l = pcb.width, pcb.length
    pcb.set_bounds(ox, oy - l, ox + w, oy + l)
    pcb.generate_pcb()
    air = pcb.generate_air(6 * mm)
    all_traces = em.geo.add(traces, traces_mirror)

    sim.commit_geometry()
    sim.mw.set_frequency_range(0.5e9, 4.5e9, npoints)
    sim.mw.set_resolution(RESOLUTION)
    sim.mesher.set_trace_refinement(all_traces, qualtity_factor=TRACE_QUALITY)
    sim.mesher.set_boundary_size(all_traces, w50 / BOUNDARY_DIV)
    sim.mesher.set_face_size(lp1, h)
    sim.mesher.set_face_size(lp2, h)
    sim.generate_mesh()
    sim.mw.bc.AbsorbingBoundary(air.top)
    sim.mw.bc.ModalPort(lp2, 2, modetype="TEM")
    sim.mw.set_method("iterative")
    return sim


def install_dump():
    """Save Corvid's inputs on every solve."""
    from emerge._emerge.solvers.corvid import SolverCorvid

    os.makedirs(DATA, exist_ok=True)
    orig_set, orig_solve = SolverCorvid.set_precon_data, SolverCorvid.solve

    def set_precon_data(self, pd, solve_ids, k0=None):
        self._dump = dict(dof_entity=pd.dof_entity[solve_ids], dof_kind=pd.dof_kind[solve_ids], edges=pd.edges,
                          tris=pd.tris, wall_vertices=pd.wall_vertices(solve_ids), nodes=pd.nodes,
                          k0=np.nan if k0 is None else k0)
        return orig_set(self, pd, solve_ids, k0)

    def solve(self, A, b, precon, id=-1):
        A = A.tocsr()
        f_ghz = float(self._dump["k0"]) * 299792458.0 / (2 * np.pi) / 1e9
        path = os.path.join(DATA, f"weng_{f_ghz:.2f}GHz.npz")
        np.savez(path, K_data=A.data, K_indices=A.indices, K_indptr=A.indptr, K_shape=A.shape, b=b, **self._dump)
        print(f"### dumped {path}  n={A.shape[0]} nnz={A.nnz}", flush=True)
        return orig_solve(self, A, b, precon, id)

    SolverCorvid.set_precon_data, SolverCorvid.solve = set_precon_data, solve


def patch_corvid_defaults():
    """Benchmark-only: WENG_CORVID="history=8,order=2" overrides corvid.Solver defaults."""
    opts = os.environ.get("WENG_CORVID")
    if not opts:
        return
    import corvid
    kw = {k: (float(v) if "." in v else int(v)) for k, v in (o.split("=") for o in opts.split(","))}
    init = corvid.Solver.__init__

    def patched(self, *a, **k):
        k.update({key: val for key, val in kw.items() if key != "order"})
        init(self, *a, **k)
        if "order" in kw:
            self.order = kw["order"]

    corvid.Solver.__init__ = patched
    print(f"### corvid overrides {kw}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    patch_corvid_defaults()
    config.set_acc_threads(NTHREADS)
    if mode == "dump":
        install_dump()
        sim = build(3)                                          # low, middle and top of the band
        sim.mw.run_sweep(nthreads=NTHREADS)
    elif mode == "dumpsweep":
        # consecutive points i0 .. i0+n-1 of the 71-point sweep (0.5-4.5 GHz), meshed for the full band
        i0, n = int(sys.argv[2]), int(sys.argv[3])
        install_dump()
        sim = build(71)
        step = 4.0e9 / 70
        sim.mw.set_frequency_range(0.5e9 + i0 * step, 0.5e9 + (i0 + n - 1) * step, n)
        sim.mw.run_sweep(nthreads=NTHREADS)
    elif mode == "sweep":
        # sweep.py sweep [direct|iterative] [i0 n]: points i0 .. i0+n-1 of the 71-point sweep (default: 5 points
        # spread over the band)
        method = sys.argv[2] if len(sys.argv) > 2 else "iterative"
        if len(sys.argv) > 4:
            i0, n = int(sys.argv[3]), int(sys.argv[4])
            sim = build(71)
            step = 4.0e9 / 70
            sim.mw.set_frequency_range(0.5e9 + i0 * step, 0.5e9 + (i0 + n - 1) * step, n)
        else:
            sim, n = build(5), 5
        sim.mw.set_method(method)
        t0 = time.perf_counter()
        sim.mw.run_sweep(nthreads=NTHREADS)
        print(f"### sweep ({method}) {n} points: {time.perf_counter() - t0:.1f}s", flush=True)
