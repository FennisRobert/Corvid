# Wiring CORVID into EMerge

CORVID solves the system EMerge already hands its solvers, `A[solve_ids][:, solve_ids] x = b`.
On top of that it needs only the mesh's edge and triangle node lists, once per mesh.
No coordinates, tets, materials, gradient matrices or boundary information.

## 1. What CORVID needs

| Data | Where it is in EMerge | When | Size |
|---|---|---|---|
| `edges`, int (2, E) | `mesh.edges` | once per mesh | 2E ints |
| `tris`, int (3, F) | `mesh.tris` | once per mesh | 3F ints |
| `solve_ids`, int (NS,) | already passed to `SolveRoutine.solve` | once per mesh (same for every frequency) | NS ints |
| `nodes`, float (3, V) *(optional, 0.3.0)* | `mesh.nodes` | once per mesh | 3V floats |
| `k0`, float *(optional, 0.3.0)* | free-space wavenumber 2πf/c₀ of the solve | per frequency | 1 float |
| `A` = K on the solved DOFs | the `Asel` / `Asorted` given to `Solver.solve` | per frequency | already there |
| `b`, (NS,) or (NS, nrhs) | `bsel` | per frequency | already there |

For model2 that's about 1 MB of mesh data for an 88,528-unknown system. Index arrays can be int32 or int64, and
shape (2, E) or (E, 2).

**Layout assumption** (`Topology.nedelec2`): global DOFs come in four blocks of second-order Nédélec,
`[edge DOF 1 | edge DOF 2 | face DOF 1 | face DOF 2]`. Each block is in `mesh.edges` / `mesh.tris` order, and the first
block is the lowest-order (Whitney) edge DOF. That's the order `compute_global_dofcodes()` reports (64, 66, 128, 129).
The ML export's discrete gradient confirms the first block is Whitney: its rows there are exactly −1/+1 from
`edges[0]` to `edges[1]`.

If the layout ever changes, use the general constructor, which takes the information per solved DOF:

```python
corvid.Topology(dof_entity, dof_kind, edges, tris)
# dof_entity[i]: edge index (kinds 0, 1) or triangle index (kind 2) of solved DOF i
# dof_kind[i]:   0 = Whitney edge DOF, 1 = other edge DOF, 2 = face DOF
```

## 2. The API

```python
import corvid

top = corvid.Topology.nedelec2(mesh.edges, mesh.tris, solve_ids, nodes=mesh.nodes)   # once per mesh, ~ms
sol = corvid.Solver(top, method="gmres").factorize(Asel, k0=k0)                    # once per frequency
x = sol.solve(bsel)               # (NS,) or (NS, nrhs); one Krylov solve per column
sol.info                          # per column: {"iterations", "residual", "converged"}
sol.memory_bytes                  # vertex-block storage (the dominant memory)
```

Defaults: `order=1` (plain damped vertex-block Schwarz, the cheapest), `rtol=1e-6` (true relative residual),
`maxiter=500`, `restart=100`, `exact="auto"`, `pml_depth=2`.

**PML.** With `exact="auto"`, rows whose entries are mostly complex (a PML's volume, where the coordinate
stretching shrinks the local wavelength by |s|) are solved exactly together with the Whitney rows. Without that,
PML problems stall. Since 0.2.1 only rows at least `pml_depth=2` mesh edges deep into the PML (graph distance from
the non-PML region, computed from `edges`) are included: the outer layer is barely stretched and the vertex blocks
handle it. On the model4 horn this takes p1/p4/p7 from 57/77/134 to 41/53/93 iterations with a smaller exact
factor; `pml_depth=1` restores 0.2.0. EMerge knows which tets are PML, so passing `exact=<bool mask over the solved
unknowns>` is the precise alternative. Use it if lossy dielectrics are present, because those are detected as
complex too: that's harmless, but they would be solved exactly. `sol.n_exact` reports how many rows were added.

**Metal walls (0.3.0).** With `nodes` and `k0`, every higher-order unknown that touches a metal vertex
(a vertex carrying a DOF that EMerge eliminated, i.e. PEC) whose longest incident edge has k0·h > `wall_kh=1.0`
is also solved exactly. On coarse meshes the slowest error modes hug the metal walls (measured on the model4
horn: 60–96 % of their energy next to PEC). model4 p1/p4/p7: 41/53/93 → 24/29/32 iterations, 3.6/4.0/6.2 s →
3.0/2.9/2.9 s; finely meshed metal (model2, model3) selects nothing and is unchanged. Without `nodes` or `k0`
the solver behaves as 0.2.1. Note: k0 is the vacuum wavenumber, so walls inside dielectrics are judged by
their vacuum kh (conservative).

**Reading the exact-row counts (0.4.1).** `sol.n_exact` is the sum of two parts: `sol.n_pml_exact` (strong PML,
fixed for a given geometry) and `sol.n_wall_exact` (metal walls with k0·h > 1). The wall part grows with frequency
along a sweep by design (more of the mesh is coarse relative to the wavelength); with a soft PML (deltamax ≈ 2) the
PML part is 0 and the whole count is walls. Imaginary parts below 1e-8 relative (round-off, seen down to 1e-27 in
EMerge matrices) are ignored by the PML detection, and fewer than 2 % complex-dominated rows count as no PML.

**Mesh density.** The exact coarse level is first order, so it needs roughly 5 or more edges per wavelength. At
about 3 per wavelength, iterations climb from ~13 to ~200. Use `method="gmres"`, or `"cocg"` when A is complex symmetric (every non-periodic EMerge system is).
COCG stores no Krylov basis and needs 0–2 more iterations.

## 3. Proposed wiring

### 3a. A solver class, next to `SolverSparta` in `_emerge/solver.py`

```python
class SolverCorvid(Solver):
    """CORVID: vertex-block Schwarz + Whitney-level coarse solve, Krylov outer."""

    req_sorter = False          # rows must stay in solve_ids order
    real_only = False
    stype = SolverType.SINGLE_MP
    name = "Corvid"

    def __init__(self, pre: str):
        super().__init__(pre)
        self.own_preconditioner = True
        self._mesh = None          # (edges, tris), set once per mesh
        self._top = None           # corvid.Topology, cached per solve_ids
        self._top_key = None
        self.method = "gmres"

    def set_mesh(self, edges, tris, solve_ids, nodes=None) -> None:
        key = (id(edges), solve_ids.shape[0], hash(solve_ids.tobytes()))
        if key != self._top_key:
            self._top = corvid.Topology.nedelec2(edges, tris, solve_ids, nodes=nodes)
            self._top_key = key

    def set_frequency(self, frequency: float) -> None:
        self.k0 = 2 * np.pi * frequency / 299792458.0      # enables the metal-wall rule (0.3.0)

    def set_symmetry(self, complex_symmetric: bool) -> None:
        self.method = "cocg" if complex_symmetric else "gmres"

    def solve(self, A, b, precon, reuse_factorization=False, id=-1):
        sol = corvid.Solver(self._top, method=self.method).factorize(A, k0=getattr(self, "k0", None))
        x = sol.solve(b)
        ok = all(i["converged"] for i in sol.info)
        its = [i["iterations"] for i in sol.info]
        logger.info(f"{_pfx(self.pre, id)} CORVID {self.method}: iterations {its} "
                    f"(exact rows: PML {sol.n_pml_exact}, metal walls {sol.n_wall_exact})")
        return x, SolveReport(solver=str(self), exit_code=0 if ok else 1, aux={"iterations": its})
```

### 3b. Getting the mesh to the solver

`SolveRoutine.solve` already receives `solve_ids`. It needs one more optional argument, set in the same places
`mldataset` is threaded through:

1. **Assembler** (where `SimJob(...)` is created, `assembler.py` ~line 813): attach `(mesh.edges, mesh.tris, mesh.nodes)`
   to the SimJob, e.g. `SimJob(..., corvid_mesh=(mesh.edges, mesh.tris, mesh.nodes))`. It's the same object for
   every frequency; the SimJob already carries the frequency for `set_frequency`.
2. **Job runners** (`microwave_3d.py`: `run_job_single`, `run_job`, and the multiprocessing worker at line ~152):
   pass `corvid_mesh=job._corvid_mesh` into `routine.solve(...)`.
3. **`SolveRoutine.solve`**: after choosing the solver, before calling it:
   ```python
   if isinstance(solver, SolverCorvid):
       if corvid_mesh is None or has_periodic_reduction:
           solver = <the default direct solver>          # see 4.
       else:
           solver.set_mesh(corvid_mesh[0], corvid_mesh[1], solve_ids, nodes=corvid_mesh[2])
           solver.set_frequency(job.freq)
   ```

### 3c. Fall back to direct when

- **Periodic boundaries.** `_reduce_periodic` changes K and `solve_ids`, so the four-block layout no longer
  describes the rows. That's not supported in version 1.
- **No convergence** (`exit_code == 1`). Re-solve with the direct solver and log a warning. It hasn't happened on
  any test problem, but it's the safe default while CORVID is new.
- **Any other element order** than second-order Nédélec.

### 3d. Thin conductors (split DoFs) are supported

A `ThinConductor` gives the tets on one side of its sheet a copied set of DoFs on the sheet's interior edges
and triangles. EMerge's `PreconData.from_field` describes this as a cut through the mesh:
- **Copied vertices.** Every interior sheet vertex gets a copy, at the same coordinates.
- **Copied DoFs.** The copied DoFs live on copies of the sheet edges and triangles that run between copied
  vertices.
- **Re-attached neighbours.** Edges and faces that touch the sheet on the copied side are attached to the
  copied vertices.

Every DoF then belongs to one edge or triangle of the cut mesh again, so CORVID needs no change. Its vertex
blocks stay on one side of the sheet, and the copied Whitney edges join the coarse level. On the Weng filter
with copper traces (2,356 copied vertices), the S-parameters match the direct solver to 6e-8, with the same
iteration counts as the PEC version. `bench/weng/compare_copper.py` repeats the check.

## 4. Things to know

- **Threads:** CORVID's kernels and its SPARTA solves share one pool, 8 threads by default (`RAYON_NUM_THREADS`
  overrides). Run CORVID with one EMerge solve worker. The solver doesn't release the GIL yet, so thread-parallel
  jobs would run one after another anyway, and processes would each start their own 8 threads.
- **Memory (0.4.0):** two parts, the vertex blocks and the exact level. Vertex blocks are stored as sparse LDLᵀ
  factors (minimum-degree ordering), about 40 % of the dense inverses of earlier versions: model2 74 MB (was 183),
  model3 44 MB (was 94); `sparse_blocks=False` restores the dense inverses. Blocks lying entirely inside the exact
  level are dropped (0.3.1). Total memory relative to a direct LDLᵀ of the same system: model2 31 %, model3 44 %,
  model4 horn with EMerge's default PML (deltamax 8) 75 %, the same horn with `deltamax=2` 42 %.
  **Soft PMLs are much cheaper:** with deltamax ≈ 2 (theoretical reflection about −50 dB at normal incidence for a
  0.6 λ thick layer) the PML no longer needs the exact solve; `exact="auto"` detects this from the matrix
  (`sol.pml_stretch`, the 99th percentile of the estimated |s|, below `pml_strong=2.5`).
- **Setup per frequency:** about 0.45 s on model2, almost all of it inverting the vertex blocks (parallel).
  The solve itself takes 0.17 s for 9–10 iterations.
- **SPARTA:** CORVID builds against SPARTA's Rust crate (GitHub commit `a178949`, CPU backend, double precision)
  for its coarse and block factorizations. `pip install .` fetches it, so the build needs git and network access.
  On macOS it also needs the Xcode command-line tools, because SPARTA compiles its Metal shaders even though CORVID
  only uses the CPU path.

## 5. Measured (tests/test_api.py, exactly the inputs above)

| Problem | Unknowns | RHS | GMRES its | COCG its | factorize | solve |
|---|---|---|---|---|---|---|
| model1 p1 | 1,692 | 2 | 8, 7 | 9, 8 | 0.02 s | 0.01 s |
| model2 p1 (second-order ABC) | 88,528 | 1 | 9 | 10 | 0.46 s | 0.17 s |
| model3 p9 | 62,122 | 2 | 10, 10 | 12, 12 | 0.38 s | 0.20 s |

Over all the stored problems (model1 p1–11, model2 p1–7, model3 p1–11), the counts stay between 7 and 10 GMRES
iterations, for k0 from 4 to 210.
