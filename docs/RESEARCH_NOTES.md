# CORVID — Coarse Operators by Recursive Volume Interpolation of De Rham complexes

Element-agglomeration AMGe for H(curl) (prototype).

Goal: a tiny package (eventual PyPI, AGPL-3.0-or-later, all OSes) that builds an
exact de Rham coarse hierarchy on agglomerated tets (parELAG's method, no code
taken) and a V-cycle for the lowest-order (Whitney) H(curl) problem. The
coarsest level is solved directly: SPARTA (>= 0.2.0, CPU backend on every OS)
when importable, scipy `splu` otherwise. The design background is in
`reference/helix/notes/AMGE_PARELAG.md`.

## Data contract with EMerge (keep minimal)

Required: `coords (V,3) f64`, `tets (T,4) i32`, `edges (E,2) i32` (row order =
Whitney DOF order, oriented edges[e,0] -> edges[e,1]), `free (E,) bool`, and the
Whitney-level operator itself (already in EMerge's memory: the [0,E) block).
Optional: `eps (T,)`, `inv_mu (T,)` weights for the extension masses.
Everything else (faces, orientations, boundary/PEC classes, all coarse levels) is
derived here. The datasets lack tets, so research/probes/emdata.py rebuilds them from
triangles (exact on model1/2/3); EMerge should just send `tets`.

## Layout

    corvid/whitney.py      fine Whitney complex: incidences + closed-form element/face masses
    corvid/level.py        Level: one polyhedral lowest-order complex (fine and coarse alike)
    corvid/agglomerate.py  ball-preserving region growing + coarse AF/AR/AP topology
    corvid/coarsen.py      L2 -> H(div) -> H(curl) traces and saddle-point extensions
    corvid/mg.py           hierarchy(), Hiptmair/Chebyshev V-cycle, direct coarsest solve
    tests/                emdata loader, identity checks, comparison against helix AMS

## Verified (2026-10-23)

Fine complex vs EMerge: curl-curl = D1^T M2 D1 to 1e-14 on model1/2/3; Whitney mass
= M1 to 1e-14 on model1 (vacuum), off by the dielectric eps on model2/3.

Every level, every model: D1c D0c = 0, D2c D1c = 0, D1 P1 = P2 D1c and
D2 P2 = P3 D2c to ~1e-14, coarse masses = P^T M P, PEC rows of free columns = 0.

Topology lessons (beyond parELAG):
- AFs must be cut along every AF boundary curve, iterated, or an edge can be
  interior to one AF and on the boundary of another (P rows overwritten).
- AEs must be manifold balls: Euler counts alone pass pinched AEs (tets around
  an edge alternate AE1/AE2), which lets two AFs share an interior edge.
- So aggregation grows each AE by gluing one element at a time only across a
  single disk of shared faces (union of balls along a disk = ball), never across
  PEC. Root+neighbours aggregation gave ~50 % invalid AEs at size 70.

Coarsening (target 32 elements/AE), free Whitney edges:
model3 9947 -> 2076 -> 429, model2 15167 -> 3908 -> 936 -> 748 (stalls: PEC).

## Result: as the Whitney-level solver inside helix palace_pc.build

Same A1 = Re+Im of R^T P_flip R, outer GMRES(200) to 1e-6 on the full N2 system:

    model3 p1  AMS x2 31 its 1.9 s | AMGe x1 18 its 0.7 s | AMGe x2 16 its 0.7 s
    model2 p1  AMS x2 94 its 8.7 s | AMGe x1 95 its 7.3 s | AMGe x2 84 its 7.8 s
    model1 p1  36 = 36 (Whitney level below the 500-DOF direct threshold)

Hierarchy setup (pure Python): 4.2 s model3, 6.0 s model2.

## Next

- Full reduced problem set (model1 all, model2 all 7, model3 ~10 ids).
- Target-size and smoother-order sweep; coarse-level stall on model2.
- Per-tet eps weights (model2/3 are dielectric).
- Then the Rust port of agglomerate + coarsen (the only slow parts).

## Second-order ABC (model2 p1, 2026-10-23): TALOS's surface-block recipe

The preconditioner applied exactly needs 40 iterations; an exact coarse solve
(complex LU of R^T P R, unflipped) needs 73. That is the ceiling for any
coarse solver, and what is left above 40 is the fine (N2) level.

    fine smoother + explicit surface block GS step (TALOS):  73 / 75  (no gain over helix's
                                                             surface-LU Chebyshev scaling, 73)
    coarse = AMGe on the FLIPPED SPD matrix, Chebyshev, x2   84
    coarse = AMGe on the complex unflipped matrix, Chebyshev o2 + surface block   94 (x1) / 99 (x2)
             same, Chebyshev o4                              diverges (complex spectrum)
             same, no surface block                          131
    coarse = AMGe on the complex unflipped matrix, symmetric GS Hiptmair, x2   73 = ceiling
             (with or without the surface block: GS handles the ABC rows itself)
    AMS (helix today)                                        94

So on the Whitney levels: Gauss-Seidel Hiptmair on the unflipped complex operator,
2 cycles. The coarse level is then solved as well as an exact LU (73 its, 6.2 s vs
AMS 94 its, 8.7 s). The remaining gap (73 -> 40) is the fine N2 smoother, not the
ABC surface. Option 2 (first-order ABC in the preconditioner) was already measured
in helix ("drop": >200 its) and is out.
GS currently lives in research/probes/coarse_gs.py (pyamg); in the package it needs its own
(Rust) sweep.

## Where model2's 6 s goes (p1, CORVID GS x2 coarse, GMRES 73 its)

Not Python: 4.3 of 5.9 s is scipy's compiled csr_matvec (10510 calls), 0.6 s
pyamg's compiled GS, ~0.3 s interpreter. About 29 fine-matrix (3.3M nnz, 2 ms)
SpMV-equivalents per iteration. Threaded Rust would give perhaps 3-5x
(bandwidth bound), i.e. ~1.2-2 s. Direct (SPARTA README: 109k EM system, 0.11 s
GPU factor, ~0.5 s 4-thread CPU) wins at this size.

Total work is pinned by the fine level: trading smoothing for iterations does not help.

    Chebyshev order / cycles      GMRES its   time
    4 / 2  (default)                 73        5.8 s
    4 / 1                           117        5.0 s
    2 / 2                           142        8.7 s
    2 / 1                           293       10.1 s
    1 / 2                           410       21.5 s

Ceilings: exact coarse 73, exact P 40. The lever is the fine N2 smoother (and
the K-vs-P shift gap); CORVID's coarse side is already at its ceiling.

## Correction and the two deficits (model2 p1)

Iterative solvers DO pay off on small problems with a good preconditioner
(~10 its x a few SpMVs = ~40 ms in scipy here). The 6 s means the
preconditioner is not good yet. It has two separate deficits:

1. Shift floor: preconditioning with Palace's shifted P instead of K. Even P^-1
   applied exactly needs 40 its.
2. Fine N2 level: the p-multigrid approximation of P^-1 takes 40 -> 73
   (CORVID's coarse side is already at its exact-LU ceiling).

Probe of multigrid on K itself (research/probes/unshifted_probe.py): pointwise GS Hiptmair on
the fine N2 level diverges for K AND for P, with or without the exact surface
block (300 its, relres 1e-2..3e+1; one config 2.7e-6). Chebyshev with the surface
LU scaling is the only fine smoother so far that works there. The mesh is strongly
graded: 44 edges per wavelength at median h, 4 at max h.

## Threaded Rust sparse products (src/lib.rs, corvid/fast.py), model2 p1

SpMV on the 3.3M-nnz fine matrix: scipy 2.0 ms; Rust 2.5 ms (1 thread),
0.59 ms (8 threads, the default), 0.55 ms (16), slower with all 28 cores.
Saturates at ~130 GB/s. Real-matrix x complex-vector kernel for R, G, P.

Same preconditioner, same iterations (identical residuals), wall time:

    GMRES  73 its   scipy 5.9 s  ->  rust 3.3 s
    COCG   86 its   scipy 6.6 s  ->  rust 3.6 s

Left in the 3.6 s COCG profile: fine/coarse complex SpMV 1.64 s, pyamg serial GS
on the coarse levels 0.77 s, real x complex SpMV 0.40 s, Chebyshev vector updates
in numpy 0.25 s, surface/coarsest SuperLU 0.23 s.
Pure SpMV floor of this algorithm: ~29 fine SpMVs/it x 0.59 ms x 73 its ~ 1.3 s.

## All-Rust solve phase (src/: csr, direct, precond, krylov; corvid._kernels.PSolver)

One-to-one port of the prototype: same iterations (GMRES 73 = 73, COCG 86 = 86 on
model2 p1 with the CORVID GS coarse), preconditioners agree to 4e-4 (lambda_max
start vector). 8 threads by default. Direct solves: SPARTA (GitHub rev a178949)
LDL^T, force_cpu, double precision. pyamg is gone from the solve path; the Rust GS
matches pyamg's to 6e-16.

model2 p1, wall time of the outer solve (setup excluded):

    step                                              GMRES       COCG
    scipy + pyamg prototype                           5.9 s       6.6 s
    + threaded Rust SpMV (fine level only)            3.3 s       3.6 s
    all Rust, CORVID GS coarse                        3.1 s       3.3 s
    coarse = SPARTA LU of the whole Whitney level     2.6 s       2.5 s (82 its)
    + fused Chebyshev step (one pass, no allocs)      2.3 s       2.3 s
    + real CSR + sparse imaginary part (12 vs 20 B)   2.0 s       2.0 s

Tried and rejected: hybrid block GS (8 blocks: 104 its), multicolour GS (same its,
slower: scattered rows), CORVID depth > 0 on model2 (coarse operators get dense:
209k/249k/233k/197k nnz on 15167/3908/936/748 rows, operator complexity 4.2).

Where the 1.9 s (COCG) goes: fine Chebyshev products 0.58, fine residuals 0.26,
Whitney LU solves 0.64 (SPARTA CPU solve ~2.8 ms for nnz(L) = 1.53M, effectively
single-threaded: 3.0 ms at 1 thread), surface LU 0.15, G/G^T 0.11.
SuperLU on the same matrix: L+U 3.8M, factor 0.16 s, solve 2.74 ms; SPARTA
factors with 2.5x less fill, in 0.12 s.

## Vertex-patch Schwarz on the UNSHIFTED K (2026-10-23): the breakthrough

Fine smoother: Chebyshev (order 2) scaled by additive vertex-patch Schwarz
(Arnold-Falk-Winther: per mesh vertex, all N2 DOFs on incident edges and faces,
exact dense inverse), plus Hiptmair's gradient step, 1 cycle. Coarse: exact SPARTA
LDL^T of R^T K R (Whitney level). Operator: K itself, no shift. All Rust.
research/probes/schwarz_sweep.py. GMRES / COCG iterations to 1e-6 (true residual):

    model1 p1-11   (n=1692)    7-8 / 7-8      0.01 s
    model2 p1-7    (n=88528)   9-10 / 10      0.21-0.24 s   (second-order ABC)
    model3 p1-11   (n=62122)   7-10 / 7-10    0.10-0.14 s   (k0 = 4.2 ... 168)

Before (Hiptmair + surface LU on the shifted P): model2 73 its, 2.0 s.
On the shifted P, Schwarz gives 48-70 its: the shift itself was the floor.
Without the gradient step: same iterations (o2: 9-10). Plain damped Schwarz
(Chebyshev order 1): 10-14 its, slightly faster.

Cost: patches mean 87 DOFs (max 172) on model2, dense inverses 22.5M entries =
361 MB (more than the 319 MB direct factor of K), setup ~0.5 s. Direct SPARTA on
K: factor 0.54 s + solve 53 ms. So at this size: comparable time, more memory.
Next: shrink the patch storage.

## PML (2026-10-23): diagnosis and fix

EMerge's PML: s = 1 - i*deltamax*(depth/thickness)^1.5, deltamax = 8 (frequency
independent), tensors Lambda = diag(sy sz/sx, sx sz/sy, sx sy/sz). Horn (model4 geometry),
regenerated headless by intercepting SolveRoutine.solve (scratchpad gen_model4_pml.py).

Same horn, res 0.15 (~5.5 edges/lambda), GMRES its, CORVID order 1:
    ABC instead of PML 13 | PML deltamax 1: 16 | 2: 29 | 4: 129 | 8: 300+
    phase-only PML (|s| = 1, arg up to 83 deg): 24   -> the phase is NOT the problem
    magnitude-only PML (real s up to 8.1):      fails -> |s| is

Mechanism: in the stretched direction the field goes as exp(-i k s x): the local
length scale is 1/(k|s|). With |k s h| up to 7.6 the PML's near-kernel (evanescent,
small-residual modes) is neither local (vertex blocks do not remove small-residual
error) nor representable in the Whitney coarse space. Two-level convergence needs the
coarse space to hold the near-kernel. Preconditioned spectrum: clustered at 1 with an
outlier or two on model2/3; a cloud straddling the origin on the PML horn; the
stationary iteration diverges (x1.7-2.5 per step).
The same mechanism with plain kh: the ABC horn at res 0.3 (3.1 edges/lambda) needs
208 its. The Whitney coarse level wants >= ~5 edges per wavelength.

Rejected: partition-of-unity / restricted Schwarz, per-region damping (3x better, not a
fix), preconditioning with a soft PML (delta=1) on the same mesh (3.7e-3 at 300),
putting only the deep PML (|Im K_ii|/|K_ii| >= tau) in the coarse level (121-200 its:
cutting the PML anywhere loses the modes at the cut).

Fix (corvid 0.2.0, Solver(exact="auto")): every PML row (most of its entries complex)
joins the exactly solved coarse level. Horn res 0.15 delta 8: 300+ -> 26 its; delta 4:
129 -> 24; the user's model4 p1/p4/p7: stalled -> 57/77/134 (coarser mesh, ~3.6
edges/lambda, so the kh limit adds on top). No PML: nothing selected, unchanged.
Cost on the horn (PML = 52 % of the DOFs, tiny air box): coarse nnz(L) 20.1M vs 40.0M
for a full direct factorization. A PML is a layer, so its share shrinks with size.

## PML depth threshold (corvid 0.2.1, 2026-10-25)

Research log: research/ (experiments) and docs/notebook/ (notebook, built by render.sh). The deep PML
needs the exact solve, the first two vertex layers do not; including them even costs iterations
(an incomplete piece of the complement in the exact level adds spurious coarse modes). Solver(pml_depth=2):

    problem          0.2.0 auto              pml_depth=2
    model4 p1        57 its 4.8 s 17.9M      41 its 3.7 s 17.0M
    model4 p4        77 its 5.7 s            53 its 4.1 s
    model4 p7       134 its 9.3 s            93 its 6.3 s
    horn res 0.15    26 its 20.1M            21 its 18.6M
    PML boxes       11-13 its                11-12 its, up to -10% factor
    non-PML models   unchanged               unchanged
pml_depth=3 cuts the factor by a further ~30% for a few more iterations.

## Metal-wall exact rows (corvid 0.3.0, 2026-10-26)

Idea from the user (antenna designer): "put the absorber where the trapped modes live". On the model4 horn
the slow modes of 0.2.1 sit next to the PEC walls of the flare (60-96 % of |v|^2 on DOFs touching a PEC
vertex, 98 % on higher-order DOFs). Adding the higher-order DOFs around PEC vertices with k0 h > 1 to the
exact level (vertex granularity; a per-DOF threshold creates ragged sets and broke the finer horn, 21 -> 66):

    model4 p1/p4/p7   41/53/93 its 3.6/4.0/6.2 s  ->  24/29/32 its 3.0/2.9/2.9 s
    horn res 0.15     21 -> 18      box 3.6 epl   78 -> 60     model1  12 -> 6
    model2, model3    unchanged (no rows selected)
API: Topology.nedelec2(..., nodes=mesh.nodes), Solver(...).factorize(K, k0=...). Optional.

## Dropping redundant vertex blocks (corvid 0.3.1, 2026-10-26)

User report: on a bigger horn 0.3.0 used more RAM than a direct solve. Breakdown on model4 p1:
exact level 306 MB + vertex blocks 203 MB = 508 MB vs direct 446 MB (114 %). Blocks whose DOFs all lie in
the exact level are redundant; dropping them:

    model4 p1/p4/p7   508 MB 24/29/32 its  ->  378 MB (85 % of direct) 18/24/29 its, 2.1-2.5 s
    horn res 0.15     643 MB (101 %) 18 its ->  507 MB (79 %) 13 its
    PML boxes         one to three fewer iterations; non-PML models unchanged
pml_depth=3 brings the horn to ~73 % of direct (+few its); sparse block factors would give ~60 %.

## Soft PML switch and sparse vertex blocks (corvid 0.4.0, 2026-10-26)

- exact="auto" estimates the PML stretch |s| per row from the matrix (rho = sum|Im|/sum|K| ~ sigma/|s|) and skips
  the exact PML solve when the 99th percentile is below pml_strong=2.5 (EMerge deltamax <= ~2.5). The estimate
  tracks the true peak: 1.8 / 3.4 / 7.0 for deltamax 2 / 4 / 8 (true 2.2 / 4.1 / 8.1).
- Vertex blocks as sparse LDL^T factors (patch.rs: bitset minimum degree, unpivoted LDL^T, dense-inverse fallback
  on a tiny pivot; none needed on any dataset). Same preconditioner to 1e-13, block memory -55..60 %, faster setup.

    total memory / direct LDL^T     0.3.1 (dense blocks)   0.4.0
    model4 horn deltamax 2          67 %                   42 %   (29 its)
    model4 horn deltamax 2, res .15 66 %                   42 %   (15 its)
    model4 horn deltamax 8          85 %                   76 %   (18 its)
    model2 / model3                 65 % / 86 %            31 % / 44 %

## Bug report: exact-row count climbing along a sweep (corvid 0.4.1, 2026-10-26)

User log (soft PML): "extra exact rows" 0, 952, 2179, 3386, 5386 over j001-j005, iterations 23 -> 29.
Diagnosis (sweep datasets, research/exp/check_sweep_rows.py):
- The count is the metal-wall rule (0.3.0), k0-dependent by design: more wall vertices have k0 h > 1 at higher
  frequency (model4 sweep 27834 -> 28595 wall rows over 90-110 GHz). With deltamax 2 the PML part is skipped,
  so j001 (lowest f) can have 0 rows. Not the PML detection, and the mesh does not change (n fixed).
- Iterations rise with frequency with or without walls; walls help at every frequency (model4: 20 vs 33 at
  90 GHz, 75 vs 300 at 110 GHz).
- Hardening anyway: EMerge matrices carry imaginary round-off down to 1e-27 relative (now ignored below 1e-8);
  ABC-only models had 30-36 boundary rows with an estimated |s| of 2.3-2.5, right at pml_strong (now: fewer
  than 2 % complex-dominated rows = no PML). Separate counters n_pml_exact / n_wall_exact.
