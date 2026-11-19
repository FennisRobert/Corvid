"""CORVID's solver for the time-harmonic Maxwell system K x = b.

Two pieces, built at different times:

  Topology  once per mesh: which unknowns belong to which vertex block, and
            which unknowns are the lowest-order (Whitney) edge DOFs
  Solver    once per frequency: factorizes K's vertex blocks and its Whitney
            block, then solves any number of right-hand sides

Preconditioner (all Rust, corvid._kernels.TwoLevelSolver): Chebyshev smoothing
scaled by vertex-block Schwarz -- for every mesh vertex, all unknowns on its
incident edges and faces, solved exactly -- around an exact solve on the
Whitney rows. It works on K itself, no shifted operator.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from . import _kernels

EDGE_WHITNEY, EDGE_OTHER, FACE = 0, 1, 2
WALL_LOOKAHEAD = 1.2      # wall rows are selected for this multiple of k0 (see Solver._wall_rows)
# coarsen_kh: modes of the V-cycle's error propagation I - V A with |mu| above DEFLATE_MU are deflated;
# the eigensolve looks for DEFLATE_K of them, doubling up to DEFLATE_KMAX while all it finds are bad.
# The deflation only needs those modes roughly: tolerance 1e-2 finds the same ones in a fifth of the
# V-cycles of 1e-4 (model5: 21 against 96, same outer iterations)
DEFLATE_MU = 0.8
DEFLATE_K = 8
DEFLATE_KMAX = 48
DEFLATE_TOL = 1e-2


class Topology:
    """Vertex blocks and Whitney rows, in the numbering of the solved system.

    General form: per solved unknown, the mesh entity it lives on (`dof_entity`:
    an edge index for kinds 0 and 1, a triangle index for kind 2) and its kind
    (`dof_kind`: 0 = lowest-order edge DOF, 1 = other edge DOF, 2 = face DOF),
    plus the node lists of the edges (E, 2) and triangles (F, 3).

    Optional, for Solver(coarsen_kh > 0) only: `tets` (T, 4) on the same vertex numbering as the
    edges, `coords` (the node coordinates) and `tet_index` (T,), the refractive index sqrt(er ur)
    of every tet (1 when not given: too low inside dielectrics, so the coarsening stops later than
    it should there).
    """

    def __init__(self, dof_entity, dof_kind, edges, tris, wall_vertices=None, coords=None, tets=None, tet_index=None):
        dof_entity = np.asarray(dof_entity, dtype=np.int64).ravel()
        dof_kind = np.asarray(dof_kind, dtype=np.int8).ravel()
        edges = _as_rows(edges, 2)
        tris = _as_rows(tris, 3)
        n = len(dof_entity)
        if dof_kind.shape != (n,):
            raise ValueError("dof_entity and dof_kind must have the same length")
        on_edge = dof_kind != FACE
        nodes = [edges[dof_entity[on_edge]], tris[dof_entity[~on_edge]]]
        rows = [np.repeat(np.flatnonzero(on_edge), 2), np.repeat(np.flatnonzero(~on_edge), 3)]
        nv = int(max(edges.max(initial=-1), tris.max(initial=-1))) + 1
        inc = sp.csc_matrix((np.ones(sum(len(r) for r in rows), np.int8),
                             (np.concatenate(rows), np.concatenate([x.ravel() for x in nodes]))), shape=(n, nv))
        inc.sum_duplicates()
        keep = np.flatnonzero(np.diff(inc.indptr))              # vertices that carry an unknown
        inc = inc[:, keep]
        inc.sort_indices()
        self.n = n
        self.dof_vertices = inc.tocsr()                 # solved DOF x kept vertex incidence
        g = sp.csr_matrix((np.ones(2 * len(edges), np.int8), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                          shape=(nv, nv))[keep][:, keep]
        self.vertex_graph = g.tocsr()                    # mesh edges between kept vertices
        self.block_ptr = inc.indptr.astype(np.int64)
        self.block_idx = inc.indices.astype(np.int32)
        self.whitney = np.flatnonzero(dof_kind == EDGE_WHITNEY).astype(np.int64)
        self.non_whitney = dof_kind != EDGE_WHITNEY
        # optional geometry for the coarsened Whitney level (Solver(coarsen_kh > 0))
        X = None
        if coords is not None:
            X = np.asarray(coords, dtype=np.float64)
            X = X.T if X.shape[0] == 3 and X.shape[1] != 3 else X
        self.coords = X
        self.edges = edges
        self.whitney_edges = dof_entity[self.whitney]
        self.tets = None if tets is None else _as_rows(tets, 4)
        self.tet_index = None
        if tet_index is not None:
            self.tet_index = np.asarray(tet_index, dtype=np.float64).ravel()
            if self.tets is None or self.tet_index.shape != (len(self.tets),):
                raise ValueError("tet_index needs tets and one value per tet")
        # optional geometry: metal-touching vertices and their longest incident edge (for the wall rule)
        self.wall_h = None
        if wall_vertices is not None and X is not None:
            h = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
            vh = np.zeros(nv)
            np.maximum.at(vh, edges[:, 0], h)
            np.maximum.at(vh, edges[:, 1], h)
            self.wall_h = np.where(np.asarray(wall_vertices, bool), vh, 0.0)[keep]

    @classmethod
    def nedelec2(cls, edges, tris, solve_ids, nodes=None, edge_dofs=2, face_dofs=2, tets=None, tet_index=None):
        """EMerge's Nedelec layout (its `Nedelec2` field, for every ElementSpace): global DOFs
        are `edge_dofs` blocks of edge DOFs followed by `face_dofs` blocks of face DOFs,
        [edge DOF 1 (Whitney) | edge DOF 2 | ... | face DOF 1 | ...], each block in mesh
        edge / triangle order. The counts are EMerge's `dofset.set3d.n_edge_dofs` and
        `n_face_dofs`; the defaults (2, 2) are the second-order mixed spaces. The first edge
        block must be the Whitney function Li grad Lj - Lj grad Li (edge DoF 0, the first
        entry of every EMerge ElementSpace).

        `solve_ids` are the global DOFs that are solved, in the order of the solved system's
        rows. With the node coordinates `nodes` ((V, 3) or (3, V)) the metal walls are located
        (vertices of eliminated DOFs) for the wall rule of Solver.factorize(K, k0). `tets` and
        `tet_index` (with `nodes`) enable Solver(coarsen_kh > 0); see the class docstring."""
        edge_dofs, face_dofs = int(edge_dofs), int(face_dofs)
        if edge_dofs < 1 or face_dofs < 0:
            raise ValueError("need edge_dofs >= 1 (the first is Whitney) and face_dofs >= 0")
        edges = _as_rows(edges, 2)
        tris = _as_rows(tris, 3)
        ne, nf = len(edges), len(tris)
        n_edge = edge_dofs * ne
        n_total = n_edge + face_dofs * nf
        g = np.asarray(solve_ids, dtype=np.int64).ravel()
        if g.size and (g.min() < 0 or g.max() >= n_total):
            raise ValueError(f"solve_ids outside the {edge_dofs}*E + {face_dofs}*F layout "
                             f"(max {g.max()}, layout size {n_total})")
        on_edge = g < n_edge
        kind = np.where(g < ne, EDGE_WHITNEY, np.where(on_edge, EDGE_OTHER, FACE)).astype(np.int8)
        ent = np.where(on_edge, g % ne, (g - n_edge) % max(nf, 1))
        wall = None
        if nodes is not None:
            nv = int(max(edges.max(initial=-1), tris.max(initial=-1))) + 1
            solved = np.zeros(n_total, bool); solved[g] = True
            elim = np.flatnonzero(~solved)
            wall = np.zeros(nv, bool)
            on_e = elim < n_edge
            wall[edges[elim[on_e] % ne].ravel()] = True
            wall[tris[(elim[~on_e] - n_edge) % max(nf, 1)].ravel()] = True
        return cls(ent, kind, edges, tris, wall_vertices=wall, coords=nodes, tets=tets, tet_index=tet_index)

    @property
    def block_entries(self) -> int:
        """Entries of the dense vertex-block inverses (x16 bytes, halved by symmetry)."""
        m = np.diff(self.block_ptr).astype(np.int64)
        return int(np.sum(m * (m + 1) // 2))


class Solver:
    """Factorize once per matrix, then solve any number of right-hand sides.

    order:   Chebyshev order of the smoother (1: plain damped Schwarz, cheapest)
    method:  "gmres" (restarted, right preconditioned) or "cocg" (complex
             symmetric K: no basis to store, a little more iterations)
    exact:   unknowns solved exactly in the coarse level besides the Whitney rows.
             "auto" adds PML rows, detected as rows whose entries are mostly complex
             (a PML's coordinate stretching shrinks the local wavelength by |s|, so
             its evanescent modes are neither local nor representable by the Whitney
             space); a boolean mask over the solved unknowns sets them explicitly;
             None adds nothing.
    pml_depth: with exact="auto", only PML rows at least this many mesh edges deep (graph
             distance from the non-PML region) are solved exactly. The outer layers of a PML are
             barely stretched and the vertex blocks handle them; 2 cut the exact level's factor
             and the iterations on the test horns, 1 is the full PML (CORVID 0.2.0).
    wall_kh: with node coordinates in the Topology and k0 passed to factorize, every
             higher-order unknown touching a metal vertex whose longest incident edge has
             k0 h > wall_kh is also solved exactly. The slowest error modes on coarse meshes
             hug the metal walls; the model4 horn goes from 41/93 to 24/32 iterations. None
             disables it.
    pml_strong: with exact="auto", the PML is solved exactly only if it is strongly stretched: the 99th
             percentile of |s| estimated from its rows (rho = sum|Im K_ij| / sum|K_ij| ~ sigma / |s|,
             |s| = 1 / sqrt(1 - rho^2)) exceeds this. Soft PMLs (EMerge deltamax <= ~2.5) converge
             without it, and skipping it saves the whole PML factorization. `sol.pml_stretch` reports
             the estimate.
    sparse_blocks: store each vertex block as a sparse LDL^T factor (minimum-degree ordering) instead of
             its dense inverse: about half the block memory, same preconditioner up to rounding. Blocks
             with a tiny pivot fall back to the dense inverse.
    threads: CPU threads for the factorization (vertex blocks and the exact level's LDL^T) and the solves.
             None uses the module default (corvid.get_num_threads(): RAYON_NUM_THREADS, else up to 8,
             changed with corvid.set_num_threads). Read at factorize(): the factorization and every solve
             on it run on that count.
    warm_start: start each solve from the previous solutions on this Solver (keep one Solver across a
             frequency sweep and call factorize(K, k0) per frequency): the minimum-residual combination
             of the stored solutions, min ||b - K V g|| over g, per column. `history` caps the stored
             solution vectors (all columns of the most recent solves): 16 is 8 frequencies of a 2-port
             problem, at 16 vectors of the system size in memory.
    refactor_tol: in a sweep, factorize(K, k0) keeps the block and coarse factors of the last real
             factorization (at k0_f) while |k0^2 / k0_f^2 - 1| < refactor_tol, only switching the
             operator to the new K. A solve on such kept factors that needs clearly more iterations
             than the last fresh one is stopped, refactored and continued, and the Solver then keeps
             factors only up to half that distance. 0 refactors every time.
    coarsen_kh: experimental, opt-in (0 = off, the default). Instead of factorizing the Whitney rows
             exactly, coarsen them: an agglomerated multilevel hierarchy (exact de Rham coarse spaces)
             in which every aggregate keeps k_loc * H <= coarsen_kh (H its bounding-box diagonal,
             k_loc = k0 * the largest refractive index in it), so it only coarsens where the mesh is
             finer than the wave needs; one V-cycle (symmetric Gauss-Seidel Hiptmair, exact solve on
             the coarsest level) replaces the exact solve, and the few modes that V-cycle amplifies
             (near-resonant modes the coarse levels misplace) are found by an eigensolve and solved
             exactly around it. 0.6 is the tested value (model5, 1.75M unknowns: 15 iterations
             against 14 with the exact solve, coarsest factor 20 % of the exact one). Needs `tets`
             and node coordinates in the Topology (and `tet_index` in dielectrics). Falls back to the
             exact solve, with a warning, when PML or metal-wall rows are solved exactly (n_exact > 0)
             or nothing coarsens. The hierarchy is built in Rust (6 s for 400k tets, three levels) and
             kept across a sweep; per frequency, the Galerkin operators and the deflation eigensolve
             (Python / SciPy) take about as long again.
    coarsen_k0: the largest k0 the hierarchy must hold for (the top of the sweep). None: 1.2 x the
             first k0, rebuilt when a later k0 exceeds it.
    """

    def __init__(self, topology: Topology, order: int = 3, method: str = "gmres", rtol: float = 1e-6,
                 maxiter: int = 500, restart: int = 100, exact="auto", pml_depth: int = 2, wall_kh=1.0,
                 pml_strong: float = 2.5, sparse_blocks: bool = True, threads: int | None = None,
                 warm_start: bool = True, refactor_tol: float = 0.2, history: int = 16,
                 coarsen_kh: float = 0.0, coarsen_k0: float | None = None):
        if method not in ("gmres", "cocg"):
            raise ValueError("method must be 'gmres' or 'cocg'")
        if threads is not None and int(threads) < 1:
            raise ValueError("threads must be at least 1")
        self.topology = topology
        self.order, self.method, self.rtol, self.maxiter, self.restart = order, method, rtol, maxiter, restart
        self.exact = exact
        self.pml_depth = pml_depth
        self.wall_kh = wall_kh
        self.pml_strong = pml_strong
        self.sparse_blocks = sparse_blocks
        self.threads = threads
        self.pml_stretch = None
        self.k0 = None
        self.n_exact = 0
        self.n_pml_exact = 0
        self.n_wall_exact = 0
        self._pc = None
        self._key = None
        self.warm_start = warm_start
        self._history: list[tuple[float, np.ndarray]] = []     # (k0, solutions) of the last solves
        self.refactored = False
        self.refactor_tol = refactor_tol
        self.history = history
        self.reused = False              # the current factors come from an earlier k0
        self._k0_factor = None           # k0 of the last real factorization
        self._ref_its = None             # iterations of the last solve on fresh factors
        self._k = None                   # the current K (CSR arrays), kept until the next solve
        self._tol_learned = None         # refactor_tol lowered after kept factors fell short
        self._wall_mask = None           # wall rows of this sweep (only grows), chosen for k0 up to _wall_k0
        self._wall_k0 = None
        self.info: list[dict] = []
        if coarsen_kh is None or float(coarsen_kh) < 0:
            raise ValueError("coarsen_kh must be >= 0 (0 turns the coarsening off)")
        self.coarsen_kh = float(coarsen_kh)
        self.coarsen_k0 = None if coarsen_k0 is None else float(coarsen_k0)
        self.coarsen_active = False      # the current factorization uses the coarsened Whitney level
        self.coarsen_levels: list[int] = []   # free Whitney edges per level of the hierarchy
        self.coarsen_setup_s = 0.0       # seconds spent building the hierarchy (once per sweep)
        self.n_deflated = 0              # modes solved exactly around the V-cycle
        self._hier = None                # (coarsen_kh, k0 it holds for, Ps, Gs) in the solved rows' order
        self._coarsen_warned: set[str] = set()

    def factorize(self, K, k0=None) -> "Solver":
        """Build the preconditioner for the solved system matrix K (n x n). `k0` (free-space
        wavenumber) enables the wall rule when the Topology knows the node coordinates.

        Called again on the same Solver (the next frequency of a sweep) with the same sparsity
        pattern and the same exact rows, it refactors in place: the coarse level's SPARTA analysis,
        the vertex-block orderings and the spectral estimate are reused."""
        if k0 is not None:
            self.k0 = float(k0)
        top = self.topology
        if (sp.isspmatrix_csc(K) or isinstance(K, sp.csc_array)) and self.method == "cocg":
            K = K.T                     # COCG needs K = K^T, so the CSC arrays are already K's CSR arrays
        K = K if sp.isspmatrix_csr(K) or isinstance(K, sp.csr_array) else sp.csr_matrix(K)
        if K.shape != (top.n, top.n):
            raise ValueError(f"K is {K.shape}, the topology has {top.n} unknowns")
        if not K.has_sorted_indices:
            K = K.copy(); K.sort_indices()
        # views where possible: no copy for a sorted complex128 CSR with int32 indices
        k = (K.indptr.astype(np.int32, copy=False), K.indices.astype(np.int32, copy=False),
             np.ascontiguousarray(K.data, dtype=np.complex128), K.shape[0], K.shape[1])
        coarse = top.whitney
        extra = self._exact_rows(K)
        if extra is not None and extra.any():
            mask = extra.copy()
            mask[top.whitney] = True
            coarse = np.flatnonzero(mask).astype(np.int64)
        self.n_exact = len(coarse) - len(top.whitney)
        self._k_nnz = int(K.nnz)
        ptr, idx = top.block_ptr, top.block_idx
        if self.n_exact:
            # vertex blocks lying entirely in the exact level are redundant with it; dropping them saves
            # their memory and, by lowering the block overlap, allows a larger smoothing step
            exact = np.zeros(top.n, bool); exact[coarse] = True
            sizes = np.diff(ptr)
            inside = np.add.reduceat(exact[idx].astype(np.int64), ptr[:-1]) == sizes
            keep = ~inside
            idx = np.ascontiguousarray(idx[np.repeat(keep, sizes)], dtype=np.int32)
            ptr = np.r_[0, np.cumsum(sizes[keep])].astype(np.int64)
        self.n_blocks = len(ptr) - 1
        ml = self._use_multilevel()
        key = (K.shape, K.nnz, len(coarse), hash(coarse.tobytes()), self.order, self.sparse_blocks, self.threads, ml)
        same = self._pc is not None and self._key is not None and self._key[0] == key \
            and _same_pattern(self._key[1], self._key[2], k[0], k[1])
        self._k = k
        self.reused = same and self._close_to_factor()
        self.refactored = same and not self.reused
        if self.reused:
            self._pc.update_operator(k)
        elif self.refactored:
            self._pc.refactor(k)
            self._k0_factor, self._ref_its = self.k0, None
        else:
            self._pc = None                                    # release the previous factorization first
            self._pc = _kernels.TwoLevelSolver(k, ptr, idx, coarse, order=self.order, sparse_blocks=self.sparse_blocks,
                                               threads=None if self.threads is None else int(self.threads),
                                               exact_coarse=not ml)
            self._k0_factor, self._ref_its = self.k0, None
        self.coarsen_active = ml
        if ml and not self.reused:
            self._setup_multilevel(k)
        self._key = (key, k[0], k[1])
        return self

    # ---- opt-in coarsened Whitney level (coarsen_kh > 0)

    def _coarsen_off(self, why: str) -> bool:
        if why not in self._coarsen_warned:
            self._coarsen_warned.add(why)
            warnings.warn(f"CORVID coarsen_kh={self.coarsen_kh}: {why}; using the exact Whitney solve", stacklevel=3)
        return False

    def _use_multilevel(self) -> bool:
        """Whether this factorization coarsens the Whitney level; (re)builds the hierarchy if needed."""
        if not self.coarsen_kh:
            return False
        top = self.topology
        if top.tets is None or top.coords is None:
            return self._coarsen_off("the Topology has no tets or node coordinates")
        if self.n_exact:
            return self._coarsen_off("extra exact rows (PML or metal walls) are present")
        k_need = self.coarsen_k0 if self.coarsen_k0 is not None else self.k0
        if k_need is None:
            return self._coarsen_off("no k0 (pass it to factorize, or set coarsen_k0)")
        h = self._hier
        if h is None or h[0] != self.coarsen_kh or k_need > h[1] * (1 + 1e-9):
            self._build_hierarchy(k_need if self.coarsen_k0 is not None else k_need * WALL_LOOKAHEAD)
        if not self._hier[2]:
            return self._coarsen_off("nothing coarsens (the mesh is already at the wavelength limit)")
        return True

    def _build_hierarchy(self, k_build: float) -> None:
        from .amge.capped import build_hierarchy
        top = self.topology
        if top.tet_index is None:
            if "no tet_index" not in self._coarsen_warned:
                self._coarsen_warned.add("no tet_index")
                warnings.warn("CORVID coarsen_kh: no tet_index in the Topology, assuming vacuum (refractive "
                              "index 1); the coarsening is too aggressive inside dielectrics", stacklevel=4)
        index = np.ones(len(top.tets)) if top.tet_index is None else top.tet_index
        free = np.zeros(len(top.edges), bool)
        free[top.whitney_edges] = True
        Ps, Gs, sizes, secs = build_hierarchy(top.coords, top.tets, top.edges, free, k_build * index, self.coarsen_kh)
        if Ps:
            # level 0 of the hierarchy is in increasing edge order; the solver's Whitney rows are in
            # solved order (top.whitney): reorder the rows of P_0 and G_0 to match
            rank = np.searchsorted(np.flatnonzero(free), top.whitney_edges)
            Ps[0], Gs[0] = Ps[0][rank].tocsr(), Gs[0][rank].tocsr()
        # kept as the CSR tuples the Rust kernels take (converted once per hierarchy, not per frequency)
        self._hier = (self.coarsen_kh, k_build, [_csr_tuple(P, False) for P in Ps], [_csr_tuple(G, False) for G in Gs])
        self.coarsen_levels, self.coarsen_setup_s = sizes, self.coarsen_setup_s + secs

    def _setup_multilevel(self, k) -> None:
        """Galerkin operators of the current K and the V-cycle (both in Rust), then its deflation."""
        top = self.topology
        self._pc.set_multilevel_galerkin(k, top.whitney, self._hier[2], self._hier[3])
        K = sp.csr_matrix((k[2], k[1], k[0]), shape=(k[3], k[4]))
        self.n_deflated = self._deflate(K[top.whitney][:, top.whitney].tocsr())

    def _deflate(self, A) -> int:
        """Eigenvectors of I - V A (V the V-cycle) with |mu| > DEFLATE_MU: the modes the V-cycle
        amplifies or barely reduces. Solved exactly around it; returns how many."""
        import scipy.sparse.linalg as spla
        n = A.shape[0]
        op = spla.LinearOperator((n, n), dtype=complex,
                                 matvec=lambda x: x - self._pc.ml_vcycle(np.ascontiguousarray(A @ x, dtype=complex)))
        v0 = np.random.default_rng(0).standard_normal(n).astype(complex)
        k = DEFLATE_K
        while True:
            kk = min(k, n - 2)
            try:
                mu, Z = spla.eigs(op, k=kk, which="LM", tol=DEFLATE_TOL, maxiter=5000, v0=v0)
            except spla.ArpackNoConvergence as e:
                mu, Z = e.eigenvalues, e.eigenvectors
            sel = np.abs(mu) > DEFLATE_MU
            if sel.all() and kk < min(DEFLATE_KMAX, n - 2):
                k *= 2                      # all found modes are bad: there may be more
                continue
            break
        if sel.all() and len(mu):
            warnings.warn(f"CORVID coarsen_kh: {len(mu)} or more modes need deflation; convergence may suffer "
                          "(a smaller coarsen_kh coarsens less)", stacklevel=4)
        Z = Z[:, sel]
        m = Z.shape[1]
        if m:
            Si = np.linalg.inv(Z.T @ (A @ Z))           # plain transpose: keeps the operator complex symmetric
            self._pc.set_deflation(np.ascontiguousarray(Z.T).ravel(), np.ascontiguousarray(Si).ravel(), m)
        return m

    def _close_to_factor(self) -> bool:
        tol = self.refactor_tol if self._tol_learned is None else min(self.refactor_tol, self._tol_learned)
        if not tol or self.k0 is None or self._k0_factor is None or self._ref_its is None:
            return False
        return abs(self.k0 ** 2 / self._k0_factor ** 2 - 1) < tol

    def _exact_rows(self, K):
        if self.exact is None:
            return None
        if isinstance(self.exact, str):
            if self.exact != "auto":
                raise ValueError("exact must be 'auto', None or a boolean mask")
            stats = _row_stats(K)
            rows = pml_rows(K, stats=stats)
            self.pml_stretch = pml_stretch(K, rows, stats=stats)
            if self.pml_stretch < self.pml_strong:
                m = np.zeros(K.shape[0], bool)          # soft PML: the vertex blocks handle it
            else:
                m = rows if self.pml_depth <= 1 else rows & (pml_depth(self.topology, rows) >= self.pml_depth)
        else:
            m = np.asarray(self.exact, dtype=bool).ravel()
            if m.shape != (K.shape[0],):
                raise ValueError("the exact mask must have one entry per solved unknown")
        self.n_pml_exact = int(m.sum())
        w = self._wall_rows()
        self.n_wall_exact = 0 if w is None else int((w & ~m).sum())
        return m if w is None else m | w

    def _wall_rows(self):
        """Higher-order unknowns on metal vertices with k0 h > wall_kh. Across a sweep the set only
        grows and is chosen for WALL_LOOKAHEAD times the current k0, so the coarse pattern changes at
        most once per such step in k0 and refactors / kept factors work in between (extra exact rows
        never hurt convergence)."""
        top = self.topology
        if self.wall_kh is None or self.k0 is None or top.wall_h is None:
            return None
        if self._wall_mask is None or self.k0 > self._wall_k0:
            self._wall_k0 = self.k0 * WALL_LOOKAHEAD
            sel = (self._wall_k0 * top.wall_h > self.wall_kh).astype(np.float64)
            touch = (np.asarray(top.dof_vertices @ sel).ravel() > 0) & top.non_whitney
            self._wall_mask = touch if self._wall_mask is None else (self._wall_mask | touch)
        return self._wall_mask

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the vertex blocks (dense inverses or sparse factors)."""
        return 0 if self._pc is None else self._pc.schwarz_bytes

    @property
    def memory_report(self) -> dict:
        """Bytes held by CORVID after factorize: the vertex blocks, the exact level's LDL^T factor
        (16 B per nonzero) and its copy of K (real/imaginary split, ~20 B per nonzero). Excludes the
        caller's own K and the transient peak of the SPARTA factorization."""
        if self._pc is None:
            return {}
        blocks = self._pc.schwarz_bytes
        coarse = self._pc.coarse_factor_nnz * 16
        kcopy = self._k_nnz * 20
        ml = self._pc.ml_bytes
        return dict(blocks=blocks, exact_factor=coarse, multilevel=ml, k_copy=kcopy, total=blocks + coarse + ml + kcopy,
                    exact_rows=self.n_exact + len(self.topology.whitney), vertex_blocks=self.n_blocks)

    def solve(self, b, x0=None) -> np.ndarray:
        """x with K x = b; b may be (n,) or (n, nrhs). Per column, `self.info`
        records iterations and the true relative residual.

        All columns are solved together (COCG: in lockstep, one preconditioner application per
        iteration for all of them). `x0` is a starting guess of b's shape; without one and with
        `warm_start`, the solutions of the previous solves on this Solver (earlier frequencies of a
        sweep) are extrapolated in k0. A guess is only used where it beats zero."""
        if self._pc is None:
            raise RuntimeError("call factorize(K) first")
        b = np.asarray(b)
        B = b.reshape(b.shape[0], -1)
        cols = [np.ascontiguousarray(B[:, j], dtype=np.complex128) for j in range(B.shape[1])]
        X0 = None
        if x0 is not None:
            X0 = np.asarray(x0).reshape(B.shape)
        elif self.warm_start:
            X0 = self._project(B)
        x0c = None if X0 is None else [np.ascontiguousarray(X0[:, j], dtype=np.complex128) for j in range(B.shape[1])]
        failed = False
        if self.reused:
            # kept factors: a capped attempt; if it falls short, refactor and continue from where it got
            cap = int(1.5 * self._ref_its) + 3
            xs, its, res = self._pc.solve_columns(cols, x0c, self.method, self.rtol, self.restart, min(cap, self.maxiter))
            if max(res) > self.rtol:
                # this problem tolerates less: keep factors only up to half this distance from now on
                self._tol_learned = 0.5 * abs(self.k0 ** 2 / self._k0_factor ** 2 - 1)
                self._pc.refactor(self._k)
                if self.coarsen_active:
                    self._setup_multilevel(self._k)
                self._k0_factor, self.reused, self.refactored = self.k0, False, True
                xs, its2, res = self._pc.solve_columns(cols, list(xs), self.method, self.rtol, self.restart, self.maxiter)
                its = [a + b for a, b in zip(its, its2)]
                failed = True        # no clean reference count: the next frequency refactors
        else:
            xs, its, res = self._pc.solve_columns(cols, x0c, self.method, self.rtol, self.restart, self.maxiter)
        if failed:
            self._ref_its = None
        elif not self.reused and self._ref_its is None:
            self._ref_its = max(its)
        self._k = None
        x = np.column_stack(xs) if len(xs) > 1 else xs[0].reshape(-1, 1)
        self.info = [dict(iterations=i, residual=r, converged=r <= self.rtol) for i, r in zip(its, res)]
        if self.warm_start and self.history > 0:
            # (k0, contiguous solution columns) per solve, most recent last
            hist = [h for h in self._history if h[1][0].shape[0] == x.shape[0] and h[0] != self.k0] + [(self.k0, xs)]
            while len(hist) > 1 and sum(len(h[1]) for h in hist) > self.history:
                hist.pop(0)                                     # oldest frequency first
            self._history = hist
        return x.reshape(b.shape)

    def _project(self, B):
        """Starting guess: X0 = V G with V the stored solutions (all columns of the most recent
        solves) and G minimizing ||B - K V G|| column by column."""
        V = [c for _, cols in self._history if cols[0].shape[0] == B.shape[0] for c in cols]
        if not V:
            return None
        KV = self._pc.mul_columns(V)
        m = len(V)
        # normal equations on the small Gram matrix, with numerically dependent directions cut off
        # (only a starting guess is needed; the solver checks it against zero)
        A = np.empty((m, m), complex)
        for i in range(m):
            for j in range(i, m):
                A[i, j] = np.vdot(KV[i], KV[j])
                A[j, i] = np.conj(A[i, j])
        rhs = np.array([[np.vdot(KV[i], B[:, c]) for c in range(B.shape[1])] for i in range(m)])
        del KV
        w, U = np.linalg.eigh(A)
        keep = w > 1e-13 * w.max() if w.size and w.max() > 0 else np.zeros(0, bool)
        if not keep.any():
            return None
        U, w = U[:, keep], w[keep]
        G = U @ ((U.conj().T @ rhs) / w[:, None])
        X0 = np.zeros(B.shape, complex)
        for i in range(m):
            X0 += V[i][:, None] * G[i][None, :]
        return X0


def _csr_tuple(M, complex_values: bool):
    """(indptr, indices, data, nrows, ncols) as the Rust kernels take a CSR matrix."""
    M = sp.csr_matrix(M)
    M.sort_indices()
    return (M.indptr.astype(np.int32), M.indices.astype(np.int32),
            np.ascontiguousarray(M.data, dtype=np.complex128 if complex_values else np.float64), M.shape[0], M.shape[1])


def _same_pattern(ip1, ix1, ip2, ix2) -> bool:
    if ip1 is ip2 and ix1 is ix2:
        return True
    return ip1.shape == ip2.shape and ix1.shape == ix2.shape and np.array_equal(ip1, ip2) and np.array_equal(ix1, ix2)


def _row_stats(K, rel_tol=1e-8):
    """(complex entries, sum |Im|, sum |a|, entries) per row of K, from one threaded pass."""
    K = K if sp.isspmatrix_csr(K) or isinstance(K, sp.csr_array) else sp.csr_matrix(K)
    ip = K.indptr.astype(np.int32, copy=False)
    cnt, sim, sab = _kernels.row_stats(ip, np.ascontiguousarray(K.data, dtype=np.complex128), rel_tol)
    return cnt, sim, sab, np.diff(ip)


def pml_rows(K, fraction=0.5, rel_tol=1e-8, min_share=0.02, stats=None):
    """Rows of K whose entries are mostly complex: the volume of a PML (or of a
    strongly lossy material). Absorbing-boundary and port terms make only part
    of a row complex, so they are not selected. Imaginary parts below `rel_tol`
    relative to the entry are round-off (EMerge's matrices carry some at 1e-27) and
    ignored; fewer than `min_share` of the rows is not a PML volume (boundary rows)."""
    cnt, _, _, nnz = stats if stats is not None else _row_stats(K, rel_tol)
    sel = cnt > fraction * np.maximum(nnz, 1)
    return sel if sel.mean() >= min_share else np.zeros(len(nnz), bool)


def pml_stretch(K, rows, q=0.99, stats=None) -> float:
    """Estimated |s| of the PML rows (quantile q): rho = sum|Im K_ij| / sum|K_ij| per row, |s| = 1/sqrt(1-rho^2)."""
    if not rows.any():
        return 1.0
    _, sim, sab, _ = stats if stats is not None else _row_stats(K)
    rho = sim[rows] / np.maximum(sab[rows], 1e-300)
    s = 1 / np.sqrt(np.clip(1 - rho ** 2, 1e-12, 1))
    return float(np.quantile(s, q))


def pml_depth(top: Topology, rows: np.ndarray) -> np.ndarray:
    """Per solved DOF: the largest graph depth of its vertices inside the PML, where a vertex is in
    the PML when most of its block's rows are `rows`, and depth is the number of mesh edges to the
    nearest vertex outside the PML (0 outside)."""
    inc = top.dof_vertices
    share = np.asarray(inc.T @ rows.astype(float)).ravel() / np.maximum(np.diff(inc.tocsc().indptr), 1)
    inside = share > 0.5
    g = top.vertex_graph
    depth = np.where(inside, -1, 0)
    frontier = np.flatnonzero(~inside)
    d = 0
    while frontier.size and (depth < 0).any():
        d += 1
        nb = np.unique(g[frontier].indices)
        nb = nb[depth[nb] < 0]
        depth[nb] = d
        frontier = nb
    depth[depth < 0] = d + 1                             # PML pockets not reachable from outside
    vd = inc.multiply(depth[None, :]).tocsr()
    return np.asarray(vd.max(axis=1).todense()).ravel()


def _as_rows(a, k):
    a = np.asarray(a, dtype=np.int64)
    if a.ndim == 2 and a.shape[1] != k and a.shape[0] == k:
        a = a.T                                  # EMerge stores (k, N)
    if a.ndim != 2 or a.shape[1] != k:
        raise ValueError(f"expected an (N, {k}) or ({k}, N) index array")
    return np.ascontiguousarray(a)
