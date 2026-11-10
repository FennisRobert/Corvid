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

import numpy as np
import scipy.sparse as sp

from . import _kernels

EDGE_WHITNEY, EDGE_OTHER, FACE = 0, 1, 2


class Topology:
    """Vertex blocks and Whitney rows, in the numbering of the solved system.

    General form: per solved unknown, the mesh entity it lives on (`dof_entity`:
    an edge index for kinds 0 and 1, a triangle index for kind 2) and its kind
    (`dof_kind`: 0 = lowest-order edge DOF, 1 = other edge DOF, 2 = face DOF),
    plus the node lists of the edges (E, 2) and triangles (F, 3).
    """

    def __init__(self, dof_entity, dof_kind, edges, tris, wall_vertices=None, coords=None):
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
        # optional geometry: metal-touching vertices and their longest incident edge (for the wall rule)
        self.wall_h = None
        if wall_vertices is not None and coords is not None:
            X = np.asarray(coords, dtype=np.float64)
            X = X.T if X.shape[0] == 3 and X.shape[1] != 3 else X
            h = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
            vh = np.zeros(nv)
            np.maximum.at(vh, edges[:, 0], h)
            np.maximum.at(vh, edges[:, 1], h)
            self.wall_h = np.where(np.asarray(wall_vertices, bool), vh, 0.0)[keep]

    @classmethod
    def nedelec2(cls, edges, tris, solve_ids, nodes=None):
        """EMerge's second-order Nedelec layout: global DOFs are four blocks,
        [edge DOF 1 (Whitney) | edge DOF 2 | face DOF 1 | face DOF 2], each in mesh
        edge / triangle order. `solve_ids` are the global DOFs that are solved,
        in the order of the solved system's rows. With the node coordinates `nodes`
        ((V, 3) or (3, V)) the metal walls are located (vertices of eliminated DOFs) for
        the wall rule of Solver.factorize(K, k0)."""
        edges = _as_rows(edges, 2)
        tris = _as_rows(tris, 3)
        ne, nf = len(edges), len(tris)
        g = np.asarray(solve_ids, dtype=np.int64).ravel()
        if g.size and (g.min() < 0 or g.max() >= 2 * ne + 2 * nf):
            raise ValueError("solve_ids outside the 2*E + 2*F second-order layout")
        kind = np.where(g < ne, EDGE_WHITNEY, np.where(g < 2 * ne, EDGE_OTHER, FACE)).astype(np.int8)
        ent = np.where(g < ne, g, np.where(g < 2 * ne, g - ne, np.where(g < 2 * ne + nf, g - 2 * ne, g - 2 * ne - nf)))
        wall = None
        if nodes is not None:
            nv = int(max(edges.max(initial=-1), tris.max(initial=-1))) + 1
            solved = np.zeros(2 * ne + 2 * nf, bool); solved[g] = True
            elim = np.flatnonzero(~solved)
            wall = np.zeros(nv, bool)
            on_e = elim < 2 * ne
            wall[edges[elim[on_e] % ne].ravel()] = True
            wall[tris[(elim[~on_e] - 2 * ne) % nf].ravel()] = True
        return cls(ent, kind, edges, tris, wall_vertices=wall, coords=nodes)

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
             than the last fresh one is stopped, refactored and continued. 0 refactors every time.
    """

    def __init__(self, topology: Topology, order: int = 3, method: str = "gmres", rtol: float = 1e-6,
                 maxiter: int = 500, restart: int = 100, exact="auto", pml_depth: int = 2, wall_kh=1.0,
                 pml_strong: float = 2.5, sparse_blocks: bool = True, threads: int | None = None,
                 warm_start: bool = True, refactor_tol: float = 0.2, history: int = 16):
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
        self.info: list[dict] = []

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
        key = (K.shape, K.nnz, len(coarse), hash(coarse.tobytes()), self.order, self.sparse_blocks, self.threads)
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
                                               threads=None if self.threads is None else int(self.threads))
            self._k0_factor, self._ref_its = self.k0, None
        self._key = (key, k[0], k[1])
        return self

    def _close_to_factor(self) -> bool:
        if not self.refactor_tol or self.k0 is None or self._k0_factor is None or self._ref_its is None:
            return False
        return abs(self.k0 ** 2 / self._k0_factor ** 2 - 1) < self.refactor_tol

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
        top = self.topology
        if self.wall_kh is None or self.k0 is None or top.wall_h is None:
            return None
        sel = (self.k0 * top.wall_h > self.wall_kh).astype(np.float64)
        touch = np.asarray(top.dof_vertices @ sel).ravel() > 0
        return touch & top.non_whitney

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
        return dict(blocks=blocks, exact_factor=coarse, k_copy=kcopy, total=blocks + coarse + kcopy,
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
        if self.reused:
            # kept factors: a capped attempt; if it falls short, refactor and continue from where it got
            cap = int(1.5 * self._ref_its) + 3
            xs, its, res = self._pc.solve_columns(cols, x0c, self.method, self.rtol, self.restart, min(cap, self.maxiter))
            if max(res) > self.rtol:
                self._pc.refactor(self._k)
                self._k0_factor, self.reused, self.refactored = self.k0, False, True
                xs, its2, res = self._pc.solve_columns(cols, list(xs), self.method, self.rtol, self.restart, self.maxiter)
                its = [a + b for a, b in zip(its, its2)]
                self._ref_its = None
        else:
            xs, its, res = self._pc.solve_columns(cols, x0c, self.method, self.rtol, self.restart, self.maxiter)
        if not self.reused and self._ref_its is None:
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
