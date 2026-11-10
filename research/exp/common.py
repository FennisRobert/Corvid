"""Shared loaders for the research experiments (not part of the package)."""
import os, time
import numpy as np
import scipy.sparse as sp
import corvid
from corvid import _kernels

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SCRATCH = "research/data"


class Problem:
    """K x = b on the solved DOFs plus the geometry of every solved DOF."""

    def __init__(self, name, pid=1):
        path = name if os.path.isabs(name) else os.path.join(DATA, name)
        if not os.path.exists(path):
            path = os.path.join(SCRATCH, name)
        if not os.path.exists(path):
            path = os.path.join("data/aegis", name)
        f = np.load(os.path.join(path, f"problem_{pid}.npz"))
        if "A_data" in f:
            A = sp.csr_matrix((f["A_data"], f["A_indices"], f["A_indptr"]), shape=tuple(f["A_shape"]))
        else:                                    # aegis datasets: A = E - k0^2 B
            csr = lambda k: sp.csr_matrix((f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]), shape=tuple(f[k + "_shape"]))
            A = (csr("E") - f["k0"] ** 2 * csr("B")).tocsr()
        self.name = name
        self.sid = sid = f["solve_ids"]
        self.K = A[sid][:, sid].tocsr()
        self.K.sort_indices()
        self.b = f["b"][sid]
        self.b = self.b.reshape(self.n if False else len(sid), -1)[:, 0].astype(complex)
        self.x_ref = f["solution"][sid] if "solution" in f else None
        self.edges, self.tris, self.nodes = f["mesh_edges"], f["mesh_tris"], f["mesh_nodes"]
        self.tets = f["mesh_tets"] if "mesh_tets" in f else None
        self.k0 = float(f["k0"]) if "k0" in f else 2 * np.pi * 90e9 / 299792458.0
        self.n = self.K.shape[0]
        ne, nf = self.edges.shape[1], self.tris.shape[1]
        g = sid
        self.kind = np.where(g < ne, 0, np.where(g < 2 * ne, 1, 2))
        self.ent = np.where(g < ne, g, np.where(g < 2 * ne, g - ne, np.where(g < 2 * ne + nf, g - 2 * ne, g - 2 * ne - nf)))
        X = self.nodes
        mid_e = X[:, self.edges].mean(1)
        mid_f = X[:, self.tris].mean(1)
        self.loc = np.where(self.kind < 2, mid_e[:, np.minimum(self.ent, ne - 1)], mid_f[:, np.minimum(self.ent, nf - 1)])
        self.pml = corvid.solver.pml_rows(self.K)
        self.top = corvid.Topology.nedelec2(self.edges, self.tris, sid)

    def kernels_csr(self, K=None):
        K = self.K if K is None else K
        return (K.indptr.astype(np.int32), K.indices.astype(np.int32), np.ascontiguousarray(K.data, dtype=np.complex128),
                K.shape[0], K.shape[1])

    def edges_per_wavelength(self):
        X = self.nodes
        h = np.linalg.norm(X[:, self.edges[0]] - X[:, self.edges[1]], axis=0)
        return 2 * np.pi / self.k0 / np.median(h), 2 * np.pi / self.k0 / np.max(h)


def baseline(p, exact="auto", method="gmres", maxiter=500, restart=500):
    """CORVID 0.2.0 as shipped: (iterations, residual, factorize s, solve s, extra exact rows, block MB)."""
    t = time.time()
    s = corvid.Solver(p.top, exact=exact, method=method, maxiter=maxiter, restart=restart).factorize(p.K)
    tf = time.time() - t
    t = time.time()
    s.solve(p.b)
    ts = time.time() - t
    return dict(its=s.info[0]["iterations"], res=s.info[0]["residual"], tf=tf, ts=ts, n_exact=s.n_exact,
                mb=s.memory_bytes / 1e6)


def gmres(matvec, prec, b, rtol=1e-6, maxiter=150, restart=None, stall_window=30, stall_factor=0.5):
    """Right-preconditioned full GMRES (MGS, one reorthogonalization). Stops early when the residual
    estimate fell by less than `stall_factor` over the last `stall_window` iterations (stagnation).
    Returns (x, its, history); history[-1] > rtol means not converged."""
    n = len(b)
    m = maxiter
    bn = np.linalg.norm(b)
    V = np.zeros((m + 1, n), complex)
    Z = np.zeros((m, n), complex)
    H = np.zeros((m + 1, m), complex)
    V[0] = b / bn
    e1 = np.zeros(m + 1, complex); e1[0] = bn
    hist = []
    for j in range(m):
        Z[j] = prec(V[j])
        w = matvec(Z[j])
        for _ in range(2):
            c = V[:j + 1].conj() @ w
            H[:j + 1, j] += c
            w = w - V[:j + 1].T @ c
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] > 1e-300:
            V[j + 1] = w / H[j + 1, j]
        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1[:j + 2], rcond=None)
        hist.append(np.linalg.norm(e1[:j + 2] - H[:j + 2, :j + 1] @ y) / bn)
        if hist[-1] < rtol:
            break
        if j >= stall_window and hist[-1] > stall_factor * hist[-1 - stall_window]:
            break
    x = Z[:j + 1].T @ y
    return x, j + 1, hist


def fmt_hist(hist, every=25):
    return " ".join(f"{hist[k]:.0e}" for k in range(every - 1, len(hist), every))

def sparta_nnz(M):
    """(factor nonzeros, factor seconds) of SPARTA's CPU LDL^T on M (lower triangle used)."""
    M = sp.csr_matrix(M); M.sort_indices()
    nnz, tf, _ = _kernels.bench_direct((M.indptr.astype(np.int32), M.indices.astype(np.int32),
                                        np.ascontiguousarray(M.data, dtype=np.complex128), M.shape[0], M.shape[1]), 1)
    return nnz, tf
