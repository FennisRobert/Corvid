"""Test-side loader for EMerge datasets (data/aegis). Not part of the package.

Rebuilds tets from the triangle list (the datasets predate a tet export), and
returns exactly what the package will ask EMerge for: coords, tets, edges,
the free-edge mask, plus the matrices the tests compare against.
"""
import numpy as np
import scipy.sparse as sp

DATA = "data/aegis"


def _csr(f, k):
    return sp.csr_matrix((f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]), shape=tuple(f[k + "_shape"]))


def tets_from_tris(tris, nV):
    tris = set(map(tuple, np.sort(np.asarray(tris).T, axis=1)))
    adj = [set() for _ in range(nV)]
    for a, b, c in tris:
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    out = set()
    for a, b, c in tris:
        for d in adj[a] & adj[b] & adj[c]:
            if all(tuple(sorted(x)) in tris for x in ((a, b, d), (a, c, d), (b, c, d))):
                out.add(tuple(sorted((a, b, c, d))))
    return np.array(sorted(out))


def load(model, pid=1):
    f = np.load(f"{DATA}/{model}/problem_{pid}.npz")
    X = f["mesh_nodes"].T
    edges = f["mesh_edges"].T
    tets = tets_from_tris(f["mesh_tris"], len(X))
    nE = len(edges)
    sid = f["solve_ids"]
    free = np.zeros(nE, bool)
    free[sid[sid < nE]] = True
    d = dict(coords=X, tets=tets, edges=edges, free=free, k0=float(f["k0"]),
             E=_csr(f, "E"), B=_csr(f, "B"), solve_ids=sid, n_edges=nE,
             n_tris=f["mesh_tris"].shape[1])
    for k in ("ABC2C", "ABC2D"):
        if k + "_data" in f:
            d[k] = _csr(f, k)
    return d
