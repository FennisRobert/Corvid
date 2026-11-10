"""Smaller Schwarz blocks: iterations vs block memory. usage: block_variants.py model pid"""
import sys, time
import numpy as np, scipy.sparse as sp
sys.path[:0] = ["tests", "research/probes"]
from test_api import emerge_inputs
from corvid import _kernels

model, pid = sys.argv[1], int(sys.argv[2])
edges, tris, sid, K, b = emerge_inputs(model, pid)
E, F = edges.T.astype(np.int64), tris.T.astype(np.int64)
ne, nf, n = len(E), len(F), K.shape[0]
nv = int(max(E.max(), F.max())) + 1
pos = -np.ones(2 * ne + 2 * nf, np.int64); pos[sid] = np.arange(n)
de1, de2 = pos[:ne], pos[ne:2 * ne]                    # solved index of edge DOFs (or -1)
df1, df2 = pos[2 * ne:2 * ne + nf], pos[2 * ne + nf:]
whit = np.flatnonzero(np.isin(np.arange(n), de1[de1 >= 0])).astype(np.int64)
whit = np.sort(de1[de1 >= 0]).astype(np.int64)

def blocks_from(rows, cols, nb):
    """blocks = columns of the (dof x block) incidence"""
    m = sp.csc_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, nb)); m.sum_duplicates(); m.sort_indices()
    keep = np.flatnonzero(np.diff(m.indptr)); m = m[:, keep]
    return m.indptr.astype(np.int64), m.indices.astype(np.int32)

def ent_dofs(dofs_per_entity, ent_ids, block_ids):
    r, c = [], []
    for d in dofs_per_entity:
        x = d[ent_ids]; ok = x >= 0
        r.append(x[ok]); c.append(block_ids[ok])
    return np.concatenate(r), np.concatenate(c)

variants = {}
# V1: vertex star, all DOFs on incident edges and faces (current)
r1, c1 = ent_dofs([de1, de2], np.repeat(np.arange(ne), 2), E.ravel())
r2, c2 = ent_dofs([df1, df2], np.repeat(np.arange(nf), 3), F.ravel())
variants["vertex: edges+faces (current)"] = blocks_from(np.r_[r1, r2], np.r_[c1, c2], nv)
# V2: vertex star, edge DOFs only + one block per face (its 2 DOFs)
rf, cf = ent_dofs([df1, df2], np.arange(nf), nv + np.arange(nf))
variants["vertex: edges only + face pairs"] = blocks_from(np.r_[r1, rf], np.r_[c1, cf], nv + nf)
# V3: edge blocks: an edge's DOFs + DOFs of the faces containing it
fe = np.stack([np.minimum(F[:, [0, 1, 0]], F[:, [1, 2, 2]]), np.maximum(F[:, [0, 1, 0]], F[:, [1, 2, 2]])], -1)
ekey = {tuple(sorted(e)): i for i, e in enumerate(E)}
f_edges = np.array([[ekey[tuple(fe[f, k])] for k in range(3)] for f in range(nf)])
re_, ce_ = ent_dofs([de1, de2], np.arange(ne), np.arange(ne))
rfe, cfe = ent_dofs([df1, df2], np.repeat(np.arange(nf), 3), f_edges.ravel())
variants["edge: edge + incident faces"] = blocks_from(np.r_[re_, rfe], np.r_[ce_, cfe], ne)
# V4: face blocks: a face's DOFs + DOFs of its 3 edges
rfa, cfa = ent_dofs([df1, df2], np.arange(nf), np.arange(nf))
rea, cea = ent_dofs([de1, de2], f_edges.ravel(), np.repeat(np.arange(nf), 3))
variants["face: face + its 3 edges"] = blocks_from(np.r_[rfa, rea], np.r_[cfa, cea], nf)

Kc = sp.csr_matrix(K); Kc.sort_indices()
k = (Kc.indptr.astype(np.int32), Kc.indices.astype(np.int32), Kc.data.astype(np.complex128), n, n)
kb = (Kc.data.nbytes + Kc.indices.nbytes) / 1e6
print(f"{model} p{pid}: n={n}, K {kb:.0f} MB")
for name, (bp, bi) in variants.items():
    m = np.diff(bp)
    for order in (2, 4):
        t0 = time.time(); s = _kernels.TwoLevelSolver(k, bp, bi, whit, order=order); tf = time.time() - t0
        t0 = time.time(); x, its, rr = s.solve(np.ascontiguousarray(b[:, 0] if b.ndim > 1 else b, dtype=np.complex128), "gmres", 1e-6, 100, 300); ts = time.time() - t0
        print(f"  {name:34s} cheb {order}: {len(m):6d} blocks, size {m.mean():5.1f} (max {m.max():3d}), "
              f"memory {s.schwarz_bytes/1e6:6.0f} MB = {s.schwarz_bytes/1e6/kb:4.2f}x K | GMRES {its:3d} its {'' if rr < 1e-6 else 'rr=%.0e ' % rr}"
              f"| factorize {tf:.2f}s solve {ts:.2f}s", flush=True)
        del s
