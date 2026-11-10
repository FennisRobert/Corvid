"""Agglomeration: partition the elements, then derive the coarse topology.

Coarse faces (AF), edges (AR) and vertices (AP) follow parELAG's rule: fine
entities with the same set of neighbouring coarse entities belong together. Two
additions over parELAG make every coarse entity a cell, which the local saddle
points need to be nonsingular and exact:

  AE  connected, a topological ball (chi = 1, boundary chi = 2), no PEC edge inside
  AF  connected through FREE edges (so PEC edges land on AF boundaries), a disk
  AR  a simple open path between two distinct APs

Entities that fail are split, never dissolved into fine elements.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .level import Level


@dataclass
class CoarseTopology:
    ae_elems: list            # per AE: element ids
    af_faces: list            # per AF: (face ids, signs relative to the AF)
    ar_edges: list            # per AR: (edge ids, signs relative to the AR)
    ap_verts: np.ndarray      # per AP: the fine vertex
    D0c: sp.csr_matrix        # AR x AP
    D1c: sp.csr_matrix        # AF x AR
    D2c: sp.csr_matrix        # AE x AF
    ar_free: np.ndarray


def _rows(A, i):
    return A.indices[A.indptr[i]:A.indptr[i + 1]], A.data[A.indptr[i]:A.indptr[i + 1]]


def _components(nodes, nbrs):
    """Connected components of `nodes` under nbrs(x) -> iterable."""
    left, out = set(nodes), []
    while left:
        s = left.pop(); comp = [s]; stack = [s]
        while stack:
            x = stack.pop()
            for y in nbrs(x):
                if y in left:
                    left.remove(y); comp.append(y); stack.append(y)
        out.append(comp)
    return out


def _bisect(nodes, nbrs):
    """Split a connected set in two by BFS from a peripheral node."""
    nodes = list(nodes); ns = set(nodes)
    for _ in range(2):                      # two sweeps: find a far node, BFS from it
        order = [nodes[0]]; seen = {nodes[0]}; i = 0
        while i < len(order):
            for y in nbrs(order[i]):
                if y in ns and y not in seen:
                    seen.add(y); order.append(y)
            i += 1
        nodes = order[::-1]
    half = len(order) // 2
    return order[:half], order[half:]


class _Topo:
    """Index helpers on one Level."""

    def __init__(self, L: Level):
        self.L = L
        self.D0, self.D1, self.D2 = L.D0.tocsr(), L.D1.tocsr(), L.D2.tocsr()
        self.D2T = self.D2.T.tocsr()        # face -> elems
        self.D1T = self.D1.T.tocsr()        # edge -> faces
        self.D0T = self.D0.T.tocsr()        # vert -> edges
        nonfree = ~L.free_edge
        # PEC face: every edge non-free (a PEC sheet face is interior but PEC)
        self.pec_face = np.array([nonfree[_rows(self.D1, f)[0]].all() for f in range(L.n_face)])
        self.nonfree = nonfree

    def elem_nbrs(self, t, cut_pec=True):
        for f in _rows(self.D2, t)[0]:
            if cut_pec and self.pec_face[f]:
                continue
            for s in _rows(self.D2T, f)[0]:
                if s != t:
                    yield s

    def closure(self, elems):
        F = np.unique(np.concatenate([_rows(self.D2, t)[0] for t in elems]))
        E = np.unique(np.concatenate([_rows(self.D1, f)[0] for f in F]))
        V = np.unique(np.concatenate([_rows(self.D0, e)[0] for e in E]))
        return V, E, F

    def boundary_faces(self, elems):
        s = np.asarray(self.D2[np.asarray(elems)].sum(axis=0)).ravel()
        return np.flatnonzero(s), s

    def boundary_edges(self, faces, signs):
        s = np.zeros(self.L.n_edge)
        for f, g in zip(faces, signs):
            c, v = _rows(self.D1, f)
            s[c] += g * v
        nz = np.flatnonzero(np.abs(s) > 0.5)
        return nz, s[nz]


def _cell(T: _Topo, t):
    F = set(_rows(T.D2, t)[0].tolist())
    E = set(np.concatenate([_rows(T.D1, f)[0] for f in F]).tolist())
    V = set(np.concatenate([_rows(T.D0, e)[0] for e in E]).tolist())
    return V, E, F


def _glue_ok(T: _Topo, cell, S):
    """Whether ball S stays a ball when element `cell` is glued on: they must
    meet in exactly one disk of shared faces, none of it PEC (no PEC face or
    edge may end up inside an AE)."""
    (V, E, F), (SV, SE, SF) = cell, S
    shared = F & SF
    if not shared or any(T.pec_face[f] for f in shared):
        return False
    fe = {f: set(_rows(T.D1, f)[0].tolist()) for f in shared}
    dE = set().union(*fe.values())
    dV = set(np.concatenate([_rows(T.D0, e)[0] for e in dE]).tolist())
    if (E & SE) != dE or (V & SV) != dV:          # touching outside the disk
        return False
    if len(dV) - len(dE) + len(shared) != 1:
        return False
    cnt = defaultdict(int)
    for es in fe.values():
        for e in es:
            cnt[e] += 1
    inner = [e for e, c in cnt.items() if c == 2]
    if any(T.nonfree[e] for e in inner):
        return False
    bverts = defaultdict(int)                        # disk boundary must be one simple loop
    for e, c in cnt.items():
        if c == 1:
            for x in _rows(T.D0, e)[0]:
                bverts[int(x)] += 1
    if any(c != 2 for c in bverts.values()):
        return False
    return len(_components(list(shared), lambda f: (g for e in fe[f] if cnt[e] == 2
                                                       for g in shared if g != f and e in fe[g]))) == 1


def grow(T: _Topo, target=8):
    """Region growing: seeds in order, then glue the neighbour sharing the most
    faces while the union stays a ball. Every AE is valid by construction."""
    n = T.L.n_elem
    cells = [_cell(T, t) for t in range(n)]
    part = -np.ones(n, int); k = 0
    for seed in range(n):
        if part[seed] >= 0:
            continue
        part[seed] = k
        S = tuple(set(x) for x in cells[seed]); members = 1
        while members < target:
            cand = defaultdict(int)
            for f in S[2]:
                for t in _rows(T.D2T, f)[0]:
                    if part[t] < 0:
                        cand[t] += 1
            added = False
            for t, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
                if _glue_ok(T, cells[t], S):
                    part[t] = k; members += 1; added = True
                    for a, b in zip(S, cells[t]):
                        a |= b
                    break
            if not added:
                break
        k += 1
    return part


def _ae_ok(T: _Topo, elems):
    V, E, F = T.closure(elems)
    if len(V) - len(E) + len(F) - len(elems) != 1:
        return False
    bf, _ = T.boundary_faces(elems)
    bE = np.unique(np.concatenate([_rows(T.D1, f)[0] for f in bf]))
    bV = np.unique(np.concatenate([_rows(T.D0, e)[0] for e in bE]))
    if len(bV) - len(bE) + len(bf) != 2:
        return False
    # manifold boundary: every boundary edge in exactly two boundary faces, and
    # the boundary faces at every boundary vertex form one fan (a pinched AE
    # passes the Euler test but lets two AFs share an interior edge)
    fe = {int(f): set(_rows(T.D1, f)[0].tolist()) for f in bf}
    ecount = defaultdict(list)
    for f, es in fe.items():
        for e in es:
            ecount[e].append(f)
    if any(len(fs) != 2 for fs in ecount.values()):
        return False
    for x in bV:
        ex = set(_rows(T.D0T, x)[0].tolist())
        fx = [f for f, es in fe.items() if es & ex]
        fxs = set(fx)
        nb = lambda f: (g for e in fe[f] & ex for g in ecount[e] if g in fxs)
        if len(_components(fx, nb)) != 1:
            return False
    inner = np.setdiff1d(E, bE)
    return not T.nonfree[inner].any()


def _af_ok(T: _Topo, faces, signs, cut=()):
    E = np.unique(np.concatenate([_rows(T.D1, f)[0] for f in faces]))
    V = np.unique(np.concatenate([_rows(T.D0, e)[0] for e in E]))
    if len(V) - len(E) + len(faces) != 1:
        return False
    be, _ = T.boundary_edges(faces, signs)
    deg = defaultdict(int)
    for e in be:
        for v in _rows(T.D0, e)[0]:
            deg[v] += 1
    if any(d != 2 for d in deg.values()):
        return False
    inner = np.setdiff1d(E, be)
    if any(int(e) in cut for e in inner):
        return False
    return not T.nonfree[inner].any() or T.pec_face[faces].all()


def coarsen_topology(L: Level, part=None, target=8) -> CoarseTopology:
    T = _Topo(L)
    if part is None:
        part = grow(T, target)

    # ---- AE: connected components, then split until every AE is a ball
    groups = defaultdict(list)
    for t, p in enumerate(part):
        groups[p].append(t)
    todo = []
    for g in groups.values():
        todo += _components(g, lambda t: T.elem_nbrs(t, cut_pec=True))
    ae = []
    while todo:
        g = todo.pop()
        if len(g) == 1 or _ae_ok(T, g):
            ae.append(sorted(g))
        else:
            sub = set(g)
            for h in _bisect(g, lambda t: (s for s in T.elem_nbrs(t) if s in sub)):
                todo += _components(h, lambda t, h=set(h): (s for s in T.elem_nbrs(t) if s in h))
    elem_ae = np.empty(L.n_elem, int)
    for i, g in enumerate(ae):
        elem_ae[g] = i

    # ---- AF: faces grouped by (AE pair, PEC); oriented outward from the lower AE
    sig = defaultdict(list)
    for f in range(L.n_face):
        el, sg = _rows(T.D2T, f)
        a = elem_ae[el]
        if len(el) == 2 and a[0] == a[1]:
            continue
        k = int(np.argmin(a))
        key = (int(a[k]), int(a[1 - k]) if len(el) == 2 else -1, bool(T.pec_face[f]))
        sig[key].append((f, float(sg[k])))
    # An AF may not contain, as an interior edge, an edge on another AF's boundary
    # (around such an edge the AEs alternate). Cut along every AF boundary and
    # regroup until that no longer happens.
    cut = set()
    while True:
        edge_faces = lambda f: (g for e in _rows(T.D1, f)[0]
                                if (not T.nonfree[e] or T.pec_face[f]) and e not in cut
                                for g in _rows(T.D1T, e)[0])
        af, af_ae = [], []
        for key, lst in sig.items():
            sgn = dict(lst)
            todo = _components(list(sgn), lambda f: (g for g in edge_faces(f) if g in sgn))
            while todo:
                c = todo.pop()
                s = [sgn[f] for f in c]
                if len(c) == 1 or _af_ok(T, c, s, cut):
                    af.append((np.array(c), np.array(s))); af_ae.append(key[:2])
                else:
                    sub = set(c)
                    for h in _bisect(c, lambda f: (g for g in edge_faces(f) if g in sub)):
                        hs = set(h)
                        todo += _components(h, lambda f: (g for g in edge_faces(f) if g in hs))
        bnd = set()
        for fs, ss in af:
            bnd.update(T.boundary_edges(fs, ss)[0].tolist())
        inner = set()
        for fs, ss in af:
            E = set(np.concatenate([_rows(T.D1, f)[0] for f in fs]).tolist())
            inner |= E - set(T.boundary_edges(fs, ss)[0].tolist())
        if not (inner & bnd):
            break
        cut |= inner & bnd
    nAF = len(af)
    r, c, v = [], [], []
    for i, (lo, hi) in enumerate(af_ae):
        r.append(lo); c.append(i); v.append(1.0)
        if hi >= 0:
            r.append(hi); c.append(i); v.append(-1.0)
    D2c = sp.csr_matrix((v, (r, c)), shape=(len(ae), nAF))

    # ---- AR: AF-boundary edges grouped by (set of AFs, free); chains between APs
    edge_afs = defaultdict(list)
    af_bdry = []
    for i, (fs, ss) in enumerate(af):
        be, bs = T.boundary_edges(fs, ss)
        af_bdry.append(dict(zip(be.tolist(), bs.tolist())))
        for e in be:
            edge_afs[int(e)].append(i)
    edge_grp = {e: (tuple(sorted(a)), bool(T.nonfree[e])) for e, a in edge_afs.items()}
    vert_edges = defaultdict(list)
    for e in edge_grp:
        for x in _rows(T.D0, e)[0]:
            vert_edges[int(x)].append(e)
    special = {x for x, es in vert_edges.items()
               if len(es) != 2 or edge_grp[es[0]] != edge_grp[es[1]]}
    other = lambda e, x: int([y for y in _rows(T.D0, e)[0] if y != x][0])
    seen, chains = set(), []

    def walk(x, e):
        path = [(e, x)]; seen.add(e)
        y = other(e, x)
        while y not in special:
            nxt = [d for d in vert_edges[y] if d not in seen]
            if not nxt:
                break
            e = nxt[0]; path.append((e, y)); seen.add(e); y = other(e, y)
        return path, y

    for x in list(special):
        for e in vert_edges[x]:
            if e not in seen:
                chains.append(walk(x, e))
    for e in edge_grp:                     # pure loops: split at two vertices
        if e in seen:
            continue
        x = int(_rows(T.D0, e)[0][0])
        loop = [e]; y = other(e, x); cur = e
        while y != x:
            cur = [d for d in vert_edges[y] if d != cur][0]; loop.append(cur); y = other(cur, y)
        a, b = x, None
        y = x
        for d in loop[: len(loop) // 2]:
            y = other(d, y)
        b = y
        special.update({a, b})
        chains.append(walk(a, e))
        rest = [d for d in vert_edges[y] if d not in seen]
        if rest:
            chains.append(walk(y, rest[0]))
    ap = sorted(special)
    ap_id = {x: i for i, x in enumerate(ap)}
    ar, d0r, d0c, d0v, ar_free = [], [], [], [], []
    for j, (path, end) in enumerate(chains):
        es = np.array([e for e, _ in path])
        # sign +1 when the fine edge points along the walk (its start vertex is the walk vertex)
        sg = np.array([-T.D0[e, x] for e, x in path], dtype=float)
        ar.append((es, sg))
        start = path[0][1]
        if start == end:
            raise RuntimeError("closed AR survived loop splitting")
        d0r += [j, j]; d0c += [ap_id[start], ap_id[end]]; d0v += [-1.0, 1.0]
        ar_free.append(not T.nonfree[es[0]])
    D0c = sp.csr_matrix((d0v, (d0r, d0c)), shape=(len(ar), len(ap)))
    edge_ar = {}
    for j, (es, sg) in enumerate(ar):
        for e, s in zip(es, sg):
            edge_ar[int(e)] = (j, s)
    r, c, v = [], [], []
    for i, bd in enumerate(af_bdry):
        acc = {}
        for e, s in bd.items():
            j, s2 = edge_ar[e]
            val = s * s2
            if acc.setdefault(j, val) != val:
                raise RuntimeError(f"AF {i}: inconsistent orientation along AR {j}")
        for j, val in acc.items():
            r.append(i); c.append(j); v.append(val)
    D1c = sp.csr_matrix((v, (r, c)), shape=(nAF, len(ar)))
    return CoarseTopology(ae, af, ar, np.array(ap), D0c, D1c, D2c, np.array(ar_free))
