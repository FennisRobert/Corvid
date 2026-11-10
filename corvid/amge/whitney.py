"""The fine level: the lowest-order de Rham complex of a tetrahedral mesh.

DOFs are de Rham integrals: H1 point values, H(curl) line integrals along each
edge from edges[e,0] to edges[e,1] (EMerge's Whitney convention, its G has -1/+1
there), H(div) fluxes through each face, L2 integrals over each tet. With that
choice the exterior derivatives are the signed incidence matrices, exactly.

Input is only what EMerge must send: node coordinates, tets, and the edge list
(edge order = Whitney DOF order). Faces are derived here and oriented by their
sorted node triple (a<b<c, normal (b-a)x(c-a)).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .level import Level

_TET_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_TET_FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])  # opposite vertex i


def _key(a, b, n):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo.astype(np.int64) * n + hi


def fine_level(coords, tets, edges, free_edges=None, eps=None, inv_mu=None) -> Level:
    X = np.asarray(coords, dtype=float).reshape(-1, 3)
    tets = np.asarray(tets, dtype=np.int64).reshape(-1, 4).copy()
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    nV, nT, nE = len(X), len(tets), len(edges)

    # positive orientation: det[x1-x0, x2-x0, x3-x0] > 0
    d = X[tets[:, 1:]] - X[tets[:, :1]]
    vol6 = np.einsum("ij,ij->i", np.cross(d[:, 0], d[:, 1]), d[:, 2])
    neg = vol6 < 0
    tets[neg, 2], tets[neg, 3] = tets[neg, 3].copy(), tets[neg, 2].copy()
    vol = np.abs(vol6) / 6.0

    # edges: global lookup, local (i<j) orientation vs global
    ekey = _key(edges[:, 0], edges[:, 1], nV)
    order = np.argsort(ekey)
    la, lb = tets[:, _TET_EDGES[:, 0]], tets[:, _TET_EDGES[:, 1]]          # (T,6)
    pos = np.searchsorted(ekey[order], _key(la, lb, nV))
    tet_e = order[pos]
    if not np.array_equal(ekey[tet_e], _key(la, lb, nV)):
        raise ValueError("a tet edge is missing from the edge list")
    tet_es = np.where(edges[tet_e, 0] == la, 1.0, -1.0)                    # (T,6)

    # faces: unique sorted triples
    loc = tets[:, _TET_FACES]                                                # (T,4,3)
    srt = np.sort(loc, axis=2)
    fkey = (srt[..., 0] * nV + srt[..., 1]) * nV + srt[..., 2]
    ukey, inv = np.unique(fkey.ravel(), return_inverse=True)
    faces = np.stack([ukey // (nV * nV), (ukey // nV) % nV, ukey % nV], axis=1)
    nF = len(faces)
    tet_f = inv.reshape(nT, 4)
    # induced outward orientation of face opposite i is (-1)^i [the remaining
    # vertices in order]; compare with the sorted triple by permutation parity
    par = np.array([_parity(p) for p in np.argsort(loc.reshape(-1, 3), axis=1)]).reshape(nT, 4)
    tet_fs = par * np.array([1.0, -1.0, 1.0, -1.0])

    # incidences
    D0 = sp.csr_matrix((np.r_[-np.ones(nE), np.ones(nE)],
                        (np.r_[np.arange(nE), np.arange(nE)], np.r_[edges[:, 0], edges[:, 1]])),
                       shape=(nE, nV))
    fa, fb, fc = faces.T
    fe = np.stack([_key(fa, fb, nV), _key(fb, fc, nV), _key(fa, fc, nV)], axis=1)
    fe_idx = order[np.searchsorted(ekey[order], fe)]
    fe_sgn = np.where(edges[fe_idx, 0] == np.stack([fa, fb, fa], 1), 1.0, -1.0) * np.array([1, 1, -1])
    D1 = sp.csr_matrix((fe_sgn.ravel(), (np.repeat(np.arange(nF), 3), fe_idx.ravel())), shape=(nF, nE))
    D2 = sp.csr_matrix((tet_fs.ravel(), (np.repeat(np.arange(nT), 4), tet_f.ravel())), shape=(nT, nF))

    # element matrices
    eps = np.ones(nT) if eps is None else np.asarray(eps, float)
    inv_mu = np.ones(nT) if inv_mu is None else np.asarray(inv_mu, float)
    Xt = X[tets]                                                             # (T,4,3)
    A = np.concatenate([np.ones((nT, 4, 1)), Xt], axis=2)
    grad = np.linalg.inv(A)[:, 1:, :].transpose(0, 2, 1)                     # (T,4,3) grad lambda_i
    lam = (np.ones((4, 4)) + np.eye(4)) / 20.0                               # int l_i l_j / V
    gg = np.einsum("tid,tjd->tij", grad, grad)
    a, b = _TET_EDGES[:, 0], _TET_EDGES[:, 1]
    M1 = (lam[a][:, a] * gg[:, b][:, :, b] - lam[a][:, b] * gg[:, b][:, :, a]
          - lam[b][:, a] * gg[:, a][:, :, b] + lam[b][:, b] * gg[:, a][:, :, a])
    M1 = M1 * (vol * eps)[:, None, None] * tet_es[:, :, None] * tet_es[:, None, :]
    R = Xt[:, None, :, :] - Xt[:, :, None, :]                                # R[t,i,k] = x_k - x_i
    M2 = np.einsum("kl,tikd,tjld->tij", lam, R, R) / (9.0 * vol)[:, None, None]
    M2 = M2 * inv_mu[:, None, None] * tet_fs[:, :, None] * tet_fs[:, None, :]
    M3 = 1.0 / vol

    # face trace masses: tangential Whitney (3x3, edges ab, bc, ac) and normal flux (1/area)
    P = X[faces]
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)            # (F,3,2)
    gram = np.einsum("fdi,fdj->fij", J, J)
    area = 0.5 * np.sqrt(np.linalg.det(gram))
    g12 = np.einsum("fdi,fij->fjd", J, np.linalg.inv(gram))                   # grad l1, l2
    g = np.concatenate([-(g12[:, :1] + g12[:, 1:]), g12], axis=1)            # (F,3,3)
    lam2 = (np.ones((3, 3)) + np.eye(3)) / 12.0
    fgg = np.einsum("fid,fjd->fij", g, g)
    ea, eb = np.array([0, 1, 0]), np.array([1, 2, 2])
    MF1 = (lam2[ea][:, ea] * fgg[:, eb][:, :, eb] - lam2[ea][:, eb] * fgg[:, eb][:, :, ea]
           - lam2[eb][:, ea] * fgg[:, ea][:, :, eb] + lam2[eb][:, eb] * fgg[:, ea][:, :, ea])
    s = np.where(edges[fe_idx, 0] == np.stack([fa, fb, fa], 1), 1.0, -1.0)
    MF1 = MF1 * area[:, None, None] * s[:, :, None] * s[:, None, :]
    MF2 = 1.0 / area

    length = np.linalg.norm(X[edges[:, 1]] - X[edges[:, 0]], axis=1)
    free = np.ones(nE, bool) if free_edges is None else np.asarray(free_edges, bool)
    return Level(D0=D0, D1=D1, D2=D2,
                 elem_edges=list(tet_e), elem_M1=list(M1),
                 elem_faces=list(tet_f), elem_M2=list(M2), elem_M3=M3,
                 face_edges=list(fe_idx), face_M1=list(MF1), face_M2=MF2,
                 w_edge=length, w_face=area, w_elem=vol, free_edge=free)


def _parity(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s
