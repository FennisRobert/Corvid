"""Space coarsening: one Level -> the next, with prolongations P1, P2, P3.

parELAG's construction without targets (DeRhamSequence::Coarsen), top down:

  L2     P3[t, AE]  = w_t / w_AE
  H(div) traces     P2[f, AF] = s_f w_f / w_AF       (unit total flux)
         into AE    min |u|_M2  s.t.  div u in span(P3[:, AE])      (facet saddle point)
  H(curl) traces    P1[e, AR] = s_e w_e / w_AR       (unit circulation)
         into AF    min |u|_M1F s.t.  curl u in span(P2[:, AF])     (facet saddle point)
         into AE    min |u|_M1  s.t.  curl u = P2 D1c               (ridge saddle point,
                                                                     regularized by div-div)

The Lagrange multiplier of each facet constraint is the coarse derivative entry;
it is checked against the coarse incidence, which is what makes the coarse level
again a complex with D = signed incidence.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from .agglomerate import CoarseTopology, _Topo, _rows, coarsen_topology
from .level import Level


def _loc(ids):
    return {int(x): i for i, x in enumerate(ids)}


def _gather(n, ids, mats, sel):
    """Assemble entity-local dense matrices over `sel` onto the index list `n`."""
    m = _loc(n); A = np.zeros((len(n), len(n)))
    for k in sel:
        p = [m[int(x)] for x in ids[k]]
        A[np.ix_(p, p)] += mats[k]
    return A


def _solve(K, rhs):
    return la.solve(K, rhs, assume_a="sym")


def coarsen(L: Level, part=None, target=8, tol=1e-8):
    T = _Topo(L)
    C: CoarseTopology = coarsen_topology(L, part, target)
    nAE, nAF, nAR = len(C.ae_elems), len(C.af_faces), len(C.ar_edges)
    D1, D2 = T.D1, T.D2

    # ---------- L2
    r, c, v = [], [], []
    w_ae = np.array([L.w_elem[g].sum() for g in C.ae_elems])
    for a, g in enumerate(C.ae_elems):
        r += list(g); c += [a] * len(g); v += list(L.w_elem[g] / w_ae[a])
    P3 = sp.csr_matrix((v, (r, c)), shape=(L.n_elem, nAE))
    M3c = np.array([(L.w_elem[g] / w_ae[a]) ** 2 @ L.elem_M3[g] for a, g in enumerate(C.ae_elems)])

    # ---------- H(div): traces
    P2 = sp.lil_matrix((L.n_face, nAF))
    w_af = np.array([L.w_face[fs].sum() for fs, _ in C.af_faces])
    for j, (fs, ss) in enumerate(C.af_faces):
        for f, s in zip(fs, ss):
            P2[f, j] = s * L.w_face[f] / w_af[j]
    face_M2c = np.array([sum(P2[f, j] ** 2 * L.face_M2[f] for f in fs) for j, (fs, _) in enumerate(C.af_faces)])

    ae_info = []
    D2c, D2cT = C.D2c.tocsr(), C.D2c.T.tocsr()
    for a, g in enumerate(C.ae_elems):
        V, Ea, Fa = T.closure(g)
        bf, _ = T.boundary_faces(g)
        Fi = np.setdiff1d(Fa, bf)
        afs = _rows(D2c, a)[0]
        ae_info.append((g, Ea, Fa, bf, Fi, afs))
        if len(Fi) == 0:
            continue
        M2a = _gather(Fa, L.elem_faces, L.elem_M2, g)
        fl = _loc(Fa); ii = [fl[f] for f in Fi]; bb = [fl[f] for f in bf]
        W3 = L.elem_M3[g]
        D2a = D2[g][:, Fa].toarray()
        B = W3[:, None] * D2a
        Tl = W3 * (L.w_elem[g] / w_ae[a])
        nu, npp = len(Fi), len(g)
        K = np.zeros((nu + npp + 1, nu + npp + 1))
        K[:nu, :nu] = M2a[np.ix_(ii, ii)]
        K[nu:nu + npp, :nu] = B[:, ii]; K[:nu, nu:nu + npp] = B[:, ii].T
        K[-1, nu:nu + npp] = Tl; K[nu:nu + npp, -1] = Tl
        R = P2[bf][:, afs].toarray()
        rhs = np.zeros((K.shape[0], len(afs)))
        rhs[:nu] = -M2a[np.ix_(ii, bb)] @ R
        rhs[nu:nu + npp] = -B[:, bb] @ R
        x = _solve(K, rhs)
        for k, j in enumerate(afs):
            for i, f in enumerate(Fi):
                P2[f, j] = x[i, k]
        dc = -x[-1]
        if np.abs(dc - D2c[a, afs].toarray().ravel()).max() > tol:
            raise RuntimeError(f"AE {a}: coarse divergence {dc} != incidence")
    P2 = P2.tocsr()
    elem_M2c, elem_faces_c = [], []
    for a, (g, Ea, Fa, bf, Fi, afs) in enumerate(ae_info):
        M2a = _gather(Fa, L.elem_faces, L.elem_M2, g)
        Pa = P2[Fa][:, afs].toarray()
        elem_M2c.append(Pa.T @ M2a @ Pa); elem_faces_c.append(afs)

    # ---------- H(curl): traces
    P1 = sp.lil_matrix((L.n_edge, nAR))
    w_ar = np.array([L.w_edge[es].sum() for es, _ in C.ar_edges])
    for j, (es, ss) in enumerate(C.ar_edges):
        for e, s in zip(es, ss):
            P1[e, j] = s * L.w_edge[e] / w_ar[j]
    D1c = C.D1c.tocsr()

    # into AF
    face_M1c, face_edges_c = [], []
    for j, (fs, ss) in enumerate(C.af_faces):
        Ef = np.unique(np.concatenate([L.face_edges[f] for f in fs]))
        be, _ = T.boundary_edges(fs, ss)
        Ei = np.setdiff1d(Ef, be)
        ars = _rows(D1c, j)[0]
        Mf = _gather(Ef, L.face_edges, L.face_M1, fs)
        el = _loc(Ef); ii = [el[e] for e in Ei]
        if len(Ei):
            bb = [el[e] for e in be]
            Wf = L.face_M2[fs]
            B = Wf[:, None] * D1[fs][:, Ef].toarray()
            Tl = Wf * np.asarray(P2[fs, j].toarray()).ravel()
            nu, npp = len(Ei), len(fs)
            K = np.zeros((nu + npp + 1, nu + npp + 1))
            K[:nu, :nu] = Mf[np.ix_(ii, ii)]
            K[nu:nu + npp, :nu] = B[:, ii]; K[:nu, nu:nu + npp] = B[:, ii].T
            K[-1, nu:nu + npp] = Tl; K[nu:nu + npp, -1] = Tl
            R = P1[be][:, ars].toarray()
            rhs = np.zeros((K.shape[0], len(ars)))
            rhs[:nu] = -Mf[np.ix_(ii, bb)] @ R
            rhs[nu:nu + npp] = -B[:, bb] @ R
            x = _solve(K, rhs)
            for k, r_ in enumerate(ars):
                for i, e in enumerate(Ei):
                    P1[e, r_] = x[i, k]
            dc = -x[-1]
            if np.abs(dc - D1c[j, ars].toarray().ravel()).max() > tol:
                raise RuntimeError(f"AF {j}: coarse curl {dc} != incidence")
        Pf = P1[Ef][:, ars].toarray()
        face_M1c.append(Pf.T @ Mf @ Pf); face_edges_c.append(ars)

    # into AE
    elem_M1c, elem_edges_c = [], []
    for a, (g, Ea, Fa, bf, Fi, afs) in enumerate(ae_info):
        ars = np.unique(np.concatenate([_rows(D1c, j)[0] for j in afs]))
        bE = np.unique(np.concatenate([_rows(D1, f)[0] for f in bf]))
        Ei = np.setdiff1d(Ea, bE)
        M1a = _gather(Ea, L.elem_edges, L.elem_M1, g)
        if len(Ei):
            M2a = _gather(Fa, L.elem_faces, L.elem_M2, g)
            el, fl = _loc(Ea), _loc(Fa)
            ii, bb = [el[e] for e in Ei], [el[e] for e in bE]
            fi = [fl[f] for f in Fi]
            WD = M2a @ D1[Fa][:, Ea].toarray()
            D2a = D2[g][:, Fa].toarray()
            Cm = D2a[:, fi].T @ (L.elem_M3[g][:, None] * D2a[:, fi])
            nu, npp = len(Ei), len(Fi)
            K = np.zeros((nu + npp, nu + npp))
            K[:nu, :nu] = M1a[np.ix_(ii, ii)]
            K[nu:, :nu] = WD[np.ix_(fi, ii)]; K[:nu, nu:] = WD[np.ix_(fi, ii)].T
            K[nu:, nu:] = -Cm
            R = P1[bE][:, ars].toarray()
            PDc = P2[Fa][:, afs].toarray() @ D1c[afs][:, ars].toarray()
            rhs = np.zeros((nu + npp, len(ars)))
            rhs[:nu] = -M1a[np.ix_(ii, bb)] @ R
            rhs[nu:] = -WD[np.ix_(fi, bb)] @ R + M2a[fi] @ PDc
            x = _solve(K, rhs)
            for k, r_ in enumerate(ars):
                for i, e in enumerate(Ei):
                    P1[e, r_] = x[i, k]
        Pa = P1[Ea][:, ars].toarray()
        elem_M1c.append(Pa.T @ M1a @ Pa); elem_edges_c.append(ars)
    P1 = P1.tocsr()

    Lc = Level(D0=C.D0c, D1=C.D1c.tocsr(), D2=C.D2c.tocsr(),
               elem_edges=elem_edges_c, elem_M1=elem_M1c,
               elem_faces=elem_faces_c, elem_M2=elem_M2c, elem_M3=M3c,
               face_edges=face_edges_c, face_M1=face_M1c, face_M2=face_M2c,
               w_edge=w_ar, w_face=w_af, w_elem=w_ae, free_edge=C.ar_free)
    return Lc, (P1, P2, P3), C


def check(L: Level, Lc: Level, P):
    """Max abs violation of each identity the construction must satisfy."""
    P1, P2, P3 = P
    out = {
        "D1c D0c": abs(Lc.D1 @ Lc.D0).max(),
        "D2c D1c": abs(Lc.D2 @ Lc.D1).max(),
        "D1 P1 - P2 D1c": abs(L.D1 @ P1 - P2 @ Lc.D1).max(),
        "D2 P2 - P3 D2c": abs(L.D2 @ P2 - P3 @ Lc.D2).max(),
        "M1c - P1'M1P1": abs(Lc.assemble_M1() - P1.T @ L.assemble_M1() @ P1).max() / abs(Lc.assemble_M1()).max(),
        "PEC rows of free cols": abs(P1[~L.free_edge][:, Lc.free_edge]).max()
        if (~L.free_edge).any() and Lc.free_edge.any() else 0.0,
    }
    return out
