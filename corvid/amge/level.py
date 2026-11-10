"""One level of the hierarchy: a lowest-order complex on a polyhedral mesh.

Without targets every level has exactly one DOF per entity, and the exterior
derivatives are the signed incidences D0 (edge x vertex), D1 (face x edge),
D2 (element x face). The geometry lives only in the entity-local mass matrices,
so the fine Whitney level and every agglomerated level share this one type.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Level:
    D0: sp.csr_matrix
    D1: sp.csr_matrix
    D2: sp.csr_matrix
    elem_edges: list          # per element: edge ids in its closure
    elem_M1: list             # per element: dense H(curl) mass on those edges
    elem_faces: list          # per element: face ids on its boundary
    elem_M2: list             # per element: dense H(div) mass on those faces
    elem_M3: np.ndarray       # per element: L2 mass (scalar)
    face_edges: list          # per face: edge ids on its boundary
    face_M1: list             # per face: dense tangential-trace H(curl) mass
    face_M2: np.ndarray       # per face: normal-trace H(div) mass (scalar)
    w_edge: np.ndarray        # length, area, volume: weights of the PV traces
    w_face: np.ndarray
    w_elem: np.ndarray
    free_edge: np.ndarray     # H(curl) DOFs not on PEC

    @property
    def n_vert(self): return self.D0.shape[1]
    @property
    def n_edge(self): return self.D0.shape[0]
    @property
    def n_face(self): return self.D1.shape[0]
    @property
    def n_elem(self): return self.D2.shape[0]

    def assemble_M1(self):
        return _assemble(self.n_edge, self.elem_edges, self.elem_M1)

    def assemble_M2(self):
        return _assemble(self.n_face, self.elem_faces, self.elem_M2)


def _assemble(n, ids, mats):
    r = np.concatenate([np.repeat(i, len(i)) for i in ids])
    c = np.concatenate([np.tile(i, len(i)) for i in ids])
    v = np.concatenate([np.asarray(m).ravel() for m in mats])
    return sp.csr_matrix((v, (r, c)), shape=(n, n))
