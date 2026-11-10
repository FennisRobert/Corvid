"""Threaded sparse products from the Rust extension, behind a scipy-like `@`."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import _kernels as _k


class FastCsr:
    """Read-only CSR matrix whose `@ vector` runs on all cores."""

    def __init__(self, A):
        A = sp.csr_matrix(A)
        A.sort_indices()
        self.dtype = np.dtype(np.complex128 if np.iscomplexobj(A.data) else np.float64)
        self.shape = A.shape
        self.nnz = A.nnz
        self.ip = A.indptr.astype(np.int32)
        self.ix = A.indices.astype(np.int32)
        self.a = np.ascontiguousarray(A.data, dtype=self.dtype)
        self._csr = A
        self._T = None

    def _kern(self, vdt):
        if self.dtype == np.complex128:
            return _k.spmv_c128, _k.residual_c128
        if vdt == np.complex128:
            return _k.spmv_rc, _k.residual_rc
        return _k.spmv_f64, _k.residual_f64

    def __matmul__(self, x):
        if x.ndim != 1:
            return self._csr @ x
        vdt = np.result_type(self.dtype, x.dtype)
        xx = np.ascontiguousarray(x, dtype=vdt)
        y = np.empty(self.shape[0], dtype=vdt)
        self._kern(vdt)[0](self.ip, self.ix, self.a, xx, y)
        return y

    def residual(self, b, x):
        """b - A x in one pass."""
        vdt = np.result_type(self.dtype, x.dtype, b.dtype)
        xx = np.ascontiguousarray(x, dtype=vdt)
        bb = np.ascontiguousarray(b, dtype=vdt)
        y = np.empty(self.shape[0], dtype=vdt)
        self._kern(vdt)[1](self.ip, self.ix, self.a, xx, bb, y)
        return y

    def gauss_seidel(self, x, b, forward=True, sweeps=1, nblocks=1):
        """In-place GS on self x = b. nblocks > 1: hybrid block GS in parallel."""
        if self.dtype == np.complex128 or np.iscomplexobj(x) or np.iscomplexobj(b):
            if self.dtype != np.complex128:
                raise TypeError("real matrix with complex vectors: split the parts")
            f = _k.gs_c128
        else:
            f = _k.gs_f64
        if not (x.flags.c_contiguous and x.dtype == self.dtype):
            raise TypeError("x must be a contiguous array of the matrix dtype")
        f(self.ip, self.ix, self.a, x, np.ascontiguousarray(b, dtype=self.dtype), forward, sweeps, nblocks)

    def diagonal(self):
        return self._csr.diagonal()

    @property
    def T(self):
        if self._T is None:
            self._T = FastCsr(self._csr.T)
            self._T._T = self
        return self._T
