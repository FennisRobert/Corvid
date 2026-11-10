"""Finds a BLAS and LAPACK for SPARTA's CPU factorization (CORVID's coarse level) at import.

CORVID links its own copy of SPARTA, so it loads the library itself, the same way SPARTA's Python
package does (this module follows SPARTA's sparta/_blas.py, by Robert Fennis and Milan Rother).
The extension links Accelerate on Apple. Elsewhere it loads one at run time from the first of these
that works:

1. ``CORVID_BLAS`` or ``SPARTA_BLAS``, a path to an OpenBLAS or MKL shared library;
2. the ``scipy-openblas32`` wheel, which this package depends on off Apple;
3. the OpenBLAS SciPy's own wheel bundles, then a conda environment's OpenBLAS or MKL.

Without any, the factorization uses SPARTA's portable loops: correct, but several times slower.
``corvid.blas_library()`` reports which one is in use. Only the coarse factorization uses BLAS;
CORVID's own kernels and the coarse triangular solves do not.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

from ._kernels import _load_blas, blas_library

_PORTABLE = "portable loops"


def _scipy_openblas32() -> list[str]:
    try:
        import scipy_openblas32
    except ImportError:
        return []
    lib = scipy_openblas32.get_lib_dir()
    names = [n for n in os.listdir(lib) if n.startswith("libscipy_openblas") and n.endswith((".dll", ".so", ".dylib"))]
    return [os.path.normpath(os.path.join(lib, n)) for n in sorted(names)]


def _bundled() -> list[str]:
    """SciPy's vendored OpenBLAS (32-bit integers, ``scipy_`` prefix), then conda's."""
    found = []
    try:
        import scipy
        site = os.path.dirname(os.path.dirname(scipy.__file__))
        found += sorted(glob.glob(os.path.join(site, "scipy.libs", "libscipy_openblas-*")))
    except ImportError:
        pass
    if sys.platform == "win32":
        bin_ = os.path.join(sys.prefix, "Library", "bin")
        patterns = ["openblas.dll", "libopenblas.dll", "mkl_rt.*.dll", "mkl_rt.dll"]
    else:
        bin_ = os.path.join(sys.prefix, "lib")
        patterns = ["libopenblas.so*", "libmkl_rt.so*"]
    for pattern in patterns:
        found += sorted(glob.glob(os.path.join(bin_, pattern)))
    return found


def _load() -> None:
    if blas_library() != _PORTABLE:
        return  # linked at build time, or loaded before
    for var in ("CORVID_BLAS", "SPARTA_BLAS"):
        explicit = os.environ.get(var)
        if explicit:
            try:
                _load_blas(explicit)
                return
            except RuntimeError as e:
                warnings.warn(f"{var}: {e}; looking for another BLAS", RuntimeWarning, stacklevel=3)
    for find in (_scipy_openblas32, _bundled):
        for path in find():
            try:
                _load_blas(path)
                return
            except RuntimeError:
                continue
    warnings.warn("CORVID found no BLAS for its coarse factorization and uses SPARTA's portable loops, "
                  "which are several times slower. Install scipy-openblas32 or set CORVID_BLAS to an "
                  "OpenBLAS or MKL library.", RuntimeWarning, stacklevel=3)


if sys.platform != "darwin":
    _load()
