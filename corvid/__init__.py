"""CORVID: Coarse Operators by Recursive Volume Interpolation of De Rham complexes.

    import corvid
    top = corvid.Topology.nedelec2(edges, tris, solve_ids)   # once per mesh
    sol = corvid.Solver(top).factorize(K)                     # once per frequency
    x = sol.solve(b)                                          # per right-hand side

Threads: corvid.Solver(top, threads=n) per solver, or corvid.set_num_threads(n) for the default.
"""
from .solver import Solver, Topology, pml_rows
from ._kernels import blas_library, get_num_threads, set_num_threads
from . import _blas  # loads the coarse factorization's BLAS off Apple; see _blas.py

__version__ = "0.5.1"
__all__ = ["Solver", "Topology", "pml_rows", "blas_library", "get_num_threads", "set_num_threads", "__version__"]
