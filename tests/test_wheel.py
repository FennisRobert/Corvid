"""Smoke test of an installed CORVID (used by the wheel workflow): a second-order Nedelec waveguide
problem from EMerge, solved with GMRES and COCG, then as a two-point sweep on one Solver.

    python tests/test_wheel.py        (run it from outside the repository, so the installed package
                                       is imported rather than the source tree)
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

import corvid

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    print(f"corvid {corvid.__version__} from {os.path.dirname(corvid.__file__)}, threads {corvid.get_num_threads()}")
    f = np.load(os.path.join(HERE, "data", "model1_p1.npz"))
    csr = lambda k: sp.csr_matrix((f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]), shape=tuple(f[k + "_shape"]))
    k0 = float(f["k0"])
    sid = f["solve_ids"]
    top = corvid.Topology.nedelec2(f["mesh_edges"], f["mesh_tris"], sid, nodes=f["mesh_nodes"])
    E, B = csr("E"), csr("B")
    b = f["b"][sid]

    def system(k):
        A = (E - k ** 2 * B).tocsr()
        return A[sid][:, sid].tocsr()

    ok = True
    for method in ("gmres", "cocg"):
        K = system(k0)
        s = corvid.Solver(top, method=method, threads=2).factorize(K, k0=k0)
        x = s.solve(b)
        res = np.max(np.linalg.norm(K @ x - b, axis=0) / np.linalg.norm(b, axis=0))
        its = [i["iterations"] for i in s.info]
        print(f"{method}: iterations {its}, true residual {res:.1e}")
        ok &= res < 1e-5

    s = corvid.Solver(top, method="cocg", threads=2)
    for k in (k0, 1.02 * k0):
        K = system(k)
        s.factorize(K, k0=k)
        x = s.solve(b)
        res = np.max(np.linalg.norm(K @ x - b, axis=0) / np.linalg.norm(b, axis=0))
        state = "kept factors" if s.reused else "refactored" if s.refactored else "factorized"
        print(f"sweep k0={k:.2f}: {state}, iterations {[i['iterations'] for i in s.info]}, true residual {res:.1e}")
        ok &= res < 1e-5

    print("passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
