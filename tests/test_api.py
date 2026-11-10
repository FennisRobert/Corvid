"""End-to-end test of the public API with exactly the data EMerge would pass.
usage: python tests/test_api.py [model pid ...]   (defaults: one problem per available model)

The full EMerge datasets live in data/aegis/ (not in the repository). Without them the test runs on
the model1 problem shipped in tests/data/."""
import os, sys, time
import numpy as np, scipy.sparse as sp
import corvid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "aegis")
FIXTURES = {("model1", 1): os.path.join(HERE, "data", "model1_p1.npz")}


def problem_path(model, pid):
    p = os.path.join(DATA, model, f"problem_{pid}.npz")
    return p if os.path.exists(p) else FIXTURES.get((model, pid), p)


def available(model, pid):
    return os.path.exists(problem_path(model, pid))


def emerge_inputs(model, pid):
    """What EMerge has at solve time: mesh edges/tris, solve_ids, the full
    matrix A = E - k0^2 B and b; the solver sees A[solve_ids][:, solve_ids]."""
    f = np.load(problem_path(model, pid))
    csr = lambda k: sp.csr_matrix((f[k + "_data"], f[k + "_indices"], f[k + "_indptr"]), shape=tuple(f[k + "_shape"]))
    A = (csr("E") - f["k0"] ** 2 * csr("B")).tocsr()
    sid = f["solve_ids"]
    return f["mesh_edges"], f["mesh_tris"], sid, A[sid][:, sid].tocsr(), f["b"][sid]


def main():
    cases = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)] or \
            [c for c in [("model1", 1), ("model2_abc2", 1), ("model3", 9)] if available(*c)]
    print(f"corvid {corvid.__version__}")
    for model, pid in cases:
        edges, tris, sid, K, b = emerge_inputs(model, pid)
        t0 = time.time(); top = corvid.Topology.nedelec2(edges, tris, sid); t_top = time.time() - t0
        for method in ("gmres", "cocg"):
            t0 = time.time(); s = corvid.Solver(top, method=method).factorize(K); t_fac = time.time() - t0
            t0 = time.time(); x = s.solve(b); t_sol = time.time() - t0
            rr = np.linalg.norm(K @ x - b) / np.linalg.norm(b)
            its = [i["iterations"] for i in s.info]
            print(f"{model} p{pid} n={K.shape[0]} rhs={b.shape[1] if b.ndim > 1 else 1}  {method}: its {its} "
                  f"residual {rr:.1e}  topology {t_top:.2f}s factorize {t_fac:.2f}s solve {t_sol:.2f}s  "
                  f"blocks {s.memory_bytes / 1e6:.0f} MB")
            assert rr < 1e-5, "did not converge"


if __name__ == "__main__":
    main()
