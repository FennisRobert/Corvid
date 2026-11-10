"""Sweep features of corvid.Solver on the EMerge datasets: lockstep columns, refactor in place, warm
starts (minimum-residual projection on earlier solutions), kept factors and their safety net.

    python tests/test_sweep.py          (run from the repo root, with the package importable)

Every check compares against an independently built solver; all solves must reach rtol.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_api import emerge_inputs, problem_path, available  # noqa: E402
import corvid  # noqa: E402

RTOL = 1e-6
THREADS = 8


def k0_of(model, pid):
    return float(np.load(problem_path(model, pid))["k0"])


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


def lockstep_equals_single(model, pid):
    E, F, sid, K, b = emerge_inputs(model, pid)
    top = corvid.Topology.nedelec2(E, F, sid)
    s = corvid.Solver(top, method="cocg", threads=THREADS, warm_start=False).factorize(K)
    B = b.reshape(b.shape[0], -1)
    if B.shape[1] < 2:
        B = np.column_stack([B[:, 0], 1j * B[:, 0] + 0.5 * np.roll(B[:, 0], 7)])
    Xb = s.solve(B)
    its_b = [i["iterations"] for i in s.info]
    its_s, Xs = [], []
    for j in range(B.shape[1]):
        Xs.append(s.solve(B[:, j]))
        its_s.append(s.info[0]["iterations"])
    diff = max(np.linalg.norm(Xb[:, j] - Xs[j]) / np.linalg.norm(Xs[j]) for j in range(B.shape[1]))
    check(its_b == its_s and diff < 1e-10, f"{model} p{pid}: lockstep {its_b} = one by one {its_s} (solution diff {diff:.0e})")


def refactor_equals_fresh(model, p1, p2):
    E, F, sid, K1, _ = emerge_inputs(model, p1)
    _, _, _, K2, b2 = emerge_inputs(model, p2)
    top = corvid.Topology.nedelec2(E, F, sid)
    s = corvid.Solver(top, method="cocg", threads=THREADS, warm_start=False, refactor_tol=0).factorize(K1)
    s.factorize(K2)
    x = s.solve(b2)
    its_r = [i["iterations"] for i in s.info]
    f = corvid.Solver(top, method="cocg", threads=THREADS, warm_start=False).factorize(K2)
    y = f.solve(b2)
    its_f = [i["iterations"] for i in f.info]
    diff = np.linalg.norm(x - y) / np.linalg.norm(y)
    # the refactor keeps max(old, new) spectral bound, so iterations may differ by a little
    check(s.refactored and abs(sum(its_r) - sum(its_f)) <= 2 and diff < 1e-5 and max(i["residual"] for i in s.info) < RTOL,
          f"{model} p{p1}->p{p2}: refactor {its_r} vs fresh {its_f} (solution diff {diff:.0e})")


def sweep(model, pids, refactor_tol):
    E, F, sid, _, _ = emerge_inputs(model, pids[0])
    top = corvid.Topology.nedelec2(E, F, sid)
    s = corvid.Solver(top, method="cocg", threads=THREADS, refactor_tol=refactor_tol)
    cold = corvid.Solver(top, method="cocg", threads=THREADS, warm_start=False, refactor_tol=0)
    tot_w = tot_c = 0
    for pid in pids:
        _, _, _, K, b = emerge_inputs(model, pid)
        k0 = k0_of(model, pid)
        s.factorize(K, k0=k0)
        x = s.solve(b)
        cold.factorize(K, k0=k0)
        cold.solve(b)
        res = np.linalg.norm(K @ x - b.reshape(x.shape), axis=0) / np.linalg.norm(b.reshape(x.shape), axis=0)
        tot_w += sum(i["iterations"] for i in s.info)
        tot_c += sum(i["iterations"] for i in cold.info)
        state = "kept" if s.reused else "refactored" if s.refactored else "new"
        check(np.max(res) < 1.5 * RTOL, f"{model} p{pid} ({state}): its {[i['iterations'] for i in s.info]} "
              f"(cold {[i['iterations'] for i in cold.info]}), true residual {np.max(res):.1e}")
    check(tot_w <= tot_c, f"{model}: warm sweep {tot_w} iterations <= cold {tot_c}")


def safety_net(model, p1, p2):
    """Factors kept far outside their range: the capped attempt fails, the solver refactors and converges."""
    E, F, sid, K1, b1 = emerge_inputs(model, p1)
    _, _, _, K2, b2 = emerge_inputs(model, p2)
    top = corvid.Topology.nedelec2(E, F, sid)
    s = corvid.Solver(top, method="cocg", threads=THREADS, refactor_tol=1e9)   # always keep factors
    s.factorize(K1, k0=k0_of(model, p1)).solve(b1)
    s.factorize(K2, k0=k0_of(model, p2))
    kept = s.reused
    x = s.solve(b2)
    res = np.max(np.linalg.norm(K2 @ x - b2.reshape(x.shape), axis=0) / np.linalg.norm(b2.reshape(x.shape), axis=0))
    check(kept and res < 1.5 * RTOL, f"{model} p{p1}->p{p2} kept factors: its {[i['iterations'] for i in s.info]}, "
          f"refactored on the way: {s.refactored}, true residual {res:.1e}")


if __name__ == "__main__":
    print(f"corvid {corvid.__version__}")
    if not available("model2_abc2", 1):
        print("skipped: the sweep tests need the EMerge datasets in data/aegis/")
        sys.exit(0)
    lockstep_equals_single("model1", 1)
    lockstep_equals_single("model3", 9)
    refactor_equals_fresh("model2_abc2", 1, 2)
    refactor_equals_fresh("model3", 1, 2)
    sweep("model1", list(range(1, 12)), refactor_tol=0.2)
    sweep("model2_abc2", list(range(1, 8)), refactor_tol=0.2)
    sweep("model3", list(range(1, 12)), refactor_tol=0.2)
    safety_net("model3", 1, 11)
    print("FAILED" if check.failed else "all passed")
    sys.exit(1 if check.failed else 0)
