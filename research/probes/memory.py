"""RAM of the Rust solve phase on model2 p1 (coarse = SPARTA LU of the Whitney level)."""
import sys, os, gc, resource
import numpy as np, scipy.sparse as sp
sys.argv = ["x", "model2_abc2", "1"]
src = open("research/probes/coarse_gs.py").read().split("variants = {}")[0]
exec(src)
from corvid._kernels import PSolver

def rss():  # current resident set, MB
    return int(os.popen(f"ps -o rss= -p {os.getpid()}").read()) / 1024

def t(A, dtype):
    A = sp.csr_matrix(A); A.sort_indices()
    return (A.indptr.astype(np.int32), A.indices.astype(np.int32), np.ascontiguousarray(A.data, dtype=dtype), A.shape[0], A.shape[1])

C, R_ = np.complex128, np.float64
K, b, pc = pp.build(rec); sm = pc.smoother
inj = pc.R.tocsc().indices.astype(np.int64)
S = pp.SurfaceBlockJacobi(pc.A, "lu").S.astype(np.int64)
args = (t(K, C), t(pc.A, C), t(sm.G, R_), t(sm.Gt, R_), t(sm.SG.A, C), S, inj, [], t(P1, C))
gc.collect()
m0 = rss()
rs = PSolver(*args, order=4, cycles=2, coarse_its=2)
del args; gc.collect()
m1 = rss()
x, its, rr = rs.solve(b.astype(C), "cocg", 1e-6, 200, 600)
del x; gc.collect(); m2 = rss()
x, its2, rr = rs.solve(b.astype(C), "gmres", 1e-6, 200, 600)
del x; gc.collect(); m3 = rss()
n = K.shape[0]
print(f"n = {n} unknowns, K nnz = {K.nnz}")
print(f"RSS before solver {m0:.0f} MB | after building it +{m1 - m0:.0f} MB | after COCG ({its} its) +{m2 - m1:.0f} MB | after GMRES ({its2} its) +{m3 - m2:.0f} MB")
print(f"peak RSS of the whole process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} MB (includes Python, scipy copies, helix setup)")

# component sizes held by the Rust solver
def split_mb(A):
    A = sp.csr_matrix(A); ni = (A.data.imag != 0).sum()
    return (A.nnz * 12 + (A.shape[0] + 1) * 8 + ni * 12 + (A.shape[0] + 1) * 8) / 1e6
def cx_mb(A): return (A.nnz * 20 + (A.shape[0] + 1) * 8) / 1e6
def re_mb(A): return (A.nnz * 12 + (A.shape[0] + 1) * 8) / 1e6
print("Rust-held data (computed):")
print(f"  K (system, real + sparse imag)      {split_mb(K):6.1f} MB")
print(f"  P (fine shifted, real + sparse imag) {split_mb(pc.A):6.1f} MB")
print(f"  G, G^T (fine gradient, real)        {re_mb(sm.G) + re_mb(sm.Gt):6.1f} MB")
print(f"  G^T P G (complex)                   {cx_mb(sm.SG.A):6.1f} MB")
print(f"  Whitney LDL^T factor nnz(L)=1.53M   {1.53e6 * 16 / 1e6:6.1f} MB (+ SPARTA's kept copy of the lower triangle {(P1.nnz / 2 + P1.shape[0] / 2) * 16 / 1e6:.1f} MB)")
print(f"  COCG vectors (~6 x n)               {6 * n * 16 / 1e6:6.1f} MB;  GMRES basis ({its2 + 1} x n)  {(its2 + 1) * n * 16 / 1e6:6.1f} MB")
from corvid._kernels import bench_direct
nnzL, tf, ts = bench_direct(t(K, C), 5)
print(f"Direct SPARTA CPU LDL^T of the full K: nnz(L) = {nnzL/1e6:.1f}M -> {nnzL*16/1e6:.0f} MB factor, factor {tf:.2f}s, solve {ts:.1f} ms")
