//! CORVID's compiled solve phase: threaded sparse kernels, the preconditioner
//! and the outer Krylov solvers. CSR with i32 indices, f64 or complex128.

mod csr;
mod direct;
mod krylov;
mod patch;
mod precond;
mod research;
mod vecops;

use csr::Csr;
use numpy::{IntoPyArray, PyArray1};
use precond::{Corvid, DistRelax, Precond, Preconditioner, TwoLevel};

use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::{AddAssign, Mul, Sub};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const ROWS_PER_TASK: usize = 512;

// ---- thread pools -------------------------------------------------------------------------------
// Every kernel, the vertex-block factorization and SPARTA's LDL^T run on the rayon pool that is
// current when they are called. A solver runs all of its work inside the pool of its own thread
// count (`run`), so different solvers in one process can use different counts and the count can be
// changed at any time. The global pool (sized once at import) serves the default count; pools for
// other counts are built on first use and kept for reuse.

/// Size of rayon's global pool, fixed at import.
static GLOBAL_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Thread count of solvers and kernels that do not name one; set_num_threads changes it.
static DEFAULT_THREADS: AtomicUsize = AtomicUsize::new(0);
static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();

/// The thread count a `threads` argument stands for (None: the module default).
fn resolve_threads(threads: Option<usize>) -> PyResult<usize> {
    match threads {
        Some(0) => Err(pyo3::exceptions::PyValueError::new_err("threads must be at least 1")),
        Some(n) => Ok(n),
        None => Ok(DEFAULT_THREADS.load(Ordering::Relaxed)),
    }
}

/// Run `f` on a pool of `threads` threads.
fn run<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    if threads == GLOBAL_THREADS.load(Ordering::Relaxed) || threads == rayon::current_num_threads() {
        return f();
    }
    let pool = {
        let mut pools = POOLS.get_or_init(Default::default).lock().unwrap();
        pools
            .entry(threads)
            .or_insert_with(|| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .thread_name(|i| format!("corvid-{i}"))
                        .build()
                        .expect("CORVID thread pool"),
                )
            })
            .clone()
    };
    pool.install(f)
}

/// Set the thread count used by solvers created without `threads` and by the free kernels.
#[pyfunction]
fn set_num_threads(threads: usize) -> PyResult<()> {
    DEFAULT_THREADS.store(resolve_threads(Some(threads))?, Ordering::Relaxed);
    Ok(())
}

/// The thread count used by solvers created without `threads` and by the free kernels.
#[pyfunction]
fn get_num_threads() -> usize {
    DEFAULT_THREADS.load(Ordering::Relaxed)
}

/// y = A x or y = b - A x; matrix entries M, vector entries T (M = f64 with
/// T = Complex64 applies a real matrix to a complex vector in one pass).
fn spmv<M, T>(ip: &[i32], ix: &[i32], a: &[M], x: &[T], y: &mut [T], b: Option<&[T]>)
where
    M: Copy + Send + Sync,
    T: Copy + Send + Sync + Default + AddAssign + Sub<Output = T> + Mul<M, Output = T>,
{
    y.par_chunks_mut(ROWS_PER_TASK).enumerate().for_each(|(c, ys)| {
        let r0 = c * ROWS_PER_TASK;
        for (k, yi) in ys.iter_mut().enumerate() {
            let i = r0 + k;
            let mut s = T::default();
            for p in ip[i] as usize..ip[i + 1] as usize {
                s += x[ix[p] as usize] * a[p];
            }
            *yi = match b {
                Some(b) => b[i] - s,
                None => s,
            };
        }
    });
}

/// One Gauss-Seidel sweep on A x = b, in place. Rows are split into `nblocks`
/// contiguous blocks swept in parallel; inside a block the sweep uses the newest
/// values, across blocks the values from the start of the sweep (hybrid GS).
/// nblocks = 1 is the exact sequential sweep. Rows with a zero diagonal are skipped.
fn gs<T>(ip: &[i32], ix: &[i32], a: &[T], x: &mut [T], b: &[T], forward: bool, nblocks: usize)
where
    T: Copy + Send + Sync + Default + PartialEq + AddAssign + Sub<Output = T> + Mul<T, Output = T>
        + std::ops::Div<Output = T>,
{
    let n = x.len();
    let nb = nblocks.max(1).min(n.max(1));
    let size = (n + nb - 1) / nb;
    let x0: Vec<T> = if nb > 1 { x.to_vec() } else { Vec::new() };
    let x0 = &x0;
    let sweep = |r0: usize, xs: &mut [T]| {
        let len = xs.len();
        let rows: Box<dyn Iterator<Item = usize>> =
            if forward { Box::new(0..len) } else { Box::new((0..len).rev()) };
        for k in rows {
            let i = r0 + k;
            let mut s = b[i];
            let mut d = T::default();
            for p in ip[i] as usize..ip[i + 1] as usize {
                let j = ix[p] as usize;
                if j == i {
                    d = a[p];
                } else {
                    let xj = if j >= r0 && j < r0 + len { xs[j - r0] } else { x0[j] };
                    s = s - a[p] * xj;
                }
            }
            if d != T::default() {
                xs[k] = s / d;
            }
        }
    };
    if nb == 1 {
        sweep(0, x);
    } else {
        x.par_chunks_mut(size).enumerate().for_each(|(c, xs)| sweep(c * size, xs));
    }
}

macro_rules! export_gs {
    ($name:ident, $t:ty) => {
        /// In-place Gauss-Seidel sweeps.
        #[pyfunction]
        fn $name(
            ip: PyReadonlyArray1<i32>, ix: PyReadonlyArray1<i32>, a: PyReadonlyArray1<$t>,
            mut x: PyReadwriteArray1<$t>, b: PyReadonlyArray1<$t>, forward: bool, sweeps: usize,
            nblocks: usize,
        ) -> PyResult<()> {
            let (ip, ix, a, b) = (ip.as_slice()?, ix.as_slice()?, a.as_slice()?, b.as_slice()?);
            let x = x.as_slice_mut()?;
            run(DEFAULT_THREADS.load(Ordering::Relaxed), || {
                for _ in 0..sweeps {
                    gs(ip, ix, a, x, b, forward, nblocks);
                }
            });
            Ok(())
        }
    };
}

export_gs!(gs_f64, f64);
export_gs!(gs_c128, Complex64);

macro_rules! export {
    ($name:ident, $resname:ident, $m:ty, $t:ty) => {
        /// y = A x
        #[pyfunction]
        fn $name(
            ip: PyReadonlyArray1<i32>, ix: PyReadonlyArray1<i32>, a: PyReadonlyArray1<$m>,
            x: PyReadonlyArray1<$t>, mut y: PyReadwriteArray1<$t>,
        ) -> PyResult<()> {
            let (ip, ix, a, x) = (ip.as_slice()?, ix.as_slice()?, a.as_slice()?, x.as_slice()?);
            let y = y.as_slice_mut()?;
            run(DEFAULT_THREADS.load(Ordering::Relaxed), || spmv(ip, ix, a, x, y, None));
            Ok(())
        }
        /// y = b - A x
        #[pyfunction]
        fn $resname(
            ip: PyReadonlyArray1<i32>, ix: PyReadonlyArray1<i32>, a: PyReadonlyArray1<$m>,
            x: PyReadonlyArray1<$t>, b: PyReadonlyArray1<$t>, mut y: PyReadwriteArray1<$t>,
        ) -> PyResult<()> {
            let (ip, ix, a, x, b) = (ip.as_slice()?, ix.as_slice()?, a.as_slice()?, x.as_slice()?, b.as_slice()?);
            let y = y.as_slice_mut()?;
            run(DEFAULT_THREADS.load(Ordering::Relaxed), || spmv(ip, ix, a, x, y, Some(b)));
            Ok(())
        }
    };
}

export!(spmv_f64, residual_f64, f64, f64);
export!(spmv_c128, residual_c128, Complex64, Complex64);
export!(spmv_rc, residual_rc, f64, Complex64);

type CsrC<'py> = (PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, Complex64>, usize, usize);
type CsrR<'py> = (PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, f64>, usize, usize);

fn csr_c(m: &CsrC) -> PyResult<Csr<Complex64>> {
    Ok(Csr::new(m.3, m.4, m.0.as_slice()?, m.1.as_slice()?, m.2.as_slice()?))
}

fn csr_r(m: &CsrR) -> PyResult<Csr<f64>> {
    Ok(Csr::new(m.3, m.4, m.0.as_slice()?, m.1.as_slice()?, m.2.as_slice()?))
}

fn err(e: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e)
}

/// The Palace-style p-multigrid with a CORVID coarse V-cycle, plus the system
/// matrix, ready to solve. Matrices come as (indptr, indices, data, nrows, ncols).
#[pyclass(unsendable)]
struct PSolver {
    k: csr::Split,
    pc: Preconditioner,
    threads: usize,
}

#[pymethods]
impl PSolver {
    #[new]
    #[pyo3(signature = (k, p, g, gt, ag, surface, inj, levels, coarsest, order=4, cycles=2, coarse_its=2, colored=true, color_min_rows=2000, patch_ptr=None, patch_idx=None, use_grad=true, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k: CsrC, p: CsrC, g: CsrR, gt: CsrR, ag: CsrC, surface: PyReadonlyArray1<i64>,
        inj: PyReadonlyArray1<i64>, levels: Vec<(CsrC, CsrR, CsrR, CsrR, CsrR, CsrC)>, coarsest: CsrC,
        order: usize, cycles: usize, coarse_its: usize, colored: bool, color_min_rows: usize,
        patch_ptr: Option<PyReadonlyArray1<i64>>, patch_idx: Option<PyReadonlyArray1<i32>>, use_grad: bool,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let threads = resolve_threads(threads)?;
        let surface: Vec<usize> = surface.as_slice()?.iter().map(|&v| v as usize).collect();
        let patches: Option<Vec<Vec<u32>>> = match (&patch_ptr, &patch_idx) {
            (Some(pp), Some(pi)) => {
                let (pp, pi) = (pp.as_slice()?, pi.as_slice()?);
                Some((0..pp.len() - 1).map(|v| pi[pp[v] as usize..pp[v + 1] as usize].iter().map(|&d| d as u32).collect()).collect())
            }
            _ => None,
        };
        let (p, g, gt, ag) = (csr_c(&p)?, csr_r(&g)?, csr_r(&gt)?, csr_c(&ag)?);
        let mut lv = Vec::new();
        for (a, pp, pt, gg, ggt, agg) in &levels {
            lv.push((csr_c(a)?, csr_r(pp)?, csr_r(pt)?, csr_r(gg)?, csr_r(ggt)?, csr_c(agg)?));
        }
        let coarsest = csr_c(&coarsest)?;
        let inj: Vec<usize> = inj.as_slice()?.iter().map(|&v| v as usize).collect();
        let k = csr::Split::from(&csr_c(&k)?);
        let pc = run(threads, move || -> Result<Preconditioner, String> {
            let fine = match patches {
                Some(patches) => DistRelax::new_schwarz(p, g, gt, ag, order, patches, use_grad)?,
                None => DistRelax::new(p, g, gt, ag, order, Some(surface))?,
            };
            let coarse = Corvid::new(lv, &coarsest, coarse_its, colored, color_min_rows)?;
            Ok(Preconditioner::new(fine, inj, coarse, cycles))
        })
        .map_err(err)?;
        Ok(PSolver { k, pc, threads })
    }

    /// z = M r (one preconditioner application)
    fn apply<'py>(&self, py: Python<'py>, r: PyReadonlyArray1<'py, Complex64>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let r = r.as_slice()?;
        Ok(run(self.threads, || self.pc.apply(r)).into_pyarray_bound(py))
    }

    /// (x, iterations, true relative residual)
    #[pyo3(signature = (b, method="gmres", rtol=1e-6, restart=200, maxiter=1000))]
    fn solve<'py>(
        &self, py: Python<'py>, b: PyReadonlyArray1<'py, Complex64>, method: &str, rtol: f64, restart: usize,
        maxiter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Complex64>>, usize, f64)> {
        let b = b.as_slice()?;
        let (x, its, rr) = match method {
            "gmres" => run(self.threads, || krylov::gmres(&self.k, b, &self.pc, rtol, restart, maxiter)),
            "cocg" => run(self.threads, || krylov::cocg(&self.k, b, &self.pc, rtol, maxiter)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("method: gmres or cocg")),
        };
        Ok((x.into_pyarray_bound(py), its, rr))
    }
}

/// Per row of a complex CSR matrix, in one threaded pass: (entries whose imaginary part exceeds
/// `rel_tol` of their magnitude, sum |Im a_ij|, sum |a_ij|). Feeds corvid.pml_rows / pml_stretch.
#[pyfunction]
#[pyo3(signature = (indptr, data, rel_tol=1e-8, threads=None))]
#[allow(clippy::type_complexity)]
fn row_stats<'py>(
    py: Python<'py>, indptr: PyReadonlyArray1<'py, i32>, data: PyReadonlyArray1<'py, Complex64>, rel_tol: f64,
    threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (ip, a) = (indptr.as_slice()?, data.as_slice()?);
    let n = ip.len().saturating_sub(1);
    let threads = resolve_threads(threads)?;
    let (mut cnt, mut sim, mut sab) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    run(threads, || {
        cnt.par_chunks_mut(ROWS_PER_TASK)
            .zip(sim.par_chunks_mut(ROWS_PER_TASK))
            .zip(sab.par_chunks_mut(ROWS_PER_TASK))
            .enumerate()
            .for_each(|(c, ((cn, si), sa))| {
                for k in 0..cn.len() {
                    let i = c * ROWS_PER_TASK + k;
                    let (mut nc, mut im, mut ab) = (0.0, 0.0, 0.0);
                    for v in &a[ip[i] as usize..ip[i + 1] as usize] {
                        let (vi, va) = (v.im.abs(), v.norm());
                        if vi > rel_tol * va {
                            nc += 1.0;
                        }
                        im += vi;
                        ab += va;
                    }
                    cn[k] = nc;
                    si[k] = im;
                    sa[k] = ab;
                }
            });
    });
    Ok((cnt.into_pyarray_bound(py), sim.into_pyarray_bound(py), sab.into_pyarray_bound(py)))
}

/// (factor nonzeros, factor seconds, solve milliseconds) of the SPARTA CPU
/// LDL^T on a complex symmetric matrix (lower triangle used).
#[pyfunction]
fn bench_direct(m: CsrC, reps: usize) -> PyResult<(usize, f64, f64)> {
    let a = csr_c(&m)?;
    let (n, lower) = a.lower(None);
    let t = std::time::Instant::now();
    let d = direct::Direct::new(n, lower).map_err(err)?;
    let tf = t.elapsed().as_secs_f64();
    let b = vec![Complex64::new(1.0, 0.5); n];
    let _ = d.solve(&b);
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let _ = d.solve(&b);
    }
    Ok((d.factor_nnz, tf, t.elapsed().as_secs_f64() / reps as f64 * 1e3))
}

/// Seconds per preconditioner component since the last call (then reset).
#[pyfunction]
fn timers() -> Vec<(&'static str, f64)> {
    precond::timers()
}

/// CORVID's solver for K x = b: vertex-patch Schwarz Chebyshev smoothing with an
/// exact coarse solve on the Whitney rows (see corvid/solver.py for the inputs).
#[pyclass(unsendable)]
struct TwoLevelSolver {
    pc: TwoLevel,
    threads: usize,
}

#[pymethods]
impl TwoLevelSolver {
    #[new]
    #[pyo3(signature = (k, patch_ptr, patch_idx, inj, order=2, sparse_blocks=false, threads=None))]
    fn new(
        k: CsrC, patch_ptr: PyReadonlyArray1<i64>, patch_idx: PyReadonlyArray1<i32>, inj: PyReadonlyArray1<i64>,
        order: usize, sparse_blocks: bool, threads: Option<usize>,
    ) -> PyResult<Self> {
        let threads = resolve_threads(threads)?;
        let (pp, pi) = (patch_ptr.as_slice()?, patch_idx.as_slice()?);
        let patches = (0..pp.len().saturating_sub(1))
            .map(|v| pi[pp[v] as usize..pp[v + 1] as usize].iter().map(|&d| d as u32).collect())
            .collect();
        let inj = inj.as_slice()?.iter().map(|&v| v as usize).collect();
        let vals = k.2.as_slice()?;
        let t = std::time::Instant::now();
        let ks = csr::Split::from_parts(k.3, k.4, k.0.as_slice()?, k.1.as_slice()?, vals);
        precond::tick(&precond::T_B_SPLIT, t);
        let pc = run(threads, move || TwoLevel::new(ks, vals, patches, inj, order, sparse_blocks)).map_err(err)?;
        Ok(TwoLevelSolver { pc, threads })
    }

    /// New values of K on the same sparsity pattern (the caller checks the pattern): reuses the
    /// coarse analysis, the vertex-block orderings and the spectral estimate's start vector.
    fn refactor(&mut self, k: CsrC) -> PyResult<()> {
        let vals = k.2.as_slice()?;
        let t = std::time::Instant::now();
        let ks = csr::Split::from_parts(k.3, k.4, k.0.as_slice()?, k.1.as_slice()?, vals);
        precond::tick(&precond::T_B_SPLIT, t);
        let pc = &mut self.pc;
        run(self.threads, move || pc.refactor(ks, vals)).map_err(err)
    }

    /// New values of K on the same pattern for the Krylov products and the smoother, keeping the
    /// block and coarse factors of the last factorization (a nearby frequency).
    fn update_operator(&mut self, k: CsrC) -> PyResult<()> {
        let ks = csr::Split::from_parts(k.3, k.4, k.0.as_slice()?, k.1.as_slice()?, k.2.as_slice()?);
        self.pc.update_operator(ks);
        Ok(())
    }

    /// K x for every column of `x` (a list of (n,) arrays), up to four columns per pass over K.
    fn mul_columns<'py>(&self, py: Python<'py>, x: Vec<PyReadonlyArray1<'py, Complex64>>) -> PyResult<Vec<Bound<'py, PyArray1<Complex64>>>> {
        let xs: Vec<&[Complex64]> = x.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
        let n = krylov::BlockOp::n(&self.pc);
        if xs.iter().any(|c| c.len() != n) {
            return Err(pyo3::exceptions::PyValueError::new_err("every column must have the system size"));
        }
        let out = run(self.threads, || {
            let mut out = Vec::with_capacity(xs.len());
            for g in (0..xs.len()).step_by(precond::MAX_COLS) {
                let e = (g + precond::MAX_COLS).min(xs.len());
                let k = e - g;
                let mut y = vec![Complex64::default(); n * k];
                krylov::BlockOp::mul_k(&self.pc, &krylov::pack(&xs[g..e]), &mut y, k);
                for c in 0..k {
                    let mut col = vec![Complex64::default(); n];
                    krylov::unpack(&y, k, c, &mut col);
                    out.push(col);
                }
            }
            out
        });
        Ok(out.into_iter().map(|c| c.into_pyarray_bound(py)).collect())
    }

    /// Power iterations of the last spectral estimate.
    #[getter]
    fn lmax_iterations(&self) -> usize {
        self.pc.lmax_iterations
    }

    /// Solve for every column of `b` (a list of (n,) arrays), optionally from starting guesses `x0`.
    /// COCG runs up to four columns in lockstep (one operator / preconditioner application per
    /// iteration for all of them); GMRES solves column by column. Returns (x per column,
    /// iterations per column, true relative residual per column).
    #[pyo3(signature = (b, x0=None, method="cocg", rtol=1e-6, restart=100, maxiter=500))]
    #[allow(clippy::type_complexity)]
    fn solve_columns<'py>(
        &self, py: Python<'py>, b: Vec<PyReadonlyArray1<'py, Complex64>>, x0: Option<Vec<PyReadonlyArray1<'py, Complex64>>>,
        method: &str, rtol: f64, restart: usize, maxiter: usize,
    ) -> PyResult<(Vec<Bound<'py, PyArray1<Complex64>>>, Vec<usize>, Vec<f64>)> {
        let bs: Vec<&[Complex64]> = b.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
        let xs: Option<Vec<&[Complex64]>> = match &x0 {
            Some(v) => Some(v.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?),
            None => None,
        };
        let n = krylov::BlockOp::n(&self.pc);
        if bs.iter().any(|c| c.len() != n) || xs.as_ref().is_some_and(|x| x.len() != bs.len() || x.iter().any(|c| c.len() != n)) {
            return Err(pyo3::exceptions::PyValueError::new_err("every column must have the system size"));
        }
        let (mut xo, mut its, mut res) = (Vec::new(), Vec::new(), Vec::new());
        match method {
            "cocg" => {
                for g in (0..bs.len()).step_by(precond::MAX_COLS) {
                    let e = (g + precond::MAX_COLS).min(bs.len());
                    let x0g = xs.as_ref().map(|x| &x[g..e]);
                    let (x, i, r) = run(self.threads, || krylov::cocg_block(&self.pc, &bs[g..e], x0g, rtol, maxiter));
                    xo.extend(x);
                    its.extend(i);
                    res.extend(r);
                }
            }
            "gmres" => {
                for (c, bc) in bs.iter().enumerate() {
                    let x0c = xs.as_ref().map(|x| x[c]);
                    let (x, i, r) = run(self.threads, || krylov::gmres_x0(&self.pc.k, bc, x0c, &self.pc, rtol, restart, maxiter));
                    xo.push(x);
                    its.push(i);
                    res.push(r);
                }
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("method: gmres or cocg")),
        }
        Ok((xo.into_iter().map(|x| x.into_pyarray_bound(py)).collect(), its, res))
    }

    /// Threads this solver runs on.
    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    /// Bytes held by the patch inverses / factors.
    #[getter]
    fn schwarz_bytes(&self) -> usize {
        self.pc.schwarz_bytes
    }

    /// Nonzeros of the exact level's LDL^T factor (SPARTA).
    #[getter]
    fn coarse_factor_nnz(&self) -> usize {
        self.pc.coarse_factor_nnz
    }

    /// Blocks stored as dense inverses although sparse storage was requested (tiny pivot).
    #[getter]
    fn dense_fallbacks(&self) -> usize {
        self.pc.dense_fallbacks
    }

    /// z = M r (one preconditioner application)
    fn apply<'py>(&self, py: Python<'py>, r: PyReadonlyArray1<'py, Complex64>) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let r = r.as_slice()?;
        Ok(run(self.threads, || self.pc.apply(r)).into_pyarray_bound(py))
    }

    /// (x, iterations, true relative residual)
    #[pyo3(signature = (b, method="gmres", rtol=1e-6, restart=100, maxiter=500))]
    fn solve<'py>(
        &self, py: Python<'py>, b: PyReadonlyArray1<'py, Complex64>, method: &str, rtol: f64, restart: usize,
        maxiter: usize,
    ) -> PyResult<(Bound<'py, PyArray1<Complex64>>, usize, f64)> {
        let b = b.as_slice()?;
        let (x, its, rr) = match method {
            "gmres" => run(self.threads, || krylov::gmres(&self.pc.k, b, &self.pc, rtol, restart, maxiter)),
            "cocg" => run(self.threads, || krylov::cocg(&self.pc.k, b, &self.pc, rtol, maxiter)),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("method: gmres or cocg")),
        };
        Ok((x.into_pyarray_bound(py), its, rr))
    }
}

#[pymodule]
fn _kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Default 8 threads: bandwidth-bound kernels are within ~8% of their best
    // there and it leaves the machine to the rest of the program. RAYON_NUM_THREADS overrides.
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(8);
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
    // outside any pool, current_num_threads() is the global pool's size
    let n = rayon::current_num_threads();
    GLOBAL_THREADS.store(n, Ordering::Relaxed);
    DEFAULT_THREADS.store(n, Ordering::Relaxed);
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(spmv_f64, m)?)?;
    m.add_function(wrap_pyfunction!(residual_f64, m)?)?;
    m.add_function(wrap_pyfunction!(spmv_c128, m)?)?;
    m.add_function(wrap_pyfunction!(residual_c128, m)?)?;
    m.add_function(wrap_pyfunction!(spmv_rc, m)?)?;
    m.add_function(wrap_pyfunction!(residual_rc, m)?)?;
    m.add_function(wrap_pyfunction!(gs_f64, m)?)?;
    m.add_function(wrap_pyfunction!(gs_c128, m)?)?;
    m.add_class::<PSolver>()?;
    m.add_class::<TwoLevelSolver>()?;
    m.add_class::<research::RxDirect>()?;
    m.add_class::<research::RxSchwarz>()?;
    m.add_class::<research::RxSweep>()?;
    m.add_function(wrap_pyfunction!(timers, m)?)?;
    m.add_function(wrap_pyfunction!(row_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bench_direct, m)?)?;
    Ok(())
}
