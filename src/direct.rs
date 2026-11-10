//! Sparse direct solves inside the preconditioner (the surface block and the
//! coarsest level): SPARTA's LDL^T on the CPU in double precision. Both
//! matrices are complex symmetric, so only the lower triangle is passed.

use crate::precond::{tick, T_B_ANALYSIS, T_B_NUMERIC};
use num_complex::Complex64 as C;
use sparta::{Analysis, Config, Solver};
use std::sync::Mutex;
use std::time::Instant;

pub struct Direct {
    solver: Mutex<CpuSolver>,
    pub factor_nnz: usize,
}

/// A SPARTA solver on its CPU backend. sparta::Solver is not Send because its GPU backend holds a
/// Metal context pointer; the CPU backend (`force_cpu`, the only one `Direct` builds) is plain Rust
/// data, so moving it to another thread is sound. This lets a solver's work run on its own rayon pool.
struct CpuSolver(Solver);
unsafe impl Send for CpuSolver {}

impl Direct {
    /// `lower`: entries (i, j, a) with i >= j of an n x n complex symmetric matrix.
    pub fn new(n: usize, mut lower: Vec<(usize, usize, C)>) -> Result<Self, String> {
        lower.sort_unstable_by_key(|&(i, j, _)| (j, i));
        let mut ptr = vec![0usize; n + 1];
        for &(_, j, _) in &lower {
            ptr[j + 1] += 1;
        }
        for j in 0..n {
            ptr[j + 1] += ptr[j];
        }
        let rows: Vec<usize> = lower.iter().map(|&(i, _, _)| i).collect();
        let vals: Vec<C> = lower.iter().map(|&(_, _, v)| v).collect();
        Self::from_csc(n, &ptr, &rows, &vals)
    }

    /// The lower triangle as compressed columns (rows ascending within each column).
    pub fn from_csc(n: usize, colptr: &[usize], rows: &[usize], vals: &[C]) -> Result<Self, String> {
        let config = Config { force_cpu: true, ..Config::default() };
        let t = Instant::now();
        let plan = Analysis::ldlt(n, colptr, rows, &config).map_err(|e| format!("{e:?}"))?;
        tick(&T_B_ANALYSIS, t);
        let factor_nnz = plan.factor_nonzeros();
        let solver = Solver::new(plan).map_err(|e| format!("{e:?}"))?;
        let d = Direct { solver: Mutex::new(CpuSolver(solver)), factor_nnz };
        d.refactor(vals)?;
        Ok(d)
    }

    /// New values on the analyzed pattern (same order as at construction): SPARTA's analysis
    /// (ordering and symbolic factorization) is reused.
    pub fn refactor(&self, vals: &[C]) -> Result<(), String> {
        let t = Instant::now();
        self.solver.lock().unwrap().0.factor_values(vals).map_err(|e| format!("{e:?}"))?;
        tick(&T_B_NUMERIC, t);
        Ok(())
    }

    pub fn solve(&self, b: &[C]) -> Vec<C> {
        self.solve_many(b, 1)
    }

    /// `nrhs` right-hand sides stored column after column; SPARTA solves the columns in parallel.
    pub fn solve_many(&self, b: &[C], nrhs: usize) -> Vec<C> {
        self.solver.lock().unwrap().0.solve_values(b, nrhs).expect("SPARTA solve")
    }
}
