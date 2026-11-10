//! Research building blocks exposed to Python (corvid/research experiments only;
//! the production solver does not use them). All double precision, SPARTA CPU.
//!
//!   RxDirect   sparse LDL^T of a complex symmetric matrix
//!   RxSchwarz  additive Schwarz over index blocks with dense inverses
//!   RxSweep    multiplicative (block Gauss-Seidel) sweep over index blocks,
//!              each block factored with SPARTA

use crate::csr::{CMat, Split};
use crate::direct::Direct;
use crate::patch::Schwarz;
use num_complex::Complex64 as C;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::time::Instant;

type CsrC<'py> = (PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, i32>, PyReadonlyArray1<'py, C>, usize, usize);

fn split(m: &CsrC) -> PyResult<Split> {
    Ok(Split::from_parts(m.3, m.4, m.0.as_slice()?, m.1.as_slice()?, m.2.as_slice()?))
}

fn err(e: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e)
}

fn blocks(ptr: &[i64], idx: &[i64]) -> Vec<Vec<usize>> {
    (0..ptr.len().saturating_sub(1)).map(|b| idx[ptr[b] as usize..ptr[b + 1] as usize].iter().map(|&i| i as usize).collect()).collect()
}

#[pyclass(unsendable)]
pub struct RxDirect {
    d: Direct,
    #[pyo3(get)]
    factor_nnz: usize,
    #[pyo3(get)]
    factor_seconds: f64,
}

#[pymethods]
impl RxDirect {
    #[new]
    fn new(m: CsrC) -> PyResult<Self> {
        let a = split(&m)?;
        let rows: Vec<usize> = (0..a.nrows()).collect();
        let t = Instant::now();
        let d = Direct::new(rows.len(), a.lower_on(&rows)).map_err(err)?;
        let factor_seconds = t.elapsed().as_secs_f64();
        Ok(RxDirect { factor_nnz: d.factor_nnz, d, factor_seconds })
    }

    fn solve<'py>(&self, py: Python<'py>, b: PyReadonlyArray1<'py, C>) -> PyResult<Bound<'py, PyArray1<C>>> {
        Ok(self.d.solve(b.as_slice()?).into_pyarray_bound(py))
    }
}

#[pyclass(unsendable)]
pub struct RxSchwarz {
    s: Schwarz,
    #[pyo3(get)]
    bytes: usize,
}

#[pymethods]
impl RxSchwarz {
    #[new]
    fn new(m: CsrC, ptr: PyReadonlyArray1<i64>, idx: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let a = split(&m)?;
        let pats = blocks(ptr.as_slice()?, idx.as_slice()?).into_iter().map(|b| b.into_iter().map(|i| i as u32).collect()).collect();
        let s = Schwarz::new(&a, pats).map_err(err)?;
        Ok(RxSchwarz { bytes: s.memory_bytes(), s })
    }

    fn apply<'py>(&self, py: Python<'py>, r: PyReadonlyArray1<'py, C>) -> PyResult<Bound<'py, PyArray1<C>>> {
        Ok(self.s.apply(r.as_slice()?).into_pyarray_bound(py))
    }
}

#[pyclass(unsendable)]
pub struct RxSweep {
    a: Split,
    blocks: Vec<Vec<usize>>,
    lu: Vec<Direct>,
    #[pyo3(get)]
    factor_nnz: usize,
    #[pyo3(get)]
    factor_seconds: f64,
}

impl RxSweep {
    fn step(&self, b: usize, r: &[C], y: &mut [C], omega: f64) {
        let rows = &self.blocks[b];
        let res: Vec<C> = rows
            .iter()
            .map(|&i| {
                let mut s = r[i];
                self.a.for_row(i, &mut |j, v| s -= v * y[j as usize]);
                s
            })
            .collect();
        let e = self.lu[b].solve(&res);
        for (&i, v) in rows.iter().zip(e) {
            y[i] += v * omega;
        }
    }
}

#[pymethods]
impl RxSweep {
    #[new]
    fn new(m: CsrC, ptr: PyReadonlyArray1<i64>, idx: PyReadonlyArray1<i64>) -> PyResult<Self> {
        let a = split(&m)?;
        let bl = blocks(ptr.as_slice()?, idx.as_slice()?);
        let t = Instant::now();
        let lu: Result<Vec<Direct>, String> = bl.iter().map(|rows| Direct::new(rows.len(), a.lower_on(rows))).collect();
        let lu = lu.map_err(err)?;
        let factor_seconds = t.elapsed().as_secs_f64();
        Ok(RxSweep { factor_nnz: lu.iter().map(|d| d.factor_nnz).sum(), a, blocks: bl, lu, factor_seconds })
    }

    /// y <- y + block Gauss-Seidel corrections for A y = r, blocks in order (forward) or reversed.
    #[pyo3(signature = (r, y, forward, omega=1.0))]
    fn sweep<'py>(&self, py: Python<'py>, r: PyReadonlyArray1<'py, C>, y: PyReadonlyArray1<'py, C>, forward: bool, omega: f64) -> PyResult<Bound<'py, PyArray1<C>>> {
        let r = r.as_slice()?;
        let mut y = y.as_slice()?.to_vec();
        let nb = self.blocks.len();
        for k in 0..nb {
            self.step(if forward { k } else { nb - 1 - k }, r, &mut y, omega);
        }
        Ok(y.into_pyarray_bound(py))
    }

    /// Additive (block Jacobi) correction: sum_b R_b^T A_b^-1 R_b r.
    fn additive<'py>(&self, py: Python<'py>, r: PyReadonlyArray1<'py, C>) -> PyResult<Bound<'py, PyArray1<C>>> {
        let r = r.as_slice()?;
        let parts: Vec<Vec<C>> = self.blocks.iter().zip(&self.lu).map(|(rows, d)| d.solve(&rows.iter().map(|&i| r[i]).collect::<Vec<_>>())).collect();
        let mut z = vec![C::default(); r.len()];
        for (rows, e) in self.blocks.iter().zip(parts) {
            for (&i, v) in rows.iter().zip(e) {
                z[i] += v;
            }
        }
        Ok(z.into_pyarray_bound(py))
    }
}
