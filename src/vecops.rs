//! Threaded vector kernels for the Krylov loops and the smoother. Long vectors are cut into fixed
//! chunks; reductions sum the chunk partials in chunk order, so results do not depend on the thread
//! count or on scheduling (iteration counts are reproducible).

use num_complex::Complex64 as C;
use rayon::prelude::*;

/// Entries per task. Below `PAR_MIN` everything runs on the calling thread.
const CHUNK: usize = 8192;
const PAR_MIN: usize = 32768;

#[inline]
fn par(n: usize) -> bool {
    n >= PAR_MIN
}

/// Deterministic sum of f over the chunks of 0..n.
fn reduce<T, F>(n: usize, f: F) -> T
where
    T: Send + Default + std::iter::Sum<T>,
    F: Fn(std::ops::Range<usize>) -> T + Sync,
{
    if !par(n) {
        return f(0..n);
    }
    let parts: Vec<T> = (0..n.div_ceil(CHUNK)).into_par_iter().map(|c| f(c * CHUNK..((c + 1) * CHUNK).min(n))).collect();
    parts.into_iter().sum()
}

/// sum conj(a) b
pub fn dotc(a: &[C], b: &[C]) -> C {
    reduce(a.len(), |r| a[r.clone()].iter().zip(&b[r]).map(|(x, y)| x.conj() * y).sum())
}

/// sum a b (no conjugate: the bilinear form of COCG)
pub fn dotu(a: &[C], b: &[C]) -> C {
    reduce(a.len(), |r| a[r.clone()].iter().zip(&b[r]).map(|(x, y)| x * y).sum())
}

pub fn norm(x: &[C]) -> f64 {
    reduce::<f64, _>(x.len(), |r| x[r].iter().map(|v| v.norm_sqr()).sum()).sqrt()
}

/// y[i] = f(i, y[i]) over all i, in parallel for long vectors.
pub fn map_mut<F: Fn(usize, &mut C) + Sync>(y: &mut [C], f: F) {
    if !par(y.len()) {
        y.iter_mut().enumerate().for_each(|(i, v)| f(i, v));
        return;
    }
    y.par_chunks_mut(CHUNK).enumerate().for_each(|(c, ys)| {
        let i0 = c * CHUNK;
        ys.iter_mut().enumerate().for_each(|(k, v)| f(i0 + k, v));
    });
}

/// (a[i], b[i]) = f(i, a[i], b[i]) over all i: two vectors updated in one pass.
pub fn map_mut2<F: Fn(usize, &mut C, &mut C) + Sync>(a: &mut [C], b: &mut [C], f: F) {
    if !par(a.len()) {
        a.iter_mut().zip(b.iter_mut()).enumerate().for_each(|(i, (x, y))| f(i, x, y));
        return;
    }
    a.par_chunks_mut(CHUNK).zip(b.par_chunks_mut(CHUNK)).enumerate().for_each(|(c, (xs, ys))| {
        let i0 = c * CHUNK;
        xs.iter_mut().zip(ys.iter_mut()).enumerate().for_each(|(k, (x, y))| f(i0 + k, x, y));
    });
}

/// A new vector with entries f(i).
pub fn build<F: Fn(usize) -> C + Sync>(n: usize, f: F) -> Vec<C> {
    let mut y = vec![C::default(); n];
    map_mut(&mut y, |i, v| *v = f(i));
    y
}

/// y += x
pub fn add(y: &mut [C], x: &[C]) {
    map_mut(y, |i, v| *v += x[i]);
}

/// y += a x
pub fn axpy(y: &mut [C], a: C, x: &[C]) {
    map_mut(y, |i, v| *v += a * x[i]);
}
