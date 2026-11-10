//! CSR matrices with u32 column indices; products run on the rayon pool when the
//! matrix is large enough for threads to pay.

use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::ops::{AddAssign, Mul, Sub};

const ROWS_PER_TASK: usize = 512;
const PAR_NNZ: usize = 100_000;

pub struct Csr<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub ip: Vec<usize>,
    pub ix: Vec<u32>,
    pub a: Vec<T>,
}

#[inline(always)]
fn row_dot<M: Copy, V>(ip: &[usize], ix: &[u32], a: &[M], x: &[V], i: usize) -> V
where
    V: Copy + Default + AddAssign + Mul<M, Output = V>,
{
    let mut s = V::default();
    for p in ip[i]..ip[i + 1] {
        s += x[ix[p] as usize] * a[p];
    }
    s
}

impl<M: Copy + Send + Sync> Csr<M> {
    pub fn new(nrows: usize, ncols: usize, ip: &[i32], ix: &[i32], a: &[M]) -> Self {
        Csr {
            nrows,
            ncols,
            ip: ip.iter().map(|&v| v as usize).collect(),
            ix: ix.iter().map(|&v| v as u32).collect(),
            a: a.to_vec(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.a.len()
    }

    /// y = A x (b = None) or y = b - A x
    pub fn apply<V>(&self, x: &[V], y: &mut [V], b: Option<&[V]>)
    where
        V: Copy + Send + Sync + Default + AddAssign + Sub<Output = V> + Mul<M, Output = V>,
    {
        let (ip, ix, a) = (&self.ip[..], &self.ix[..], &self.a[..]);
        let body = |r0: usize, ys: &mut [V]| {
            for (k, yi) in ys.iter_mut().enumerate() {
                let i = r0 + k;
                let s = row_dot(ip, ix, a, x, i);
                *yi = match b {
                    Some(b) => b[i] - s,
                    None => s,
                };
            }
        };
        if self.nnz() >= PAR_NNZ {
            y.par_chunks_mut(ROWS_PER_TASK).enumerate().for_each(|(c, ys)| body(c * ROWS_PER_TASK, ys));
        } else {
            body(0, y);
        }
    }

    pub fn mul<V>(&self, x: &[V]) -> Vec<V>
    where
        V: Copy + Send + Sync + Default + AddAssign + Sub<Output = V> + Mul<M, Output = V>,
    {
        let mut y = vec![V::default(); self.nrows];
        self.apply(x, &mut y, None);
        y
    }

    pub fn residual<V>(&self, b: &[V], x: &[V]) -> Vec<V>
    where
        V: Copy + Send + Sync + Default + AddAssign + Sub<Output = V> + Mul<M, Output = V>,
    {
        let mut y = vec![V::default(); self.nrows];
        self.apply(x, &mut y, Some(b));
        y
    }
}

impl Csr<C> {
    pub fn diag(&self) -> Vec<C> {
        (0..self.nrows)
            .map(|i| {
                (self.ip[i]..self.ip[i + 1])
                    .find(|&p| self.ix[p] as usize == i)
                    .map(|p| self.a[p])
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Sequential Gauss-Seidel sweep on A x = b, in place (zero-diagonal rows skipped).
    pub fn gs(&self, x: &mut [C], b: &[C], forward: bool) {
        let mut row = |i: usize| {
            let mut s = b[i];
            let mut d = C::default();
            for p in self.ip[i]..self.ip[i + 1] {
                let j = self.ix[p] as usize;
                if j == i {
                    d = self.a[p];
                } else {
                    s -= self.a[p] * x[j];
                }
            }
            if d != C::default() {
                x[i] = s / d;
            }
        };
        if forward {
            (0..self.nrows).for_each(&mut row);
        } else {
            (0..self.nrows).rev().for_each(&mut row);
        }
    }

    /// Lower triangle (i >= j) of the principal submatrix on `rows` (None: all
    /// rows), as (i, j, a) in local numbering.
    pub fn lower(&self, rows: Option<&[usize]>) -> (usize, Vec<(usize, usize, C)>) {
        let all: Vec<usize>;
        let rows = match rows {
            Some(r) => r,
            None => {
                all = (0..self.nrows).collect();
                &all
            }
        };
        let mut pos = vec![usize::MAX; self.ncols];
        for (k, &r) in rows.iter().enumerate() {
            pos[r] = k;
        }
        let mut t = Vec::new();
        for (k, &r) in rows.iter().enumerate() {
            for p in self.ip[r]..self.ip[r + 1] {
                let c = pos[self.ix[p] as usize];
                if c != usize::MAX && k >= c {
                    t.push((k, c, self.a[p]));
                }
            }
        }
        (rows.len(), t)
    }
}

/// Rows grouped by colour: no two rows of one colour are coupled, so a colour
/// can be relaxed in parallel. Greedy (first-fit) on the matrix pattern.
pub struct Coloring {
    pub colors: Vec<Vec<u32>>,
}

impl Coloring {
    pub fn greedy<T>(a: &Csr<T>) -> Self {
        let n = a.nrows;
        let mut color = vec![u32::MAX; n];
        let mut mark: Vec<usize> = Vec::new();
        let mut ncol = 0u32;
        for i in 0..n {
            mark.clear();
            for p in a.ip[i]..a.ip[i + 1] {
                let c = color[a.ix[p] as usize];
                if c != u32::MAX {
                    mark.push(c as usize);
                }
            }
            let mut used = vec![false; mark.iter().max().map_or(0, |m| m + 1) + 1];
            for &c in &mark {
                used[c] = true;
            }
            let c = used.iter().position(|u| !u).unwrap() as u32;
            color[i] = c;
            ncol = ncol.max(c + 1);
        }
        let mut colors = vec![Vec::new(); ncol as usize];
        for (i, &c) in color.iter().enumerate() {
            colors[c as usize].push(i as u32);
        }
        Coloring { colors }
    }
}

struct SyncPtr(*mut C);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

impl Csr<C> {
    /// Multicolour Gauss-Seidel sweep: colours in order (reverse when
    /// backward), rows of one colour in parallel.
    pub fn gs_colored(&self, x: &mut [C], b: &[C], forward: bool, col: &Coloring) {
        let px = SyncPtr(x.as_mut_ptr());
        let px = &px;
        let relax = |i: usize| {
            let x = px.0;
            let mut s = b[i];
            let mut d = C::default();
            for p in self.ip[i]..self.ip[i + 1] {
                let j = self.ix[p] as usize;
                if j == i {
                    d = self.a[p];
                } else {
                    // SAFETY: rows of one colour are not coupled, so x[j] is not
                    // written while this colour is relaxed; x[i] is written once.
                    s -= self.a[p] * unsafe { *x.add(j) };
                }
            }
            if d != C::default() {
                unsafe { *x.add(i) = s / d };
            }
        };
        let order: Box<dyn Iterator<Item = &Vec<u32>>> =
            if forward { Box::new(col.colors.iter()) } else { Box::new(col.colors.iter().rev()) };
        for rows in order {
            if rows.len() >= 256 {
                rows.par_chunks(64).for_each(|ch| ch.iter().for_each(|&i| relax(i as usize)));
            } else {
                rows.iter().for_each(|&i| relax(i as usize));
            }
        }
    }
}

impl Csr<C> {
    /// One fused 4th-kind Chebyshev step (rows in parallel):
    ///   y += d;  r -= A d;  dn = c1 d + c2 D^-1 r
    /// `dn` is only written on rows with `skip[i] == false`; the caller fills the
    /// skipped (surface) rows afterwards from the updated r and the old d.
    #[allow(clippy::too_many_arguments)]
    pub fn cheb_step(&self, d: &[C], dn: &mut [C], y: &mut [C], r: &mut [C], d_inv: &[C], skip: &[bool], c1: f64, c2: f64) {
        let (ip, ix, a) = (&self.ip[..], &self.ix[..], &self.a[..]);
        let body = |r0: usize, dn: &mut [C], y: &mut [C], r: &mut [C]| {
            for k in 0..dn.len() {
                let i = r0 + k;
                let ad = row_dot(ip, ix, a, d, i);
                y[k] += d[i];
                r[k] -= ad;
                if !skip[i] {
                    dn[k] = d[i] * c1 + d_inv[i] * r[k] * c2;
                }
            }
        };
        if self.nnz() >= PAR_NNZ {
            dn.par_chunks_mut(ROWS_PER_TASK)
                .zip(y.par_chunks_mut(ROWS_PER_TASK))
                .zip(r.par_chunks_mut(ROWS_PER_TASK))
                .enumerate()
                .for_each(|(c, ((dn, y), r))| body(c * ROWS_PER_TASK, dn, y, r));
        } else {
            body(0, dn, y, r);
        }
    }
}

/// The complex operators the smoothers and Krylov solvers apply.
pub trait CMat: Sync {
    fn nrows(&self) -> usize;
    /// Visit row i's entries (column, value), each column once, columns ascending.
    fn for_row(&self, i: usize, f: &mut dyn FnMut(u32, C));
    /// y = A x (b = None) or y = b - A x
    fn apply(&self, x: &[C], y: &mut [C], b: Option<&[C]>);
    fn diag(&self) -> Vec<C>;
    fn to_csr(&self) -> Csr<C>;
    #[allow(clippy::too_many_arguments)]
    fn cheb_step(&self, d: &[C], dn: &mut [C], y: &mut [C], r: &mut [C], d_inv: &[C], skip: &[bool], c1: f64, c2: f64);

    fn mul(&self, x: &[C]) -> Vec<C> {
        let mut y = vec![C::default(); self.nrows()];
        self.apply(x, &mut y, None);
        y
    }

    /// Lower triangle (i >= j) of the principal submatrix on `rows`, local numbering.
    fn lower_on(&self, rows: &[usize]) -> Vec<(usize, usize, C)> {
        let mut pos = std::collections::HashMap::with_capacity(rows.len());
        for (k, &r) in rows.iter().enumerate() {
            pos.insert(r as u32, k);
        }
        let mut t = Vec::new();
        for (k, &r) in rows.iter().enumerate() {
            self.for_row(r, &mut |j, v| {
                if let Some(&c) = pos.get(&j) {
                    if k >= c {
                        t.push((k, c, v));
                    }
                }
            });
        }
        t
    }
    fn residual(&self, b: &[C], x: &[C]) -> Vec<C> {
        let mut y = vec![C::default(); self.nrows()];
        self.apply(x, &mut y, Some(b));
        y
    }
}

impl CMat for Csr<C> {
    fn nrows(&self) -> usize { self.nrows }
    fn for_row(&self, i: usize, f: &mut dyn FnMut(u32, C)) {
        for p in self.ip[i]..self.ip[i + 1] {
            f(self.ix[p], self.a[p]);
        }
    }
    fn apply(&self, x: &[C], y: &mut [C], b: Option<&[C]>) { Csr::apply(self, x, y, b) }
    fn diag(&self) -> Vec<C> { Csr::diag(self) }
    fn to_csr(&self) -> Csr<C> {
        Csr { nrows: self.nrows, ncols: self.ncols, ip: self.ip.clone(), ix: self.ix.clone(), a: self.a.clone() }
    }
    fn cheb_step(&self, d: &[C], dn: &mut [C], y: &mut [C], r: &mut [C], d_inv: &[C], skip: &[bool], c1: f64, c2: f64) {
        Csr::cheb_step(self, d, dn, y, r, d_inv, skip, c1, c2)
    }
}

/// A complex matrix in the cheaper of two layouts for its products:
///   - split: a real CSR of everything plus a real CSR of the imaginary parts, when the imaginary
///     part lives on few entries (absorbing and port surfaces): 12 instead of 20 bytes per nonzero;
///   - full: complex values on the same pattern, when many entries are complex (lossy dielectrics,
///     PMLs), where a second index pass would cost more than it saves.
/// `re.ip` / `re.ix` are the pattern in both cases.
pub struct Split {
    re: Csr<f64>,
    im: Csr<f64>,
    full: Option<Vec<C>>,
}

/// Above this share of complex entries the full complex layout is used.
const FULL_COMPLEX_SHARE: f64 = 0.2;

impl Split {
    /// From a complex CSR given as slices (sorted columns).
    pub fn from_parts(nrows: usize, ncols: usize, ip: &[i32], ix: &[i32], a: &[C]) -> Self {
        let ipu: Vec<usize> = ip.iter().map(|&v| v as usize).collect();
        let ixu: Vec<u32> = ix.iter().map(|&v| v as u32).collect();
        let n_im = a.par_iter().filter(|v| v.im != 0.0).count();
        if n_im as f64 > FULL_COMPLEX_SHARE * a.len() as f64 {
            let re = Csr { nrows, ncols, ip: ipu, ix: ixu, a: Vec::new() };
            let im = Csr { nrows, ncols, ip: vec![0; nrows + 1], ix: Vec::new(), a: Vec::new() };
            return Split { re, im, full: Some(a.to_vec()) };
        }
        let (mut jp, mut jx, mut jv) = (vec![0usize; nrows + 1], Vec::new(), Vec::new());
        for i in 0..nrows {
            for p in ipu[i]..ipu[i + 1] {
                if a[p].im != 0.0 {
                    jx.push(ixu[p]);
                    jv.push(a[p].im);
                }
            }
            jp[i + 1] = jx.len();
        }
        let re = Csr { nrows, ncols, ip: ipu, ix: ixu, a: a.iter().map(|v| v.re).collect() };
        Split { re, im: Csr { nrows, ncols, ip: jp, ix: jx, a: jv }, full: None }
    }

    pub fn from(a: &Csr<C>) -> Self {
        let ip: Vec<i32> = a.ip.iter().map(|&v| v as i32).collect();
        let ix: Vec<i32> = a.ix.iter().map(|&v| v as i32).collect();
        Self::from_parts(a.nrows, a.ncols, &ip, &ix, &a.a)
    }

    fn nnz(&self) -> usize {
        self.re.ix.len()
    }

    /// Y = A X (b = None) or Y = B - A X for K columns stored interleaved (X[i * K + c]): every
    /// matrix entry is read once for all K columns.
    pub fn apply_k<const K: usize>(&self, x: &[C], y: &mut [C], b: Option<&[C]>) {
        if let Some(fa) = &self.full {
            return self.apply_k_full::<K>(fa, x, y, b);
        }
        let (rp, rx, ra) = (&self.re.ip[..], &self.re.ix[..], &self.re.a[..]);
        let (jp, jx, ja) = (&self.im.ip[..], &self.im.ix[..], &self.im.a[..]);
        let body = |r0: usize, ys: &mut [C]| {
            for (k, yi) in ys.chunks_exact_mut(K).enumerate() {
                let i = r0 + k;
                let mut s = [C::default(); K];
                for p in rp[i]..rp[i + 1] {
                    let (a, j) = (ra[p], rx[p] as usize * K);
                    for c in 0..K {
                        s[c] += x[j + c] * a;
                    }
                }
                for q in jp[i]..jp[i + 1] {
                    let (a, j) = (ja[q], jx[q] as usize * K);
                    for c in 0..K {
                        let v = x[j + c] * a;
                        s[c] += C::new(-v.im, v.re);
                    }
                }
                match b {
                    Some(b) => {
                        for c in 0..K {
                            yi[c] = b[i * K + c] - s[c];
                        }
                    }
                    None => yi.copy_from_slice(&s),
                }
            }
        };
        if self.nnz() >= PAR_NNZ {
            y.par_chunks_mut(ROWS_PER_TASK * K).enumerate().for_each(|(c, ys)| body(c * ROWS_PER_TASK, ys));
        } else {
            body(0, y);
        }
    }

    /// apply_k on the full complex layout. Two partial sums per column keep more products in flight.
    fn apply_k_full<const K: usize>(&self, fa: &[C], x: &[C], y: &mut [C], b: Option<&[C]>) {
        let (rp, rx) = (&self.re.ip[..], &self.re.ix[..]);
        let body = |r0: usize, ys: &mut [C]| {
            for (k, yi) in ys.chunks_exact_mut(K).enumerate() {
                let i = r0 + k;
                let (p0, p1) = (rp[i], rp[i + 1]);
                let mut s0 = [C::default(); K];
                let mut s1 = [C::default(); K];
                let mut p = p0;
                while p + 1 < p1 {
                    let (a0, j0) = (fa[p], rx[p] as usize * K);
                    let (a1, j1) = (fa[p + 1], rx[p + 1] as usize * K);
                    for c in 0..K {
                        s0[c] += x[j0 + c] * a0;
                        s1[c] += x[j1 + c] * a1;
                    }
                    p += 2;
                }
                if p < p1 {
                    let (a0, j0) = (fa[p], rx[p] as usize * K);
                    for c in 0..K {
                        s0[c] += x[j0 + c] * a0;
                    }
                }
                match b {
                    Some(b) => {
                        for c in 0..K {
                            yi[c] = b[i * K + c] - (s0[c] + s1[c]);
                        }
                    }
                    None => {
                        for c in 0..K {
                            yi[c] = s0[c] + s1[c];
                        }
                    }
                }
            }
        };
        if self.nnz() >= PAR_NNZ {
            y.par_chunks_mut(ROWS_PER_TASK * K).enumerate().for_each(|(c, ys)| body(c * ROWS_PER_TASK, ys));
        } else {
            body(0, y);
        }
    }

    /// Positions in this matrix's (= the caller's CSR) value array of the lower triangle of the principal
    /// submatrix on `rows`, as compressed columns in local numbering: (colptr, row indices, positions).
    pub fn lower_csc_positions(&self, rows: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let n = rows.len();
        let mut local = vec![u32::MAX; self.re.nrows];
        for (k, &r) in rows.iter().enumerate() {
            local[r] = k as u32;
        }
        // entries (i, j, p) with i >= j, grouped by column j
        let mut count = vec![0usize; n + 1];
        for (i, &r) in rows.iter().enumerate() {
            for p in self.re.ip[r]..self.re.ip[r + 1] {
                let j = local[self.re.ix[p] as usize];
                if j != u32::MAX && (j as usize) <= i {
                    count[j as usize + 1] += 1;
                }
            }
        }
        for j in 0..n {
            count[j + 1] += count[j];
        }
        let colptr = count.clone();
        let mut ri = vec![0usize; colptr[n]];
        let mut pos = vec![0usize; colptr[n]];
        // rows visited in increasing local index, so every column's rows come out ascending
        for (i, &r) in rows.iter().enumerate() {
            for p in self.re.ip[r]..self.re.ip[r + 1] {
                let j = local[self.re.ix[p] as usize];
                if j != u32::MAX && (j as usize) <= i {
                    let q = count[j as usize];
                    ri[q] = i;
                    pos[q] = p;
                    count[j as usize] += 1;
                }
            }
        }
        (colptr, ri, pos)
    }

    #[inline(always)]
    fn row(&self, x: &[C], i: usize) -> C {
        if let Some(fa) = &self.full {
            return row_dot(&self.re.ip, &self.re.ix, fa, x, i);
        }
        let s = row_dot(&self.re.ip, &self.re.ix, &self.re.a, x, i);
        if self.im.ip[i] == self.im.ip[i + 1] {
            return s;
        }
        let t = row_dot(&self.im.ip, &self.im.ix, &self.im.a, x, i);
        s + C::new(-t.im, t.re)
    }
}

impl CMat for Split {
    fn nrows(&self) -> usize { self.re.nrows }

    fn for_row(&self, i: usize, f: &mut dyn FnMut(u32, C)) {
        if let Some(fa) = &self.full {
            for p in self.re.ip[i]..self.re.ip[i + 1] {
                f(self.re.ix[p], fa[p]);
            }
            return;
        }
        let mut q = self.im.ip[i];
        let qe = self.im.ip[i + 1];
        for p in self.re.ip[i]..self.re.ip[i + 1] {
            let j = self.re.ix[p];
            let mut v = C::new(self.re.a[p], 0.0);
            if q < qe && self.im.ix[q] == j {
                v.im = self.im.a[q];
                q += 1;
            }
            f(j, v);
        }
    }

    fn apply(&self, x: &[C], y: &mut [C], b: Option<&[C]>) {
        let body = |r0: usize, ys: &mut [C]| {
            for (k, yi) in ys.iter_mut().enumerate() {
                let s = self.row(x, r0 + k);
                *yi = match b { Some(b) => b[r0 + k] - s, None => s };
            }
        };
        if self.nnz() >= PAR_NNZ {
            y.par_chunks_mut(ROWS_PER_TASK).enumerate().for_each(|(c, ys)| body(c * ROWS_PER_TASK, ys));
        } else {
            body(0, y);
        }
    }

    fn diag(&self) -> Vec<C> {
        (0..self.nrows())
            .map(|i| {
                let mut d = C::default();
                self.for_row(i, &mut |j, v| {
                    if j as usize == i {
                        d = v;
                    }
                });
                d
            })
            .collect()
    }

    fn to_csr(&self) -> Csr<C> {
        let n = self.re.nrows;
        if let Some(fa) = &self.full {
            return Csr { nrows: n, ncols: self.re.ncols, ip: self.re.ip.clone(), ix: self.re.ix.clone(), a: fa.clone() };
        }
        let mut a: Vec<C> = self.re.a.iter().map(|&v| C::new(v, 0.0)).collect();
        for i in 0..n {
            for q in self.im.ip[i]..self.im.ip[i + 1] {
                let j = self.im.ix[q];
                let p = (self.re.ip[i]..self.re.ip[i + 1]).find(|&p| self.re.ix[p] == j).expect("imag pattern within real");
                a[p].im = self.im.a[q];
            }
        }
        Csr { nrows: n, ncols: self.re.ncols, ip: self.re.ip.clone(), ix: self.re.ix.clone(), a }
    }

    fn cheb_step(&self, d: &[C], dn: &mut [C], y: &mut [C], r: &mut [C], d_inv: &[C], skip: &[bool], c1: f64, c2: f64) {
        let body = |r0: usize, dn: &mut [C], y: &mut [C], r: &mut [C]| {
            for k in 0..dn.len() {
                let i = r0 + k;
                let ad = self.row(d, i);
                y[k] += d[i];
                r[k] -= ad;
                if !skip[i] {
                    dn[k] = d[i] * c1 + d_inv[i] * r[k] * c2;
                }
            }
        };
        if self.nnz() >= PAR_NNZ {
            dn.par_chunks_mut(ROWS_PER_TASK)
                .zip(y.par_chunks_mut(ROWS_PER_TASK))
                .zip(r.par_chunks_mut(ROWS_PER_TASK))
                .enumerate()
                .for_each(|(c, ((dn, y), r))| body(c * ROWS_PER_TASK, dn, y, r));
        } else {
            body(0, dn, y, r);
        }
    }
}
