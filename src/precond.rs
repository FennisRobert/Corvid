//! The preconditioner, translated one-to-one from the Python prototype
//! (helix palace_pc.PalacePC + CORVID's Gauss-Seidel coarse V-cycle):
//!
//!   fine (N2):   Chebyshev-4 on P scaled by Jacobi + exact surface block,
//!                Chebyshev-4 on G^T P G (Hiptmair), `cycles` stationary V-cycles
//!   Whitney:     injection R (an index gather / scatter)
//!   CORVID:      `coarse_its` stationary V-cycles, GS Hiptmair per level,
//!                Galerkin operators, direct coarsest solve (SPARTA, CPU)

use crate::csr::{CMat, Coloring, Csr, Split};
use crate::direct::Direct;
use crate::patch::Schwarz;
use num_complex::Complex64 as C;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Nanoseconds spent per component (diagnostics, read by `timers()`).
pub static T_SURF: AtomicU64 = AtomicU64::new(0);
pub static T_FINE_SPMV: AtomicU64 = AtomicU64::new(0);
pub static T_GS: AtomicU64 = AtomicU64::new(0);
pub static T_COARSEST: AtomicU64 = AtomicU64::new(0);
pub static T_CORVID: AtomicU64 = AtomicU64::new(0);
pub static T_PC: AtomicU64 = AtomicU64::new(0);
pub static T_RES: AtomicU64 = AtomicU64::new(0);
pub static T_GT: AtomicU64 = AtomicU64::new(0);
pub static T_CHEB_ALL: AtomicU64 = AtomicU64::new(0);
pub static T_B_ANALYSIS: AtomicU64 = AtomicU64::new(0);
pub static T_B_NUMERIC: AtomicU64 = AtomicU64::new(0);
pub static T_B_BLOCKS: AtomicU64 = AtomicU64::new(0);
pub static T_B_LMAX: AtomicU64 = AtomicU64::new(0);
pub static T_B_SPLIT: AtomicU64 = AtomicU64::new(0);
pub static LMAX_ITS: AtomicU64 = AtomicU64::new(0);

pub fn tick(c: &AtomicU64, t: Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn timers() -> Vec<(&'static str, f64)> {
    [("surface LU", &T_SURF), ("fine Chebyshev SpMV", &T_FINE_SPMV), ("CORVID GS", &T_GS),
     ("coarsest LU", &T_COARSEST), ("CORVID total", &T_CORVID), ("fine residuals", &T_RES), ("G / G^T", &T_GT),
     ("Chebyshev runs (all)", &T_CHEB_ALL), ("preconditioner total", &T_PC),
     ("build: K split", &T_B_SPLIT), ("build: coarse analysis", &T_B_ANALYSIS), ("build: coarse numeric", &T_B_NUMERIC),
     ("build: vertex blocks", &T_B_BLOCKS), ("build: lambda_max", &T_B_LMAX)]
        .iter()
        .map(|(n, c)| (*n, c.swap(0, Ordering::Relaxed) as f64 * 1e-9))
        .chain(std::iter::once(("lambda_max iterations", LMAX_ITS.swap(0, Ordering::Relaxed) as f64)))
        .collect()
}

pub use crate::vecops::norm;
use crate::vecops::{add, map_mut};
use rayon::prelude::*;

/// M^-1 of a Chebyshev smoother: Jacobi, optionally an exact solve on some rows.
pub struct Scaling {
    d_inv: Vec<C>,
    surface: Option<(Vec<usize>, Direct)>,
    skip: Vec<bool>,
    schwarz: Option<Schwarz>,
}

impl Scaling {
    pub fn new<A: CMat>(a: &A, surface_rows: Option<Vec<usize>>) -> Result<Self, String> {
        let d_inv: Vec<C> = a.diag().iter().map(|d| if *d == C::default() { C::new(1.0, 0.0) } else { 1.0 / d }).collect();
        let mut skip = vec![false; a.nrows()];
        let surface = match surface_rows {
            Some(rows) if !rows.is_empty() => {
                for &i in &rows {
                    skip[i] = true;
                }
                let lower = a.lower_on(&rows);
                Some((rows.clone(), Direct::new(rows.len(), lower)?))
            }
            _ => None,
        };
        Ok(Scaling { d_inv, surface, skip, schwarz: None })
    }

    pub fn schwarz<A: CMat>(a: &A, patches: Vec<Vec<u32>>) -> Result<Self, String> {
        Self::schwarz_storage(a, patches, false)
    }

    pub fn schwarz_storage<A: CMat>(a: &A, patches: Vec<Vec<u32>>, sparse: bool) -> Result<Self, String> {
        let n = a.nrows();
        Ok(Scaling { d_inv: Vec::new(), surface: None, skip: vec![false; n], schwarz: Some(Schwarz::with_storage(a, patches, sparse)?) })
    }

    /// z on the surface rows only: the exact block solve of r there.
    fn surface_solve(&self, r: &[C]) -> Option<(&[usize], Vec<C>)> {
        self.surface.as_ref().map(|(rows, lu)| {
            let t = Instant::now();
            let rs: Vec<C> = rows.iter().map(|&i| r[i]).collect();
            let xs = lu.solve(&rs);
            tick(&T_SURF, t);
            (&rows[..], xs)
        })
    }

    pub fn apply(&self, r: &[C]) -> Vec<C> {
        if let Some(sw) = &self.schwarz {
            let t = Instant::now();
            let z = sw.apply(r);
            tick(&T_SURF, t);
            return z;
        }
        let mut z: Vec<C> = r.iter().zip(&self.d_inv).map(|(a, b)| a * b).collect();
        if let Some((rows, xs)) = self.surface_solve(r) {
            for (&i, v) in rows.iter().zip(xs) {
                z[i] = v;
            }
        }
        z
    }
}

/// 4th-kind Chebyshev (Phillips & Fischer 2022).
pub struct Chebyshev {
    order: usize,
    lmax: f64,
    scale: Scaling,
}

impl Chebyshev {
    pub fn new<A: CMat>(a: &A, order: usize, sf: f64, scale: Scaling) -> Self {
        let lmax = sf * lambda_max(a, &scale, 1e-4, 1000);
        Chebyshev { order, lmax, scale }
    }

    /// y <- Chebyshev(y) for A y = b; y = None starts from zero. Each step is one
    /// fused pass over A (`Csr::cheb_step`) plus the surface block solve.
    pub fn run<A: CMat>(&self, a: &A, b: &[C], y: Option<Vec<C>>) -> Vec<C> {
        let lm = self.lmax;
        let (mut y, mut r) = match y {
            None => (vec![C::default(); b.len()], b.to_vec()),
            Some(y) => {
                let r = a.residual(b, &y);
                (y, r)
            }
        };
        let c0 = 4.0 / (3.0 * lm);
        let mut d = self.scale.apply(&r);
        map_mut(&mut d, |_, v| *v *= c0);
        let mut dn = vec![C::default(); b.len()];
        if self.scale.schwarz.is_some() {
            let mut ad = vec![C::default(); b.len()];
            for k in 1..self.order {
                let kf = k as f64;
                let (c1, c2) = ((2.0 * kf - 1.0) / (2.0 * kf + 3.0), (8.0 * kf + 4.0) / ((2.0 * kf + 3.0) * lm));
                add(&mut y, &d);
                let t = Instant::now();
                a.apply(&d, &mut ad, None);
                tick(&T_FINE_SPMV, t);
                map_mut(&mut r, |i, ri| *ri -= ad[i]);
                let z = self.scale.apply(&r);
                map_mut(&mut d, |i, di| *di = *di * c1 + z[i] * c2);
            }
            add(&mut y, &d);
            return y;
        }
        for k in 1..self.order {
            let kf = k as f64;
            let (c1, c2) = ((2.0 * kf - 1.0) / (2.0 * kf + 3.0), (8.0 * kf + 4.0) / ((2.0 * kf + 3.0) * lm));
            let t = Instant::now();
            a.cheb_step(&d, &mut dn, &mut y, &mut r, &self.scale.d_inv, &self.scale.skip, c1, c2);
            tick(&T_FINE_SPMV, t);
            if let Some((rows, xs)) = self.scale.surface_solve(&r) {
                for (&i, v) in rows.iter().zip(xs) {
                    dn[i] = d[i] * c1 + v * c2;
                }
            }
            std::mem::swap(&mut d, &mut dn);
        }
        add(&mut y, &d);
        y
    }
}

/// Power iteration on M^-1 A to a relative tolerance (Palace's SpectralNorm).
fn lambda_max<A: CMat>(a: &A, m: &Scaling, tol: f64, max_it: usize) -> f64 {
    let n = a.nrows();
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut v: Vec<C> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            C::new((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5, 0.0)
        })
        .collect();
    let nv = norm(&v);
    v.iter_mut().for_each(|x| *x /= nv);
    let mut lam = 0.0;
    for it in 0..max_it {
        LMAX_ITS.store(it as u64 + 1, Ordering::Relaxed);
        let w = m.apply(&a.mul(&v));
        let l = norm(&w);
        v = w;
        map_mut(&mut v, |_, x| *x /= l);
        if lam > 0.0 && (l - lam).abs() < tol * l {
            return l;
        }
        lam = l;
    }
    lam
}

/// Hiptmair distributive relaxation with Chebyshev on A and on G^T A G.
pub struct DistRelax {
    pub a: Split,
    g: Csr<f64>,
    gt: Csr<f64>,
    ag: Csr<C>,
    s: Chebyshev,
    sg: Chebyshev,
    use_grad: bool,
}

impl DistRelax {
    pub fn new(a: Csr<C>, g: Csr<f64>, gt: Csr<f64>, ag: Csr<C>, order: usize, surface: Option<Vec<usize>>) -> Result<Self, String> {
        let a = Split::from(&a);
        let s = Chebyshev::new(&a, order, 1.1, Scaling::new(&a, surface)?);
        let sg = Chebyshev::new(&ag, order, 1.1, Scaling::new(&ag, None)?);
        Ok(DistRelax { a, g, gt, ag, s, sg, use_grad: true })
    }

    /// Chebyshev scaled by vertex-patch Schwarz; `use_grad` keeps Hiptmair's
    /// gradient correction on top.
    #[allow(clippy::too_many_arguments)]
    pub fn new_schwarz(a: Csr<C>, g: Csr<f64>, gt: Csr<f64>, ag: Csr<C>, order: usize, patches: Vec<Vec<u32>>, use_grad: bool) -> Result<Self, String> {
        let a = Split::from(&a);
        let sc = Scaling::schwarz(&a, patches)?;
        let s = Chebyshev::new(&a, order, 1.1, sc);
        let sg = Chebyshev::new(&ag, order, 1.1, Scaling::new(&ag, None)?);
        Ok(DistRelax { a, g, gt, ag, s, sg, use_grad })
    }

    fn grad(&self, b: &[C], y: &mut Vec<C>) {
        if !self.use_grad {
            return;
        }
        let t = Instant::now();
        let r = self.a.residual(b, y);
        tick(&T_RES, t);
        let t = Instant::now();
        let rg = self.gt.mul(&r);
        tick(&T_GT, t);
        let t = Instant::now();
        let z = self.sg.run(&self.ag, &rg, None);
        tick(&T_CHEB_ALL, t);
        let t = Instant::now();
        let gz = self.g.mul(&z);
        add(y, &gz);
        tick(&T_GT, t);
    }

    pub fn pre(&self, b: &[C]) -> Vec<C> {
        let t = Instant::now();
        let mut y = self.s.run(&self.a, b, None);
        tick(&T_CHEB_ALL, t);
        self.grad(b, &mut y);
        y
    }

    pub fn post(&self, b: &[C], mut y: Vec<C>) -> Vec<C> {
        self.grad(b, &mut y);
        let t = Instant::now();
        let y = self.s.run(&self.a, b, Some(y));
        tick(&T_CHEB_ALL, t);
        y
    }
}

/// One CORVID level's Gauss-Seidel Hiptmair smoother.
pub struct GsLevel {
    pub a: Csr<C>,
    p: Csr<f64>,
    pt: Csr<f64>,
    g: Csr<f64>,
    gt: Csr<f64>,
    ag: Csr<C>,
    col: Option<(Coloring, Coloring)>,
}

impl GsLevel {
    fn sweep(&self, which: usize, x: &mut [C], b: &[C], forward: bool) {
        let m = if which == 0 { &self.a } else { &self.ag };
        match &self.col {
            Some((ca, cg)) => m.gs_colored(x, b, forward, if which == 0 { ca } else { cg }),
            None => m.gs(x, b, forward),
        }
    }

    fn grad(&self, b: &[C], y: &mut Vec<C>, forward: bool) {
        let r = self.a.residual(b, y);
        let rg = self.gt.mul(&r);
        let mut z = vec![C::default(); rg.len()];
        self.sweep(1, &mut z, &rg, forward);
        add(y, &self.g.mul(&z));
    }

    fn pre(&self, b: &[C]) -> Vec<C> {
        let t = Instant::now();
        let mut y = vec![C::default(); b.len()];
        self.sweep(0, &mut y, b, true);
        self.grad(b, &mut y, true);
        tick(&T_GS, t);
        y
    }

    fn post(&self, b: &[C], mut y: Vec<C>) -> Vec<C> {
        let t = Instant::now();
        self.grad(b, &mut y, false);
        self.sweep(0, &mut y, b, false);
        tick(&T_GS, t);
        y
    }
}

pub struct Corvid {
    levels: Vec<GsLevel>,
    coarsest: Direct,
    its: usize,
}

impl Corvid {
    /// `colored`: multicolour (parallel) Gauss-Seidel on levels with at least
    /// `min_rows` rows, sequential below.
    pub fn new(levels: Vec<(Csr<C>, Csr<f64>, Csr<f64>, Csr<f64>, Csr<f64>, Csr<C>)>, coarsest: &Csr<C>, its: usize, colored: bool, min_rows: usize) -> Result<Self, String> {
        let levels = levels
            .into_iter()
            .map(|(a, p, pt, g, gt, ag)| {
                let col = (colored && a.nrows >= min_rows).then(|| (Coloring::greedy(&a), Coloring::greedy(&ag)));
                GsLevel { a, p, pt, g, gt, ag, col }
            })
            .collect();
        let (n, lower) = coarsest.lower(None);
        Ok(Corvid { levels, coarsest: Direct::new(n, lower)?, its })
    }

    fn vcycle(&self, l: usize, b: &[C]) -> Vec<C> {
        if l == self.levels.len() {
            let t = Instant::now();
            let x = self.coarsest.solve(b);
            tick(&T_COARSEST, t);
            return x;
        }
        let lv = &self.levels[l];
        let mut y = lv.pre(b);
        let r = lv.a.residual(b, &y);
        let rc = lv.pt.mul(&r);
        let ec = self.vcycle(l + 1, &rc);
        add(&mut y, &lv.p.mul(&ec));
        lv.post(b, y)
    }

    /// `its` stationary V-cycles from zero.
    pub fn apply(&self, b: &[C]) -> Vec<C> {
        let mut x = self.vcycle(0, b);
        let a0 = &self.levels.first().map(|l| &l.a);
        for _ in 1..self.its {
            let r = match a0 {
                Some(a) => a.residual(b, &x),
                None => return x,
            };
            add(&mut x, &self.vcycle(0, &r));
        }
        x
    }
}

/// What the Krylov solvers need from a preconditioner.
pub trait Precond {
    fn apply(&self, b: &[C]) -> Vec<C>;
}

impl Precond for Preconditioner {
    fn apply(&self, b: &[C]) -> Vec<C> {
        Preconditioner::apply(self, b)
    }
}

/// The production preconditioner for K itself (no shift):
///
///   pre:    Chebyshev (`order`) scaled by vertex-patch Schwarz
///   coarse: exact solve of K on the Whitney (lowest-order edge) rows
///   post:   the same Chebyshev from the corrected iterate
///
/// Symmetric for complex symmetric K, so COCG applies. Every operator works on K interleaved
/// columns at once (`apply_k`), so a solve with several right-hand sides reads K and the block
/// factors once per iteration for all of them and hands SPARTA all coarse columns in one call.
/// `refactor` takes new values on the same pattern (the next frequency of a sweep) and reuses the
/// SPARTA analysis, the block orderings and the spectral estimate's vector.
pub struct TwoLevel {
    pub k: Split,
    schwarz: Schwarz,
    order: usize,
    lmax: f64,
    /// last power-iteration vector, the warm start of the next estimate
    lvec: Vec<C>,
    inj: Vec<usize>,
    coarse: Direct,
    /// positions in K's value array of the coarse factor's entries, in SPARTA's order
    coarse_pos: Vec<usize>,
    pub schwarz_bytes: usize,
    pub dense_fallbacks: usize,
    pub coarse_factor_nnz: usize,
    pub lmax_iterations: usize,
}

/// Chebyshev safety factor over the estimated largest eigenvalue of S K.
const LMAX_SAFETY: f64 = 1.1;
/// Power iteration on the indefinite S K: its norm ratio oscillates rather than converge to tight
/// tolerances, so the estimate is the largest ratio seen, after at least LMAX_MINIT iterations and
/// until the ratio stagnates to LMAX_TOL.
const LMAX_TOL: f64 = 1e-3;
const LMAX_MINIT: usize = 15;
const LMAX_MAXIT: usize = 1000;
/// Power iterations after a refactor (they start from the last vector).
const LMAX_REFACTOR_MINIT: usize = 10;
const LMAX_REFACTOR_MAXIT: usize = 20;

impl TwoLevel {
    /// `k_values`: K's values in CSR order (the pattern is the one `k` was built from).
    pub fn new(k: Split, k_values: &[C], patches: Vec<Vec<u32>>, inj: Vec<usize>, order: usize, sparse: bool) -> Result<Self, String> {
        let (colptr, rows, coarse_pos) = k.lower_csc_positions(&inj);
        let vals: Vec<C> = coarse_pos.par_iter().map(|&p| k_values[p]).collect();
        let coarse = Direct::from_csc(inj.len(), &colptr, &rows, &vals)?;
        let t = Instant::now();
        let schwarz = Schwarz::with_storage(&k, patches, sparse)?;
        tick(&T_B_BLOCKS, t);
        let n = k.nrows();
        let mut tl = TwoLevel {
            schwarz_bytes: schwarz.memory_bytes(),
            dense_fallbacks: schwarz.dense_fallbacks(),
            coarse_factor_nnz: coarse.factor_nnz,
            k,
            schwarz,
            order,
            lmax: 0.0,
            lvec: random_unit(n),
            inj,
            coarse,
            coarse_pos,
            lmax_iterations: 0,
        };
        tl.estimate_lmax(LMAX_MINIT, LMAX_MAXIT);
        Ok(tl)
    }

    /// New values of K on the same pattern.
    pub fn refactor(&mut self, k: Split, k_values: &[C]) -> Result<(), String> {
        let vals: Vec<C> = self.coarse_pos.par_iter().map(|&p| k_values[p]).collect();
        self.coarse.refactor(&vals)?;
        let t = Instant::now();
        self.schwarz.refactor(&k)?;
        tick(&T_B_BLOCKS, t);
        self.k = k;
        self.schwarz_bytes = self.schwarz.memory_bytes();
        self.dense_fallbacks = self.schwarz.dense_fallbacks();
        self.estimate_lmax(LMAX_REFACTOR_MINIT, LMAX_REFACTOR_MAXIT);
        Ok(())
    }

    /// New values of K for the Krylov products and the smoother's residuals, keeping the block and
    /// coarse factors (and the spectral estimate) of the last factorization.
    pub fn update_operator(&mut self, k: Split) {
        self.k = k;
    }

    /// Power iteration on S K from the last vector (Palace's SpectralNorm).
    fn estimate_lmax(&mut self, min_it: usize, max_it: usize) {
        let t = Instant::now();
        let n = self.k.nrows();
        let mut v = std::mem::take(&mut self.lvec);
        let mut w = vec![C::default(); n];
        let (mut lam, mut top) = (0.0f64, 0.0f64);
        let mut its = 0;
        let min_it = min_it.min(max_it);
        for _ in 0..max_it {
            its += 1;
            self.k.apply_k::<1>(&v, &mut w, None);
            let z = self.schwarz.apply_k::<1>(&w);
            let l = norm(&z);
            v = z;
            map_mut(&mut v, |_, x| *x /= l);
            top = top.max(l);
            if its >= min_it && lam > 0.0 && (l - lam).abs() < LMAX_TOL * l {
                break;
            }
            lam = l;
        }
        self.lvec = v;
        self.lmax = LMAX_SAFETY * top;
        self.lmax_iterations = its;
        LMAX_ITS.store(its as u64, Ordering::Relaxed);
        tick(&T_B_LMAX, t);
    }

    /// Chebyshev (4th kind) smoothing of A Y = B from Y (zero if None), K interleaved columns.
    fn smooth_k<const K: usize>(&self, b: &[C], y: Option<Vec<C>>) -> Vec<C> {
        let n = b.len();
        let lm = self.lmax;
        let (mut y, mut r) = match y {
            None => (vec![C::default(); n], b.to_vec()),
            Some(y) => {
                let mut r = vec![C::default(); n];
                let t = Instant::now();
                self.k.apply_k::<K>(&y, &mut r, Some(b));
                tick(&T_FINE_SPMV, t);
                (y, r)
            }
        };
        let c0 = 4.0 / (3.0 * lm);
        let t = Instant::now();
        let mut d = self.schwarz.apply_k::<K>(&r);
        tick(&T_SURF, t);
        map_mut(&mut d, |_, v| *v *= c0);
        let mut ad = vec![C::default(); n];
        for k in 1..self.order {
            let kf = k as f64;
            let (c1, c2) = ((2.0 * kf - 1.0) / (2.0 * kf + 3.0), (8.0 * kf + 4.0) / ((2.0 * kf + 3.0) * lm));
            add(&mut y, &d);
            let t = Instant::now();
            self.k.apply_k::<K>(&d, &mut ad, None);
            tick(&T_FINE_SPMV, t);
            map_mut(&mut r, |i, ri| *ri -= ad[i]);
            let t = Instant::now();
            let z = self.schwarz.apply_k::<K>(&r);
            tick(&T_SURF, t);
            map_mut(&mut d, |i, di| *di = *di * c1 + z[i] * c2);
        }
        add(&mut y, &d);
        y
    }

    /// Z = M B for K interleaved columns.
    fn apply_k<const K: usize>(&self, b: &[C]) -> Vec<C> {
        let t = Instant::now();
        let n = b.len() / K;
        let mut y = self.smooth_k::<K>(b, None);
        let mut r = vec![C::default(); b.len()];
        let ts = Instant::now();
        self.k.apply_k::<K>(&y, &mut r, Some(b));
        tick(&T_FINE_SPMV, ts);
        let nc = self.inj.len();
        // coarse right-hand sides column after column, as SPARTA takes them
        let mut rc = vec![C::default(); nc * K];
        rc.par_chunks_mut(nc).enumerate().for_each(|(c, col)| {
            col.iter_mut().zip(&self.inj).for_each(|(v, &i)| *v = r[i * K + c]);
        });
        let tc = Instant::now();
        let ec = self.coarse.solve_many(&rc, K);
        tick(&T_COARSEST, tc);
        for (q, &i) in self.inj.iter().enumerate() {
            for c in 0..K {
                y[i * K + c] += ec[c * nc + q];
            }
        }
        debug_assert_eq!(y.len(), n * K);
        let y = self.smooth_k::<K>(b, Some(y));
        tick(&T_PC, t);
        y
    }
}

fn random_unit(n: usize) -> Vec<C> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut v: Vec<C> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            C::new((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5, 0.0)
        })
        .collect();
    let nv = norm(&v);
    map_mut(&mut v, |_, x| *x /= nv);
    v
}

/// Largest number of columns the operators handle at once; more are solved in groups.
pub const MAX_COLS: usize = 4;

impl crate::krylov::BlockOp for TwoLevel {
    fn n(&self) -> usize {
        self.k.nrows()
    }
    fn mul_k(&self, x: &[C], y: &mut [C], k: usize) {
        match k {
            1 => self.k.apply_k::<1>(x, y, None),
            2 => self.k.apply_k::<2>(x, y, None),
            3 => self.k.apply_k::<3>(x, y, None),
            4 => self.k.apply_k::<4>(x, y, None),
            _ => panic!("at most {MAX_COLS} columns"),
        }
    }
    fn prec_k(&self, r: &[C], k: usize) -> Vec<C> {
        match k {
            1 => self.apply_k::<1>(r),
            2 => self.apply_k::<2>(r),
            3 => self.apply_k::<3>(r),
            4 => self.apply_k::<4>(r),
            _ => panic!("at most {MAX_COLS} columns"),
        }
    }
}

impl Precond for TwoLevel {
    fn apply(&self, b: &[C]) -> Vec<C> {
        self.apply_k::<1>(b)
    }
}

/// The two-level p-multigrid around CORVID.
pub struct Preconditioner {
    fine: DistRelax,
    inj: Vec<usize>,
    coarse: Corvid,
    cycles: usize,
}

impl Preconditioner {
    pub fn new(fine: DistRelax, inj: Vec<usize>, coarse: Corvid, cycles: usize) -> Self {
        Preconditioner { fine, inj, coarse, cycles }
    }

    fn vcycle(&self, b: &[C]) -> Vec<C> {
        let mut y = self.fine.pre(b);
        let t = Instant::now();
        let r = self.fine.a.residual(b, &y);
        tick(&T_RES, t);
        let rc: Vec<C> = self.inj.iter().map(|&i| r[i]).collect();
        let t = Instant::now();
        let ec = self.coarse.apply(&rc);
        tick(&T_CORVID, t);
        for (&i, v) in self.inj.iter().zip(ec) {
            y[i] += v;
        }
        self.fine.post(b, y)
    }

    pub fn apply(&self, b: &[C]) -> Vec<C> {
        let t = Instant::now();
        let mut y = self.vcycle(b);
        for _ in 1..self.cycles {
            let t = Instant::now();
            let r = self.fine.a.residual(b, &y);
            tick(&T_RES, t);
            add(&mut y, &self.vcycle(&r));
        }
        tick(&T_PC, t);
        y
    }
}
