//! Outer solvers, as in helix palace_pc: right-preconditioned restarted GMRES
//! (MGS Arnoldi, Givens) and COCG for complex symmetric systems. Both return
//! (x, iterations, true relative residual).

use crate::csr::CMat;
use crate::precond::Precond;
use crate::vecops::{add, axpy, build, dotc, dotu, map_mut, map_mut2, norm};
use num_complex::Complex64 as C;

/// A starting guess is used only if it reduces the residual below this fraction of ||b||. A guess
/// that barely beats zero can still converge slower than zero (its residual is rich in slow modes).
const X0_GAIN: f64 = 0.1;

pub fn gmres<K: CMat, M: Precond>(k: &K, b: &[C], m: &M, rtol: f64, restart: usize, maxiter: usize) -> (Vec<C>, usize, f64) {
    gmres_x0(k, b, None, m, rtol, restart, maxiter)
}

/// GMRES from `x0` if it beats zero.
pub fn gmres_x0<K: CMat, M: Precond>(k: &K, b: &[C], x0: Option<&[C]>, m: &M, rtol: f64, restart: usize, maxiter: usize) -> (Vec<C>, usize, f64) {
    let n = b.len();
    let bn = norm(b);
    let mut x = match x0 {
        Some(x0) if norm(&k.residual(b, x0)) < X0_GAIN * bn => x0.to_vec(),
        _ => vec![C::default(); n],
    };
    if bn == 0.0 {
        return (x, 0, 0.0);
    }
    let mut its = 0;
    while its < maxiter {
        let r = k.residual(b, &x);
        let beta = norm(&r);
        if beta / bn < rtol {
            break;
        }
        let mm = restart.min(maxiter - its);
        let mut v: Vec<Vec<C>> = Vec::with_capacity(mm + 1);
        v.push(build(n, |i| r[i] / beta));
        let mut h = vec![vec![C::default(); mm]; mm + 1];
        let (mut cs, mut sn) = (vec![C::default(); mm], vec![C::default(); mm]);
        let mut g = vec![C::default(); mm + 1];
        g[0] = C::new(beta, 0.0);
        let mut done = 0;
        for j in 0..mm {
            let mut w = k.mul(&m.apply(&v[j]));
            for i in 0..=j {
                let hij = dotc(&v[i], &w);
                h[i][j] = hij;
                axpy(&mut w, -hij, &v[i]);
            }
            let hn = norm(&w);
            h[j + 1][j] = C::new(hn, 0.0);
            v.push(if hn != 0.0 { build(n, |i| w[i] / hn) } else { w });
            for i in 0..j {
                let t = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i].conj() * h[i][j] + cs[i].conj() * h[i + 1][j];
                h[i][j] = t;
            }
            let (h1, h2) = (h[j][j], h[j + 1][j]);
            let den = (h1.norm_sqr() + h2.norm_sqr()).sqrt();
            if den != 0.0 {
                cs[j] = h1.conj() / den;
                sn[j] = h2.conj() / den;
            } else {
                cs[j] = C::new(1.0, 0.0);
                sn[j] = C::default();
            }
            h[j][j] = cs[j] * h1 + sn[j] * h2;
            h[j + 1][j] = C::default();
            g[j + 1] = -sn[j].conj() * g[j];
            g[j] = cs[j] * g[j];
            its += 1;
            done = j + 1;
            if g[j + 1].norm() / bn < rtol {
                break;
            }
        }
        // back substitution on the triangular H, then x += M (V y)
        let mut y = vec![C::default(); done];
        for i in (0..done).rev() {
            let mut s = g[i];
            for l in i + 1..done {
                s -= h[i][l] * y[l];
            }
            y[i] = s / h[i][i];
        }
        let mut vy = vec![C::default(); n];
        for (vi, &yi) in v.iter().zip(&y) {
            axpy(&mut vy, yi, vi);
        }
        let dx = m.apply(&vy);
        add(&mut x, &dx);
    }
    let rr = norm(&k.residual(b, &x)) / bn;
    (x, its, rr)
}

pub fn cocg<K: CMat, M: Precond>(k: &K, b: &[C], m: &M, rtol: f64, maxiter: usize) -> (Vec<C>, usize, f64) {
    let n = b.len();
    let mut x = vec![C::default(); n];
    let mut r = b.to_vec();
    let bn = norm(b);
    let z = m.apply(&r);
    let mut p = z.clone();
    let mut rho = dotu(&r, &z);
    let mut q = vec![C::default(); n];
    let mut its = 0;
    while its < maxiter {
        k.apply(&p, &mut q, None);
        let pq = dotu(&p, &q);
        if pq == C::default() {
            break;
        }
        let alpha = rho / pq;
        map_mut2(&mut x, &mut r, |i, xi, ri| {
            *xi += alpha * p[i];
            *ri -= alpha * q[i];
        });
        its += 1;
        if norm(&r) / bn < rtol {
            break;
        }
        let z = m.apply(&r);
        let rho_new = dotu(&r, &z);
        let beta = rho_new / rho;
        map_mut(&mut p, |i, pi| *pi = z[i] + beta * *pi);
        rho = rho_new;
    }
    let rr = norm(&k.residual(b, &x)) / bn;
    (x, its, rr)
}

/// An operator and preconditioner acting on `k` interleaved columns at once (x[i * k + c]).
pub trait BlockOp: Sync {
    fn n(&self) -> usize;
    /// Y = K X
    fn mul_k(&self, x: &[C], y: &mut [C], k: usize);
    /// Z = M R
    fn prec_k(&self, r: &[C], k: usize) -> Vec<C>;
}

/// Interleave columns: out[i * k + c] = cols[c][i].
pub fn pack(cols: &[&[C]]) -> Vec<C> {
    let k = cols.len();
    let n = cols[0].len();
    let mut out = vec![C::default(); n * k];
    if k == 1 {
        out.copy_from_slice(cols[0]);
        return out;
    }
    map_mut(&mut out, |i, v| *v = cols[i % k][i / k]);
    out
}

/// Column c of an interleaved block, written into `col`.
pub fn unpack(x: &[C], k: usize, c: usize, col: &mut [C]) {
    if k == 1 {
        col.copy_from_slice(x);
        return;
    }
    map_mut(col, |i, v| *v = x[i * k + c]);
}

/// COCG on every column of `b` in lockstep (at most `crate::precond::MAX_COLS` columns): one
/// operator and one preconditioner application per iteration for all still active columns. Each
/// column keeps its own recurrence, so it converges exactly as if solved alone. `x0` (per column)
/// is used where it beats zero. Returns (x, iterations, true relative residual) per column.
pub fn cocg_block<O: BlockOp>(op: &O, b: &[&[C]], x0: Option<&[&[C]]>, rtol: f64, maxiter: usize) -> (Vec<Vec<C>>, Vec<usize>, Vec<f64>) {
    let k = b.len();
    let n = op.n();
    let bn: Vec<f64> = b.iter().map(|c| norm(c)).collect();
    let mut x: Vec<Vec<C>> = vec![vec![C::default(); n]; k];
    let mut r: Vec<Vec<C>> = b.iter().map(|c| c.to_vec()).collect();
    if let Some(x0) = x0 {
        let mut ax = vec![C::default(); n * k];
        op.mul_k(&pack(x0), &mut ax, k);
        let mut col = vec![C::default(); n];
        for c in 0..k {
            unpack(&ax, k, c, &mut col);
            let rc = build(n, |i| b[c][i] - col[i]);
            if norm(&rc) < X0_GAIN * bn[c] {
                x[c].copy_from_slice(x0[c]);
                r[c] = rc;
            }
        }
    }
    let mut its = vec![0usize; k];
    let mut active: Vec<usize> = (0..k).filter(|&c| bn[c] > 0.0 && norm(&r[c]) / bn[c] >= rtol).collect();
    let mut p: Vec<Vec<C>> = vec![Vec::new(); k];
    let mut rho = vec![C::default(); k];
    let mut q = vec![C::default(); n];
    if !active.is_empty() {
        let z = op.prec_k(&pack(&active.iter().map(|&c| &r[c][..]).collect::<Vec<_>>()), active.len());
        for (a, &c) in active.iter().enumerate() {
            let mut zc = vec![C::default(); n];
            unpack(&z, active.len(), a, &mut zc);
            rho[c] = dotu(&r[c], &zc);
            p[c] = zc;
        }
    }
    while !active.is_empty() {
        let ka = active.len();
        let mut qk = vec![C::default(); n * ka];
        op.mul_k(&pack(&active.iter().map(|&c| &p[c][..]).collect::<Vec<_>>()), &mut qk, ka);
        let mut still = Vec::with_capacity(ka);
        for (a, &c) in active.iter().enumerate() {
            unpack(&qk, ka, a, &mut q);
            let pq = dotu(&p[c], &q);
            if pq == C::default() {
                continue;
            }
            let alpha = rho[c] / pq;
            let pc = &p[c];
            map_mut2(&mut x[c], &mut r[c], |i, xi, ri| {
                *xi += alpha * pc[i];
                *ri -= alpha * q[i];
            });
            its[c] += 1;
            if norm(&r[c]) / bn[c] >= rtol && its[c] < maxiter {
                still.push(c);
            }
        }
        active = still;
        if active.is_empty() {
            break;
        }
        let ka = active.len();
        let z = op.prec_k(&pack(&active.iter().map(|&c| &r[c][..]).collect::<Vec<_>>()), ka);
        for (a, &c) in active.iter().enumerate() {
            unpack(&z, ka, a, &mut q);
            let rho_new = dotu(&r[c], &q);
            let beta = rho_new / rho[c];
            let zc = &q;
            map_mut(&mut p[c], |i, pi| *pi = zc[i] + beta * *pi);
            rho[c] = rho_new;
        }
    }
    // true residuals, all columns in one product
    let mut ax = vec![C::default(); n * k];
    op.mul_k(&pack(&x.iter().map(|c| &c[..]).collect::<Vec<_>>()), &mut ax, k);
    let mut col = vec![C::default(); n];
    let res = (0..k)
        .map(|c| {
            if bn[c] == 0.0 {
                return 0.0;
            }
            unpack(&ax, k, c, &mut col);
            norm(&build(n, |i| b[c][i] - col[i])) / bn[c]
        })
        .collect();
    (x, its, res)
}
