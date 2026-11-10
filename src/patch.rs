//! Vertex-patch (Arnold-Falk-Winther) additive Schwarz: for every mesh vertex,
//! the DOFs on its incident edges and faces, solved exactly. z = sum_v R_v^T A_v^-1 R_v r.
//!
//! A is complex symmetric, so every A_v is too. Each block is stored either as
//!   - a sparse LDL^T factor after a minimum-degree ordering (default): only the
//!     factor's nonzeros, about half the memory of the dense inverse, or
//!   - the packed upper triangle of its dense inverse (fallback when a pivot of the
//!     unpivoted LDL^T is tiny, or when sparse storage is switched off).

use crate::csr::CMat;
use num_complex::Complex64 as C;
use rayon::prelude::*;
use std::sync::Mutex;

/// Block entries per parallel task in `apply`.
const ENTRIES_PER_TASK: usize = 16384;
/// Unknowns per parallel task in the gather of `apply`.
const ROWS_PER_TASK: usize = 4096;

enum Block {
    /// packed upper triangle of the inverse, row by row
    Dense(Vec<C>),
    /// P A P^T = L D L^T, L unit lower triangular stored by columns (strictly lower part)
    Ldl { perm: Vec<u16>, colptr: Vec<u32>, rows: Vec<u16>, vals: Vec<C>, dinv: Vec<C> },
}

/// The blocks plus the bookkeeping that lets `apply` run without locks or allocations per block:
/// every block writes its local solution into its own range of one flat buffer (`off`), and every
/// unknown then sums its entries of that buffer (`tptr` / `tpos`, the transpose of the block lists).
pub struct Schwarz {
    n: usize,
    patches: Vec<(Vec<u32>, Block)>,
    /// block p owns buffer entries off[p]..off[p + 1]
    off: Vec<usize>,
    /// block index boundaries of the parallel tasks, cut at about ENTRIES_PER_TASK entries
    tasks: Vec<usize>,
    /// buffer entries of unknown d: tpos[tptr[d]..tptr[d + 1]]
    tptr: Vec<usize>,
    tpos: Vec<usize>,
    /// the flat buffer, kept between applications
    buf: Mutex<Vec<C>>,
    /// blocks stored as sparse LDL^T where possible
    sparse: bool,
}

/// Dense inverse by Gauss-Jordan with partial pivoting (row-major, m x m).
fn invert(mut a: Vec<C>, m: usize) -> Result<Vec<C>, String> {
    let mut inv = vec![C::default(); m * m];
    for i in 0..m {
        inv[i * m + i] = C::new(1.0, 0.0);
    }
    for k in 0..m {
        let p = (k..m).max_by(|&i, &j| a[i * m + k].norm().partial_cmp(&a[j * m + k].norm()).unwrap()).unwrap();
        if a[p * m + k].norm() == 0.0 {
            return Err("singular patch matrix".into());
        }
        if p != k {
            for j in 0..m {
                a.swap(p * m + j, k * m + j);
                inv.swap(p * m + j, k * m + j);
            }
        }
        let piv = 1.0 / a[k * m + k];
        for j in 0..m {
            a[k * m + j] *= piv;
            inv[k * m + j] *= piv;
        }
        for i in 0..m {
            if i != k {
                let f = a[i * m + k];
                if f != C::default() {
                    for j in 0..m {
                        let (akj, ikj) = (a[k * m + j], inv[k * m + j]);
                        a[i * m + j] -= f * akj;
                        inv[i * m + j] -= f * ikj;
                    }
                }
            }
        }
    }
    Ok(inv)
}

/// Upper triangle of a symmetric m x m row-major matrix, packed by rows,
/// symmetrised (averaging the two triangles removes rounding asymmetry).
fn pack_upper(full: &[C], m: usize) -> Vec<C> {
    let mut out = Vec::with_capacity(m * (m + 1) / 2);
    for i in 0..m {
        for j in i..m {
            out.push((full[i * m + j] + full[j * m + i]) * 0.5);
        }
    }
    out
}

/// Greedy minimum-degree ordering of the symmetric pattern of `a` (bitset adjacency).
fn min_degree(a: &[C], m: usize) -> Vec<usize> {
    let w = (m + 63) / 64;
    let mut adj = vec![0u64; m * w];
    for i in 0..m {
        for j in 0..m {
            if i != j && (a[i * m + j] != C::default() || a[j * m + i] != C::default()) {
                adj[i * w + j / 64] |= 1 << (j % 64);
            }
        }
    }
    let mut alive = vec![0u64; w];
    for i in 0..m {
        alive[i / 64] |= 1 << (i % 64);
    }
    let mut order = Vec::with_capacity(m);
    for _ in 0..m {
        let mut best = (usize::MAX, 0);
        for i in 0..m {
            if alive[i / 64] >> (i % 64) & 1 == 1 {
                let deg: u32 = (0..w).map(|k| (adj[i * w + k] & alive[k]).count_ones()).sum();
                if (deg as usize) < best.0 {
                    best = (deg as usize, i);
                }
            }
        }
        let v = best.1;
        alive[v / 64] &= !(1 << (v % 64));
        // neighbours of v become a clique
        let nb: Vec<u64> = (0..w).map(|k| adj[v * w + k] & alive[k]).collect();
        for i in 0..m {
            if nb[i / 64] >> (i % 64) & 1 == 1 {
                for k in 0..w {
                    adj[i * w + k] |= nb[k];
                }
                adj[i * w + i / 64] &= !(1 << (i % 64));
            }
        }
        order.push(v);
    }
    order
}

/// Unpivoted LDL^T of the permuted block; None if a pivot is tiny (caller falls back to dense).
fn ldl(a: &[C], m: usize) -> Option<Block> {
    if m > u16::MAX as usize {
        return None;
    }
    ldl_perm(a, m, min_degree(a, m))
}

/// LDL^T with a given ordering (reused across frequencies: the block pattern does not change).
fn ldl_perm(a: &[C], m: usize, perm: Vec<usize>) -> Option<Block> {
    let mut b = vec![C::default(); m * m]; // lower triangle of P A P^T, row-major
    for i in 0..m {
        for j in 0..=i {
            b[i * m + j] = (a[perm[i] * m + perm[j]] + a[perm[j] * m + perm[i]]) * 0.5;
        }
    }
    let scale = (0..m).map(|i| b[i * m + i].norm()).fold(0.0f64, f64::max).max(1e-300);
    let mut d = vec![C::default(); m];
    for k in 0..m {
        let dk = b[k * m + k];
        if dk.norm() < 1e-10 * scale {
            return None;
        }
        d[k] = dk;
        let inv = 1.0 / dk;
        // column k below the pivot, before scaling: b_ij -= (b_ik / d_k) b_jk for k < j <= i
        let col: Vec<(usize, C)> = (k + 1..m).filter_map(|i| {
            let v = b[i * m + k];
            (v != C::default()).then_some((i, v))
        }).collect();
        for (a_i, &(i, bik)) in col.iter().enumerate() {
            let l = bik * inv;
            for &(j, bjk) in &col[..=a_i] {
                b[i * m + j] -= l * bjk;
            }
            b[i * m + k] = l;
        }
    }
    let mut colptr = Vec::with_capacity(m + 1);
    let (mut rows, mut vals) = (Vec::new(), Vec::new());
    colptr.push(0u32);
    for k in 0..m {
        for i in k + 1..m {
            let l = b[i * m + k];
            if l != C::default() {
                rows.push(i as u16);
                vals.push(l);
            }
        }
        colptr.push(rows.len() as u32);
    }
    Some(Block::Ldl {
        perm: perm.iter().map(|&p| p as u16).collect(),
        colptr,
        rows,
        vals,
        dinv: d.iter().map(|&x| 1.0 / x).collect(),
    })
}

/// Scratch space of one task, reused across its blocks.
#[derive(Default)]
struct Work {
    c: Vec<C>,
}

impl Block {
    fn bytes(&self) -> usize {
        match self {
            Block::Dense(up) => up.len() * 16,
            Block::Ldl { perm, colptr, rows, vals, dinv } => perm.len() * 2 + colptr.len() * 4 + rows.len() * 2 + vals.len() * 16 + dinv.len() * 16,
        }
    }

    /// OUT = A_v^-1 R[dofs] for K interleaved columns (R[d * K + c], OUT[i * K + c]): every factor
    /// entry is read once for all K columns.
    fn solve_into_k<const K: usize>(&self, r: &[C], dofs: &[u32], out: &mut [C], work: &mut Work) {
        let m = dofs.len();
        match self {
            Block::Dense(up) => {
                let work = &mut work.c;
                work.clear();
                for &d in dofs {
                    work.extend_from_slice(&r[d as usize * K..d as usize * K + K]);
                }
                out.fill(C::default());
                let mut p = 0;
                for i in 0..m {
                    let a = up[p];
                    let mut zi = [C::default(); K];
                    for c in 0..K {
                        zi[c] = a * work[i * K + c];
                    }
                    for j in i + 1..m {
                        let a = up[p + j - i];
                        for c in 0..K {
                            zi[c] += a * work[j * K + c];
                            out[j * K + c] += a * work[i * K + c];
                        }
                    }
                    for c in 0..K {
                        out[i * K + c] += zi[c];
                    }
                    p += m - i;
                }
            }
            Block::Ldl { perm, colptr, rows, vals, dinv } => {
                let work = &mut work.c;
                work.clear();
                for &p in perm.iter() {
                    let d = dofs[p as usize] as usize * K;
                    work.extend_from_slice(&r[d..d + K]);
                }
                let y = &mut work[..];
                for k in 0..m {
                    let mut yk = [C::default(); K];
                    yk.copy_from_slice(&y[k * K..k * K + K]);
                    for p in colptr[k] as usize..colptr[k + 1] as usize {
                        let (l, i) = (vals[p], rows[p] as usize * K);
                        for c in 0..K {
                            y[i + c] -= l * yk[c];
                        }
                    }
                }
                for k in 0..m {
                    for c in 0..K {
                        y[k * K + c] *= dinv[k];
                    }
                }
                for k in (0..m).rev() {
                    let mut s = [C::default(); K];
                    s.copy_from_slice(&y[k * K..k * K + K]);
                    for p in colptr[k] as usize..colptr[k + 1] as usize {
                        let (l, i) = (vals[p], rows[p] as usize * K);
                        for c in 0..K {
                            s[c] -= l * y[i + c];
                        }
                    }
                    y[k * K..k * K + K].copy_from_slice(&s);
                }
                for (k, &p) in perm.iter().enumerate() {
                    out[p as usize * K..p as usize * K + K].copy_from_slice(&y[k * K..k * K + K]);
                }
            }
        }
    }
}

/// The block of A on `dofs`, as a sparse LDL^T (with `perm` if given, else a fresh minimum-degree
/// ordering) or, if that fails or `sparse` is off, the packed dense inverse.
fn factor_block<A: CMat>(a: &A, dofs: &[u32], sparse: bool, perm: Option<Vec<usize>>) -> Result<Block, String> {
    let m = dofs.len();
    // dofs ascend (block lists are built sorted), so local positions come from a binary search
    let sorted = dofs.windows(2).all(|w| w[0] < w[1]);
    let mut pos = std::collections::HashMap::new();
    if !sorted {
        for (k, &d) in dofs.iter().enumerate() {
            pos.insert(d, k);
        }
    }
    let mut loc = vec![C::default(); m * m];
    for (k, &d) in dofs.iter().enumerate() {
        a.for_row(d as usize, &mut |j, v| {
            let c = if sorted { dofs.binary_search(&j).ok() } else { pos.get(&j).copied() };
            if let Some(c) = c {
                loc[k * m + c] = v;
            }
        });
    }
    let ldl_block = if !sparse || m > u16::MAX as usize {
        None
    } else {
        match perm {
            Some(p) if p.len() == m => ldl_perm(&loc, m, p),
            _ => ldl(&loc, m),
        }
    };
    Ok(match ldl_block {
        Some(b) => b,
        None => Block::Dense(pack_upper(&invert(loc, m)?, m)),
    })
}

impl Schwarz {
    pub fn new<A: CMat>(a: &A, patches: Vec<Vec<u32>>) -> Result<Self, String> {
        Self::with_storage(a, patches, false)
    }

    /// `sparse`: store blocks as sparse LDL^T factors where possible (about half the memory).
    pub fn with_storage<A: CMat>(a: &A, patches: Vec<Vec<u32>>, sparse: bool) -> Result<Self, String> {
        let n = a.nrows();
        let built: Result<Vec<_>, String> = patches
            .into_par_iter()
            .map(|dofs| {
                let block = factor_block(a, &dofs, sparse, None)?;
                Ok((dofs, block))
            })
            .collect();
        let patches = built?;
        let mut s = Self::index(n, patches);
        s.sparse = sparse;
        Ok(s)
    }

    /// New values of A on the same pattern: every block is factored again, reusing its ordering.
    pub fn refactor<A: CMat>(&mut self, a: &A) -> Result<(), String> {
        let sparse = self.sparse;
        self.patches.par_iter_mut().try_for_each(|(dofs, block)| {
            let perm = match block {
                Block::Ldl { perm, .. } => Some(perm.iter().map(|&p| p as usize).collect()),
                Block::Dense(_) => None,
            };
            *block = factor_block(a, dofs, sparse, perm)?;
            Ok::<(), String>(())
        })
    }

    /// Buffer offsets, task boundaries and the per-unknown gather lists.
    fn index(n: usize, patches: Vec<(Vec<u32>, Block)>) -> Self {
        let mut off = Vec::with_capacity(patches.len() + 1);
        off.push(0usize);
        let mut tasks = vec![0usize];
        let mut count = vec![0usize; n + 1];
        for (p, (dofs, _)) in patches.iter().enumerate() {
            off.push(off[p] + dofs.len());
            if off[p + 1] - off[*tasks.last().unwrap()] >= ENTRIES_PER_TASK {
                tasks.push(p + 1);
            }
            for &d in dofs {
                count[d as usize + 1] += 1;
            }
        }
        if *tasks.last().unwrap() != patches.len() {
            tasks.push(patches.len());
        }
        for d in 0..n {
            count[d + 1] += count[d];
        }
        let tptr = count.clone();
        let mut tpos = vec![0usize; *off.last().unwrap()];
        for (p, (dofs, _)) in patches.iter().enumerate() {
            for (k, &d) in dofs.iter().enumerate() {
                tpos[count[d as usize]] = off[p] + k;
                count[d as usize] += 1;
            }
        }
        Schwarz { n, patches, off, tasks, tptr, tpos, buf: Mutex::new(Vec::new()), sparse: false }
    }

    pub fn memory_bytes(&self) -> usize {
        self.patches.iter().map(|(d, b)| d.len() * 4 + b.bytes()).sum::<usize>()
            + (self.off.len() + self.tptr.len() + self.tpos.len()) * 8
            + self.off.last().copied().unwrap_or(0) * 16
    }

    pub fn dense_fallbacks(&self) -> usize {
        self.patches.iter().filter(|(_, b)| matches!(b, Block::Dense(_))).count()
    }

    pub fn apply(&self, r: &[C]) -> Vec<C> {
        self.apply_k::<1>(r)
    }

    /// z = sum_v R_v^T A_v^-1 R_v r for K interleaved columns.
    pub fn apply_k<const K: usize>(&self, r: &[C]) -> Vec<C> {
        let total = *self.off.last().unwrap() * K;
        // the kept buffer; a concurrent caller (none today) gets its own
        let mut guard = self.buf.try_lock().ok();
        let mut own = Vec::new();
        let buf: &mut Vec<C> = match guard.as_deref_mut() {
            Some(b) => b,
            None => &mut own,
        };
        buf.resize(total, C::default());
        // one disjoint buffer slice per task
        let mut slices = Vec::with_capacity(self.tasks.len());
        let mut rest = &mut buf[..];
        for t in self.tasks.windows(2) {
            let (head, tail) = rest.split_at_mut((self.off[t[1]] - self.off[t[0]]) * K);
            slices.push((t[0], t[1], head));
            rest = tail;
        }
        slices.into_par_iter().for_each(|(p0, p1, out)| {
            let base = self.off[p0];
            let mut work = Work::default();
            for p in p0..p1 {
                let (dofs, block) = &self.patches[p];
                let o = &mut out[(self.off[p] - base) * K..(self.off[p + 1] - base) * K];
                block.solve_into_k::<K>(r, dofs, o, &mut work);
            }
        });
        let buf = &buf[..];
        let mut z = vec![C::default(); self.n * K];
        z.par_chunks_mut(ROWS_PER_TASK * K).enumerate().for_each(|(c, zs)| {
            let d0 = c * ROWS_PER_TASK;
            for (k, zd) in zs.chunks_exact_mut(K).enumerate() {
                let d = d0 + k;
                let mut s = [C::default(); K];
                for &q in &self.tpos[self.tptr[d]..self.tptr[d + 1]] {
                    for c in 0..K {
                        s[c] += buf[q * K + c];
                    }
                }
                zd.copy_from_slice(&s);
            }
        });
        z
    }
}
