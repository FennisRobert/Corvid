//! Element-agglomeration AMGe for the Whitney level, with wavelength-capped aggregates: the Rust port of
//! corvid/amge (whitney.py, agglomerate.py, coarsen.py, capped.py). Same construction, same checks:
//!
//!   fine level     lowest-order de Rham complex of the tet mesh (incidences, element / face masses)
//!   aggregation    region growing into topological balls that never cross PEC, refused once an
//!                  aggregate's k_loc * bounding-box diagonal would exceed kappa
//!   coarse topology  AE (balls), AF (disks, cut along every AF boundary), AR (chains between APs)
//!   coarse spaces  parELAG's harmonic extensions: L2 -> H(div) -> H(curl), each a local saddle point
//!                  whose multiplier must equal the coarse incidence (so the coarse level is a complex)
//!
//! Output: per level the prolongation between free edges and the gradient on free edges.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Phase timer, printed to stderr when CORVID_AMGE_TIMING is set.
struct Clock(Option<Instant>);
impl Clock {
    fn new() -> Self {
        Clock(std::env::var_os("CORVID_AMGE_TIMING").map(|_| Instant::now()))
    }
    fn lap(&mut self, what: &str) {
        if let Some(t) = self.0 {
            let (ns, flops, nmax) = saddle_stats();
            let extra = if ns > 0 { format!("   ({ns} saddle systems, sum n^3 = {flops}M, largest n = {nmax}){}", sub_report()) } else { String::new() };
            eprintln!("  amge {what:<28} {:8.3}s{extra}", t.elapsed().as_secs_f64());
            self.0 = Some(Instant::now());
        }
    }
}

// ------------------------------------------------------------------ small sparse / dense helpers

/// CSR with f64 values; rows hold sorted column indices.
#[derive(Clone, Default)]
pub struct Sp {
    pub nrows: usize,
    pub ncols: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

impl Sp {
    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[i], self.ptr[i + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }

    pub fn cols(&self, i: usize) -> &[usize] {
        &self.idx[self.ptr[i]..self.ptr[i + 1]]
    }

    /// From per-row (col, value) lists; duplicate columns are summed, zeros kept.
    pub fn from_rows(ncols: usize, rows: Vec<Vec<(usize, f64)>>) -> Sp {
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        ptr.push(0);
        for mut r in rows {
            r.sort_by_key(|x| x.0);
            let mut last: Option<usize> = None;
            for (c, v) in r {
                if last == Some(c) {
                    *val.last_mut().unwrap() += v;
                } else {
                    idx.push(c);
                    val.push(v);
                    last = Some(c);
                }
            }
            ptr.push(idx.len());
        }
        Sp { nrows: ptr.len() - 1, ncols, ptr, idx, val }
    }

    pub fn transpose(&self) -> Sp {
        let mut ptr = vec![0usize; self.ncols + 1];
        for &j in &self.idx {
            ptr[j + 1] += 1;
        }
        for j in 0..self.ncols {
            ptr[j + 1] += ptr[j];
        }
        let mut next = ptr.clone();
        let mut idx = vec![0usize; self.idx.len()];
        let mut val = vec![0.0; self.idx.len()];
        for i in 0..self.nrows {
            let (c, v) = self.row(i);
            for (&j, &x) in c.iter().zip(v) {
                idx[next[j]] = i;
                val[next[j]] = x;
                next[j] += 1;
            }
        }
        Sp { nrows: self.ncols, ncols: self.nrows, ptr, idx, val }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (c, v) = self.row(i);
        match c.binary_search(&j) {
            Ok(p) => v[p],
            Err(_) => 0.0,
        }
    }

    /// A B (for the identity checks)
    pub fn matmul(&self, b: &Sp) -> Sp {
        let rows: Vec<Vec<(usize, f64)>> = (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                let mut acc: HashMap<usize, f64> = HashMap::new();
                let (c, v) = self.row(i);
                for (&k, &a) in c.iter().zip(v) {
                    let (c2, v2) = b.row(k);
                    for (&j, &x) in c2.iter().zip(v2) {
                        *acc.entry(j).or_insert(0.0) += a * x;
                    }
                }
                acc.into_iter().collect()
            })
            .collect();
        Sp::from_rows(b.ncols, rows)
    }

    pub fn max_abs_diff(&self, b: &Sp) -> f64 {
        let mut m: f64 = 0.0;
        for i in 0..self.nrows {
            let mut acc: HashMap<usize, f64> = HashMap::new();
            let (c, v) = self.row(i);
            for (&j, &x) in c.iter().zip(v) {
                *acc.entry(j).or_insert(0.0) += x;
            }
            let (c, v) = b.row(i);
            for (&j, &x) in c.iter().zip(v) {
                *acc.entry(j).or_insert(0.0) -= x;
            }
            m = acc.values().fold(m, |a, &x| a.max(x.abs()));
        }
        m
    }

    /// Rows `r` and columns `c` (each a selection mask), renumbered in increasing order.
    pub fn select(&self, r: &[bool], c: &[bool]) -> Sp {
        let mut cmap = vec![usize::MAX; self.ncols];
        let mut nc = 0;
        for (j, &k) in c.iter().enumerate() {
            if k {
                cmap[j] = nc;
                nc += 1;
            }
        }
        let rows = (0..self.nrows)
            .filter(|&i| r[i])
            .map(|i| {
                let (cc, v) = self.row(i);
                cc.iter().zip(v).filter(|(&j, _)| cmap[j] != usize::MAX).map(|(&j, &x)| (cmap[j], x)).collect()
            })
            .collect();
        Sp::from_rows(nc, rows)
    }
}

/// Row-major dense matrix.
#[derive(Clone)]
struct Mat {
    r: usize,
    c: usize,
    d: Vec<f64>,
}

impl Mat {
    fn zeros(r: usize, c: usize) -> Mat {
        Mat { r, c, d: vec![0.0; r * c] }
    }
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.d[i * self.c + j]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.c + j] = v;
    }
    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        self.d[i * self.c + j] += v;
    }
    fn matmul(&self, b: &Mat) -> Mat {
        let mut o = Mat::zeros(self.r, b.c);
        if b.c == 0 {
            return o;
        }
        for (arow, orow) in self.d.chunks_exact(self.c.max(1)).zip(o.d.chunks_exact_mut(b.c)) {
            for (&a, brow) in arow.iter().zip(b.d.chunks_exact(b.c)) {
                if a != 0.0 {
                    orow.iter_mut().zip(brow).for_each(|(o, &x)| *o += a * x);
                }
            }
        }
        o
    }
    /// self^T b without forming the transpose
    fn tmatmul(&self, b: &Mat) -> Mat {
        let mut o = Mat::zeros(self.c, b.c);
        if b.c == 0 || self.c == 0 {
            return o;
        }
        for (arow, brow) in self.d.chunks_exact(self.c).zip(b.d.chunks_exact(b.c)) {
            for (&a, orow) in arow.iter().zip(o.d.chunks_exact_mut(b.c)) {
                if a != 0.0 {
                    orow.iter_mut().zip(brow).for_each(|(o, &x)| *o += a * x);
                }
            }
        }
        o
    }
    fn t(&self) -> Mat {
        let mut o = Mat::zeros(self.c, self.r);
        for i in 0..self.r {
            for j in 0..self.c {
                o.d[j * self.r + i] = self.d[i * self.c + j];
            }
        }
        o
    }
    /// P^T A P; both products skip zeros (of A, then of P)
    fn ptap(a: &Mat, p: &Mat) -> Mat {
        p.tmatmul(&a.matmul(p))
    }
}

/// Solve K X = B (K n x n, B n x m) by LU with partial pivoting; Err when singular.
fn solve_dense(mut k: Mat, mut b: Mat) -> Result<Mat, String> {
    let n = k.r;
    let m = b.c;
    if m == 0 {
        return Ok(b);
    }
    for col in 0..n {
        let mut p = col;
        let mut best = k.d[col * n + col].abs();
        for i in col + 1..n {
            let v = k.d[i * n + col].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best == 0.0 {
            return Err("singular local saddle point".into());
        }
        if p != col {
            let (a, c) = k.d.split_at_mut(p * n);
            a[col * n..col * n + n].swap_with_slice(&mut c[..n]);
            if m > 0 {
                let (a, c) = b.d.split_at_mut(p * m);
                a[col * m..col * m + m].swap_with_slice(&mut c[..m]);
            }
        }
        let piv = k.d[col * n + col];
        let (top, rest) = k.d.split_at_mut((col + 1) * n);
        let prow = &top[col * n + col..col * n + n];
        let (btop, brest) = b.d.split_at_mut((col + 1) * m);
        let bprow = &btop[col * m..col * m + m];
        for (krow, brow) in rest.chunks_exact_mut(n).zip(brest.chunks_exact_mut(m.max(1))) {
            let f = krow[col] / piv;
            if f == 0.0 {
                continue;
            }
            krow[col..].iter_mut().zip(prow).for_each(|(x, &y)| *x -= f * y);
            if m > 0 {
                brow.iter_mut().zip(bprow).for_each(|(x, &y)| *x -= f * y);
            }
        }
    }
    // back substitution, row by row on all right-hand sides at once
    let mut s = vec![0.0; m];
    for col in (0..n).rev() {
        s.copy_from_slice(&b.d[col * m..col * m + m]);
        for i in col + 1..n {
            let kci = k.d[col * n + i];
            if kci != 0.0 {
                let bi = &b.d[i * m..i * m + m];
                s.iter_mut().zip(bi).for_each(|(x, &y)| *x -= kci * y);
            }
        }
        let piv = k.d[col * n + col];
        b.d[col * m..col * m + m].iter_mut().zip(&s).for_each(|(x, &y)| *x = y / piv);
    }
    Ok(b)
}

fn sorted_unique(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v.dedup();
    v
}

/// Position lookup in a sorted, duplicate-free index list (binary search, no hashing).
struct LocalMap<'a>(&'a [usize]);
impl LocalMap<'_> {
    /// position of x, which must be in the list
    #[inline]
    fn at(&self, x: usize) -> usize {
        self.0.binary_search(&x).expect("index not in the local list")
    }
    #[inline]
    fn find(&self, x: usize) -> Option<usize> {
        self.0.binary_search(&x).ok()
    }
}

fn local_map(ids: &[usize]) -> LocalMap<'_> {
    debug_assert!(ids.windows(2).all(|w| w[0] < w[1]), "local_map needs a sorted, duplicate-free list");
    LocalMap(ids)
}

/// Sorted a minus sorted b
fn setdiff(a: &[usize], b: &[usize]) -> Vec<usize> {
    let bs: HashSet<usize> = b.iter().copied().collect();
    a.iter().copied().filter(|x| !bs.contains(x)).collect()
}

// ------------------------------------------------------------------ levels

/// One lowest-order complex on a polyhedral mesh (fine tets and every agglomerated level).
pub struct Level {
    pub d0: Sp, // edge x vertex
    pub d1: Sp, // face x edge
    pub d2: Sp, // element x face
    elem_edges: Vec<Vec<usize>>,
    elem_m1: Vec<Mat>,
    elem_faces: Vec<Vec<usize>>,
    elem_m2: Vec<Mat>,
    elem_m3: Vec<f64>,
    face_edges: Vec<Vec<usize>>,
    face_m1: Vec<Mat>,
    face_m2: Vec<f64>,
    w_edge: Vec<f64>,
    w_face: Vec<f64>,
    w_elem: Vec<f64>,
    pub free_edge: Vec<bool>,
}

impl Level {
    pub fn n_edge(&self) -> usize {
        self.d0.nrows
    }
    pub fn n_face(&self) -> usize {
        self.d1.nrows
    }
    pub fn n_elem(&self) -> usize {
        self.d2.nrows
    }
    pub fn n_free(&self) -> usize {
        self.free_edge.iter().filter(|&&f| f).count()
    }

    /// Gradient on the free edges, columns: the vertices that touch one (renumbered).
    pub fn grad(&self) -> Sp {
        let rows: Vec<bool> = self.free_edge.clone();
        let mut touch = vec![false; self.d0.ncols];
        for e in 0..self.n_edge() {
            if self.free_edge[e] {
                for &v in self.d0.cols(e) {
                    touch[v] = true;
                }
            }
        }
        // a vertex column is kept when it has a nonzero in a free row
        let g = self.d0.select(&rows, &vec![true; self.d0.ncols]);
        let mut nz = vec![false; self.d0.ncols];
        for i in 0..g.nrows {
            let (c, v) = g.row(i);
            for (&j, &x) in c.iter().zip(v) {
                if x != 0.0 {
                    nz[j] = true;
                }
            }
        }
        let keep: Vec<bool> = (0..self.d0.ncols).map(|j| touch[j] && nz[j]).collect();
        g.select(&vec![true; g.nrows], &keep)
    }
}

const TET_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn ekey(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Closed-form gradients of the barycentric coordinates of a tet (rows: vertices) and its volume.
fn bary_grads(xt: &[[f64; 3]; 4]) -> ([[f64; 3]; 4], f64) {
    let d1 = sub3(xt[1], xt[0]);
    let d2 = sub3(xt[2], xt[0]);
    let d3 = sub3(xt[3], xt[0]);
    let v6 = dot3(cross3(d1, d2), d3);
    // grad l_1 = (d2 x d3) / v6, grad l_2 = (d3 x d1) / v6, grad l_3 = (d1 x d2) / v6, grad l_0 = -sum
    let g1 = cross3(d2, d3).map(|v| v / v6);
    let g2 = cross3(d3, d1).map(|v| v / v6);
    let g3 = cross3(d1, d2).map(|v| v / v6);
    let g0 = [-(g1[0] + g2[0] + g3[0]), -(g1[1] + g2[1] + g3[1]), -(g1[2] + g2[2] + g3[2])];
    ([g0, g1, g2, g3], v6.abs() / 6.0)
}

/// Sorted (key, index) table with binary-search lookup (no hashing of millions of keys).
struct Table<K: Ord + Copy + Send + Sync> {
    keys: Vec<K>,
    ids: Vec<usize>,
}

impl<K: Ord + Copy + Send + Sync> Table<K> {
    fn new(keys: Vec<K>) -> Self {
        let mut kv: Vec<(K, usize)> = keys.into_iter().enumerate().map(|(i, k)| (k, i)).collect();
        kv.par_sort_unstable_by_key(|x| x.0);
        Table { keys: kv.iter().map(|x| x.0).collect(), ids: kv.iter().map(|x| x.1).collect() }
    }
    fn get(&self, k: K) -> Option<usize> {
        self.keys.binary_search(&k).ok().map(|p| self.ids[p])
    }
}

/// The fine level (corvid/amge/whitney.py: fine_level, vacuum weights).
pub fn fine_level(x: &[[f64; 3]], tets_in: &[[usize; 4]], edges: &[[usize; 2]], free: &[bool]) -> Result<Level, String> {
    let (nv, nt, ne) = (x.len(), tets_in.len(), edges.len());
    // positively oriented tets
    let tets: Vec<[usize; 4]> = tets_in
        .par_iter()
        .map(|tt| {
            let mut tt = *tt;
            let v6 = dot3(cross3(sub3(x[tt[1]], x[tt[0]]), sub3(x[tt[2]], x[tt[0]])), sub3(x[tt[3]], x[tt[0]]));
            if v6 < 0.0 {
                tt.swap(2, 3);
            }
            tt
        })
        .collect();
    let emap = Table::new(edges.iter().map(|ab| ekey(ab[0], ab[1])).collect());
    let tet_edges: Vec<Result<([usize; 6], [f64; 6]), String>> = tets
        .par_iter()
        .map(|tt| {
            let mut e6 = [0usize; 6];
            let mut s6 = [0f64; 6];
            for (k, &(a, b)) in TET_EDGES.iter().enumerate() {
                let (la, lb) = (tt[a], tt[b]);
                let e = emap.get(ekey(la, lb)).ok_or("a tet edge is missing from the edge list")?;
                e6[k] = e;
                s6[k] = if edges[e][0] == la { 1.0 } else { -1.0 };
            }
            Ok((e6, s6))
        })
        .collect();
    let mut tet_e = Vec::with_capacity(nt);
    let mut tet_es = Vec::with_capacity(nt);
    for r in tet_edges {
        let (e, s) = r?;
        tet_e.push(e);
        tet_es.push(s);
    }
    // faces: unique sorted triples, numbered in increasing key order (as np.unique)
    let fkeys: Vec<(usize, usize, usize)> = tets
        .par_iter()
        .flat_map_iter(|tt| {
            TET_FACES.iter().map(move |f| {
                let mut s = [tt[f[0]], tt[f[1]], tt[f[2]]];
                s.sort_unstable();
                (s[0], s[1], s[2])
            })
        })
        .collect();
    let mut ukeys = fkeys.clone();
    ukeys.par_sort_unstable();
    ukeys.dedup();
    let nf = ukeys.len();
    let (tet_f, tet_fs): (Vec<[usize; 4]>, Vec<[f64; 4]>) = (0..nt)
        .into_par_iter()
        .map(|t| {
            let mut f4 = [0usize; 4];
            let mut s4 = [0f64; 4];
            for (i, f) in TET_FACES.iter().enumerate() {
                let loc = [tets[t][f[0]], tets[t][f[1]], tets[t][f[2]]];
                f4[i] = ukeys.binary_search(&fkeys[4 * t + i]).unwrap();
                // parity of the permutation sorting loc
                let inv = (loc[0] > loc[1]) as i32 + (loc[0] > loc[2]) as i32 + (loc[1] > loc[2]) as i32;
                s4[i] = if inv % 2 == 0 { 1.0 } else { -1.0 } * if i % 2 == 0 { 1.0 } else { -1.0 };
            }
            (f4, s4)
        })
        .unzip();
    drop(fkeys);
    // incidences
    let d0 = Sp::from_rows(nv, edges.iter().map(|ab| vec![(ab[0], -1.0), (ab[1], 1.0)]).collect());
    let face_rows: Vec<([usize; 3], [f64; 3])> = ukeys
        .par_iter()
        .map(|&(fa, fb, fc)| {
            let es = [emap.get((fa, fb)).unwrap(), emap.get((fb, fc)).unwrap(), emap.get((fa, fc)).unwrap()];
            let starts = [fa, fb, fa];
            let s: [f64; 3] = std::array::from_fn(|k| if edges[es[k]][0] == starts[k] { 1.0 } else { -1.0 });
            (es, s)
        })
        .collect();
    let d1 = Sp::from_rows(ne, face_rows.iter().map(|(es, s)| vec![(es[0], s[0]), (es[1], s[1]), (es[2], -s[2])]).collect());
    let d2 = Sp::from_rows(nf, (0..nt).map(|t| (0..4).map(|i| (tet_f[t][i], tet_fs[t][i])).collect()).collect());

    // element masses
    let lam = |i: usize, j: usize| if i == j { 2.0 / 20.0 } else { 1.0 / 20.0 };
    let masses: Vec<(Mat, Mat, f64)> = (0..nt)
        .into_par_iter()
        .map(|t| {
            let xt: [[f64; 3]; 4] = std::array::from_fn(|i| x[tets[t][i]]);
            let (grad, vol) = bary_grads(&xt);
            let gg = |i: usize, j: usize| dot3(grad[i], grad[j]);
            let mut m1 = Mat::zeros(6, 6);
            for (p, &(ap, bp)) in TET_EDGES.iter().enumerate() {
                for (q, &(aq, bq)) in TET_EDGES.iter().enumerate() {
                    let v = lam(ap, aq) * gg(bp, bq) - lam(ap, bq) * gg(bp, aq) - lam(bp, aq) * gg(ap, bq) + lam(bp, bq) * gg(ap, aq);
                    m1.set(p, q, v * vol * tet_es[t][p] * tet_es[t][q]);
                }
            }
            // R[i][k] = x_k - x_i
            let mut m2 = Mat::zeros(4, 4);
            for i in 0..4 {
                for j in 0..4 {
                    let mut s = 0.0;
                    for k in 0..4 {
                        for l in 0..4 {
                            s += lam(k, l) * dot3(sub3(xt[k], xt[i]), sub3(xt[l], xt[j]));
                        }
                    }
                    m2.set(i, j, s / (9.0 * vol) * tet_fs[t][i] * tet_fs[t][j]);
                }
            }
            (m1, m2, vol)
        })
        .collect();
    let mut elem_m1 = Vec::with_capacity(nt);
    let mut elem_m2 = Vec::with_capacity(nt);
    let mut vol = Vec::with_capacity(nt);
    for (a, b, v) in masses {
        elem_m1.push(a);
        elem_m2.push(b);
        vol.push(v);
    }
    let elem_m3: Vec<f64> = vol.iter().map(|v| 1.0 / v).collect();

    // face trace masses
    let lam2 = |i: usize, j: usize| if i == j { 2.0 / 12.0 } else { 1.0 / 12.0 };
    let fm: Vec<(Mat, f64)> = (0..nf)
        .into_par_iter()
        .map(|f| {
            let (fa, fb, fc) = ukeys[f];
            let j1 = sub3(x[fb], x[fa]);
            let j2 = sub3(x[fc], x[fa]);
            let (g11, g12, g22) = (dot3(j1, j1), dot3(j1, j2), dot3(j2, j2));
            let det = g11 * g22 - g12 * g12;
            let area = 0.5 * det.sqrt();
            let (i11, i12, i22) = (g22 / det, -g12 / det, g11 / det);
            let gl1: [f64; 3] = std::array::from_fn(|d| j1[d] * i11 + j2[d] * i12);
            let gl2: [f64; 3] = std::array::from_fn(|d| j1[d] * i12 + j2[d] * i22);
            let g = [[-(gl1[0] + gl2[0]), -(gl1[1] + gl2[1]), -(gl1[2] + gl2[2])], gl1, gl2];
            let fgg = |i: usize, j: usize| dot3(g[i], g[j]);
            let (ea, eb) = ([0usize, 1, 0], [1usize, 2, 2]);
            let s = face_rows[f].1;
            let mut m = Mat::zeros(3, 3);
            for p in 0..3 {
                for q in 0..3 {
                    let v = lam2(ea[p], ea[q]) * fgg(eb[p], eb[q]) - lam2(ea[p], eb[q]) * fgg(eb[p], ea[q])
                        - lam2(eb[p], ea[q]) * fgg(ea[p], eb[q])
                        + lam2(eb[p], eb[q]) * fgg(ea[p], ea[q]);
                    m.set(p, q, v * area * s[p] * s[q]);
                }
            }
            (m, area)
        })
        .collect();
    let (face_m1, area): (Vec<Mat>, Vec<f64>) = fm.into_iter().unzip();
    let face_m2: Vec<f64> = area.iter().map(|a| 1.0 / a).collect();
    let w_edge: Vec<f64> = edges.iter().map(|ab| dot3(sub3(x[ab[1]], x[ab[0]]), sub3(x[ab[1]], x[ab[0]])).sqrt()).collect();
    Ok(Level {
        d0,
        d1,
        d2,
        elem_edges: tet_e.iter().map(|e| e.to_vec()).collect(),
        elem_m1,
        elem_faces: tet_f.iter().map(|f| f.to_vec()).collect(),
        elem_m2,
        elem_m3,
        face_edges: face_rows.iter().map(|r| r.0.to_vec()).collect(),
        face_m1,
        face_m2,
        w_edge,
        w_face: area,
        w_elem: vol,
        free_edge: free.to_vec(),
    })
}

// ------------------------------------------------------------------ topology helpers

struct Topo<'a> {
    l: &'a Level,
    d2t: Sp, // face -> elems
    d1t: Sp, // edge -> faces
    d0t: Sp, // vertex -> edges
    pec_face: Vec<bool>,
    nonfree: Vec<bool>,
}

impl<'a> Topo<'a> {
    fn new(l: &'a Level) -> Self {
        let nonfree: Vec<bool> = l.free_edge.iter().map(|&f| !f).collect();
        let pec_face = (0..l.n_face()).into_par_iter().map(|f| l.d1.cols(f).iter().all(|&e| nonfree[e])).collect();
        Topo { l, d2t: l.d2.transpose(), d1t: l.d1.transpose(), d0t: l.d0.transpose(), pec_face, nonfree }
    }

    fn elem_nbrs(&self, t: usize, cut_pec: bool) -> Vec<usize> {
        let mut out = Vec::new();
        for &f in self.l.d2.cols(t) {
            if cut_pec && self.pec_face[f] {
                continue;
            }
            for &s in self.d2t.cols(f) {
                if s != t {
                    out.push(s);
                }
            }
        }
        out
    }

    /// (V, E, F) of a set of elements, each sorted.
    fn closure(&self, elems: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let f = sorted_unique(elems.iter().flat_map(|&t| self.l.d2.cols(t).iter().copied()).collect());
        let e = sorted_unique(f.iter().flat_map(|&x| self.l.d1.cols(x).iter().copied()).collect());
        let v = sorted_unique(e.iter().flat_map(|&x| self.l.d0.cols(x).iter().copied()).collect());
        (v, e, f)
    }

    /// Faces with a nonzero signed sum over the elements (the boundary), sorted.
    fn boundary_faces(&self, elems: &[usize]) -> Vec<usize> {
        let mut s: HashMap<usize, f64> = HashMap::new();
        for &t in elems {
            let (c, v) = self.l.d2.row(t);
            for (&f, &x) in c.iter().zip(v) {
                *s.entry(f).or_insert(0.0) += x;
            }
        }
        sorted_unique(s.into_iter().filter(|(_, x)| *x != 0.0).map(|(f, _)| f).collect())
    }

    /// Boundary edges (sorted) and their signs of a signed set of faces.
    fn boundary_edges(&self, faces: &[usize], signs: &[f64]) -> (Vec<usize>, Vec<f64>) {
        let mut s: HashMap<usize, f64> = HashMap::new();
        for (&f, &g) in faces.iter().zip(signs) {
            let (c, v) = self.l.d1.row(f);
            for (&e, &x) in c.iter().zip(v) {
                *s.entry(e).or_insert(0.0) += g * x;
            }
        }
        let mut be: Vec<(usize, f64)> = s.into_iter().filter(|(_, x)| x.abs() > 0.5).collect();
        be.sort_by_key(|x| x.0);
        (be.iter().map(|x| x.0).collect(), be.iter().map(|x| x.1).collect())
    }
}

/// Connected components of `nodes` under `nbrs`.
fn components(nodes: &[usize], nbrs: &dyn Fn(usize) -> Vec<usize>) -> Vec<Vec<usize>> {
    let mut left: HashSet<usize> = nodes.iter().copied().collect();
    let mut out = Vec::new();
    for &s in nodes {
        if !left.remove(&s) {
            continue;
        }
        let mut comp = vec![s];
        let mut stack = vec![s];
        while let Some(x) = stack.pop() {
            for y in nbrs(x) {
                if left.remove(&y) {
                    comp.push(y);
                    stack.push(y);
                }
            }
        }
        out.push(comp);
    }
    out
}

/// Split a connected set in two by BFS from a peripheral node.
fn bisect(nodes: &[usize], nbrs: &dyn Fn(usize) -> Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let ns: HashSet<usize> = nodes.iter().copied().collect();
    let mut start = nodes[0];
    let mut order = Vec::new();
    for _ in 0..2 {
        order = vec![start];
        let mut seen: HashSet<usize> = HashSet::from([start]);
        let mut i = 0;
        while i < order.len() {
            for y in nbrs(order[i]) {
                if ns.contains(&y) && seen.insert(y) {
                    order.push(y);
                }
            }
            i += 1;
        }
        start = *order.last().unwrap();
    }
    let half = order.len() / 2;
    let b = order.split_off(half);
    (order, b)
}

// ------------------------------------------------------------------ aggregation

struct Cell {
    v: Vec<usize>,
    e: Vec<usize>,
    f: Vec<usize>,
}

fn cell(t: &Topo, el: usize) -> Cell {
    let f: Vec<usize> = t.l.d2.cols(el).to_vec();
    let e = sorted_unique(f.iter().flat_map(|&x| t.l.d1.cols(x).iter().copied()).collect());
    let v = sorted_unique(e.iter().flat_map(|&x| t.l.d0.cols(x).iter().copied()).collect());
    Cell { v, e, f }
}

/// The ball being grown: membership by stamp (entity -> id of the ball that holds it).
struct Ball {
    id: u32,
    v: Vec<u32>,
    e: Vec<u32>,
    f: Vec<u32>,
}

impl Ball {
    fn has_v(&self, x: usize) -> bool {
        self.v[x] == self.id
    }
    fn has_e(&self, x: usize) -> bool {
        self.e[x] == self.id
    }
    fn has_f(&self, x: usize) -> bool {
        self.f[x] == self.id
    }
    fn add(&mut self, c: &Cell) {
        let id = self.id;
        c.v.iter().for_each(|&x| self.v[x] = id);
        c.e.iter().for_each(|&x| self.e[x] = id);
        c.f.iter().for_each(|&x| self.f[x] = id);
    }
}

/// Whether ball S stays a ball when `c` is glued on (agglomerate._glue_ok).
fn glue_ok(t: &Topo, c: &Cell, s: &Ball) -> bool {
    let shared: Vec<usize> = c.f.iter().copied().filter(|&f| s.has_f(f)).collect();
    if shared.is_empty() || shared.iter().any(|&f| t.pec_face[f]) {
        return false;
    }
    // edges of the shared disk with multiplicity (1: disk boundary, 2: inside the disk)
    let mut all: Vec<usize> = shared.iter().flat_map(|&f| t.l.d1.cols(f).iter().copied()).collect();
    all.sort_unstable();
    let mut cnt: Vec<(usize, usize)> = Vec::with_capacity(all.len());
    for e in all {
        match cnt.last_mut() {
            Some(l) if l.0 == e => l.1 += 1,
            _ => cnt.push((e, 1)),
        }
    }
    // the cell may touch the ball only along that disk
    let ce: Vec<usize> = c.e.iter().copied().filter(|&e| s.has_e(e)).collect(); // sorted (c.e is)
    if ce.len() != cnt.len() || ce.iter().zip(&cnt).any(|(a, b)| *a != b.0) {
        return false;
    }
    let mut dv: Vec<usize> = cnt.iter().flat_map(|&(e, _)| t.l.d0.cols(e).iter().copied()).collect();
    dv.sort_unstable();
    dv.dedup();
    let cv: Vec<usize> = c.v.iter().copied().filter(|&v| s.has_v(v)).collect();
    if cv != dv {
        return false;
    }
    if dv.len() as i64 - cnt.len() as i64 + shared.len() as i64 != 1 {
        return false;
    }
    if cnt.iter().any(|&(e, k)| k == 2 && t.nonfree[e]) {
        return false;
    }
    // the disk boundary is one simple loop: every boundary vertex in exactly two boundary edges
    let mut bv: Vec<usize> = cnt.iter().filter(|x| x.1 == 1).flat_map(|&(e, _)| t.l.d0.cols(e).iter().copied()).collect();
    bv.sort_unstable();
    let mut i = 0;
    while i < bv.len() {
        let mut j = i;
        while j < bv.len() && bv[j] == bv[i] {
            j += 1;
        }
        if j - i != 2 {
            return false;
        }
        i = j;
    }
    // the shared faces are connected through their inner (count 2) edges: union-find
    let ns = shared.len();
    let mut parent: Vec<usize> = (0..ns).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let mut comps = ns;
    for &(e, k) in &cnt {
        if k != 2 {
            continue;
        }
        let mut owners = shared.iter().enumerate().filter(|(_, &f)| t.l.d1.cols(f).contains(&e)).map(|(i, _)| i);
        if let (Some(a), Some(b)) = (owners.next(), owners.next()) {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
                comps -= 1;
            }
        }
    }
    comps == 1
}

/// Region growing with the wavelength cap (corvid/amge/capped.py: _grow). Sequential (greedy), with
/// stamps for ball membership and candidate counts kept up to date as elements are glued on.
fn grow_capped(t: &Topo, bmin: &[[f64; 3]], bmax: &[[f64; 3]], kl: &[f64], kappa: f64, target: usize) -> Vec<usize> {
    let nel = t.l.n_elem();
    let cells: Vec<Cell> = (0..nel).into_par_iter().map(|el| cell(t, el)).collect();
    let mut part = vec![usize::MAX; nel];
    let mut ball = Ball { id: u32::MAX, v: vec![u32::MAX; t.l.d0.ncols], e: vec![u32::MAX; t.l.n_edge()], f: vec![u32::MAX; t.l.n_face()] };
    let mut cand: Vec<(usize, usize)> = Vec::new(); // (element, faces shared with the ball)
    let mut k = 0usize;
    for seed in 0..nel {
        if part[seed] != usize::MAX {
            continue;
        }
        part[seed] = k;
        ball.id = k as u32;
        cand.clear();
        // count the new faces' neighbours (faces already in the ball were counted when they joined)
        let add_faces = |el: usize, ball: &Ball, part: &[usize], cand: &mut Vec<(usize, usize)>| {
            for &f in &cells[el].f {
                if ball.has_f(f) {
                    continue;
                }
                for &o in t.d2t.cols(f) {
                    if o != el && part[o] == usize::MAX {
                        match cand.iter_mut().find(|x| x.0 == o) {
                            Some(x) => x.1 += 1,
                            None => cand.push((o, 1)),
                        }
                    }
                }
            }
        };
        add_faces(seed, &ball, &part, &mut cand);
        ball.add(&cells[seed]);
        let (mut lo, mut hi, mut km, mut m) = (bmin[seed], bmax[seed], kl[seed], 1);
        while m < target && !cand.is_empty() {
            cand.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let mut chosen = None;
            for &(el, _) in cand.iter() {
                let lo2: [f64; 3] = std::array::from_fn(|d| lo[d].min(bmin[el][d]));
                let hi2: [f64; 3] = std::array::from_fn(|d| hi[d].max(bmax[el][d]));
                let km2 = km.max(kl[el]);
                if km2 * dot3(sub3(hi2, lo2), sub3(hi2, lo2)).sqrt() > kappa {
                    continue;
                }
                if glue_ok(t, &cells[el], &ball) {
                    chosen = Some((el, lo2, hi2, km2));
                    break;
                }
            }
            let Some((el, lo2, hi2, km2)) = chosen else { break };
            part[el] = k;
            m += 1;
            lo = lo2;
            hi = hi2;
            km = km2;
            cand.retain(|x| x.0 != el);
            add_faces(el, &ball, &part, &mut cand);
            ball.add(&cells[el]);
        }
        k += 1;
    }
    part
}

// ------------------------------------------------------------------ coarse topology

struct CoarseTopology {
    ae_elems: Vec<Vec<usize>>,
    af_faces: Vec<(Vec<usize>, Vec<f64>)>,
    /// boundary edges (sorted) and their signs of every AF
    af_be: Vec<(Vec<usize>, Vec<f64>)>,
    ar_edges: Vec<(Vec<usize>, Vec<f64>)>,
    d0c: Sp,
    d1c: Sp,
    d2c: Sp,
    ar_free: Vec<bool>,
}

fn ae_ok(t: &Topo, elems: &[usize]) -> bool {
    let (v, e, f) = t.closure(elems);
    if v.len() as i64 - e.len() as i64 + f.len() as i64 - elems.len() as i64 != 1 {
        return false;
    }
    let bf = t.boundary_faces(elems);
    let be = sorted_unique(bf.iter().flat_map(|&x| t.l.d1.cols(x).iter().copied()).collect());
    let bv = sorted_unique(be.iter().flat_map(|&x| t.l.d0.cols(x).iter().copied()).collect());
    if bv.len() as i64 - be.len() as i64 + bf.len() as i64 != 2 {
        return false;
    }
    let mut ecount: HashMap<usize, Vec<usize>> = HashMap::new();
    for &f in &bf {
        for &x in t.l.d1.cols(f) {
            ecount.entry(x).or_default().push(f);
        }
    }
    if ecount.values().any(|fs| fs.len() != 2) {
        return false;
    }
    for &x in &bv {
        let ex: HashSet<usize> = t.d0t.cols(x).iter().copied().collect();
        let fx: Vec<usize> = bf.iter().copied().filter(|&f| t.l.d1.cols(f).iter().any(|e| ex.contains(e))).collect();
        let fxs: HashSet<usize> = fx.iter().copied().collect();
        let nb = |f: usize| -> Vec<usize> {
            t.l.d1.cols(f).iter().filter(|e| ex.contains(e)).flat_map(|e| ecount[e].iter().copied()).filter(|g| fxs.contains(g)).collect()
        };
        if components(&fx, &nb).len() != 1 {
            return false;
        }
    }
    let inner = setdiff(&e, &be);
    !inner.iter().any(|&x| t.nonfree[x])
}

fn af_ok(t: &Topo, faces: &[usize], signs: &[f64], cut: &HashSet<usize>) -> bool {
    let e = sorted_unique(faces.iter().flat_map(|&f| t.l.d1.cols(f).iter().copied()).collect());
    let v = sorted_unique(e.iter().flat_map(|&x| t.l.d0.cols(x).iter().copied()).collect());
    if v.len() as i64 - e.len() as i64 + faces.len() as i64 != 1 {
        return false;
    }
    let (be, _) = t.boundary_edges(faces, signs);
    let mut deg: HashMap<usize, usize> = HashMap::new();
    for &x in &be {
        for &vv in t.l.d0.cols(x) {
            *deg.entry(vv).or_insert(0) += 1;
        }
    }
    if deg.values().any(|&d| d != 2) {
        return false;
    }
    let inner = setdiff(&e, &be);
    if inner.iter().any(|x| cut.contains(x)) {
        return false;
    }
    !inner.iter().any(|&x| t.nonfree[x]) || faces.iter().all(|&f| t.pec_face[f])
}

fn coarsen_topology(t: &Topo, part: &[usize]) -> Result<CoarseTopology, String> {
    let mut clk = Clock::new();
    let l = t.l;
    // ---- AE: connected components (never across PEC), split until every AE is a ball
    let ngroups = part.iter().max().map_or(0, |m| m + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ngroups];
    for (el, &p) in part.iter().enumerate() {
        groups[p].push(el);
    }
    let pieces: Vec<(Vec<usize>, bool)> = groups
        .par_iter()
        .flat_map_iter(|g| {
            let gs: HashSet<usize> = g.iter().copied().collect();
            components(g, &|x| t.elem_nbrs(x, true).into_iter().filter(|y| gs.contains(y)).collect())
                .into_iter()
                .map(|c| {
                    let ok = c.len() == 1 || ae_ok(t, &c);
                    (c, ok)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut ae: Vec<Vec<usize>> = Vec::with_capacity(pieces.len());
    let mut retry: Vec<Vec<usize>> = Vec::new();
    for (g, ok) in pieces {
        if ok {
            ae.push(sorted_unique(g));
        } else {
            retry.push(g);
        }
    }
    while let Some(g) = retry.pop() {
        if g.len() == 1 || ae_ok(t, &g) {
            ae.push(sorted_unique(g));
            continue;
        }
        let sub: HashSet<usize> = g.iter().copied().collect();
        let (h1, h2) = bisect(&g, &|x| t.elem_nbrs(x, true).into_iter().filter(|y| sub.contains(y)).collect());
        for h in [h1, h2] {
            let hs: HashSet<usize> = h.iter().copied().collect();
            retry.extend(components(&h, &|x| t.elem_nbrs(x, true).into_iter().filter(|y| hs.contains(y)).collect()));
        }
    }
    let mut elem_ae = vec![0usize; l.n_elem()];
    for (i, g) in ae.iter().enumerate() {
        for &el in g {
            elem_ae[el] = i;
        }
    }
    clk.lap("topology: AE");

    // ---- AF: faces grouped by (AE pair, PEC), oriented outward from the lower AE
    let mut keyed: Vec<((usize, i64, bool), usize, f64)> = (0..l.n_face())
        .into_par_iter()
        .filter_map(|f| {
            let (el, sg) = t.d2t.row(f);
            let a: Vec<usize> = el.iter().map(|&x| elem_ae[x]).collect();
            if el.len() == 2 && a[0] == a[1] {
                return None;
            }
            let k = if el.len() == 2 && a[1] < a[0] { 1 } else { 0 };
            let other = if el.len() == 2 { a[1 - k] as i64 } else { -1 };
            Some(((a[k], other, t.pec_face[f]), f, sg[k]))
        })
        .collect();
    keyed.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut groups: Vec<((usize, i64), &[((usize, i64, bool), usize, f64)])> = Vec::new();
    let mut i = 0;
    while i < keyed.len() {
        let mut j = i;
        while j < keyed.len() && keyed[j].0 == keyed[i].0 {
            j += 1;
        }
        groups.push(((keyed[i].0 .0, keyed[i].0 .1), &keyed[i..j]));
        i = j;
    }
    let mut cut = vec![false; l.n_edge()];
    let (af, af_ae, af_be) = loop {
        let edge_faces = |f: usize| -> Vec<usize> {
            let mut out = Vec::new();
            for &e in l.d1.cols(f) {
                if (!t.nonfree[e] || t.pec_face[f]) && !cut[e] {
                    out.extend(t.d1t.cols(e).iter().copied());
                }
            }
            out
        };
        let cut_set: HashSet<usize> = (0..l.n_edge()).filter(|&e| cut[e]).collect();
        let per_group: Vec<Vec<((Vec<usize>, Vec<f64>), (usize, i64))>> = groups
            .par_iter()
            .map(|(key, lst)| {
                let sgn: HashMap<usize, f64> = lst.iter().map(|x| (x.1, x.2)).collect();
                let fs: Vec<usize> = lst.iter().map(|x| x.1).collect();
                let mut out = Vec::new();
                let mut todo = components(&fs, &|f| edge_faces(f).into_iter().filter(|g| sgn.contains_key(g)).collect());
                while let Some(c) = todo.pop() {
                    let s: Vec<f64> = c.iter().map(|f| sgn[f]).collect();
                    if c.len() == 1 || af_ok(t, &c, &s, &cut_set) {
                        out.push(((c, s), *key));
                    } else {
                        let sub: HashSet<usize> = c.iter().copied().collect();
                        let (h1, h2) = bisect(&c, &|f| edge_faces(f).into_iter().filter(|g| sub.contains(g)).collect());
                        for h in [h1, h2] {
                            let hs: HashSet<usize> = h.iter().copied().collect();
                            todo.extend(components(&h, &|f| edge_faces(f).into_iter().filter(|g| hs.contains(g)).collect()));
                        }
                    }
                }
                out
            })
            .collect();
        let mut af: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        let mut af_ae: Vec<(usize, i64)> = Vec::new();
        for g in per_group {
            for (a, k) in g {
                af.push(a);
                af_ae.push(k);
            }
        }
        // boundary edges of every AF, and its inner edges
        let be_in: Vec<((Vec<usize>, Vec<f64>), Vec<usize>)> = af
            .par_iter()
            .map(|(fs, ss)| {
                let (be, bs) = t.boundary_edges(fs, ss);
                let all = sorted_unique(fs.iter().flat_map(|&f| l.d1.cols(f).iter().copied()).collect());
                let inner = setdiff(&all, &be);
                ((be, bs), inner)
            })
            .collect();
        let mut bnd = vec![false; l.n_edge()];
        for ((be, _), _) in &be_in {
            be.iter().for_each(|&e| bnd[e] = true);
        }
        let mut clash = false;
        for (_, inner) in &be_in {
            for &e in inner {
                if bnd[e] {
                    cut[e] = true;
                    clash = true;
                }
            }
        }
        if !clash {
            break (af, af_ae, be_in.into_iter().map(|x| x.0).collect::<Vec<_>>());
        }
    };
    let n_af = af.len();
    let mut d2c_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ae.len()];
    for (i, &(lo, hi)) in af_ae.iter().enumerate() {
        d2c_rows[lo].push((i, 1.0));
        if hi >= 0 {
            d2c_rows[hi as usize].push((i, -1.0));
        }
    }
    let d2c = Sp::from_rows(n_af, d2c_rows);
    clk.lap("topology: AF");

    // ---- AR: AF-boundary edges grouped by (set of AFs, free); chains between APs
    let ne = l.n_edge();
    let mut edge_afs: Vec<Vec<u32>> = vec![Vec::new(); ne];
    for (i, (be, _)) in af_be.iter().enumerate() {
        for &e in be {
            edge_afs[e].push(i as u32); // AF ids increase, so each list is sorted
        }
    }
    let on_af: Vec<bool> = edge_afs.iter().map(|a| !a.is_empty()).collect();
    let same_grp = |a: usize, b: usize| edge_afs[a] == edge_afs[b] && t.nonfree[a] == t.nonfree[b];
    let nvert = l.d0.ncols;
    let mut vert_edges: Vec<Vec<usize>> = vec![Vec::new(); nvert];
    for e in 0..ne {
        if on_af[e] {
            for &x in l.d0.cols(e) {
                vert_edges[x].push(e);
            }
        }
    }
    let mut special: Vec<bool> = (0..nvert)
        .map(|x| {
            let es = &vert_edges[x];
            !es.is_empty() && (es.len() != 2 || !same_grp(es[0], es[1]))
        })
        .collect();
    let other = |e: usize, x: usize| -> usize { *l.d0.cols(e).iter().find(|&&y| y != x).unwrap() };
    let mut seen = vec![false; ne];
    let mut chains: Vec<(Vec<(usize, usize)>, usize)> = Vec::new();
    let walk = |x: usize, e: usize, seen: &mut Vec<bool>, special: &Vec<bool>| -> (Vec<(usize, usize)>, usize) {
        let mut path = vec![(e, x)];
        seen[e] = true;
        let mut y = other(e, x);
        while !special[y] {
            match vert_edges[y].iter().copied().find(|&d| !seen[d]) {
                None => break,
                Some(d) => {
                    path.push((d, y));
                    seen[d] = true;
                    y = other(d, y);
                }
            }
        }
        (path, y)
    };
    for x in 0..nvert {
        if !special[x] {
            continue;
        }
        for &e in &vert_edges[x] {
            if !seen[e] {
                let c = walk(x, e, &mut seen, &special);
                chains.push(c);
            }
        }
    }
    for e in 0..ne {
        // pure loops: split at two vertices
        if !on_af[e] || seen[e] {
            continue;
        }
        let x = l.d0.cols(e)[0];
        let mut lp = vec![e];
        let mut y = other(e, x);
        let mut cur = e;
        while y != x {
            cur = *vert_edges[y].iter().find(|&&d| d != cur).unwrap();
            lp.push(cur);
            y = other(cur, y);
        }
        let mut b = x;
        for &d in &lp[..lp.len() / 2] {
            b = other(d, b);
        }
        special[x] = true;
        special[b] = true;
        let c = walk(x, e, &mut seen, &special);
        chains.push(c);
        if let Some(r) = vert_edges[b].iter().copied().find(|&d| !seen[d]) {
            let c = walk(b, r, &mut seen, &special);
            chains.push(c);
        }
    }
    let mut ap_id = vec![usize::MAX; nvert];
    let mut nap = 0;
    for x in 0..nvert {
        if special[x] {
            ap_id[x] = nap;
            nap += 1;
        }
    }
    let mut ar: Vec<(Vec<usize>, Vec<f64>)> = Vec::with_capacity(chains.len());
    let mut d0c_rows = Vec::with_capacity(chains.len());
    let mut ar_free = Vec::with_capacity(chains.len());
    for (path, end) in &chains {
        let es: Vec<usize> = path.iter().map(|p| p.0).collect();
        // +1 when the fine edge points along the walk (its start vertex is the walk vertex)
        let sg: Vec<f64> = path.iter().map(|&(e, x)| -l.d0.get(e, x)).collect();
        let start = path[0].1;
        if start == *end {
            return Err("closed AR survived loop splitting".into());
        }
        d0c_rows.push(vec![(ap_id[start], -1.0), (ap_id[*end], 1.0)]);
        ar_free.push(!t.nonfree[es[0]]);
        ar.push((es, sg));
    }
    let d0c = Sp::from_rows(nap, d0c_rows);
    let mut edge_ar: Vec<(usize, f64)> = vec![(usize::MAX, 0.0); ne];
    for (j, (es, sg)) in ar.iter().enumerate() {
        for (&e, &s) in es.iter().zip(sg) {
            edge_ar[e] = (j, s);
        }
    }
    let d1c_rows: Vec<Result<Vec<(usize, f64)>, String>> = af_be
        .par_iter()
        .enumerate()
        .map(|(i, (be, bs))| {
            let mut acc: Vec<(usize, f64)> = Vec::new();
            for (&e, &s) in be.iter().zip(bs) {
                let (j, s2) = edge_ar[e];
                let val = s * s2;
                match acc.iter().find(|x| x.0 == j) {
                    Some(x) if x.1 != val => return Err(format!("AF {i}: inconsistent orientation along AR {j}")),
                    Some(_) => {}
                    None => acc.push((j, val)),
                }
            }
            Ok(acc)
        })
        .collect();
    let d1c = Sp::from_rows(ar.len(), d1c_rows.into_iter().collect::<Result<Vec<_>, _>>()?);
    clk.lap("topology: AR");
    Ok(CoarseTopology { ae_elems: ae, af_faces: af, af_be, ar_edges: ar, d0c, d1c, d2c, ar_free })
}

// ------------------------------------------------------------------ coarse spaces

/// Sparse matrix under construction whose entries are set (overwritten), row by row.
struct RowSet {
    rows: Vec<Vec<(usize, f64)>>,
}

impl RowSet {
    fn new(n: usize) -> Self {
        RowSet { rows: vec![Vec::new(); n] }
    }
    fn set(&mut self, i: usize, j: usize, v: f64) {
        match self.rows[i].iter_mut().find(|x| x.0 == j) {
            Some(x) => x.1 = v,
            None => self.rows[i].push((j, v)),
        }
    }
    fn get(&self, i: usize, j: usize) -> f64 {
        self.rows[i].iter().find(|x| x.0 == j).map_or(0.0, |x| x.1)
    }
    /// Set row i to `vals` (row i must be empty: interior rows belong to exactly one AE / AF).
    fn fill(&mut self, i: usize, vals: Vec<(usize, f64)>) {
        debug_assert!(self.rows[i].is_empty());
        self.rows[i] = vals;
    }
    /// dense rows `r` x columns `c`
    fn dense(&self, r: &[usize], c: &[usize]) -> Mat {
        let cm = local_map(c);
        let mut m = Mat::zeros(r.len(), c.len());
        for (i, &ri) in r.iter().enumerate() {
            for &(j, v) in &self.rows[ri] {
                if let Some(k) = cm.find(j) {
                    m.set(i, k, v);
                }
            }
        }
        m
    }
    fn into_sp(self, ncols: usize) -> Sp {
        Sp::from_rows(ncols, self.rows)
    }
}

/// Assemble entity-local dense matrices over `sel` onto the index list `n`.
fn gather(n: &[usize], ids: &[Vec<usize>], mats: &[Mat], sel: &[usize]) -> Mat {
    let m = local_map(n);
    let mut a = Mat::zeros(n.len(), n.len());
    for &k in sel {
        let p: Vec<usize> = ids[k].iter().map(|&x| m.at(x)).collect();
        for (i, &pi) in p.iter().enumerate() {
            for (j, &pj) in p.iter().enumerate() {
                a.add(pi, pj, mats[k].at(i, j));
            }
        }
    }
    a
}

/// rows `r`, columns `c` of a dense matrix (local indices)
fn submat(a: &Mat, r: &[usize], c: &[usize]) -> Mat {
    let mut m = Mat::zeros(r.len(), c.len());
    for (i, &ri) in r.iter().enumerate() {
        for (j, &cj) in c.iter().enumerate() {
            m.set(i, j, a.at(ri, cj));
        }
    }
    m
}

/// rows `r`, columns `c` of a sparse matrix, dense
fn sp_dense(s: &Sp, r: &[usize], c: &[usize]) -> Mat {
    let cm = local_map(c);
    let mut m = Mat::zeros(r.len(), c.len());
    for (i, &ri) in r.iter().enumerate() {
        let (cc, v) = s.row(ri);
        for (&j, &x) in cc.iter().zip(v) {
            if let Some(k) = cm.find(j) {
                m.set(i, k, x);
            }
        }
    }
    m
}

/// The saddle point  [M B^T T; B 0 0; T^T 0 0]-type systems of coarsen.py, assembled and solved.
/// Returns the solution (rows: unknowns, multipliers, last row the total-flux multiplier if any).
static SUB_NS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0),
];
fn sub_tick(i: usize, t: Instant) {
    SUB_NS[i].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}
fn sub_report() -> String {
    let v: Vec<f64> = SUB_NS.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9).collect();
    if v.iter().all(|&x| x == 0.0) {
        return String::new();
    }
    format!("   [thread-seconds: gather {:.2}, WD/Cm/rhs {:.2}, solve {:.2}, P^T M P {:.2}, other {:.2}]", v[0], v[1], v[2], v[3], v[4])
}

static SADDLE_STATS: [std::sync::atomic::AtomicU64; 3] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// (systems, sum of n^3 / 1e6, largest n) since the last call
fn saddle_stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (SADDLE_STATS[0].swap(0, Relaxed), SADDLE_STATS[1].swap(0, Relaxed) / 1_000_000, SADDLE_STATS[2].swap(0, Relaxed))
}

fn saddle(m: &Mat, b: &Mat, tl: Option<&[f64]>, cm: Option<&Mat>, rhs_u: &Mat, rhs_p: &Mat) -> Result<Mat, String> {
    let (nu, np) = (m.r, b.r);
    {
        use std::sync::atomic::Ordering::Relaxed;
        let n = (nu + np + tl.is_some() as usize) as u64;
        SADDLE_STATS[0].fetch_add(1, Relaxed);
        SADDLE_STATS[1].fetch_add(n * n * n, Relaxed);
        SADDLE_STATS[2].fetch_max(n, Relaxed);
    }
    let extra = tl.is_some() as usize;
    let n = nu + np + extra;
    let mut k = Mat::zeros(n, n);
    for i in 0..nu {
        for j in 0..nu {
            k.set(i, j, m.at(i, j));
        }
    }
    for i in 0..np {
        for j in 0..nu {
            k.set(nu + i, j, b.at(i, j));
            k.set(j, nu + i, b.at(i, j));
        }
    }
    if let Some(c) = cm {
        for i in 0..np {
            for j in 0..np {
                k.set(nu + i, nu + j, -c.at(i, j));
            }
        }
    }
    if let Some(tl) = tl {
        for i in 0..np {
            k.set(n - 1, nu + i, tl[i]);
            k.set(nu + i, n - 1, tl[i]);
        }
    }
    let nr = rhs_u.c;
    let mut rhs = Mat::zeros(n, nr);
    for i in 0..nu {
        for j in 0..nr {
            rhs.set(i, j, rhs_u.at(i, j));
        }
    }
    for i in 0..np {
        for j in 0..nr {
            rhs.set(nu + i, j, rhs_p.at(i, j));
        }
    }
    solve_dense(k, rhs)
}

/// rows `rows` of `p` := the rows of `x` (first rows.len() rows; extra rows of x, e.g. multipliers, ignored)
fn put_rows(p: &mut Mat, rows: &[usize], x: &Mat) {
    let c = p.c;
    for (i, &r) in rows.iter().enumerate() {
        p.d[r * c..(r + 1) * c].copy_from_slice(&x.d[i * c..(i + 1) * c]);
    }
}

fn scale_rows(m: &mut Mat, w: &[f64]) {
    let c = m.c;
    for (row, &s) in m.d.chunks_exact_mut(c.max(1)).zip(w) {
        row.iter_mut().for_each(|x| *x *= s);
    }
}

fn neg(a: &Mat) -> Mat {
    Mat { r: a.r, c: a.c, d: a.d.iter().map(|x| -x).collect() }
}

fn sub(a: &Mat, b: &Mat) -> Mat {
    Mat { r: a.r, c: a.c, d: a.d.iter().zip(&b.d).map(|(x, y)| x - y).collect() }
}

const DC_TOL: f64 = 1e-8;

/// One level to the next (coarsen.py: coarsen). Returns the coarse level, P1 (fine edges x ARs),
/// P2 (fine faces x AFs) and the AE of every coarse element (lists of fine elements).
fn coarsen(t: &Topo, part: &[usize]) -> Result<(Level, Sp, Sp, Vec<Vec<usize>>), String> {
    let mut clk = Clock::new();
    let l = t.l;
    let c = coarsen_topology(t, part)?;
    clk.lap("coarsen_topology");
    let (n_ae, n_af, n_ar) = (c.ae_elems.len(), c.af_faces.len(), c.ar_edges.len());
    let (d1, d2) = (&l.d1, &l.d2);
    let d2c = &c.d2c;
    let d1c = &c.d1c;

    // ---------- L2
    let w_ae: Vec<f64> = c.ae_elems.iter().map(|g| g.iter().map(|&x| l.w_elem[x]).sum()).collect();
    let m3c: Vec<f64> = c.ae_elems.iter().enumerate().map(|(a, g)| g.iter().map(|&x| (l.w_elem[x] / w_ae[a]).powi(2) * l.elem_m3[x]).sum()).collect();

    // ---------- H(div): traces
    let mut p2 = RowSet::new(l.n_face());
    let w_af: Vec<f64> = c.af_faces.iter().map(|(fs, _)| fs.iter().map(|&f| l.w_face[f]).sum()).collect();
    for (j, (fs, ss)) in c.af_faces.iter().enumerate() {
        for (&f, &s) in fs.iter().zip(ss) {
            p2.set(f, j, s * l.w_face[f] / w_af[j]);
        }
    }
    let face_m2c: Vec<f64> = c.af_faces.iter().enumerate().map(|(j, (fs, _))| fs.iter().map(|&f| p2.get(f, j).powi(2) * l.face_m2[f]).sum()).collect();

    struct AeInfo {
        ea: Vec<usize>,
        fa: Vec<usize>,
        bf: Vec<usize>,
        fi: Vec<usize>,
        afs: Vec<usize>,
    }
    let ae_info: Vec<AeInfo> = c
        .ae_elems
        .par_iter()
        .enumerate()
        .map(|(a, g)| {
            let (_, ea, fa) = t.closure(g);
            let bf = t.boundary_faces(g);
            let fi = setdiff(&fa, &bf);
            AeInfo { ea, fa, bf, fi, afs: d2c.cols(a).to_vec() }
        })
        .collect();

    // H(div) into each AE: min |u|_M2 s.t. div u in span(P3[:, AE]); the AE's P2 block (rows: its
    // faces) is assembled from the traces and the solution, and gives the coarse mass right away
    type Rows = Vec<(usize, Vec<(usize, f64)>)>; // interior rows of P, each filled once
    type Ext = (Rows, Mat, Mat); // (interior rows, local P block, coarse mass)
    let ext2: Vec<Result<Ext, String>> = (0..n_ae)
        .into_par_iter()
        .map(|a| {
            let g = &c.ae_elems[a];
            let inf = &ae_info[a];
            let m2a = gather(&inf.fa, &l.elem_faces, &l.elem_m2, g);
            let fl = local_map(&inf.fa);
            let ii: Vec<usize> = inf.fi.iter().map(|&f| fl.at(f)).collect();
            let bb: Vec<usize> = inf.bf.iter().map(|&f| fl.at(f)).collect();
            let r = p2.dense(&inf.bf, &inf.afs);
            let mut pa = Mat::zeros(inf.fa.len(), inf.afs.len());
            put_rows(&mut pa, &bb, &r);
            let mut out = Vec::new();
            if !inf.fi.is_empty() {
                let w3: Vec<f64> = g.iter().map(|&x| l.elem_m3[x]).collect();
                let mut bmat = sp_dense(d2, g, &inf.fa);
                scale_rows(&mut bmat, &w3);
                let tl: Vec<f64> = g.iter().enumerate().map(|(i, &x)| w3[i] * l.w_elem[x] / w_ae[a]).collect();
                let all: Vec<usize> = (0..g.len()).collect();
                let rhs_u = neg(&submat(&m2a, &ii, &bb).matmul(&r));
                let rhs_p = neg(&submat(&bmat, &all, &bb).matmul(&r));
                let x = saddle(&submat(&m2a, &ii, &ii), &submat(&bmat, &all, &ii), Some(&tl), None, &rhs_u, &rhs_p)?;
                let last = x.r - 1;
                for (k, &j) in inf.afs.iter().enumerate() {
                    if (-x.at(last, k) - d2c.get(a, j)).abs() > DC_TOL {
                        return Err(format!("AE {a}: coarse divergence does not match the incidence"));
                    }
                }
                put_rows(&mut pa, &ii, &x);
                for (i, &f) in inf.fi.iter().enumerate() {
                    out.push((f, inf.afs.iter().enumerate().map(|(k, &j)| (j, x.at(i, k))).collect()));
                }
            }
            let m = Mat::ptap(&m2a, &pa);
            Ok((out, pa, m))
        })
        .collect();
    let mut elem_m2c = Vec::with_capacity(n_ae);
    let mut pa2 = Vec::with_capacity(n_ae);
    for r in ext2 {
        let (out, pa, m) = r?;
        for (f, row) in out {
            p2.fill(f, row);
        }
        pa2.push(pa);
        elem_m2c.push(m);
    }

    clk.lap("H(div) extension + masses");
    // ---------- H(curl): traces
    let mut p1 = RowSet::new(l.n_edge());
    let w_ar: Vec<f64> = c.ar_edges.iter().map(|(es, _)| es.iter().map(|&e| l.w_edge[e]).sum()).collect();
    for (j, (es, ss)) in c.ar_edges.iter().enumerate() {
        for (&e, &s) in es.iter().zip(ss) {
            p1.set(e, j, s * l.w_edge[e] / w_ar[j]);
        }
    }

    // into each AF: min |u|_M1F s.t. curl u in span(P2[:, AF])
    let ext_f: Vec<Result<(Rows, Vec<usize>, Mat), String>> = (0..n_af)
        .into_par_iter()
        .map(|j| {
            let (fs, _) = &c.af_faces[j];
            let ef = sorted_unique(fs.iter().flat_map(|&f| l.face_edges[f].iter().copied()).collect());
            let be = &c.af_be[j].0;
            let ei = setdiff(&ef, be);
            let ars = d1c.cols(j).to_vec();
            let mf = gather(&ef, &l.face_edges, &l.face_m1, fs);
            let el = local_map(&ef);
            let ii: Vec<usize> = ei.iter().map(|&e| el.at(e)).collect();
            let bb: Vec<usize> = be.iter().map(|&e| el.at(e)).collect();
            let r = p1.dense(be, &ars);
            let mut pf = Mat::zeros(ef.len(), ars.len());
            put_rows(&mut pf, &bb, &r);
            let mut out = Vec::new();
            if !ei.is_empty() {
                let wf: Vec<f64> = fs.iter().map(|&f| l.face_m2[f]).collect();
                let mut bmat = sp_dense(d1, fs, &ef);
                scale_rows(&mut bmat, &wf);
                let tl: Vec<f64> = fs.iter().enumerate().map(|(i, &f)| wf[i] * p2.get(f, j)).collect();
                let all: Vec<usize> = (0..fs.len()).collect();
                let rhs_u = neg(&submat(&mf, &ii, &bb).matmul(&r));
                let rhs_p = neg(&submat(&bmat, &all, &bb).matmul(&r));
                let x = saddle(&submat(&mf, &ii, &ii), &submat(&bmat, &all, &ii), Some(&tl), None, &rhs_u, &rhs_p)?;
                let last = x.r - 1;
                for (k, &r_) in ars.iter().enumerate() {
                    if (-x.at(last, k) - d1c.get(j, r_)).abs() > DC_TOL {
                        return Err(format!("AF {j}: coarse curl does not match the incidence"));
                    }
                }
                put_rows(&mut pf, &ii, &x);
                for (i, &e) in ei.iter().enumerate() {
                    out.push((e, ars.iter().enumerate().map(|(k, &r_)| (r_, x.at(i, k))).collect()));
                }
            }
            Ok((out, ars, Mat::ptap(&mf, &pf)))
        })
        .collect();
    let mut face_ars = Vec::with_capacity(n_af);
    let mut face_m1c = Vec::with_capacity(n_af);
    for r in ext_f {
        let (out, ars, m) = r?;
        for (e, row) in out {
            p1.fill(e, row);
        }
        face_ars.push(ars);
        face_m1c.push(m);
    }

    clk.lap("H(curl) into AF + masses");
    // into each AE: min |u|_M1 s.t. curl u = P2 D1c (regularized by div-div)
    let ext_e: Vec<Result<(Rows, Vec<usize>, Mat), String>> = (0..n_ae)
        .into_par_iter()
        .map(|a| {
            let g = &c.ae_elems[a];
            let inf = &ae_info[a];
            let ars = sorted_unique(inf.afs.iter().flat_map(|&j| d1c.cols(j).iter().copied()).collect());
            let be_ = sorted_unique(inf.bf.iter().flat_map(|&f| d1.cols(f).iter().copied()).collect());
            let ei = setdiff(&inf.ea, &be_);
            let t0 = Instant::now();
            let m1a = gather(&inf.ea, &l.elem_edges, &l.elem_m1, g);
            sub_tick(0, t0);
            let el = local_map(&inf.ea);
            let ii: Vec<usize> = ei.iter().map(|&e| el.at(e)).collect();
            let bb: Vec<usize> = be_.iter().map(|&e| el.at(e)).collect();
            let r = p1.dense(&be_, &ars);
            let mut pa = Mat::zeros(inf.ea.len(), ars.len());
            put_rows(&mut pa, &bb, &r);
            let mut out = Vec::new();
            if !ei.is_empty() {
                let t0 = Instant::now();
                let m2a = gather(&inf.fa, &l.elem_faces, &l.elem_m2, g);
                sub_tick(0, t0);
                let t0 = Instant::now();
                let fl = local_map(&inf.fa);
                let fi: Vec<usize> = inf.fi.iter().map(|&f| fl.at(f)).collect();
                // WD[fi, :] = M2a[fi, :] D1[Fa, Ea], through D1's sparsity
                let mut wd = Mat::zeros(fi.len(), inf.ea.len());
                let d1loc: Vec<Vec<(usize, f64)>> = inf
                    .fa
                    .iter()
                    .map(|&f| {
                        let (c, v) = d1.row(f);
                        c.iter().zip(v).map(|(&e, &x)| (el.at(e), x)).collect()
                    })
                    .collect();
                for (i, &fr) in fi.iter().enumerate() {
                    let row = &mut wd.d[i * inf.ea.len()..(i + 1) * inf.ea.len()];
                    for (fl_, d) in d1loc.iter().enumerate() {
                        let m = m2a.at(fr, fl_);
                        if m != 0.0 {
                            for &(e, x) in d {
                                row[e] += m * x;
                            }
                        }
                    }
                }
                let d2fi = submat(&sp_dense(d2, g, &inf.fa), &(0..g.len()).collect::<Vec<_>>(), &fi);
                let mut w = d2fi.clone();
                scale_rows(&mut w, &g.iter().map(|&x| l.elem_m3[x]).collect::<Vec<_>>());
                let cm = d2fi.tmatmul(&w);
                let pdc = pa2[a].matmul(&sp_dense(d1c, &inf.afs, &ars));
                let rhs_u = neg(&submat(&m1a, &ii, &bb).matmul(&r));
                let allfi: Vec<usize> = (0..fi.len()).collect();
                let rhs_p = sub(&submat(&m2a, &fi, &(0..inf.fa.len()).collect::<Vec<_>>()).matmul(&pdc), &submat(&wd, &allfi, &bb).matmul(&r));
                sub_tick(1, t0);
                let t0 = Instant::now();
                let x = saddle(&submat(&m1a, &ii, &ii), &submat(&wd, &allfi, &ii), None, Some(&cm), &rhs_u, &rhs_p)?;
                sub_tick(2, t0);
                put_rows(&mut pa, &ii, &x);
                for (i, &e) in ei.iter().enumerate() {
                    out.push((e, ars.iter().enumerate().map(|(k, &r_)| (r_, x.at(i, k))).collect()));
                }
            }
            let t0 = Instant::now();
            let mc = Mat::ptap(&m1a, &pa);
            sub_tick(3, t0);
            Ok((out, ars, mc))
        })
        .collect();
    let mut ae_ars = Vec::with_capacity(n_ae);
    let mut elem_m1c = Vec::with_capacity(n_ae);
    for r in ext_e {
        let (out, ars, m) = r?;
        for (e, row) in out {
            p1.fill(e, row);
        }
        ae_ars.push(ars);
        elem_m1c.push(m);
    }
    drop(pa2);

    clk.lap("H(curl) into AE + masses");
    let p1 = p1.into_sp(n_ar);
    let lc = Level {
        d0: c.d0c,
        d1: c.d1c.clone(),
        d2: c.d2c.clone(),
        elem_edges: ae_ars,
        elem_m1: elem_m1c,
        elem_faces: ae_info.iter().map(|i| i.afs.clone()).collect(),
        elem_m2: elem_m2c,
        elem_m3: m3c,
        face_edges: face_ars,
        face_m1: face_m1c,
        face_m2: face_m2c,
        w_edge: w_ar,
        w_face: w_af,
        w_elem: w_ae,
        free_edge: c.ar_free,
    };
    Ok((lc, p1, p2.into_sp(n_af), c.ae_elems))
}

/// Max violations of D1 P1 = P2 D1c, D1c D0c = 0, D2c D1c = 0 and of the PEC rows (coarsen.check).
fn check(l: &Level, lc: &Level, p1: &Sp, p2: &Sp) -> [f64; 4] {
    let a = l.d1.matmul(p1).max_abs_diff(&p2.matmul(&lc.d1));
    let z1 = lc.d1.matmul(&lc.d0);
    let b = z1.val.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let z2 = lc.d2.matmul(&lc.d1);
    let c = z2.val.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut pec: f64 = 0.0;
    for e in 0..l.n_edge() {
        if !l.free_edge[e] {
            let (cc, v) = p1.row(e);
            for (&j, &x) in cc.iter().zip(v) {
                if lc.free_edge[j] {
                    pec = pec.max(x.abs());
                }
            }
        }
    }
    [a, b, c, pec]
}

// ------------------------------------------------------------------ the hierarchy

pub struct Hierarchy {
    /// level l+1 -> level l on free edges (level 0 in increasing edge order)
    pub ps: Vec<Sp>,
    /// gradient of every smoothed level on its free edges
    pub gs: Vec<Sp>,
    pub sizes: Vec<usize>,
    /// per level: max violation of the de Rham identities (should be ~1e-14)
    pub checks: Vec<[f64; 4]>,
}

pub const STALL: f64 = 0.85;

/// corvid/amge/capped.py: build_hierarchy, in Rust.
#[allow(clippy::too_many_arguments)]
pub fn hierarchy(
    x: &[[f64; 3]], tets: &[[usize; 4]], edges: &[[usize; 2]], free: &[bool], k_tet: &[f64], kappa: f64, target: usize,
    max_levels: usize, do_check: bool,
) -> Result<Hierarchy, String> {
    let mut clk = Clock::new();
    let mut l = fine_level(x, tets, edges, free)?;
    clk.lap("fine_level");
    let mut bmin: Vec<[f64; 3]> = tets.iter().map(|t| std::array::from_fn(|d| t.iter().map(|&v| x[v][d]).fold(f64::INFINITY, f64::min))).collect();
    let mut bmax: Vec<[f64; 3]> = tets.iter().map(|t| std::array::from_fn(|d| t.iter().map(|&v| x[v][d]).fold(f64::NEG_INFINITY, f64::max))).collect();
    let mut kl = k_tet.to_vec();
    let mut h = Hierarchy { ps: Vec::new(), gs: vec![l.grad()], sizes: vec![l.n_free()], checks: Vec::new() };
    while h.ps.len() < max_levels.saturating_sub(1) {
        let t = Topo::new(&l);
        clk.lap("Topo::new");
        let part = grow_capped(&t, &bmin, &bmax, &kl, kappa, target);
        clk.lap("grow");
        let nparts = part.iter().max().map_or(0, |m| m + 1);
        if nparts as f64 > 0.9 * l.n_elem() as f64 {
            break;
        }
        let (lc, p1, p2, ae) = coarsen(&t, &part)?;
        drop(t);
        clk.lap("coarsen total");
        let nf = lc.n_free();
        if nf == 0 || nf as f64 > STALL * l.n_free() as f64 {
            break;
        }
        if do_check {
            h.checks.push(check(&l, &lc, &p1, &p2));
        }
        h.ps.push(p1.select(&l.free_edge, &lc.free_edge));
        h.gs.push(lc.grad());
        h.sizes.push(nf);
        bmin = ae.iter().map(|g| std::array::from_fn(|d| g.iter().map(|&e| bmin[e][d]).fold(f64::INFINITY, f64::min))).collect();
        bmax = ae.iter().map(|g| std::array::from_fn(|d| g.iter().map(|&e| bmax[e][d]).fold(f64::NEG_INFINITY, f64::max))).collect();
        kl = ae.iter().map(|g| g.iter().map(|&e| kl[e]).fold(0.0, f64::max)).collect();
        l = lc;
        clk.lap("select / grad / geometry");
    }
    h.gs.truncate(h.ps.len());
    Ok(h)
}
