//! Products with the tall sample matrices of the Stokes problem (a row per component and
//! quadrature point), split into row blocks across threads; nalgebra runs them on one.
//! The matrices may be stored in f32 where they are only read: every product
//! accumulates in f64, so storage rounding (2⁻²⁴ per entry) is the only change.

use nalgebra::{DMatrix, DVector, Scalar};
use rayon::prelude::*;

/// Rows per parallel block: large enough to amortize the split, small enough to stay in cache.
const BLOCK: usize = 4096;
/// Rows per scaled f64 block of a Gram product: one per thread is alive at a time.
const GRAM_ROWS: usize = 1024;

/// Storage element of a sample matrix.
pub trait Entry: Scalar + Copy + Into<f64> + Send + Sync {
    fn of(x: f64) -> Self;
}

impl Entry for f64 {
    #[inline]
    fn of(x: f64) -> Self {
        x
    }
}

impl Entry for f32 {
    #[inline]
    fn of(x: f64) -> Self {
        x as f32
    }
}

/// A zero matrix of any storage element.
pub fn zeros<T: Entry>(rows: usize, cols: usize) -> DMatrix<T> {
    DMatrix::from_element(rows, cols, T::of(0.0))
}

/// `A x`.
pub fn mul<T: Entry>(a: &DMatrix<T>, x: &DVector<f64>) -> DVector<f64> {
    let (m, n) = a.shape();
    let col = |j: usize| &a.as_slice()[j * m..(j + 1) * m];
    let mut out = DVector::zeros(m);
    out.as_mut_slice()
        .par_chunks_mut(BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            let (r0, len) = (b * BLOCK, chunk.len());
            for j in 0..n {
                let xj = x[j];
                for (o, v) in chunk.iter_mut().zip(&col(j)[r0..r0 + len]) {
                    *o += xj * (*v).into();
                }
            }
        });
    out
}

/// `Aᵀ y`.
pub fn tr_mul<T: Entry>(a: &DMatrix<T>, y: &DVector<f64>) -> DVector<f64> {
    let m = a.nrows();
    DVector::from_vec(
        (0..a.ncols())
            .into_par_iter()
            .map(|j| {
                a.as_slice()[j * m..(j + 1) * m]
                    .iter()
                    .zip(y.iter())
                    .map(|(v, w)| (*v).into() * w)
                    .sum()
            })
            .collect(),
    )
}

/// `Σ_p s_p A_pᵀ A_p` with `A_p` the `per` rows of point `p` and `s_p ≥ 0`: a weighted
/// Gram matrix, formed one block of scaled f64 rows at a time instead of from a scaled copy.
pub fn weighted_gram<T: Entry>(a: &DMatrix<T>, per: usize, s: &[f64]) -> DMatrix<f64> {
    let n = a.ncols();
    let points = (GRAM_ROWS / per).max(1);
    (0..s.len().div_ceil(points))
        .into_par_iter()
        .map(|b| {
            let (p0, p1) = (b * points, ((b + 1) * points).min(s.len()));
            let blk = DMatrix::from_fn(per * (p1 - p0), n, |r, j| {
                s[p0 + r / per].sqrt() * a[(per * p0 + r, j)].into()
            });
            blk.tr_mul(&blk)
        })
        .reduce(|| DMatrix::zeros(n, n), |x, y| x + y)
}
