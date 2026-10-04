use nalgebra::{Cholesky, DMatrix, DVector};
use rayon::prelude::*;

/// Block-diagonal 8×8 Cholesky factors, one per spherical-harmonic mode.
pub fn block_cholesky(matrix: &DMatrix<f64>) -> Vec<Cholesky<f64, nalgebra::Dyn>> {
    let n = matrix.nrows();
    let blocks = n / 8;
    (0..blocks)
        .into_par_iter()
        .map(|block| {
            let start = block * 8;
            let local = matrix.view((start, start), (8, 8)).into_owned();
            if let Some(factor) = local.clone().cholesky() {
                return factor;
            }
            let scale = local
                .diagonal()
                .iter()
                .map(|value| value.abs())
                .fold(1.0_f64, f64::max);
            let mut shifted = local;
            for trial in 0..6 {
                let delta = scale * 10.0_f64.powi(trial - 8);
                for i in 0..8 {
                    shifted[(i, i)] += delta;
                }
                if let Some(factor) = shifted.clone().cholesky() {
                    return factor;
                }
            }
            DMatrix::identity(8, 8).cholesky().expect("identity is SPD")
        })
        .collect()
}

pub fn apply_block_inverse(
    factors: &[Cholesky<f64, nalgebra::Dyn>],
    rhs: &DVector<f64>,
) -> DVector<f64> {
    let mut out = DVector::zeros(rhs.len());
    for (block, factor) in factors.iter().enumerate() {
        let start = block * 8;
        let local = DVector::from_iterator(8, rhs.rows(start, 8).iter().copied());
        let solved = factor.solve(&local);
        out.rows_mut(start, 8).copy_from(&solved);
    }
    out
}
