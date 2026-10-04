use nalgebra::{DMatrix, DVector};

use super::precond_b1::{apply_block_inverse, block_cholesky};

fn pcg(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    factors: &[nalgebra::Cholesky<f64, nalgebra::Dyn>],
) -> (DVector<f64>, f64) {
    let mut x = DVector::zeros(rhs.len());
    let mut r = rhs.clone();
    let mut z = apply_block_inverse(factors, &r);
    let mut p = z.clone();
    let mut rz = r.dot(&z);
    for _ in 0..rhs.len() + 2 {
        let ap = matrix * &p;
        let denom = p.dot(&ap);
        if denom.abs() < 1e-30 || rz.abs() < 1e-30 {
            break;
        }
        let alpha = rz / denom;
        x += &p * alpha;
        r -= &ap * alpha;
        if r.norm() <= 1e-10 * rhs.norm().max(1.0) {
            break;
        }
        z = apply_block_inverse(factors, &r);
        let next = r.dot(&z);
        p = &z + p * (next / rz);
        rz = next;
    }
    let residual = (matrix * &x - rhs).norm() / rhs.norm().max(1e-30);
    (x, residual)
}

#[derive(Clone, Debug)]
pub struct RangeSolve {
    pub velocity: DVector<f64>,
    pub lambda: DVector<f64>,
    pub pcg_residual: f64,
}

/// Range-space rigid projection: u = y0 − Y λ, with B u = 0.
pub fn range_space_velocity(
    stiffness: &DMatrix<f64>,
    constraint: &DMatrix<f64>,
    load: &DVector<f64>,
    gamma: f64,
) -> Result<RangeSolve, String> {
    let k_gamma = stiffness + constraint.transpose() * constraint * gamma;
    let factors = block_cholesky(&k_gamma);
    let (y0, y0_residual) = pcg(&k_gamma, load, &factors);
    let mut y = DMatrix::zeros(load.len(), 6);
    let mut pcg_residual = y0_residual;
    for mode in 0..6 {
        let column = constraint.row(mode).transpose();
        let (solved, residual) = pcg(&k_gamma, &column, &factors);
        pcg_residual = pcg_residual.max(residual);
        y.set_column(mode, &solved);
    }
    let schur = constraint * &y;
    let schur_rhs = constraint * &y0;
    let lambda = match schur.cholesky() {
        Some(factor) => factor.solve(&schur_rhs),
        None => return Err("6x6 rigid Schur complement is singular".into()),
    };
    Ok(RangeSolve {
        velocity: y0 - y * &lambda,
        lambda,
        pcg_residual,
    })
}
