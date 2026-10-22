//! Quadrature rules of §9.4: Gauss–Jacobi in `ζ` with weight `ζ²`, and a product
//! Gauss–Legendre × uniform rule on the sphere.

use std::f64::consts::PI;

use nalgebra::DMatrix;

use crate::common::math::DVec3;

/// Eigen-decomposition of a Jacobi matrix: nodes and first-component weights.
fn golub_welsch(diag: &[f64], off: &[f64], mu0: f64) -> (Vec<f64>, Vec<f64>) {
    let n = diag.len();
    let mut j = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        j[(i, i)] = diag[i];
        if i + 1 < n {
            j[(i, i + 1)] = off[i];
            j[(i + 1, i)] = off[i];
        }
    }
    let eig = j.symmetric_eigen();
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| (eig.eigenvalues[i], mu0 * eig.eigenvectors[(0, i)].powi(2)))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

/// `n`-point Gauss–Legendre rule on `[-1, 1]`.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let diag = vec![0.0; n];
    let off: Vec<f64> = (1..n)
        .map(|k| {
            let k = k as f64;
            k / (4.0 * k * k - 1.0).sqrt()
        })
        .collect();
    golub_welsch(&diag, &off, 2.0)
}

/// `n`-point rule with `∫₀¹ f(ζ) ζ² dζ ≈ Σ ω_i f(ζ_i)`: Gauss–Jacobi `(α, β) = (0, 2)` on
/// `x = 2ζ − 1`. Exact for polynomials of degree `2n − 1`.
pub fn gauss_jacobi_zeta2(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (alpha, beta) = (0.0_f64, 2.0_f64);
    let diag: Vec<f64> = (0..n)
        .map(|k| {
            let k = k as f64;
            if k == 0.0 {
                (beta - alpha) / (alpha + beta + 2.0)
            } else {
                (beta * beta - alpha * alpha)
                    / ((2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 2.0))
            }
        })
        .collect();
    let off: Vec<f64> = (1..n)
        .map(|k| {
            let k = k as f64;
            let s = 2.0 * k + alpha + beta;
            2.0 / s * (k * (k + alpha) * (k + beta) * (k + alpha + beta) / (s * s - 1.0)).sqrt()
        })
        .collect();
    // ∫ (1 + x)² dx over [-1, 1] = 8/3; the map to ζ and the ζ² density contribute 1/8.
    let (x, w) = golub_welsch(&diag, &off, 8.0 / 3.0);
    (
        x.iter().map(|x| 0.5 * (1.0 + x)).collect(),
        w.iter().map(|w| w / 8.0).collect(),
    )
}

/// Directions and solid-angle weights (sum `4π`) of the product rule.
pub fn sphere_grid(n_theta: usize, n_phi: usize) -> (Vec<DVec3>, Vec<f64>) {
    let (z, wz) = gauss_legendre(n_theta);
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    let mut weights = Vec::with_capacity(n_theta * n_phi);
    for (zi, wi) in z.iter().zip(&wz) {
        let s = (1.0 - zi * zi).sqrt();
        for j in 0..n_phi {
            // An offset avoids placing nodes on the meridian of the polar axis symmetry.
            let phi = 2.0 * PI * (j as f64 + 0.5) / n_phi as f64;
            dirs.push(DVec3::new(s * phi.cos(), s * phi.sin(), *zi));
            weights.push(wi * 2.0 * PI / n_phi as f64);
        }
    }
    (dirs, weights)
}
