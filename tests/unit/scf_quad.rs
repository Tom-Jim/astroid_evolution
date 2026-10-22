//! Unit tests of `scf::quad`, moved out of `src/scf/quad.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use nalgebra::DMatrix;
    pub use std::f64::consts::PI;
}
use ctx::*;

use astroid_evolution::scf::quad::*;

#[test]
fn jacobi_rule_integrates_zeta_powers() {
    let (z, w) = gauss_jacobi_zeta2(5);
    for k in 0..10 {
        let got: f64 = z.iter().zip(&w).map(|(z, w)| w * z.powi(k)).sum();
        assert!((got - 1.0 / (k as f64 + 3.0)).abs() < 1e-13, "k={k}");
    }
    assert!(z.iter().all(|z| *z > 0.0 && *z < 1.0));
}

#[test]
fn sphere_rule_integrates_low_degree_polynomials() {
    let (d, w) = sphere_grid(8, 16);
    let total: f64 = w.iter().sum();
    assert!((total - 4.0 * PI).abs() < 1e-12);
    let zz: f64 = d.iter().zip(&w).map(|(d, w)| w * d.z * d.z).sum();
    assert!((zz - 4.0 * PI / 3.0).abs() < 1e-12);
    let xxyy: f64 = d
        .iter()
        .zip(&w)
        .map(|(d, w)| w * d.x * d.x * d.y * d.y)
        .sum();
    assert!((xxyy - 4.0 * PI / 15.0).abs() < 1e-12);
    let odd: f64 = d.iter().zip(&w).map(|(d, w)| w * d.x * d.y * d.z).sum();
    assert!(odd.abs() < 1e-12);
}
