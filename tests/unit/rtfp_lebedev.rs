//! Unit tests of `rtfp::lebedev`, moved out of `src/rtfp/lebedev.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use std::sync::OnceLock;
}
use ctx::*;

#[test]
fn rule_integrates_low_degree_polynomials() {
    let r = astroid_evolution::rtfp::lebedev::rule();
    let total: f64 = r.weights.iter().sum();
    assert!((total - 4.0 * std::f64::consts::PI).abs() < 1e-13);
    // ∮ x² dΩ = 4π/3, ∮ x²y²z² dΩ = 4π/105.
    let x2: f64 = r
        .dirs
        .iter()
        .zip(&r.weights)
        .map(|(d, w)| w * d.x * d.x)
        .sum();
    let xyz: f64 = r
        .dirs
        .iter()
        .zip(&r.weights)
        .map(|(d, w)| w * (d.x * d.y * d.z).powi(2))
        .sum();
    assert!((x2 - 4.0 * std::f64::consts::PI / 3.0).abs() < 1e-13);
    assert!((xyz - 4.0 * std::f64::consts::PI / 105.0).abs() < 1e-13);
}
