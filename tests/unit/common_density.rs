//! Unit tests of `common::density`, moved out of `src/common/density.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::{DMat3, DVec3, Sym6, mat_sym, sym_mat};
}
use ctx::*;

use astroid_evolution::common::density::*;

fn sheared() -> CauchyKernel {
    CauchyKernel::try_anisotropic(
        DVec3::new(0.2, -0.1, 0.05),
        1.4,
        [0.9, 0.5, 0.7, 0.2, -0.1, 0.15],
        1.3,
    )
    .unwrap()
}

#[test]
fn derivatives_match_central_differences() {
    let density = CauchyDensity::try_new(
        0.2,
        vec![
            sheared(),
            CauchyKernel::try_new(DVec3::ZERO, 0.8, 0.7, 2.1).unwrap(),
        ],
    )
    .unwrap();
    let y = DVec3::new(0.4, 0.2, -0.3);
    let eps = 1e-5;
    for axis in 0..3 {
        let e = DVec3::AXES[axis] * eps;
        let g = (density.rho(y + e) - density.rho(y - e)) / (2.0 * eps);
        assert!((density.grad(y)[axis] - g).abs() < 1e-8);
        let hcol = (density.grad(y + e) - density.grad(y - e)) / (2.0 * eps);
        let h = sym_mat(density.hessian(y));
        assert!((h.col(axis) - hcol).length() < 1e-7);
    }
}

#[test]
fn centre_value_and_hessian() {
    let k = sheared();
    assert!((kernel_rho(&k, k.center) - k.weight).abs() < 1e-15);
    let h = kernel_hessian(&k, k.center);
    for i in 0..6 {
        assert!((h[i] + 2.0 * k.alpha * k.weight * k.metric[i]).abs() < 1e-14);
    }
}

#[test]
fn missing_alpha_is_rejected() {
    let err = CauchyKernel::try_new(DVec3::ZERO, 1.0, 1.0, f64::NAN).unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
    assert!(
        CauchyKernel::try_anisotropic(DVec3::ZERO, 1.0, [1.0, 1.0, -1.0, 0.0, 0.0, 0.0], 1.0)
            .is_err()
    );
}

#[test]
fn ln_gamma_matches_factorials() {
    assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-13);
    assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-13);
}
