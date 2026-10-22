//! Unit tests of `scf::jets`, moved out of `src/scf/jets.rs`.

mod ctx {
    pub use std::sync::OnceLock;
}
use ctx::*;

use astroid_evolution::scf::jets::*;

#[test]
fn product_and_derivative_follow_the_leibniz_rule() {
    let (x, y, z) = (
        Jet::var(0, 0.7, 4),
        Jet::var(1, -0.3, 4),
        Jet::var(2, 1.1, 4),
    );
    let f = x.mul(&y).mul(&z).add(&x.mul(&x));
    // f = x y z + x^2
    assert!((f.val() - (0.7 * -0.3 * 1.1 + 0.49)).abs() < 1e-14);
    let fx = f.deriv(0);
    assert!((fx.val() - (-0.3 * 1.1 + 1.4)).abs() < 1e-14);
    let fxx = fx.deriv(0);
    assert!((fxx.val() - 2.0).abs() < 1e-14);
    let fyz = f.deriv(1).deriv(2);
    assert!((fyz.val() - 0.7).abs() < 1e-14);
}

#[test]
fn composition_matches_closed_forms() {
    let x = Jet::var(0, 2.0, 4);
    let s = x.sqrt();
    let h = 0.1;
    assert!((s.eval([h, 0.0, 0.0]) - (2.0 + h).sqrt()).abs() < 1e-7);
    let r = x.recip();
    assert!((r.eval([h, 0.0, 0.0]) - 1.0 / (2.0 + h)).abs() < 1e-6);
    let ms = x.mul_x(0, 2.0);
    // (2 + h)(2 + h) = 4 + 4h + h^2
    assert!((ms.eval([h, 0.0, 0.0]) - (2.0 + h) * (2.0 + h)).abs() < 1e-14);
}

#[test]
fn a_zero_homogeneous_jet_scales_exactly_along_a_ray() {
    // u_x = x/|x| is homogeneous of degree 0, so ∂^α u(ρ û) = ρ^{-|α|} ∂^α u(û).
    // The basis reuses one directional jet for every radius on a ray by this rule.
    let dir = {
        let p = [0.2_f64, -0.5, 0.7];
        let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    };
    let ord = 4;
    let at = |rho: f64| {
        let p = [dir[0] * rho, dir[1] * rho, dir[2] * rho];
        let x = [
            Jet::var(0, p[0], ord),
            Jet::var(1, p[1], ord),
            Jet::var(2, p[2], ord),
        ];
        let r2 = x[0].mul(&x[0]).add(&x[1].mul(&x[1])).add(&x[2].mul(&x[2]));
        x[0].mul(&r2.sqrt().recip())
    };
    let j0 = at(1.7);
    let j1 = at(4.2);
    let ratio: f64 = 1.7 / 4.2;
    let deg = &tables().deg;
    for i in 0..count(ord) {
        let scaled = j0.c[i] * ratio.powi(i32::from(deg[i]));
        assert!(
            (scaled - j1.c[i]).abs() <= 1e-9 * (1.0 + scaled.abs()),
            "coeff {i}: {scaled} vs {}",
            j1.c[i]
        );
    }
}
