//! Unit tests of `rheo::basis`, moved out of `src/rheo/basis.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::rheo::model::{StarSurface, VelocityField};
    pub use std::f64::consts::PI;
}
use ctx::*;

use astroid_evolution::rheo::basis::*;

#[test]
fn sigma_kappa_match_the_jacobi_polynomials() {
    for l in 1..=8 {
        for k in 0..K {
            let p = psi_poly(l, k);
            assert!(
                (poly_eval(&p, 1.0) - 1.0).abs() < 1e-9,
                "psi(1) l={l} k={k}"
            );
            let d1 = poly_derivative(&p);
            let d2 = poly_derivative(&d1);
            let s = poly_eval(&d1, 1.0);
            let kp = poly_eval(&d2, 1.0);
            assert!(
                (s - sigma(l, k)).abs() < 1e-7 * s.abs().max(1.0),
                "sigma l={l} k={k}"
            );
            assert!(
                (kp - kappa(l, k)).abs() < 1e-7 * kp.abs().max(1.0),
                "kappa l={l} k={k}"
            );
        }
    }
}

#[test]
fn jacobi_basis_is_l2_orthogonal() {
    // int psi_j psi_k zeta^2 dzeta = delta / (2k + 2l + 3).
    let l = 3;
    for j in 0..K {
        for k in 0..K {
            let (pj, pk) = (psi_poly(l, j), psi_poly(l, k));
            // Exact: multiply the monomial coefficients and integrate term by term.
            let mut acc = 0.0;
            for (a, ca) in pj.iter().enumerate() {
                for (b, cb) in pk.iter().enumerate() {
                    acc += ca * cb / (a + b + 3) as f64;
                }
            }
            let want = if j == k {
                1.0 / (2 * k + 2 * l + 3) as f64
            } else {
                0.0
            };
            assert!((acc - want).abs() < 1e-9, "j={j} k={k} {acc}");
        }
    }
}

#[test]
fn rigid_modes_have_no_surface_shear_constraint() {
    for l in 1..=1 {
        assert_eq!(constraint_poloidal(l, 0), 0.0);
        assert_eq!(constraint_toroidal(l, 0), 0.0);
    }
}

#[test]
fn harmonics_are_orthonormal() {
    let l_max = 5;
    let n = 60000;
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let count = (l_max + 1) * (l_max + 1);
    let mut gram = vec![0.0; count * count];
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = golden * i as f64;
        let y = sh_all(l_max, [r * phi.cos(), r * phi.sin(), z]);
        for a in 0..count {
            for b in 0..count {
                gram[a * count + b] += y[a] * y[b] * 4.0 * PI / n as f64;
            }
        }
    }
    for a in 0..count {
        for b in 0..count {
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((gram[a * count + b] - want).abs() < 2e-3, "{a} {b}");
        }
    }
}
