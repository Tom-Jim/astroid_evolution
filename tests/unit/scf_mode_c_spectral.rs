//! Unit tests of `scf::mode_c_spectral`, moved out of `src/scf/mode_c_spectral.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::scf::quad::gauss_legendre;
    pub use astroid_evolution::shsmooth::harmonics::{coeff_count, sh_all_into};
    pub use rayon::prelude::*;
    pub use std::sync::Mutex;
}
use ctx::*;

use astroid_evolution::scf::mode_c_spectral::*;
use astroid_evolution::shsmooth::harmonics::{lm_index, sh_all};

fn legendre(l: usize, x: f64) -> f64 {
    let mut p0 = 1.0;
    if l == 0 {
        return p0;
    }
    let mut p1 = x;
    if l == 1 {
        return p1;
    }
    for k in 1..l {
        let kf = k as f64;
        let next = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
        p0 = p1;
        p1 = next;
    }
    p1
}

#[test]
fn addition_theorem_matches_the_legendre_polynomial() {
    let a = sh_all(4, [0.3, -0.4, 0.8660254037844386]);
    let raw = [-0.2_f64, 0.5, 0.842614978];
    let n = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
    let dir_b = [raw[0] / n, raw[1] / n, raw[2] / n];
    let b = sh_all(4, dir_b);
    let dir_a = [0.3, -0.4, 0.8660254037844386];
    let dot = dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1] + dir_a[2] * dir_b[2];
    for l in 0..=4 {
        let mut sum = 0.0;
        for m in -(l as i32)..=l as i32 {
            sum += a[lm_index(l, m)] * b[lm_index(l, m)];
        }
        let expect = (2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * legendre(l, dot);
        assert!(
            (sum - expect).abs() < 1.0e-12,
            "l={l} sum={sum} expect={expect}"
        );
    }
}

#[test]
fn ryugu_width_is_inside_the_truncation_gate() {
    let beta = spectrum(0.2).expect("kernel width 0.2");
    assert_eq!(beta.len(), L_MAX + 1);
    assert!(beta[0] > 0.0);
}
