//! Unit tests of `scf::harmonics`, moved out of `src/scf/harmonics.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rheo::basis::{lm_index, sh_all};
    pub use astroid_evolution::scf::jets::Jet;
    pub use nalgebra::{Cholesky, DMatrix, DVector};
    pub use std::f64::consts::{PI, SQRT_2};
}
use ctx::*;

use astroid_evolution::scf::harmonics::*;

fn point(i: usize, n: usize) -> DVec3 {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
    let r = (1.0 - z * z).sqrt();
    DVec3::new(
        r * (golden * i as f64).cos(),
        r * (golden * i as f64).sin(),
        z,
    )
}

#[test]
fn jets_reproduce_the_scalar_recursion_and_its_gradient() {
    let l_max = 5;
    let w = point(7, 31);
    let x = [
        Jet::var(0, w.x, 3),
        Jet::var(1, w.y, 3),
        Jet::var(2, w.z, 3),
    ];
    let jets = sh_jets(l_max, &x);
    let plain = sh_all(l_max, w.to_array());
    let h = 1e-6;
    for k in 0..plain.len() {
        assert!((jets[k].val() - plain[k]).abs() < 1e-13, "value {k}");
        // The recursion is a polynomial in the arguments, so shifting the argument
        // (without renormalising) must follow the jet.
        let shifted = |axis: usize, s: f64| {
            let mut p = w.to_array();
            p[axis] += s;
            sh_all(l_max, p)[k]
        };
        for axis in 0..3 {
            let fd = (shifted(axis, h) - shifted(axis, -h)) / (2.0 * h);
            assert!(
                (jets[k].grad()[axis] - fd).abs() < 1e-6,
                "gradient {k} {axis}"
            );
        }
    }
}

