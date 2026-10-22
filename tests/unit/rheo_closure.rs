//! Unit tests of `rheo::closure`, moved out of `src/rheo/closure.rs`.

mod ctx {
    pub use astroid_evolution::rheo::params::{Material, MuProjMode};
}
use ctx::*;

use astroid_evolution::rheo::closure::*;

fn case() -> DegeneracyCase {
    DegeneracyCase {
        pressure: 2000.0,
        nu: 0.6,
        z: 6.0,
        d_p: 0.005,
        rho_p: 2500.0,
        f_h: 0.0,
        c_c: 1.0,
        c_t: 1.0,
        dev_h_frob: 0.0,
        shear: 4.0,
        mu_s: 0.38,
        mu_2: 0.64,
        i_0: 0.279,
        cohesion: 0.0,
        gamma_eps: 0.0,
    }
}

#[test]
fn classical_limit_is_exact_without_extensions() {
    let r = degeneracy(&case());
    assert!(r.cohesion_ratio < 1e-15 && r.tide_ratio < 1e-15);
    assert!(r.inertial_rel_diff < 1e-14);
    assert!(r.friction_abs_diff < 1e-14);
    assert!(r.flow_stress_rel_diff < 1e-14);
}

#[test]
fn extensions_vanish_in_the_stated_limits() {
    // F_H, cohesion, tide and regularisation all scaled by a vanishing parameter.
    let mut last = f64::INFINITY;
    for e in [1e-2, 1e-4, 1e-6, 1e-8] {
        let mut c = case();
        c.f_h = 1e-3 * e;
        c.cohesion = 2000.0 * e;
        c.dev_h_frob = 1.0 * e;
        c.gamma_eps = 4.0 * e;
        let r = degeneracy(&c);
        assert!(r.flow_stress_rel_diff < last, "monotone approach, e={e}");
        last = r.flow_stress_rel_diff;
        assert!(r.cohesion_ratio < 1e-1 * e.max(1e-8) * 1e3);
    }
    assert!(last < 1e-7, "{last}");
}

#[test]
fn mu_is_monotone_with_the_stated_range() {
    let mut prev = 0.0;
    for k in 0..60 {
        let mu = mu_of_inertial(10f64.powf(-6.0 + 0.2 * k as f64), 0.38, 0.64, 0.279);
        assert!(mu >= prev && mu >= 0.38 && mu < 0.64);
        prev = mu;
    }
}
