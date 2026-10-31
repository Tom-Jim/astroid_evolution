//! Unit tests of `scf::orbit` (§10.4.6, hypothesis A-O).

use std::sync::Arc;

use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::forward::{Forward, Loading};
use astroid_evolution::scf::orbit::{Necks, OrbitLaw, OrbitParams};
use astroid_evolution::scf::spin::SpinSchedule;

/// Phobos today: GM of Mars, a₀ = 9376 km, ȧ₀ = −3.75 cm/yr, over 40 Myr.
fn phobos() -> (SpinSchedule, OrbitParams) {
    let spin = SpinSchedule::parse("orbit:4.282837e13,9.376e6,-1.1879e-9,1.2e15").unwrap();
    (spin, OrbitParams { primary_radius: 3.3895e6, limit_every: 0 })
}

/// Without mass loss the exact update is the closed law of §10.4.1 at every step boundary
/// (Proposition 10.4.6.1), whatever the steps.
#[test]
fn without_loss_the_update_is_the_closed_orbit_law() {
    let (spin, prm) = phobos();
    let m0 = 1.0603e16;
    let law = OrbitLaw::new(&prm, &spin, m0).unwrap();
    let mut s = OrbitLaw::start(&spin, 0.0, m0);
    let mut t = 0.0;
    for k in 0..40 {
        let tau = 1.0e12 * (1.0 + (k % 7) as f64);
        law.advance(&mut s, tau, 0.0).unwrap();
        t += tau;
        let want = spin.semi_major_axis(t);
        assert!((s.a - want).abs() <= 1e-12 * want, "t = {t:e}: {} vs {want}", s.a);
        assert!((law.mean_motion(s.a).powi(2) - spin.tide(t)).abs() <= 1e-10 * spin.tide(t));
    }
    assert_eq!((s.mass, s.escaped), (m0, 0.0));
}

/// With losses the decay runs at the mass of each step: `a^{13/2}` drops by `κ M_n τ_n`, the
/// ledger `M + escaped` stays `M₀` exactly, and a lighter satellite decays more slowly.
#[test]
fn losses_slow_the_decay_by_the_mass_of_each_step() {
    let (spin, prm) = phobos();
    let m0 = 1.0603e16;
    let law = OrbitLaw::new(&prm, &spin, m0).unwrap();
    let kappa = -6.5 * 9.376e6_f64.powf(5.5) * -1.1879e-9 / m0;
    let mut s = OrbitLaw::start(&spin, 0.0, m0);
    let mut light = s;
    let mut want = s.a.powf(6.5);
    for k in 0..10 {
        let (tau, lost) = (2.0e13, 0.01 * s.mass * (k as f64 + 1.0));
        want -= kappa * s.mass * tau;
        law.advance(&mut s, tau, lost).unwrap();
        assert!((s.a.powf(6.5) - want).abs() <= 1e-12 * want, "step {k}");
        assert!((s.mass + s.escaped - m0).abs() <= 1e-6, "ledger {} + {}", s.mass, s.escaped);
        law.advance(&mut light, tau, 0.0).unwrap();
        assert!(s.a >= light.a, "the losing satellite decayed faster");
    }
    assert!(law.adot(s.a, s.mass).abs() < law.adot(s.a, m0).abs());
}

/// Extreme losses (99.9 % in one step, then nearly all of the rest) never let the orbit
/// cross the primary: the rate falls with the mass, the steps bounded by `Δt_ω` and the
/// time to the primary end the orbit exactly there, a step past it is refused, and a
/// loss of the whole mass is refused.
#[test]
fn extreme_losses_keep_the_orbit_above_the_primary() {
    let (spin, prm) = phobos();
    let m0 = 1.0603e16;
    let law = OrbitLaw::new(&prm, &spin, m0).unwrap();
    let mut s = OrbitLaw::start(&spin, 0.0, m0);
    let n_dot = law.ndot(s.a, s.mass);
    law.advance(&mut s, 1.0e13, 0.999 * m0).unwrap();
    // A thousandfold lighter satellite: a thousandfold slower decay.
    let ratio = law.ndot(s.a, s.mass) / n_dot;
    assert!((ratio - 1e-3).abs() < 2e-5, "ṅ ratio {ratio}");
    let whole = s.mass;
    assert!(law.advance(&mut s.clone(), 1.0, whole).is_err(), "the whole mass was lost");
    assert!(law.advance(&mut s.clone(), 1.0, -1.0).is_err(), "a negative loss was taken");
    let too_long = law.time_to_floor(&s) * (1.0 + 1e-6);
    assert!(law.advance(&mut s.clone(), too_long, 0.0).is_err(), "a step crossed the primary");

    let (eps, omega_ref) = (0.01, 2.2795e-4);
    let mut steps = 0;
    while !law.ended(&s) {
        let n = law.mean_motion(s.a);
        let dt_omega = eps * n.max(omega_ref) / law.ndot(s.a, s.mass).abs();
        let dt = dt_omega.min(law.time_to_floor(&s));
        let a = s.a;
        let half = 0.5 * s.mass;
        law.advance(&mut s, dt, half).unwrap();
        assert!(s.a <= a && s.a >= prm.primary_radius, "a went from {a} to {}", s.a);
        // Δt_ω bounds the first-order change of n by ε; the exact change is at most a few
        // per cent above it.
        assert!(law.mean_motion(s.a) <= n * (1.0 + 1.05 * eps), "n jumped from {n} to {}", law.mean_motion(s.a));
        steps += 1;
        assert!(steps < 100_000, "the orbit never reached the primary");
    }
    assert_eq!(s.a, prm.primary_radius);
    assert!(s.mass > 0.0 && (s.mass + s.escaped - m0).abs() <= 1e-6 * m0);
}

/// A uniform sphere has the point-mass field outside: its necks are the Hill saddles at
/// `±r_H e_x`, `r_H = (GM / 3n²)^{1/3}`, with the axes `±e_x`.
#[test]
fn the_necks_of_a_sphere_are_the_hill_points() {
    let (topo, dirs) = Topology::icosphere(4);
    let r = 1000.0;
    let nv = dirs.len();
    let body = Body::new(Arc::new(topo), dirs.iter().map(|d| *d * r).collect(), 2000.0, vec![], vec![0.0; nv]).unwrap();
    let g = 6.6743e-11;
    let fwd = Forward::bind(&body, g).unwrap();
    let mass = fwd.moments.mass;
    let r_hill = 2.5 * r;
    let n2 = g * mass / (3.0 * r_hill.powi(3));
    let necks = Necks::locate(&fwd, g, mass, Loading { omega: n2.sqrt(), tide: n2 }).unwrap();
    assert!((necks.r_hill - r_hill).abs() < 1e-9 * r_hill);
    for (neck, side) in necks.necks.iter().zip([1.0, -1.0]) {
        let want = fwd.moments.centroid + DVec3::X * (side * r_hill);
        assert!((neck.point - want).length() < 2e-3 * r_hill, "{:?} vs {want:?}", neck.point);
        assert!(neck.axis.dot(DVec3::X * side) > 0.9999, "{:?}", neck.axis);
    }
    // Γ lies inside both planes: a negative margin of about 1 − R/r_H.
    let margin = necks.margin(&body.positions);
    assert!((margin + 1.0 - r / r_hill).abs() < 0.01, "margin {margin}");
    assert!(!necks.beyond(DVec3::X * (0.99 * r_hill)) && necks.beyond(DVec3::X * (1.01 * r_hill)));
    assert!(necks.beyond(DVec3::X * (-1.01 * r_hill)));
    // The cap brings a point on the axis onto the plane ahead; none lies behind.
    let cap = necks.cap(DVec3::X * r, DVec3::X).unwrap();
    assert!((cap - (r_hill - r)).abs() < 2e-3 * r_hill);
    assert!(necks.cap(DVec3::Z * r, DVec3::Z).is_none());
}

/// The orbit section needs an orbit schedule and a decaying orbit.
#[test]
fn the_orbit_law_is_refused_without_a_decaying_orbit() {
    let prm = OrbitParams { primary_radius: 3.3895e6, limit_every: 0 };
    let table = SpinSchedule::parse("const:1e-4@1e10").unwrap();
    assert!(OrbitLaw::new(&prm, &table, 1e16).is_err());
    let rising = SpinSchedule::parse("orbit:4.282837e13,2.3463e7,1.09e-12,1e15").unwrap();
    assert!(OrbitLaw::new(&prm, &rising, 1e16).is_err());
}
