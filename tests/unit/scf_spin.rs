//! Unit tests of `scf::spin`, moved out of `src/scf/spin.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
}
use ctx::*;

use astroid_evolution::scf::spin::*;

#[test]
fn stages_hold_omega_until_their_end() {
    let s = SpinSchedule::parse("stages:100@1e-4,50@2e-4").unwrap();
    assert_eq!(s.t_end(), 150.0);
    let a = s.sample(10.0);
    assert_eq!((a.omega, a.t_next), (1e-4, 100.0));
    let b = s.sample(100.0);
    assert_eq!((b.omega, b.t_next), (2e-4, 150.0));
}

#[test]
fn linear_and_table_interpolate() {
    let l = SpinSchedule::parse("linear:1e-4,1e-7,1000").unwrap();
    assert!((l.sample(500.0).omega - 1.5e-4).abs() < 1e-18);
    let t = SpinSchedule::parse("table:0@1e-4,10@3e-4").unwrap();
    assert!((t.sample(5.0).omega - 2e-4).abs() < 1e-18);
    assert!(SpinSchedule::parse("linear:1e-4,-1e-3,1000").is_err());
    assert!(SpinSchedule::parse("table:0@1,0@2").is_err());
    assert!(SpinSchedule::parse("sometimes:1").is_err());
}

#[test]
fn orbit_is_synchronous_and_decays() {
    // Phobos today: a = 9376 km, ȧ ≈ −1.8 cm/yr.
    let yr = 3.15576e7;
    let s = SpinSchedule::parse(&format!(
        "orbit:4.282837e13,9.376e6,{},{}",
        -0.018 / yr,
        1e7 * yr
    ))
    .unwrap();
    let n0 = s.sample(0.0).omega;
    assert!(
        (2.0 * std::f64::consts::PI / n0 / 3600.0 - 7.65).abs() < 0.01,
        "period {} h",
        2.0 * std::f64::consts::PI / n0 / 3600.0
    );
    assert!(s.semi_major_axis(1e7 * yr) < 9.376e6);
    assert!((s.tide(0.0) - n0 * n0).abs() < 1e-20);
}

#[test]
fn a_table_that_starts_late_holds_its_first_spin_until_the_first_node() {
    let t = SpinSchedule::parse("table:100@1e-4,200@3e-4").unwrap();
    let early = t.sample(10.0);
    assert_eq!(
        (early.omega, early.omega_dot, early.t_next),
        (1e-4, 0.0, 100.0)
    );
    let first = t.sample(100.0);
    assert!((first.omega_dot - 2e-6).abs() < 1e-18);
    assert_eq!(first.t_next, 200.0);
}
