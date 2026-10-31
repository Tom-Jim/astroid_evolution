//! Unit tests of `scf::nbody` (§12.3–12.5).

use std::path::PathBuf;

use astroid_evolution::common::math::{DMat3, DVec3};
use astroid_evolution::scf::body::Body;
use astroid_evolution::scf::breakup::{BreakupParams, Piece, Rigid};
use astroid_evolution::scf::forward::Forward;
use astroid_evolution::scf::nbody::gravity::{NEAR, forces};
use astroid_evolution::scf::nbody::{Fragment, G, Primary, Ring, System, free_rotation};
use astroid_evolution::scf::params::Project;

fn project() -> Project {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("dist/.stage/assets/projects");
    let text = std::fs::read_to_string(dir.join("phobos_breakup.toml")).unwrap();
    Project::parse(&text, dir).unwrap()
}

/// A uniform ellipsoid of semi-axes `axes` (m) as a piece in its principal frame.
fn ellipsoid(axes: [f64; 3], level: usize) -> Piece {
    let (topo, pos) = Body::ellipsoid(level, axes);
    let nv = pos.len();
    let body = Body::new(topo, pos, 1900.0, vec![], vec![0.0; nv]).unwrap();
    let fwd = Forward::bind(&body, G).unwrap();
    Piece { body, fwd, com: DVec3::ZERO, rot: DMat3::IDENTITY, motion: Rigid { a: DVec3::ZERO, theta: DVec3::ZERO, c: DVec3::ZERO } }
}

/// Far from the origin, where a negligible primary sits.
const AWAY: DVec3 = DVec3::new(1.0e7, 0.0, 0.0);

fn conserved(frags: &[Fragment]) -> (DVec3, DVec3) {
    frags.iter().fold((DVec3::ZERO, DVec3::ZERO), |(p, l), f| (p + f.v * f.mass, l + f.x.cross(f.v) * f.mass + f.l))
}

/// The splitting of the free rigid body keeps the angular momentum in space exactly, the
/// body-frame momentum on its sphere, and the rotational energy to second order.
#[test]
fn the_free_rigid_body_keeps_its_invariants() {
    let inertia = DVec3::new(1.0, 2.0, 3.5);
    let l = DVec3::new(0.3, 1.0, 0.2);
    let mut rot = DMat3::IDENTITY;
    let energy = |rot: &DMat3| {
        let pi = rot.transpose() * l;
        0.5 * (pi / inertia).dot(pi)
    };
    let e0 = energy(&rot);
    for _ in 0..20_000 {
        free_rotation(&mut rot, l, inertia, 0.01);
    }
    let e1 = energy(&rot);
    assert!(((rot.transpose() * rot) - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, 1e-10), "rotation drifted from orthogonal");
    assert!((e1 - e0).abs() < 1e-4 * e0, "energy {e0} -> {e1}");
}

/// Two uniform spheres attract as point masses, and the pair force is continuous across the
/// switch from the RTFP volume integral to the quadrupole expansion.
#[test]
fn near_and_far_pair_forces_agree() {
    let primary = Primary { gm: 1e-20, radius: 1.0, j2: 0.0 };
    let pair = |b: Piece, d: f64| {
        let a = Fragment::new(0, ellipsoid([1000.0; 3], 4), AWAY, DVec3::ZERO, DMat3::IDENTITY, DVec3::ZERO).unwrap();
        let b = Fragment::new(1, b, AWAY + DVec3::new(d, 0.0, 0.0), DVec3::ZERO, DMat3::IDENTITY, DVec3::ZERO).unwrap();
        let s = a.radius + b.radius;
        let (f, _) = forces(&primary, &[a, b]).unwrap();
        (f, s)
    };
    let (f, s) = pair(ellipsoid([600.0; 3], 4), 4000.0);
    assert!(4000.0 < NEAR * s, "the sphere pair must be close");
    let (ma, mb) = (1900.0 * 4.0 / 3.0 * std::f64::consts::PI * 1e9, 1900.0 * 4.0 / 3.0 * std::f64::consts::PI * 2.16e8);
    let want = G * ma * mb / 4000.0_f64.powi(2);
    assert!((f[1].0.x + want).abs() < 5e-3 * want, "sphere pair {} vs {want}", f[1].0.x);
    assert!((f[0].0 + f[1].0).length() < 1e-9 * want, "action and reaction");

    let elongated = || ellipsoid([900.0, 400.0, 300.0], 4);
    let s = pair(elongated(), 1e9).1;
    let (inside, _) = pair(elongated(), 0.999 * NEAR * s);
    let (outside, _) = pair(elongated(), 1.001 * NEAR * s);
    // At one distance: the forces times d².
    let (fi, fo) = (inside[1].0 * 0.999f64.powi(2), outside[1].0 * 1.001f64.powi(2));
    assert!((fi - fo).length() < 3e-3 * fo.length(), "near {fi:?} vs far {fo:?}");
}

/// About a primary, two fragments keep the total angular momentum to rounding and the
/// energy to second order in the step (far apart, where the energy is exact).
#[test]
fn fragments_about_a_primary_conserve_angular_momentum_and_energy() {
    let p = project();
    let prm: BreakupParams = p.breakup.clone().unwrap();
    let primary = Primary { gm: 4.282837e13, radius: 3.3895e6, j2: 0.0 };
    let mut sys = System::new(&p, &prm, primary);
    let a = 7.0e6;
    let v = (primary.gm / a).sqrt();
    for (x, piece) in [(DVec3::new(a, 0.0, 0.0), ellipsoid([900.0, 500.0, 400.0], 3)), (DVec3::new(a, 2.0e5, 0.0), ellipsoid([700.0, 600.0, 500.0], 3))] {
        let dir = DVec3::Z.cross(x).normalize();
        sys.add_pieces(vec![piece], x, dir * v, DMat3::IDENTITY, DVec3::new(0.0, 1e-4, v / a)).unwrap();
    }
    let (_, l0) = conserved(&sys.fragments);
    let e0 = sys.frame().unwrap().energy;
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / primary.gm).sqrt();
    let dt = period / 2000.0;
    for _ in 0..2000 {
        sys.advance(dt).unwrap();
    }
    let (_, l1) = conserved(&sys.fragments);
    let e1 = sys.frame().unwrap().energy;
    assert!((l1 - l0).length() < 1e-11 * l0.length(), "angular momentum {l0:?} -> {l1:?}");
    assert!((e1 - e0).abs() < 1e-5 * e0.abs(), "energy {e0} -> {e1}");
}

/// Touching fragments that approach merge into one body with their mass, momentum and
/// angular momentum.
#[test]
fn a_merger_keeps_mass_momentum_and_angular_momentum() {
    let p = project();
    let prm: BreakupParams = p.breakup.clone().unwrap();
    let primary = Primary { gm: 1e-20, radius: 1.0, j2: 0.0 };
    let mut sys = System::new(&p, &prm, primary);
    sys.add_pieces(vec![ellipsoid([800.0, 500.0, 400.0], 3)], AWAY, DVec3::new(0.01, 0.0, 0.0), DMat3::IDENTITY, DVec3::new(0.0, 0.0, 2e-4))
        .unwrap();
    sys.add_pieces(vec![ellipsoid([500.0; 3], 3)], AWAY + DVec3::new(1250.0, 200.0, 0.0), DVec3::new(-0.02, 0.0, 0.0), DMat3::IDENTITY, DVec3::new(1e-4, 0.0, 0.0))
        .unwrap();
    let mass: f64 = sys.fragments.iter().map(|f| f.mass).sum();
    let (p0, l0) = conserved(&sys.fragments);
    sys.advance(1.0).unwrap();
    assert_eq!(sys.fragments.len(), 1, "the touching pair did not merge");
    let (p1, l1) = conserved(&sys.fragments);
    assert!((sys.fragments[0].mass - mass).abs() < 1e-9 * mass, "mass {} vs {mass}", sys.fragments[0].mass);
    assert!((p1 - p0).length() < 1e-9 * p0.length(), "momentum {p0:?} -> {p1:?}");
    assert!((l1 - l0).length() < 1e-9 * l0.length(), "angular momentum {l0:?} -> {l1:?}");
    assert_eq!(sys.events.iter().filter(|e| e.kind == "merge").count(), 1);
}

/// The ring conserves mass with its two sinks, keeps `Σ ≥ 0`, and spreads.
#[test]
fn the_ring_conserves_mass_and_spreads() {
    let (gm, rp) = (4.282837e13, 3.3895e6);
    let mut ring = Ring::new(gm, rp, 200, 1.0, 2420.0);
    let mid = 0.5 * (ring.inner() + ring.roche());
    let mass = 1.0e16;
    ring.deposit(mid, mass);
    let width = |ring: &Ring| {
        let m: Vec<f64> = ring.r.iter().zip(&ring.sigma).map(|(r, s)| r * s).collect();
        let total: f64 = m.iter().sum();
        let c = ring.r.iter().zip(&m).map(|(r, w)| r * w).sum::<f64>() / total;
        (ring.r.iter().zip(&m).map(|(r, w)| (r - c).powi(2) * w).sum::<f64>() / total).sqrt()
    };
    assert!(width(&ring) < ring.dr, "the deposit is not in one cell");
    // A Phobos mass in one cell has ν ≈ 0.02 m²/s: a million years spreads it over cells.
    let steps = ring.evolve(3.15576e13).unwrap();
    assert!(steps > 0);
    let total = ring.mass() + ring.to_primary + ring.to_moonlets;
    assert!((total - mass).abs() < 1e-9 * mass, "ring {} + primary {} + moonlets {} vs {mass}", ring.mass(), ring.to_primary, ring.to_moonlets);
    assert!(ring.sigma.iter().all(|s| *s >= 0.0));
    assert!(width(&ring) > 3.0 * ring.dr, "the ring did not spread: width {}", width(&ring));
    // Inside the primary or beyond the Roche limit nothing stays in the ring.
    ring.deposit(0.5 * rp, 1.0);
    ring.deposit(2.0 * ring.roche(), 2.0);
    assert!((ring.mass() + ring.to_primary + ring.to_moonlets - mass - 3.0).abs() < 1e-9 * mass);
}
