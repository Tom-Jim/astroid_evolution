//! Unit tests of `scf::cycle` (§10.4.7, hypothesis A-Λ) and of the primary's light (§11.6).

use astroid_evolution::common::math::{DMat3, DVec3};
use astroid_evolution::scf::cycle::{Cycle, Libration, chi_of, eccentricity_coefficient};
use astroid_evolution::scf::evolving::thermal::{PrimaryLight, eclipse_visible, primary_irradiance};
use astroid_evolution::scf::orbit::{OrbitLaw, OrbitParams};
use astroid_evolution::scf::spin::SpinSchedule;

/// Proposition 10.4.7.1(1): the classical 19/8 above 3n/2, 25/8 when every lag is reversed
/// (Phobos), and the regime of a primary between n/2 and n.
#[test]
fn the_eccentricity_coefficient_follows_the_signs_of_the_lags() {
    let n = 1.0e-4;
    assert!((eccentricity_coefficient(10.0 * n, n) - 19.0 / 8.0).abs() < 1e-15);
    assert!((eccentricity_coefficient(0.3 * n, n) - 25.0 / 8.0).abs() < 1e-15);
    // Phobos: Mars spins at 7.088e-5 rad/s, Phobos orbits at 2.28e-4.
    assert!((eccentricity_coefficient(7.0882e-5, 2.2795e-4) - 3.125).abs() < 1e-15);
    // n/2 < Ω < n: S₁ = −1, S₂ = +1, S₃ = −1.
    let want = (0.75 - 3.0 / 16.0 - 147.0 / 16.0 - 9.0 / 8.0) / -3.0;
    assert!((eccentricity_coefficient(0.7 * n, n) - want).abs() < 1e-15);
}

/// Proposition 10.4.7.1(2): over any steps of A-O, `e = e₀ (a/a₀)^x` exactly.
#[test]
fn the_eccentricity_follows_the_semi_major_axis_exactly() {
    let spin = SpinSchedule::parse("orbit:4.282837e13,9.376e6,-1.1879e-9,1.2e15").unwrap();
    let prm = OrbitParams { primary_radius: 3.3895e6, limit_every: 0 };
    let m0 = 1.0603e16;
    let law = OrbitLaw::new(&prm, &spin, m0).unwrap();
    let mut s = OrbitLaw::start(&spin, 0.0, m0);
    s.e = 0.0151;
    let (a0, x) = (s.a, 6.1);
    for k in 0..30 {
        law.advance(&mut s, 1.0e13 * (1.0 + (k % 5) as f64), 0.0, x).unwrap();
        let want = 0.0151 * (s.a / a0).powf(x);
        assert!((s.e - want).abs() <= 1e-13 * want, "e = {} vs {want}", s.e);
    }
    assert!(s.a < 0.9 * a0 && s.e < 0.6 * 0.0151, "the orbit and e did not decay: {s:?}");
}

/// Proposition 10.4.7.2: the steady state of the damped libration equation, integrated
/// directly from rest over many damping times, is `Γ`, amplitude and phase.
#[test]
fn the_libration_is_the_steady_state_of_its_equation() {
    let (n, e, chi, q) = (1.0, 0.0151, 0.39, 100.0);
    let lib = Libration::new(e, chi, q);
    let rhs = |t: f64, y: [f64; 2]| [y[1], -chi.sqrt() * n / q * y[1] - chi * n * n * (y[0] - 2.0 * e * (n * t).sin())];
    let (mut t, mut y) = (0.0, [0.0, 0.0]);
    let h = 2.0 * std::f64::consts::PI / n / 400.0;
    let orbits = 4000;
    for _ in 0..orbits * 400 {
        let k1 = rhs(t, y);
        let k2 = rhs(t + h / 2.0, [y[0] + h / 2.0 * k1[0], y[1] + h / 2.0 * k1[1]]);
        let k3 = rhs(t + h / 2.0, [y[0] + h / 2.0 * k2[0], y[1] + h / 2.0 * k2[1]]);
        let k4 = rhs(t + h, [y[0] + h * k3[0], y[1] + h * k3[1]]);
        for i in 0..2 {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
    }
    // At t = 2π·orbits, M ≡ 0: γ = γ_c and γ̇ = n γ_s.
    let free = (-0.5 * chi.sqrt() * n / q * t).exp();
    assert!(free < 1e-6, "the free libration has not decayed: {free}");
    assert!((y[0] - lib.gamma_c).abs() < 1e-6 * lib.amplitude(), "γ = {} vs γ_c = {}", y[0], lib.gamma_c);
    assert!((y[1] / n - lib.gamma_s).abs() < 1e-6 * lib.amplitude(), "γ̇/n = {} vs γ_s = {}", y[1] / n, lib.gamma_s);
    // Undamped: −2eχ/(1 − χ), and resonance at χ = 1.
    let free = Libration::new(e, chi, f64::INFINITY);
    assert!((free.gamma_s + 2.0 * e * chi / (1.0 - chi)).abs() < 1e-15 && free.gamma_c == 0.0);
    assert!(Libration::new(e, 0.999, q).amplitude() > 10.0 * free.amplitude());
}

/// Phobos: the inertia of its triaxial ellipsoid (13.0 × 11.4 × 9.1 km) and e = 0.0151 give
/// the forced libration measured by Burmeister et al. (2018), 1.14° ± 0.03°.
#[test]
fn phobos_librates_by_the_measured_angle() {
    let (a, b, c) = (13.0e3_f64, 11.39e3_f64, 9.07e3_f64);
    let inertia = [b * b + c * c, a * a + c * c, a * a + b * b, 0.0, 0.0, 0.0];
    let chi = chi_of(inertia);
    let deg = Libration::new(0.0151, chi, 100.0).amplitude().to_degrees();
    assert!((chi - 0.39).abs() < 0.01, "χ = {chi}");
    assert!((deg - 1.14).abs() < 0.1, "libration {deg}° against the measured 1.14°");
}

/// The exact load in the body frame of a librating satellite on a Kepler orbit, less the
/// mean, per unit mass at `x`: the tide `(GM/r³)(3r̂r̂ᵀ − I)x`, the centrifugal `Ω² x_⊥` and
/// the Euler term `−Ω̇ e_z × x`, with the body angle `M + γ(M)`.
fn exact_cycle(n: f64, e: f64, lib: Libration, x: DVec3, m: f64) -> DVec3 {
    let mut ea = m;
    for _ in 0..50 {
        ea -= (ea - e * ea.sin() - m) / (1.0 - e * ea.cos());
    }
    let f = 2.0 * ((1.0 + e).sqrt() * (ea / 2.0).sin()).atan2((1.0 - e).sqrt() * (ea / 2.0).cos());
    let r = 1.0 - e * ea.cos();
    let (gs, gc) = (lib.gamma_s, lib.gamma_c);
    let gamma = gs * m.sin() + gc * m.cos();
    let psi = f - m - gamma;
    let rhat = DVec3::new(psi.cos(), psi.sin(), 0.0);
    let tide = (rhat * (3.0 * rhat.dot(x)) - x) * (n * n / r.powi(3));
    let spin = n * (1.0 + gs * m.cos() - gc * m.sin());
    let spin_dot = n * n * (-gs * m.sin() - gc * m.cos());
    let total = tide + DVec3::new(x.x, x.y, 0.0) * (spin * spin) - DVec3::Z.cross(x) * spin_dot;
    let mean = DVec3::new(2.0 * x.x, -x.y, -x.z) * (n * n) + DVec3::new(x.x, x.y, 0.0) * (n * n);
    total - mean
}

/// The boxed `δg` of §10.4.7 is the exact load to first order: its error falls as `e²`.
#[test]
fn the_cyclic_load_is_the_exact_field_to_first_order() {
    let n = 2.28e-4;
    let pts = [DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.3, -0.8, 0.4), DVec3::new(-0.5, 0.6, -0.7)];
    let err = |e: f64| {
        let lib = Libration::new(e, 0.39, 100.0);
        let c = Cycle::new(n, e, lib);
        let (mut worst, mut size) = (0.0_f64, 0.0_f64);
        for x in pts {
            for j in 0..16 {
                let m = Cycle::phase(j, 16);
                let want = exact_cycle(n, e, lib, x, m);
                worst = worst.max((c.delta_g(x, m) - want).length());
                size = size.max(want.length());
            }
        }
        (worst, size)
    };
    let (e1, s1) = err(0.02);
    let (e2, s2) = err(0.01);
    assert!(e1 < 0.1 * s1, "first-order error {e1:e} against the cycle {s1:e}");
    let order = (e1 / e2).log2();
    assert!((order - 2.0).abs() < 0.2, "the error falls as e^{order:.2}, not e²");
    assert!((s1 / s2 - 2.0).abs() < 0.05, "the cycle is not linear in e");
}

/// Proposition 10.4.7.3 on a uniform ellipsoid: no net force, no torque about x or y, and
/// about z only the damping torque C (√χ n / Q) γ̇.
#[test]
fn the_cyclic_load_is_self_equilibrated() {
    let (a, b, c) = (1.3_f64, 1.1_f64, 0.9_f64);
    let k = 24;
    let mut pts = Vec::new();
    for i in 0..k {
        for j in 0..k {
            for l in 0..k {
                let u = DVec3::new(i as f64 + 0.5, j as f64 + 0.5, l as f64 + 0.5) / k as f64 * 2.0 - DVec3::ONE;
                if u.length_squared() <= 1.0 {
                    pts.push(DVec3::new(a * u.x, b * u.y, c * u.z));
                }
            }
        }
    }
    let i_xx: f64 = pts.iter().map(|p| p.y * p.y + p.z * p.z).sum();
    let i_yy: f64 = pts.iter().map(|p| p.x * p.x + p.z * p.z).sum();
    let i_zz: f64 = pts.iter().map(|p| p.x * p.x + p.y * p.y).sum();
    let chi = chi_of([i_xx, i_yy, i_zz, 0.0, 0.0, 0.0]);
    for q in [f64::INFINITY, 100.0] {
        let lib = Libration::new(0.0151, chi, q);
        let cyc = Cycle::new(1.0, 0.0151, lib);
        for j in 0..8 {
            let m = Cycle::phase(j, 8);
            let force: DVec3 = pts.iter().map(|p| cyc.delta_g(*p, m)).sum();
            let torque: DVec3 = pts.iter().map(|p| p.cross(cyc.delta_g(*p, m))).sum();
            let scale = pts.iter().map(|p| p.length() * cyc.delta_g(*p, m).length()).sum::<f64>();
            assert!(force.length() < 1e-12 * scale, "net force {force:?}");
            assert!(torque.x.abs() < 1e-12 * scale && torque.y.abs() < 1e-12 * scale, "{torque:?}");
            let gamma_dot = lib.gamma_s * m.cos() - lib.gamma_c * m.sin();
            let damping = if q.is_finite() { i_zz * chi.sqrt() / q * gamma_dot } else { 0.0 };
            assert!((torque.z - damping).abs() < 1e-10 * scale, "z torque {} vs damping {damping}", torque.z);
        }
    }
}

/// The instantaneous necks breathe with `r` and turn with the primary.
#[test]
fn the_necks_follow_the_phase() {
    use astroid_evolution::scf::orbit::{Neck, Necks};
    let necks = Necks {
        necks: [Neck { point: DVec3::X * 10.0, axis: DVec3::X }, Neck { point: -DVec3::X * 10.0, axis: -DVec3::X }],
        r_hill: 10.0,
        centre: DVec3::ZERO,
    };
    let lib = Libration::new(0.1, 0.39, 100.0);
    let c = Cycle::new(1.0, 0.1, lib);
    // At pericentre (M = 0) the Hill region is smallest.
    let peri = c.necks_at(&necks, 0.0);
    assert!((peri.r_hill - 9.0).abs() < 1e-12 && (peri.necks[0].point.length() - 9.0).abs() < 1e-12);
    let m = 1.0;
    let at = c.necks_at(&necks, m);
    let angle = at.necks[0].axis.y.atan2(at.necks[0].axis.x);
    assert!((angle - c.psi(m)).abs() < 1e-12, "axis at {angle} vs ψ = {}", c.psi(m));
    assert!(at.beyond(at.necks[0].point * 1.01) && !at.beyond(at.necks[0].point * 0.99));
}

/// §11.6: the eclipse fraction and the primary's light.
#[test]
fn the_primary_eclipses_and_lights_a_satellite() {
    let (a_s, a_p) = (0.003, 0.37);
    assert_eq!(eclipse_visible(a_s, a_p, 1.0), 1.0);
    assert!(eclipse_visible(a_s, a_p, 0.1).abs() < 1e-12, "a total eclipse");
    // Half the Sun's disk behind the limb (to O(a_s / a_p)).
    assert!((eclipse_visible(a_s, a_p, a_p) - 0.5).abs() < 0.01);
    let mut last = 0.0;
    for k in 0..=40 {
        let v = eclipse_visible(a_s, a_p, a_p - a_s + 2.0 * a_s * k as f64 / 40.0);
        assert!(v >= last - 1e-12 && (0.0..=1.0).contains(&v), "not monotone through the penumbra");
        last = v;
    }
    let light = PrimaryLight { temperature: 209.8, albedo: 0.25 };
    let (ratio, flux) = (0.3614, 590.0);
    let sigma = 5.670_374_419e-8;
    // Sun behind the satellite: the primary's lit face, full phase.
    let full = primary_irradiance(&light, ratio, flux, -DVec3::X);
    let want = ratio * ratio * (sigma * 209.8_f64.powi(4) + 2.0 / 3.0 * 0.25 * flux);
    assert!((full - want).abs() < 1e-12 * want);
    // Sun behind the primary: its night side, heat only.
    let new = primary_irradiance(&light, ratio, flux, DVec3::X);
    assert!((new - ratio * ratio * sigma * 209.8_f64.powi(4)).abs() < 1e-9 * new);
    let _ = DMat3::IDENTITY;
}
