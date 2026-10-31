//! Unit tests of `scf::runout` (mathstrict §2.8b, hypothesis A-C3).

use astroid_evolution::common::math::{DMat3, DVec3, mat_sym};
use astroid_evolution::rheo::{Material, MaterialSpec, SolverSpec};
use std::sync::Arc;

use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::forward::Loading;
use astroid_evolution::scf::params::ModeCParams;
use astroid_evolution::scf::runout::{
    RunoutField, THETA, eroded_masses, moved_source, report, route_relief,
};
use astroid_evolution::scf::surface::SurfaceIndex;
use astroid_evolution::scf::transport::{Spread, mode_c};

fn material(mu_s: f64, mu_2: f64) -> Material {
    Material::new(
        MaterialSpec {
            nu0: 0.54,
            z0: 6.0,
            d_p: 0.05,
            rho_p: 2420.0,
            mu_s,
            mu_2,
            i_0: 0.279,
            cohesion: 4.0,
            cohesion_surface: 0.5,
            cohesion_depth: 1.0,
            c_c: 1.0,
            f_h: 1.0e-4,
            c_t: 1.0,
            gamma_eps: 1.0e-9,
            p_min: 1.0e-6,
            h_s: 1.0,
            eps_overhang: 0.01,
            mu_proj_mode: "dynamic".into(),
        },
        SolverSpec {
            max_iter: 100,
            tol: 1.0e-12,
        },
    )
    .unwrap()
}

/// A sphere of radius `r` with face normals and centres.
struct Sphere {
    topo: Arc<Topology>,
    positions: Vec<DVec3>,
    index: SurfaceIndex,
}

impl Sphere {
    fn new(level: usize, r: f64) -> Self {
        let (topo, dirs) = Topology::icosphere(level);
        let positions: Vec<DVec3> = dirs.iter().map(|d| *d * r).collect();
        let index = SurfaceIndex::build(&positions, &topo.faces);
        Self {
            topo: Arc::new(topo),
            positions,
            index,
        }
    }

    /// The deposition spread of width `w` (radians) on this sphere.
    fn spread(&self, area: &[f64], w: f64) -> Spread {
        let nv = self.positions.len();
        let body = Body::new(self.topo.clone(), self.positions.clone(), 1.0, vec![], vec![0.0; nv]).unwrap();
        Spread::new(&body, area, w * body.r_mean())
    }

    fn face(&self, f: usize) -> (DVec3, DVec3) {
        let [a, b, c] = self.topo.faces[f].map(|v| self.positions[v as usize]);
        ((a + b + c) / 3.0, (b - a).cross(c - a).normalize())
    }

    /// `g_face` from a field of the face centre and normal.
    fn gravity(&self, g: impl Fn(DVec3, DVec3) -> DVec3) -> Vec<DVec3> {
        (0..self.topo.faces.len())
            .map(|f| {
                let (c, n) = self.face(f);
                g(c, n)
            })
            .collect()
    }

    fn field<'a>(&'a self, g: &'a [DVec3], load: Loading, m: &'a Material, time: f64) -> RunoutField<'a> {
        RunoutField {
            topo: &self.topo,
            index: &self.index,
            positions: &self.positions,
            g_face: g,
            hess_face: &[],
            rho_face: &[],
            grav_const: 0.0,
            load,
            material: m,
            necks: None,
            time_limit: time,
            stop: 1.0e-3,
            theta: THETA,
        }
    }

    /// Vertex whose direction is closest to `dir`.
    fn vertex(&self, dir: DVec3) -> usize {
        let dir = dir.normalize();
        let x = &self.positions;
        (0..x.len())
            .max_by(|&a, &b| x[a].normalize().dot(dir).total_cmp(&x[b].normalize().dot(dir)))
            .unwrap()
    }
}

fn end_point(sphere: &Sphere, ids: [usize; 3], w: [f64; 3]) -> DVec3 {
    (0..3).map(|k| sphere.positions[ids[k]] * w[k]).sum()
}

/// Frictionless and rotating, gravity normal to the surface: the velocity turns at the
/// rate 2Ω, so after a quarter inertial period (t = π/(4ω)) the packet has moved along a
/// quarter circle of radius v/(2ω) to the right of its initial heading.
#[test]
fn a_frictionless_packet_follows_the_inertial_circle() {
    let sphere = Sphere::new(5, 1.0e7);
    let m = material(1e-12, 2e-12);
    let g = sphere.gravity(|_, n| n * -1.0e-3);
    let omega = 1.0e-3;
    let load = Loading { omega, tide: 0.0 };
    let time = std::f64::consts::PI / (4.0 * omega);
    let field = sphere.field(&g, load, &m, time);
    let pole = sphere.vertex(DVec3::Z);
    let v0 = 1.0;
    let arrival = field.launch(pole, DVec3::new(v0, 0.0, 0.0));
    assert!(arrival.timed_out);
    assert_eq!(arrival.hops, 0);
    let start = sphere.positions[pole];
    let end = DVec3::new(arrival.path[11][0], arrival.path[11][1], arrival.path[11][2]);
    let shift = end - start;
    // Quarter circle of radius r = v/(2ω) = 500 m: Δx = r, Δy = −r.
    let r = v0 / (2.0 * omega);
    assert!((shift.x - r).abs() < 0.02 * r, "{shift:?}");
    assert!((shift.y + r).abs() < 0.02 * r, "{shift:?}");
    let arc = std::f64::consts::FRAC_PI_2 * r;
    assert!((arrival.distance - arc).abs() < 0.02 * arc, "{}", arrival.distance);
}

/// No rotation, a level surface and a constant friction coefficient: a packet launched at
/// v₀ stops after v₀²/(2μg).
#[test]
fn a_coulomb_packet_stops_at_the_analytic_distance() {
    let sphere = Sphere::new(5, 1.0e7);
    let mu = 0.5;
    let m = material(mu, mu * (1.0 + 1e-12));
    let g0 = 1.0e-3;
    let g = sphere.gravity(|_, n| n * -g0);
    let field = sphere.field(&g, Loading { omega: 0.0, tide: 0.0 }, &m, 1.0e9);
    let pole = sphere.vertex(DVec3::Z);
    let v0 = 1.0;
    let arrival = field.launch(pole, DVec3::new(v0, 0.0, 0.0));
    assert!(!arrival.timed_out && !arrival.escaped);
    let want = v0 * v0 / (2.0 * mu * g0);
    assert!((arrival.distance - want).abs() < 0.01 * want, "{} vs {want}", arrival.distance);
}

/// Ryugu at 3.3 h: self-gravity of 30 m³/s² at 450 m and ω = 5.29e-4 s⁻¹. At ±40° the
/// effective slope (tan ≈ 0.84) exceeds μ_s = tan 35°, so packets released there run
/// toward the equator, mirror-symmetrically, and halving the substep barely moves them.
#[test]
fn packets_from_mid_latitudes_migrate_toward_the_equator() {
    let (gm, r, omega) = (30.0, 450.0, 5.29e-4);
    let sphere = Sphere::new(5, r);
    let m = material(0.70, 0.96);
    let load = Loading { omega, tide: 0.0 };
    let g = sphere.gravity(|c, _| {
        -c.normalize() * (gm / c.length_squared()) + DVec3::new(c.x, c.y, 0.0) * (omega * omega)
    });
    let lat = |p: DVec3| (p.z / p.length()).asin().to_degrees();
    let north = sphere.vertex(DVec3::new(40f64.to_radians().cos(), 0.0, 40f64.to_radians().sin()));
    let south = sphere.vertex(DVec3::new(40f64.to_radians().cos(), 0.0, -(40f64.to_radians().sin())));
    let mut ends = Vec::new();
    for theta in [THETA, THETA / 2.0] {
        let mut field = sphere.field(&g, load, &m, 1.0e6);
        field.theta = theta;
        let arrivals = field.trace(&[north, south]);
        for a in &arrivals {
            assert!(!a.escaped && !a.timed_out, "{a:?}");
            assert!(a.dlat < -10f64.to_radians(), "dlat {}", a.dlat.to_degrees());
            assert!(a.distance > 50.0);
        }
        let (n_end, s_end) = (
            end_point(&sphere, arrivals[0].ids, arrivals[0].weights),
            end_point(&sphere, arrivals[1].ids, arrivals[1].weights),
        );
        assert!(lat(n_end).abs() < 20.0 && lat(s_end).abs() < 20.0);
        assert!((lat(n_end) + lat(s_end)).abs() < 3.0, "{} {}", lat(n_end), lat(s_end));
        ends.push(n_end);
    }
    // PO-16: halving θ_r moves the arrival by less than a few node spacings.
    let spacing = r * (4.0 * std::f64::consts::PI / sphere.positions.len() as f64).sqrt();
    assert!((ends[0] - ends[1]).length() < 3.0 * spacing, "{:?}", ends);
}

/// Above the critical spin the equator pushes the packet off; it flies away, counts as
/// escaped and its mass returns to the source.
#[test]
fn an_unbound_packet_escapes_and_stays_at_its_source() {
    let (gm, r): (f64, f64) = (30.0, 450.0);
    let omega = 1.3 * (gm / (r * r * r)).sqrt();
    let sphere = Sphere::new(4, r);
    let m = material(0.70, 0.96);
    let g = sphere.gravity(|c, _| {
        -c.normalize() * (gm / c.length_squared()) + DVec3::new(c.x, c.y, 0.0) * (omega * omega)
    });
    let field = sphere.field(&g, Loading { omega, tide: 0.0 }, &m, 1.0e8);
    let equator = sphere.vertex(DVec3::X);
    let a = &field.trace(&[equator])[0];
    assert!(a.escaped && a.hops >= 1, "{a:?}");
    assert_eq!(a.distance, 0.0);
    let back = end_point(&sphere, a.ids, a.weights);
    assert!((back - sphere.positions[equator]).length() < 1e-6 * r);
}

/// Gravity and potential Hessian `GM (3 x xᵀ − r² I) / r⁵` of a point mass at the origin.
fn point_mass(gm: f64, x: DVec3) -> (DVec3, DMat3) {
    let r = x.length();
    let xx = DMat3::from_cols(x * x.x, x * x.y, x * x.z);
    (x * (-gm / r.powi(3)), (xx * 3.0 - DMat3::IDENTITY * (r * r)) * (gm / r.powi(5)))
}

/// A far flight takes its self-gravity from the exterior field (§2.8b). Off a
/// non-rotating sphere a packet ejected straight up at `v² = 1.2 GM/R` rises to
/// `R / (1 − v² R / 2GM) − R` and falls back onto its source; ejected at 45° it reaches the
/// apoapsis of its Kepler ellipse, both to 2 % at `θ_r = 0.2` (the velocity-Verlet flight is
/// second order). The prefetch rows serve each flight in a few batched calls. Without the
/// exterior field the face expansion, extrapolated a radius up, turns outward and the
/// packet is lost.
#[test]
fn a_far_flight_follows_the_kepler_orbit_in_few_exterior_calls() {
    let (gm, r) = (100.0, 1000.0);
    let sphere = Sphere::new(5, r);
    let m = material(0.70, 0.96);
    let g = sphere.gravity(|c, _| point_mass(gm, c).0);
    let hess: Vec<_> = (0..sphere.topo.faces.len()).map(|f| mat_sym(point_mass(gm, sphere.face(f).0).1)).collect();
    let mut field = sphere.field(&g, Loading { omega: 0.0, tide: 0.0 }, &m, 1.0e7);
    field.hess_face = &hess;
    let pole = sphere.vertex(DVec3::Z);
    let up = sphere.positions[pole].normalize();
    let v2 = 1.2 * gm / r;

    let (calls, points) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
    let exterior = |xs: &[DVec3]| {
        calls.set(calls.get() + 1);
        points.set(points.get() + xs.len());
        assert!(xs.iter().all(|x| x.length() > r), "an exterior request inside the body");
        Ok(xs.iter().map(|x| point_mass(gm, *x)).collect())
    };
    let fly = |v0: DVec3| {
        (calls.set(0), points.set(0));
        let a = field.launch_with(pole, v0, Some(&exterior)).unwrap();
        (a, calls.get(), points.get())
    };

    // Straight up: energy alone fixes the height, and the packet comes down where it left.
    let (a, n_calls, n_points) = fly(up * v2.sqrt());
    let climb = r / (1.0 - v2 * r / (2.0 * gm)) - r;
    assert!(!a.escaped && !a.timed_out && a.hops == 1, "{a:?}");
    assert!((a.max_hop - climb).abs() < 0.02 * climb, "height {:.1} m, energy {climb:.1} m", a.max_hop);
    let back = end_point(&sphere, a.ids, a.weights);
    assert!((back - sphere.positions[pole]).length() < 0.02 * r, "landed {:.1} m away", (back - sphere.positions[pole]).length());
    assert!((1..=4).contains(&n_calls) && n_points <= 200, "{n_calls} calls, {n_points} points");

    // At 45°: the apoapsis of the ellipse with energy ε = v²/2 − GM/R and angular momentum
    // h = R v sin 45°.
    let tilt = (up + DVec3::X).normalize();
    let (a, n_calls, n_points) = fly(tilt * v2.sqrt());
    let eps = 0.5 * v2 - gm / r;
    let h = r * v2.sqrt() * std::f64::consts::FRAC_1_SQRT_2;
    let e = (1.0 + 2.0 * eps * h * h / (gm * gm)).sqrt();
    let apo = -gm / (2.0 * eps) * (1.0 + e) - r;
    assert!(!a.escaped && a.hops >= 1, "{a:?}");
    assert!((a.max_hop - apo).abs() < 0.02 * apo, "apoapsis height {:.1} m, Kepler {apo:.1} m", a.max_hop);
    assert!((1..=4).contains(&n_calls) && n_points <= 200, "{n_calls} calls, {n_points} points");

    let lost = field.launch(pole, up * v2.sqrt());
    assert!(lost.escaped || (lost.max_hop - climb).abs() > 0.3 * climb, "the face expansion alone matched: {lost:?}");

    // Past the escape speed the flight leaves and the packet stays at its source.
    let (gone, _, _) = fly(up * (2.2 * gm / r).sqrt());
    assert!(gone.escaped && gone.distance == 0.0, "{gone:?}");
}

/// The necks of A-O (§10.4.6) on a sphere in the Hill field `W = U + ½ n²(3x² − z²)` with
/// `r_H = 2.5 R`. A packet ejected toward the primary below the critical Jacobi energy
/// `−W(L₁)` cannot leave (Proposition 10.4.6.3): it falls back and keeps its mass. Above
/// it the packet transits the neck, is lost, and its mass is laid down nowhere.
#[test]
fn a_packet_leaves_through_a_neck_only_above_the_critical_energy() {
    use astroid_evolution::scf::orbit::{Neck, Necks};
    let (gm, r): (f64, f64) = (100.0, 1000.0);
    let r_hill = 2.5 * r;
    let n2 = gm / (3.0 * r_hill.powi(3));
    let frame = |x: DVec3| DVec3::new(3.0 * x.x, 0.0, -x.z) * n2;
    let sphere = Sphere::new(5, r);
    let m = material(0.70, 0.96);
    let g = sphere.gravity(|c, _| point_mass(gm, c).0 + frame(c));
    let hess: Vec<_> = (0..sphere.topo.faces.len()).map(|f| mat_sym(point_mass(gm, sphere.face(f).0).1)).collect();
    let load = Loading { omega: n2.sqrt(), tide: n2 };
    let necks = Necks {
        necks: [
            Neck { point: DVec3::X * r_hill, axis: DVec3::X },
            Neck { point: DVec3::X * -r_hill, axis: -DVec3::X },
        ],
        r_hill,
    };
    let mut field = sphere.field(&g, load, &m, 1.0e7);
    field.hess_face = &hess;
    field.necks = Some(&necks);
    let exterior = |xs: &[DVec3]| Ok(xs.iter().map(|x| point_mass(gm, *x)).collect());
    let source = sphere.vertex(DVec3::X);
    let x0 = sphere.positions[source];
    let w = |x: DVec3| gm / x.length() + 0.5 * n2 * (3.0 * x.x * x.x - x.z * x.z);
    let v_crit = (2.0 * (w(x0) - w(DVec3::X * r_hill))).sqrt();
    let launch = |f: f64| field.launch_with(source, x0.normalize() * (f * v_crit), Some(&exterior)).unwrap();

    let bound = launch(0.9);
    assert!(!bound.escaped && !bound.lost && bound.hops >= 1, "{bound:?}");
    assert!(bound.max_hop > 0.2 * r, "the packet barely rose: {}", bound.max_hop);
    let gone = launch(1.3);
    assert!(gone.escaped && gone.lost, "{gone:?}");

    // A lost packet is laid down nowhere; one that stays keeps all its mass.
    let area = vec![1.0; sphere.positions.len()];
    let mut j = vec![0.0; area.len()];
    j[source] = 2.0;
    let laid = moved_source(&area, &j, std::slice::from_ref(&gone));
    assert!(laid.iter().all(|v| *v == 0.0));
    let kept: f64 = moved_source(&area, &j, std::slice::from_ref(&bound)).iter().sum();
    assert!((kept - 2.0).abs() < 1e-12);
}

/// Moving the sources and spreading them conserves mass exactly; arrivals at their own
/// sources reproduce the isotropic kernel of §2.8.
#[test]
fn the_runout_kernel_conserves_mass_and_reduces_to_the_isotropic_kernel() {
    let (gm, r, omega) = (30.0, 450.0, 5.29e-4);
    let sphere = Sphere::new(4, r);
    let nv = sphere.positions.len();
    let m = material(0.70, 0.96);
    let load = Loading { omega, tide: 0.0 };
    let g = sphere.gravity(|c, _| {
        -c.normalize() * (gm / c.length_squared()) + DVec3::new(c.x, c.y, 0.0) * (omega * omega)
    });
    let area = vec![4.0 * std::f64::consts::PI * r * r / nv as f64; nv];
    let rho_s = vec![1307.0; nv];
    let movable = vec![true; nv];
    // Yield excess at mid-latitudes only.
    let f_delta: Vec<f64> = sphere
        .positions
        .iter()
        .map(|p| p.normalize())
        .map(|d| if d.z.abs() > 0.5 && d.z.abs() < 0.8 { 0.3 } else { 0.0 })
        .collect();
    let prm = ModeCParams {
        k_e: 1e-3,
        sigma_0: 1.0,
        kernel_width: 0.2,
        runout: true,
        runout_time: 1.0e6,
        runout_stop: 1.0e-3,
    };
    let sources: Vec<usize> = (0..nv).filter(|&i| f_delta[i] > 0.0).collect();
    let arrivals = sphere.field(&g, load, &m, 1.0e6).trace(&sources);
    let spread = sphere.spread(&area, prm.kernel_width);
    let moved = mode_c(&spread, &area, &rho_s, &movable, &f_delta, &prm, Some(&arrivals));
    let ero: f64 = (0..nv).map(|i| moved.j_ero[i] * area[i]).sum();
    let dep: f64 = (0..nv).map(|i| moved.j_dep[i] * area[i]).sum();
    assert!((ero - dep).abs() <= 1e-12 * ero, "{ero} {dep}");
    let source = moved_source(&area, &moved.j_ero, &arrivals);
    let src: f64 = (0..nv).map(|i| source[i] * area[i]).sum();
    assert!((src - ero).abs() <= 1e-12 * ero);
    let rep = report(&eroded_masses(&area, &moved.j_ero, &arrivals, 1.0), &arrivals);
    assert_eq!(rep.traced, sources.len());
    assert!(rep.mean_dlat < 0.0, "mass-weighted Δ|lat| {}", rep.mean_dlat);

    // A packet that cannot move (no rotation, radial gravity) stays at its source.
    let still_g = sphere.gravity(|c, _| -c.normalize() * (gm / c.length_squared()));
    let still = sphere
        .field(&still_g, Loading { omega: 0.0, tide: 0.0 }, &m, 1.0e6)
        .trace(&sources);
    let a = mode_c(&spread, &area, &rho_s, &movable, &f_delta, &prm, Some(&still));
    let b = mode_c(&spread, &area, &rho_s, &movable, &f_delta, &prm, None);
    for i in 0..nv {
        assert!((a.j_dep[i] - b.j_dep[i]).abs() <= 1e-9 * b.j_dep[i].abs().max(1e-30), "{i}");
    }
}

/// The relief removed on mid-latitude slopes of a fast spinner comes back, as a radial
/// deposit, with exactly its mass in the linearized mass row, and closer to the equator.
#[test]
fn routed_relief_returns_its_mass_toward_the_equator() {
    let (gm, r, omega) = (30.0, 450.0, 5.29e-4);
    let sphere = Sphere::new(4, r);
    let nv = sphere.positions.len();
    let m = material(0.70, 0.96);
    let load = Loading { omega, tide: 0.0 };
    let g = sphere.gravity(|c, _| {
        -c.normalize() * (gm / c.length_squared()) + DVec3::new(c.x, c.y, 0.0) * (omega * omega)
    });
    let area = vec![4.0 * std::f64::consts::PI * r * r / nv as f64; nv];
    // (DM)_i = ρ A_i on a sphere; 0.5 m of relief on the 35°–55° bands.
    let jac: Vec<f64> = area.iter().map(|a| 1193.0 * a).collect();
    let removed: Vec<f64> = sphere
        .positions
        .iter()
        .map(|p| p.normalize())
        .map(|d| if d.z.abs() > 0.57 && d.z.abs() < 0.82 { 0.5 } else { 0.0 })
        .collect();
    let prm = ModeCParams {
        k_e: 1e-3,
        sigma_0: 1.0,
        kernel_width: 0.2,
        runout: true,
        runout_time: 1.0e6,
        runout_stop: 1.0e-3,
    };
    let field = sphere.field(&g, load, &m, 1.0e6);
    let routing = route_relief(&field, &area, &jac, &removed, &sphere.spread(&area, prm.kernel_width), None).unwrap();
    let out: f64 = (0..nv).map(|i| jac[i] * removed[i]).sum();
    let back: f64 = (0..nv).map(|i| jac[i] * routing.delta[i]).sum();
    assert!((back - out).abs() <= 1e-12 * out, "{back} vs {out}");
    assert_eq!(routing.masses.len(), routing.arrivals.len());
    // Mean |sin latitude| of the removed and of the returned mass.
    let mean_z = |w: &dyn Fn(usize) -> f64| {
        let (num, den) = (0..nv).fold((0.0, 0.0), |(n, d), i| {
            (n + w(i) * sphere.positions[i].normalize().z.abs(), d + w(i))
        });
        num / den
    };
    let from = mean_z(&|i| jac[i] * removed[i]);
    let to = mean_z(&|i| jac[i] * routing.delta[i]);
    assert!(to < from - 0.2, "removed at |z| {from:.3}, returned at {to:.3}");
}
