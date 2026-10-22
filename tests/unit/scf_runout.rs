//! Unit tests of `scf::runout` (mathstrict §2.8b, hypothesis A-C3).

use astroid_evolution::common::math::DVec3;
use astroid_evolution::rheo::{Material, MaterialSpec, SolverSpec};
use astroid_evolution::scf::body::Topology;
use astroid_evolution::scf::forward::Loading;
use astroid_evolution::scf::params::ModeCParams;
use astroid_evolution::scf::runout::{
    RunoutField, THETA, eroded_masses, moved_source, report, route_relief,
};
use astroid_evolution::scf::transport::{DirIndex, mode_c};

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
    topo: Topology,
    positions: Vec<DVec3>,
    index: DirIndex,
}

impl Sphere {
    fn new(level: usize, r: f64) -> Self {
        let topo = Topology::icosphere(level);
        let positions = topo.dirs.iter().map(|d| *d * r).collect();
        let index = DirIndex::build(&topo.dirs);
        Self {
            topo,
            positions,
            index,
        }
    }

    fn face(&self, f: usize) -> (DVec3, DVec3) {
        let [a, b, c] = self.topo.faces[f].map(|v| self.positions[v as usize]);
        let centre = (a + b + c) / 3.0;
        let mut n = (b - a).cross(c - a).normalize();
        if n.dot(centre) < 0.0 {
            n = -n;
        }
        (centre, n)
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
            load,
            material: m,
            time_limit: time,
            stop: 1.0e-3,
            theta: THETA,
        }
    }

    /// Vertex whose direction is closest to `dir`.
    fn vertex(&self, dir: DVec3) -> usize {
        let dir = dir.normalize();
        (0..self.topo.dirs.len())
            .max_by(|&a, &b| self.topo.dirs[a].dot(dir).total_cmp(&self.topo.dirs[b].dot(dir)))
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
    let spacing = r * (4.0 * std::f64::consts::PI / sphere.topo.dirs.len() as f64).sqrt();
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

/// Moving the sources and spreading them conserves mass exactly; arrivals at their own
/// sources reproduce the isotropic kernel of §2.8.
#[test]
fn the_runout_kernel_conserves_mass_and_reduces_to_the_isotropic_kernel() {
    let (gm, r, omega) = (30.0, 450.0, 5.29e-4);
    let sphere = Sphere::new(4, r);
    let nv = sphere.topo.dirs.len();
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
        .topo
        .dirs
        .iter()
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
    let moved = mode_c(&sphere.topo, &area, &rho_s, &movable, &f_delta, &prm, Some(&arrivals));
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
    let a = mode_c(&sphere.topo, &area, &rho_s, &movable, &f_delta, &prm, Some(&still));
    let b = mode_c(&sphere.topo, &area, &rho_s, &movable, &f_delta, &prm, None);
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
    let nv = sphere.topo.dirs.len();
    let m = material(0.70, 0.96);
    let load = Loading { omega, tide: 0.0 };
    let g = sphere.gravity(|c, _| {
        -c.normalize() * (gm / c.length_squared()) + DVec3::new(c.x, c.y, 0.0) * (omega * omega)
    });
    let area = vec![4.0 * std::f64::consts::PI * r * r / nv as f64; nv];
    // (DM)_i = ρ A_i on a sphere; 0.5 m of relief on the 35°–55° bands.
    let jac: Vec<f64> = area.iter().map(|a| 1193.0 * a).collect();
    let removed: Vec<f64> = sphere
        .topo
        .dirs
        .iter()
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
    let routing = route_relief(&field, &area, &jac, &removed, &prm);
    let out: f64 = (0..nv).map(|i| jac[i] * removed[i]).sum();
    let back: f64 = (0..nv).map(|i| jac[i] * routing.delta[i]).sum();
    assert!((back - out).abs() <= 1e-12 * out, "{back} vs {out}");
    assert_eq!(routing.masses.len(), routing.arrivals.len());
    // Mean |sin latitude| of the removed and of the returned mass.
    let mean_z = |w: &dyn Fn(usize) -> f64| {
        let (num, den) = (0..nv).fold((0.0, 0.0), |(n, d), i| {
            (n + w(i) * sphere.topo.dirs[i].z.abs(), d + w(i))
        });
        num / den
    };
    let from = mean_z(&|i| jac[i] * removed[i]);
    let to = mean_z(&|i| jac[i] * routing.delta[i]);
    assert!(to < from - 0.2, "removed at |z| {from:.3}, returned at {to:.3}");
}
