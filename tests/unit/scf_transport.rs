//! Unit tests of `scf::transport` (§2.8, §8.0 G5): erosion, the mesh heat-kernel spread,
//! geodesic foot points and conservative transport of `ρ_s`.

use std::sync::Arc;

use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::params::ModeCParams;
use astroid_evolution::scf::transport::*;

fn sphere(level: usize, r: f64) -> Body {
    let (topo, dirs) = Topology::icosphere(level);
    let nv = dirs.len();
    Body::new(Arc::new(topo), dirs.iter().map(|d| *d * r).collect(), 1.0, vec![], vec![1.0; nv])
        .unwrap()
}

fn params(width: f64) -> ModeCParams {
    ModeCParams {
        k_e: 0.1,
        sigma_0: 10.0,
        kernel_width: width,
        runout: false,
        runout_time: 1.0,
        runout_stop: 0.5,
    }
}

/// The step-averaged erosion integrates `dρ_s/dt = −λ ρ_s` exactly, so the layer
/// left after any step is `ρ_s e^{−λ dt} ≥ 0`, and short steps recover the rate.
#[test]
fn step_averaged_erosion_is_exact_and_positive() {
    let prm = ModeCParams {
        k_e: 1e-3,
        sigma_0: 1.0,
        ..params(0.2)
    };
    let (area, rho_s, movable, f) = (vec![1.0; 3], vec![9.0, 0.5, 0.0], vec![true; 3], vec![0.4; 3]);
    let lambda = prm.k_e * 0.4 / prm.sigma_0;
    for dt in [1e-3, 1e2, 1e4, 1e8] {
        let (j, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, dt);
        for i in 0..3 {
            let left = rho_s[i] - dt * j[i];
            assert!(
                left >= 0.0
                    && (left - rho_s[i] * (-lambda * dt).exp()).abs() <= 1e-12 * rho_s[i].max(1.0),
                "{dt} {i}"
            );
        }
    }
    let (j0, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, 0.0);
    let (j1, _, _) = erosion_over(&area, &rho_s, &movable, &f, &prm, 1e-6);
    assert!((j0[0] - j1[0]).abs() < 1e-9 * j0[0]);
}

/// `∫ J_dep = ∫ J_ero` to rounding, and the deposit is nonnegative (§2.8).
#[test]
fn mode_c_conserves_mass_exactly() {
    let body = sphere(3, 100.0);
    let area = body.vertex_areas();
    let x = &body.positions;
    let nv = x.len();
    let movable: Vec<bool> = (0..nv).map(|i| x[i].z > 30.0).collect();
    let f: Vec<f64> = (0..nv).map(|i| 0.03 * x[i].z).collect();
    let prm = params(0.3);
    let spread = Spread::new(&body, &area, prm.kernel_width * body.r_mean());
    let mc = mode_c(&spread, &area, &vec![5.0; nv], &movable, &f, &prm, None);
    let ero: f64 = (0..nv).map(|i| mc.j_ero[i] * area[i]).sum();
    let dep: f64 = (0..nv).map(|i| mc.j_dep[i] * area[i]).sum();
    assert!(ero > 0.0);
    assert!((ero - dep).abs() < 1e-12 * ero, "{ero} vs {dep}");
    assert!(mc.j_dep.iter().all(|j| *j >= 0.0));
}

/// A point source spread by `t = w²/2` has, on a sphere much larger than `w`, the
/// second moment `2w²` of a planar Gaussian of width `w`.
#[test]
fn heat_kernel_spread_has_the_gaussian_width() {
    let body = sphere(5, 1.0);
    let area = body.vertex_areas();
    let w = 0.1;
    let pole = (0..body.positions.len()).max_by(|a, b| {
        body.positions[*a].z.total_cmp(&body.positions[*b].z)
    });
    let mut j = vec![0.0; area.len()];
    let p = pole.unwrap();
    j[p] = 1.0 / area[p];
    let out = Spread::new(&body, &area, w).spread(&j);
    let mass: f64 = out.iter().zip(&area).map(|(u, a)| u * a).sum();
    assert!((mass - 1.0).abs() < 1e-12);
    let second: f64 = (0..out.len())
        .map(|i| out[i] * area[i] * (body.positions[i] - body.positions[p]).length_squared())
        .sum();
    assert!((second / (2.0 * w * w) - 1.0).abs() < 0.15, "{}", second / (2.0 * w * w));
}

/// On a sphere the straightest geodesic of length `L` from a vertex ends near the
/// great-circle point at arc `L`.
#[test]
fn geodesic_follows_the_great_circle() {
    let body = sphere(4, 1.0);
    let (x, topo) = (&body.positions, &body.topo);
    let a = 0;
    let t = x[a].cross(DVec3::Z).normalize_or(DVec3::X);
    let len = 0.5;
    let (f, w) = geodesic(topo, x, a, t * len);
    let end: DVec3 = (0..3).map(|k| x[topo.faces[f][k] as usize] * w[k]).sum();
    let arc = x[a].dot(end.normalize()).clamp(-1.0, 1.0).acos();
    assert!((arc - len).abs() < 0.02, "arc {arc} for {len}");
}

/// Transport by a rigid rotation keeps `Σ A ρ_s` and moves a bump along the flow.
#[test]
fn transport_conserves_and_translates_a_bump() {
    let body = sphere(4, 1.0);
    let x = &body.positions;
    let nv = x.len();
    let normals = body.vertex_normals();
    let v: Vec<DVec3> = x.iter().map(|p| DVec3::new(-p.y, p.x, 0.0) * 0.05).collect();
    let flow = surface_flow(&body, &normals, &v, &vec![true; nv]);
    let field: Vec<f64> = x
        .iter()
        .map(|d| (-(d.x - 1.0).powi(2) * 6.0 - d.y * d.y * 6.0 - d.z * d.z * 6.0).exp())
        .collect();
    let area = body.vertex_areas();
    let (out, fix) = transport(&body, &flow, &field, &area, &area, 1.0);
    let mass = |f: &[f64]| (0..nv).map(|i| f[i] * area[i]).sum::<f64>();
    assert!((mass(&out) - mass(&field)).abs() < 1e-12 * mass(&field));
    assert!(fix < 0.1);
    let cy = |f: &[f64]| (0..nv).map(|i| f[i] * x[i].y * area[i]).sum::<f64>();
    assert!(cy(&out) > cy(&field) + 1e-3, "{} -> {}", cy(&field), cy(&out));
}
