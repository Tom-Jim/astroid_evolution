//! Unit tests of `scf::stokes`, moved out of `src/scf/stokes.rs`.

mod ctx {
    pub use astroid_evolution::common::density::CauchyDensity;
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rheo::Material;
    pub use astroid_evolution::rheo::basis::sh_all;
    pub use astroid_evolution::rheo::closure::dynamic_friction;
    pub use astroid_evolution::scf::basis::{
        PresLayout, PresRows, SurfaceStrain, VelLayout, VelRows, Want, eval_pressure,
        eval_velocity_rays, surface_strain,
    };
    pub use astroid_evolution::scf::harmonics::SurfaceFit;
    pub use astroid_evolution::scf::params::StokesParams;
    pub use astroid_evolution::scf::quad::{gauss_jacobi_zeta2, sphere_grid};
    pub use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
    pub use std::f64::consts::PI;
}
use ctx::*;

use astroid_evolution::rheo::{MaterialSpec, SolverSpec};
use astroid_evolution::scf::stokes::*;

pub fn material() -> Material {
    Material::new(
        MaterialSpec {
            nu0: 0.6,
            z0: 6.0,
            d_p: 0.05,
            rho_p: 2500.0,
            mu_s: 0.38,
            mu_2: 0.64,
            i_0: 0.279,
            cohesion: 25.0,
            cohesion_surface: 25.0,
            cohesion_depth: 1.0,
            c_c: 1.0,
            f_h: 1.0e-4,
            c_t: 1.0,
            gamma_eps: 1.0e-9,
            p_min: 1.0e-6,
            h_s: 5.0,
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

pub fn params() -> StokesParams {
    StokesParams {
        l_surface: 4,
        l_velocity: 3,
        l_pressure: 4,
        k_pressure: 4,
        n_theta: 10,
        n_phi: 20,
        n_radial: 5,
        p_ext: 0.0,
        picard_max: 60,
        tol_picard: 1e-8,
        tol_momentum: 1e-6,
        theta_eta: 1.0,
        continuation: 2,
        rigid_sigma_min: 1e-6,
    }
}

#[test]
fn uniform_sphere_is_hydrostatic_and_at_rest() {
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let surf = SurfaceFit {
        l_max: 0,
        coeff: vec![r0 * (4.0 * PI).sqrt()],
    };
    let density = CauchyDensity::uniform(rho).unwrap();
    let prm = params();
    // A closed set of surface nodes: the product grid.
    let (dirs, solid) = sphere_grid(8, 16);
    let surface = SurfaceSet::build(&surf, dirs, solid);
    let ops = Operators::build(&surf, &density, surface, &prm).unwrap();
    let g_eff: Vec<DVec3> = ops.vol.pts.iter().map(|x| *x * -k).collect();
    let tide = vec![[0.0; 6]; ops.vol.pts.len()];
    let tide_nn = vec![0.0; ops.surface.pts.len()];
    let ld = Loads {
        g_eff: &g_eff,
        tide: &tide,
        tide_nn: &tide_nn,
        g_const: g,
    };
    let sol = solve(&ops, &ld, &prm, &material(), None).unwrap();
    let umax = sol.u.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(umax < 1e-12, "velocity {umax}");
    // p(r) = rho k (R^2 - r^2) / 2
    for (p, x) in ops.vol.pts.iter().enumerate().step_by(37) {
        let exact = 0.5 * rho * k * (r0 * r0 - x.length_squared());
        let got = sol.p_tilde[p];
        assert!(
            (got - exact).abs() < 2e-3 * exact.abs().max(1.0) + 1e-3,
            "p at r={}: {got} vs {exact}",
            x.length()
        );
    }
    assert!(sol.eps_rot < 1e-2, "rot {}", sol.eps_rot);
}

#[test]
fn strain_gram_matches_the_scaled_product() {
    let nu = 7;
    let np = 5;
    let mut d = DMatrix::zeros(6 * np, nu);
    for row in 0..d.nrows() {
        for col in 0..nu {
            d[(row, col)] = ((row * 3 + col * 5) % 17) as f64 * 0.01 - 0.05;
        }
    }
    let eta = [1.0, 2.5, 0.0, 4.0, 0.3];
    let weight = [0.2, 0.1, 0.4, 0.05, 0.8];
    let grams = strain_grams(&d);
    let got = stiffness_from_grams(&grams, &eta, &weight, nu);
    let mut scaled = d.clone();
    for p in 0..np {
        let s = (2.0 * eta[p] * weight[p]).sqrt();
        for c in 0..6 {
            scaled.row_mut(6 * p + c).scale_mut(s);
        }
    }
    let want = scaled.tr_mul(&scaled);
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = got
        .iter()
        .zip(want.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff <= 1e-12 * scale.max(1.0),
        "gram stiffness differs by {diff} on a matrix of scale {scale}"
    );
}

/// §4.3a: the profile equals `c_s` on the surface, tends to `c_b` at depth, and its
/// slope is the derivative of the same function.
#[test]
fn two_layer_cohesion_profile() {
    let mut m = material();
    m.spec.cohesion = 6.0;
    m.spec.cohesion_surface = 0.5;
    m.spec.cohesion_depth = 3.0;
    assert!((m.cohesion_at(0.0) - 0.5).abs() < 1e-15);
    assert!((m.cohesion_at(1e3) - 6.0).abs() < 1e-12);
    for d in [0.1, 1.0, 3.0, 10.0] {
        let h = 1e-6;
        let fd = (m.cohesion_at(d + h) - m.cohesion_at(d - h)) / (2.0 * h);
        assert!((fd - m.cohesion_slope(d)).abs() < 1e-7, "slope at {d}");
        assert!(m.cohesion_at(d) > m.cohesion_at(d * 0.5));
    }
    // A homogeneous profile gives the single-layer viscosity of §4.3.
    let one = material();
    let a = eta_eval(&one, one.cohesion_at(4.0), 1e-6, 30.0, 1e-9).eta;
    let b = eta_eval(&one, one.spec.cohesion, 1e-6, 30.0, 1e-9).eta;
    assert_eq!(a, b);
}

/// The depth gradient `∇d = (ω̂ − n/(n·ω̂))/ζ − ω̂` that carries `∇c` into `∇η`
/// matches central differences of `d(x) = R(x/|x|) − |x|` on a bumpy surface.
#[test]
fn depth_gradient_matches_finite_differences() {
    let (dirs, w) = sphere_grid(16, 32);
    let radii: Vec<f64> = dirs
        .iter()
        .map(|d| 100.0 * (1.0 + 0.08 * (3.0 * d.z * d.z - 1.0) + 0.05 * d.x * d.y))
        .collect();
    let surf = SurfaceFit::fit(4, &dirs, &radii, &w).unwrap();
    let density = CauchyDensity::try_new(1000.0, vec![]).unwrap();
    let vol = VolumeSet::build(&surf, &density, 4, 8, 3);
    let depth_at = |x: DVec3| surf.radius(x.normalize()) - x.length();
    let h = 1e-4;
    for p in (0..vol.pts.len()).step_by(7) {
        let x = vol.pts[p];
        assert!((depth_at(x) - vol.depth[p]).abs() < 1e-9 * 100.0);
        let fd = DVec3::new(
            depth_at(x + DVec3::X * h) - depth_at(x - DVec3::X * h),
            depth_at(x + DVec3::Y * h) - depth_at(x - DVec3::Y * h),
            depth_at(x + DVec3::Z * h) - depth_at(x - DVec3::Z * h),
        ) / (2.0 * h);
        assert!(
            (fd - vol.gdepth[p]).length() < 1e-4,
            "point {p}: {fd} vs {}",
            vol.gdepth[p]
        );
    }
}

#[test]
fn chart_strain_rebuilds_the_cartesian_tensor() {
    use astroid_evolution::common::math::DVec3;
    use astroid_evolution::scf::mechanics::cartesian_strain;
    // A symmetric tensor and a skewed chart: e1, e2 neither unit nor orthogonal.
    let d = [[0.3, -0.2, 0.5], [-0.2, -0.7, 0.1], [0.5, 0.1, 0.4]];
    let apply = |v: DVec3| {
        let v = v.to_array();
        DVec3::new(
            d[0][0] * v[0] + d[0][1] * v[1] + d[0][2] * v[2],
            d[1][0] * v[0] + d[1][1] * v[1] + d[1][2] * v[2],
            d[2][0] * v[0] + d[2][1] * v[1] + d[2][2] * v[2],
        )
    };
    let e1 = DVec3::new(1.3, 0.2, -0.1);
    let e2 = DVec3::new(0.4, 0.9, 0.3);
    let n = e1.cross(e2).normalize();
    let rebuilt = cartesian_strain(
        [e1, e2],
        n,
        [e1.dot(e1), e1.dot(e2), e2.dot(e2)],
        [e1.dot(apply(e1)), e1.dot(apply(e2)), e2.dot(apply(e2))],
        [n.dot(apply(e1)), n.dot(apply(e2))],
        n.dot(apply(n)),
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!((rebuilt[i][j] - d[i][j]).abs() < 1e-12, "{i}{j}: {}", rebuilt[i][j]);
        }
    }
}
