//! Unit tests of `scf::stokes` on the general domain of hypothesis (G).

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

use std::sync::Arc;

use astroid_evolution::rheo::{MaterialSpec, SolverSpec};
use astroid_evolution::rtfp::mesh::BodyMesh;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::domain::SurfacePoints;
use astroid_evolution::scf::stokes::*;
use astroid_evolution::scf::surface::SurfaceIndex;

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
        solver: "primal-dual".into(),
        pd_tol: 1e-8,
        pd_max: 400_000,
        pd_pressure_every: 10,
        n_boundary: 1024,
        cut_depth: 3,
    }
}

/// The Stokes operators of a uniform polyhedral sphere of radius `r0` (§8.0).
pub fn sphere_ops(r0: f64, rho: f64, prm: &StokesParams) -> Operators {
    let (topo, dirs) = Topology::icosphere(4);
    let nv = dirs.len();
    let body = Body::new(Arc::new(topo), dirs.iter().map(|d| *d * r0).collect(), rho, vec![], vec![0.0; nv])
        .unwrap();
    let mesh = BodyMesh::build(&body.mesh().unwrap()).unwrap();
    let index = SurfaceIndex::build(&body.positions, &body.topo.faces);
    let faces = SurfacePoints::faces(&body.face_geometry());
    Operators::build(&SurfaceFit::sphere(body.r_star()), &body.density().unwrap(), &mesh, &index, faces, prm)
        .unwrap()
}

/// A uniform sphere in its own gravity is at rest, and the Nitsche pressure (G3) is the
/// hydrostatic `ρ k (R² − r²) / 2` up to the polyhedral approximation of the sphere.
#[test]
fn uniform_sphere_is_hydrostatic_and_at_rest() {
    let (r0, rho, g) = (1000.0, 2000.0, 6.6743e-11);
    let k = 4.0 * PI / 3.0 * g * rho;
    let prm = params();
    let ops = sphere_ops(r0, rho, &prm);
    let g_eff: Vec<DVec3> = ops.vol.pts.iter().map(|x| *x * -k).collect();
    let g_bnd: Vec<DVec3> = ops.bnd.pts.iter().map(|x| *x * -k).collect();
    let tide = vec![[0.0; 6]; ops.vol.pts.len()];
    let tide_bnd = vec![[0.0; 6]; ops.bnd.pts.len()];
    let tide_nn = vec![0.0; ops.faces.pts.len()];
    let ld = Loads {
        g_eff: &g_eff,
        tide: &tide,
        g_bnd: &g_bnd,
        tide_bnd: &tide_bnd,
        tide_nn: &tide_nn,
        g_const: g,
    };
    let sol = solve(&ops, &ld, &prm, &material(), None).unwrap();
    let umax = sol.u.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(umax < 1e-12, "velocity {umax}");
    let centre = 0.5 * rho * k * r0 * r0;
    for (p, x) in ops.vol.pts.iter().enumerate() {
        let exact = 0.5 * rho * k * (r0 * r0 - x.length_squared());
        assert!((sol.p_tilde[p] - exact).abs() < 5e-3 * centre, "p at r={}: {} vs {exact}", x.length(), sol.p_tilde[p]);
    }
    assert!(sol.eps_rot < 1e-2, "rot {}", sol.eps_rot);
}

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

/// The depth is the distance to the surface (G6), not the radial gap: the foot
/// `x − d ∇d` lies on `r = R(ω̂)`, the gap is normal there (`∇d = −n`), `|∇d| = 1`,
/// and `d` never exceeds the radial gap `R(x̂) − |x|`.
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
