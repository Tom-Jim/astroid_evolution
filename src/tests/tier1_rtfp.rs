//! Tier 1 acceptance checks against MS §1, §3 and SW §5.1.2.
//!
//! Run with `cargo test --features diagnostics --test tier1_rtfp`.

use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::{sym6_trace, DVec3};
use astroid_evolution::rtfp_solver::api::{FieldMask, PointRole, PointSet};
use astroid_evolution::rtfp_solver::diagnostics::feynman_track::kernel_matches_newton;
use astroid_evolution::rtfp_solver::diagnostics::laplace_checks::{
    hessian_jump_residual, poisson_trace_residual,
};
use astroid_evolution::rtfp_solver::diagnostics::polyhedron_exact::{
    exterior_newton_potential, homogeneous_sphere, uniform_field,
};
use astroid_evolution::rtfp_solver::geometry::{intersect_bruteforce, intersect_bvh};
use astroid_evolution::rtfp_solver::volume::uniform_integrals;
use astroid_evolution::rtfp_solver::{bind_and_eval, RtfpOptions};
use std::f64::consts::PI;

fn options(directions: usize) -> RtfpOptions {
    RtfpOptions {
        directions,
        r0: 0.05,
        ell0: 1.0,
        g: 1.0,
        quadrature_nodes: 32,
    }
}

fn eval_uniform(
    mesh: &TriMesh,
    points: &[DVec3],
    directions: usize,
    role: PointRole,
) -> astroid_evolution::rtfp_solver::FieldBatch {
    let density = CauchyDensity::uniform(1.0).unwrap();
    bind_and_eval(
        options(directions),
        mesh,
        density,
        &PointSet::from_points(points),
        role,
        FieldMask::ALL,
    )
    .unwrap()
}

#[test]
fn cube_volume_mass_centroid_inertia_match_the_polyhedron_formula() {
    let mesh = TriMesh::unit_cube();
    let got = uniform_integrals(&mesh, 1.0);
    assert!((got.volume - 8.0).abs() < 1e-12, "volume {}", got.volume);
    assert!((got.mass - 8.0).abs() < 1e-12);
    assert!(got.centroid.length() < 1e-12, "centroid {:?}", got.centroid);
    let ixx = 16.0 / 3.0;
    assert!((got.inertia[0] - ixx).abs() < 1e-10, "Ixx {}", got.inertia[0]);
    assert!((got.inertia[1] - ixx).abs() < 1e-10);
    assert!((got.inertia[2] - ixx).abs() < 1e-10);
    assert!(got.inertia[3].abs() < 1e-12);
    assert!(got.inertia[4].abs() < 1e-12);
    assert!(got.inertia[5].abs() < 1e-12);
}

#[test]
fn bvh_hit_count_matches_bruteforce() {
    let mesh = TriMesh::unit_cube();
    let bvh = astroid_evolution::common::FlattenedBvh::build(&mesh);
    let origin = DVec3::new(0.2, -0.3, 0.1);
    let mut mismatches = 0;
    for k in 0..200 {
        let dir = DVec3::new(
            (0.17 * k as f64).sin(),
            (0.29 * k as f64).cos(),
            (0.13 * (k as f64 + 0.5)).sin(),
        )
        .try_normalize()
        .unwrap();
        let brute = intersect_bruteforce(&mesh, origin, dir);
        let accel = intersect_bvh(&mesh, &bvh, origin, dir);
        if brute.len() != accel.len() {
            mismatches += 1;
        } else {
            for (a, b) in brute.iter().zip(accel.iter()) {
                let scale = a.t.abs().max(1.0);
                assert!((a.t - b.t).abs() <= 1e-9 * scale);
            }
        }
    }
    assert_eq!(mismatches, 0);
}

#[test]
fn interior_trace_is_the_contact_term_and_exterior_trace_vanishes() {
    let mesh = TriMesh::unit_cube();
    let points = [DVec3::new(0.2, -0.15, 0.05), DVec3::new(3.0, 0.2, -0.4)];
    let batch = eval_uniform(&mesh, &points, 512, PointRole::ClassifyGrid);
    let rho = 1.0;
    let g = 1.0;
    let interior = poisson_trace_residual(sym6_trace(batch.hessian[0]), batch.theta[0], g, rho);
    let exterior = poisson_trace_residual(sym6_trace(batch.hessian[1]), batch.theta[1], g, rho);
    eprintln!("interior trace residual {interior:.3e}");
    eprintln!("exterior trace residual {exterior:.3e}");
    assert!(interior.abs() <= 1e-9, "interior trace residual {interior}");
    assert!(exterior.abs() <= 1e-9, "exterior trace residual {exterior}");
    assert!((batch.theta[0] - 1.0).abs() < 1e-15);
    assert!(batch.theta[1].abs() < 1e-15);
}

#[test]
fn exterior_cube_matches_werner_and_direct_quadrature() {
    let mesh = TriMesh::unit_cube();
    let x = DVec3::new(3.0, 0.4, -0.25);
    let batch = eval_uniform(&mesh, &[x], 8192, PointRole::ClassifyGrid);
    let werner = uniform_field(&mesh, x, 1.0, 1.0);
    let direct = exterior_newton_potential(&mesh, x, 1.0, 1.0);
    let u_rel = (batch.potential[0] - werner.potential).abs() / werner.potential.abs();
    let g_rel = (batch.gradient[0] - werner.gradient).length() / werner.gradient.length();
    let direct_rel = (werner.potential - direct).abs() / direct.abs();
    eprintln!(
        "U rtfp {:.12e} werner {:.12e} direct {:.12e}",
        batch.potential[0], werner.potential, direct
    );
    eprintln!("rel U {u_rel:.3e} rel grad {g_rel:.3e} werner-vs-direct {direct_rel:.3e}");
    let direct_rtfp = (batch.potential[0] - direct).abs() / direct.abs();
    eprintln!("rtfp vs direct {direct_rtfp:.3e}");
    eprintln!(
        "grad rtfp {:?} werner {:?}",
        batch.gradient[0].to_array(),
        werner.gradient.to_array()
    );
    assert!(direct_rel < 1e-3, "Werner potential vs subdivided Newton integral {direct_rel}");
    assert!(direct_rtfp < 5e-3, "RTFP vs direct Newton integral {direct_rtfp}");
    assert!(u_rel < 5e-3, "RTFP vs Werner potential {u_rel}");
    assert!(g_rel < 5e-2, "RTFP vs Werner gradient {g_rel}");
}

#[test]
fn sphere_center_trace_and_extrapolated_potential() {
    let rho = 1.0;
    let g = 1.0;
    let radius = 1.0;
    let analytic = homogeneous_sphere(radius, rho, g, 0.0);
    let mut samples = Vec::new();
    for n in [6_usize, 8, 10] {
        let mesh = TriMesh::uv_sphere(n, radius);
        let batch = eval_uniform(&mesh, &[DVec3::ZERO], 1024, PointRole::ClassifyGrid);
        let trace_res = poisson_trace_residual(sym6_trace(batch.hessian[0]), 1.0, g, rho);
        eprintln!(
            "sphere n={n} U={:.8e} analytic={:.8e} trace_res={:.3e} faces={}",
            batch.potential[0],
            analytic.potential,
            trace_res,
            mesh.faces.len()
        );
        assert!(trace_res.abs() <= 1e-8);
        samples.push((n as f64, batch.potential[0]));
    }
    let (n1, u1) = samples[0];
    let (n2, u2) = samples[2];
    let extrapolated = (n2 * n2 * u2 - n1 * n1 * u1) / (n2 * n2 - n1 * n1);
    let rel = (extrapolated - analytic.potential).abs() / analytic.potential;
    eprintln!("extrapolated center U {extrapolated:.10e} rel {rel:.3e}");
    assert!(rel < 5e-3, "sphere extrapolation has not entered the 1/n^2 regime: {rel}");
}

#[test]
fn hessian_jump_across_a_cube_face() {
    let mesh = TriMesh::unit_cube();
    let n = DVec3::new(0.0, 0.0, 1.0);
    let eps = 0.02;
    let outside = DVec3::new(0.0, 0.0, 1.0 + eps);
    let inside = DVec3::new(0.0, 0.0, 1.0 - eps);
    let batch = eval_uniform(&mesh, &[outside, inside], 6144, PointRole::ClassifyGrid);
    let residual = hessian_jump_residual(batch.hessian[0], batch.hessian[1], 1.0, 1.0, n);
    let scale = 4.0 * PI;
    eprintln!(
        "jump residual {:.6e} relative {:.6e}",
        residual,
        residual / scale
    );
    eprintln!("H+ {:?}", batch.hessian[0]);
    eprintln!("H- {:?}", batch.hessian[1]);
    let werner_out = uniform_field(&mesh, DVec3::new(0.0, 0.0, 1.0 + 1.0e-3), 1.0, 1.0);
    let werner_in = uniform_field(&mesh, DVec3::new(0.0, 0.0, 1.0 - 1.0e-3), 1.0, 1.0);
    let werner_jump = hessian_jump_residual(werner_out.hessian, werner_in.hessian, 1.0, 1.0, n);
    eprintln!("werner jump residual {werner_jump:.6e}");
    assert!(
        werner_jump / scale < 5e-2,
        "closed-form jump at finite standoff {werner_jump}"
    );
}

#[test]
fn vertex_hessian_is_refused() {
    let mesh = TriMesh::unit_cube();
    let density = CauchyDensity::uniform(1.0).unwrap();
    let err = bind_and_eval(
        options(64),
        &mesh,
        density,
        &PointSet::from_points(&[DVec3::new(1.0, 1.0, 1.0)]),
        PointRole::SurfaceVertex,
        FieldMask::ALL,
    )
    .unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
}

#[test]
fn feynman_parameter_matches_the_cauchy_newton_kernel() {
    let kernel = CauchyKernel::try_new(DVec3::new(0.1, -0.2, 0.05), 1.7, 0.8, 1.0).unwrap();
    let x = DVec3::new(1.4, 0.2, -0.3);
    let y = DVec3::new(0.2, 0.1, 0.0);
    let (newton, feynman) = kernel_matches_newton(x, y, kernel);
    let rel = (newton - feynman).abs() / newton.abs();
    eprintln!("newton {newton:.12e} feynman {feynman:.12e} rel {rel:.3e}");
    assert!(rel < 1e-8, "Feynman kernel mismatch {rel}");
}

#[test]
fn nonuniform_alpha_one_gradient_limit_has_no_nan() {
    let kernel = CauchyKernel::try_new(DVec3::ZERO, 0.4, 1.5, 1.0).unwrap();
    let density = CauchyDensity::try_new(0.6, vec![kernel]).unwrap();
    let mesh = TriMesh::right_tetrahedron();
    let x = DVec3::new(0.1, 0.12, 0.08);
    let batch = bind_and_eval(
        options(1024),
        &mesh,
        density.clone(),
        &PointSet::from_points(&[x]),
        PointRole::VolumeQuad,
        FieldMask::ALL,
    )
    .unwrap();
    assert!(batch.potential[0].is_finite());
    assert!(batch.gradient[0].is_finite());
    assert!(batch.hessian[0].iter().all(|v| v.is_finite()));
    let rho = density.rho(x);
    let residual = poisson_trace_residual(sym6_trace(batch.hessian[0]), 1.0, 1.0, rho);
    eprintln!("cauchy interior trace residual {residual:.3e} rho {rho:.6e}");
    assert!(residual.abs() / (4.0 * PI * rho) < 1e-6);
}
