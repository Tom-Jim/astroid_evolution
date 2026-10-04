//! Tier 1 acceptance. The RTFP solver runs on the hardware ray tracer only; the
//! Werner, Feynman and Carlson references are used here as error benchmarks.
//! Target: relative error of at most 1e-2 against each reference.

use std::time::Instant;

use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::{DVec3, Sym6};
use astroid_evolution::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use astroid_evolution::rtfp_solver::reference::ellipsoid::Ellipsoid;
use astroid_evolution::rtfp_solver::reference::feynman::Feynman;
use astroid_evolution::rtfp_solver::reference::werner;
use astroid_evolution::rtfp_solver::volume::uniform_integrals;
use astroid_evolution::rtfp_solver::{RtfpOptions, RtfpSolver};
use std::f64::consts::PI;

const TOLERANCE: f64 = 1.0e-2;
/// The Feynman reference itself is good to about 2e-3.
const FEYNMAN_TOLERANCE: f64 = 1.0e-2;
const G: f64 = 1.0;

fn solver(mesh: &TriMesh, density: CauchyDensity) -> RtfpSolver {
    let options = RtfpOptions { g: G, ..RtfpOptions::default() };
    RtfpSolver::bind(options, mesh.clone(), density).expect("hardware ray tracing is available")
}

fn eval(solver: &RtfpSolver, points: &[DVec3], role: PointRole) -> FieldBatch {
    solver
        .eval(&PointSet::from_points(points), role, FieldMask::ALL)
        .expect("field evaluation")
}

fn relative(value: f64, reference: f64, floor: f64) -> f64 {
    (value - reference).abs() / reference.abs().max(floor)
}

fn relative_vec(value: DVec3, reference: DVec3) -> f64 {
    // Floor: a gradient that vanishes by symmetry is judged against the field scale.
    (value - reference).length() / reference.length().max(0.1)
}

fn relative_hessian(value: Sym6, reference: Sym6) -> f64 {
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = (0..6).fold(0.0_f64, |m, i| m.max((value[i] - reference[i]).abs()));
    diff / scale.max(1e-300)
}

struct Errors {
    potential: f64,
    gradient: f64,
    hessian: f64,
}

fn check(
    label: &str,
    batch: &FieldBatch,
    i: usize,
    potential: f64,
    gradient: DVec3,
    hessian: Sym6,
) -> Errors {
    let errors = Errors {
        potential: relative(batch.potential[i], potential, 1e-300),
        gradient: relative_vec(batch.gradient[i], gradient),
        hessian: relative_hessian(batch.hessian[i], hessian),
    };
    println!(
        "{label}: U {:.2e}  grad {:.2e}  hess {:.2e}",
        errors.potential, errors.gradient, errors.hessian
    );
    errors
}

fn assert_within(label: &str, errors: &Errors) {
    assert_bound(label, errors, TOLERANCE);
}

fn assert_bound(label: &str, errors: &Errors, bound: f64) {
    assert!(errors.potential <= bound, "{label}: potential {:.3e}", errors.potential);
    assert!(errors.gradient <= bound, "{label}: gradient {:.3e}", errors.gradient);
    assert!(errors.hessian <= bound, "{label}: hessian {:.3e}", errors.hessian);
}

/// Uniform density against the exact Werner–Scheeres polyhedron.
fn against_werner(mesh: &TriMesh, interior: &[DVec3], exterior: &[DVec3], name: &str) {
    let rho = 2.5;
    let solver = solver(mesh, CauchyDensity::uniform(rho).unwrap());
    for (points, role, kind) in [(interior, PointRole::Field, "interior"), (exterior, PointRole::Field, "exterior")] {
        let batch = eval(&solver, points, role);
        for (i, x) in points.iter().enumerate() {
            let w = werner::uniform_polyhedron(mesh, *x, G, rho);
            let label = format!("{name} {kind} {i}");
            assert_within(&label, &check(&label, &batch, i, w.potential, w.gradient, w.hessian));
        }
    }
    let faces: Vec<DVec3> = (0..mesh.faces.len().min(24))
        .step_by((mesh.faces.len() / 24).max(1))
        .map(|f| {
            let [a, b, c] = mesh.triangle(f);
            (a + b + c) * (1.0 / 3.0)
        })
        .collect();
    let batch = eval(&solver, &faces, PointRole::SurfaceFace);
    for (i, x) in faces.iter().enumerate() {
        let w = werner::uniform_polyhedron(mesh, *x, G, rho);
        let label = format!("{name} face {i}");
        assert_within(&label, &check(&label, &batch, i, w.potential, w.gradient, w.hessian));
    }
}

#[test]
fn cube_matches_werner() {
    against_werner(
        &TriMesh::unit_cube(),
        &[DVec3::new(0.1, -0.2, 0.3), DVec3::new(0.9, 0.8, -0.7), DVec3::new(0.0, 0.0, 0.0)],
        &[DVec3::new(1.5, 0.2, 0.1), DVec3::new(-2.0, 1.7, 3.0), DVec3::new(1.05, 1.05, 1.05)],
        "cube",
    );
}

#[test]
fn tetrahedron_matches_werner() {
    against_werner(
        &TriMesh::right_tetrahedron(),
        &[DVec3::new(0.2, 0.2, 0.2), DVec3::new(0.05, 0.05, 0.05)],
        &[DVec3::new(1.0, 1.0, 1.0), DVec3::new(-0.3, 0.3, 0.3)],
        "tetra",
    );
}

#[test]
fn sphere_matches_werner() {
    against_werner(
        &TriMesh::uv_sphere(24, 1.0),
        &[DVec3::new(0.3, 0.1, -0.2), DVec3::new(0.0, 0.0, 0.8)],
        &[DVec3::new(1.6, 0.0, 0.0), DVec3::new(0.7, 0.7, 1.2)],
        "sphere",
    );
}

/// A bumpy sphere: non-convex and not star-shaped about most interior points.
fn bumpy() -> TriMesh {
    let base = TriMesh::uv_sphere(32, 1.0);
    let vertices = base
        .vertices
        .iter()
        .map(|v| {
            let theta = (v.z / v.length()).acos();
            let phi = v.y.atan2(v.x);
            *v * (1.0 + 0.35 * (5.0 * phi).cos() * (3.0 * theta).sin().powi(2) - 0.25 * (2.0 * theta).cos())
        })
        .collect();
    TriMesh::try_new(vertices, base.faces).expect("bumpy sphere")
}

#[test]
fn non_convex_body_matches_werner() {
    let mesh = bumpy();
    against_werner(
        &mesh,
        &[DVec3::new(0.2, 0.3, 0.1), DVec3::new(0.0, 0.0, 0.5), DVec3::new(0.5, -0.1, -0.2)],
        &[DVec3::new(2.0, 0.0, 0.3), DVec3::new(0.0, 0.0, 2.0), DVec3::new(1.1, 0.0, 0.0)],
        "bumpy",
    );
}

#[test]
fn near_surface_points_stay_accurate() {
    let mesh = TriMesh::unit_cube();
    let rho = 1.0;
    let solver = solver(&mesh, CauchyDensity::uniform(rho).unwrap());
    let points = [
        DVec3::new(0.3, 0.2, 1.0 - 1e-3),
        DVec3::new(0.3, 0.2, 1.0 + 1e-3),
        DVec3::new(-1.0 + 1e-2, 0.9, 0.9),
        DVec3::new(0.2, -1.0 - 1e-2, 0.1),
    ];
    let batch = eval(&solver, &points, PointRole::Field);
    for (i, x) in points.iter().enumerate() {
        let w = werner::uniform_polyhedron(&mesh, *x, G, rho);
        let label = format!("near surface {i}");
        assert_within(&label, &check(&label, &batch, i, w.potential, w.gradient, w.hessian));
    }
}

#[test]
fn vertex_hessian_is_refused_and_vertex_gradient_is_served() {
    let mesh = TriMesh::unit_cube();
    let solver = solver(&mesh, CauchyDensity::uniform(1.0).unwrap());
    let vertices = PointSet::from_points(&mesh.vertices);
    assert!(solver.eval(&vertices, PointRole::SurfaceVertex, FieldMask::ALL).is_err());
    let batch = solver
        .eval(&vertices, PointRole::SurfaceVertex, FieldMask::GRAD)
        .expect("vertex gradient");
    for (i, v) in mesh.vertices.iter().enumerate() {
        let w = werner::uniform_polyhedron(&mesh, *v, G, 1.0);
        assert!(relative_vec(batch.gradient[i], w.gradient) <= TOLERANCE, "vertex {i}");
    }
}

fn cauchy(centers: &[([f64; 3], f64, f64, f64)], bulk: f64) -> CauchyDensity {
    let kernels = centers
        .iter()
        .map(|&(c, weight, sigma, alpha)| CauchyKernel::try_new(DVec3::from_array(c), weight, sigma, alpha).unwrap())
        .collect();
    CauchyDensity::try_new(bulk, kernels).unwrap()
}

#[test]
#[ignore = "slow high-precision Feynman benchmark; run with --ignored"]
fn feynman_reference_reduces_to_werner_for_a_flat_kernel() {
    // sigma -> 0 makes (1 + sigma^2 r^2)^-alpha equal to 1 to 1e-8 over the body.
    let mesh = TriMesh::unit_cube();
    let density = cauchy(&[([0.2, 0.1, 0.0], 3.0, 1.0e-4, 1.0)], 0.0);
    let feynman = Feynman::new(&mesh, &density, G);
    for x in [DVec3::new(0.2, 0.1, 0.3), DVec3::new(2.0, 0.5, -1.0)] {
        let (u, g) = feynman.potential_gradient(x);
        let w = werner::uniform_polyhedron(&mesh, x, G, 3.0);
        assert!(relative(u, w.potential, 1e-300) < FEYNMAN_TOLERANCE, "potential {u} vs {}", w.potential);
        assert!(relative_vec(g, w.gradient) < FEYNMAN_TOLERANCE, "gradient {g:?} vs {:?}", w.gradient);
    }
}

fn against_feynman(mesh: &TriMesh, density: CauchyDensity, interior: &[DVec3], exterior: &[DVec3], name: &str) {
    let solver = solver(mesh, density.clone());
    let reference = Feynman::new(mesh, &density, G);
    for (points, kind) in [(interior, "interior"), (exterior, "exterior")] {
        let batch = eval(&solver, points, PointRole::Field);
        for (i, x) in points.iter().enumerate() {
            let (u, g) = reference.potential_gradient(*x);
            let h = reference.hessian(*x).expect("Feynman Hessian");
            let label = format!("{name} {kind} {i}");
            assert_bound(&label, &check(&label, &batch, i, u, g, h), FEYNMAN_TOLERANCE);
        }
    }
}

#[test]
#[ignore = "slow high-precision Feynman benchmark; run with --ignored"]
fn cauchy_alpha_one_matches_feynman() {
    let density = cauchy(&[([0.3, -0.2, 0.1], 2.0, 1.5, 1.0), ([-0.4, 0.3, 0.2], 1.0, 0.8, 1.0)], 0.5);
    against_feynman(
        &TriMesh::unit_cube(),
        density,
        &[DVec3::new(0.1, 0.2, -0.3), DVec3::new(0.6, -0.5, 0.7)],
        &[DVec3::new(1.7, 0.2, 0.4), DVec3::new(-1.5, -1.5, 1.2)],
        "cube alpha=1",
    );
}

#[test]
#[ignore = "slow high-precision Feynman benchmark; run with --ignored"]
fn cauchy_general_alpha_matches_feynman() {
    let density = cauchy(&[([0.2, 0.1, -0.3], 1.5, 2.0, 2.0), ([-0.3, 0.0, 0.4], 2.0, 1.0, 0.5)], 0.0);
    against_feynman(
        &TriMesh::uv_sphere(14, 1.0),
        density,
        &[DVec3::new(0.2, -0.3, 0.1)],
        &[DVec3::new(0.0, 1.8, 0.5)],
        "sphere alpha mix",
    );
}

#[test]
#[ignore = "slow high-precision Feynman benchmark; run with --ignored"]
fn cauchy_face_hessian_matches_feynman() {
    let density = cauchy(&[([0.1, 0.2, 0.3], 2.0, 1.2, 1.0)], 0.7);
    let mesh = TriMesh::unit_cube();
    let solver = solver(&mesh, density.clone());
    let reference = Feynman::new(&mesh, &density, G);
    for face in [0usize, 5, 10] {
        let [a, b, c] = mesh.triangle(face);
        let x = (a + b + c) * (1.0 / 3.0);
        let batch = eval(&solver, &[x], PointRole::SurfaceFace);
        let (u, g) = reference.potential_gradient(x);
        let h = reference.face_hessian(x, face).expect("face Hessian");
        let label = format!("face {face}");
        assert_bound(&label, &check(&label, &batch, 0, u, g, h), FEYNMAN_TOLERANCE);
    }
}

#[test]
fn contact_term_gives_the_poisson_trace() {
    let density = cauchy(&[([0.0, 0.0, 0.0], 2.0, 1.0, 1.0)], 0.3);
    let mesh = TriMesh::uv_sphere(20, 1.0);
    let solver = solver(&mesh, density.clone());
    let inside = DVec3::new(0.2, 0.1, -0.3);
    let outside = DVec3::new(0.0, 0.0, 2.2);
    let batch = eval(&solver, &[inside, outside], PointRole::Field);
    let trace = |i: usize| batch.hessian[i][0] + batch.hessian[i][1] + batch.hessian[i][2];
    assert!(relative(trace(0), -4.0 * PI * G * density.rho(inside), 1e-300) <= TOLERANCE);
    assert!(trace(1).abs() <= TOLERANCE * 4.0 * PI * G * density.rho(inside));
    assert_eq!(batch.theta, vec![1.0, 0.0]);
}

#[test]
fn ellipsoid_mesh_matches_carlson() {
    // Mesh vertices on the ellipsoid. The polyhedron differs from the smooth body by
    // O(h^2), so this benchmark is held to a coarser bound than the exact references.
    let body = Ellipsoid::new(DVec3::ZERO, [1.6, 1.0, 0.7]);
    let base = TriMesh::uv_sphere(64, 1.0);
    let vertices = base
        .vertices
        .iter()
        .map(|v| DVec3::new(v.x * 1.6, v.y * 1.0, v.z * 0.7))
        .collect();
    let mesh = TriMesh::try_new(vertices, base.faces).unwrap();
    let rho = 1.7;
    let solver = solver(&mesh, CauchyDensity::uniform(rho).unwrap());
    let points = [DVec3::new(0.3, 0.2, 0.1), DVec3::new(3.0, 1.0, -1.0), DVec3::new(0.0, 0.0, 1.5)];
    let batch = eval(&solver, &points, PointRole::Field);
    for (i, x) in points.iter().enumerate() {
        let exact = body.field(*x, G, rho);
        let w = werner::uniform_polyhedron(&mesh, *x, G, rho);
        // The polyhedron is within a few 1e-3 of the smooth ellipsoid ...
        assert!(relative(w.potential, exact.potential, 1e-300) < 1e-2, "werner vs carlson {i}");
        // ... and RTFP reproduces that polyhedron to the tolerance.
        let label = format!("ellipsoid {i}");
        assert_within(&label, &check(&label, &batch, i, w.potential, w.gradient, w.hessian));
    }
}

#[test]
fn cauchy_sphere_matches_carlson_layers() {
    // Kernels centred on a sphere are functions of the radius: Carlson stacks homogeneous shells.
    let density = cauchy(&[([0.0, 0.0, 0.0], 2.0, 1.3, 1.0), ([0.0, 0.0, 0.0], 1.0, 0.7, 2.0)], 0.4);
    let mesh = TriMesh::uv_sphere(64, 1.0);
    let body = Ellipsoid::new(DVec3::ZERO, [1.0, 1.0, 1.0]);
    let solver = solver(&mesh, density.clone());
    let points = [DVec3::new(0.3, 0.2, -0.4), DVec3::new(0.0, 1.8, 0.6)];
    let batch = eval(&solver, &points, PointRole::Field);
    for (i, x) in points.iter().enumerate() {
        let exact = body.cauchy_field(*x, G, &density).expect("carlson layers");
        let label = format!("carlson cauchy {i}");
        // the 64-latitude polyhedron is within ~1e-3 of the sphere
        assert_bound(&label, &check(&label, &batch, i, exact.potential, exact.gradient, exact.hessian), TOLERANCE);
    }
}

#[test]
fn polyhedron_volume_matches_the_divergence_formula() {
    let cube = uniform_integrals(&TriMesh::unit_cube(), 3.0);
    assert!((cube.volume - 8.0).abs() < 1e-12);
    assert!((cube.mass - 24.0).abs() < 1e-12);
}

#[test]
fn cauchy_mass_matches_the_kernel_integral() {
    // Mass of a wide kernel over a unit-radius sphere against the 1-D radial integral.
    let (weight, sigma) = (2.0, 0.5);
    let density = cauchy(&[([0.0, 0.0, 0.0], weight, sigma, 1.0)], 0.0);
    let mesh = TriMesh::uv_sphere(48, 1.0);
    let solver = solver(&mesh, density);
    let moments = solver.density_integrals().expect("density integrals");
    let n = 20_000;
    let mut exact = 0.0;
    for k in 0..n {
        let r = (k as f64 + 0.5) / n as f64;
        exact += 4.0 * PI * r * r * weight / (1.0 + sigma * sigma * r * r) / n as f64;
    }
    // The 48-latitude polyhedron has slightly less volume than the unit sphere.
    let volume_ratio = uniform_integrals(&mesh, 1.0).volume / (4.0 / 3.0 * PI);
    assert!(relative(moments.mass, exact * volume_ratio, 1e-300) < 5e-3, "{} vs {}", moments.mass, exact);
    assert!(moments.centroid.length() < 1e-6);
}

#[test]
fn speed_on_a_dense_mesh() {
    // 20k-face body, 64 gradient points near the surface.
    let mesh = TriMesh::uv_sphere(100, 1.0);
    let density = cauchy(&[([0.1, 0.0, 0.0], 1.0, 2.0, 1.0), ([-0.2, 0.1, 0.0], 1.0, 3.0, 1.0)], 1.0);
    let solver = solver(&mesh, density);
    let points: Vec<DVec3> = (0..64)
        .map(|k| {
            let t = k as f64 * 0.61803398875 * 2.0 * PI;
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / 64.0;
            let r = (1.0 - z * z).sqrt();
            DVec3::new(r * t.cos(), r * t.sin(), z) * 1.01
        })
        .collect();
    let start = Instant::now();
    let batch = solver
        .eval(&PointSet::from_points(&points), PointRole::Field, FieldMask::GRAD)
        .expect("dense evaluation");
    let seconds = start.elapsed().as_secs_f64();
    println!("{} faces, 64 points: {seconds:.2} s", mesh.faces.len());
    assert!(batch.gradient.iter().all(|g| g.is_finite()));
    assert!(seconds < 60.0, "dense evaluation took {seconds:.1} s");
}
