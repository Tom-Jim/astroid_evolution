//! Acceptance of the standalone rheology and friction-angle solver.
//! The chain runs on the Metal GPU in double-single arithmetic.

use astroid_evolution::rheo_friction::selfcheck::{self, Check};
use astroid_evolution::rheo_friction::{RheoGpu, selfcheck::reference_material};

fn gpu() -> RheoGpu {
    RheoGpu::new().expect("a Metal device is available")
}

fn assert_pass(c: &Check) {
    println!("{:<64} {:.3e} <= {:.1e}", c.name, c.value, c.limit);
    assert!(
        c.passed(),
        "{}: {:e} exceeds {:e}",
        c.name,
        c.value,
        c.limit
    );
}

#[test]
fn rigid_motions_carry_no_surface_shear() {
    assert_pass(&selfcheck::rigid_motion(&gpu()).unwrap());
}

#[test]
fn spherical_harmonic_recursion_is_double_single_accurate() {
    assert_pass(&selfcheck::radius_recursion(&gpu()).unwrap());
}

#[test]
fn velocity_trace_matches_the_potentials() {
    assert_pass(&selfcheck::velocity_trace(&gpu()).unwrap());
}

#[test]
fn strain_rate_matches_a_velocity_stencil() {
    assert_pass(&selfcheck::strain_rate_stencil(&gpu()).unwrap());
}

#[test]
fn fixed_sphere_matches_proposition_10_3() {
    assert_pass(&selfcheck::sphere_closed_form(&gpu()).unwrap());
}

#[test]
fn closure_matches_the_f64_reference() {
    for c in selfcheck::closure_double_single(&gpu()).unwrap() {
        assert_pass(&c);
    }
}

#[test]
fn static_mode_pins_the_projection_to_mu_s() {
    assert_pass(&selfcheck::static_projection(&gpu()).unwrap());
}

#[test]
fn over_strength_tide_is_reported_not_converged() {
    assert_pass(&selfcheck::tide_beyond_strength(&gpu()).unwrap());
}

#[test]
fn friction_ranges_hold() {
    assert_pass(&selfcheck::friction_ranges(&gpu()).unwrap());
}

#[test]
fn missing_constants_are_model_not_closed() {
    let mut spec = reference_material(0.0, "dynamic").unwrap().spec;
    spec.mu_2 = spec.mu_s;
    let solver = astroid_evolution::rheo_friction::SolverSpec {
        max_iter: 10,
        tol: 1e-9,
    };
    let err = astroid_evolution::rheo_friction::Material::new(spec, solver).unwrap_err();
    assert_eq!(err.kind, "ModelNotClosed");
}
