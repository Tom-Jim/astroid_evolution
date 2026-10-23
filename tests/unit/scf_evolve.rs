//! Unit tests of `scf::evolve`, moved out of `src/scf/evolve.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rheo::RheoGpu;
    pub use astroid_evolution::scf::basis::{Want, contract_velocity_rays};
    pub use astroid_evolution::scf::body::Body;
    pub use astroid_evolution::scf::cores::{self, CoreReport, Quadrature};
    pub use astroid_evolution::scf::forward::Forward;
    pub use astroid_evolution::scf::forward::Loading;
    pub use astroid_evolution::scf::frame::{FrameReport, realign, vertex_areas};
    pub use astroid_evolution::scf::mechanics::{Mechanics, evaluate};
    pub use astroid_evolution::scf::params::Project;
    pub use astroid_evolution::scf::relax::{AcceptError, accept};
    pub use astroid_evolution::scf::socp::{Kind, SocpInput, SocpSeed, mass_jacobian, solve};
    pub use astroid_evolution::scf::spin::SpinSchedule;
    pub use astroid_evolution::scf::transport::{
        DirIndex, locate_face, locate_vertex, mode_c, mode_c_over, surface_flow, transport,
    };
    pub use nalgebra::{Matrix3, Vector3};
    pub use serde::Serialize;
}
use ctx::*;

use std::sync::Arc;

use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::evolve::radial_increment;
use astroid_evolution::scf::evolve::{IDENTITY, radial_increments, step_fields};
use astroid_evolution::scf::frame::{FrameDeparture, FrameReport};

#[test]
fn radial_increment_keeps_a_transverse_face_and_drops_a_grazing_one() {
    assert!((radial_increment(0.1, 0.2) - 0.5).abs() < 1e-12);
    assert!((radial_increment(0.1, -0.2) + 0.5).abs() < 1e-12);
    assert_eq!(radial_increment(10.0, 0.01), 0.0);
    assert_eq!(radial_increment(10.0, -0.01), 0.0);
    assert_eq!(radial_increment(10.0, 0.0), 0.0);
}

#[test]
fn a_spin_step_records_zero_radial_increment() {
    let zeros = radial_increments(false, 0.4, 5, Some(&[1.0, 2.0, 3.0, 4.0, 5.0]));
    assert!(zeros.iter().all(|value| *value == 0.0));
    let scaled = radial_increments(true, 0.5, 3, Some(&[1.0, 2.0, 4.0]));
    assert_eq!(scaled, vec![0.5, 1.0, 2.0]);
}

#[test]
fn realigned_fields_keep_the_post_step_vertex_count() {
    let topo = Arc::new(Topology::icosphere(0));
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let body = Body::new(
        topo.clone(),
        vec![1.0; nv],
        1000.0,
        Vec::<CauchyKernel>::new(),
        vec![0.0; nv],
    )
    .unwrap();
    let f_delta: Vec<f64> = (0..nv).map(|i| i as f64).collect();
    let movable: Vec<bool> = (0..nv).map(|i| i % 2 == 0).collect();
    let v_sl = topo.dirs.clone();
    let slack: Vec<f64> = (0..nf).map(|f| f as f64).collect();
    let g_face = topo.face_dirs.clone();
    let d_radial: Vec<f64> = (0..nv).map(|i| 0.1 * i as f64).collect();
    let report = FrameReport {
        applied: true,
        passes: 1,
        translation: DVec3::ZERO,
        rotation: IDENTITY,
        angle_deg: 0.0,
        before: FrameDeparture {
            centroid: 0.0,
            off_diagonal: 0.0,
            ordering: 0.0,
        },
        after: FrameDeparture {
            centroid: 0.0,
            off_diagonal: 0.0,
            ordering: 0.0,
        },
        mass_change: 0.0,
        volume_change: 0.0,
        max_radius_change: 0.0,
    };
    let fields = step_fields(
        &vec![2.0; nv],
        &f_delta,
        &movable,
        &v_sl,
        &topo.dirs,
        &f_delta,
        &f_delta,
        &d_radial,
        &slack,
        &g_face,
        &vec![[3.0, 3.0, 3.0, 0.0, 0.0, 0.0]; slack.len()],
        &[],
        false,
        Some(&report),
        &body,
    );
    // An isotropic stress is unchanged by the frame rotation.
    assert_eq!(fields.stress.len(), slack.len());
    for t in &fields.stress {
        assert!((t[0] - 3.0).abs() < 1e-12 && (t[1] - 3.0).abs() < 1e-12 && (t[2] - 3.0).abs() < 1e-12);
        assert!(t[3].abs() < 1e-12 && t[4].abs() < 1e-12 && t[5].abs() < 1e-12);
    }
    assert_eq!(fields.f_delta.len(), body.radius.len());
    assert_eq!(fields.v_sl.len(), body.radius.len());
    assert_eq!(fields.normal.len(), body.radius.len());
    assert_eq!(fields.movable.len(), body.radius.len());
    assert_eq!(fields.d_radial.len(), body.radius.len());
    assert_eq!(fields.slack.len(), body.topo.n_faces());
    assert_eq!(fields.radius_from, vec![2.0; nv]);
    assert!(!fields.flow_active);
    for i in 0..nv {
        assert!(
            (fields.f_delta[i] - f_delta[i]).abs() < 1.0e-6,
            "vertex {i}: {} vs {}",
            fields.f_delta[i],
            f_delta[i]
        );
        assert_eq!(fields.movable[i], movable[i]);
        assert!((fields.d_radial[i] - d_radial[i]).abs() < 1.0e-12);
        assert!((fields.v_sl[i] - v_sl[i]).length() < 1.0e-6);
    }
    for f in 0..nf {
        assert!(
            (fields.slack[f] - slack[f]).abs() < 1.0e-6,
            "face {f}: {} vs {}",
            fields.slack[f],
            slack[f]
        );
    }
}

#[test]
fn slope_promotion_is_local_to_the_supercritical_faces() {
    use astroid_evolution::scf::evolve::slope_promotion;
    // Two faces sharing the edge 1–2; face 0 is supercritical, face 1 is not.
    let faces = [[0u32, 1, 2], [1, 2, 3]];
    let slack = [0.05, 0.0];
    let tol = 1e-3;
    // Nothing yields: every loose vertex of face 0 is promoted, vertex 3 is not.
    let (movable, promoted) =
        slope_promotion(&faces, &[false; 4], &slack, &[1.0, 1.0, 1.0, 1.0], tol);
    assert_eq!(movable, vec![true, true, true, false]);
    assert_eq!(promoted, vec![true, true, true, false]);
    // A vertex without loose material stays locked.
    let (movable, _) = slope_promotion(&faces, &[false; 4], &slack, &[1.0, 0.0, 1.0, 1.0], tol);
    assert_eq!(movable, vec![true, false, true, false]);
    // A yielded vertex elsewhere (3) does not switch the promotion off, and a yielded
    // vertex of the supercritical face (0) is movable without being promoted.
    let (movable, promoted) =
        slope_promotion(&faces, &[true, false, false, true], &slack, &[1.0; 4], tol);
    assert_eq!(movable, vec![true, true, true, true]);
    assert_eq!(promoted, vec![false, true, true, false]);
    // Below the tolerance nothing is promoted.
    let (movable, promoted) = slope_promotion(&faces, &[false; 4], &[1e-4, 0.0], &[1.0; 4], tol);
    assert!(movable.iter().all(|m| !*m) && promoted.iter().all(|p| !*p));
}
