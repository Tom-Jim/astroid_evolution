//! Unit tests of `app::snapshot`, moved out of `src/app/snapshot.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::scf::evolve::{State, StepFields, StepRecord};
    pub use rayon::prelude::*;
}
use ctx::*;

use std::sync::Arc;

use astroid_evolution::app::snapshot::{
    morphed_position, pack, pack_initial, posed_normals, posed_positions, scatter_face_slack,
    smoothstep,
};
use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::{Body, Topology};
use astroid_evolution::scf::evolve::{State, StepFields, StepRecord};

fn record() -> StepRecord {
    StepRecord {
        step: 2,
        t: 3.0,
        advance: 1.0,
        omega: 0.2,
        kind: "spin-only",
        dt: 1.0,
        dt_bound: "omega",
        dt_omega: 1.0,
        quasi_static: true,
        core_mass_fraction: 1.0,
        mass: 5.0,
        min_radial: 1.0,
        ..StepRecord::default()
    }
}

#[test]
fn morphed_radius_is_the_scaled_ray() {
    let dir = [0.0, 1.0, 0.0];
    let start = morphed_position(dir, 2.0, 4.0, 0.0, 0.5);
    assert!(start[0].abs() < 1.0e-6);
    assert!((start[1] - 1.0).abs() < 1.0e-5);
    let end = morphed_position(dir, 2.0, 4.0, 1.0, 0.5);
    assert!((end[1] - 2.0).abs() < 1.0e-5);
    assert!((smoothstep(0.5) - 0.5).abs() < 1.0e-6);
    let mid = morphed_position(dir, 2.0, 4.0, 0.5, 1.0);
    assert!((mid[1] - 3.0).abs() < 1.0e-5);
}

#[test]
fn the_screen_pose_matches_the_shader_mix() {
    let from = [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]];
    let to = [[0.0, 0.0, 3.0], [6.0, 0.0, 0.0]];
    assert_eq!(posed_positions(&from, &to, 0.0), from);
    assert_eq!(posed_positions(&from, &to, 1.0), to);
    let s = smoothstep(0.25);
    let mid = posed_positions(&from, &to, 0.25);
    assert!((mid[0][2] - (1.0 + 2.0 * s)).abs() < 1.0e-6);
    assert!((mid[1][0] - (2.0 + 4.0 * s)).abs() < 1.0e-6);
    assert!((mid[0][0]).abs() < 1.0e-6);
    assert!((mid[0][1]).abs() < 1.0e-6);

    let from_n = [[0.0, 1.0, 0.0]];
    let to_n = [[0.0, 0.0, 1.0]];
    let start_n = posed_normals(&from_n, &to_n, 0.0);
    let end_n = posed_normals(&from_n, &to_n, 1.0);
    assert!((start_n[0][1] - 1.0).abs() < 1.0e-6);
    assert!((end_n[0][2] - 1.0).abs() < 1.0e-6);
    let mid_n = posed_normals(&from_n, &to_n, 0.5);
    let unit = (0.5f32).sqrt();
    assert!((mid_n[0][1] - unit).abs() < 1.0e-5);
    assert!((mid_n[0][2] - unit).abs() < 1.0e-5);
}

#[test]
fn face_slack_scatters_by_the_incident_maximum_and_the_area_weight() {
    let faces = [[0, 1, 2], [0, 2, 3]];
    let (max, avg) = scatter_face_slack(4, &faces, &[2.0, 6.0], &[1.0, 3.0]);
    assert!((max[0] - 6.0).abs() < 1.0e-6);
    assert!((max[1] - 2.0).abs() < 1.0e-6);
    assert!((max[2] - 6.0).abs() < 1.0e-6);
    assert!((avg[0] - 5.0).abs() < 1.0e-5);
    assert!((avg[1] - 2.0).abs() < 1.0e-5);
    assert!((avg[3] - 6.0).abs() < 1.0e-5);
}

#[test]
fn a_spin_step_snapshot_keeps_zero_radial_increment_and_stays_on_the_seed() {
    let topo = Arc::new(Topology::icosphere(0));
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        vec![2.0; count],
        1000.0,
        Vec::<CauchyKernel>::new(),
        vec![0.0; count],
    )
    .unwrap();
    let state = State {
        body,
        level: 0,
        t: 3.0,
        step: 2,
        omega: 0.2,
        frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        frame_translation: DVec3::ZERO,
        velocity: Vec::new(),
    };
    let fields = StepFields {
        radius_from: vec![1.0; count],
        f_delta: vec![0.0; count],
        movable: vec![false; count],
        v_sl: vec![DVec3::ZERO; count],
        normal: state.body.topo.dirs.clone(),
        j_ero: vec![0.0; count],
        j_dep: vec![0.0; count],
        d_radial: vec![0.0; count],
        slack: vec![0.0; state.body.topo.n_faces()],
        g_face: vec![DVec3::ZERO; state.body.topo.n_faces()],
        flow_active: false,
    };
    let snapshot = pack(&record(), &state, &fields, 0.5, 0.01, 1.5);
    assert_eq!(snapshot.field.len(), state.body.radius.len());
    assert!(snapshot.field.iter().all(|sample| sample.d_radial == 0.0));
    assert!(!snapshot.flow_active);
    for (shown, dir) in snapshot.position_to.iter().zip(&state.body.topo.dirs) {
        let expected = morphed_position([dir.x, dir.y, dir.z], 2.0, 2.0, 1.0, 0.5);
        assert!((shown[0] - expected[0]).abs() < 1.0e-5);
        assert!((shown[1] - expected[1]).abs() < 1.0e-5);
        assert!((shown[2] - expected[2]).abs() < 1.0e-5);
    }
    assert_eq!(
        snapshot.stream.len(),
        astroid_evolution::app::snapshot::PARTICLE_COUNT
            * astroid_evolution::app::snapshot::STREAMLINE_STEPS
    );
    for particle in 0..astroid_evolution::app::snapshot::PARTICLE_COUNT {
        let seed = (particle * count) / astroid_evolution::app::snapshot::PARTICLE_COUNT;
        for step in 0..astroid_evolution::app::snapshot::STREAMLINE_STEPS {
            let point = snapshot.stream
                [particle * astroid_evolution::app::snapshot::STREAMLINE_STEPS + step];
            assert_eq!(point.verts, [seed as u32; 3]);
        }
    }
}

#[test]
fn the_initial_snapshot_is_a_still_tier2_pose() {
    let topo = Arc::new(Topology::icosphere(0));
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        vec![2.0; count],
        1000.0,
        Vec::<CauchyKernel>::new(),
        vec![0.0; count],
    )
    .unwrap();
    let state = State {
        body,
        level: 0,
        t: 0.0,
        step: 0,
        omega: 0.1,
        frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        frame_translation: DVec3::ZERO,
        velocity: Vec::new(),
    };
    let snapshot = pack_initial(&state, 9.0, 0.5, 0.01);
    assert_eq!(snapshot.position_from, snapshot.position_to);
    assert_eq!(snapshot.normal_from, snapshot.normal_to);
    assert_eq!((snapshot.step, snapshot.mass), (0, 9.0));
    assert!(snapshot.field.iter().all(|sample| sample.f_delta == 0.0));
    assert_eq!(
        snapshot.stream.len(),
        astroid_evolution::app::snapshot::PARTICLE_COUNT
            * astroid_evolution::app::snapshot::STREAMLINE_STEPS
    );
    let shown = posed_positions(&snapshot.position_from, &snapshot.position_to, 0.5);
    assert_eq!(shown, snapshot.position_to);
}
