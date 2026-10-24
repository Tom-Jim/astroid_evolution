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
    let (topo, dirs) = Topology::icosphere(0);
    let topo = Arc::new(topo);
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        dirs.iter().map(|d| *d * 2.0).collect(),
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
        positions_from: dirs.clone(),
        f_delta: vec![0.0; count],
        movable: vec![false; count],
        v_sl: vec![DVec3::ZERO; count],
        normal: dirs.clone(),
        j_ero: vec![0.0; count],
        j_dep: vec![0.0; count],
        d_normal: vec![0.0; count],
        slack: vec![0.0; state.body.topo.n_faces()],
        g_face: vec![DVec3::ZERO; state.body.topo.n_faces()],
        // Uniaxial tension 5 Pa along x, compression along z.
        stress: vec![[5.0, 1.0, -2.0, 0.0, 0.0, 0.0]; state.body.topo.n_faces()],
        runout: Vec::new(),
        flow_active: false,
    };
    let snapshot = pack(&record(), &state, &fields, 0.5, 0.01, 1.5);
    assert!((snapshot.tension_max - 5.0).abs() < 1e-5);
    for (sample, dir) in snapshot.field.iter().zip(&dirs) {
        assert!((sample.sigma1 - 5.0).abs() < 1e-4);
        let [x, y, z] = sample.sigma_dir;
        // The tangent projection of ±x: unit, normal to the radial direction.
        let len = (x * x + y * y + z * z).sqrt();
        if dir.x.abs() < 0.99 {
            assert!((len - 1.0).abs() < 1e-4, "{len}");
            assert!((x as f64 * dir.x + y as f64 * dir.y + z as f64 * dir.z).abs() < 0.05);
            assert!(x.abs() > y.abs() && x.abs() > z.abs() || dir.x.abs() > 0.7);
        }
    }
    assert_eq!(snapshot.field.len(), state.body.positions.len());
    assert!(snapshot.field.iter().all(|sample| sample.d_normal == 0.0));
    assert!(!snapshot.flow_active);
    for (shown, dir) in snapshot.position_to.iter().zip(&dirs) {
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
    let (topo, dirs) = Topology::icosphere(0);
    let topo = Arc::new(topo);
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        dirs.iter().map(|d| *d * 2.0).collect(),
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

#[test]
fn the_history_returns_each_recorded_step_within_quantization() {
    use astroid_evolution::app::history::History;
    let (topo, dirs) = Topology::icosphere(2);
    let topo = Arc::new(topo);
    let count = topo.n_vertices();
    let faces = topo.n_faces();
    let mut history = History::default();
    let mut originals = Vec::new();
    for step in 0..3 {
        // A bump that grows with the step, and a stress that changes with it.
        let radius: Vec<f64> = dirs
            .iter()
            .map(|d| 2.0 + 0.1 * step as f64 * d.z.max(0.0))
            .collect();
        let body = Body::new(
            topo.clone(),
            dirs.iter().zip(&radius).map(|(d, r)| *d * *r).collect(),
            1000.0,
            Vec::<CauchyKernel>::new(),
            vec![0.0; count],
        )
        .unwrap();
        let state = State {
            body,
            level: 2,
            t: step as f64,
            step,
            omega: 0.2,
            frame_rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            frame_translation: DVec3::ZERO,
            velocity: Vec::new(),
        };
        let fields = StepFields {
            positions_from: state.body.positions.clone(),
            f_delta: (0..count).map(|i| i as f64 * 1e-3).collect(),
            movable: vec![true; count],
            v_sl: vec![DVec3::new(0.0, 1e-9, 0.0); count],
            normal: dirs.clone(),
            j_ero: vec![0.0; count],
            j_dep: vec![0.0; count],
            d_normal: vec![0.0; count],
            slack: vec![0.0; faces],
            g_face: vec![DVec3::ZERO; faces],
            stress: vec![[step as f64 + 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]; faces],
            // One straight 12-point path that hops halfway.
            runout: (0..12)
                .map(|k| [2.0, 0.1 * k as f64, 0.0, f64::from(u8::from(k > 5))])
                .collect(),
            flow_active: true,
        };
        let snap = pack(
            &StepRecord {
                step,
                t: step as f64,
                ..record()
            },
            &state,
            &fields,
            0.4,
            0.01,
            1.0,
        );
        history.record(&snap);
        originals.push(snap);
    }
    assert_eq!(history.len(), 3);
    for (i, original) in originals.iter().enumerate() {
        let back = history.frame(i).unwrap();
        assert_eq!(back.step, original.step);
        assert_eq!(back.faces, original.faces);
        assert_eq!(back.position_from, back.position_to);
        for (a, b) in back.position_to.iter().zip(&original.position_to) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-4, "step {i}: {a:?} vs {b:?}");
            }
        }
        for (a, b) in back.field.iter().zip(&original.field) {
            assert!((a.f_delta - b.f_delta).abs() < 1e-4);
            assert!((a.sigma1 - b.sigma1).abs() < 1e-3 * b.sigma1.abs().max(1.0));
        }
        assert!((back.tension_max - original.tension_max).abs() < 1e-9);
        assert_eq!(back.runout.len(), original.runout.len());
        for (a, b) in back.runout.iter().zip(&original.runout) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 2e-4, "{a:?} vs {b:?}");
            }
            assert_eq!(a[3], b[3]);
        }
        assert_eq!(history.stats(i).unwrap().step, original.step);
    }
    assert!(history.frame(3).is_none());
}
