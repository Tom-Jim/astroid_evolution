use std::path::PathBuf;
use std::sync::Arc;

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::pipeline::{
    Evolver, fuse_frame, load_original, to_mesh_frame,
};
use astroid_evolution::asteroid_app::render::{colors, delta_color};
use astroid_evolution::asteroid_app::resources::{AppConfig, Scene};
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf_simulator::params::Project;
use astroid_evolution::sh_smooth::Mesh;
use bevy::prelude::*;

fn project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/projects")
}

fn project_text() -> String {
    std::fs::read_to_string(project_dir().join("ryugu_cauchy.toml")).expect("ryugu project")
}

fn config() -> AppConfig {
    AppConfig {
        project_text: project_text(),
        project_dir: project_dir(),
        spin: "const:4.986655e-4@3.15576e13".into(),
        max_steps: Some(1),
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[test]
fn delta_colors_are_finite_neutral_at_zero_and_antisymmetric() {
    let neutral = delta_color(0.0, 5.0);
    assert!((neutral[0] - neutral[2]).abs() < 1.0e-6);
    let out = delta_color(5.0, 5.0);
    let inward = delta_color(-5.0, 5.0);
    assert!(out[0] > out[2] && inward[2] > inward[0]);
    for (d, l) in [(f32::NAN, 1.0), (1.0, 0.0), (1.0e9, 1.0), (-1.0e9, 1.0)] {
        assert!(
            delta_color(d, l)
                .iter()
                .all(|c| c.is_finite() && (0.0..=1.0).contains(c))
        );
    }
    assert!(colors(&[0.0, 1.0, -2.0]).len() == 3);
}

#[test]
fn tier2_frame_maps_back_with_rotation_translation_and_origin() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let p = [DVec3::new(1.0, 2.0, 3.0)];
    let same = to_mesh_frame(&p, identity, DVec3::ZERO, DVec3::ZERO);
    assert_eq!(same[0], [1.0, 2.0, 3.0]);
    // x_mesh = Rᵀ x + t + o with R a quarter turn about z: Rᵀ(1,0,0) = (0,-1,0).
    let quarter = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let q = to_mesh_frame(
        &[DVec3::new(1.0, 0.0, 0.0)],
        quarter,
        DVec3::new(10.0, 0.0, 0.0),
        DVec3::new(0.0, 100.0, 0.0),
    );
    assert!((q[0][0] - 10.0).abs() < 1e-12);
    assert!((q[0][1] - 99.0).abs() < 1e-12);
    assert!(q[0][2].abs() < 1e-12);
}

#[test]
fn fusion_adds_the_coarse_displacement_and_keeps_vertices_and_faces() {
    let high = Mesh::from_radial(24, |_, _| 100.0).unwrap();
    let coarse = Mesh::from_radial(8, |_, _| 110.0).unwrap();
    let frame = fuse_frame(&high, &coarse, 3, 1.5, 2.0e-4, 7.0).unwrap();
    assert_eq!(frame.vertices.len(), high.vertices.len());
    assert_eq!(frame.to_mesh(&high).faces, high.faces);
    assert_eq!(
        (frame.step, frame.time, frame.omega, frame.mass),
        (3, 1.5, 2.0e-4, 7.0)
    );
    for (fused, delta) in frame.vertices.iter().zip(&frame.radial_delta) {
        assert!((f64::from(*delta) - 10.0).abs() < 1.0, "delta {delta}");
        assert!((norm(*fused) - 110.0).abs() < 1.0);
    }
    // No displacement leaves the original untouched.
    let same = fuse_frame(&high, &high, 0, 0.0, 0.0, 0.0).unwrap();
    assert!(same.radial_delta.iter().all(|d| d.abs() < 1.0e-3));
}

#[test]
fn the_viewer_starts_with_the_original_glb_and_no_tier2_state() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AsteroidPlugin {
        config: config(),
        windowed: false,
    });
    app.update();
    let project = Project::parse(&project_text(), project_dir()).unwrap();
    let original = load_original(&project).unwrap();
    let scene = app.world().resource::<Scene>();
    assert_eq!(scene.original.vertices, original.vertices);
    assert_eq!(scene.original.faces, original.faces);
    assert_eq!(scene.original.length_unit, "m");
    let view = scene.view_positions(&scene.original.vertices);
    let reach = view
        .iter()
        .map(|v| norm([f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]))
        .fold(0.0, f64::max);
    assert!((reach - 1.0).abs() < 1.0e-5);
}

#[test]
fn tier2_output_fuses_with_the_original_at_start_and_after_a_step() {
    let project = Project::parse(&project_text(), project_dir()).unwrap();
    let original = Arc::new(load_original(&project).unwrap());
    let evolver = Evolver::new(
        &project_text(),
        &project_dir(),
        original.clone(),
        &config().spin,
    )
    .expect("smooth, relax");
    let first = evolver.frame().unwrap();
    assert_eq!(first.vertices.len(), original.vertices.len());
    assert!(first.radial_delta.iter().all(|d| d.is_finite()));
    // The ancient-Ryugu rollback must visibly change the shape.
    assert!(first.delta_max - first.delta_min > 1.0);
    assert!(first.mass > 0.0);

    let mut frames = Vec::new();
    evolver
        .run(
            Some(1),
            || Ok(()),
            |frame| {
                frames.push(frame);
                Ok(())
            },
        )
        .expect("one tier-2 step");
    assert!(!frames.is_empty());
    let last = frames.last().unwrap();
    assert_eq!(last.vertices.len(), original.vertices.len());
    assert!(last.vertices.iter().flatten().all(|v| v.is_finite()));
}
