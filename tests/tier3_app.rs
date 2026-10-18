use std::path::PathBuf;
use std::sync::Arc;

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::pipeline::SurfaceFrame;
use astroid_evolution::asteroid_app::resources::{
    AppConfig, Evolution, Phase, Scene, display_imported, display_positions, lock_scale,
};
use astroid_evolution::asteroid_app::ui::hud_lines;
use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::scf_simulator::body::{Body, Topology};
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
        project_path: "assets/projects/ryugu_cauchy.toml".into(),
        mesh_path: "assets/models/ryugu.glb".into(),
        mesh_unit: "km".into(),
        spin: "const:4.986655e-4@3.15576e13".into(),
        max_steps: Some(1),
    }
}

#[test]
fn the_view_keeps_file_axes_and_locks_scale_from_the_imported_mesh() {
    let samples = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]];
    let scale = lock_scale(&samples);
    assert!((scale - 0.2).abs() < 1.0e-12);
    let shown = display_positions(&[[0.0, 4.0, 5.0]], scale);
    assert!((shown[0][0]).abs() < 1.0e-6);
    assert!((shown[0][1] - 0.8).abs() < 1.0e-6);
    assert!((shown[0][2] - 1.0).abs() < 1.0e-6);
    let placed = display_imported(&[[1.0, 2.0, 3.0]], [1.0, 0.0, 0.0], 0.5);
    assert!((placed[0][0]).abs() < 1.0e-6);
    assert!((placed[0][1] - 1.0).abs() < 1.0e-6);
    assert!((placed[0][2] - 1.5).abs() < 1.0e-6);
}

#[test]
fn a_frame_is_the_tier2_body_itself() {
    let topo = Arc::new(Topology::icosphere(1));
    let count = topo.n_vertices();
    let body = Body::new(
        topo.clone(),
        vec![2.0; count],
        1000.0,
        Vec::<CauchyKernel>::new(),
        vec![1.0; count],
    )
    .unwrap();
    let frame = SurfaceFrame::from_body(&body, 4, 12.0, 3.0e-4, 9.0);
    assert_eq!(frame.vertices.len(), count);
    assert_eq!(frame.faces, topo.faces);
    assert_eq!(
        (frame.step, frame.time, frame.omega, frame.mass),
        (4, 12.0, 3.0e-4, 9.0)
    );
    assert!((frame.radius_min - 2.0).abs() < 1.0e-12);
    assert!((frame.radius_max - 2.0).abs() < 1.0e-12);
    for (shown, position) in frame.vertices.iter().zip(body.positions()) {
        assert!((shown[0] - position.x).abs() < 1.0e-12);
        assert!((shown[1] - position.y).abs() < 1.0e-12);
        assert!((shown[2] - position.z).abs() < 1.0e-12);
    }
}

fn scene() -> Scene {
    Scene {
        vertices: Vec::new(),
        faces: Vec::new(),
        center: [0.0, 0.0, 0.0],
        scale: 0.002,
    }
}

#[test]
fn the_corner_label_names_the_project_the_mesh_and_the_constant_spin() {
    let waiting = hud_lines(
        &config(),
        &scene(),
        &Evolution {
            phase: Phase::Preparing,
            status: "computing the first tier 2 step".into(),
            ..Evolution::default()
        },
        0.0,
        "combined",
    )
    .join("\n");
    assert!(waiting.contains("project  assets/projects/ryugu_cauchy.toml"));
    assert!(waiting.contains("mesh     assets/models/ryugu.glb  (km)"));
    assert!(waiting.contains("spin     const:4.986655e-4@3.15576e13"));
    assert!(waiting.contains("showing  imported glb"));
    assert!(waiting.contains("file axes"));
    assert!(waiting.contains("1 = 500.0 m"));

    let shown = hud_lines(
        &config(),
        &scene(),
        &Evolution {
            phase: Phase::Running,
            status: "tier 2 running".into(),
            stats: Some(astroid_evolution::asteroid_app::resources::FrameStats {
                step: 3,
                time: 1.0,
                omega: 4.986655e-4,
                mass: 1.0,
                radius_min: 400.0,
                radius_max: 500.0,
                yielded_fraction: 0.25,
                max_slack: 1.0e-3,
                rho_s_total: 10.0,
                max_displacement: 0.4,
            }),
            ..Evolution::default()
        },
        0.4,
        "yield",
    )
    .join("\n");
    assert!(shown.contains("showing  tier 2 step 3"));
    assert!(shown.contains("tier 2 frame"));
}

#[test]
fn a_headless_viewer_records_the_input_and_does_not_start_tier2() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins).add_plugins(AsteroidPlugin {
        config: config(),
        windowed: false,
    });
    app.update();
    let scene = app.world().resource::<Scene>();
    assert_eq!(scene.vertices.len(), 98306);
    assert_eq!(scene.faces.len(), 196608);
    assert!(
        scene.scale > 0.001 && scene.scale < 0.003,
        "{}",
        scene.scale
    );
    let evolution = app.world().resource::<Evolution>();
    assert_eq!(evolution.phase, Phase::Idle);
    assert!(evolution.receiver.lock().unwrap().is_none());
    assert!(evolution.stats.is_none());
    let stored = app.world().resource::<AppConfig>();
    assert_eq!(stored.project_path, "assets/projects/ryugu_cauchy.toml");
    assert_eq!(stored.mesh_path, "assets/models/ryugu.glb");
    assert_eq!(stored.mesh_unit, "km");
    assert!(stored.spin.starts_with("const:"));
}
