use std::process::Command;

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::components::{StarMesh, SurfaceFields};
use astroid_evolution::asteroid_app::driver::ScfActive;
use astroid_evolution::asteroid_app::playback::{PlaybackPlugin, SnapshotFrame, write_snapshot};
use astroid_evolution::asteroid_app::render::field_overlay::{coolwarm, plasma, turbo, viridis};
use bevy::prelude::*;

#[test]
fn colormap_is_centered_and_finite() {
    let neutral = coolwarm(0.0);
    assert!((neutral[0] - neutral[2]).abs() < 1.0e-6);
    let hot = coolwarm(0.8);
    let cold = coolwarm(-0.8);
    assert!((hot[0] - cold[2]).abs() < 1.0e-6);
    assert!((hot[2] - cold[0]).abs() < 1.0e-6);
    assert!((hot[1] - cold[1]).abs() < 1.0e-6);
    for sample in [0.0, 1.0, -1.0, 2.0, f64::NAN] {
        for color in [viridis(sample), plasma(sample), turbo(sample)] {
            assert!(
                color
                    .iter()
                    .all(|channel| channel.is_finite() && (0.0..=1.0).contains(channel))
            );
        }
    }
}

#[test]
fn playback_rebuilds_the_mesh_without_the_scf_plugin() {
    let path = std::env::temp_dir().join("asteroid-playback-tier3.aevb");
    write_snapshot(
        &path,
        &[SnapshotFrame {
            step: 2,
            time: 2.0,
            omega: [0.0, 0.0, 0.1],
            directions: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            radii: vec![1.0, 1.1, 0.9, 1.0],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            yield_criterion: vec![-0.2, 0.3, -0.1, -0.4],
        }],
    )
    .unwrap();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AsteroidPlugin { windowed: false })
        .add_plugins(PlaybackPlugin { path });
    app.update();
    assert!(app.world().get_resource::<ScfActive>().is_none());
    let mut meshes = app.world_mut().query::<&StarMesh>();
    let radii = meshes.iter(app.world()).next().unwrap().radii.clone();
    let mut fields = app.world_mut().query::<&SurfaceFields>();
    let mobile = fields.iter(app.world()).next().unwrap().mobile.clone();
    assert_eq!(radii.len(), 4);
    assert_eq!(radii[1].to_bits(), 1.1f64.to_bits());
    assert!(mobile[1]);
    assert!(!mobile[0]);
}

#[test]
fn headless_general_mesh_runs_without_claiming_stable() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mesh = root.join("assets/mesh/octahedron.obj");
    let mass = 1602.4101348454262_f64;
    let text = std::fs::read_to_string(root.join("assets/projects/sample_project.toml"))
        .expect("sample project")
        .replace("../models/ryugu.glb", mesh.to_str().expect("mesh path"))
        .replace("mass_target = 4.50e11", "mass_target = 1602.4101348454262")
        .replace("value = 4.50e11", "value = 1602.4101348454262")
        .replace("uncertainty = 5.85e9", "uncertainty = 20.0")
        .replace(
            "mass_tol = 0.013",
            &format!("mass_tol = {:.16}", 20.0 / mass),
        );
    let dir = std::env::temp_dir().join("astroid-evolution-tier3");
    std::fs::create_dir_all(&dir).unwrap();
    let project = dir.join("octahedron-general.toml");
    std::fs::write(&project, text).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_asteroid-app"))
        .args([
            "--project",
            project.to_str().unwrap(),
            "--headless-test",
            "--max-steps",
            "1",
        ])
        .output()
        .expect("asteroid-app");
    let err = String::from_utf8_lossy(&output.stderr);
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "status {:?}\n{err}\n{text}",
        output.status.code()
    );
    assert!(!err.contains("ModelUnsupported"), "{err}");
    assert!(text.contains("model general_mesh"), "{text}");
    assert!(!text.contains("verification_level passed"), "{text}");
    assert!(!text.contains("status stable"), "{text}");
}
