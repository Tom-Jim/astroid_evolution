use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::camera::Exposure;
use bevy::pbr::wireframe::Wireframe;
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

use crate::asteroid_app::pipeline::{self, WorkerMessage};
use crate::asteroid_app::render::{build_mesh, update_mesh};
use crate::asteroid_app::resources::{
    display_imported, display_positions, AppConfig, Evolution, FrameStats, Phase, Scene,
};

#[derive(Component)]
pub struct SurfaceView;

#[derive(Component)]
pub struct Hud;

const START_YAW: f32 = 0.6;
const START_PITCH: f32 = 0.35;
const START_RADIUS: f32 = 3.4;

/// Key light is a few thousand lux, not daylight, so it does not clip against the
/// default photographic exposure. Ambient fills the night side.
pub fn spawn_scene(
    mut commands: Commands,
    scene: Res<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Exposure { ev100: 8.0 },
        PanOrbitCamera {
            yaw: Some(START_YAW),
            pitch: Some(START_PITCH),
            radius: Some(START_RADIUS),
            zoom_lower_limit: 0.4,
            ..default()
        },
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 1_400.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_arc(
            Vec3::NEG_Z,
            Vec3::new(0.45, -1.0, 0.3).normalize(),
        )),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 500.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_rotation_arc(
            Vec3::NEG_Z,
            Vec3::new(-0.55, -0.35, -0.75).normalize(),
        )),
    ));
    commands.spawn((
        Hud,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.93, 0.94, 0.96)),
        BackgroundColor(Color::srgba(0.04, 0.05, 0.06, 0.78)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            max_width: Val::Px(720.0),
            padding: UiRect::all(Val::Px(10.0)),
            ..default()
        },
    ));
    commands.spawn((
        SurfaceView,
        NoFrustumCulling,
        Mesh3d(meshes.add(build_mesh(
            display_imported(&scene.vertices, scene.center, scene.scale),
            &scene.faces,
        ))),
        MeshMaterial3d(materials.add(surface_material())),
    ));
}

fn surface_material() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.5),
        perceptual_roughness: 1.0,
        metallic: 0.0,
        ..default()
    }
}

/// Start Tier 2 as soon as the window exists. Tier 3 does not wait for a key.
pub fn begin_evolution(config: Res<AppConfig>, mut evolution: ResMut<Evolution>) {
    if evolution.phase == Phase::Idle {
        start_worker(&mut evolution, &config);
    }
}

fn start_worker(evolution: &mut Evolution, config: &AppConfig) {
    let receiver = pipeline::spawn(
        config.project_text.clone(),
        config.project_dir.clone(),
        config.spin.clone(),
        config.max_steps,
        evolution.control.clone(),
    );
    *evolution.receiver.lock().unwrap() = Some(receiver);
    evolution.phase = Phase::Preparing;
    evolution.status = "loading the mesh into tier 2".into();
}

/// Space pauses or resumes. W toggles wireframe. R resets the camera.
pub fn control_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<AppConfig>,
    mut evolution: ResMut<Evolution>,
    mut commands: Commands,
    views: Query<(Entity, Has<Wireframe>), With<SurfaceView>>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    if keys.just_pressed(KeyCode::Space) {
        match evolution.phase {
            Phase::Idle => start_worker(&mut evolution, &config),
            Phase::Running => {
                evolution.control.paused.store(true, Ordering::Relaxed);
                evolution.phase = Phase::Paused;
            }
            Phase::Paused => {
                evolution.control.paused.store(false, Ordering::Relaxed);
                evolution.phase = Phase::Running;
            }
            Phase::Preparing | Phase::Done | Phase::Failed => {}
        }
    }
    if keys.just_pressed(KeyCode::KeyW) {
        for (entity, wireframe) in &views {
            if wireframe {
                commands.entity(entity).remove::<Wireframe>();
            } else {
                commands.entity(entity).insert(Wireframe);
            }
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        for mut camera in &mut cameras {
            camera.target_focus = Vec3::ZERO;
            camera.target_yaw = START_YAW;
            camera.target_pitch = START_PITCH;
            camera.target_radius = START_RADIUS;
        }
    }
}

/// Drain the worker. The imported glTF stays up until Tier 2 emits a step, then each
/// newer body replaces the one on screen.
pub fn poll_worker(
    mut commands: Commands,
    mut evolution: ResMut<Evolution>,
    scene: Res<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    view: Query<&Mesh3d, With<SurfaceView>>,
) {
    let mut latest = None;
    let mut closed = false;
    let mut messages = Vec::new();
    if let Some(receiver) = evolution.receiver.lock().unwrap().as_ref() {
        loop {
            match receiver.try_recv() {
                Ok(message) => messages.push(message),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    closed = true;
                    break;
                }
            }
        }
    }
    for message in messages {
        match message {
            WorkerMessage::Status(text) => evolution.status = text,
            WorkerMessage::Frame(frame) => {
                if evolution.phase == Phase::Preparing {
                    evolution.phase = Phase::Running;
                }
                latest = Some(frame);
            }
            WorkerMessage::Finished(text) => {
                evolution.phase = Phase::Done;
                evolution.status = text;
            }
            WorkerMessage::Failed(text) => {
                evolution.phase = Phase::Failed;
                evolution.status = text;
            }
        }
    }
    if closed
        && matches!(
            evolution.phase,
            Phase::Preparing | Phase::Running | Phase::Paused
        )
    {
        evolution.phase = Phase::Failed;
        evolution.status = "worker stopped unexpectedly".into();
    }
    let Some(frame) = latest else {
        return;
    };
    evolution.stats = Some(FrameStats {
        step: frame.step,
        time: frame.time,
        omega: frame.omega,
        mass: frame.mass,
        radius_min: frame.radius_min,
        radius_max: frame.radius_max,
    });
    let positions = display_positions(&frame.vertices, scene.scale);
    match view.single() {
        Ok(handle) => {
            if let Some(mut mesh) = meshes.get_mut(&handle.0) {
                let same_count = mesh
                    .attribute(Mesh::ATTRIBUTE_POSITION)
                    .is_some_and(|values| values.len() == positions.len());
                if same_count {
                    update_mesh(&mut mesh, positions);
                } else {
                    *mesh = build_mesh(positions, &frame.faces);
                }
            }
        }
        Err(_) => {
            commands.spawn((
                SurfaceView,
                NoFrustumCulling,
                Mesh3d(meshes.add(build_mesh(positions, &frame.faces))),
                MeshMaterial3d(materials.add(surface_material())),
            ));
        }
    }
}

pub fn refresh_hud(
    config: Res<AppConfig>,
    scene: Res<Scene>,
    evolution: Res<Evolution>,
    mut label: Query<&mut Text, With<Hud>>,
) {
    let Ok(mut text) = label.single_mut() else {
        return;
    };
    text.0 = hud_lines(&config, &scene, &evolution).join("\n");
}

pub fn hud_lines(config: &AppConfig, scene: &Scene, evolution: &Evolution) -> Vec<String> {
    let mut lines = vec![
        format!("project  {}", config.project_path),
        format!("mesh     {}  ({})", config.mesh_path, config.mesh_unit),
        format!("spin     {}", config.spin),
        format!("{:?}: {}", evolution.phase, evolution.status),
    ];
    if let Some(stats) = evolution.stats {
        lines.push(format!("showing  tier 2 step {}", stats.step));
        lines.push(format!(
            "step {}   t {:.4e} s   omega {:.6e} rad/s",
            stats.step, stats.time, stats.omega
        ));
        lines.push(format!(
            "tier 2 body   mass {:.6e} kg   radius [{:.1}, {:.1}] m",
            stats.mass, stats.radius_min, stats.radius_max
        ));
        lines.push(format!(
            "view     tier 2 frame   1 = {:.1} m",
            1.0 / scene.scale
        ));
    } else {
        lines.push("showing  imported glb".into());
        if scene.scale > 0.0 {
            lines.push(format!("view     file axes   1 = {:.1} m", 1.0 / scene.scale));
        }
    }
    lines.push(
        "Space pause/resume   W wireframe   R reset view   LMB orbit  RMB pan  wheel zoom".into(),
    );
    lines
}
