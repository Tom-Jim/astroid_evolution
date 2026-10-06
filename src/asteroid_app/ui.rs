use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;

use bevy::camera::visibility::NoFrustumCulling;
use bevy::pbr::wireframe::Wireframe;
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCamera;

use crate::asteroid_app::pipeline::{self, WorkerMessage};
use crate::asteroid_app::render::{build_mesh, colors, update_mesh};
use crate::asteroid_app::resources::{AppConfig, Evolution, FrameStats, Phase, Scene};

#[derive(Component)]
pub struct SurfaceView;

#[derive(Component)]
pub struct Hud;

const START_YAW: f32 = 0.6;
const START_PITCH: f32 = 0.4;
const START_RADIUS: f32 = 3.4;

/// Camera (Bevy pan-orbit controller), lights, HUD, and the original model.
pub fn spawn_scene(
    mut commands: Commands,
    scene: Res<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        PanOrbitCamera {
            yaw: Some(START_YAW),
            pitch: Some(START_PITCH),
            radius: Some(START_RADIUS),
            zoom_lower_limit: 1.2,
            ..default()
        },
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, 0.6, 2.8, 0.0)),
    ));
    commands.spawn((
        Hud,
        Text::new(""),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.92, 0.94, 0.96)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
    let original = &scene.original;
    let mesh = build_mesh(
        scene.view_positions(&original.vertices),
        colors(&vec![0.0; original.vertices.len()]),
        &original.faces,
    );
    commands.spawn((
        SurfaceView,
        NoFrustumCulling,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
    ));
}

/// Space: start, then pause/resume. W: wireframe. R: reset the camera.
pub fn control_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<AppConfig>,
    scene: Res<Scene>,
    mut evolution: ResMut<Evolution>,
    mut commands: Commands,
    views: Query<(Entity, Has<Wireframe>), With<SurfaceView>>,
    mut cameras: Query<&mut PanOrbitCamera>,
) {
    if keys.just_pressed(KeyCode::Space) {
        match evolution.phase {
            Phase::Idle => {
                let receiver = pipeline::spawn(
                    config.project_text.clone(),
                    config.project_dir.clone(),
                    scene.original.clone(),
                    config.spin.clone(),
                    config.max_steps,
                    evolution.control.clone(),
                );
                *evolution.receiver.lock().unwrap() = Some(receiver);
                evolution.phase = Phase::Preparing;
                evolution.status = "starting".into();
            }
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

/// Drain the worker and show its newest fused frame.
pub fn poll_worker(
    mut evolution: ResMut<Evolution>,
    scene: Res<Scene>,
    view: Query<&Mesh3d, With<SurfaceView>>,
    mut meshes: ResMut<Assets<Mesh>>,
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
        delta_min: frame.delta_min,
        delta_max: frame.delta_max,
    });
    let Ok(handle) = view.single() else {
        return;
    };
    if let Some(mut mesh) = meshes.get_mut(&handle.0) {
        update_mesh(
            &mut mesh,
            scene.view_positions(&frame.vertices),
            colors(&frame.radial_delta),
        );
    }
}

pub fn refresh_hud(evolution: Res<Evolution>, mut label: Query<&mut Text, With<Hud>>) {
    let Ok(mut text) = label.single_mut() else {
        return;
    };
    let mut lines = vec![format!("{:?}: {}", evolution.phase, evolution.status)];
    if let Some(s) = evolution.stats {
        lines.push(format!(
            "step {}   t {:.4e} s   omega {:.6e} rad/s",
            s.step, s.time, s.omega
        ));
        lines.push(format!(
            "mass {:.6e} kg   dR [{:.2}, {:.2}] m (red out, blue in)",
            s.mass, s.delta_min, s.delta_max
        ));
    }
    lines.push(
        "Space start/pause   W wireframe   R reset view   LMB orbit  RMB pan  wheel zoom".into(),
    );
    text.0 = lines.join("\n");
}
