use bevy::camera::Exposure;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::camera_controller::pan_orbit_camera::prelude::*;
use bevy::mesh::MeshTag;
use bevy::pbr::wireframe::Wireframe;
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

use crate::asteroid_app::material::{AsteroidMaterial, PlaybackParams};
use crate::asteroid_app::pipeline::{self, EvolutionPool};
use crate::asteroid_app::playback::{Playback, VisMode, vis_name};
use crate::asteroid_app::render::{imported_mesh, particle_mesh};
use crate::asteroid_app::resources::{
    AppConfig, Evolution, Hud, Phase, RegolithView, Scene, SurfaceView, display_imported,
};

const START_YAW: f32 = 0.6;
const START_PITCH: f32 = 0.35;
const START_RADIUS: f32 = 3.4;

fn home_transform() -> Transform {
    let x = START_RADIUS * START_PITCH.cos() * START_YAW.sin();
    let y = START_RADIUS * START_PITCH.sin();
    let z = START_RADIUS * START_PITCH.cos() * START_YAW.cos();
    Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y)
}

/// Key light is a few thousand lux, not daylight, so it does not clip against the
/// default photographic exposure. Ambient fills the night side.
pub fn spawn_scene(
    mut commands: Commands,
    scene: Res<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AsteroidMaterial>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let home = home_transform();
    commands.spawn((
        Camera3d::default(),
        Exposure { ev100: 8.0 },
        PanOrbitCamera::default().with_initial_anchor_depth(f64::from(START_RADIUS)),
        home,
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
    let surface = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]]));
    let stream = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4], [0.0; 4]]));
    let material = materials.add(AsteroidMaterial {
        params: PlaybackParams::default(),
        surface,
        stream,
    });
    let positions = display_imported(&scene.vertices, scene.center, scene.scale);
    commands.spawn((
        SurfaceView,
        NoFrustumCulling,
        MeshTag::new(0),
        Mesh3d(meshes.add(imported_mesh(positions, &scene.faces))),
        MeshMaterial3d(material.clone()),
    ));
    commands.spawn((
        RegolithView,
        NoFrustumCulling,
        Pickable::IGNORE,
        MeshTag::new(1),
        Mesh3d(meshes.add(particle_mesh())),
        MeshMaterial3d(material),
    ));
}

/// Start Tier 2 as soon as the window exists. Tier 3 does not wait for a key.
pub fn begin_evolution(
    config: Res<AppConfig>,
    scene: Res<Scene>,
    pool: Res<EvolutionPool>,
    mut evolution: ResMut<Evolution>,
) {
    if evolution.phase == Phase::Idle {
        start_worker(&mut evolution, &config, scene.scale, &*pool);
    }
}

fn start_worker(evolution: &mut Evolution, config: &AppConfig, scale: f64, pool: &EvolutionPool) {
    let receiver = pipeline::spawn(
        config.project_text.clone(),
        config.project_dir.clone(),
        config.spin.clone(),
        config.max_steps,
        scale,
        evolution.control.clone(),
        pool,
    );
    *evolution.receiver.lock().unwrap() = Some(receiver);
    evolution.phase = Phase::Preparing;
    evolution.status = "loading the mesh into tier 2".into();
}

/// Space pauses or resumes. W toggles wireframe. R resets the camera. 1–4 select the layer.
pub fn control_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<AppConfig>,
    scene: Res<Scene>,
    pool: Res<EvolutionPool>,
    mut evolution: ResMut<Evolution>,
    mut mode: ResMut<VisMode>,
    mut commands: Commands,
    views: Query<(Entity, Has<Wireframe>), With<SurfaceView>>,
    mut cameras: Query<(&mut Transform, &mut PanOrbitCamera)>,
) {
    if keys.just_pressed(KeyCode::Space) {
        match evolution.phase {
            Phase::Idle => start_worker(&mut evolution, &config, scene.scale, &*pool),
            Phase::Running => {
                evolution.control.set_paused(true);
                evolution.phase = Phase::Paused;
            }
            Phase::Paused => {
                evolution.control.set_paused(false);
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
        let home = home_transform();
        for (mut transform, mut camera) in &mut cameras {
            *transform = home;
            camera.last_anchor_depth = -f64::from(START_RADIUS);
        }
    }
    if keys.just_pressed(KeyCode::Digit1) {
        mode.mode = 1;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mode.mode = 2;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mode.mode = 3;
    } else if keys.just_pressed(KeyCode::Digit4) {
        mode.mode = 4;
    }
}

pub fn refresh_hud(
    config: Res<AppConfig>,
    scene: Res<Scene>,
    evolution: Res<Evolution>,
    playback: Res<Playback>,
    mode: Res<VisMode>,
    mut label: Query<&mut Text, With<Hud>>,
) {
    let Ok(mut text) = label.single_mut() else {
        return;
    };
    **text = hud_lines(
        &config,
        &scene,
        &evolution,
        playback.tau,
        vis_name(mode.mode),
    )
    .join("\n");
}

pub fn hud_lines(
    config: &AppConfig,
    scene: &Scene,
    evolution: &Evolution,
    tau: f32,
    layer: &str,
) -> Vec<String> {
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
            "yield {:.3}   max slack {:.3e}   rho_s {:.3e} kg   max |dR| {:.3e} m",
            stats.yielded_fraction, stats.max_slack, stats.rho_s_total, stats.max_displacement
        ));
        lines.push(format!("tau {tau:.3}   layer {layer}"));
        lines.push(format!(
            "view     tier 2 frame   1 = {:.1} m",
            1.0 / scene.scale
        ));
    } else {
        lines.push("showing  imported glb".into());
        if scene.scale > 0.0 {
            lines.push(format!(
                "view     file axes   1 = {:.1} m",
                1.0 / scene.scale
            ));
        }
    }
    lines.push(
        "Space pause/resume   W wireframe   R reset view   1 yield  2 slope  3 regolith  4 combined   RMB orbit  LMB pan  wheel zoom"
            .into(),
    );
    lines
}

pub fn stop_on_exit(mut exit: MessageReader<AppExit>, evolution: Res<Evolution>) {
    for _ in exit.read() {
        evolution.control.request_quit();
    }
}
