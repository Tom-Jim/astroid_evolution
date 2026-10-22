use bevy::camera::Exposure;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::camera_controller::pan_orbit_camera::prelude::*;
use bevy::mesh::MeshTag;
use bevy::pbr::wireframe::{Wireframe, WireframeColor};
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

use crate::app::material::{AsteroidMaterial, PlaybackParams};
use crate::app::pipeline::{self, EvolutionPool};
use crate::app::playback::{Playback, VisMode, vis_name};
use crate::app::render::{imported_mesh, particle_mesh};
use crate::app::resources::{
    AppConfig, Evolution, Hud, Phase, RegolithView, Scene, SurfaceView, display_imported,
};

/// One palette for the native window and the browser page (`web/index.html`
/// mirrors these values in CSS).
pub mod theme {
    use bevy::prelude::Color;
    /// Scene background, `#1c1f24`.
    pub const BACKGROUND: Color = Color::srgb(0.11, 0.12, 0.14);
    /// HUD panel, `rgba(10, 13, 15, 0.78)`.
    pub const PANEL: Color = Color::srgba(0.04, 0.05, 0.06, 0.78);
    /// HUD text, `#edeff5`.
    pub const TEXT: Color = Color::srgb(0.93, 0.94, 0.96);
    pub const FONT_PX: f32 = 15.0;
    pub const MARGIN_PX: f32 = 12.0;
    pub const PADDING_PX: f32 = 10.0;
}

const START_YAW: f32 = 0.6;
const START_PITCH: f32 = 0.35;
const START_RADIUS: f32 = 3.4;
/// Same yellow as the slope-hazard tint in `asteroid.wesl`.
const WIRE_COLOR: Color = Color::srgb(1.0, 0.85, 0.15);

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
            font_size: FontSize::Px(theme::FONT_PX),
            ..default()
        },
        TextColor(theme::TEXT),
        BackgroundColor(theme::PANEL),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(theme::MARGIN_PX),
            left: Val::Px(theme::MARGIN_PX),
            max_width: Val::Px(1000.0),
            padding: UiRect::all(Val::Px(theme::PADDING_PX)),
            ..default()
        },
    ));
    let surface = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]]));
    let stream = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4], [0.0; 4]]));
    let cores = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]; 3]));
    let material = materials.add(AsteroidMaterial {
        params: PlaybackParams::default(),
        surface,
        stream,
        cores,
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
    let receiver = pipeline::spawn(config, scale, evolution.control.clone(), pool);
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
                commands
                    .entity(entity)
                    .remove::<Wireframe>()
                    .remove::<WireframeColor>();
            } else {
                commands
                    .entity(entity)
                    .insert((Wireframe, WireframeColor { color: WIRE_COLOR }));
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
    // P: save the window to dist/.stage/output/screenshots.
    if keys.just_pressed(KeyCode::KeyP) {
        save_screenshot(&mut commands, "key");
    }
    // I: interior density cutaway through the centre, facing the camera;
    // [ and ] slide the cut plane.
    if keys.just_pressed(KeyCode::KeyI) {
        mode.interior = !mode.interior;
        mode.cut_offset = 0.0;
    }
    if mode.interior {
        if keys.just_pressed(KeyCode::BracketLeft) {
            mode.cut_offset -= 0.05;
        }
        if keys.just_pressed(KeyCode::BracketRight) {
            mode.cut_offset += 0.05;
        }
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
    let mut lines = hud_lines(
        &config,
        &scene,
        &evolution,
        playback.tau,
        vis_name(mode.mode),
    );
    if mode.interior {
        let [lo, hi] = playback.rho_range;
        lines.insert(
            lines.len() - 1,
            format!(
                "interior rho {lo:.0} -> {hi:.0} kg/m^3 (indigo -> yellow, contour every {:.0})   cut {:+.2}",
                (hi - lo) / 10.0,
                mode.cut_offset
            ),
        );
    }
    **text = lines.join("\n");
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
        format!("status   {:?}: {}", evolution.phase, evolution.status),
    ];
    if let Some(stats) = evolution.stats {
        lines.push(format!("showing  tier 2 step {}", stats.step));
        lines.push(format!(
            "time     t {:.4e} s ({:.3} Myr)   omega {:.6e} rad/s   period {:.4} h",
            stats.time,
            stats.time / 3.15576e13,
            stats.omega,
            std::f64::consts::TAU / stats.omega / 3600.0
        ));
        if let Ok(schedule) = crate::scf::spin::SpinSchedule::parse(&config.spin) {
            let tide = schedule.tide(stats.time);
            if tide > 0.0 {
                lines.push(format!(
                    "orbit    a {:.1} km   period {:.4} h   tide GM/a^3 {:.3e} s^-2",
                    schedule.semi_major_axis(stats.time) / 1e3,
                    std::f64::consts::TAU / tide.sqrt() / 3600.0,
                    tide
                ));
            }
        }
        lines.push(format!(
            "body     mass {:.6e} kg   radius [{:.1}, {:.1}] m",
            stats.mass, stats.radius_min, stats.radius_max
        ));
        lines.push(format!(
            "surface  yield {:.3}   max slack {:.3e}   rho_s {:.3e} kg   max |dR| {:.3e} m",
            stats.yielded_fraction, stats.max_slack, stats.rho_s_total, stats.max_displacement
        ));
        lines.push(format!(
            "balance  eps_mom {:.3e}   eps_lam {:.3e}   picard {} ({:.3e})",
            stats.eps_momentum, stats.eps_lambda, stats.picard_iterations, stats.eps_picard
        ));
        lines.push(format!(
            "sheet    mobile {:.3}   tensile {:.3e}   Ek^-1 {:.3e}   mass residual {:.3e}",
            stats.mobile_area_fraction,
            stats.tensile_fraction,
            stats.inverse_ekman,
            stats.mass_residual
        ));
        lines.push(format!(
            "cfl      bound {}   dt {:.3e} s   dt_cfl {:.3e} s",
            stats.dt_bound, stats.dt, stats.dt_cfl
        ));
        lines.push(format!(
            "layer    {layer}: {}   tau {tau:.3}",
            layer_legend(layer)
        ));
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
    lines.push("keys     Space pause   W wireframe   R reset view   1 yield  2 slope  3 regolith  4 combined   I interior  [ ] cut   P screenshot".into());
    lines.push("mouse    right drag orbit   left drag pan   wheel zoom".into());
    lines
}

/// What the colours of a surface layer mean (`shader/asteroid.wesl`).
fn layer_legend(layer: &str) -> &'static str {
    match layer {
        "yield" => "orange = yielded, by F_delta / p95",
        "slope" => "pulsing yellow = slope slack above tolerance",
        "regolith" => "tan = loose layer rho_s, moving stripes = flow",
        _ => "yield + slope + regolith",
    }
}

/// Write the window to `dist/.stage/output/screenshots/<tag>-<unix ms>.png`.
pub fn save_screenshot(commands: &mut Commands, tag: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use bevy::render::view::screenshot::{Screenshot, save_to_disk};
        let dir = std::path::Path::new("dist/.stage/output/screenshots");
        let _ = std::fs::create_dir_all(dir);
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(dir.join(format!("{tag}-{ms}.png"))));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (commands, tag);
}

/// `AE_AUTOSHOT=<seconds>`: every period, save the surface view and then the
/// interior cutaway (for unattended checks of a running evolution).
pub fn autoshot(
    time: Res<Time>,
    mut mode: ResMut<VisMode>,
    mut commands: Commands,
    mut next: Local<Option<(f32, u8)>>,
) {
    let Some(period) = std::env::var("AE_AUTOSHOT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    else {
        return;
    };
    let now = time.elapsed_secs();
    let (at, phase) = next.get_or_insert((period, 0));
    if now < *at {
        return;
    }
    match *phase {
        0 => {
            mode.interior = false;
            *phase = 1;
            *at = now + 1.0;
        }
        1 => {
            save_screenshot(&mut commands, "surface");
            mode.interior = true;
            *phase = 2;
            *at = now + 1.0;
        }
        _ => {
            save_screenshot(&mut commands, "interior");
            mode.interior = false;
            *phase = 0;
            *at = now + period;
        }
    }
}

pub fn stop_on_exit(mut exit: MessageReader<AppExit>, evolution: Res<Evolution>) {
    for _ in exit.read() {
        evolution.control.request_quit();
    }
}
