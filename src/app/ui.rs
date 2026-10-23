use bevy::camera::Exposure;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::camera_controller::pan_orbit_camera::prelude::*;
use bevy::mesh::MeshTag;
use bevy::picking::Pickable;
use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;

use crate::app::material::{AsteroidMaterial, PlaybackParams};
use crate::app::pipeline::{self, EvolutionPool};
use crate::app::playback::VisMode;
use crate::app::render::{imported_mesh, quad_mesh};
use crate::app::snapshot::{PARTICLE_COUNT, TENSION_GLYPHS};
use crate::scf::evolve::RUNOUT_PATHS;
use crate::scf::runout::PATH_SAMPLES;
use crate::app::resources::{
    AppConfig, Evolution, Phase, RegolithView, Scene, SurfaceView, display_imported,
};

/// Scene background, near-black `#050608`, under the HUD panels of `ui/src/style.css`.
pub const BACKGROUND: Color = Color::srgb(0.02, 0.024, 0.031);

const START_YAW: f32 = 0.6;
const START_PITCH: f32 = 0.35;
pub const START_RADIUS: f32 = 3.4;
/// Same yellow as the slope-hazard tint in `asteroid.wesl`.
pub const WIRE_COLOR: Color = Color::srgb(1.0, 0.85, 0.15);

pub fn home_transform() -> Transform {
    let x = START_RADIUS * START_PITCH.cos() * START_YAW.sin();
    let y = START_RADIUS * START_PITCH.sin();
    let z = START_RADIUS * START_PITCH.cos() * START_YAW.cos();
    Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y)
}

/// Key light is a few thousand lux, not daylight, so it does not clip against the
/// default photographic exposure. Ambient fills the night side.
/// The wireframe in the browser, where WebGPU has no line polygon mode and Bevy's
/// `WireframePlugin` does not load. `playback::sync_wire` shows it and keeps it on the
/// surface.
#[cfg(target_arch = "wasm32")]
pub fn spawn_wire(
    mut commands: Commands,
    scene: Res<Scene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use crate::app::render::{edge_mesh, lifted};
    let positions = display_imported(&scene.vertices, scene.center, scene.scale);
    commands.spawn((
        crate::app::resources::WireView,
        NoFrustumCulling,
        Pickable::IGNORE,
        Visibility::Hidden,
        Mesh3d(meshes.add(edge_mesh(lifted(&positions), &scene.faces))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: WIRE_COLOR,
            unlit: true,
            ..default()
        })),
    ));
}

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
    let surface = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]]));
    let stream = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4], [0.0; 4]]));
    let cores = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]; 3]));
    let runout = buffers.add(ShaderBuffer::from(vec![[0.0f32; 4]]));
    let material = materials.add(AsteroidMaterial {
        params: PlaybackParams::default(),
        surface,
        stream,
        cores,
        runout,
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
        Mesh3d(meshes.add(quad_mesh(PARTICLE_COUNT))),
        MeshMaterial3d(material.clone()),
    ));
    // Runout trails: tag 3, one segment per pair of path samples (§2.8b).
    commands.spawn((
        NoFrustumCulling,
        Pickable::IGNORE,
        MeshTag::new(3),
        Mesh3d(meshes.add(quad_mesh(RUNOUT_PATHS * (PATH_SAMPLES - 1)))),
        MeshMaterial3d(material.clone()),
    ));
    // Tension ticks: tag 2, placed on the vertices by the shader.
    commands.spawn((
        NoFrustumCulling,
        Pickable::IGNORE,
        MeshTag::new(2),
        Mesh3d(meshes.add(quad_mesh(TENSION_GLYPHS))),
        MeshMaterial3d(material),
    ));
}

/// Start Tier 2 once the window has drawn for a second. Its first step keeps the GPU
/// busy for seconds; started with the window, it would hold back the first frame and
/// leave the window blank until that step ends.
pub fn begin_evolution(
    time: Res<Time<Real>>,
    config: Res<AppConfig>,
    scene: Res<Scene>,
    pool: Res<EvolutionPool>,
    mut evolution: ResMut<Evolution>,
) {
    if evolution.phase == Phase::Idle && time.elapsed_secs() > 1.0 {
        start_worker(&mut evolution, &config, scene.scale, &*pool);
    }
}

pub fn start_worker(evolution: &mut Evolution, config: &AppConfig, scale: f64, pool: &EvolutionPool) {
    let receiver = pipeline::spawn(config, scale, evolution.control.clone(), pool);
    *evolution.receiver.lock().unwrap() = Some(receiver);
    evolution.phase = Phase::Preparing;
    evolution.status = "loading the mesh into tier 2".into();
}

/// Folder for window captures. The desktop shell points it at the user's pictures; the
/// browser build downloads the file instead.
#[derive(Resource, Clone)]
pub struct ShotDir(pub std::path::PathBuf);

impl Default for ShotDir {
    fn default() -> Self {
        Self("dist/.stage/output/screenshots".into())
    }
}

/// Write the window to `<ShotDir>/<tag>-<unix ms>.png`.
pub fn save_screenshot(commands: &mut Commands, dir: &ShotDir, tag: &str) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};
    #[cfg(not(target_arch = "wasm32"))]
    let ms = {
        let _ = std::fs::create_dir_all(&dir.0);
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis())
    };
    #[cfg(target_arch = "wasm32")]
    let ms = js_sys::Date::now() as u128;
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(dir.0.join(format!("{tag}-{ms}.png"))));
}

/// `AE_AUTOSHOT=<seconds>`: every period, save the surface view and then the
/// interior cutaway (for unattended checks of a running evolution).
pub fn autoshot(
    time: Res<Time>,
    mut mode: ResMut<VisMode>,
    dir: Res<ShotDir>,
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
            save_screenshot(&mut commands, &dir, "surface");
            mode.interior = true;
            *phase = 2;
            *at = now + 1.0;
        }
        _ => {
            save_screenshot(&mut commands, &dir, "interior");
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
