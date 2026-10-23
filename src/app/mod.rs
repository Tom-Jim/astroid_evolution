//! Tier 3 window. Tier 2 stays off the frame loop. Each accepted step becomes a
//! playback snapshot; the shader morphs radii and colours the surface fields.

pub mod bridge;
pub mod history;
pub mod material;
pub mod pan_orbit_input;
pub mod pipeline;
pub mod playback;
pub mod render;
pub mod resources;
pub mod snapshot;
pub mod ui;

use bevy::camera_controller::pan_orbit_camera::prelude::*;
use bevy::light::GlobalAmbientLight;
use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;

use crate::app::material::AsteroidMaterial;
use crate::app::pan_orbit_input::CustomInputPlugin;
use crate::app::pipeline::{EvolutionPool, view_fit};
use crate::app::playback::{Playback, VisMode};
use crate::app::resources::{AppConfig, Evolution, Scene};
use crate::scf::params::Project;

pub struct AsteroidPlugin {
    pub config: AppConfig,
    /// `false` parses the project and skips the window, the lights, and the worker (tests).
    pub windowed: bool,
}

impl Plugin for AsteroidPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .init_resource::<Evolution>()
            .init_resource::<bridge::DataRoot>()
            .init_resource::<bridge::Notice>()
            .init_resource::<ui::ShotDir>()
            .add_systems(Startup, load_scene);
        if self.windowed {
            bevy::asset::embedded_asset!(app, "shader/asteroid.wesl");
            app.insert_resource(ClearColor(ui::BACKGROUND))
                .insert_resource(GlobalAmbientLight {
                    color: Color::WHITE,
                    brightness: 180.0,
                    affects_lightmapped_meshes: true,
                })
                .insert_resource(EvolutionPool::new())
                .init_resource::<Playback>()
                .init_resource::<VisMode>()
                .init_resource::<history::History>()
                .add_plugins((
                    MeshPickingPlugin,
                    DefaultPanOrbitCameraPlugins,
                    CustomInputPlugin,
                    MaterialPlugin::<AsteroidMaterial>::default(),
                    WireframePlugin::default(),
                ))
                .add_systems(Startup, ui::spawn_scene.after(load_scene))
                .add_systems(First, bridge::forward_pointer)
                .add_systems(
                    Update,
                    (
                        bridge::apply_commands,
                        ui::begin_evolution,
                        playback::poll_snapshot,
                        playback::advance_playback,
                        playback::sync_material,
                        ui::autoshot,
                        bridge::publish_status,
                    )
                        .chain(),
                )
                .add_systems(Last, ui::stop_on_exit);
            #[cfg(target_arch = "wasm32")]
            app.add_systems(Startup, ui::spawn_wire.after(ui::spawn_scene))
                .add_systems(Update, playback::sync_wire.after(playback::sync_material));
        }
    }
}

/// Open the viewer on the page canvas, after the project bytes are in memory.
#[cfg(target_arch = "wasm32")]
pub fn run_viewer(config: AppConfig) {
    App::new()
        .add_plugins(wasm_plugins())
        .add_plugins(AsteroidPlugin {
            config,
            windowed: true,
        })
        .run();
}

#[cfg(target_arch = "wasm32")]
fn wasm_plugins() -> bevy::app::PluginGroupBuilder {
    use bevy::asset::{AssetMetaCheck, AssetPlugin};
    use bevy::render::RenderPlugin;
    use bevy::render::settings::{Backends, RenderCreation, WgpuSettings, WgpuSettingsPriority};
    use bevy::window::{PresentMode, Window, WindowPlugin};

    DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#bevy".into()),
                fit_canvas_to_parent: true,
                prevent_default_event_handling: true,
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        })
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                backends: Some(Backends::BROWSER_WEBGPU),
                priority: WgpuSettingsPriority::WebGPU,
                ..default()
            })),
            ..default()
        })
        .set(AssetPlugin {
            meta_check: AssetMetaCheck::Never,
            ..default()
        })
}

fn load_scene(mut commands: Commands, config: Res<AppConfig>) {
    commands.insert_resource(scene_of(&config).unwrap_or_else(|err| panic!("{err}")));
}

/// The imported mesh of the project, fitted to the view.
pub fn scene_of(config: &AppConfig) -> Result<Scene, String> {
    let project = Project::parse(&config.project_text, config.project_dir.clone())
        .map_err(|err| format!("project: {err}"))?;
    let fit = view_fit(&project).map_err(|err| format!("model: {err}"))?;
    Ok(Scene {
        vertices: fit.vertices,
        faces: fit.faces,
        center: fit.center,
        scale: fit.scale,
    })
}
