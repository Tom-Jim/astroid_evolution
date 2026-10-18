//! Tier 3 window. Tier 2 stays off the frame loop. Each accepted step becomes a
//! playback snapshot; the shader morphs radii and colours the surface fields.

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

use crate::asteroid_app::material::AsteroidMaterial;
use crate::asteroid_app::pan_orbit_input::CustomInputPlugin;
use crate::asteroid_app::pipeline::{EvolutionPool, view_fit};
use crate::asteroid_app::playback::{Playback, VisMode};
use crate::asteroid_app::resources::{AppConfig, Evolution, Scene};
use crate::scf_simulator::params::Project;

pub struct AsteroidPlugin {
    pub config: AppConfig,
    /// `false` parses the project and skips the window, the lights, and the worker (tests).
    pub windowed: bool,
}

impl Plugin for AsteroidPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .init_resource::<Evolution>()
            .add_systems(Startup, load_scene);
        if self.windowed {
            bevy::asset::embedded_asset!(app, "shaders/asteroid.wesl");
            app.insert_resource(ClearColor(Color::srgb(0.11, 0.12, 0.14)))
                .insert_resource(GlobalAmbientLight {
                    color: Color::WHITE,
                    brightness: 180.0,
                    affects_lightmapped_meshes: true,
                })
                .insert_resource(EvolutionPool::new())
                .init_resource::<Playback>()
                .init_resource::<VisMode>()
                .add_plugins((
                    MeshPickingPlugin,
                    DefaultPanOrbitCameraPlugins,
                    CustomInputPlugin,
                    MaterialPlugin::<AsteroidMaterial>::default(),
                    WireframePlugin::default(),
                ))
                .add_systems(Startup, ui::spawn_scene.after(load_scene))
                .add_systems(Startup, ui::begin_evolution.after(ui::spawn_scene))
                .add_systems(
                    Update,
                    (
                        playback::poll_snapshot,
                        playback::advance_playback,
                        playback::sync_material,
                        ui::control_keys,
                        ui::refresh_hud,
                    )
                        .chain(),
                )
                .add_systems(Last, ui::stop_on_exit);
        }
    }
}

fn load_scene(mut commands: Commands, config: Res<AppConfig>) {
    let project = Project::parse(&config.project_text, config.project_dir.clone())
        .unwrap_or_else(|err| panic!("project: {err}"));
    let fit = view_fit(&project).unwrap_or_else(|err| panic!("model: {err}"));
    commands.insert_resource(Scene {
        vertices: fit.vertices,
        faces: fit.faces,
        center: fit.center,
        scale: fit.scale,
    });
}
