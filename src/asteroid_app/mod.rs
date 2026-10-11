//! Tier 3 window. It starts Tier 2 on the project mesh and draws each new body.
//! Rendering does not smooth or fuse that body.

pub mod pipeline;
pub mod render;
pub mod resources;
pub mod ui;

use bevy::light::GlobalAmbientLight;
use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

use crate::asteroid_app::pipeline::view_fit;
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
            app.insert_resource(ClearColor(Color::srgb(0.11, 0.12, 0.14)))
                .insert_resource(GlobalAmbientLight {
                    color: Color::WHITE,
                    brightness: 180.0,
                    affects_lightmapped_meshes: true,
                })
                .add_plugins((PanOrbitCameraPlugin, WireframePlugin::default()))
                .add_systems(
                    Startup,
                    (ui::spawn_scene, ui::begin_evolution).after(load_scene),
                )
                .add_systems(
                    Update,
                    (ui::control_keys, ui::poll_worker, ui::refresh_hud).chain(),
                );
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
