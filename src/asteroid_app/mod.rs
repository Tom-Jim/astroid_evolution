//! Tier 3 viewer: shows the original model, and on Space fuses it live with the Tier 2
//! evolution (`pipeline`). Render systems only read the pipeline output.

pub mod pipeline;
pub mod render;
pub mod resources;
pub mod ui;

use std::sync::Arc;

use bevy::pbr::wireframe::WireframePlugin;
use bevy::prelude::*;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

use crate::asteroid_app::resources::{AppConfig, Evolution, Scene};
use crate::scf_simulator::params::Project;

pub struct AsteroidPlugin {
    pub config: AppConfig,
    /// `false` loads the original model without any rendering (tests).
    pub windowed: bool,
}

impl Plugin for AsteroidPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .init_resource::<Evolution>()
            .add_systems(Startup, load_scene);
        if self.windowed {
            app.add_plugins((PanOrbitCameraPlugin, WireframePlugin::default()))
                .add_systems(Startup, ui::spawn_scene.after(load_scene))
                .add_systems(
                    Update,
                    (ui::control_keys, ui::poll_worker, ui::refresh_hud).chain(),
                );
        }
    }
}

fn load_scene(mut commands: Commands, config: Res<AppConfig>) {
    let project = Project::parse(&config.project_text, config.project_dir.clone())
        .unwrap_or_else(|e| panic!("project: {e}"));
    let original = pipeline::load_original(&project).unwrap_or_else(|e| panic!("model: {e}"));
    commands.insert_resource(Scene::new(Arc::new(original)));
}
