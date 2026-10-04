//! Tier 3 viewer. Render and playback systems only read the physics components.

pub mod components;
pub mod driver;
pub mod events;
pub mod playback;
pub mod render;
pub mod resources;
pub mod ui;

use bevy::prelude::*;

use crate::asteroid_app::events::ParamPatch;
use crate::asteroid_app::render::surface_mesh::{ensure_surface_view, sync_wireframe};
use crate::asteroid_app::render::vector_overlay::draw_vectors;
use crate::asteroid_app::render::{sync_surface_mesh, update_field_colors};
use crate::asteroid_app::resources::{OrbitCamera, StepControl, StepPhase, ViewerConfig};
use crate::asteroid_app::ui::camera::{orbit_camera, spawn_camera};
use crate::asteroid_app::ui::diagnostics_view::{refresh_diagnostics, spawn_diagnostics};
use crate::asteroid_app::ui::timeline::timeline_keys;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub struct PublishBody;

pub struct AsteroidPlugin {
    pub windowed: bool,
}

impl Plugin for AsteroidPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ViewerConfig>()
            .init_resource::<OrbitCamera>()
            .init_resource::<StepControl>()
            .init_resource::<StepPhase>()
            .add_message::<ParamPatch>()
            .configure_sets(Startup, PublishBody);
        if self.windowed {
            app.add_plugins(bevy::pbr::wireframe::WireframePlugin::default())
                .add_systems(
                    Startup,
                    (
                        spawn_camera,
                        spawn_diagnostics,
                        ensure_surface_view.after(PublishBody),
                    ),
                )
                .add_systems(
                    Update,
                    (
                        orbit_camera,
                        timeline_keys,
                        refresh_diagnostics,
                        sync_surface_mesh,
                        update_field_colors,
                        sync_wireframe,
                        draw_vectors,
                    ),
                );
        }
    }
}
