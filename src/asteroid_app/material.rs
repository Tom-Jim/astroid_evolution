//! Custom material following Bevy 0.20 `examples/shader/shader_material.rs`
//! and `examples/shader/storage_buffer.rs`.
//!
//! The source lives next to this module and is registered with `embedded_asset!`,
//! the same path Bevy uses for `bevy_pbr`'s own `.wesl` files.

use bevy::asset::AssetPath;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

fn asteroid_shader() -> ShaderRef {
    ShaderRef::Path(
        AssetPath::from_path_buf(bevy::asset::embedded_path!("shaders/asteroid.wesl"))
            .with_source("embedded"),
    )
}

/// Per-frame playback. The large arrays live in the storage buffers.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct PlaybackParams {
    pub tau: f32,
    pub time: f32,
    pub fade: f32,
    pub mode: u32,
    pub f_delta_p95: f32,
    pub slack_p95: f32,
    pub rho_s_p95: f32,
    pub slack_tol: f32,
    pub vertex_count: u32,
    pub particle_count: u32,
    pub streamline_steps: u32,
    pub has_field: u32,
}

impl Default for PlaybackParams {
    fn default() -> Self {
        Self {
            tau: 0.0,
            time: 0.0,
            fade: 1.0,
            mode: 4,
            f_delta_p95: 1.0,
            slack_p95: 1.0,
            rho_s_p95: 1.0,
            slack_tol: 0.0,
            vertex_count: 0,
            particle_count: 0,
            streamline_steps: 0,
            has_field: 0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct AsteroidMaterial {
    #[uniform(0)]
    pub params: PlaybackParams,
    #[storage(1, read_only)]
    pub surface: Handle<ShaderBuffer>,
    #[storage(2, read_only)]
    pub stream: Handle<ShaderBuffer>,
}

impl Material for AsteroidMaterial {
    fn vertex_shader() -> ShaderRef {
        asteroid_shader()
    }

    fn fragment_shader() -> ShaderRef {
        asteroid_shader()
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}
