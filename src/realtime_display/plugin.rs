//! Bevy 插件组织与系统编排 (AsteroidDisplayPlugin)
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;

use super::debug_renderer::debug_render_gizmos_system;
use super::events::{ChangeDisplayModeEvent, DisplayMode};
use super::field_renderer::update_field_colors_system;
use super::mesh_renderer::{spawn_asteroid_mesh_system, update_asteroid_mesh_geometry_system};
use super::resources::{AsteroidDisplayData, ColormapType, DisplaySettings};

/// 小行星演化与引力可视化 Bevy Plugin
pub struct AsteroidDisplayPlugin;

impl Plugin for AsteroidDisplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DisplaySettings>()
            .init_resource::<AsteroidDisplayData>()
            .add_systems(
                Startup,
                (setup_display_scene_system, spawn_asteroid_mesh_system).chain(),
            )
            .add_systems(
                Update,
                (
                    handle_display_keyboard_system,
                    update_field_colors_system,
                    update_asteroid_mesh_geometry_system,
                    debug_render_gizmos_system,
                ),
            );
    }
}

/// 场景初始化系统: 建立 3D 观察相机、太阳定向光源与环境天光
pub fn setup_display_scene_system(mut commands: Commands) {
    // 1. 3D 透视相机 (斜向俯视小行星，距离 1600 米) 并附带环境光组件
    commands.spawn((
        Camera3d::default(),
        AmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            ..default()
        },
        Transform::from_xyz(0.0, 500.0, 1600.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // 2. 模拟太阳平行主光源
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.5, 0.0)),
    ));
}

/// 键盘交互控制系统: 切换可视化物理量、调试法向、射线与色表
pub fn handle_display_keyboard_system(
    mut commands: Commands,
    keyboard: Option<Res<ButtonInput<KeyCode>>>,
    mut settings: ResMut<DisplaySettings>,
) {
    let Some(keyboard) = keyboard else {
        return;
    };
    let mut changed = false;

    if keyboard.just_pressed(KeyCode::Digit1) {
        settings.mode = DisplayMode::SurfaceRealistic;
        changed = true;
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        settings.mode = DisplayMode::PotentialU;
        changed = true;
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        settings.mode = DisplayMode::GravityMagnitude;
        changed = true;
    } else if keyboard.just_pressed(KeyCode::Digit4) {
        settings.mode = DisplayMode::HessianTrace;
        changed = true;
    } else if keyboard.just_pressed(KeyCode::Digit5) {
        settings.mode = DisplayMode::SlopeViolation;
        changed = true;
    } else if keyboard.just_pressed(KeyCode::Digit6) {
        settings.mode = DisplayMode::MassResidual;
        changed = true;
    }

    if changed {
        commands.trigger(ChangeDisplayModeEvent {
            mode: settings.mode,
        });
        println!("--> 切换显示模式: {:?}", settings.mode);
    }

    // 切换法向箭头
    if keyboard.just_pressed(KeyCode::KeyN) {
        settings.show_normals = !settings.show_normals;
        println!("--> 切换法向调试显示: {}", settings.show_normals);
    }

    // 切换引力射线
    if keyboard.just_pressed(KeyCode::KeyR) {
        settings.show_rays = !settings.show_rays;
        println!("--> 切换射线调试显示: {}", settings.show_rays);
    }

    // 轮换伪彩色表 (Turbo -> Viridis -> Coolwarm -> Grayscale)
    if keyboard.just_pressed(KeyCode::KeyC) {
        settings.colormap = match settings.colormap {
            ColormapType::Turbo => ColormapType::Viridis,
            ColormapType::Viridis => ColormapType::Coolwarm,
            ColormapType::Coolwarm => ColormapType::Grayscale,
            ColormapType::Grayscale | ColormapType::Inferno => ColormapType::Turbo,
        };
        println!("--> 切换色表: {:?}", settings.colormap);
    }
}
