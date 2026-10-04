use bevy::input::mouse::{AccumulatedMouseScroll, MouseMotion};
use bevy::prelude::*;

use crate::asteroid_app::resources::OrbitCamera;

pub fn spawn_camera(mut commands: Commands) {
    let camera = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(eye(&camera)).looking_at(camera.target, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));
}

pub fn orbit_camera(
    mut camera: ResMut<OrbitCamera>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: MessageReader<MouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut transforms: Query<&mut Transform, With<Camera3d>>,
) {
    let mut delta = Vec2::ZERO;
    for event in motion.read() {
        delta += event.delta;
    }
    if buttons.pressed(MouseButton::Left) {
        camera.yaw -= delta.x * 0.005;
        camera.pitch = (camera.pitch + delta.y * 0.005).clamp(-1.2, 1.2);
    }
    if buttons.pressed(MouseButton::Middle) {
        let right = Vec3::new(camera.yaw.cos(), 0.0, -camera.yaw.sin());
        let distance = camera.distance;
        camera.target += right * delta.x * 0.002 * distance;
        camera.target.y += delta.y * 0.002 * distance;
    }
    if scroll.delta.y != 0.0 {
        camera.distance = (camera.distance * (1.0 - scroll.delta.y * 0.08)).clamp(1.2, 20.0);
    }
    if let Ok(mut transform) = transforms.single_mut() {
        transform.translation = eye(&camera);
        transform.look_at(camera.target, Vec3::Y);
    }
}

fn eye(camera: &OrbitCamera) -> Vec3 {
    camera.target
        + Vec3::new(
            camera.distance * camera.pitch.cos() * camera.yaw.sin(),
            camera.distance * camera.pitch.sin(),
            camera.distance * camera.pitch.cos() * camera.yaw.cos(),
        )
}
