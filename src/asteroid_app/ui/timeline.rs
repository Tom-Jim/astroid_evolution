use bevy::prelude::*;

use crate::asteroid_app::resources::{
    Colormap, OrbitCamera, ScalarField, StepControl, VectorField, ViewerConfig,
};

pub fn timeline_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut control: ResMut<StepControl>,
    mut viewer: ResMut<ViewerConfig>,
    mut camera: ResMut<OrbitCamera>,
) {
    if keys.just_pressed(KeyCode::Space) {
        *control = if *control == StepControl::Run {
            StepControl::Pause
        } else {
            StepControl::Run
        };
    }
    if keys.just_pressed(KeyCode::KeyN) {
        *control = StepControl::StepOnce;
    }
    if keys.just_pressed(KeyCode::KeyV) {
        viewer.active_vector = match viewer.active_vector {
            VectorField::None => VectorField::SlideVelocity,
            VectorField::SlideVelocity => VectorField::Gravity,
            VectorField::Gravity => VectorField::SurfaceNormal,
            VectorField::SurfaceNormal => VectorField::None,
        };
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *camera = OrbitCamera::default();
    }
    let scalar = if keys.just_pressed(KeyCode::Digit1) {
        Some(ScalarField::YieldCriterion)
    } else if keys.just_pressed(KeyCode::Digit2) {
        Some(ScalarField::Viscosity)
    } else if keys.just_pressed(KeyCode::Digit3) {
        Some(ScalarField::Pressure)
    } else if keys.just_pressed(KeyCode::Digit4) {
        Some(ScalarField::FrictionAngle)
    } else if keys.just_pressed(KeyCode::Digit5) {
        Some(ScalarField::ShearRate)
    } else if keys.just_pressed(KeyCode::Digit6) {
        Some(ScalarField::SurfaceDensity)
    } else if keys.just_pressed(KeyCode::Digit7) {
        Some(ScalarField::SlideSpeed)
    } else if keys.just_pressed(KeyCode::Digit8) {
        Some(ScalarField::None)
    } else {
        None
    };
    if let Some(scalar) = scalar {
        viewer.active_scalar = scalar;
        viewer.colormap = if scalar == ScalarField::YieldCriterion {
            Colormap::CoolWarm
        } else {
            Colormap::Turbo
        };
    }
}
