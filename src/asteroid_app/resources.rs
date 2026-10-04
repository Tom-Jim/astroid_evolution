use bevy::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScalarField {
    #[default]
    None,
    YieldCriterion,
    Pressure,
    Viscosity,
    ShearRate,
    FrictionAngle,
    SurfaceDensity,
    SlideSpeed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VectorField {
    #[default]
    None,
    SlideVelocity,
    Gravity,
    SurfaceNormal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Colormap {
    #[default]
    Viridis,
    Plasma,
    CoolWarm,
    Turbo,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct ViewerConfig {
    pub active_scalar: ScalarField,
    pub active_vector: VectorField,
    pub colormap: Colormap,
    pub wireframe: bool,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            active_scalar: ScalarField::YieldCriterion,
            active_vector: VectorField::None,
            colormap: Colormap::CoolWarm,
            wireframe: false,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.4,
            distance: 3.4,
            target: Vec3::ZERO,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepControl {
    Run,
    #[default]
    Pause,
    StepOnce,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepPhase {
    #[default]
    Idle,
    Stepping,
}
