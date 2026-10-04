use bevy::prelude::*;

/// Star-shaped surface. Vertex positions are `x_i = R_i ω̂_i`.
#[derive(Component, Clone, Debug)]
pub struct StarMesh {
    pub directions: Vec<[f64; 3]>,
    pub radii: Vec<f64>,
    pub faces: Vec<[u32; 3]>,
    pub revision: u64,
}

#[derive(Component, Clone, Debug)]
pub struct SurfaceFields {
    pub yield_criterion: Vec<f64>,
    pub pressure: Vec<f64>,
    pub viscosity: Vec<f64>,
    pub shear_rate: Vec<f64>,
    pub friction_angle: Vec<f64>,
    pub surface_density: Vec<f64>,
    pub slide_speed: Vec<f64>,
    pub slide_velocity: Vec<[f64; 3]>,
    pub gravity: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    pub mobile: Vec<bool>,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct StepRecord {
    pub step: u64,
    pub time: f64,
    pub omega: [f64; 3],
    pub yielded: bool,
    pub mobile_fraction: f64,
    pub mass: f64,
    pub mass_error: f64,
}

#[derive(Component)]
pub struct SurfaceView;
