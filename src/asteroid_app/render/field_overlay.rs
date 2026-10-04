use bevy::prelude::*;

use crate::asteroid_app::components::{StarMesh, SurfaceFields};
use crate::asteroid_app::resources::{Colormap, ScalarField, ViewerConfig};

pub fn coolwarm(value: f64) -> [f32; 4] {
    let signed = if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let magnitude = signed.abs() as f32;
    let neutral = 0.9;
    let hot = [
        neutral + (0.86 - neutral) * magnitude,
        neutral * (1.0 - 0.75 * magnitude),
        neutral * (1.0 - magnitude),
    ];
    if signed >= 0.0 {
        [hot[0], hot[1], hot[2], 1.0]
    } else {
        [hot[2], hot[1], hot[0], 1.0]
    }
}

pub fn viridis(normalized: f64) -> [f32; 4] {
    let x = unit(normalized);
    [
        0.267 + x * (0.993 - 0.267) * (0.35 + 0.65 * x),
        0.005 + x * 0.85,
        0.329 + (1.0 - x) * 0.35 - x * x * 0.2,
        1.0,
    ]
}

pub fn plasma(normalized: f64) -> [f32; 4] {
    let x = unit(normalized);
    [
        0.05 + 0.9 * x,
        0.03 + 0.55 * x * (1.0 - 0.4 * x),
        0.53 + 0.4 * (1.0 - x),
        1.0,
    ]
}

pub fn turbo(normalized: f64) -> [f32; 4] {
    let x = unit(normalized);
    let r = (0.135 + x * (4.0 * (1.0 - x) + x)).clamp(0.0, 1.0);
    let g = (x * (1.4 - x)).clamp(0.0, 1.0);
    let b = (1.0 - x * x).clamp(0.0, 1.0);
    [r, g, b, 1.0]
}

fn unit(value: f64) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

pub fn map_scalar(value: f64, lower: f64, upper: f64, colormap: Colormap) -> [f32; 4] {
    match colormap {
        Colormap::CoolWarm => coolwarm(value),
        Colormap::Viridis => viridis(normalize(value, lower, upper)),
        Colormap::Plasma => plasma(normalize(value, lower, upper)),
        Colormap::Turbo => turbo(normalize(value, lower, upper)),
    }
}

fn normalize(value: f64, lower: f64, upper: f64) -> f64 {
    let span = upper - lower;
    if !span.is_finite() || span.abs() < 1.0e-30 {
        0.5
    } else {
        (value - lower) / span
    }
}

pub fn colors_for(star: &StarMesh, scalar: ScalarField, colormap: Colormap) -> Vec<[f32; 4]> {
    let values = vec![0.0; star.radii.len()];
    paint(&values, scalar, colormap)
}

pub fn colors_for_fields(
    fields: &SurfaceFields,
    scalar: ScalarField,
    colormap: Colormap,
) -> Vec<[f32; 4]> {
    let values = match scalar {
        ScalarField::None => vec![0.0; fields.yield_criterion.len()],
        ScalarField::YieldCriterion => fields.yield_criterion.clone(),
        ScalarField::Pressure => fields.pressure.clone(),
        ScalarField::Viscosity => fields.viscosity.clone(),
        ScalarField::ShearRate => fields.shear_rate.clone(),
        ScalarField::FrictionAngle => fields.friction_angle.clone(),
        ScalarField::SurfaceDensity => fields.surface_density.clone(),
        ScalarField::SlideSpeed => fields.slide_speed.clone(),
    };
    paint(&values, scalar, colormap)
}

fn paint(values: &[f64], scalar: ScalarField, colormap: Colormap) -> Vec<[f32; 4]> {
    let colormap = if scalar == ScalarField::YieldCriterion {
        Colormap::CoolWarm
    } else {
        colormap
    };
    let lower = values.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    values
        .iter()
        .map(|value| map_scalar(*value, lower, upper, colormap))
        .collect()
}

pub fn update_field_colors(
    config: Res<ViewerConfig>,
    bodies: Query<(&StarMesh, &SurfaceFields)>,
    view: Query<&Mesh3d, With<crate::asteroid_app::components::SurfaceView>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok((star, fields)) = bodies.single() else {
        return;
    };
    if !config.is_changed() && star.revision == 0 {
        return;
    }
    let Ok(handle) = view.single() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&handle.0) else {
        return;
    };
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        colors_for_fields(fields, config.active_scalar, config.colormap),
    );
}
