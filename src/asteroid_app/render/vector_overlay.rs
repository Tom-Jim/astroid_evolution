use bevy::prelude::*;

use crate::asteroid_app::components::{StarMesh, SurfaceFields};
use crate::asteroid_app::resources::{VectorField, ViewerConfig};

pub fn draw_vectors(
    config: Res<ViewerConfig>,
    bodies: Query<(&StarMesh, &SurfaceFields)>,
    mut gizmos: Gizmos,
) {
    if config.active_vector == VectorField::None {
        return;
    }
    let Ok((star, fields)) = bodies.single() else {
        return;
    };
    let samples: Vec<usize> = (0..star.radii.len())
        .step_by(4.max(star.radii.len() / 48))
        .collect();
    let mut longest = 1.0e-8_f32;
    let vectors: Vec<[f64; 3]> = samples
        .iter()
        .map(|&index| match config.active_vector {
            VectorField::None => [0.0; 3],
            VectorField::SlideVelocity => fields.slide_velocity[index],
            VectorField::Gravity => fields.gravity[index],
            VectorField::SurfaceNormal => fields.normals[index],
        })
        .collect();
    for vector in &vectors {
        let length =
            ((vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]) as f32).sqrt();
        longest = longest.max(length);
    }
    let color = match config.active_vector {
        VectorField::SlideVelocity => Color::srgb(0.95, 0.78, 0.15),
        VectorField::Gravity => Color::srgb(0.35, 0.75, 0.95),
        VectorField::SurfaceNormal => Color::srgb(0.85, 0.9, 0.95),
        VectorField::None => Color::WHITE,
    };
    for (sample, vector) in samples.iter().zip(vectors) {
        if config.active_vector == VectorField::SlideVelocity && !fields.mobile[*sample] {
            continue;
        }
        let origin = Vec3::new(
            (star.directions[*sample][0] * star.radii[*sample]) as f32,
            (star.directions[*sample][1] * star.radii[*sample]) as f32,
            (star.directions[*sample][2] * star.radii[*sample]) as f32,
        );
        let direction = Vec3::new(vector[0] as f32, vector[1] as f32, vector[2] as f32) / longest;
        let end = origin + direction * 0.35;
        if origin.distance(end) > 1.0e-4 {
            gizmos.arrow(origin, end, color);
        }
    }
}
