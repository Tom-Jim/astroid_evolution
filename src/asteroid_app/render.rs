use bevy::asset::RenderAssetUsages;
use bevy::color::{LinearRgba, Mix, Srgba};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Neutral grey for no displacement, red for outward, blue for inward (`limit` saturates).
pub fn delta_color(delta: f32, limit: f32) -> [f32; 4] {
    let t = if delta.is_finite() && limit > 0.0 {
        (delta / limit).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let target = if t >= 0.0 {
        Srgba::new(0.86, 0.18, 0.12, 1.0)
    } else {
        Srgba::new(0.12, 0.25, 0.86, 1.0)
    };
    LinearRgba::from(Srgba::gray(0.88).mix(&target, t.abs())).to_f32_array()
}

pub fn colors(radial_delta: &[f32]) -> Vec<[f32; 4]> {
    let limit = radial_delta
        .iter()
        .filter(|d| d.is_finite())
        .fold(0.0_f32, |m, d| m.max(d.abs()));
    radial_delta
        .iter()
        .map(|d| delta_color(*d, limit))
        .collect()
}

pub fn build_mesh(positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>, faces: &[[u32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(faces.as_flattened().to_vec()));
    mesh.compute_smooth_normals();
    mesh
}

/// Replace positions and colours in place; normals are recomputed by Bevy.
pub fn update_mesh(mesh: &mut Mesh, positions: Vec<[f32; 3]>, colors: Vec<[f32; 4]>) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.compute_smooth_normals();
}
