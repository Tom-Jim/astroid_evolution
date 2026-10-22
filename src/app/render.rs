use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::app::snapshot::smooth_normals;

/// Triangle mesh. Positions are the pose currently on screen.
pub fn surface_mesh(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, faces: &[[u32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(faces.as_flattened().to_vec()));
    mesh
}

pub fn write_surface(mesh: &mut Mesh, positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
}

/// Startup mesh, before the first snapshot. Normals are accumulated once, here.
pub fn imported_mesh(positions: Vec<[f32; 3]>, faces: &[[u32; 3]]) -> Mesh {
    let normals = smooth_normals(&positions, faces);
    surface_mesh(positions, normals, faces)
}

/// `count` quads; `position.xy` is the corner in `[-1, 1]²` and `position.z` the quad
/// index. The vertex shader places each quad (a regolith particle on its streamline,
/// or a tension tick on its vertex), so these coordinates are not the drawn positions.
pub fn quad_mesh(count: usize) -> Mesh {
    let mut positions = Vec::with_capacity(count * 4);
    let mut indices = Vec::with_capacity(count * 6);
    let corners = [[-1.0f32, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    for particle in 0..count {
        let base = (particle * 4) as u32;
        let id = particle as f32;
        for corner in corners {
            positions.push([corner[0], corner[1], id]);
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}
