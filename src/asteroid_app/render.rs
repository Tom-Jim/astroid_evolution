use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

/// Triangle mesh for the Tier 2 body. Colour comes from the material, not from a displacement map.
pub fn build_mesh(positions: Vec<[f32; 3]>, faces: &[[u32; 3]]) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(faces.as_flattened().to_vec()));
    mesh.compute_smooth_normals();
    mesh
}

/// Replace positions in place. Normals are recomputed from the new shape.
pub fn update_mesh(mesh: &mut Mesh, positions: Vec<[f32; 3]>) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.compute_smooth_normals();
}
