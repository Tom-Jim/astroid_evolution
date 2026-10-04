use bevy::asset::RenderAssetUsages;
use bevy::pbr::wireframe::Wireframe;
use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_resource::PrimitiveTopology;

use crate::asteroid_app::components::{StarMesh, SurfaceFields, SurfaceView};
use crate::asteroid_app::render::field_overlay::{colors_for, colors_for_fields};
use crate::asteroid_app::resources::ViewerConfig;

pub fn vertex_positions(directions: &[[f64; 3]], radii: &[f64]) -> Vec<[f32; 3]> {
    directions
        .iter()
        .zip(radii)
        .map(|(direction, radius)| {
            [
                (direction[0] * radius) as f32,
                (direction[1] * radius) as f32,
                (direction[2] * radius) as f32,
            ]
        })
        .collect()
}

pub fn area_weighted_normals(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut accum = vec![[0.0f32; 3]; positions.len()];
    for face in faces {
        let a = positions[face[0] as usize];
        let b = positions[face[1] as usize];
        let c = positions[face[2] as usize];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for index in face {
            let slot = &mut accum[*index as usize];
            slot[0] += normal[0];
            slot[1] += normal[1];
            slot[2] += normal[2];
        }
    }
    for normal in &mut accum {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 1.0e-12 {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            *normal = [0.0, 1.0, 0.0];
        }
    }
    accum
}

pub fn build_surface_mesh(star: &StarMesh, colors: &[[f32; 4]]) -> Mesh {
    let positions = vertex_positions(&star.directions, &star.radii);
    let normals = area_weighted_normals(&positions, &star.faces);
    let mut indices = Vec::with_capacity(star.faces.len() * 3);
    for face in &star.faces {
        indices.extend_from_slice(face);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors.to_vec());
    mesh.insert_indices(Indices::U32(indices));
    if let Some((min, max)) = bounds(&positions) {
        mesh.final_aabb = Some(Aabb3d::from_min_max(min, max));
    }
    mesh
}

fn bounds(positions: &[[f32; 3]]) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for position in positions {
        let point = Vec3::from_array(*position);
        min = min.min(point);
        max = max.max(point);
    }
    (min.x <= max.x).then_some((min, max))
}

pub fn ensure_surface_view(
    mut commands: Commands,
    star: Query<&StarMesh>,
    fields: Query<&SurfaceFields>,
    existing: Query<(), With<SurfaceView>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<ViewerConfig>,
) {
    if !existing.is_empty() {
        return;
    }
    let Ok(star) = star.single() else {
        return;
    };
    let colors = if let Ok(fields) = fields.single() {
        colors_for_fields(fields, config.active_scalar, config.colormap)
    } else {
        colors_for(star, config.active_scalar, config.colormap)
    };
    commands.spawn((
        SurfaceView,
        Mesh3d(meshes.add(build_surface_mesh(star, &colors))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::default(),
    ));
}

pub fn sync_wireframe(
    config: Res<ViewerConfig>,
    mut commands: Commands,
    views: Query<(Entity, Option<&Wireframe>), With<SurfaceView>>,
) {
    if !config.is_changed() {
        return;
    }
    for (entity, wireframe) in &views {
        if config.wireframe && wireframe.is_none() {
            commands.entity(entity).insert(Wireframe);
        }
        if !config.wireframe && wireframe.is_some() {
            commands.entity(entity).remove::<Wireframe>();
        }
    }
}
pub fn sync_surface_mesh(
    star: Query<(&StarMesh, &SurfaceFields)>,
    view: Query<&Mesh3d, With<SurfaceView>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut applied: Local<u64>,
    config: Res<ViewerConfig>,
) {
    let Ok((star, fields)) = star.single() else {
        return;
    };
    if star.revision == *applied {
        return;
    }
    let Ok(mesh_handle) = view.single() else {
        return;
    };
    let Some(mut mesh) = meshes.get_mut(&mesh_handle.0) else {
        return;
    };
    let colors = colors_for_fields(fields, config.active_scalar, config.colormap);
    let positions = vertex_positions(&star.directions, &star.radii);
    let normals = area_weighted_normals(&positions, &star.faces);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    if let Some((min, max)) = bounds(&positions) {
        mesh.final_aabb = Some(Aabb3d::from_min_max(min, max));
    }
    *applied = star.revision;
}
