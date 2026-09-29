//! 表面网格实体构建与几何刷新
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues};

use super::field_renderer::compute_vertex_colors;
use super::resources::{AsteroidDisplayData, AsteroidMeshHandles, DisplaySettings};

/// 表面多面体实体标记 Component (SurfaceMeshEntity)
#[derive(Component, Debug, Default, Clone)]
pub struct SurfaceMeshEntity;

/// 从坐标点、面片拓扑与法向量构建 Bevy 3D 网格对象
pub fn build_bevy_mesh(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    normals: &[[f32; 3]],
    colors: Option<&[[f32; 4]]>,
) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    // 1. 顶点位置
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices.to_vec()),
    );

    // 2. 顶点法相
    if !normals.is_empty() {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals.to_vec()),
        );
    } else {
        mesh.duplicate_vertices();
        mesh.compute_flat_normals();
    }

    // 3. 顶点颜色 (用于伪彩色着色)
    if let Some(cols) = colors {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(cols.to_vec()),
        );
    } else {
        let default_cols = vec![[0.22, 0.20, 0.18, 1.0]; vertices.len()];
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(default_cols),
        );
    }

    // 4. 面片拓扑索引
    let mut indices_u32 = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        indices_u32.push(face[0]);
        indices_u32.push(face[1]);
        indices_u32.push(face[2]);
    }
    mesh.insert_indices(Indices::U32(indices_u32));

    mesh
}

/// Bevy ECS 启动系统: 创建小行星表面渲染网格与材质实体
pub fn spawn_asteroid_mesh_system(
    mut commands: Commands,
    data: Res<AsteroidDisplayData>,
    settings: Res<DisplaySettings>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    if data.vertices.is_empty() {
        return;
    }

    let (Some(mut meshes), Some(mut materials)) = (meshes, materials) else {
        return;
    };

    let (colors, _min_v, _max_v) = compute_vertex_colors(&data, &settings);
    let mesh = build_bevy_mesh(&data.vertices, &data.faces, &data.normals, Some(&colors));
    let mesh_handle = meshes.add(mesh);

    let material = StandardMaterial {
        base_color: Color::srgb(1.0, 1.0, 1.0),
        perceptual_roughness: 0.9,
        metallic: 0.05,
        reflectance: 0.04, // 典型 C 型小行星低反照率 (~0.045)
        double_sided: true,
        cull_mode: None,
        ..default()
    };
    let mat_handle = materials.add(material);

    commands.insert_resource(AsteroidMeshHandles {
        mesh: mesh_handle.clone(),
        material: mat_handle.clone(),
    });

    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(mat_handle),
        Transform::IDENTITY,
        SurfaceMeshEntity,
    ));
}

/// Bevy ECS 系统: 当几何形状或半径模态演化更新时，同步刷新 Bevy 顶点缓冲区
pub fn update_asteroid_mesh_geometry_system(
    data: Res<AsteroidDisplayData>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    handles: Option<Res<AsteroidMeshHandles>>,
) {
    if !data.is_changed() {
        return;
    }

    let (Some(mut meshes), Some(handles)) = (meshes, handles) else {
        return;
    };

    let Some(mut mesh) = meshes.get_mut(&handles.mesh) else {
        return;
    };

    if !data.vertices.is_empty() {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(data.vertices.clone()),
        );
    }

    if !data.normals.is_empty() {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(data.normals.clone()),
        );
    }
}
