use crate::rtfp::mesh::Mesh;
use crate::rtfp::bvh::{build_flattened_bvh, FlattenedBvh};

pub fn update_radial_mesh(mesh: &Mesh, directions: &[[f64; 3]], radii: &[f64]) -> Result<Mesh, String> {
    if directions.len() != radii.len() || directions.len() != mesh.vertex_count() {
        return Err("径向网格更新的节点数量不一致".into());
    }
    let mut out = mesh.clone();
    let centroid = (0..mesh.vertex_count()).fold([0.0; 3], |mut c, i| {
        c[0] += mesh.xyz[3*i]; c[1] += mesh.xyz[3*i+1]; c[2] += mesh.xyz[3*i+2]; c
    });
    let inv_n = 1.0 / mesh.vertex_count().max(1) as f64;
    for (i, (u, r)) in directions.iter().zip(radii).enumerate() {
        if !r.is_finite() || *r <= 0.0 { return Err("半径必须为有限正数".into()); }
        out.xyz[3*i] = centroid[0]*inv_n + r*u[0]; out.xyz[3*i+1] = centroid[1]*inv_n + r*u[1]; out.xyz[3*i+2] = centroid[2]*inv_n + r*u[2];
    }
    Ok(out)
}

/// 几何更新后的原子操作：同步生成新网格与新的扁平 BVH，避免下一轮
/// RTFP 继续使用旧几何索引。
pub fn update_mesh_and_bvh(mesh: &Mesh, directions: &[[f64; 3]], radii: &[f64]) -> Result<(Mesh, FlattenedBvh), String> {
    let updated = update_radial_mesh(mesh, directions, radii)?;
    if updated.face_count() == 0 { return Err("更新后的网格没有三角面".into()); }
    Ok((updated.clone(), build_flattened_bvh(&updated)))
}
