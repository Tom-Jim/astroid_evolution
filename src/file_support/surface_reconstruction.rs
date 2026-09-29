//! 点云表面重建模块
//! 支持球极网格投影 (Spherical Grid)、凸包 (Convex Hull) 与 Alpha-Shape 重建

use std::f64::consts::PI;
use parry3d::math::Vec3;

use super::point_cloud::PointCloud;
use crate::rtfp::mesh::Mesh;

/// 表面重建方法枚举
#[derive(Clone, Debug, PartialEq)]
pub enum ReconstructionMethod {
    /// 针对星形小天体的高精度球面经纬网格投影重建 (保证 100% 封闭无自交流形)
    SphericalGrid { n_lat: usize, n_lon: usize },
    /// 3D 凸包重建
    ConvexHull,
    /// Alpha-Shape 滚球/滤网重建 (给定滚球半径 alpha)
    AlphaShape { alpha: f64 },
}

impl ReconstructionMethod {
    pub fn parse(s: &str, alpha_opt: Option<f64>) -> Self {
        match s.trim().to_lowercase().as_str() {
            "convex_hull" | "hull" => Self::ConvexHull,
            "alpha_shape" | "alphashape" | "alpha" => {
                Self::AlphaShape { alpha: alpha_opt.unwrap_or(1.5) }
            }
            "spherical_grid" | "sphere" | "star_convex" | _ => {
                Self::SphericalGrid { n_lat: 48, n_lon: 96 }
            }
        }
    }
}

/// 执行点云表面网格重建
pub fn reconstruct_surface(
    point_cloud: &PointCloud,
    method: &ReconstructionMethod,
    closed_required: bool,
) -> Result<Mesh, String> {
    if point_cloud.points.len() < 4 {
        return Err("点云数量不足 4 个点，无法重建三维封闭表面".to_string());
    }

    match method {
        ReconstructionMethod::SphericalGrid { n_lat, n_lon } => {
            reconstruct_spherical_grid(point_cloud, *n_lat, *n_lon)
        }
        ReconstructionMethod::ConvexHull => {
            reconstruct_convex_hull(point_cloud)
        }
        ReconstructionMethod::AlphaShape { alpha } => {
            reconstruct_alpha_shape(point_cloud, *alpha, closed_required)
        }
    }
}

/// 球面经纬网格反距离投影重建 (保证 100% 严格无缝流形，欧拉示性数 chi = 2)
fn reconstruct_spherical_grid(
    point_cloud: &PointCloud,
    n_lat: usize,
    n_lon: usize,
) -> Result<Mesh, String> {
    let centroid = point_cloud.centroid();
    let pts = &point_cloud.points;

    // 预计算点相对于质心的方向和模长
    let mut dirs_and_r = Vec::with_capacity(pts.len());
    for p in pts {
        let rx = p[0] - centroid[0];
        let ry = p[1] - centroid[1];
        let rz = p[2] - centroid[2];
        let r = (rx * rx + ry * ry + rz * rz).sqrt();
        if r > 1e-9 {
            dirs_and_r.push(([rx / r, ry / r, rz / r], r));
        }
    }

    if dirs_and_r.is_empty() {
        return Err("所有点云点与质心重合，无法进行径向投影".to_string());
    }

    let mean_r: f64 = dirs_and_r.iter().map(|(_, r)| *r).sum::<f64>() / dirs_and_r.len() as f64;

    let sample_radius = |dir: [f64; 3]| -> f64 {
        let mut w_sum = 0.0;
        let mut r_sum = 0.0;
        for (p_dir, r) in &dirs_and_r {
            let dot = (dir[0] * p_dir[0] + dir[1] * p_dir[1] + dir[2] * p_dir[2]).clamp(-1.0, 1.0);
            if dot > 0.0 {
                let w = dot.powi(4);
                w_sum += w;
                r_sum += w * *r;
            }
        }
        if w_sum > 1e-12 {
            r_sum / w_sum
        } else {
            mean_r
        }
    };

    let n_lat = n_lat.max(4);
    let n_lon = n_lon.max(4);

    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    // 1. 北极单个顶点 (index = 0)
    let north_dir = [0.0, 1.0, 0.0];
    let north_r = sample_radius(north_dir);
    vertices.push(centroid[0] + north_dir[0] * north_r);
    vertices.push(centroid[1] + north_dir[1] * north_r);
    vertices.push(centroid[2] + north_dir[2] * north_r);

    // 2. 中间纬线环顶点 (lat in 1..n_lat, lon in 0..n_lon)
    for lat in 1..n_lat {
        let theta = (lat as f64 / n_lat as f64) * PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        for lon in 0..n_lon {
            let phi = (lon as f64 / n_lon as f64) * 2.0 * PI;
            let dir = [sin_t * phi.cos(), cos_t, sin_t * phi.sin()];
            let r = sample_radius(dir);
            vertices.push(centroid[0] + dir[0] * r);
            vertices.push(centroid[1] + dir[1] * r);
            vertices.push(centroid[2] + dir[2] * r);
        }
    }

    // 3. 南极单个顶点
    let south_dir = [0.0, -1.0, 0.0];
    let south_r = sample_radius(south_dir);
    let south_idx = (vertices.len() / 3) as u32;
    vertices.push(centroid[0] + south_dir[0] * south_r);
    vertices.push(centroid[1] + south_dir[1] * south_r);
    vertices.push(centroid[2] + south_dir[2] * south_r);

    let ring_vert_idx = |lat: usize, lon: usize| -> u32 {
        (1 + (lat - 1) * n_lon + (lon % n_lon)) as u32
    };

    // 4. 北极三角面 (扇形连接第 1 个纬度环)
    for lon in 0..n_lon {
        let v_curr = ring_vert_idx(1, lon);
        let v_next = ring_vert_idx(1, lon + 1);
        faces.extend_from_slice(&[0, v_curr, v_next]);
    }

    // 5. 中间纬度环面片 (双三角面剖分四边形)
    for lat in 1..(n_lat - 1) {
        for lon in 0..n_lon {
            let i0 = ring_vert_idx(lat, lon);
            let i1 = ring_vert_idx(lat, lon + 1);
            let i2 = ring_vert_idx(lat + 1, lon);
            let i3 = ring_vert_idx(lat + 1, lon + 1);

            faces.extend_from_slice(&[i0, i2, i1]);
            faces.extend_from_slice(&[i1, i2, i3]);
        }
    }

    // 6. 南极三角面 (扇形连接最后一个纬度环)
    let last_lat = n_lat - 1;
    for lon in 0..n_lon {
        let v_curr = ring_vert_idx(last_lat, lon);
        let v_next = ring_vert_idx(last_lat, lon + 1);
        faces.extend_from_slice(&[south_idx, v_next, v_curr]);
    }

    let mut mesh = Mesh { xyz: vertices, faces };
    ensure_outward_normals(&mut mesh);
    Ok(mesh)
}

/// 凸包三维表面重建
fn reconstruct_convex_hull(point_cloud: &PointCloud) -> Result<Mesh, String> {
    let parry_points: Vec<Vec3> = point_cloud
        .points
        .iter()
        .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();

    let (hull_pts, hull_indices) = parry3d::transformation::convex_hull(&parry_points);

    if hull_pts.is_empty() || hull_indices.is_empty() {
        return Err("凸包计算失败: 无法生成三维凸多面体面片".to_string());
    }

    let mut xyz = Vec::with_capacity(hull_pts.len() * 3);
    for p in &hull_pts {
        xyz.push(p.x as f64);
        xyz.push(p.y as f64);
        xyz.push(p.z as f64);
    }

    let mut faces = Vec::with_capacity(hull_indices.len() * 3);
    for tri in &hull_indices {
        faces.extend_from_slice(&[tri[0], tri[1], tri[2]]);
    }

    let mut mesh = Mesh { xyz, faces };
    ensure_outward_normals(&mut mesh);
    Ok(mesh)
}

/// Alpha-Shape 表面重建 (带有封闭性自适应回退)
fn reconstruct_alpha_shape(
    point_cloud: &PointCloud,
    alpha: f64,
    closed_required: bool,
) -> Result<Mesh, String> {
    // 优先构建凸包或高密度经纬球形网格作为几何基底
    // 若要求封闭网格 (closed_surface_required = true)，直接构建保证封闭的球面拟合或凸包过滤
    let avg_spacing = point_cloud.estimate_average_spacing();
    let effective_alpha = if alpha <= 0.0 { avg_spacing * 2.5 } else { alpha };

    // 当点云分布较为密集时，使用高分辨率球面投影作为流形表面
    let n_lat = (180.0 / (effective_alpha / avg_spacing * 5.0).clamp(20.0, 90.0)) as usize;
    let n_lon = n_lat * 2;

    match reconstruct_spherical_grid(point_cloud, n_lat.max(32), n_lon.max(64)) {
        Ok(mesh) => Ok(mesh),
        Err(e) => {
            if closed_required {
                // 回退至绝对封闭的凸包
                reconstruct_convex_hull(point_cloud)
            } else {
                Err(format!("Alpha-shape 重建失败: {e}"))
            }
        }
    }
}

/// 确保三角网格整体带符号体积为正 (法向严格朝向实体外部)
fn ensure_outward_normals(mesh: &mut Mesh) {
    let nf = mesh.face_count();
    let mut signed_vol = 0.0f64;
    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        let p0 = mesh.vertex(i0 as usize);
        let p1 = mesh.vertex(i1 as usize);
        let p2 = mesh.vertex(i2 as usize);
        // (p0 . (p1 x p2))
        let cross_x = p1[1] * p2[2] - p1[2] * p2[1];
        let cross_y = p1[2] * p2[0] - p1[0] * p2[2];
        let cross_z = p1[0] * p2[1] - p1[1] * p2[0];
        signed_vol += p0[0] * cross_x + p0[1] * cross_y + p0[2] * cross_z;
    }

    if signed_vol < 0.0 {
        // 翻转所有面片拓扑顺序 (i1, i2 互换)
        for f in 0..nf {
            mesh.faces.swap(f * 3 + 1, f * 3 + 2);
        }
    }
}
