//! 表面网格定义、GLB 加载、点云构建与 1mm 观测面生成
//! 参照 ryugu_cauchy_gravity/src/rust/compute/native/mesh.rs

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// 顶点坐标 xyz (米)
    pub xyz: Vec<f64>,
    /// 三角形面索引 (基于 0)
    pub faces: Vec<u32>,
}

impl Mesh {
    /// 从 GLB 文件加载三角网格并按 scale 缩放至 SI 单位 (米)
    pub fn load_glb(path: &Path, scale: f64) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let model = gltf::Gltf::from_slice(&bytes)
            .map_err(|e| format!("{}: invalid GLB: {e}", path.display()))?;
        let blob = model
            .blob
            .as_deref()
            .ok_or_else(|| format!("{}: GLB has no binary buffer", path.display()))?;
        let primitive = model
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .filter(|p| p.mode() == gltf::mesh::Mode::Triangles)
            .max_by_key(|p| p.indices().map_or(0, |indices| indices.count()))
            .ok_or_else(|| format!("{}: no indexed triangle primitive", path.display()))?;
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => Some(blob),
            gltf::buffer::Source::Uri(_) => None,
        });
        let positions = reader
            .read_positions()
            .ok_or_else(|| format!("{}: mesh has no POSITION attribute", path.display()))?;
        let mut mesh = Mesh::default();
        for point in positions {
            mesh.xyz.extend_from_slice(&[
                point[0] as f64 * scale,
                point[1] as f64 * scale,
                point[2] as f64 * scale,
            ]);
        }
        mesh.faces = reader
            .read_indices()
            .ok_or_else(|| format!("{}: mesh has no index accessor", path.display()))?
            .into_u32()
            .collect();
        if mesh.faces.len() % 3 != 0 {
            return Err(format!("{}: triangle index count is invalid", path.display()));
        }

        // 统一法向朝外 (检查带符号体积)
        let mut signed_six_volume = 0.0f64;
        for face in 0..mesh.face_count() {
            let [i0, i1, i2] = mesh.face(face);
            signed_six_volume += dot(
                mesh.vertex(i0 as usize),
                cross(mesh.vertex(i1 as usize), mesh.vertex(i2 as usize)),
            );
        }
        if signed_six_volume < 0.0 {
            for face_idx in 0..mesh.face_count() {
                mesh.faces.swap(face_idx * 3 + 1, face_idx * 3 + 2);
            }
        }
        Ok(mesh)
    }

    /// 从表面点云程序化构建闭合表面三角网格
    pub fn from_point_cloud(points: &[[f64; 3]], n_lat: usize, n_lon: usize) -> Result<Self, String> {
        if points.is_empty() {
            return Err("point cloud is empty".into());
        }

        let mut vertices = Vec::with_capacity((n_lat + 1) * (n_lon + 1) * 3);
        let mut faces = Vec::with_capacity(n_lat * n_lon * 6);

        use std::f64::consts::PI;
        for lat in 0..=n_lat {
            let theta = (lat as f64 / n_lat as f64) * PI;
            let sin_t = theta.sin();
            let cos_t = theta.cos();

            for lon in 0..=n_lon {
                let phi = (lon as f64 / n_lon as f64) * 2.0 * PI;
                let dir = [sin_t * phi.cos(), cos_t, sin_t * phi.sin()];

                // 反距离加权寻找点云中的径向拟合值
                let mut w_sum = 0.0;
                let mut r_sum = 0.0;
                for p in points {
                    let r = norm(*p);
                    if r < 1e-6 { continue; }
                    let p_dir = [p[0] / r, p[1] / r, p[2] / r];
                    let cos_angle = dot(dir, p_dir).clamp(-1.0, 1.0);
                    if cos_angle > 0.75 {
                        let w = (cos_angle - 0.75).powi(4);
                        w_sum += w;
                        r_sum += w * r;
                    }
                }

                let radius = if w_sum > 1e-6 { r_sum / w_sum } else { 500.0 };
                vertices.extend_from_slice(&[dir[0] * radius, dir[1] * radius, dir[2] * radius]);
            }
        }

        for lat in 0..n_lat {
            for lon in 0..n_lon {
                let i0 = (lat * (n_lon + 1) + lon) as u32;
                let i1 = (lat * (n_lon + 1) + (lon + 1)) as u32;
                let i2 = ((lat + 1) * (n_lon + 1) + lon) as u32;
                let i3 = ((lat + 1) * (n_lon + 1) + (lon + 1)) as u32;

                faces.extend_from_slice(&[i0, i2, i1]);
                faces.extend_from_slice(&[i1, i2, i3]);
            }
        }

        Ok(Mesh { xyz: vertices, faces })
    }

    pub fn vertex_count(&self) -> usize {
        self.xyz.len() / 3
    }

    pub fn face_count(&self) -> usize {
        self.faces.len() / 3
    }

    pub fn vertex(&self, i: usize) -> [f64; 3] {
        [self.xyz[3 * i], self.xyz[3 * i + 1], self.xyz[3 * i + 2]]
    }

    pub fn face(&self, f: usize) -> [u32; 3] {
        [self.faces[3 * f], self.faces[3 * f + 1], self.faces[3 * f + 2]]
    }

    /// 面积加权顶点法向量 (严格朝向外侧)
    pub fn vertex_normals(&self) -> Vec<[f64; 3]> {
        let mut normals = vec![[0.0f64; 3]; self.vertex_count()];
        for f in 0..self.face_count() {
            let [i0, i1, i2] = self.face(f);
            let (p0, p1, p2) = (self.vertex(i0 as usize), self.vertex(i1 as usize), self.vertex(i2 as usize));
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let n = cross(e1, e2);
            for idx in [i0 as usize, i1 as usize, i2 as usize] {
                normals[idx][0] += n[0];
                normals[idx][1] += n[1];
                normals[idx][2] += n[2];
            }
        }
        for (i, n) in normals.iter_mut().enumerate() {
            let len = norm(*n);
            if len > 1e-30 {
                *n = [n[0] / len, n[1] / len, n[2] / len];
            } else {
                let p = self.vertex(i);
                let r = norm(p);
                *n = if r > 1e-30 { [p[0] / r, p[1] / r, p[2] / r] } else { [0.0, 1.0, 0.0] };
            }
            let p = self.vertex(i);
            if dot(*n, p) < 0.0 {
                *n = [-n[0], -n[1], -n[2]];
            }
        }
        normals
    }

    /// 生成严格位于模型表面外侧指定高度 (例如 1mm = 0.001 m) 的观测点集合
    pub fn observation_points(&self, standoff_m: f64) -> Vec<[f64; 3]> {
        let normals = self.vertex_normals();
        (0..self.vertex_count())
            .map(|i| {
                let p = self.vertex(i);
                let n = normals[i];
                [
                    p[0] + standoff_m * n[0],
                    p[1] + standoff_m * n[1],
                    p[2] + standoff_m * n[2],
                ]
            })
            .collect()
    }

    /// 计算多面体包围体积与质心
    pub fn volume_and_centroid(&self) -> (f64, [f64; 3]) {
        let mut total_vol = 0.0f64;
        let mut c_weighted = [0.0f64; 3];

        for f in 0..self.face_count() {
            let [i0, i1, i2] = self.face(f);
            let p0 = self.vertex(i0 as usize);
            let p1 = self.vertex(i1 as usize);
            let p2 = self.vertex(i2 as usize);

            let tet_vol = dot(p0, cross(p1, p2)) / 6.0;
            total_vol += tet_vol;

            for k in 0..3 {
                c_weighted[k] += tet_vol * 0.25 * (p0[k] + p1[k] + p2[k]);
            }
        }

        let centroid = if total_vol.abs() > 1e-12 {
            [
                c_weighted[0] / total_vol,
                c_weighted[1] / total_vol,
                c_weighted[2] / total_vol,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };

        (total_vol.abs(), centroid)
    }

    /// 转换为跨模块数据契约结构 GeometryState (docs/software.md 第 6 节)
    pub fn to_geometry_state(&self) -> crate::model_calculation::contracts::GeometryState {
        let (volume, centroid) = self.volume_and_centroid();
        let normals_f64 = self.vertex_normals();
        let nv = self.vertex_count();

        let mut vertices = Vec::with_capacity(nv);
        let mut normals = Vec::with_capacity(nv);
        let mut radii = Vec::with_capacity(nv);

        for i in 0..nv {
            let p = self.vertex(i);
            vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            let n = normals_f64[i];
            normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
            let dx = p[0] - centroid[0];
            let dy = p[1] - centroid[1];
            let dz = p[2] - centroid[2];
            radii.push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
        }

        let nf = self.face_count();
        let mut faces = Vec::with_capacity(nf);
        for f in 0..nf {
            faces.push(self.face(f));
        }

        crate::model_calculation::contracts::GeometryState {
            vertices,
            faces,
            normals,
            radii,
            volume,
            centroid,
        }
    }

    /// 从 GeometryState 构造 Mesh
    pub fn from_geometry_state(geo: &crate::model_calculation::contracts::GeometryState) -> Self {
        let mut xyz = Vec::with_capacity(geo.vertices.len() * 3);
        for v in &geo.vertices {
            xyz.push(v[0] as f64);
            xyz.push(v[1] as f64);
            xyz.push(v[2] as f64);
        }
        let mut faces = Vec::with_capacity(geo.faces.len() * 3);
        for f in &geo.faces {
            faces.extend_from_slice(f);
        }
        Mesh { xyz, faces }
    }
}

pub fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn mesh_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
