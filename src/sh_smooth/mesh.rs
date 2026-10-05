//! Triangle mesh in the same OBJ dialect the rest of the project writes.
//!
//! Smoothing replaces vertex positions and copies `faces` verbatim. A polygon
//! that is not already a triangle is rejected: triangulating it would change
//! the face count.

use std::path::Path;

use super::error::ShError;
use super::math::Vec3;

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub length_unit: String,
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Latitude-longitude sphere. `radius_at(theta, phi)` is the radial function.
    pub fn from_radial(n_lat: usize, radius_at: impl Fn(f64, f64) -> f64) -> Result<Self, ShError> {
        let n_lat = n_lat.max(2);
        let n_lon = n_lat * 2;
        let mut vertices = Vec::new();
        let north = radius_at(0.0, 0.0);
        if !(north.is_finite() && north > 0.0) {
            return Err(ShError::geometry(
                "radial function must be finite and positive",
            ));
        }
        vertices.push([0.0, 0.0, north]);
        for i in 1..n_lat {
            let theta = std::f64::consts::PI * (i as f64) / (n_lat as f64);
            for j in 0..n_lon {
                let phi = 2.0 * std::f64::consts::PI * (j as f64) / (n_lon as f64);
                let r = radius_at(theta, phi);
                if !(r.is_finite() && r > 0.0) {
                    return Err(ShError::geometry(
                        "radial function must be finite and positive",
                    ));
                }
                vertices.push([
                    r * theta.sin() * phi.cos(),
                    r * theta.sin() * phi.sin(),
                    r * theta.cos(),
                ]);
            }
        }
        let south_r = radius_at(std::f64::consts::PI, 0.0);
        if !(south_r.is_finite() && south_r > 0.0) {
            return Err(ShError::geometry(
                "radial function must be finite and positive",
            ));
        }
        vertices.push([0.0, 0.0, -south_r]);
        let south = vertices.len() as u32 - 1;
        let mut faces = Vec::new();
        for j in 0..n_lon {
            let a = 1 + j as u32;
            let b = 1 + ((j + 1) % n_lon) as u32;
            faces.push([0, a, b]);
        }
        for i in 0..(n_lat - 2) {
            for j in 0..n_lon {
                let row = 1 + i * n_lon;
                let a = (row + j) as u32;
                let b = (row + (j + 1) % n_lon) as u32;
                let c = (row + n_lon + j) as u32;
                let d = (row + n_lon + (j + 1) % n_lon) as u32;
                faces.push([a, d, b]);
                faces.push([a, c, d]);
            }
        }
        let row = 1 + (n_lat - 2) * n_lon;
        for j in 0..n_lon {
            let a = (row + j) as u32;
            let b = (row + (j + 1) % n_lon) as u32;
            faces.push([a, south, b]);
        }
        Ok(Self {
            vertices,
            faces,
            length_unit: "1".to_string(),
        })
    }

    pub fn read_obj(path: &Path) -> Result<Self, ShError> {
        let text = std::fs::read_to_string(path).map_err(|err| {
            ShError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })?;
        Self::parse_obj(&text)
    }

    pub fn parse_obj(text: &str) -> Result<Self, ShError> {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut length_unit = "1".to_string();
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# length_unit ") {
                length_unit = rest.trim().to_string();
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let head = parts.next().unwrap_or("");
            match head {
                "v" => {
                    let x = parse_coord(parts.next(), line_no)?;
                    let y = parse_coord(parts.next(), line_no)?;
                    let z = parse_coord(parts.next(), line_no)?;
                    if ![x, y, z].iter().all(|v| v.is_finite()) {
                        return Err(ShError::model_not_closed(format!(
                            "line {}: vertex is not finite",
                            line_no + 1
                        )));
                    }
                    vertices.push([x, y, z]);
                }
                "f" => {
                    let corners: Vec<&str> = parts.collect();
                    if corners.len() != 3 {
                        return Err(ShError::model_not_closed(format!(
                            "line {}: face has {} corners; only triangles are accepted so the face count stays unchanged",
                            line_no + 1,
                            corners.len()
                        )));
                    }
                    let mut face = [0u32; 3];
                    for (k, token) in corners.iter().enumerate() {
                        face[k] = resolve_index(token, vertices.len(), line_no)?;
                    }
                    if face[0] == face[1] || face[1] == face[2] || face[2] == face[0] {
                        return Err(ShError::model_not_closed(format!(
                            "line {}: degenerate triangle with a repeated vertex; faces are not dropped",
                            line_no + 1
                        )));
                    }
                    faces.push(face);
                }
                _ => {}
            }
        }
        if vertices.len() < 3 || faces.is_empty() {
            return Err(ShError::model_not_closed(
                "mesh needs at least 3 vertices and 1 triangle",
            ));
        }
        Ok(Self {
            vertices,
            faces,
            length_unit,
        })
    }

    pub fn to_obj(&self, note: &str) -> String {
        let mut text = String::new();
        text.push_str("# astroid_evolution_mesh 1\n");
        text.push_str(&format!("# length_unit {}\n", self.length_unit));
        text.push_str(&format!("# source_length_unit {}\n", self.length_unit));
        text.push_str(&format!(
            "# mesh_hash {}\n",
            mesh_hash(&self.vertices, &self.faces)
        ));
        text.push_str("# source sh_smooth\n");
        text.push_str("# welded_positions 0\n# dropped_degenerate 0\n# triangulated_strips 0\n# triangulated_fans 0\n");
        if !note.is_empty() {
            text.push_str(&format!("# sh_smooth {note}\n"));
        }
        for v in &self.vertices {
            text.push_str(&format!("v {:.17} {:.17} {:.17}\n", v[0], v[1], v[2]));
        }
        for f in &self.faces {
            text.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        text
    }

    pub fn write_obj(&self, path: &Path, note: &str) -> Result<(), ShError> {
        std::fs::write(path, self.to_obj(note)).map_err(|err| {
            ShError::invalid_input(format!("cannot write {}: {err}", path.display()))
        })
    }

    /// Face lines only, so a smoothed file can be compared with its input.
    pub fn face_lines(&self) -> Vec<String> {
        self.faces
            .iter()
            .map(|f| format!("f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1))
            .collect()
    }
}

fn parse_coord(token: Option<&str>, line_no: usize) -> Result<f64, ShError> {
    let token = token.ok_or_else(|| {
        ShError::model_not_closed(format!(
            "line {}: vertex needs three coordinates",
            line_no + 1
        ))
    })?;
    token.parse::<f64>().map_err(|_| {
        ShError::model_not_closed(format!("line {}: bad coordinate '{token}'", line_no + 1))
    })
}

fn resolve_index(token: &str, nverts: usize, line_no: usize) -> Result<u32, ShError> {
    let index_str = token.split('/').next().unwrap_or("");
    let raw: i32 = index_str.parse().map_err(|_| {
        ShError::model_not_closed(format!("line {}: bad face index '{token}'", line_no + 1))
    })?;
    let idx = if raw > 0 {
        raw as usize - 1
    } else if raw < 0 {
        nverts
            .checked_sub(raw.unsigned_abs() as usize)
            .ok_or_else(|| {
                ShError::model_not_closed(format!(
                    "line {}: face index {raw} is outside the vertex array",
                    line_no + 1
                ))
            })?
    } else {
        return Err(ShError::model_not_closed(format!(
            "line {}: face index 0 is not valid",
            line_no + 1
        )));
    };
    if idx >= nverts {
        return Err(ShError::model_not_closed(format!(
            "line {}: face index {raw} is outside the vertex array",
            line_no + 1
        )));
    }
    Ok(idx as u32)
}

pub fn mesh_hash(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> String {
    let mut hasher = crc32fast::Hasher::new();
    for v in vertices {
        hasher.update(&v[0].to_le_bytes());
        hasher.update(&v[1].to_le_bytes());
        hasher.update(&v[2].to_le_bytes());
    }
    for f in faces {
        for index in f {
            hasher.update(&index.to_le_bytes());
        }
    }
    format!("{:08x}", hasher.finalize())
}

/// Where the radial expansion is centred.
#[derive(Clone, Copy, Debug)]
pub enum Center {
    Auto,
    Origin,
    At([f64; 3]),
}

impl Center {
    pub fn resolve(self, mesh: &Mesh) -> Result<[f64; 3], ShError> {
        match self {
            Center::Origin => Ok([0.0, 0.0, 0.0]),
            Center::At(c) => {
                if !c.iter().all(|v| v.is_finite()) {
                    return Err(ShError::invalid_input("center must be finite"));
                }
                Ok(c)
            }
            Center::Auto => Ok(volume_centroid(mesh).unwrap_or_else(|| vertex_mean(mesh))),
        }
    }
}

fn vertex_mean(mesh: &Mesh) -> [f64; 3] {
    let n = mesh.vertices.len() as f64;
    let mut s = [0.0; 3];
    for v in &mesh.vertices {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

/// Closed-mesh volume centroid. `None` when the signed volume is negligible.
fn volume_centroid(mesh: &Mesh) -> Option<[f64; 3]> {
    let mut vol6 = 0.0;
    let mut acc = [0.0; 3];
    for f in &mesh.faces {
        let a = Vec3::from_array(mesh.vertices[f[0] as usize]);
        let b = Vec3::from_array(mesh.vertices[f[1] as usize]);
        let c = Vec3::from_array(mesh.vertices[f[2] as usize]);
        let tet = a.dot(b.cross(c));
        vol6 += tet;
        let sum = a + b + c;
        acc[0] += tet * sum.x;
        acc[1] += tet * sum.y;
        acc[2] += tet * sum.z;
    }
    let scale = mesh
        .vertices
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(1.0);
    if !vol6.is_finite() || vol6.abs() <= 1e-14 * scale * scale * scale {
        return None;
    }
    Some([
        acc[0] / (4.0 * vol6),
        acc[1] / (4.0 * vol6),
        acc[2] / (4.0 * vol6),
    ])
}

#[derive(Clone, Debug)]
pub struct RadialSamples {
    pub center: [f64; 3],
    pub dir: Vec<[f64; 3]>,
    pub radius: Vec<f64>,
    pub weight: Vec<f64>,
}

impl Mesh {
    /// Unit directions and positive radii about `center`, with solid-angle weights.
    pub fn radial_samples(&self, center: [f64; 3]) -> Result<RadialSamples, ShError> {
        let mut dir = Vec::with_capacity(self.vertices.len());
        let mut radius = Vec::with_capacity(self.vertices.len());
        for (i, v) in self.vertices.iter().enumerate() {
            let d = Vec3::new(v[0] - center[0], v[1] - center[1], v[2] - center[2]);
            let len = d.length();
            if !(len.is_finite() && len > 0.0) {
                return Err(ShError::geometry(format!(
                    "vertex {i} is not a positive radius about the expansion center"
                )));
            }
            let u = d * (1.0 / len);
            dir.push(u.to_array());
            radius.push(len);
        }
        let weight = solid_angle_weights(&dir, &self.faces);
        Ok(RadialSamples {
            center,
            dir,
            radius,
            weight,
        })
    }
}

fn solid_angle_weights(dirs: &[[f64; 3]], faces: &[[u32; 3]]) -> Vec<f64> {
    let mut w = vec![0.0; dirs.len()];
    for f in faces {
        let a = Vec3::from_array(dirs[f[0] as usize]);
        let b = Vec3::from_array(dirs[f[1] as usize]);
        let c = Vec3::from_array(dirs[f[2] as usize]);
        let triple = a.dot(b.cross(c));
        let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
        let omega = 2.0 * triple.atan2(den);
        if !omega.is_finite() {
            continue;
        }
        let share = omega.abs() / 3.0;
        for k in 0..3 {
            w[f[k] as usize] += share;
        }
    }
    let total: f64 = w.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return vec![1.0; dirs.len()];
    }
    w
}
