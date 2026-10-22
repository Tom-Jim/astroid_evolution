//! Triangle mesh in the same OBJ dialect the rest of the project writes.
//!
//! Smoothing replaces vertex positions and copies `faces` verbatim. A polygon
//! that is not already a triangle is rejected: triangulating it would change
//! the face count.

use std::path::Path;

use crate::common::error::SimError;
use crate::common::math::DVec3 as Vec3;

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
    pub fn from_radial(
        n_lat: usize,
        radius_at: impl Fn(f64, f64) -> f64,
    ) -> Result<Self, SimError> {
        let n_lat = n_lat.max(2);
        let n_lon = n_lat * 2;
        let mut vertices = Vec::new();
        let north = radius_at(0.0, 0.0);
        if !(north.is_finite() && north > 0.0) {
            return Err(SimError::geometry(
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
                    return Err(SimError::geometry(
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
            return Err(SimError::geometry(
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

    pub fn read_obj(path: &Path) -> Result<Self, SimError> {
        let text = std::fs::read_to_string(path).map_err(|err| {
            SimError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })?;
        Self::parse_obj(&text)
    }

    /// Triangles only: smoothing must keep the face list exactly.
    pub fn parse_obj(text: &str) -> Result<Self, SimError> {
        let obj = crate::common::obj::parse(text, true)?;
        Ok(Self {
            vertices: obj.vertices,
            faces: obj.faces,
            length_unit: obj.length_unit.unwrap_or_else(|| "1".into()),
        })
    }

    pub fn write_obj(&self, path: &Path, note: &str) -> Result<(), SimError> {
        let mut header = format!(
            "# astroid_evolution_mesh 1\n# length_unit {0}\n# source_length_unit {0}\n# mesh_hash {1}\n\
             # source sh_smooth\n# welded_positions 0\n# dropped_degenerate 0\n# triangulated_strips 0\n# triangulated_fans 0\n",
            self.length_unit,
            crate::common::obj::hash(&self.vertices, &self.faces)
        );
        if !note.is_empty() {
            header.push_str(&format!("# sh_smooth {note}\n"));
        }
        crate::common::obj::write(path, &header, self.vertices.iter().copied(), &self.faces)
    }

    /// Face lines only, so a smoothed file can be compared with its input.
    pub fn face_lines(&self) -> Vec<String> {
        self.faces
            .iter()
            .map(|f| format!("f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1))
            .collect()
    }
}

/// Where the radial expansion is centred.
#[derive(Clone, Copy, Debug)]
pub enum Center {
    Auto,
    Origin,
    At([f64; 3]),
}

impl Center {
    pub fn resolve(self, mesh: &Mesh) -> Result<[f64; 3], SimError> {
        match self {
            Center::Origin => Ok([0.0, 0.0, 0.0]),
            Center::At(c) => {
                if !c.iter().all(|v| v.is_finite()) {
                    return Err(SimError::invalid_input("center must be finite"));
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
    pub fn radial_samples(&self, center: [f64; 3]) -> Result<RadialSamples, SimError> {
        let mut dir = Vec::with_capacity(self.vertices.len());
        let mut radius = Vec::with_capacity(self.vertices.len());
        for (i, v) in self.vertices.iter().enumerate() {
            let d = Vec3::new(v[0] - center[0], v[1] - center[1], v[2] - center[2]);
            let len = d.length();
            if !(len.is_finite() && len > 0.0) {
                return Err(SimError::geometry(format!(
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
