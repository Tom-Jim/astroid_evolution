use crate::common::error::CommonError;
use crate::common::math::DVec3;

/// Owned closed triangle mesh. Faces are ordered so a positive volume has outward normals.
#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<DVec3>,
    pub faces: Vec<[u32; 3]>,
}

impl TriMesh {
    pub fn try_new(vertices: Vec<DVec3>, faces: Vec<[u32; 3]>) -> Result<Self, CommonError> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(CommonError::model_not_closed(
                "a solid mesh needs at least 4 vertices and 4 triangles",
            ));
        }
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err(CommonError::model_not_closed(
                "mesh vertices must be finite",
            ));
        }
        for f in &faces {
            if f.iter().any(|&i| i as usize >= vertices.len()) {
                return Err(CommonError::model_not_closed(
                    "mesh face index is outside the vertex array",
                ));
            }
            if f[0] == f[1] || f[1] == f[2] || f[2] == f[0] {
                return Err(CommonError::model_not_closed(
                    "degenerate triangle with a repeated vertex",
                ));
            }
        }
        let mesh = Self { vertices, faces };
        let volume = mesh.signed_volume();
        if !volume.is_finite() || volume <= 0.0 {
            return Err(CommonError::model_not_closed(
                "mesh signed volume is non-positive; face winding is not the outward orientation",
            ));
        }
        Ok(mesh)
    }

    pub fn triangle(&self, face: usize) -> [DVec3; 3] {
        let [i, j, k] = self.faces[face];
        [
            self.vertices[i as usize],
            self.vertices[j as usize],
            self.vertices[k as usize],
        ]
    }

    pub fn face_normal(&self, face: usize) -> Option<DVec3> {
        let [a, b, c] = self.triangle(face);
        (b - a).cross(c - a).try_normalize()
    }

    /// `V = (1/6) Σ_f det(v0, v1, v2)` for outward faces.
    pub fn signed_volume(&self) -> f64 {
        let mut v = 0.0;
        for f in 0..self.faces.len() {
            let [a, b, c] = self.triangle(f);
            v += a.dot(b.cross(c));
        }
        v / 6.0
    }

    pub fn unit_cube() -> Self {
        let vertices = vec![
            DVec3::new(-1.0, -1.0, -1.0),
            DVec3::new(1.0, -1.0, -1.0),
            DVec3::new(1.0, 1.0, -1.0),
            DVec3::new(-1.0, 1.0, -1.0),
            DVec3::new(-1.0, -1.0, 1.0),
            DVec3::new(1.0, -1.0, 1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(-1.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        Self::try_new(vertices, faces).expect("unit cube is a closed outward mesh")
    }

    /// Right tetrahedron with vertices `(0,0,0), (1,0,0), (0,1,0), (0,0,1)`.
    pub fn right_tetrahedron() -> Self {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        Self::try_new(vertices, faces).expect("tetrahedron winding")
    }

    /// Latitude-longitude sphere of radius `radius` centered at the origin.
    pub fn uv_sphere(n_lat: usize, radius: f64) -> Self {
        let n_lat = n_lat.max(2);
        let n_lon = n_lat * 2;
        let mut vertices = Vec::new();
        vertices.push(DVec3::new(0.0, 0.0, radius));
        for i in 1..n_lat {
            let theta = std::f64::consts::PI * (i as f64) / (n_lat as f64);
            for j in 0..n_lon {
                let phi = 2.0 * std::f64::consts::PI * (j as f64) / (n_lon as f64);
                vertices.push(DVec3::new(
                    radius * theta.sin() * phi.cos(),
                    radius * theta.sin() * phi.sin(),
                    radius * theta.cos(),
                ));
            }
        }
        vertices.push(DVec3::new(0.0, 0.0, -radius));
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
        Self::try_new(vertices, faces).expect("uv sphere")
    }
}
