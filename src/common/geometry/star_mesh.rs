//! Star-shaped surface stored as one component. Directions, radii, and faces are SoA.

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;

#[derive(Clone, Debug)]
pub struct StarMesh {
    pub dir_x: Vec<f64>,
    pub dir_y: Vec<f64>,
    pub dir_z: Vec<f64>,
    pub radius: Vec<f64>,
    pub faces: Vec<[u32; 3]>,
}

impl StarMesh {
    pub fn len(&self) -> usize {
        self.radius.len()
    }

    pub fn is_empty(&self) -> bool {
        self.radius.is_empty()
    }

    pub fn direction(&self, i: usize) -> DVec3 {
        DVec3::new(self.dir_x[i], self.dir_y[i], self.dir_z[i])
    }

    /// `x_i = center + R_i ω̂_i` with unit directions.
    pub fn realize(&self, center: DVec3) -> Result<Vec<DVec3>, String> {
        if self.dir_x.len() != self.radius.len()
            || self.dir_y.len() != self.radius.len()
            || self.dir_z.len() != self.radius.len()
        {
            return Err("star mesh direction and radius arrays differ".into());
        }
        let mut out = Vec::with_capacity(self.radius.len());
        for i in 0..self.radius.len() {
            let r = self.radius[i];
            let u = self.direction(i);
            let norm = u.length();
            if !r.is_finite() || r <= 0.0 {
                return Err("radius must be a finite positive length".into());
            }
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-8 {
                return Err("node direction must be a unit vector".into());
            }
            out.push(center + u * r);
        }
        Ok(out)
    }

    pub fn to_tri_mesh(&self, center: DVec3) -> Result<TriMesh, String> {
        let vertices = self.realize(center)?;
        TriMesh::try_new(vertices, self.faces.clone()).map_err(|err| err.to_string())
    }
}
