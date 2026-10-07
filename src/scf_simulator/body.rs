//! The star-shaped body: vertex directions `ω̂_i`, faces, and variable radii `R_i`.
//!
//! An imported mesh is that polyhedron. Each vertex keeps its own direction and radius
//! about the vertex centroid, and its face stays the imported triangle. An ellipsoid,
//! which has no imported faces, is the subdivided icosphere of `body.level`. The frame
//! is the standard one (§10.4.4 step 8b): centroid at the origin, inertia principal
//! axes along `e_x, e_y, e_z`, spin about `e_z`. Directions and topology stay fixed
//! between realignments; a geological step moves vertices by changing `R_i`. There is
//! no tetrahedral volume mesh anywhere in Tier 2.

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::{TriMesh, align_radial_cones};
use crate::common::math::DVec3;
use crate::rtfp_solver::rtfp::hw_rt::HardwareMesh;

use super::error::ScfError;

/// Directions, faces and the quantities that depend on them only.
#[derive(Clone, Debug)]
pub struct Topology {
    pub dirs: Vec<DVec3>,
    pub faces: Vec<[u32; 3]>,
    /// Unit direction of the mean of each face's vertex directions (fixed).
    pub face_dirs: Vec<DVec3>,
    /// Solid angle of each spherical triangle (sums to `4π`).
    pub face_solid: Vec<f64>,
    /// Dual solid angle of each vertex (a third of its incident faces).
    pub vertex_solid: Vec<f64>,
    pub vertex_neighbors: Vec<Vec<usize>>,
    pub vertex_faces: Vec<Vec<usize>>,
    /// Faces sharing an edge, three entries per face (`u32::MAX` never occurs on a closed mesh).
    pub face_neighbors: Vec<[usize; 3]>,
}

impl Topology {
    pub fn new(dirs: Vec<DVec3>, faces: Vec<[u32; 3]>) -> Self {
        let nv = dirs.len();
        let mut face_dirs = Vec::with_capacity(faces.len());
        let mut face_solid = Vec::with_capacity(faces.len());
        let mut vertex_solid = vec![0.0; nv];
        let mut vertex_neighbors = vec![Vec::new(); nv];
        let mut vertex_faces = vec![Vec::new(); nv];
        for (f, tri) in faces.iter().enumerate() {
            let [a, b, c] = tri.map(|i| dirs[i as usize]);
            let mean = (a + b + c).try_normalize().unwrap_or(a);
            face_dirs.push(mean);
            let num = a.dot(b.cross(c));
            let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
            let omega = 2.0 * num.atan2(den).abs();
            face_solid.push(omega);
            for (k, &i) in tri.iter().enumerate() {
                let i = i as usize;
                vertex_solid[i] += omega / 3.0;
                vertex_faces[i].push(f);
                for (j, &o) in tri.iter().enumerate() {
                    if j != k && !vertex_neighbors[i].contains(&(o as usize)) {
                        vertex_neighbors[i].push(o as usize);
                    }
                }
            }
        }
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        let mut face_neighbors = vec![[usize::MAX; 3]; faces.len()];
        for (f, tri) in faces.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (tri[e], tri[(e + 1) % 3]);
                if let Some(&g) = edges.get(&(b, a)) {
                    face_neighbors[f][e] = g;
                    let ge = faces[g]
                        .iter()
                        .position(|&v| v == b)
                        .expect("shared edge belongs to the neighbour");
                    face_neighbors[g][ge] = f;
                } else {
                    edges.insert((a, b), f);
                }
            }
        }
        Self {
            dirs,
            faces,
            face_dirs,
            face_solid,
            vertex_solid,
            vertex_neighbors,
            vertex_faces,
            face_neighbors,
        }
    }

    /// Star nodes of an imported polyhedron, centred on the vertex mean.
    ///
    /// Faces are the imported triangles. Where two non-adjacent faces overlap in
    /// radial projection, their vertices are moved tangentially at constant radius
    /// until the cones are disjoint, so a later radius update (§10.4.4 step 8a)
    /// or the step-8b resample cannot make those faces intersect. A mesh that is
    /// already a star about the vertex mean is unchanged.
    pub fn star_from_mesh(
        vertices: &[DVec3],
        faces: Vec<[u32; 3]>,
    ) -> Result<(Self, Vec<f64>, DVec3), ScfError> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(ScfError::geometry(
                "a solid mesh needs at least 4 vertices and 4 triangles",
            ));
        }
        let origin = vertices.iter().copied().fold(DVec3::ZERO, |acc, v| acc + v)
            * (1.0 / vertices.len() as f64);
        let mut placed = Vec::with_capacity(vertices.len());
        for (i, v) in vertices.iter().enumerate() {
            let offset = *v - origin;
            let r = offset.length();
            if !(r.is_finite() && r > 0.0) {
                return Err(ScfError::geometry(format!(
                    "vertex {i} lies on the centroid and is not a star node"
                )));
            }
            placed.push(offset);
        }
        align_radial_cones(&mut placed, &faces)?;
        let mut dirs = Vec::with_capacity(placed.len());
        let mut radius = Vec::with_capacity(placed.len());
        for (i, p) in placed.iter().enumerate() {
            let Some(d) = p.try_normalize() else {
                return Err(ScfError::geometry(format!(
                    "vertex {i} lies on the centroid and is not a star node"
                )));
            };
            dirs.push(d);
            radius.push(p.length());
        }
        let topo = Self::new(dirs, faces);
        TriMesh::try_new(placed, topo.faces.clone())?;
        Ok((topo, radius, origin))
    }

    /// Subdivided icosahedron with outward, counter-clockwise faces.
    ///
    /// Used for `shape = "ellipsoid"` only. An imported mesh does not pass through this.
    pub fn icosphere(level: usize) -> Self {
        let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut verts: Vec<DVec3> = [
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ]
        .iter()
        .map(|p| DVec3::from_array(*p).try_normalize().unwrap())
        .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..level {
            let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
            let mut next = Vec::with_capacity(faces.len() * 4);
            for tri in &faces {
                let mut mid = [0u32; 3];
                for e in 0..3 {
                    let (a, b) = (tri[e], tri[(e + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    mid[e] = *cache.entry(key).or_insert_with(|| {
                        let m = (verts[a as usize] + verts[b as usize])
                            .try_normalize()
                            .unwrap();
                        verts.push(m);
                        (verts.len() - 1) as u32
                    });
                }
                next.push([tri[0], mid[0], mid[2]]);
                next.push([tri[1], mid[1], mid[0]]);
                next.push([tri[2], mid[2], mid[1]]);
                next.push([mid[0], mid[1], mid[2]]);
            }
            faces = next;
        }
        Self::new(verts, faces)
    }

    pub fn n_vertices(&self) -> usize {
        self.dirs.len()
    }

    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub topo: Arc<Topology>,
    pub radius: Vec<f64>,
    pub bulk: f64,
    pub kernels: Vec<CauchyKernel>,
    /// Loose surface layer `ρ_s` (kg m⁻²) per vertex.
    pub rho_s: Vec<f64>,
}

impl Body {
    pub fn new(
        topo: Arc<Topology>,
        radius: Vec<f64>,
        bulk: f64,
        kernels: Vec<CauchyKernel>,
        rho_s: Vec<f64>,
    ) -> Result<Self, ScfError> {
        if radius.len() != topo.n_vertices() || rho_s.len() != topo.n_vertices() {
            return Err(ScfError::geometry("body arrays do not match the topology"));
        }
        if radius.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
            return Err(ScfError::geometry("a vertex radius is not positive"));
        }
        Ok(Self {
            topo,
            radius,
            bulk,
            kernels,
            rho_s,
        })
    }

    pub fn positions(&self) -> Vec<DVec3> {
        self.topo
            .dirs
            .iter()
            .zip(&self.radius)
            .map(|(d, r)| *d * *r)
            .collect()
    }

    pub fn mesh(&self) -> Result<TriMesh, ScfError> {
        TriMesh::try_new(self.positions(), self.topo.faces.clone()).map_err(ScfError::from)
    }

    pub fn density(&self) -> Result<CauchyDensity, ScfError> {
        CauchyDensity::try_new(self.bulk, self.kernels.clone()).map_err(ScfError::from)
    }

    /// Face centroids, unit normals and areas of the polyhedron.
    pub fn face_geometry(&self) -> Vec<(DVec3, DVec3, f64)> {
        let p = self.positions();
        self.topo
            .faces
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| p[i as usize]);
                let cross = (b - a).cross(c - a);
                let area2 = cross.length();
                let n = cross.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0));
                ((a + b + c) * (1.0 / 3.0), n, 0.5 * area2)
            })
            .collect()
    }

    /// Smallest `n·ê` on the radial graph, and the face that carries it.
    ///
    /// A grazing or overhanging face keeps its radius. Callers report this pair;
    /// it does not reject the body.
    pub fn min_radial(&self) -> (f64, usize) {
        let geo = self.face_geometry();
        let mut worst = f64::INFINITY;
        let mut face = 0usize;
        for (fi, (_, n, _)) in geo.iter().enumerate() {
            let c = n.dot(self.topo.face_dirs[fi]);
            if c < worst {
                worst = c;
                face = fi;
            }
        }
        (worst, face)
    }

    /// Smallest nearest-vertex spacing, `h_{Γ,min}` of the CFL bound (§10.4.1).
    pub fn min_edge(&self) -> f64 {
        let p = self.positions();
        let mut h = f64::INFINITY;
        for (i, nb) in self.topo.vertex_neighbors.iter().enumerate() {
            for &j in nb {
                h = h.min((p[i] - p[j]).length());
            }
        }
        h
    }

    /// Re-sample a closed mesh along the fixed directions from `origin` (outermost crossing).
    pub fn radii_from_mesh(
        mesh: &TriMesh,
        origin: DVec3,
        dirs: &[DVec3],
    ) -> Result<Vec<f64>, ScfError> {
        let hw = HardwareMesh::build(mesh)?;
        let mut batch = hw.new_batch();
        for d in dirs {
            batch.push(origin, *d, None);
        }
        let hits = hw.trace(&batch, false)?;
        let mut buf = Vec::new();
        let mut out = Vec::with_capacity(dirs.len());
        for i in 0..dirs.len() {
            buf.clear();
            hits.distances_into(i, &mut buf);
            match buf.last() {
                Some(r) if *r > 0.0 => out.push(*r),
                _ => {
                    return Err(ScfError::geometry(format!(
                        "ray {i} from the centroid does not leave the body"
                    )));
                }
            }
        }
        Ok(out)
    }

    /// Ellipsoid with semi-axes `(a, b, c)`: the radius along `ω̂` is `1/√(Σ ω_i²/a_i²)`.
    pub fn ellipsoid(topo: Arc<Topology>, axes: [f64; 3]) -> Vec<f64> {
        topo.dirs
            .iter()
            .map(|d| {
                1.0 / ((d.x / axes[0]).powi(2) + (d.y / axes[1]).powi(2) + (d.z / axes[2]).powi(2))
                    .sqrt()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::geometry::TriMesh;
    use std::f64::consts::PI;

    #[test]
    fn icosphere_is_a_closed_outward_sphere() {
        let t = Topology::icosphere(2);
        assert_eq!(t.n_vertices(), 162);
        assert_eq!(t.n_faces(), 320);
        let total: f64 = t.face_solid.iter().sum();
        assert!((total - 4.0 * PI).abs() < 1e-10);
        let vs: f64 = t.vertex_solid.iter().sum();
        assert!((vs - 4.0 * PI).abs() < 1e-10);
        assert!(
            t.face_neighbors
                .iter()
                .all(|n| n.iter().all(|&x| x != usize::MAX))
        );
        let body = Body::new(Arc::new(t), vec![10.0; 162], 1.0, vec![], vec![0.0; 162]).unwrap();
        let mesh = body.mesh().unwrap();
        assert!(mesh.signed_volume() > 0.0);
        let v = 4.0 / 3.0 * PI * 1000.0;
        assert!((mesh.signed_volume() / v - 1.0).abs() < 0.05);
        let (mu, face) = body.min_radial();
        assert!(mu > 0.95, "sphere min radial {mu} at face {face}");
        let mut pulled = body.radius.clone();
        pulled[0] = 4.0;
        let tilted = Body::new(
            body.topo.clone(),
            pulled,
            1.0,
            vec![],
            vec![0.0; body.topo.n_vertices()],
        )
        .unwrap();
        let (mu, face) = tilted.min_radial();
        assert!(
            mu.is_finite() && mu < 0.95,
            "tilted min radial {mu} at face {face}"
        );
        assert!(face < tilted.topo.n_faces());
    }

    #[test]
    fn an_imported_mesh_keeps_every_vertex_and_face() {
        let cube = TriMesh::unit_cube();
        let shift = DVec3::new(3.0, -2.0, 5.0);
        let moved: Vec<DVec3> = cube.vertices.iter().map(|v| *v + shift).collect();
        let (topo, radius, origin) = Topology::star_from_mesh(&moved, cube.faces.clone()).unwrap();
        assert_eq!(topo.n_vertices(), cube.vertices.len());
        assert_eq!(topo.n_faces(), cube.faces.len());
        assert_ne!(topo.n_vertices(), 162);
        assert_ne!(topo.n_faces(), 320);
        assert!((origin - shift).length() < 1e-12);
        for (i, v) in moved.iter().enumerate() {
            let back = topo.dirs[i] * radius[i] + origin;
            assert!((back - *v).length() < 1e-12, "{i}");
        }
    }
}
