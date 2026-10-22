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
use crate::rtfp::mesh::BodyMesh;

use crate::common::error::SimError;

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
    ) -> Result<(Self, Vec<f64>, DVec3), SimError> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(SimError::geometry(
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
                return Err(SimError::geometry(format!(
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
                return Err(SimError::geometry(format!(
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

    /// Icosphere of `level` about the volume centroid of a closed mesh, each radius
    /// the outermost hit of its ray (`shape = "mesh_rays"`). Facets far below the
    /// node spacing (boulders) are not resolved; the regolith-scale surface is.
    pub fn star_from_rays(
        level: usize,
        vertices: &[DVec3],
        faces: &[[u32; 3]],
    ) -> Result<(Self, Vec<f64>, DVec3), SimError> {
        use rayon::prelude::*;
        let mesh = TriMesh::try_new(vertices.to_vec(), faces.to_vec())?;
        // Volume centroid from the signed tetrahedra on the origin.
        let (mut six_v, mut moment) = (0.0, DVec3::ZERO);
        for f in 0..faces.len() {
            let [a, b, c] = mesh.triangle(f);
            let v = a.dot(b.cross(c));
            six_v += v;
            moment += (a + b + c) * v;
        }
        let origin = moment * (1.0 / (4.0 * six_v));
        let topo = Self::icosphere(level);
        let index = super::transport::DirIndex::build(&topo.dirs);
        let dirs = &topo.dirs;
        let hits: Vec<(usize, f64)> = (0..faces.len())
            .into_par_iter()
            .flat_map_iter(|f| {
                let [a, b, c] = mesh.triangle(f).map(|p| p - origin);
                let centre = (a + b + c).normalize_or_zero();
                let alpha = [a, b, c]
                    .iter()
                    .map(|p| p.normalize_or_zero().dot(centre).clamp(-1.0, 1.0).acos())
                    .fold(0.0, f64::max)
                    + 1e-9;
                let mut near = Vec::new();
                index.collect_within(centre, alpha, &mut near);
                let (e1, e2) = (b - a, c - a);
                near.into_iter().filter_map(move |i| {
                    // Möller–Trumbore from the origin along dirs[i].
                    let d = dirs[i];
                    let p = d.cross(e2);
                    let det = e1.dot(p);
                    if det.abs() < 1e-300 {
                        return None;
                    }
                    let s = -a;
                    let u = s.dot(p) / det;
                    let q = s.cross(e1);
                    let v = d.dot(q) / det;
                    let t = e2.dot(q) / det;
                    (u >= -1e-12 && v >= -1e-12 && u + v <= 1.0 + 1e-12 && t > 0.0)
                        .then_some((i, t))
                })
            })
            .collect();
        let mut radius = vec![0.0_f64; dirs.len()];
        for (i, t) in hits {
            radius[i] = radius[i].max(t);
        }
        if let Some(i) = radius.iter().position(|r| *r <= 0.0) {
            return Err(SimError::geometry(format!(
                "ray {i} from the volume centroid misses the mesh"
            )));
        }
        Ok((topo, radius, origin))
    }

    /// Subdivided icosahedron with outward, counter-clockwise faces.
    ///
    /// Used for `shape = "ellipsoid"` and `shape = "mesh_rays"`.
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
    ) -> Result<Self, SimError> {
        if radius.len() != topo.n_vertices() || rho_s.len() != topo.n_vertices() {
            return Err(SimError::geometry("body arrays do not match the topology"));
        }
        if radius.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
            return Err(SimError::geometry("a vertex radius is not positive"));
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

    pub fn mesh(&self) -> Result<TriMesh, SimError> {
        TriMesh::try_new(self.positions(), self.topo.faces.clone())
    }

    pub fn density(&self) -> Result<CauchyDensity, SimError> {
        CauchyDensity::try_new(self.bulk, self.kernels.clone())
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
    ) -> Result<Vec<f64>, SimError> {
        let hw = BodyMesh::build(mesh)?;
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
                    return Err(SimError::geometry(format!(
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
