//! The body: a closed polyhedron carried by its vertex positions `X_i` (§8.1, G4).
//!
//! Connectivity is fixed for a run; a geological step moves vertices along their
//! area-weighted normals `N_i`, and the frame realignment of step 8b is a rigid
//! motion of the vertices (G8). Nothing here assumes the body is star-shaped about
//! the origin. An imported mesh is used as it is; `shape = "mesh_rays"` and the
//! ellipsoid build an icosphere-connected polyhedron, which is only a choice of
//! initial triangulation. There is no tetrahedral volume mesh anywhere in Tier 2.

use std::collections::HashMap;
use std::sync::Arc;

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;

use crate::common::error::SimError;

/// Faces and adjacency, which do not change during a run.
#[derive(Clone, Debug)]
pub struct Topology {
    pub faces: Vec<[u32; 3]>,
    pub vertex_neighbors: Vec<Vec<usize>>,
    pub vertex_faces: Vec<Vec<usize>>,
    /// Faces sharing an edge: entry `e` is the face across the edge `(t[e], t[e + 1])`.
    pub face_neighbors: Vec<[usize; 3]>,
    /// `Some(level)` when the faces are those of `icosphere(level)`, so that a snapshot
    /// need not store them.
    pub icosphere: Option<usize>,
}

impl Topology {
    pub fn new(n_vertices: usize, faces: Vec<[u32; 3]>) -> Self {
        let mut vertex_neighbors = vec![Vec::new(); n_vertices];
        let mut vertex_faces = vec![Vec::new(); n_vertices];
        for (f, tri) in faces.iter().enumerate() {
            for (k, &i) in tri.iter().enumerate() {
                let i = i as usize;
                vertex_faces[i].push(f);
                for (j, &o) in tri.iter().enumerate() {
                    if j != k && !vertex_neighbors[i].contains(&(o as usize)) {
                        vertex_neighbors[i].push(o as usize);
                    }
                }
            }
        }
        let mut edges: HashMap<(u32, u32), usize> = HashMap::with_capacity(faces.len() * 3 / 2);
        let mut face_neighbors = vec![[usize::MAX; 3]; faces.len()];
        for (f, tri) in faces.iter().enumerate() {
            for e in 0..3 {
                let (a, b) = (tri[e], tri[(e + 1) % 3]);
                if let Some(&g) = edges.get(&(b, a)) {
                    face_neighbors[f][e] = g;
                    let ge = (0..3)
                        .find(|&k| faces[g][k] == b && faces[g][(k + 1) % 3] == a)
                        .expect("shared edge belongs to the neighbour");
                    face_neighbors[g][ge] = f;
                } else {
                    edges.insert((a, b), f);
                }
            }
        }
        Self {
            faces,
            vertex_neighbors,
            vertex_faces,
            face_neighbors,
            icosphere: None,
        }
    }

    /// Subdivided icosahedron with outward, counter-clockwise faces, and its unit vertices.
    pub fn icosphere(level: usize) -> (Self, Vec<DVec3>) {
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
        .map(|p| DVec3::from_array(*p).normalize())
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
                        verts.push((verts[a as usize] + verts[b as usize]).normalize());
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
        let mut topo = Self::new(verts.len(), faces);
        topo.icosphere = Some(level);
        (topo, verts)
    }

    pub fn n_vertices(&self) -> usize {
        self.vertex_faces.len()
    }

    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
}

/// Volume centroid of a closed, outward-oriented triangle mesh (signed tetrahedra on the origin).
pub fn volume_centroid(vertices: &[DVec3], faces: &[[u32; 3]]) -> DVec3 {
    let (mut six_v, mut moment) = (0.0, DVec3::ZERO);
    for t in faces {
        let [a, b, c] = t.map(|i| vertices[i as usize]);
        let v = a.dot(b.cross(c));
        six_v += v;
        moment += (a + b + c) * v;
    }
    moment * (1.0 / (4.0 * six_v))
}

#[derive(Clone, Debug)]
pub struct Body {
    pub topo: Arc<Topology>,
    /// Vertex positions `X_i` (m).
    pub positions: Vec<DVec3>,
    pub bulk: f64,
    pub kernels: Vec<CauchyKernel>,
    /// Loose surface layer `ρ_s` (kg m⁻²) per vertex.
    pub rho_s: Vec<f64>,
}

impl Body {
    pub fn new(
        topo: Arc<Topology>,
        positions: Vec<DVec3>,
        bulk: f64,
        kernels: Vec<CauchyKernel>,
        rho_s: Vec<f64>,
    ) -> Result<Self, SimError> {
        if positions.len() != topo.n_vertices() || rho_s.len() != topo.n_vertices() {
            return Err(SimError::geometry("body arrays do not match the topology"));
        }
        if positions.iter().any(|p| !p.is_finite()) {
            return Err(SimError::geometry("a vertex position is not finite"));
        }
        Ok(Self {
            topo,
            positions,
            bulk,
            kernels,
            rho_s,
        })
    }

    /// An icosphere-connected polyhedron: each unit vertex of `icosphere(level)` scaled
    /// by the outermost crossing of its ray from the volume centroid of a closed mesh
    /// (`shape = "mesh_rays"`). Returns the body frame origin in the input frame.
    pub fn rays_from_mesh(
        level: usize,
        vertices: &[DVec3],
        faces: &[[u32; 3]],
    ) -> Result<(Arc<Topology>, Vec<DVec3>, DVec3), SimError> {
        Self::rays_from(level, vertices, faces, volume_centroid(vertices, faces))
    }

    /// `rays_from_mesh` with the rays from `origin`. From a point inside the solid every ray
    /// leaves it, so a body whose centroid lies outside (a crescent) is still represented.
    pub fn rays_from(
        level: usize,
        vertices: &[DVec3],
        faces: &[[u32; 3]],
        origin: DVec3,
    ) -> Result<(Arc<Topology>, Vec<DVec3>, DVec3), SimError> {
        TriMesh::try_new(vertices.to_vec(), faces.to_vec())?;
        let (topo, dirs) = Topology::icosphere(level);
        // The outermost crossing of each ray, found inward from beyond the body in f64 on
        // the CPU: the same radii on every platform, which a grazing ray needs.
        let index = super::surface::SurfaceIndex::build(vertices, faces);
        let far = 2.0 * vertices.iter().map(|v| (*v - origin).length()).fold(0.0, f64::max);
        let out = dirs
            .iter()
            .enumerate()
            .map(|(i, d)| match index.first_hit(origin + *d * far, *d * -far, 0.0, 1.0) {
                Some((t, _)) if t < 1.0 => Ok(*d * (far * (1.0 - t))),
                _ => Err(SimError::geometry(format!(
                    "ray {i} from {origin:?} misses the mesh"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((Arc::new(topo), out, origin))
    }

    /// An imported closed mesh used as it is, centred on its volume centroid.
    pub fn from_mesh(
        vertices: &[DVec3],
        faces: Vec<[u32; 3]>,
    ) -> Result<(Arc<Topology>, Vec<DVec3>, DVec3), SimError> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(SimError::geometry(
                "a solid mesh needs at least 4 vertices and 4 triangles",
            ));
        }
        TriMesh::try_new(vertices.to_vec(), faces.clone())?;
        let origin = volume_centroid(vertices, &faces);
        let placed: Vec<DVec3> = vertices.iter().map(|v| *v - origin).collect();
        Ok((Arc::new(Topology::new(placed.len(), faces)), placed, origin))
    }

    /// Ellipsoid with semi-axes `(a, b, c)` on the icosphere of `level`.
    pub fn ellipsoid(level: usize, axes: [f64; 3]) -> (Arc<Topology>, Vec<DVec3>) {
        let (topo, dirs) = Topology::icosphere(level);
        let positions = dirs
            .iter()
            .map(|d| {
                let r = 1.0
                    / ((d.x / axes[0]).powi(2) + (d.y / axes[1]).powi(2) + (d.z / axes[2]).powi(2))
                        .sqrt();
                *d * r
            })
            .collect();
        (Arc::new(topo), positions)
    }

    pub fn positions(&self) -> &[DVec3] {
        &self.positions
    }

    pub fn mesh(&self) -> Result<TriMesh, SimError> {
        TriMesh::try_new(self.positions.clone(), self.topo.faces.clone())
    }

    pub fn density(&self) -> Result<CauchyDensity, SimError> {
        CauchyDensity::try_new(self.bulk, self.kernels.clone())
    }

    /// Face centroids, unit normals and areas of the polyhedron.
    pub fn face_geometry(&self) -> Vec<(DVec3, DVec3, f64)> {
        let p = &self.positions;
        self.topo
            .faces
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| p[i as usize]);
                let cross = (b - a).cross(c - a);
                let n = cross.try_normalize().unwrap_or(DVec3::Z);
                ((a + b + c) * (1.0 / 3.0), n, 0.5 * cross.length())
            })
            .collect()
    }

    /// `A_i = ∂Vol/∂X_i = ⅓ Σ_{f∋i} A_f n_f` (G4).
    pub fn area_vectors(&self) -> Vec<DVec3> {
        let p = &self.positions;
        let mut a = vec![DVec3::ZERO; p.len()];
        for t in &self.topo.faces {
            let [i, j, k] = t.map(|v| v as usize);
            // A_f n_f = ½ (b − a) × (c − a).
            let an = (p[j] - p[i]).cross(p[k] - p[i]) * (1.0 / 6.0);
            a[i] += an;
            a[j] += an;
            a[k] += an;
        }
        a
    }

    /// Unit normals `N_i = A_i / |A_i|` (G4).
    pub fn vertex_normals(&self) -> Vec<DVec3> {
        self.area_vectors()
            .into_iter()
            .map(|a| a.try_normalize().unwrap_or(DVec3::Z))
            .collect()
    }

    /// Vertex areas `A_i = ⅓ Σ_{f∋i} A_f`.
    pub fn vertex_areas(&self) -> Vec<f64> {
        let p = &self.positions;
        let mut a = vec![0.0; p.len()];
        for t in &self.topo.faces {
            let [i, j, k] = t.map(|v| v as usize);
            let s = (p[j] - p[i]).cross(p[k] - p[i]).length() / 6.0;
            a[i] += s;
            a[j] += s;
            a[k] += s;
        }
        a
    }

    /// `R_* = max_Γ |x|`, the radius of the enclosing ball of the basis (G1).
    pub fn r_star(&self) -> f64 {
        self.positions.iter().map(|p| p.length()).fold(0.0, f64::max)
    }

    /// Mean vertex distance from the frame origin: the length scale `R̄`.
    pub fn r_mean(&self) -> f64 {
        self.positions.iter().map(|p| p.length()).sum::<f64>() / self.positions.len().max(1) as f64
    }
}
