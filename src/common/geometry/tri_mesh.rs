use rayon::prelude::*;

use crate::common::error::SimError;
use crate::common::math::DVec3;

/// Owned closed triangle mesh. Faces are ordered so a positive volume has outward normals.
#[derive(Clone, Debug)]
pub struct TriMesh {
    pub vertices: Vec<DVec3>,
    pub faces: Vec<[u32; 3]>,
}

/// One connected boundary shell of a mesh under hypothesis (G).
#[derive(Clone, Debug)]
pub struct Shell {
    /// Face indices of the shell.
    pub faces: Vec<usize>,
    /// Its signed volume: positive for an outer shell, negative for a cavity.
    pub signed_volume: f64,
    pub role: ShellRole,
}

/// What a shell bounds: the exterior of the solid, or a sealed cavity inside
/// the outer shell `outer` (an index into `TriMesh::components`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellRole {
    Outer,
    Cavity { outer: usize },
}

impl TriMesh {
    pub fn try_new(vertices: Vec<DVec3>, faces: Vec<[u32; 3]>) -> Result<Self, SimError> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(SimError::model_not_closed(
                "a solid mesh needs at least 4 vertices and 4 triangles",
            ));
        }
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err(SimError::model_not_closed("mesh vertices must be finite"));
        }
        for f in &faces {
            if f.iter().any(|&i| i as usize >= vertices.len()) {
                return Err(SimError::model_not_closed(
                    "mesh face index is outside the vertex array",
                ));
            }
            if f[0] == f[1] || f[1] == f[2] || f[2] == f[0] {
                return Err(SimError::model_not_closed(
                    "degenerate triangle with a repeated vertex",
                ));
            }
        }
        let mesh = Self { vertices, faces };
        mesh.validate_solid()?;
        Ok(mesh)
    }

    /// Closed edges, consistent winding, non-degenerate faces, no self-intersection,
    /// positive solid volume, and inward-pointing cavity shells.
    pub fn validate_solid(&self) -> Result<(), SimError> {
        let scale = bounding_scale(&self.vertices).max(1e-12);
        let area_tol = 1e-14 * scale * scale;
        for index in 0..self.faces.len() {
            let [a, b, c] = self.triangle(index);
            let area2 = (b - a).cross(c - a).length();
            if !area2.is_finite() || area2 <= area_tol {
                return Err(SimError::model_not_closed(format!(
                    "degenerate triangle {index}"
                )));
            }
        }
        validate_edges(&self.faces)?;
        validate_no_self_intersection(self, scale)?;
        validate_components_and_volume(self, scale)?;
        Ok(())
    }

    pub fn triangle(&self, face: usize) -> [DVec3; 3] {
        let [i, j, k] = self.faces[face];
        [
            self.vertices[i as usize],
            self.vertices[j as usize],
            self.vertices[k as usize],
        ]
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

    /// The boundary shells of hypothesis (G): one or more outward-wound outer
    /// components and the inward-wound cavity shells nested in them. On a mesh
    /// accepted by `try_new` every cavity lies inside an outer shell.
    pub fn components(&self) -> Vec<Shell> {
        let components = face_components(self);
        let scale = bounding_scale(&self.vertices).max(1e-12);
        let mut shells: Vec<Shell> = components
            .iter()
            .map(|faces| Shell {
                faces: faces.clone(),
                signed_volume: component_volume(self, faces),
                role: ShellRole::Outer,
            })
            .collect();
        let positive: Vec<usize> = shells
            .iter()
            .enumerate()
            .filter(|(_, s)| s.signed_volume > 0.0)
            .map(|(i, _)| i)
            .collect();
        for i in 0..shells.len() {
            if shells[i].signed_volume > 0.0 {
                continue;
            }
            let host = positive
                .iter()
                .find(|&o| shell_inside(self, &shells[i].faces, &shells[*o].faces, scale));
            if let Some(&o) = host {
                shells[i].role = ShellRole::Cavity { outer: o };
            }
        }
        shells
    }

    /// The sealed cavities of the mesh: one harmonic flux mode of (G1) is
    /// added per cavity shell.
    pub fn cavities(&self) -> Vec<Shell> {
        self.components()
            .into_iter()
            .filter(|s| matches!(s.role, ShellRole::Cavity { .. }))
            .collect()
    }

    /// An interior point `z_j` of every sealed cavity, for the flux modes of
    /// Proposition 8.1.5. The vertex centroid is tried first, then nudged
    /// vertices; a candidate counts when the shell's own ray test puts it
    /// inside.
    pub fn cavity_interior_points(&self) -> Vec<DVec3> {
        let scale = bounding_scale(&self.vertices).max(1e-12);
        self.cavities()
            .iter()
            .filter_map(|shell| {
                let mut verts: Vec<DVec3> = Vec::new();
                for &f in &shell.faces {
                    for v in self.faces[f] {
                        let p = self.vertices[v as usize];
                        if !verts.contains(&p) {
                            verts.push(p);
                        }
                    }
                }
                if verts.is_empty() {
                    return None;
                }
                let centre = verts.iter().copied().sum::<DVec3>() / verts.len() as f64;
                let mut candidates = vec![centre];
                for &v in verts.iter().take(24) {
                    candidates.push(v + (centre - v) * 0.1);
                    candidates.push(v + (centre - v) * 0.5);
                }
                candidates
                    .into_iter()
                    .find(|&p| point_inside_shell(self, &shell.faces, p, scale))
            })
            .collect()
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

fn bounding_scale(vertices: &[DVec3]) -> f64 {
    let mut min = vertices[0];
    let mut max = vertices[0];
    for v in vertices {
        min.x = min.x.min(v.x);
        min.y = min.y.min(v.y);
        min.z = min.z.min(v.z);
        max.x = max.x.max(v.x);
        max.y = max.y.max(v.y);
        max.z = max.z.max(v.z);
    }
    (max - min).length()
}

fn validate_edges(faces: &[[u32; 3]]) -> Result<(), SimError> {
    use std::collections::BTreeMap;
    let mut uses: BTreeMap<(u32, u32), [u32; 2]> = BTreeMap::new();
    for face in faces {
        for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let (lo, hi, forward) = if edge.0 < edge.1 {
                (edge.0, edge.1, 0)
            } else {
                (edge.1, edge.0, 1)
            };
            let slot = uses.entry((lo, hi)).or_insert([0, 0]);
            slot[forward] += 1;
        }
    }
    for ((lo, hi), [forward, back]) in uses {
        if forward == 1 && back == 1 {
            continue;
        }
        if forward + back == 2 {
            return Err(SimError::model_not_closed(format!(
                "edge {lo}-{hi} is wound the same way on both faces"
            )));
        }
        return Err(SimError::model_not_closed(format!(
            "edge {lo}-{hi} is not a closed manifold edge"
        )));
    }
    Ok(())
}

fn validate_no_self_intersection(mesh: &TriMesh, scale: f64) -> Result<(), SimError> {
    let eps = 1e-10 * scale;
    let mut min = mesh.vertices[0];
    let mut max = mesh.vertices[0];
    for v in &mesh.vertices {
        min.x = min.x.min(v.x);
        min.y = min.y.min(v.y);
        min.z = min.z.min(v.z);
        max.x = max.x.max(v.x);
        max.y = max.y.max(v.y);
        max.z = max.z.max(v.z);
    }
    let extent = (max.x - min.x)
        .max(max.y - min.y)
        .max(max.z - min.z)
        .max(scale)
        .max(1e-12);
    let cells_f = (mesh.faces.len() as f64).cbrt().ceil().clamp(4.0, 128.0);
    let cells = cells_f as i32;
    let cell = extent / cells_f;
    let origins = [min.x, min.y, min.z];
    let index_of = |value: f64, axis: usize| -> i32 {
        let i = ((value - origins[axis]) / cell).floor();
        if !i.is_finite() {
            return 0;
        }
        (i as i64).clamp(0, cells as i64) as i32
    };
    let mut boxes = Vec::with_capacity(mesh.faces.len());
    for i in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(i);
        let lo = [
            a.x.min(b.x).min(c.x),
            a.y.min(b.y).min(c.y),
            a.z.min(b.z).min(c.z),
        ];
        let hi = [
            a.x.max(b.x).max(c.x),
            a.y.max(b.y).max(c.y),
            a.z.max(b.z).max(c.z),
        ];
        boxes.push((
            [index_of(lo[0], 0), index_of(lo[1], 1), index_of(lo[2], 2)],
            [index_of(hi[0], 0), index_of(hi[1], 1), index_of(hi[2], 2)],
        ));
    }
    let side = (cells + 1) as usize;
    let mut grid = vec![Vec::<usize>::new(); side * side * side];
    for (i, (lo, hi)) in boxes.iter().enumerate() {
        for ix in lo[0]..=hi[0] {
            for iy in lo[1]..=hi[1] {
                for iz in lo[2]..=hi[2] {
                    let slot = ((ix as usize * side + iy as usize) * side) + iz as usize;
                    grid[slot].push(i);
                }
            }
        }
    }
    let threads = rayon::current_num_threads().max(1);
    let span = grid.len().div_ceil(threads);
    let hit = (0..threads)
        .into_par_iter()
        .filter_map(|worker| {
            let begin = worker * span;
            let end = (begin + span).min(grid.len());
            let mut best: Option<(usize, usize, f64)> = None;
            for slot in begin..end {
                let bucket = &grid[slot];
                if bucket.len() < 2 {
                    continue;
                }
                let ix = (slot / (side * side)) as i32;
                let rem = slot % (side * side);
                let iy = (rem / side) as i32;
                let iz = (rem % side) as i32;
                let here = [ix, iy, iz];
                for a in 0..bucket.len() {
                    for b in (a + 1)..bucket.len() {
                        let i = bucket[a];
                        let j = bucket[b];
                        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
                        if lo == hi || !pair_owned_by(here, boxes[lo].0, boxes[hi].0) {
                            continue;
                        }
                        let (fa, fb) = (mesh.faces[lo], mesh.faces[hi]);
                        let found = match shared_vertex_count(fa, fb) {
                            0 => intersection_gap(mesh.triangle(lo), mesh.triangle(hi), eps),
                            1 => fan_gap(fa, fb, mesh.triangle(lo), mesh.triangle(hi), eps),
                            _ => None,
                        };
                        if let Some(found) = found {
                            let cand = (lo, hi, found.gap);
                            if best.is_none_or(|kept| (cand.0, cand.1) < (kept.0, kept.1)) {
                                best = Some(cand);
                            }
                        }
                    }
                }
            }
            best
        })
        .reduce_with(|a, b| if (a.0, a.1) <= (b.0, b.1) { a } else { b });
    if let Some((lo, hi, gap)) = hit {
        return Err(SimError::model_not_closed(format!(
            "triangles {lo} and {hi} intersect (gap {:.4} m)",
            gap
        )));
    }
    Ok(())
}

/// A pair is tested in the lowest cell both boxes occupy, and nowhere else.
fn pair_owned_by(cell: [i32; 3], a: [i32; 3], b: [i32; 3]) -> bool {
    cell[0] == a[0].max(b[0]) && cell[1] == a[1].max(b[1]) && cell[2] == a[2].max(b[2])
}

fn shared_vertex_count(a: [u32; 3], b: [u32; 3]) -> usize {
    let mut shared = 0;
    for vertex in a {
        if b.contains(&vertex) {
            shared += 1;
        }
    }
    shared
}

struct InteriorHit {
    gap: f64,
}

/// Two triangles that share exactly one vertex `v`. They overlap beyond `v` iff the edge
/// opposite `v` in one of them crosses the other triangle; the edges at `v` meet the other
/// triangle at `v` itself, and testing them would report that touch, or round-off near it,
/// as a crossing (a flat fan, or a hinge just off flat).
fn fan_gap(fa: [u32; 3], fb: [u32; 3], a: [DVec3; 3], b: [DVec3; 3], eps: f64) -> Option<InteriorHit> {
    let ia = (0..3).find(|&i| fb.contains(&fa[i]))?;
    let ib = (0..3).find(|&i| fb[i] == fa[ia])?;
    let opposite = |t: [DVec3; 3], i: usize| (t[(i + 1) % 3], t[(i + 2) % 3]);
    let (p, q) = opposite(a, ia);
    let (r, s) = opposite(b, ib);
    let hits = [segment_interior_gap(p, q, b, eps), segment_interior_gap(r, s, a, eps)];
    let coplanar = hits.iter().all(Option::is_none)
        && (point_in_triangle_strict(p, b, eps)
            || point_in_triangle_strict(q, b, eps)
            || point_in_triangle_strict(r, a, eps)
            || point_in_triangle_strict(s, a, eps))
        && [p, q].iter().all(|x| (b[1] - b[0]).cross(b[2] - b[0]).normalize_or_zero().dot(*x - b[0]).abs() <= eps);
    hits.into_iter()
        .flatten()
        .min_by(|x, y| x.gap.total_cmp(&y.gap))
        .or(coplanar.then_some(InteriorHit { gap: 0.0 }))
}

/// Distance from an interior intersection point to the triangle boundary, in metres.
/// `None` when the triangles do not meet in their interiors.
fn intersection_gap(a: [DVec3; 3], b: [DVec3; 3], eps: f64) -> Option<InteriorHit> {
    let na = (a[1] - a[0]).cross(a[2] - a[0]);
    let nb = (b[1] - b[0]).cross(b[2] - b[0]);
    let da = [0, 1, 2].map(|i| na.dot(b[i] - a[0]));
    let db = [0, 1, 2].map(|i| nb.dot(a[i] - b[0]));
    let a_scale = na.length().max(eps);
    let b_scale = nb.length().max(eps);
    let a_coplanar = da.iter().all(|d| d.abs() <= 1e-9 * a_scale);
    let b_coplanar = db.iter().all(|d| d.abs() <= 1e-9 * b_scale);
    if a_coplanar && b_coplanar {
        return coplanar_overlap(a, b, eps).then_some(InteriorHit { gap: 0.0 });
    }
    if da.iter().all(|d| *d > eps) || da.iter().all(|d| *d < -eps) {
        return None;
    }
    if db.iter().all(|d| *d > eps) || db.iter().all(|d| *d < -eps) {
        return None;
    }
    let mut best: Option<InteriorHit> = None;
    for edge in [[a[0], a[1]], [a[1], a[2]], [a[2], a[0]]] {
        if let Some(hit) = segment_interior_gap(edge[0], edge[1], b, eps) {
            if best.as_ref().is_none_or(|kept| hit.gap < kept.gap) {
                best = Some(hit);
            }
        }
    }
    for edge in [[b[0], b[1]], [b[1], b[2]], [b[2], b[0]]] {
        if let Some(hit) = segment_interior_gap(edge[0], edge[1], a, eps) {
            if best.as_ref().is_none_or(|kept| hit.gap < kept.gap) {
                best = Some(hit);
            }
        }
    }
    best
}

fn segment_interior_gap(p: DVec3, q: DVec3, tri: [DVec3; 3], eps: f64) -> Option<InteriorHit> {
    let dir = q - p;
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    let scale = e1.length() * e2.length() * dir.length();
    if scale == 0.0 || det.abs() <= 1e-12 * scale {
        return None;
    }
    let inv = 1.0 / det;
    let tvec = p - tri[0];
    let u = tvec.dot(pv) * inv;
    let qv = tvec.cross(e1);
    let v = dir.dot(qv) * inv;
    let t = e2.dot(qv) * inv;
    let margin = (eps / scale.max(eps)).clamp(1e-8, 1e-3);
    if t > margin && t < 1.0 - margin && u > margin && v > margin && u + v < 1.0 - margin {
        let point = p + dir * t;
        Some(InteriorHit {
            gap: distance_to_boundary(point, tri),
        })
    } else {
        None
    }
}

fn distance_to_boundary(point: DVec3, tri: [DVec3; 3]) -> f64 {
    let mut gap = f64::MAX;
    for edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
        let side = edge[1] - edge[0];
        let len = side.length();
        if len == 0.0 {
            continue;
        }
        gap = gap.min((point - edge[0]).cross(side).length() / len);
    }
    gap
}

fn coplanar_overlap(a: [DVec3; 3], b: [DVec3; 3], eps: f64) -> bool {
    for vertex in b {
        if point_in_triangle_strict(vertex, a, eps) {
            return true;
        }
    }
    for vertex in a {
        if point_in_triangle_strict(vertex, b, eps) {
            return true;
        }
    }
    for ea in [[a[0], a[1]], [a[1], a[2]], [a[2], a[0]]] {
        for eb in [[b[0], b[1]], [b[1], b[2]], [b[2], b[0]]] {
            if segments_cross(ea[0], ea[1], eb[0], eb[1], eps) {
                return true;
            }
        }
    }
    false
}

fn point_in_triangle_strict(p: DVec3, tri: [DVec3; 3], eps: f64) -> bool {
    let n = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    let axis = if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() {
        0
    } else if n.y.abs() >= n.z.abs() {
        1
    } else {
        2
    };
    let project = |v: DVec3| -> [f64; 2] {
        match axis {
            0 => [v.y, v.z],
            1 => [v.x, v.z],
            _ => [v.x, v.y],
        }
    };
    let p = project(p);
    let t = tri.map(project);
    let bary = barycentric_2d(p, t);
    bary.iter().all(|w| *w > eps)
}

fn barycentric_2d(p: [f64; 2], t: [[f64; 2]; 3]) -> [f64; 3] {
    let den = (t[1][1] - t[2][1]) * (t[0][0] - t[2][0]) + (t[2][0] - t[1][0]) * (t[0][1] - t[2][1]);
    if den.abs() <= 1e-18 {
        return [0.0; 3];
    }
    let w0 =
        ((t[1][1] - t[2][1]) * (p[0] - t[2][0]) + (t[2][0] - t[1][0]) * (p[1] - t[2][1])) / den;
    let w1 =
        ((t[2][1] - t[0][1]) * (p[0] - t[2][0]) + (t[0][0] - t[2][0]) * (p[1] - t[2][1])) / den;
    [w0, w1, 1.0 - w0 - w1]
}

fn segments_cross(a: DVec3, b: DVec3, c: DVec3, d: DVec3, eps: f64) -> bool {
    let n = (b - a).cross(d - c);
    if n.length() <= eps {
        return false;
    }
    let volume = n.dot(c - a);
    if volume.abs() > eps * n.length() {
        return false;
    }
    let axis = if n.x.abs() >= n.y.abs() && n.x.abs() >= n.z.abs() {
        0
    } else if n.y.abs() >= n.z.abs() {
        1
    } else {
        2
    };
    let drop = |v: DVec3| -> [f64; 2] {
        match axis {
            0 => [v.y, v.z],
            1 => [v.x, v.z],
            _ => [v.x, v.y],
        }
    };
    proper_2d(drop(a), drop(b), drop(c), drop(d), eps)
}

fn proper_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], eps: f64) -> bool {
    let orient = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let ab_c = orient(a, b, c);
    let ab_d = orient(a, b, d);
    let cd_a = orient(c, d, a);
    let cd_b = orient(c, d, b);
    ab_c * ab_d < -eps && cd_a * cd_b < -eps
}

fn validate_components_and_volume(mesh: &TriMesh, scale: f64) -> Result<(), SimError> {
    let components = face_components(mesh);
    let mut shells = Vec::new();
    for faces in &components {
        let volume = component_volume(mesh, faces);
        if !volume.is_finite() || volume.abs() <= 1e-14 * scale * scale * scale {
            return Err(SimError::model_not_closed(
                "a mesh shell has zero signed volume",
            ));
        }
        shells.push((faces.clone(), volume));
    }
    let mut positive: Vec<usize> = shells
        .iter()
        .enumerate()
        .filter(|(_, (_, volume))| *volume > 0.0)
        .map(|(index, _)| index)
        .collect();
    if positive.is_empty() {
        return Err(SimError::model_not_closed(
            "mesh signed volume is non-positive; face winding is not the outward orientation",
        ));
    }
    positive.sort_by(|&i, &j| {
        shells[j]
            .1
            .partial_cmp(&shells[i].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (index, (faces, volume)) in shells.iter().enumerate() {
        if *volume > 0.0 {
            for other in &positive {
                if *other == index {
                    continue;
                }
                if shell_inside(mesh, faces, &shells[*other].0, scale) {
                    return Err(SimError::model_not_closed(
                        "an interior shell is wound as a solid instead of a cavity",
                    ));
                }
            }
            continue;
        }
        let host = positive
            .iter()
            .find(|outer| shell_inside(mesh, faces, &shells[**outer].0, scale));
        if host.is_none() {
            return Err(SimError::model_not_closed(
                "a cavity shell is not inside a positively oriented solid",
            ));
        }
    }
    let total: f64 = shells.iter().map(|(_, volume)| *volume).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(SimError::model_not_closed(
            "mesh signed volume is non-positive; face winding is not the outward orientation",
        ));
    }
    Ok(())
}

fn face_components(mesh: &TriMesh) -> Vec<Vec<usize>> {
    use std::collections::HashMap;
    let n = mesh.faces.len();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, face) in mesh.faces.iter().enumerate() {
        for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let key = if edge.0 < edge.1 {
                (edge.0, edge.1)
            } else {
                (edge.1, edge.0)
            };
            edge_faces.entry(key).or_default().push(i);
        }
    }
    let mut neighbor = vec![Vec::new(); n];
    for faces in edge_faces.values() {
        for a in 0..faces.len() {
            for b in (a + 1)..faces.len() {
                neighbor[faces[a]].push(faces[b]);
                neighbor[faces[b]].push(faces[a]);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut faces = Vec::new();
        seen[start] = true;
        while let Some(face) = stack.pop() {
            faces.push(face);
            for next in &neighbor[face] {
                if !seen[*next] {
                    seen[*next] = true;
                    stack.push(*next);
                }
            }
        }
        components.push(faces);
    }
    components
}

fn component_volume(mesh: &TriMesh, faces: &[usize]) -> f64 {
    let mut volume = 0.0;
    for &face in faces {
        let [a, b, c] = mesh.triangle(face);
        volume += a.dot(b.cross(c));
    }
    volume / 6.0
}

fn shell_inside(mesh: &TriMesh, inner: &[usize], outer: &[usize], scale: f64) -> bool {
    inner.iter().all(|&face| {
        let [a, b, c] = mesh.triangle(face);
        let sample = (a + b + c) * (1.0 / 3.0);
        point_inside_shell(mesh, outer, sample, scale)
    })
}

fn point_inside_shell(mesh: &TriMesh, faces: &[usize], point: DVec3, scale: f64) -> bool {
    let dirs = [
        DVec3::new(1.0, 0.137, 0.071),
        DVec3::new(0.083, 1.0, 0.191),
        DVec3::new(0.173, 0.059, 1.0),
    ];
    for dir in dirs {
        let mut hits = 0;
        let mut grazing = false;
        for &face in faces {
            match ray_hit(point, dir, mesh.triangle(face), 1e-9 * scale) {
                RayClass::Miss => {}
                RayClass::Hit => hits += 1,
                RayClass::Grazing => grazing = true,
            }
        }
        if !grazing {
            return hits % 2 == 1;
        }
    }
    false
}

enum RayClass {
    Miss,
    Hit,
    Grazing,
}

fn ray_hit(origin: DVec3, dir: DVec3, tri: [DVec3; 3], eps: f64) -> RayClass {
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let p = dir.cross(e2);
    let det = e1.dot(p);
    let scale = e1.length() * e2.length() * dir.length();
    if scale == 0.0 || det.abs() <= 1e-12 * scale {
        return RayClass::Grazing;
    }
    let inv = 1.0 / det;
    let tvec = origin - tri[0];
    let u = tvec.dot(p) * inv;
    let q = tvec.cross(e1);
    let v = dir.dot(q) * inv;
    let t = e2.dot(q) * inv;
    if t <= eps {
        return RayClass::Miss;
    }
    if u < -1e-8 || v < -1e-8 || u + v > 1.0 + 1e-8 {
        return RayClass::Miss;
    }
    if u < 1e-7 || v < 1e-7 || u + v > 1.0 - 1e-7 {
        RayClass::Grazing
    } else {
        RayClass::Hit
    }
}
