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
        mesh.validate_solid()?;
        Ok(mesh)
    }

    /// Closed edges, consistent winding, non-degenerate faces, no self-intersection,
    /// positive solid volume, and inward-pointing cavity shells.
    pub fn validate_solid(&self) -> Result<(), CommonError> {
        let scale = bounding_scale(&self.vertices).max(1e-12);
        let area_tol = 1e-14 * scale * scale;
        for index in 0..self.faces.len() {
            let [a, b, c] = self.triangle(index);
            let area2 = (b - a).cross(c - a).length();
            if !area2.is_finite() || area2 <= area_tol {
                return Err(CommonError::model_not_closed(format!(
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

fn validate_edges(faces: &[[u32; 3]]) -> Result<(), CommonError> {
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
            return Err(CommonError::model_not_closed(format!(
                "edge {lo}-{hi} is wound the same way on both faces"
            )));
        }
        return Err(CommonError::model_not_closed(format!(
            "edge {lo}-{hi} is not a closed manifold edge"
        )));
    }
    Ok(())
}

fn validate_no_self_intersection(mesh: &TriMesh, scale: f64) -> Result<(), CommonError> {
    use std::collections::{HashMap, HashSet};
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
    let cells = (mesh.faces.len() as f64).cbrt().ceil().clamp(4.0, 48.0);
    let cell = extent / cells;
    let index_of = |value: f64, origin: f64| -> i32 { ((value - origin) / cell).floor() as i32 };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
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
        let i0 = [
            index_of(lo[0], min.x),
            index_of(lo[1], min.y),
            index_of(lo[2], min.z),
        ];
        let i1 = [
            index_of(hi[0], min.x),
            index_of(hi[1], min.y),
            index_of(hi[2], min.z),
        ];
        let span =
            (i1[0] - i0[0] + 1) as i64 * (i1[1] - i0[1] + 1) as i64 * (i1[2] - i0[2] + 1) as i64;
        if span > 64 {
            grid.entry((i32::MIN, i32::MIN, i32::MIN))
                .or_default()
                .push(i);
            continue;
        }
        for ix in i0[0]..=i1[0] {
            for iy in i0[1]..=i1[1] {
                for iz in i0[2]..=i1[2] {
                    grid.entry((ix, iy, iz)).or_default().push(i);
                }
            }
        }
    }
    let mut seen = HashSet::new();
    let mut check = |i: usize, j: usize| -> Result<(), CommonError> {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        if lo == hi || !seen.insert((lo, hi)) {
            return Ok(());
        }
        if share_edge(mesh.faces[lo], mesh.faces[hi]) {
            return Ok(());
        }
        if let Some(hit) = intersection_gap(mesh.triangle(lo), mesh.triangle(hi), eps) {
            return Err(CommonError::model_not_closed(format!(
                "triangles {lo} and {hi} intersect (gap {:.4} m)",
                hit.gap
            )));
        }
        Ok(())
    };
    let oversized = grid
        .get(&(i32::MIN, i32::MIN, i32::MIN))
        .cloned()
        .unwrap_or_default();
    for face in 0..mesh.faces.len() {
        for &other in &oversized {
            check(face, other)?;
        }
    }
    for bucket in grid.values() {
        for a in 0..bucket.len() {
            for b in (a + 1)..bucket.len() {
                check(bucket[a], bucket[b])?;
            }
        }
    }
    Ok(())
}

fn share_edge(a: [u32; 3], b: [u32; 3]) -> bool {
    shared_vertex_count(a, b) >= 2
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

fn validate_components_and_volume(mesh: &TriMesh, scale: f64) -> Result<(), CommonError> {
    let components = face_components(mesh);
    let mut shells = Vec::new();
    for faces in &components {
        let volume = component_volume(mesh, faces);
        if !volume.is_finite() || volume.abs() <= 1e-14 * scale * scale * scale {
            return Err(CommonError::model_not_closed(
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
        return Err(CommonError::model_not_closed(
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
                    return Err(CommonError::model_not_closed(
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
            return Err(CommonError::model_not_closed(
                "a cavity shell is not inside a positively oriented solid",
            ));
        }
    }
    let total: f64 = shells.iter().map(|(_, volume)| *volume).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(CommonError::model_not_closed(
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

/// How far radial-cone alignment moved the imported vertices.
#[derive(Clone, Copy, Debug)]
pub struct RadialAlignReport {
    /// Longest tangential displacement, in the same units as the mesh.
    pub max_offset: f64,
    /// Force passes that were applied. Zero when no cones overlapped.
    pub passes: usize,
}

/// Separate radial cones of triangles that share one vertex and no edge.
///
/// The origin is the coordinate origin. Vertices that sit in a crossed pair are
/// drawn toward their neighbours, keeping radius, vertex count and face count.
/// Afterwards a positive radius field on these directions is an embedded
/// polyhedron, which is what §10.4.4 step 8b resampling requires. A convex mesh
/// is left unchanged.
pub fn align_radial_cones(
    vertices: &mut [DVec3],
    faces: &[[u32; 3]],
) -> Result<RadialAlignReport, CommonError> {
    for face in faces {
        if face.iter().any(|&i| i as usize >= vertices.len()) {
            return Err(CommonError::model_not_closed(
                "mesh face index is outside the vertex array",
            ));
        }
    }
    let n = vertices.len();
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (face, tri) in faces.iter().enumerate() {
        for vertex in tri {
            incident[*vertex as usize].push(face);
        }
    }
    let original = vertices.to_vec();
    let mut passes = 0usize;
    let mut bad = usize::MAX;
    for _ in 0..120 {
        let found = relax_crossed_vertices(vertices, faces, &incident);
        bad = found;
        if bad == 0 {
            break;
        }
        passes += 1;
    }
    let max_offset = vertices
        .iter()
        .zip(&original)
        .map(|(now, was)| (*now - *was).length())
        .fold(0.0, f64::max);
    if bad > 0 {
        return Err(CommonError::model_not_closed(format!(
            "radial cones still overlap on {bad} face pairs after alignment (max offset {max_offset:.3} m)"
        )));
    }
    Ok(RadialAlignReport { max_offset, passes })
}

/// Pull vertices of overlapping cones toward their neighbours. A global push
/// fights itself; this only moves vertices that sit in a crossed pair.
fn relax_crossed_vertices(
    vertices: &mut [DVec3],
    faces: &[[u32; 3]],
    incident: &[Vec<usize>],
) -> usize {
    let mut dir: Vec<DVec3> = vertices
        .iter()
        .map(|v| v.try_normalize().unwrap_or(DVec3::new(0.0, 0.0, 1.0)))
        .collect();
    let radius: Vec<f64> = vertices.iter().map(|v| v.length()).collect();
    let mut bad_vertex = vec![false; vertices.len()];
    let mut bad = 0usize;
    for (s, faces_at) in incident.iter().enumerate() {
        if faces_at.len() < 2 {
            continue;
        }
        let sd = dir[s];
        let tmp = if sd.x.abs() < 0.9 {
            DVec3::new(1.0, 0.0, 0.0)
        } else {
            DVec3::new(0.0, 1.0, 0.0)
        };
        let Some(e1) = sd.cross(tmp).try_normalize() else {
            continue;
        };
        let Some(e2) = sd.cross(e1).try_normalize() else {
            continue;
        };
        for i in 0..faces_at.len() {
            for j in (i + 1)..faces_at.len() {
                let fa = faces[faces_at[i]];
                let fb = faces[faces_at[j]];
                if shared_vertex_count(fa, fb) != 1 {
                    continue;
                }
                let mut crossed = false;
                for e in 0..3 {
                    let a = tangent_of(dir[fa[e] as usize], e1, e2);
                    let b = tangent_of(dir[fa[(e + 1) % 3] as usize], e1, e2);
                    for g in 0..3 {
                        let c = tangent_of(dir[fb[g] as usize], e1, e2);
                        let d = tangent_of(dir[fb[(g + 1) % 3] as usize], e1, e2);
                        if segments_cross_2d(a, b, c, d) {
                            crossed = true;
                        }
                    }
                }
                if crossed {
                    bad += 1;
                    for v in fa.into_iter().chain(fb) {
                        bad_vertex[v as usize] = true;
                    }
                }
            }
        }
    }
    if bad == 0 {
        return 0;
    }
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
    for face in faces {
        for e in 0..3 {
            let a = face[e] as usize;
            let b = face[(e + 1) % 3] as usize;
            if !neighbors[a].contains(&b) {
                neighbors[a].push(b);
            }
            if !neighbors[b].contains(&a) {
                neighbors[b].push(a);
            }
        }
    }
    const KEEP: f64 = 0.8;
    for i in 0..vertices.len() {
        if !bad_vertex[i] || neighbors[i].is_empty() {
            continue;
        }
        let mut acc = DVec3::ZERO;
        for &nb in &neighbors[i] {
            acc = acc + dir[nb];
        }
        let Some(mean) = (acc * (1.0 / neighbors[i].len() as f64)).try_normalize() else {
            continue;
        };
        let Some(next) = (dir[i] * KEEP + mean * (1.0 - KEEP)).try_normalize() else {
            continue;
        };
        dir[i] = next;
        vertices[i] = next * radius[i];
    }
    bad
}

fn tangent_of(dir: DVec3, e1: DVec3, e2: DVec3) -> (f64, f64) {
    (dir.dot(e1), dir.dot(e2))
}

fn segments_cross_2d(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let ab_c = orient2(a, b, c);
    let ab_d = orient2(a, b, d);
    let cd_a = orient2(c, d, a);
    let cd_b = orient2(c, d, b);
    ab_c * ab_d < 0.0 && cd_a * cd_b < 0.0
}

fn orient2(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

#[cfg(test)]
mod solid_rules {
    use super::TriMesh;
    use crate::common::math::DVec3;

    #[test]
    fn a_convex_mesh_needs_no_radial_alignment() {
        let cube = TriMesh::unit_cube();
        let mut vertices = cube.vertices.clone();
        let report = super::align_radial_cones(&mut vertices, &cube.faces).unwrap();
        assert_eq!(report.passes, 0);
        assert!(report.max_offset < 1e-12);
    }

    /// `output/out.obj` is a closed mesh whose radial cones overlap, so a later
    /// radius resample self-intersects. Alignment must make every positive radius
    /// field on those directions an embedded polyhedron. Release only: the mesh
    /// has ~2e5 faces.
    #[cfg(not(debug_assertions))]
    #[test]
    fn radial_alignment_makes_equal_radii_intersection_free() {
        let path = std::path::Path::new("output/out.obj");
        if !path.exists() {
            return;
        }
        let text = std::fs::read_to_string(path).unwrap();
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for line in text.lines() {
            let mut parts = line.split_whitespace();
            match parts.next() {
                Some("v") => {
                    let x: f64 = parts.next().unwrap().parse().unwrap();
                    let y: f64 = parts.next().unwrap().parse().unwrap();
                    let z: f64 = parts.next().unwrap().parse().unwrap();
                    vertices.push(DVec3::new(x, y, z));
                }
                Some("f") => {
                    let ids: Vec<u32> = parts
                        .map(|token| token.split('/').next().unwrap().parse::<i64>().unwrap())
                        .map(|id| (id - 1) as u32)
                        .collect();
                    if ids.len() == 3 {
                        faces.push([ids[0], ids[1], ids[2]]);
                    }
                }
                _ => {}
            }
        }
        let n = vertices.len();
        let origin =
            vertices.iter().copied().fold(DVec3::ZERO, |acc, v| acc + v) * (1.0 / n as f64);
        for v in &mut vertices {
            *v = *v - origin;
        }
        let before = TriMesh::try_new(vertices.clone(), faces.clone())
            .expect("centred import")
            .signed_volume();
        let report = super::align_radial_cones(&mut vertices, &faces).unwrap();
        let after = TriMesh::try_new(vertices.clone(), faces.clone())
            .expect("aligned mesh")
            .signed_volume();
        let mean_r = vertices.iter().map(|v| v.length()).sum::<f64>() / n as f64;
        let sphere: Vec<DVec3> = vertices
            .iter()
            .map(|v| v.try_normalize().unwrap() * mean_r)
            .collect();
        TriMesh::try_new(sphere, faces).unwrap_or_else(|err| {
            panic!("equal radius after alignment: {err}");
        });
        eprintln!(
            "radial align passes {} max offset {:.4} m volume {:.6}",
            report.passes,
            report.max_offset,
            after / before
        );
        assert!(report.max_offset < 20.0, "{}", report.max_offset);
        assert!((after - before).abs() / before < 0.01, "{}", after / before);
    }

    #[test]
    fn reference_solids_are_closed() {
        TriMesh::unit_cube();
        TriMesh::right_tetrahedron();
        let sphere = TriMesh::uv_sphere(4, 1.0);
        assert!(sphere.signed_volume() > 0.0);
    }

    #[test]
    fn an_open_edge_is_rejected() {
        let cube = TriMesh::unit_cube();
        let faces = cube.faces[..cube.faces.len() - 1].to_vec();
        let err = TriMesh::try_new(cube.vertices, faces).unwrap_err();
        assert!(
            err.detail.contains("not a closed manifold edge"),
            "{}",
            err.detail
        );
    }

    #[test]
    fn a_face_wound_the_same_way_is_rejected() {
        let cube = TriMesh::unit_cube();
        let mut faces = cube.faces.clone();
        faces[0] = [faces[0][0], faces[0][2], faces[0][1]];
        let err = TriMesh::try_new(cube.vertices, faces).unwrap_err();
        assert!(err.detail.contains("wound the same way"), "{}", err.detail);
    }

    #[test]
    fn a_degenerate_triangle_is_rejected() {
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
        let err = TriMesh::try_new(vertices, faces).unwrap_err();
        assert!(err.detail.contains("degenerate triangle"), "{}", err.detail);
    }

    #[test]
    fn crossing_shells_are_rejected() {
        let (mut vertices, mut faces) = shifted_tet(DVec3::ZERO, 0);
        let (extra, extra_faces) = shifted_tet(DVec3::new(0.25, 0.25, -0.15), 4);
        vertices.extend(extra);
        faces.extend(extra_faces);
        let err = TriMesh::try_new(vertices, faces).unwrap_err();
        assert!(err.detail.contains("intersect"), "{}", err.detail);
    }

    #[test]
    fn an_inward_cavity_inside_a_solid_is_accepted() {
        let (mut vertices, mut faces) = cube_shell(1.0, 0, false);
        let (inner, inner_faces) = cube_shell(0.4, 8, true);
        vertices.extend(inner);
        faces.extend(inner_faces);
        let mesh = TriMesh::try_new(vertices, faces).expect("cavity");
        assert!(mesh.signed_volume() > 0.0);
    }

    #[test]
    fn a_positive_inner_shell_is_rejected() {
        let (mut vertices, mut faces) = cube_shell(1.0, 0, false);
        let (inner, inner_faces) = cube_shell(0.4, 8, false);
        vertices.extend(inner);
        faces.extend(inner_faces);
        let err = TriMesh::try_new(vertices, faces).unwrap_err();
        assert!(err.detail.contains("wound as a solid"), "{}", err.detail);
    }

    #[test]
    fn a_cavity_outside_the_solid_is_rejected() {
        let (mut vertices, mut faces) = cube_shell(1.0, 0, false);
        let (mut inner, inner_faces) = cube_shell(0.4, 8, true);
        for vertex in &mut inner {
            *vertex = *vertex + DVec3::new(3.0, 0.0, 0.0);
        }
        vertices.extend(inner);
        faces.extend(inner_faces);
        let err = TriMesh::try_new(vertices, faces).unwrap_err();
        assert!(
            err.detail
                .contains("not inside a positively oriented solid"),
            "{}",
            err.detail
        );
    }

    fn cube_shell(scale: f64, offset: u32, inward: bool) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let cube = TriMesh::unit_cube();
        let vertices = cube.vertices.into_iter().map(|v| v * scale).collect();
        let faces = cube
            .faces
            .into_iter()
            .map(|face| {
                let shifted = [face[0] + offset, face[1] + offset, face[2] + offset];
                if inward {
                    [shifted[0], shifted[2], shifted[1]]
                } else {
                    shifted
                }
            })
            .collect();
        (vertices, faces)
    }

    fn shifted_tet(shift: DVec3, offset: u32) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let tet = TriMesh::right_tetrahedron();
        let vertices = tet.vertices.into_iter().map(|v| v + shift).collect();
        let faces = tet
            .faces
            .into_iter()
            .map(|face| [face[0] + offset, face[1] + offset, face[2] + offset])
            .collect();
        (vertices, faces)
    }
}
