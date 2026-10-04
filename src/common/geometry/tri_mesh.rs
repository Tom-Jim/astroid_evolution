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
        if triangles_intersect(mesh.triangle(lo), mesh.triangle(hi), eps) {
            return Err(CommonError::model_not_closed(format!(
                "triangles {lo} and {hi} intersect"
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
    let mut shared = 0;
    for vertex in a {
        if b.contains(&vertex) {
            shared += 1;
        }
    }
    shared >= 2
}

fn triangles_intersect(a: [DVec3; 3], b: [DVec3; 3], eps: f64) -> bool {
    let na = (a[1] - a[0]).cross(a[2] - a[0]);
    let nb = (b[1] - b[0]).cross(b[2] - b[0]);
    let da = [0, 1, 2].map(|i| na.dot(b[i] - a[0]));
    let db = [0, 1, 2].map(|i| nb.dot(a[i] - b[0]));
    let a_scale = na.length().max(eps);
    let b_scale = nb.length().max(eps);
    let a_coplanar = da.iter().all(|d| d.abs() <= 1e-9 * a_scale);
    let b_coplanar = db.iter().all(|d| d.abs() <= 1e-9 * b_scale);
    if a_coplanar && b_coplanar {
        return coplanar_overlap(a, b, eps);
    }
    if da.iter().all(|d| *d > eps) || da.iter().all(|d| *d < -eps) {
        return false;
    }
    if db.iter().all(|d| *d > eps) || db.iter().all(|d| *d < -eps) {
        return false;
    }
    for edge in [[a[0], a[1]], [a[1], a[2]], [a[2], a[0]]] {
        if segment_hits_interior(edge[0], edge[1], b, eps) {
            return true;
        }
    }
    for edge in [[b[0], b[1]], [b[1], b[2]], [b[2], b[0]]] {
        if segment_hits_interior(edge[0], edge[1], a, eps) {
            return true;
        }
    }
    false
}

fn segment_hits_interior(p: DVec3, q: DVec3, tri: [DVec3; 3], eps: f64) -> bool {
    let dir = q - p;
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    let scale = e1.length() * e2.length() * dir.length();
    if scale == 0.0 || det.abs() <= 1e-12 * scale {
        return false;
    }
    let inv = 1.0 / det;
    let tvec = p - tri[0];
    let u = tvec.dot(pv) * inv;
    let qv = tvec.cross(e1);
    let v = dir.dot(qv) * inv;
    let t = e2.dot(qv) * inv;
    let margin = (eps / scale.max(eps)).clamp(1e-8, 1e-3);
    t > margin && t < 1.0 - margin && u > margin && v > margin && u + v < 1.0 - margin
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

#[cfg(test)]
mod solid_rules {
    use super::TriMesh;
    use crate::common::math::DVec3;

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
