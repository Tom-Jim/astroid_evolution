//! Interior tetrahedral mesh of a closed triangle surface.
//! Boundary faces keep a map back to the input triangles. Failure is an error,
//! never a radial sampling of the surface.

use std::collections::HashMap;

use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::backend::hw_rt::HardwareMesh;
use crate::rtfp_solver::geometry::ray_query::unique_crossing_times;
use crate::scf_simulator::error::ScfError;

#[derive(Clone, Debug)]
pub struct BoundaryFace {
    pub vertices: [u32; 3],
    pub tet: usize,
    pub surface_face: usize,
    pub normal: DVec3,
    pub area: f64,
}

#[derive(Clone, Debug)]
pub struct TetMesh {
    pub vertices: Vec<DVec3>,
    pub tets: Vec<[u32; 4]>,
    pub neighbors: Vec<[i32; 4]>,
    pub boundary: Vec<BoundaryFace>,
    pub quad_points: Vec<DVec3>,
    pub quad_weights: Vec<f64>,
    pub quad_tet: Vec<usize>,
    pub volumes: Vec<f64>,
    pub surface_to_boundary: Vec<Vec<usize>>,
}

pub fn tetrahedralize(mesh: &TriMesh) -> Result<TetMesh, ScfError> {
    if let Some(tet) = as_single_tet(mesh) {
        return finish(mesh, tet.0, tet.1);
    }
    if let Some(built) = as_solid_box(mesh) {
        return finish(mesh, built.0, built.1);
    }
    let mut divisions = if mesh.faces.len() > 5_000 { 4 } else { 6 };
    loop {
        if let Some(built) = interior_lattice(mesh, divisions)? {
            return finish(mesh, built.0, built.1);
        }
        if divisions >= 8 {
            return Err(ScfError::model_unsupported(
                "tetrahedralization produced no positive-volume tetrahedra; the surface was not replaced by radial samples",
            ));
        }
        divisions += 2;
    }
}

pub fn tet_volume(vertices: &[DVec3], tet: [u32; 4]) -> f64 {
    let p = tet.map(|index| vertices[index as usize]);
    (p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])) / 6.0
}

pub fn integrate_density(mesh: &TetMesh, sample: impl Fn(DVec3) -> f64) -> (f64, DVec3, f64) {
    let mut mass = 0.0;
    let mut moment = DVec3::ZERO;
    let mut volume = 0.0;
    for (tet, &weight) in mesh.volumes.iter().enumerate() {
        let p = mesh.tets[tet].map(|index| mesh.vertices[index as usize]);
        let center = (p[0] + p[1] + p[2] + p[3]) * 0.25;
        let rho = sample(center);
        mass += rho * weight;
        moment = moment + center * (rho * weight);
        volume += weight;
    }
    let centroid = if mass.abs() > 0.0 {
        moment * (1.0 / mass)
    } else {
        DVec3::ZERO
    };
    (mass, centroid, volume)
}

fn finish(
    surface: &TriMesh,
    vertices: Vec<DVec3>,
    tets: Vec<[u32; 4]>,
) -> Result<TetMesh, ScfError> {
    if tets.is_empty() {
        return Err(ScfError::model_unsupported(
            "tetrahedralization produced an empty complex",
        ));
    }
    let mut volumes = Vec::with_capacity(tets.len());
    for tet in &tets {
        let volume = tet_volume(&vertices, *tet);
        if !(volume.is_finite() && volume > 0.0) {
            return Err(ScfError::model_not_closed(
                "a tetrahedron has non-positive volume",
            ));
        }
        volumes.push(volume);
    }
    let neighbors = adjacency(&tets);
    let boundary = boundary_faces(surface, &vertices, &tets, &neighbors);
    let mut surface_to_boundary = vec![Vec::<usize>::new(); surface.faces.len()];
    for (index, face) in boundary.iter().enumerate() {
        if face.surface_face < surface_to_boundary.len() {
            surface_to_boundary[face.surface_face].push(index);
        }
    }
    let (quad_points, quad_weights, quad_tet) = quadrature(&vertices, &tets, &volumes);
    Ok(TetMesh {
        vertices,
        tets,
        neighbors,
        boundary,
        quad_points,
        quad_weights,
        quad_tet,
        volumes,
        surface_to_boundary,
    })
}

fn as_single_tet(mesh: &TriMesh) -> Option<(Vec<DVec3>, Vec<[u32; 4]>)> {
    if mesh.vertices.len() != 4 || mesh.faces.len() != 4 {
        return None;
    }
    let tet = [0, 1, 2, 3];
    let volume = tet_volume(&mesh.vertices, tet);
    let ordered = if volume > 0.0 { tet } else { [0, 1, 3, 2] };
    Some((mesh.vertices.clone(), vec![ordered]))
}

fn as_solid_box(mesh: &TriMesh) -> Option<(Vec<DVec3>, Vec<[u32; 4]>)> {
    let (min, max) = bounds(&mesh.vertices);
    let extent = max - min;
    if extent.x <= 0.0 || extent.y <= 0.0 || extent.z <= 0.0 {
        return None;
    }
    let box_volume = extent.x * extent.y * extent.z;
    if (mesh.signed_volume() - box_volume).abs() > 1e-6 * box_volume {
        return None;
    }
    for face in 0..mesh.faces.len() {
        let normal = mesh.face_normal(face)?;
        let axis = normal.x.abs().max(normal.y.abs()).max(normal.z.abs());
        if (axis - 1.0).abs() > 1e-6 {
            return None;
        }
    }
    Some(structured_box(min, max, 2))
}

fn interior_lattice(
    mesh: &TriMesh,
    divisions: usize,
) -> Result<Option<(Vec<DVec3>, Vec<[u32; 4]>)>, ScfError> {
    let (min, max) = bounds(&mesh.vertices);
    let scene = HardwareMesh::build(mesh).map_err(|err| ScfError::model_not_closed(err.detail))?;
    let divisions = divisions.max(2);
    let mut nodes = Vec::new();
    for ix in 0..=divisions {
        for iy in 0..=divisions {
            for iz in 0..=divisions {
                let x = min.x + (max.x - min.x) * (ix as f64) / (divisions as f64);
                let y = min.y + (max.y - min.y) * (iy as f64) / (divisions as f64);
                let z = min.z + (max.z - min.z) * (iz as f64) / (divisions as f64);
                nodes.push(DVec3::new(x, y, z));
            }
        }
    }
    let dir = DVec3::new(1.0, 1.0e-3, -2.0e-3);
    let rays: Vec<(DVec3, DVec3)> = nodes.iter().map(|point| (*point, dir)).collect();
    let rows = scene
        .intersect_batch(&rays)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let mut inside = Vec::with_capacity(nodes.len());
    for row in &rows {
        let times: Vec<f64> = row.iter().map(|hit| hit.t).collect();
        inside.push(unique_crossing_times(&times).len() % 2 == 1);
    }
    let stride =
        |ix: usize, iy: usize, iz: usize| (ix * (divisions + 1) + iy) * (divisions + 1) + iz;
    let mut vertices = Vec::new();
    let mut index_of = vec![u32::MAX; nodes.len()];
    let mut tets = Vec::new();
    for ix in 0..divisions {
        for iy in 0..divisions {
            for iz in 0..divisions {
                let corners = [
                    stride(ix, iy, iz),
                    stride(ix + 1, iy, iz),
                    stride(ix + 1, iy + 1, iz),
                    stride(ix, iy + 1, iz),
                    stride(ix, iy, iz + 1),
                    stride(ix + 1, iy, iz + 1),
                    stride(ix + 1, iy + 1, iz + 1),
                    stride(ix, iy + 1, iz + 1),
                ];
                if !corners.iter().all(|&corner| inside[corner]) {
                    continue;
                }
                let mut local = [0u32; 8];
                for (slot, &corner) in corners.iter().enumerate() {
                    if index_of[corner] == u32::MAX {
                        index_of[corner] = vertices.len() as u32;
                        vertices.push(nodes[corner]);
                    }
                    local[slot] = index_of[corner];
                }
                for tet in box_tets(local) {
                    if tet_volume(&vertices, tet) > 0.0 {
                        tets.push(tet);
                    }
                }
            }
        }
    }
    if tets.is_empty() {
        return Ok(None);
    }
    Ok(Some((vertices, tets)))
}

fn structured_box(min: DVec3, max: DVec3, divisions: usize) -> (Vec<DVec3>, Vec<[u32; 4]>) {
    let mut vertices = Vec::new();
    let mut index = HashMap::<(usize, usize, usize), u32>::new();
    for ix in 0..=divisions {
        for iy in 0..=divisions {
            for iz in 0..=divisions {
                let point = DVec3::new(
                    min.x + (max.x - min.x) * (ix as f64) / (divisions as f64),
                    min.y + (max.y - min.y) * (iy as f64) / (divisions as f64),
                    min.z + (max.z - min.z) * (iz as f64) / (divisions as f64),
                );
                index.insert((ix, iy, iz), vertices.len() as u32);
                vertices.push(point);
            }
        }
    }
    let mut tets = Vec::new();
    for ix in 0..divisions {
        for iy in 0..divisions {
            for iz in 0..divisions {
                let local = [
                    index[&(ix, iy, iz)],
                    index[&(ix + 1, iy, iz)],
                    index[&(ix + 1, iy + 1, iz)],
                    index[&(ix, iy + 1, iz)],
                    index[&(ix, iy, iz + 1)],
                    index[&(ix + 1, iy, iz + 1)],
                    index[&(ix + 1, iy + 1, iz + 1)],
                    index[&(ix, iy + 1, iz + 1)],
                ];
                tets.extend(box_tets(local));
            }
        }
    }
    (vertices, tets)
}

fn box_tets(v: [u32; 8]) -> [[u32; 4]; 5] {
    [
        [v[0], v[1], v[2], v[5]],
        [v[0], v[2], v[3], v[7]],
        [v[0], v[2], v[7], v[5]],
        [v[0], v[5], v[7], v[4]],
        [v[2], v[5], v[6], v[7]],
    ]
}

fn adjacency(tets: &[[u32; 4]]) -> Vec<[i32; 4]> {
    let mut map: HashMap<(u32, u32, u32), (usize, usize)> = HashMap::new();
    let mut neighbors = vec![[-1; 4]; tets.len()];
    for (tet_index, tet) in tets.iter().enumerate() {
        for face in 0..4 {
            let mut ids = face_vertices(*tet, face);
            ids.sort_unstable();
            let key = (ids[0], ids[1], ids[2]);
            if let Some((other, other_face)) = map.remove(&key) {
                neighbors[tet_index][face] = other as i32;
                neighbors[other][other_face] = tet_index as i32;
            } else {
                map.insert(key, (tet_index, face));
            }
        }
    }
    neighbors
}

fn face_vertices(tet: [u32; 4], face: usize) -> [u32; 3] {
    match face {
        0 => [tet[1], tet[2], tet[3]],
        1 => [tet[0], tet[3], tet[2]],
        2 => [tet[0], tet[1], tet[3]],
        _ => [tet[0], tet[2], tet[1]],
    }
}

fn boundary_faces(
    surface: &TriMesh,
    vertices: &[DVec3],
    tets: &[[u32; 4]],
    neighbors: &[[i32; 4]],
) -> Vec<BoundaryFace> {
    let mut faces = Vec::new();
    for (tet_index, tet) in tets.iter().enumerate() {
        for local in 0..4 {
            if neighbors[tet_index][local] >= 0 {
                continue;
            }
            let ids = face_vertices(*tet, local);
            let p = ids.map(|index| vertices[index as usize]);
            let cross = (p[1] - p[0]).cross(p[2] - p[0]);
            let area = 0.5 * cross.length();
            if area == 0.0 {
                continue;
            }
            let normal = cross * (1.0 / cross.length());
            let center = (p[0] + p[1] + p[2]) * (1.0 / 3.0);
            let mut best = 0usize;
            let mut best_score = f64::MAX;
            for (surface_index, face) in surface.faces.iter().enumerate() {
                let q = face.map(|index| surface.vertices[index as usize]);
                let mid = (q[0] + q[1] + q[2]) * (1.0 / 3.0);
                let score = (mid - center).length_squared();
                if score < best_score {
                    best_score = score;
                    best = surface_index;
                }
            }
            faces.push(BoundaryFace {
                vertices: ids,
                tet: tet_index,
                surface_face: best,
                normal,
                area,
            });
        }
    }
    faces
}

fn quadrature(
    vertices: &[DVec3],
    tets: &[[u32; 4]],
    volumes: &[f64],
) -> (Vec<DVec3>, Vec<f64>, Vec<usize>) {
    let alpha = 0.585_410_196_624_968_5;
    let beta = 0.138_196_601_125_010_5;
    let bary = [
        [alpha, beta, beta, beta],
        [beta, alpha, beta, beta],
        [beta, beta, alpha, beta],
        [beta, beta, beta, alpha],
    ];
    let mut points = Vec::new();
    let mut weights = Vec::new();
    let mut owner = Vec::new();
    for (tet_index, tet) in tets.iter().enumerate() {
        let p = tet.map(|index| vertices[index as usize]);
        for coords in bary {
            let point = p[0] * coords[0] + p[1] * coords[1] + p[2] * coords[2] + p[3] * coords[3];
            points.push(point);
            weights.push(volumes[tet_index] / 4.0);
            owner.push(tet_index);
        }
    }
    (points, weights, owner)
}

fn bounds(vertices: &[DVec3]) -> (DVec3, DVec3) {
    let mut min = vertices[0];
    let mut max = vertices[0];
    for vertex in vertices {
        min.x = min.x.min(vertex.x);
        min.y = min.y.min(vertex.y);
        min.z = min.z.min(vertex.z);
        max.x = max.x.max(vertex.x);
        max.y = max.y.max(vertex.y);
        max.z = max.z.max(vertex.z);
    }
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_tets_recover_the_box_volume_and_map_the_boundary() {
        let mesh = TriMesh::unit_cube();
        let volume = tetrahedralize(&mesh).unwrap();
        let sum: f64 = volume.volumes.iter().sum();
        assert!((sum - mesh.signed_volume()).abs() < 1e-8, "{sum}");
        assert!(!volume.boundary.is_empty());
        assert!(
            volume
                .surface_to_boundary
                .iter()
                .any(|faces| !faces.is_empty())
        );
        assert_eq!(volume.quad_points.len(), volume.tets.len() * 4);
        let (_mass, centroid, _) = integrate_density(&volume, |_| 1.0);
        assert!(centroid.length() < 1e-8, "{centroid:?}");
    }

    #[test]
    fn one_tetrahedron_is_kept_as_one_element() {
        let mesh = TriMesh::right_tetrahedron();
        let volume = tetrahedralize(&mesh).unwrap();
        assert_eq!(volume.tets.len(), 1);
        assert!((volume.volumes[0] - mesh.signed_volume()).abs() < 1e-12);
    }

    #[test]
    fn a_concave_closed_mesh_still_tetrahedralizes() {
        let mesh = concave_prism();
        assert!(super::super::geometry::star_margin(&mesh, crate::common::math::DVec3::ZERO) < 0.2);
        let volume = tetrahedralize(&mesh).unwrap();
        let sum: f64 = volume.volumes.iter().sum();
        assert!(sum > 0.0);
        assert!(!volume.boundary.is_empty());
        assert!(!volume.neighbors.is_empty());
    }

    fn concave_prism() -> TriMesh {
        let z0 = 0.0;
        let z1 = 1.0;
        let footprint = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let mut vertices = Vec::new();
        for z in [z0, z1] {
            for corner in footprint {
                vertices.push(crate::common::math::DVec3::new(corner[0], corner[1], z));
            }
        }
        let mut faces = Vec::new();
        let n = 6u32;
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push([i, j, n + i]);
            faces.push([j, n + j, n + i]);
        }
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push([0 + j, 0 + i, 0]);
        }
        // Bottom and top are fans from the first vertex, which is valid for this concave polygon
        // only if the fan stays inside. Use an explicit triangulation instead.
        faces.clear();
        let bottom = [[0u32, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]];
        let top = [[6u32, 8, 7], [6, 9, 8], [6, 10, 9], [6, 11, 10]];
        faces.extend(bottom);
        faces.extend(top);
        for i in 0..n {
            let j = (i + 1) % n;
            faces.push([i, n + i, j]);
            faces.push([j, n + i, n + j]);
        }
        let mut volume = 0.0;
        for face in &faces {
            let a = vertices[face[0] as usize];
            let b = vertices[face[1] as usize];
            let c = vertices[face[2] as usize];
            volume += a.dot(b.cross(c));
        }
        if volume < 0.0 {
            for face in &mut faces {
                face.swap(1, 2);
            }
        }
        TriMesh::try_new(vertices, faces).expect("concave prism")
    }
}
