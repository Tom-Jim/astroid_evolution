//! Unit tests of `common::geometry::tri_mesh`, moved out of `src/common/geometry/tri_mesh.rs`.

mod ctx {
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use rayon::prelude::*;
}
use ctx::*;

use astroid_evolution::common::geometry::tri_mesh::TriMesh;
use astroid_evolution::common::math::DVec3;

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
