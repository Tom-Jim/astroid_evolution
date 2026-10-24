//! Unit tests of `scf::body` (§8.0, G4).

use std::f64::consts::PI;
use std::sync::Arc;

use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::*;

#[test]
fn icosphere_is_a_closed_outward_sphere() {
    let (t, dirs) = Topology::icosphere(2);
    assert_eq!((t.n_vertices(), t.n_faces()), (162, 320));
    assert!(t.face_neighbors.iter().all(|n| n.iter().all(|&x| x != usize::MAX)));
    let body = Body::new(Arc::new(t), dirs.iter().map(|d| *d * 10.0).collect(), 1.0, vec![], vec![0.0; 162])
        .unwrap();
    let v = body.mesh().unwrap().signed_volume();
    assert!(v > 0.0 && (v / (4.0 / 3.0 * PI * 1000.0) - 1.0).abs() < 0.05);
    // Outward vertex normals on a sphere point along the position.
    assert!(body.vertex_normals().iter().zip(&body.positions).all(|(n, x)| n.dot(x.normalize()) > 0.99));
}

/// `A_i = ∂Vol/∂X_i` (G4): a normal increment `d_i` changes the volume by `|A_i| d_i`.
#[test]
fn area_vectors_are_the_volume_gradient() {
    let cube = TriMesh::unit_cube();
    let (topo, x, _) = Body::from_mesh(&cube.vertices, cube.faces.clone()).unwrap();
    let body = Body::new(topo.clone(), x.clone(), 1.0, vec![], vec![0.0; x.len()]).unwrap();
    let a = body.area_vectors();
    let vol = |p: Vec<DVec3>| TriMesh::try_new(p, topo.faces.clone()).unwrap().signed_volume();
    let h = 1e-6;
    for i in 0..x.len() {
        let mut p = x.clone();
        p[i] += a[i].normalize() * h;
        let fd = (vol(p) - vol(x.clone())) / h;
        assert!((fd - a[i].length()).abs() < 1e-6, "vertex {i}: {fd} vs {}", a[i].length());
    }
    let total: f64 = body.vertex_areas().iter().sum();
    assert!((total - 24.0).abs() < 1e-12, "{total}");
}

#[test]
fn an_imported_mesh_keeps_every_vertex_and_face() {
    let cube = TriMesh::unit_cube();
    let shift = DVec3::new(3.0, -2.0, 5.0);
    let moved: Vec<DVec3> = cube.vertices.iter().map(|v| *v + shift).collect();
    let (topo, x, origin) = Body::from_mesh(&moved, cube.faces.clone()).unwrap();
    assert_eq!((topo.n_vertices(), topo.n_faces()), (cube.vertices.len(), cube.faces.len()));
    let centre = cube.vertices.iter().fold(DVec3::ZERO, |s, v| s + *v) / 8.0;
    assert!((origin - (centre + shift)).length() < 1e-12);
    for (i, v) in moved.iter().enumerate() {
        assert!((x[i] + origin - *v).length() < 1e-12, "{i}");
    }
}
