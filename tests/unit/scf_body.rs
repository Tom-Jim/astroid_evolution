//! Unit tests of `scf::body`, moved out of `src/scf/body.rs`.

mod ctx {
    pub use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::geometry::{TriMesh, align_radial_cones};
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rtfp::mesh::BodyMesh;
    pub use std::collections::HashMap;
    pub use std::sync::Arc;
}
use ctx::*;

use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::scf::body::*;
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
