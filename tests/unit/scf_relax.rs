//! Unit tests of `scf::relax`.

use astroid_evolution::scf::socp::Kind;
use astroid_evolution::scf::body::Topology;
use astroid_evolution::scf::relax::*;

#[test]
fn subcritical_vertices_stay_locked() {
    let (topo, _) = Topology::icosphere(2);
    let nv = topo.n_vertices();
    let nf = topo.n_faces();
    let tri = topo.faces[0];
    let mut slack = vec![0.0; nf];
    slack[0] = 0.1;

    let protect = vec![Kind::Mobile; nv];
    let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
    assert!(slope[0]);
    assert_eq!(slope.iter().filter(|on| **on).count(), 1);
    for i in 0..nv {
        let on_face = tri.contains(&(i as u32));
        assert_eq!(
            kind[i] == Kind::Mobile,
            on_face,
            "vertex {i} mobile={} but on the supercritical face={on_face}",
            kind[i] == Kind::Mobile
        );
    }

    let mut protect = vec![Kind::Mobile; nv];
    protect[tri[0] as usize] = Kind::Locked;
    let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
    assert!(slope[0]);
    assert_eq!(kind[tri[0] as usize], Kind::Locked);
    assert_eq!(kind[tri[1] as usize], Kind::Mobile);
    assert_eq!(kind[tri[2] as usize], Kind::Mobile);

    let protect = vec![Kind::Locked; nv];
    let (kind, slope) = friction_lock(&topo, &protect, &slack, 1.0e-3);
    assert!(!slope[0]);
    assert!(kind.iter().all(|k| *k == Kind::Locked));
}
