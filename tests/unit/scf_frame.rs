//! Unit tests of `scf::frame`, moved out of `src/scf/frame.rs`.

mod ctx {
    pub use astroid_evolution::common::density::CauchyKernel;
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::scf::body::Body;
    pub use astroid_evolution::scf::forward::{Forward, Moments, inertia_matrix};
    pub use astroid_evolution::scf::params::GeometryParams;
    pub use astroid_evolution::scf::transport::{DirIndex, interp_vertex_field};
    pub use nalgebra::{Matrix3, SymmetricEigen, Vector3};
}
use ctx::*;

use astroid_evolution::scf::frame::*;
use std::sync::Arc;

use astroid_evolution::scf::body::Topology;

fn prm() -> GeometryParams {
    GeometryParams {
        c0: 0.2,
        eps_n: 0.01,
        eps_mass: 1e-6,
        trust0: 0.01,
        trust_min: 1e-5,
        trust_shrink: 0.5,
        varrho: 1e3,
        sh_fit_tol: 1e-2,
        eps_frame: 1e-3,
        eps_deg: 1e-6,
    }
}

#[test]
fn tilted_offset_ellipsoid_is_brought_to_the_standard_frame() {
    let topo = Arc::new(Topology::icosphere(4));
    let axes = [0.6, 0.8, 1.0];
    // Ellipsoid whose long axis is e_x and short axis e_z, i.e. a wrong frame.
    let radius = Body::ellipsoid(topo.clone(), axes);
    let nv = topo.n_vertices();
    let body = Body::new(topo.clone(), radius, 1000.0, vec![], vec![1.0; nv]).unwrap();
    // The frame is wrong: I_zz is the smallest moment.
    let g = 1.0e-9;
    let (out, rep, _) = realign(&body, None, g, &prm(), false).unwrap();
    assert!(rep.applied);
    assert!(rep.after.ordering < 1e-3, "{:?}", rep.after);
    assert!(rep.after.off_diagonal < 1e-3, "{:?}", rep.after);
    assert!(rep.after.centroid < 1e-3, "{:?}", rep.after);
    // The new long axis (max R) is along e_x? No: e_z is the max-inertia (short body) axis,
    // so the shortest radius must now lie on z and the longest on x or y.
    let zi = (0..nv)
        .max_by(|a, b| topo.dirs[*a].z.partial_cmp(&topo.dirs[*b].z).unwrap())
        .unwrap();
    assert!((out.radius[zi] - 0.6).abs() < 0.03, "{}", out.radius[zi]);
    assert!(rep.mass_change.abs() < 5e-3);
}

#[test]
fn kernel_centres_follow_the_mesh() {
    let topo = Arc::new(Topology::icosphere(4));
    let nv = topo.n_vertices();
    let radius = Body::ellipsoid(topo.clone(), [1.0, 0.8, 0.6]);
    let k = CauchyKernel::try_new(DVec3::new(0.3, 0.1, 0.2), 800.0, 0.3, 2.0).unwrap();
    let body = Body::new(topo, radius, 1000.0, vec![k], vec![0.0; nv]).unwrap();
    let g = 1.0e-9;
    let (out, rep, _) = realign(&body, None, g, &prm(), false).unwrap();
    assert!(rep.applied);
    assert!(rep.after.centroid < 1e-3, "{:?}", rep.after);
    assert!(rep.after.off_diagonal < 1e-3, "{:?}", rep.after);
    // The kernel moved with the rigid map x' = Q (x - c) (a rigid map preserves distances
    // to the origin only up to the translation).
    let c = rep.translation;
    let q = rep.rotation;
    let old = DVec3::new(0.3, 0.1, 0.2) - c;
    let expect = DVec3::new(
        q[0][0] * old.x + q[0][1] * old.y + q[0][2] * old.z,
        q[1][0] * old.x + q[1][1] * old.y + q[1][2] * old.z,
        q[2][0] * old.x + q[2][1] * old.y + q[2][2] * old.z,
    );
    let got = out.kernels[0].center;
    assert!((got - expect).length() < 1e-9, "{got:?} vs {expect:?}");
}
