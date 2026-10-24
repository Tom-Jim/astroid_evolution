//! Unit tests of `scf::frame`: the realignment of step 8b is a rigid motion (G8).

use astroid_evolution::common::density::CauchyKernel;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::scf::body::Body;
use astroid_evolution::scf::frame::*;
use astroid_evolution::scf::params::GeometryParams;

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
        theta_g: 0.4,
    }
}

fn ellipsoid(axes: [f64; 3], kernels: Vec<CauchyKernel>) -> Body {
    let (topo, x) = Body::ellipsoid(4, axes);
    let rho_s = x.iter().map(|p| 1.0 + 0.5 * p.x).collect();
    Body::new(topo, x, 1000.0, kernels, rho_s).unwrap()
}

#[test]
fn tilted_offset_ellipsoid_is_brought_to_the_standard_frame() {
    // Long axis e_x, short axis e_z: I_zz is the smallest moment, a wrong frame.
    let body = ellipsoid([0.6, 0.8, 1.0], vec![]);
    let (out, rep, _) = realign(&body, None, 1.0e-9, &prm(), false).unwrap();
    assert!(rep.applied);
    assert!(rep.after.ordering < 1e-3 && rep.after.off_diagonal < 1e-3 && rep.after.centroid < 1e-3);
    // The short axis (0.6) now lies on z; the motion is rigid, so ρ_s stays on its vertex.
    let top = out.positions.iter().map(|p| p.z.abs()).fold(0.0, f64::max);
    assert!((top - 0.6).abs() < 1e-3, "{top}");
    assert_eq!(out.rho_s, body.rho_s);
    assert!(rep.mass_change.abs() < 1e-9);
}

#[test]
fn undo_frame_inverts_the_realignment_exactly() {
    let k = CauchyKernel::try_new(DVec3::new(0.2, -0.1, 0.15), 500.0, 0.4, 2.0).unwrap();
    let body = ellipsoid([0.6, 1.0, 0.8], vec![k]);
    let (moved, rep, _) = realign(&body, None, 1.0e-9, &prm(), false).unwrap();
    assert!(rep.applied && rep.angle_deg > 45.0, "{}", rep.angle_deg);
    let back = undo_frame(&moved, rep.rotation, rep.translation).unwrap();
    let worst = (0..body.positions.len())
        .map(|i| (back.positions[i] - body.positions[i]).length())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "position error {worst}");
    assert!((back.kernels[0].center - k.center).length() < 1e-12);
    let ds = back.kernels[0].metric_mat() - k.metric_mat();
    assert!(ds.to_cols_array().iter().all(|v| v.abs() < 1e-9));
    let same = undo_frame(&body, astroid_evolution::scf::evolve::IDENTITY, DVec3::ZERO).unwrap();
    assert_eq!(same.positions, body.positions);
}
