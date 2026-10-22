//! Unit tests of `scf::cores`, moved out of `src/scf/cores.rs`.

mod ctx {
    pub use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
    pub use astroid_evolution::common::error::SimError;
    pub use astroid_evolution::common::math::{DMat3, DVec3, mat_sym};
    pub use nalgebra::{DMatrix, DVector, Matrix3};
    pub use rayon::prelude::*;
}
use ctx::*;

use astroid_evolution::scf::cores::*;

fn params() -> CoreParams {
    CoreParams {
        density_mode: "lagrangian_cores".into(),
        core_shape: "anisotropic".into(),
        eps_xi: 1e-3,
        eps_rho: 1e-3,
        gn_max: 30,
    }
}

#[test]
fn affine_flow_is_reproduced_exactly() {
    // v = a + L x with tr L = 0: the transported kernel is again a Cauchy kernel.
    let l = DMat3::from_cols(
        DVec3::new(0.1, 0.3, -0.2),
        DVec3::new(-0.4, 0.05, 0.1),
        DVec3::new(0.2, -0.1, -0.15),
    );
    let a = DVec3::new(0.3, -0.1, 0.2);
    let v = move |p: &[DVec3]| p.iter().map(|x| a + l * *x).collect::<Vec<_>>();
    let k = CauchyKernel::try_anisotropic(
        DVec3::new(0.1, 0.2, -0.3),
        2.0,
        [1.3, 0.9, 1.1, 0.2, -0.1, 0.05],
        1.3,
    )
    .unwrap();
    let tau = 0.05;
    let mut rep = CoreReport::default();
    let out = predict(&[k], tau, &v, &|_| true, 1e-4, &params(), &mut rep);
    let s_ratio = out[0].metric_mat().determinant() / k.metric_mat().determinant();
    assert!((s_ratio - 1.0).abs() < 1e-10, "det ratio {s_ratio}");
    // Check against the exact flow map of the affine field at random points.
    let flow = |x: DVec3, t: f64| {
        let n = 400;
        let h = t / n as f64;
        let mut y = x;
        for _ in 0..n {
            let k1 = a + l * y;
            let k2 = a + l * (y + k1 * (h / 2.0));
            let k3 = a + l * (y + k2 * (h / 2.0));
            let k4 = a + l * (y + k3 * h);
            y += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
        }
        y
    };
    let old = CauchyDensity::try_new(0.0, vec![k]).unwrap();
    let new = CauchyDensity::try_new(0.0, out).unwrap();
    for x in [
        DVec3::new(0.3, 0.1, 0.0),
        DVec3::new(-0.5, 0.4, 0.2),
        DVec3::new(0.0, -0.3, 0.6),
    ] {
        let back = flow(x, -tau);
        let e = (new.rho(x) - old.rho(back)).abs() / old.rho(back);
        assert!(e < 1e-6, "relative defect {e}");
    }
}

#[test]
fn refit_recovers_a_shifted_core_and_keeps_the_mass() {
    let truth = CauchyKernel::try_anisotropic(
        DVec3::new(0.2, -0.1, 0.05),
        3.0,
        [4.0, 2.0, 3.0, 0.5, 0.0, -0.3],
        1.5,
    )
    .unwrap();
    let guess = CauchyKernel::try_anisotropic(
        DVec3::new(0.15, -0.05, 0.0),
        3.0,
        [3.5, 2.5, 3.0, 0.0, 0.0, 0.0],
        1.5,
    )
    .unwrap();
    let (z, wz) = astroid_evolution::scf::quad::gauss_legendre(10);
    let mut pts = Vec::new();
    let mut w = Vec::new();
    for (i, x) in z.iter().enumerate() {
        for (j, y) in z.iter().enumerate() {
            for (k, zz) in z.iter().enumerate() {
                pts.push(DVec3::new(*x, *y, *zz));
                w.push(wz[i] * wz[j] * wz[k]);
            }
        }
    }
    let quad = Quadrature { pts, w };
    let td = CauchyDensity::try_new(1.0, vec![truth]).unwrap();
    let target: Vec<f64> = quad.pts.iter().map(|x| td.rho(*x)).collect();
    let mut rep = CoreReport::default();
    let p = CoreParams {
        eps_rho: 1e-6,
        ..params()
    };
    let mass0: f64 = {
        let g = CauchyDensity::try_new(1.0, vec![guess]).unwrap();
        quad.pts
            .iter()
            .zip(&quad.w)
            .map(|(x, w)| w * g.rho(*x))
            .sum()
    };
    let fit = refit(1.0, &[guess], &quad, &target, &p, &mut rep);
    let fd = CauchyDensity::try_new(1.0, fit.clone()).unwrap();
    let mass1: f64 = quad
        .pts
        .iter()
        .zip(&quad.w)
        .map(|(x, w)| w * fd.rho(*x))
        .sum();
    assert!(
        (mass1 - mass0).abs() < 1e-6 * mass0,
        "mass moved {mass0} → {mass1}"
    );
    let before = misfit(
        &CauchyDensity::try_new(1.0, vec![guess]).unwrap(),
        &quad,
        &target,
    );
    assert!(
        rep.e_rho < 0.2 * before,
        "misfit {before:.2e} → {:.2e}",
        rep.e_rho
    );
}
