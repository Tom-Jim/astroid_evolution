//! Unit tests of `scf::basis`, moved out of `src/scf/basis.rs`.

mod ctx {
    pub use astroid_evolution::common::math::DVec3;
    pub use astroid_evolution::rheo::basis::{K, lm_index};
    pub use astroid_evolution::scf::harmonics::SurfaceFit;
    pub use nalgebra::DMatrix;
}
use ctx::*;

const S2: f64 = std::f64::consts::SQRT_2;

use astroid_evolution::scf::basis::*;
use std::f64::consts::PI;

/// `R = 100 (1 + 0.08 (3z² − 1) + 0.05 xy)`, a degree-2 star surface (the star-shaped
/// special case of the basis), projected exactly on a product rule.
fn bumpy_surface() -> SurfaceFit {
    let (dirs, w) = astroid_evolution::scf::quad::sphere_grid(16, 32);
    let n = (4 + 1) * (4 + 1);
    let mut coeff = vec![0.0; n];
    for (d, w) in dirs.iter().zip(&w) {
        let r = 100.0 * (1.0 + 0.08 * (3.0 * d.z * d.z - 1.0) + 0.05 * d.x * d.y);
        let y = astroid_evolution::rheo::basis::sh_all(4, d.to_array());
        for a in 0..n {
            coeff[a] += w * r * y[a];
        }
    }
    SurfaceFit { l_max: 4, coeff }
}

fn radius(surf: &SurfaceFit, w: DVec3) -> f64 {
    let y = astroid_evolution::rheo::basis::sh_all(surf.l_max, w.to_array());
    surf.coeff.iter().zip(&y).map(|(c, y)| c * y).sum()
}

#[test]
fn velocity_modes_are_divergence_free_and_match_finite_differences() {
    let surf = bumpy_surface();
    let lay = VelLayout::new(3);
    let x0 = DVec3::new(31.0, -22.0, 40.0);
    let rows = eval_velocity(&surf, &lay, &[x0], Want::ALL);
    let h = 1e-3;
    let shift = |axis: usize, s: f64| {
        let mut p = x0.to_array();
        p[axis] += s;
        eval_velocity(&surf, &lay, &[DVec3::from_array(p)], Want::V)
    };
    let plus: Vec<_> = (0..3).map(|a| shift(a, h)).collect();
    let minus: Vec<_> = (0..3).map(|a| shift(a, -h)).collect();
    let mid = shift(0, 0.0);
    for j in 0..lay.n_u {
        let scale = (0..3).map(|a| rows.v[(a, j)].abs()).fold(1e-30, f64::max);
        let tr = rows.d[(0, j)] + rows.d[(1, j)] + rows.d[(2, j)];
        assert!(
            tr.abs() < 1e-8 * scale / 40.0 * 10.0 + 1e-12,
            "trace dof {j}: {tr}"
        );
        // gradient by central differences of the value rows
        let gr = |a: usize, b: usize| (plus[b].v[(a, j)] - minus[b].v[(a, j)]) / (2.0 * h);
        let dxx = gr(0, 0);
        let dxy = 0.5 * (gr(0, 1) + gr(1, 0));
        assert!(
            (dxx - rows.d[(0, j)]).abs() < 1e-5 * scale / 40.0,
            "Dxx dof {j}"
        );
        assert!(
            (S2 * dxy - rows.d[(3, j)]).abs() < 1e-5 * scale / 40.0,
            "Dxy dof {j}"
        );
        // Laplacian
        let mut lap = [0.0; 3];
        for a in 0..3 {
            for b in 0..3 {
                lap[a] += (plus[b].v[(a, j)] - 2.0 * mid.v[(a, j)] + minus[b].v[(a, j)]) / (h * h);
            }
            // h = 1e-3, so a double-single error in v is amplified by 1/h².
            let diff = (lap[a] - rows.lap[(a, j)]).abs();
            let mag = lap[a].abs().max(rows.lap[(a, j)].abs());
            assert!(
                diff < 2.0e-3 * mag + 1.0e-4,
                "Laplacian dof {j} comp {a}: {} vs {}",
                lap[a],
                rows.lap[(a, j)]
            );
        }
    }
}

#[test]
fn l1_k0_modes_are_rigid_on_a_sphere() {
    let surf = SurfaceFit {
        l_max: 0,
        coeff: vec![100.0 * (4.0 * PI).sqrt()],
    };
    let lay = VelLayout::new(2);
    let x0 = DVec3::new(10.0, 20.0, -30.0);
    let rows = eval_velocity(&surf, &lay, &[x0], Want::VD);
    for m in -1..=1 {
        for tor in [false, true] {
            let j = lay.dof(1, m, tor, 0);
            for c in 0..6 {
                assert!(rows.d[(c, j)].abs() < 1e-10, "mode {m} {tor} comp {c}");
            }
        }
    }
    // A genuinely deforming mode has non-zero strain.
    let j = lay.dof(2, 0, false, 0);
    assert!((0..6).map(|c| rows.d[(c, j)].abs()).sum::<f64>() > 1e-6);
}

#[test]
fn pressure_basis_vanishes_on_the_surface() {
    let surf = bumpy_surface();
    let lay = PresLayout { l_max: 3, k_max: 3 };
    let w = DVec3::new(0.3, -0.5, 0.8).try_normalize().unwrap();
    let x = w * radius(&surf, w);
    let rows = eval_pressure(&surf, &lay, &[x], true);
    assert!(rows.val.iter().all(|v| v.abs() < 1e-12));
    // The lift reproduces Y_lm there.
    let y = astroid_evolution::rheo::basis::sh_all(3, w.to_array());
    for l in 0..=3 {
        for m in -(l as i32)..=(l as i32) {
            assert!((rows.lift_val[(0, lm_index(l, m))] - y[lm_index(l, m)]).abs() < 1e-10);
        }
    }
}

/// The hydrostatic modes `(1 − ζ²) P_k^{(0,1/2)}(2ζ² − 1)` are even in `r`:
/// their gradient vanishes linearly at the centre (no cusp, §8.1.3).
#[test]
fn hydrostatic_pressure_modes_are_smooth_at_the_centre() {
    let surf = bumpy_surface();
    let lay = PresLayout { l_max: 2, k_max: 3 };
    let dirs = [
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, -0.6, 0.8),
        DVec3::new(-0.48, 0.6, -0.64),
    ];
    let grad_at = |r: f64| {
        let pts: Vec<DVec3> = dirs.iter().map(|d| *d * r).collect();
        let rows = eval_pressure(&surf, &lay, &pts, true);
        let mut worst = 0.0f64;
        for k in 0..lay.k_max {
            let col = lay.index(0, 0, k);
            for p in 0..pts.len() {
                let g = DVec3::new(
                    rows.grad[(3 * p, col)],
                    rows.grad[(3 * p + 1, col)],
                    rows.grad[(3 * p + 2, col)],
                );
                worst = worst.max(g.length());
            }
        }
        worst
    };
    let (g1, g2) = (grad_at(1.0), grad_at(0.5));
    assert!(
        g1 < 1e-2 && (g2 / g1 - 0.5).abs() < 0.05,
        "gradient {g1:e} -> {g2:e} is not O(r)"
    );
}

#[test]
fn contraction_matches_the_dense_product() {
    let surf = bumpy_surface();
    let lay = VelLayout::new(3);
    let pts = [
        DVec3::new(40.0, 10.0, 20.0),
        DVec3::new(-15.0, 30.0, 25.0),
        DVec3::new(8.0, -12.0, 50.0),
        DVec3::new(22.0, 18.0, -9.0),
    ];
    let rows = eval_velocity(&surf, &lay, &pts, Want::VD);
    let u: Vec<f64> = (0..lay.n_u)
        .map(|j| 0.01 * ((j * 17) % 11) as f64 - 0.05)
        .collect();
    let coeff = nalgebra::DVector::from_column_slice(&u);
    let expect_v = &rows.v * &coeff;
    let expect_d = &rows.d * &coeff;
    let got = contract_velocity_rays(&surf, &lay, &pts, 1, Want::VD, &u);
    let worst = |got: &[f64], expect: &nalgebra::DVector<f64>| {
        got.iter()
            .zip(expect.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    };
    let scale = expect_v.amax() + expect_d.amax() + 1e-30;
    let miss = worst(&got.v, &expect_v).max(worst(&got.d, &expect_d));
    assert!(
        miss < 1e-8 * scale,
        "velocity contraction left the dense product by {miss:.3e}"
    );
    let strain = surface_strain(&surf, &lay, &pts).expect("strain basis");
    let got_d = strain.contract(&u).expect("strain product");
    let miss_d = worst(&got_d, &expect_d);
    assert!(
        miss_d < 1e-8 * scale,
        "strain product left the dense product by {miss_d:.3e}"
    );
    let play = PresLayout { l_max: 4, k_max: 3 };
    let pres = eval_pressure(&surf, &play, &pts, false);
    let pc: Vec<f64> = (0..play.n_p()).map(|j| 0.02 * (j as f64) - 0.4).collect();
    let lift: Vec<f64> = (0..play.n_lift()).map(|j| 0.01 * (j as f64)).collect();
    let expect_p = &pres.val * nalgebra::DVector::from_column_slice(&pc)
        + &pres.lift_val * nalgebra::DVector::from_column_slice(&lift);
    let got_p = contract_pressure(&surf, &play, &pts, &pc, &lift);
    let miss_p = worst(&got_p, &expect_p);
    assert!(
        miss_p < 1e-8 * (expect_p.amax() + 1e-30),
        "pressure contraction left the dense product by {miss_p:.3e}"
    );
    // Classification packs several radii on each fixed direction.
    let axis = DVec3::new(0.2, -0.5, 0.8).try_normalize().unwrap();
    let other = DVec3::new(-0.7, 0.1, 0.4).try_normalize().unwrap();
    let mut ray_pts = Vec::new();
    for dir in [axis, other] {
        for s in 1..=6 {
            ray_pts.push(dir * (30.0 + 10.0 * s as f64));
        }
    }
    let ray_rows = eval_velocity_rays(&surf, &lay, &ray_pts, 6, Want::VD);
    let ray_v = &ray_rows.v * &coeff;
    let ray_d = &ray_rows.d * &coeff;
    let ray_got = contract_velocity_rays(&surf, &lay, &ray_pts, 6, Want::VD, &u);
    let ray_scale = ray_v.amax() + ray_d.amax() + 1e-30;
    let ray_miss = worst(&ray_got.v, &ray_v).max(worst(&ray_got.d, &ray_d));
    assert!(
        ray_miss < 1e-8 * ray_scale,
        "ray contraction left the dense product by {ray_miss:.3e}"
    );
}
