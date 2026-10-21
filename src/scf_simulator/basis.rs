//! The divergence-free modal basis of §8 and the scalar pressure basis of §8.7.
//!
//! Evaluation runs on the GPU in double-single arithmetic. The operators are the
//! Taylor jets of `P[Φ] = ∇(Φ + x·∇Φ) − x ΔΦ` and `T[Ψ] = ∇Ψ × x`.
//!
//! Velocity: `V_{lkm} = ∇×∇×(Φ x)`, `Φ = R² ψ_{lk}(ζ) Y_{lm}`;
//! `W_{lkm} = ∇×(Ψ x)`, `Ψ = R ψ_{lk}(ζ) Y_{lm}`; `ζ = r / R(ω̂)`, `ω̂ = x/r`.
//! `R(ω̂)` is the spherical-harmonic surface. Every derivative of the basis enters
//! analytically: `v` and `∇v` need jets of order 3, `Δv` and `∇∇v` order 4.

use nalgebra::DMatrix;

use crate::common::math::DVec3;
use crate::rheo_friction::basis::{K, lm_index};

use super::harmonics::SurfaceFit;

#[cfg(test)]
const S2: f64 = std::f64::consts::SQRT_2;

/// Velocity unknown layout: for every `(l, m)` with `l >= 1` eight consecutive entries,
/// poloidal `k = 0..K` then toroidal `k = 0..K`.
#[derive(Clone, Copy, Debug)]
pub struct VelLayout {
    pub l_max: usize,
    pub n_u: usize,
}

impl VelLayout {
    pub fn new(l_max: usize) -> Self {
        Self {
            l_max,
            n_u: 2 * K * l_max * (l_max + 2),
        }
    }

    #[inline]
    pub fn dof(&self, l: usize, m: i32, toroidal: bool, k: usize) -> usize {
        let blk = (l * l - 1) as i64 + l as i64 + i64::from(m);
        2 * K * blk as usize + if toroidal { K } else { 0 } + k
    }

    /// `(l, m, toroidal, k)` of every unknown, in index order.
    pub fn modes(&self) -> Vec<(usize, i32, bool, usize)> {
        let mut out = Vec::with_capacity(self.n_u);
        for l in 1..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                for tor in [false, true] {
                    for k in 0..K {
                        out.push((l, m, tor, k));
                    }
                }
            }
        }
        out
    }

    /// Coefficients in the layout of `rheo_friction::VelocityField`.
    pub fn to_rheo(&self, u: &[f64]) -> crate::rheo_friction::VelocityField {
        let n = (self.l_max + 1) * (self.l_max + 1);
        let mut q = vec![[0.0; K]; n];
        let mut s = vec![[0.0; K]; n];
        for l in 1..=self.l_max {
            for m in -(l as i32)..=(l as i32) {
                for k in 0..K {
                    q[lm_index(l, m)][k] = u[self.dof(l, m, false, k)];
                    s[lm_index(l, m)][k] = u[self.dof(l, m, true, k)];
                }
            }
        }
        crate::rheo_friction::VelocityField {
            l_max: self.l_max,
            q,
            s,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Want {
    pub v: bool,
    pub d: bool,
    pub lap: bool,
    pub gd: bool,
}

impl Want {
    pub const V: Self = Self {
        v: true,
        d: false,
        lap: false,
        gd: false,
    };
    pub const VD: Self = Self {
        v: true,
        d: true,
        lap: false,
        gd: false,
    };
    pub const ALL: Self = Self {
        v: true,
        d: true,
        lap: true,
        gd: true,
    };

    pub(crate) fn order(self) -> usize {
        if self.lap || self.gd {
            4
        } else if self.d {
            3
        } else {
            2
        }
    }
}

/// Velocity rows at a set of points. Row `3p + a` of `v`, `6p + c` of `d` (components
/// `xx, yy, zz, √2 xy, √2 xz, √2 yz`, so that `d·d = D:D`), `3p + a` of `lap`, and row
/// `18p + 6k + c` of `gd` (`∂_k D`).
pub struct VelRows {
    pub n_points: usize,
    pub v: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub lap: DMatrix<f64>,
    pub gd: DMatrix<f64>,
}

/// Evaluate the velocity basis at `pts`. Each point is its own ray.
pub fn eval_velocity(surf: &SurfaceFit, lay: &VelLayout, pts: &[DVec3], want: Want) -> VelRows {
    eval_velocity_rays(surf, lay, pts, 1, want)
}

/// Same evaluation, with `n_per` samples packed along each ray.
///
/// Points are ray-major: sample `s` of ray `i` is `pts[i * n_per + s]`, and every
/// sample of a ray is a positive multiple of the same direction. Order-3 kernels
/// scale the degree-0 jets of `R` and `Y` along that ray.
pub fn eval_velocity_rays(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
) -> VelRows {
    #[cfg(target_os = "macos")]
    {
        return super::basis_gpu::velocity(surf, lay, pts, n_per, want)
            .unwrap_or_else(|err| panic!("velocity basis GPU: {err}"));
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    {
        return super::basis_web::velocity(surf, lay, pts, n_per, want)
            .unwrap_or_else(|err| panic!("velocity basis GPU: {err}"));
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _ = (surf, lay, pts, n_per, want);
        panic!("velocity basis requires Metal");
    }
}

/// Scalar basis `(1 − ζ) ψ_{lk}(ζ) Y_{lm}`, `0 <= l <= l_p`, `k < k_p`, and the
/// boundary lift `ζ^l Y_{lm}`.
#[derive(Clone, Copy, Debug)]
pub struct PresLayout {
    pub l_max: usize,
    pub k_max: usize,
}

impl PresLayout {
    pub fn n_p(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1) * self.k_max
    }

    pub fn n_lift(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1)
    }

    #[inline]
    pub fn index(&self, l: usize, m: i32, k: usize) -> usize {
        lm_index(l, m) * self.k_max + k
    }
}

/// Values (`P × n`) and gradients (`3P × n`) of the interior basis and of the lift.
pub struct PresRows {
    pub val: DMatrix<f64>,
    pub grad: DMatrix<f64>,
    pub lift_val: DMatrix<f64>,
    pub lift_grad: DMatrix<f64>,
}

/// Velocity and strain of one coefficient vector, contracted on the device.
///
/// `v` has length `3 * pts.len()` and `d` has length `6 * pts.len()`, both
/// point-major. This is the same product as `eval_velocity_rays` followed by
/// a matrix-vector multiply, without assembling either matrix.
pub struct ContractedVelocity {
    pub v: Vec<f64>,
    pub d: Vec<f64>,
}

pub fn contract_velocity_rays(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
    u: &[f64],
) -> ContractedVelocity {
    #[cfg(target_os = "macos")]
    {
        let got = super::basis_gpu::contract_velocity(surf, lay, pts, n_per, want, u)
            .unwrap_or_else(|err| panic!("velocity contraction GPU: {err}"));
        return ContractedVelocity { v: got.v, d: got.d };
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    {
        let got = super::basis_web::contract_velocity(surf, lay, pts, n_per, want, u)
            .unwrap_or_else(|err| panic!("velocity contraction GPU: {err}"));
        return ContractedVelocity { v: got.v, d: got.d };
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _ = (surf, lay, pts, n_per, want, u);
        panic!("velocity contraction requires Metal");
    }
}

/// Strain basis at face centres, resident in one Metal buffer.
pub struct SurfaceStrain {
    #[cfg(target_os = "macos")]
    inner: super::basis_gpu::GpuStrain,
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    inner: super::basis_web::GpuStrain,
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    _private: (),
}

impl SurfaceStrain {
    /// `D(Bu)` at each stored point, six scaled components, point-major.
    pub fn contract(&self, u: &[f64]) -> Result<Vec<f64>, String> {
        #[cfg(any(target_os = "macos", target_arch = "wasm32"))]
        {
            self.inner.contract(u)
        }
        #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
        {
            let _ = u;
            Err("strain contraction requires Metal".into())
        }
    }
}

pub fn surface_strain(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
) -> Result<SurfaceStrain, String> {
    #[cfg(target_os = "macos")]
    {
        Ok(SurfaceStrain {
            inner: super::basis_gpu::strain_basis(surf, lay, pts)?,
        })
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    {
        Ok(SurfaceStrain {
            inner: super::basis_web::strain_basis(surf, lay, pts)?,
        })
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _ = (surf, lay, pts);
        Err("strain basis requires Metal".into())
    }
}

/// Scalar pressure `P c + lift`, contracted on the device. One value per point.
pub fn contract_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    pres_c: &[f64],
    lift: &[f64],
) -> Vec<f64> {
    #[cfg(target_os = "macos")]
    {
        return super::basis_gpu::contract_pressure(surf, lay, pts, pres_c, lift)
            .unwrap_or_else(|err| panic!("pressure contraction GPU: {err}"));
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    {
        return super::basis_web::contract_pressure(surf, lay, pts, pres_c, lift)
            .unwrap_or_else(|err| panic!("pressure contraction GPU: {err}"));
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _ = (surf, lay, pts, pres_c, lift);
        panic!("pressure contraction requires Metal");
    }
}

pub fn eval_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    gradients: bool,
) -> PresRows {
    #[cfg(target_os = "macos")]
    {
        return super::basis_gpu::pressure(surf, lay, pts, gradients)
            .unwrap_or_else(|err| panic!("pressure basis GPU: {err}"));
    }
    #[cfg(all(target_arch = "wasm32", not(target_os = "macos")))]
    {
        return super::basis_web::pressure(surf, lay, pts, gradients)
            .unwrap_or_else(|err| panic!("pressure basis GPU: {err}"));
    }
    #[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
    {
        let _ = (surf, lay, pts, gradients);
        panic!("pressure basis requires Metal");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn bumpy_surface() -> SurfaceFit {
        // R = 100 (1 + 0.08 Y20-like + 0.05 xy-like) fitted on a dense grid.
        let (dirs, w) = super::super::quad::sphere_grid(16, 32);
        let radii: Vec<f64> = dirs
            .iter()
            .map(|d| 100.0 * (1.0 + 0.08 * (3.0 * d.z * d.z - 1.0) + 0.05 * d.x * d.y))
            .collect();
        SurfaceFit::fit(4, &dirs, &radii, &w).unwrap()
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
                    lap[a] +=
                        (plus[b].v[(a, j)] - 2.0 * mid.v[(a, j)] + minus[b].v[(a, j)]) / (h * h);
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
        let x = w * surf.radius(w);
        let rows = eval_pressure(&surf, &lay, &[x], true);
        assert!(rows.val.iter().all(|v| v.abs() < 1e-12));
        // The lift reproduces Y_lm there.
        let y = crate::rheo_friction::basis::sh_all(3, w.to_array());
        for l in 0..=3 {
            for m in -(l as i32)..=(l as i32) {
                assert!((rows.lift_val[(0, lm_index(l, m))] - y[lm_index(l, m)]).abs() < 1e-10);
            }
        }
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
}
