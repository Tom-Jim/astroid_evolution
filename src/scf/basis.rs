//! The divergence-free modal basis of §8 and the scalar pressure basis of §8.1.3.
//!
//! Evaluation runs on the GPU in double-single arithmetic. The operators are the
//! Taylor jets of `P[Φ] = ∇(Φ + x·∇Φ) − x ΔΦ` and `T[Ψ] = ∇Ψ × x`.
//!
//! Velocity: `V_{lkm} = ∇×∇×(Φ x)`, `Φ = R² ψ_{lk}(ζ) Y_{lm}`;
//! `W_{lkm} = ∇×(Ψ x)`, `Ψ = R ψ_{lk}(ζ) Y_{lm}`; `ζ = r / R(ω̂)`, `ω̂ = x/r`.
//! `R(ω̂)` is the spherical-harmonic surface. Every derivative of the basis enters
//! analytically: `v` and `∇v` need jets of order 3, `Δv` and `∇∇v` order 4.

use nalgebra::DMatrix;

use crate::common::linalg::Entry;
use crate::common::math::DVec3;
use crate::rheo::basis::{K, lm_index};

use super::harmonics::SurfaceFit;

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
    pub const D: Self = Self {
        v: false,
        d: true,
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
/// `18p + 6k + c` of `gd` (`∂_k D`). Stored in f64, or in f32 where the rows are only
/// read through the f64-accumulating products of `common::linalg`.
pub struct VelRows<T: Entry = f64> {
    pub n_points: usize,
    pub v: DMatrix<T>,
    pub d: DMatrix<T>,
    pub lap: DMatrix<T>,
    pub gd: DMatrix<T>,
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
    eval_velocity_rays_as(surf, lay, pts, n_per, want)
}

/// The same rows stored as `T`.
pub fn eval_velocity_rays_as<T: Entry>(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
    n_per: usize,
    want: Want,
) -> VelRows<T> {
    super::basis_gpu::velocity(surf, lay, pts, n_per, want)
        .unwrap_or_else(|err| panic!("velocity basis GPU: {err}"))
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
    let got = super::basis_gpu::contract_velocity(surf, lay, pts, n_per, want, u)
        .unwrap_or_else(|err| panic!("velocity contraction GPU: {err}"));
    ContractedVelocity { v: got.v, d: got.d }
}

/// Strain basis at face centres, resident on the device.
pub struct SurfaceStrain {
    inner: super::basis_gpu::GpuStrain,
}

impl SurfaceStrain {
    /// `D(Bu)` at each stored point, six scaled components, point-major.
    pub fn contract(&self, u: &[f64]) -> Result<Vec<f64>, String> {
        self.inner.contract(u)
    }
}

pub fn surface_strain(
    surf: &SurfaceFit,
    lay: &VelLayout,
    pts: &[DVec3],
) -> Result<SurfaceStrain, String> {
    Ok(SurfaceStrain {
        inner: super::basis_gpu::resident_basis(surf, lay, pts, Want::D)?,
    })
}

/// Scalar pressure `P c + lift`, contracted on the device. One value per point.
pub fn contract_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    pres_c: &[f64],
    lift: &[f64],
) -> Vec<f64> {
    super::basis_gpu::contract_pressure(surf, lay, pts, pres_c, lift)
        .unwrap_or_else(|err| panic!("pressure contraction GPU: {err}"))
}

pub fn eval_pressure(
    surf: &SurfaceFit,
    lay: &PresLayout,
    pts: &[DVec3],
    gradients: bool,
) -> PresRows {
    super::basis_gpu::pressure(surf, lay, pts, gradients)
        .unwrap_or_else(|err| panic!("pressure basis GPU: {err}"))
}
