//! Ray accumulation of the potential, gradient and Hessian (MS §3).
//!
//! For a ray leaving `x` along `−u` the three fields are
//!
//! ```text
//! U  = G ∮ dΩ Σ_chords ∫ ρ R dR
//! ∇U = −G ∮ dΩ  u  Σ_chords ∫ ρ dR
//! ∇∇U = G ∮ dΩ T(u) Σ_chords f.p. ∫ ρ / R dR  −  (4π/3) G θ ρ(x) I
//! ```
//!
//! with `T(u) = 3uuᵀ − I`. Each ray stands for a solid-angle element `w`.

use crate::common::density::CauchyDensity;
use crate::common::math::{DVec3, Sym6, quadrupole};
use crate::rtfp_solver::api::FieldMask;
use crate::rtfp_solver::rtfp::chords;
use crate::rtfp_solver::rtfp::hw_rt::Hits;
use crate::rtfp_solver::rtfp::radial::{self, General, Integrals, Need, RayKernel};

/// One generalized-Cauchy kernel in the form `coeff · (σ⁻² + |y − c|²)^{-α}`.
struct Kernel {
    center: DVec3,
    inv_sigma2: f64,
    coeff: f64,
    alpha: f64,
}

/// The density in the form the ray integrals consume.
pub struct Medium {
    bulk: f64,
    kernels: Vec<Kernel>,
}

impl Medium {
    pub fn new(density: &CauchyDensity) -> Self {
        let kernels = density
            .kernels
            .iter()
            .map(|k| {
                let inv_sigma2 = 1.0 / (k.sigma * k.sigma);
                Kernel {
                    center: k.center,
                    inv_sigma2,
                    coeff: k.weight * k.sigma.powf(-2.0 * k.alpha),
                    alpha: k.alpha,
                }
            })
            .collect();
        Self { bulk: density.bulk, kernels }
    }
}

/// Field sums of one point, before the factor `G`.
#[derive(Clone, Copy)]
pub struct Sums {
    pub potential: f64,
    pub gradient: DVec3,
    pub hessian: [f64; 6],
    /// Total solid angle of the accepted rays.
    pub solid_angle: f64,
}

impl Sums {
    pub const ZERO: Sums = Sums {
        potential: 0.0,
        gradient: DVec3::ZERO,
        hessian: [0.0; 6],
        solid_angle: 0.0,
    };

    pub fn merge(&mut self, o: &Sums) {
        self.potential += o.potential;
        self.gradient = self.gradient + o.gradient;
        for i in 0..6 {
            self.hessian[i] += o.hessian[i];
        }
        self.solid_angle += o.solid_angle;
    }
}

/// A point with its per-kernel offsets.
pub struct PointKernels {
    offset: Vec<DVec3>,
    a0: Vec<f64>,
}

impl PointKernels {
    pub fn new(medium: &Medium, x: DVec3) -> Self {
        let offset: Vec<DVec3> = medium.kernels.iter().map(|k| x - k.center).collect();
        let a0 = offset
            .iter()
            .zip(&medium.kernels)
            .map(|(p, k)| k.inv_sigma2 + p.length_squared())
            .collect();
        Self { offset, a0 }
    }
}

/// Everything the accumulation of one point's rays needs.
pub struct RayContext<'a> {
    pub medium: &'a Medium,
    pub point: &'a PointKernels,
    pub general: &'a General,
    pub want: FieldMask,
    /// Finite-part scale `ℓ` (the body radius). The result does not depend on it.
    pub ell: f64,
    pub body_scale: f64,
}

/// Accumulate rays `first..last` of `hits`; ray `i` has solid-angle weight `weights[i - first_ray]`
/// and direction `dirs[...]` away from the point.
pub fn accumulate(
    ctx: &RayContext<'_>,
    hits: &Hits,
    first_hit_ray: usize,
    dirs: &[DVec3],
    weights: &[f64],
) -> Sums {
    let need = Need {
        potential: ctx.want.potential,
        hessian: ctx.want.hessian,
    };
    let mut sums = Sums::ZERO;
    let mut dists = Vec::new();
    let mut segs = Vec::new();
    let mut ratios = Vec::new();
    for (j, (&dir, &w)) in dirs.iter().zip(weights).enumerate() {
        sums.solid_angle += w;
        dists.clear();
        hits.distances_into(first_hit_ray + j, &mut dists);
        chords::intervals(&dists, ctx.body_scale, &mut segs);
        if segs.is_empty() {
            continue;
        }
        let u = -dir;
        ratios.clear();
        ratios.extend(segs.iter().map(|&(r0, r1)| {
            if need.hessian {
                (r1 / if r0 == 0.0 { ctx.ell } else { r0 }).ln()
            } else {
                0.0
            }
        }));
        let mut total = Integrals::default();
        let bulk = ctx.medium.bulk;
        if bulk != 0.0 {
            for (&(r0, r1), &ln_ratio) in segs.iter().zip(&ratios) {
                total.u += 0.5 * bulk * (r1 * r1 - r0 * r0);
                total.g += bulk * (r1 - r0);
                total.h += bulk * ln_ratio;
            }
        }
        for (k, kernel) in ctx.medium.kernels.iter().enumerate() {
            let p = ctx.point.offset[k];
            let b = p.dot(u);
            let perp = p - u * b;
            let d2 = kernel.inv_sigma2 + perp.length_squared();
            let geometry = RayKernel { b, d: d2.sqrt(), a0: ctx.point.a0[k] };
            for (&(r0, r1), &ln_ratio) in segs.iter().zip(&ratios) {
                let i = if kernel.alpha == 1.0 {
                    radial::alpha_one(geometry, r0, r1, ln_ratio, need)
                } else {
                    ctx.general.integrals(kernel.alpha, geometry, r0, r1, ln_ratio)
                };
                total.u += kernel.coeff * i.u;
                total.g += kernel.coeff * i.g;
                total.h += kernel.coeff * i.h;
            }
        }
        sums.potential += w * total.u;
        sums.gradient = sums.gradient - u * (w * total.g);
        if need.hessian {
            let t: Sym6 = quadrupole(u);
            for i in 0..6 {
                sums.hessian[i] += w * total.h * t[i];
            }
        }
    }
    sums
}
