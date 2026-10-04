//! Density coefficients consumed by the GPU ray integral.
//!
//! The potential, gradient and Hessian themselves are formed in
//! `field_kernel.metal`. Along a ray leaving `x` in direction `dir`,
//! `u = −dir` and
//!
//! ```text
//! U  = G ∮ dΩ Σ_chords ∫ ρ R dR
//! ∇U = −G ∮ dΩ  u  Σ_chords ∫ ρ dR
//! ∇∇U = G ∮ dΩ T(u) Σ_chords f.p. ∫ ρ / R dR  −  (4π/3) G θ ρ(x) I
//! ```
//!
//! with `T(u) = 3uuᵀ − I`. A kernel with `α = 1` uses the closed antiderivative;
//! any other `α` uses the fixed Gauss–Legendre rule.

use crate::common::density::CauchyDensity;
use crate::common::math::DVec3;

/// One kernel `coeff · (σ⁻² + |y − c|²)^{−α}`.
#[derive(Clone, Copy)]
pub(crate) struct KernelCoeff {
    pub center: DVec3,
    pub inv_sigma2: f64,
    pub coeff: f64,
    pub alpha: f64,
}

/// The density in the form the ray integrals consume.
pub struct Medium {
    bulk: f64,
    kernels: Vec<KernelCoeff>,
}

impl Medium {
    pub fn new(density: &CauchyDensity) -> Self {
        let kernels = density
            .kernels
            .iter()
            .map(|k| {
                let inv_sigma2 = 1.0 / (k.sigma * k.sigma);
                KernelCoeff {
                    center: k.center,
                    inv_sigma2,
                    coeff: k.weight * k.sigma.powf(-2.0 * k.alpha),
                    alpha: k.alpha,
                }
            })
            .collect();
        Self {
            bulk: density.bulk,
            kernels,
        }
    }

    pub(crate) fn bulk(&self) -> f64 {
        self.bulk
    }

    pub(crate) fn kernels(&self) -> &[KernelCoeff] {
        &self.kernels
    }
}
