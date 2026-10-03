//! Cauchy density `ρ_K` and its classical derivatives (MS §1.1).

use crate::common::error::CommonError;
use crate::common::math::{DVec3, Sym6};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CauchyKernel {
    pub center: DVec3,
    pub weight: f64,
    pub sigma: f64,
    pub alpha: f64,
}

impl CauchyKernel {
    pub fn try_new(
        center: DVec3,
        weight: f64,
        sigma: f64,
        alpha: f64,
    ) -> Result<Self, CommonError> {
        if !center.is_finite() || !weight.is_finite() {
            return Err(CommonError::model_not_closed(
                "Cauchy kernel center and weight must be finite",
            ));
        }
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(CommonError::model_not_closed(
                "Cauchy kernel sigma must be finite and positive",
            ));
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(CommonError::model_not_closed(
                "Cauchy kernel alpha must be finite and positive; no implicit alpha=1",
            ));
        }
        Ok(Self {
            center,
            weight,
            sigma,
            alpha,
        })
    }
}

/// Physical density is `bulk + Σ kernels`. `bulk` may be zero. A negative bulk is rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct CauchyDensity {
    pub bulk: f64,
    pub kernels: Vec<CauchyKernel>,
}

impl CauchyDensity {
    pub fn try_new(bulk: f64, kernels: Vec<CauchyKernel>) -> Result<Self, CommonError> {
        if !(bulk.is_finite() && bulk >= 0.0) {
            return Err(CommonError::model_not_closed(
                "bulk density must be finite and non-negative",
            ));
        }
        if bulk == 0.0 && kernels.is_empty() {
            return Err(CommonError::model_not_closed(
                "density model is empty: bulk is zero and no Cauchy kernel was supplied",
            ));
        }
        Ok(Self { bulk, kernels })
    }

    pub fn uniform(bulk: f64) -> Result<Self, CommonError> {
        Self::try_new(bulk, Vec::new())
    }

    #[inline]
    pub fn rho(&self, y: DVec3) -> f64 {
        let mut s = self.bulk;
        for k in &self.kernels {
            s += kernel_rho(k, y);
        }
        s
    }

    #[inline]
    pub fn grad(&self, y: DVec3) -> DVec3 {
        let mut g = DVec3::ZERO;
        for k in &self.kernels {
            g = g + kernel_grad(k, y);
        }
        g
    }

    #[inline]
    pub fn hessian(&self, y: DVec3) -> Sym6 {
        let mut h = [0.0; 6];
        for k in &self.kernels {
            let hk = kernel_hessian(k, y);
            for i in 0..6 {
                h[i] += hk[i];
            }
        }
        h
    }
}

#[inline]
fn kernel_rho(k: &CauchyKernel, y: DVec3) -> f64 {
    let r2 = (y - k.center).length_squared();
    let q = 1.0 + k.sigma * k.sigma * r2;
    k.weight * q.powf(-k.alpha)
}

/// `∂_i ρ = −2 α w σ² q^{−α−1} (y − c)_i`.
#[inline]
fn kernel_grad(k: &CauchyKernel, y: DVec3) -> DVec3 {
    let d = y - k.center;
    let s2 = k.sigma * k.sigma;
    let q = 1.0 + s2 * d.length_squared();
    let coeff = -2.0 * k.alpha * k.weight * s2 * q.powf(-k.alpha - 1.0);
    d * coeff
}

/// Second derivatives of one Cauchy kernel. Layout is `(xx, yy, zz, xy, xz, yz)`.
#[inline]
fn kernel_hessian(k: &CauchyKernel, y: DVec3) -> Sym6 {
    let d = y - k.center;
    let s2 = k.sigma * k.sigma;
    let q = 1.0 + s2 * d.length_squared();
    let q1 = q.powf(-k.alpha - 1.0);
    let q2 = q.powf(-k.alpha - 2.0);
    let diag = -2.0 * k.alpha * k.weight * s2 * q1;
    let outer = 4.0 * k.alpha * (k.alpha + 1.0) * k.weight * s2 * s2 * q2;
    [
        diag + outer * d.x * d.x,
        diag + outer * d.y * d.y,
        diag + outer * d.z * d.z,
        outer * d.x * d.y,
        outer * d.x * d.z,
        outer * d.y * d.z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_central_difference() {
        let k = CauchyKernel::try_new(DVec3::new(0.2, -0.1, 0.05), 1.4, 0.7, 1.3).unwrap();
        let density = CauchyDensity::try_new(0.2, vec![k]).unwrap();
        let y = DVec3::new(0.4, 0.2, -0.3);
        let analytic = density.grad(y);
        let eps = 1e-6;
        let mut numeric = DVec3::ZERO;
        for axis in 0..3 {
            let mut yp = y;
            let mut ym = y;
            match axis {
                0 => {
                    yp.x += eps;
                    ym.x -= eps;
                }
                1 => {
                    yp.y += eps;
                    ym.y -= eps;
                }
                _ => {
                    yp.z += eps;
                    ym.z -= eps;
                }
            }
            let d = (density.rho(yp) - density.rho(ym)) / (2.0 * eps);
            match axis {
                0 => numeric.x = d,
                1 => numeric.y = d,
                _ => numeric.z = d,
            }
        }
        assert!((analytic - numeric).length() < 1e-7);
        let h = density.hessian(y);
        assert!((h[0] + h[1] + h[2] - laplacian_from_grad(&density, y)).abs() < 1e-6);
    }

    fn laplacian_from_grad(density: &CauchyDensity, y: DVec3) -> f64 {
        let eps = 1e-5;
        let axes = [
            DVec3::new(eps, 0.0, 0.0),
            DVec3::new(0.0, eps, 0.0),
            DVec3::new(0.0, 0.0, eps),
        ];
        let mut lap = 0.0;
        for a in axes {
            lap += (density.grad(y + a) - density.grad(y - a)).dot(a) / (a.length_squared() * 2.0);
        }
        lap
    }

    #[test]
    fn missing_alpha_is_rejected() {
        let err = CauchyKernel::try_new(DVec3::ZERO, 1.0, 1.0, f64::NAN).unwrap_err();
        assert_eq!(err.kind, "ModelNotClosed");
    }
}
