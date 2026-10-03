//! Ray factors of one Cauchy kernel (MS §1.4–1.6).
//!
//! `b = (x − c) · u`, `d² = σ⁻² + |x − c|² − b²`, `A0 = b² + d²`.

use crate::common::math::DVec3;

#[derive(Clone, Copy, Debug)]
pub struct RayFactors {
    pub b: f64,
    pub d: f64,
    pub d2: f64,
    pub a0: f64,
    pub coeff: f64,
}

pub fn ray_factors(x: DVec3, u: DVec3, center: DVec3, sigma: f64, weight: f64, alpha: f64) -> RayFactors {
    let p = x - center;
    let b = p.dot(u);
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let d2 = inv_sigma2 + p.length_squared() - b * b;
    let d = d2.sqrt();
    let a0 = b * b + d2;
    RayFactors {
        b,
        d,
        d2,
        a0,
        coeff: weight * sigma.powf(-2.0 * alpha),
    }
}
