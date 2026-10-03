//! One-sided limits of a uniform body and the Poisson trace.

use crate::common::math::{sym6_max_abs_diff, sym6_outer, DVec3, Sym6};

pub fn poisson_trace_residual(trace: f64, theta: f64, g: f64, rho: f64) -> f64 {
    trace + 4.0 * std::f64::consts::PI * g * theta * rho
}

pub fn hessian_jump_residual(h_plus: Sym6, h_minus: Sym6, g: f64, rho: f64, normal: DVec3) -> f64 {
    let mut expected = sym6_outer(normal);
    for v in &mut expected {
        *v *= 4.0 * std::f64::consts::PI * g * rho;
    }
    let mut jump = [0.0; 6];
    for i in 0..6 {
        jump[i] = h_plus[i] - h_minus[i];
    }
    sym6_max_abs_diff(jump, expected)
}
