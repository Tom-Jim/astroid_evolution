#[inline]
pub fn flow_viscosity(
    cohesion: f64,
    mu_d: f64,
    pressure: f64,
    shear_rate: f64,
    gamma_dot_eps: f64,
) -> f64 {
    let numerator = cohesion + mu_d * pressure.max(0.0);
    let denom = (shear_rate * shear_rate + gamma_dot_eps * gamma_dot_eps).sqrt();
    if denom <= 1e-18 {
        0.0
    } else {
        numerator / denom
    }
}
use nalgebra::Matrix3;
#[inline]
pub fn flow_stress(eta: f64, d_prime: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let flat = [
        d_prime[0][0],
        d_prime[0][1],
        d_prime[0][2],
        d_prime[1][0],
        d_prime[1][1],
        d_prime[1][2],
        d_prime[2][0],
        d_prime[2][1],
        d_prime[2][2],
    ];
    let dp = Matrix3::from_row_slice(&flat);
    let tau = (2.0 * eta) * dp;
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = tau[(r, c)];
        }
    }
    out
}
#[inline]
pub fn equivalent_flow_stress(tau_flow: &[[f64; 3]; 3]) -> f64 {
    let flat = [
        tau_flow[0][0],
        tau_flow[0][1],
        tau_flow[0][2],
        tau_flow[1][0],
        tau_flow[1][1],
        tau_flow[1][2],
        tau_flow[2][0],
        tau_flow[2][1],
        tau_flow[2][2],
    ];
    let mat = Matrix3::from_row_slice(&flat);
    (0.5 * mat.norm_squared()).max(0.0).sqrt()
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf_simulator::stokes::strain_rate::{
        compute_deviatoric_strain_rate, compute_shear_rate, compute_strain_rate,
    };
    #[test]
    fn test_constitutive_consistency() {
        let gamma_dot = 2.0;
        let p = 500.0;
        let mu_d = 0.45;
        let c = 50.0;
        let eps = 1e-8;
        let mut vel_grad = [[0.0; 3]; 3];
        vel_grad[0][1] = gamma_dot;
        let d = compute_strain_rate(&vel_grad);
        let d_prime = compute_deviatoric_strain_rate(&d);
        let actual_gamma_dot = compute_shear_rate(&d_prime);
        assert!((actual_gamma_dot - gamma_dot).abs() < 1e-12);
        let eta = flow_viscosity(c, mu_d, p, actual_gamma_dot, eps);
        let expected_eta = (c + mu_d * p) / gamma_dot;
        assert!((eta - expected_eta).abs() < 1e-6);
        let tau = flow_stress(eta, &d_prime);
        let tau_eq = equivalent_flow_stress(&tau);
        let expected_tau_eq = c + mu_d * p;
        assert!(
            (tau_eq - expected_tau_eq).abs() < 1e-5,
            "tau_eq ({}) 必须等于 c + mu_d * p ({})",
            tau_eq,
            expected_tau_eq
        );
    }
}
