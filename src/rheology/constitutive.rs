//! 流动本构关系与动态有效黏度
//! 严格依照 docs/software.md 第 9.3 节 与 docs/mathstrict.md 第 4.1、4.6 节

/// 计算正规化流变有效动力黏度 eta
///
/// 公式依据 docs/software.md 第 9.3 节 与 docs/mathstrict.md 式 (3704):
/// eta = (c + mu_d * p) / sqrt(dot_gamma^2 + dot_gamma_eps^2)
///
/// 其中 dot_gamma_eps 为数值正则化参数，用于防止在零剪切率或刚性区发生除零奇异。
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

/// 计算流动偏应力张量 tau_flow = 2 * eta * D'
///
/// 严格依据文档规范：必须保留因子 2，以与等效应力定义 sqrt(1/2 tau : tau) 一致。
#[inline]
pub fn flow_stress(eta: f64, d_prime: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut tau = [[0.0; 3]; 3];
    let two_eta = 2.0 * eta;
    for i in 0..3 {
        for j in 0..3 {
            tau[i][j] = two_eta * d_prime[i][j];
        }
    }
    tau
}

/// 计算偏应力的等效应力 tau_flow_eq = sqrt(1/2 tau_flow : tau_flow)
#[inline]
pub fn equivalent_flow_stress(tau_flow: &[[f64; 3]; 3]) -> f64 {
    let mut contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            contraction += tau_flow[i][j] * tau_flow[i][j];
        }
    }
    (0.5 * contraction).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rheology::strain_rate::{compute_deviatoric_strain_rate, compute_shear_rate, compute_strain_rate};

    #[test]
    fn test_constitutive_consistency() {
        let gamma_dot = 2.0;
        let p = 500.0;
        let mu_d = 0.45;
        let c = 50.0;
        let eps = 1e-8;

        // 剪切应变率张量: vel_grad[0][1] = gamma_dot
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

        // 严格一致性校验: tau_eq 应当精确等于 c + mu_d * p
        let expected_tau_eq = c + mu_d * p;
        assert!(
            (tau_eq - expected_tau_eq).abs() < 1e-5,
            "tau_eq ({}) 必须等于 c + mu_d * p ({})",
            tau_eq, expected_tau_eq
        );
    }
}
