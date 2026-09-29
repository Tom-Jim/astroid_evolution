//! 联合流变模型向经典无黏聚 mu(I) 模型的退化检验
//! 严格依照 docs/software.md 第 9 节 与 docs/mathstrict.md 第 4.7-4.16 节

use crate::rheology::cohesion::{cohesive_pressure, effective_pressure};
use crate::rheology::constitutive::{equivalent_flow_stress, flow_stress, flow_viscosity};
use crate::rheology::friction_law::{mu_classic, mu_dynamic};
use crate::rheology::inertial_number::{classic_inertial_number, modified_inertial_number};
use crate::rheology::tide_stress::tidal_backstress;

/// 检验退化条件与误差评测结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegeneracyCheckResult {
    /// 内聚比率 p_c / p (Condition A)
    pub cohesion_ratio: f64,
    /// 潮汐应力比率 ||T_tide||_F / p (Condition B)
    pub tide_stress_ratio: f64,
    /// 惯性数相对偏差 |I_c - I| / max(I, 1e-12)
    pub inertial_number_rel_diff: f64,
    /// 摩擦系数绝对偏差 |mu_d - mu_cl|
    pub friction_abs_diff: f64,
    /// 等效剪应力相对偏差 |tau_flow_eq - mu_cl * p| / (mu_cl * p)
    pub flow_stress_rel_diff: f64,
}

/// 针对给定状态和参数，定量评测联合模型与经典 mu(I) 模型的退化一致性
pub fn check_degeneracy_to_classic(
    pressure: f64,
    solid_fraction: f64,
    coordination_number: f64,
    grain_diameter: f64,
    grain_density: f64,
    cohesion_force: f64,
    c_c: f64,
    c_t: f64,
    hessian_w: &[[f64; 3]; 3],
    d_prime: &[[f64; 3]; 3],
    shear_rate: f64,
    mu_s: f64,
    mu_2: f64,
    i_0: f64,
    cohesion_macro: f64,
    eps: f64,
) -> DegeneracyCheckResult {
    // 1. 内聚压力与有效正压力
    let p_c = cohesive_pressure(c_c, coordination_number, solid_fraction, cohesion_force, grain_diameter);
    let p_star = effective_pressure(pressure, p_c);
    let cohesion_ratio = if pressure > 1e-12 { p_c / pressure } else { 0.0 };

    // 2. 潮汐背应力
    let t_tide = tidal_backstress(c_t, grain_density, solid_fraction, grain_diameter, hessian_w);
    let mut sum_sq = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            sum_sq += t_tide[i][j] * t_tide[i][j];
        }
    }
    let t_tide_norm = sum_sq.sqrt();
    let tide_stress_ratio = if pressure > 1e-12 { t_tide_norm / pressure } else { 0.0 };

    // 3. 惯性数
    let i_classic = classic_inertial_number(grain_diameter, shear_rate, pressure, grain_density);
    let i_c = modified_inertial_number(grain_diameter, shear_rate, p_star, grain_density);
    let inertial_number_rel_diff = (i_c - i_classic).abs() / i_classic.max(1e-12);

    // 4. 摩擦系数
    let mu_cl = mu_classic(i_classic, mu_s, mu_2, i_0);
    let mu_d = mu_dynamic(i_c, mu_s, mu_2, i_0);
    let friction_abs_diff = (mu_d - mu_cl).abs();

    // 5. 等效流动应力
    let eta = flow_viscosity(cohesion_macro, mu_d, pressure, shear_rate, eps);
    let tau_flow = flow_stress(eta, d_prime);
    let tau_flow_eq = equivalent_flow_stress(&tau_flow);
    let expected_classic_tau = mu_cl * pressure;
    let flow_stress_rel_diff = (tau_flow_eq - expected_classic_tau).abs() / expected_classic_tau.max(1e-12);

    DegeneracyCheckResult {
        cohesion_ratio,
        tide_stress_ratio,
        inertial_number_rel_diff,
        friction_abs_diff,
        flow_stress_rel_diff,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rheology::strain_rate::{compute_deviatoric_strain_rate, compute_shear_rate, compute_strain_rate};

    #[test]
    fn test_exact_asymptotic_degeneracy() {
        let p = 2000.0;
        let nu = 0.6;
        let z = 6.0;
        let d = 0.005;
        let rho_p = 2500.0;

        // 条件 A, B, C: 内聚力 = 0, 宏观黏聚 = 0, Hessian = 0
        let f_h = 0.0;
        let c_c = 1.0;
        let c_t = 1.0;
        let c_macro = 0.0;
        let hessian_w = [[0.0; 3]; 3];

        let gamma_dot = 4.0;
        let mut vel_grad = [[0.0; 3]; 3];
        vel_grad[0][1] = gamma_dot;
        let d_mat = compute_strain_rate(&vel_grad);
        let d_prime = compute_deviatoric_strain_rate(&d_mat);
        let actual_gamma = compute_shear_rate(&d_prime);

        let res = check_degeneracy_to_classic(
            p,
            nu,
            z,
            d,
            rho_p,
            f_h,
            c_c,
            c_t,
            &hessian_w,
            &d_prime,
            actual_gamma,
            0.38,
            0.64,
            0.279,
            c_macro,
            1e-8,
        );

        // 在条件满足时，退化偏差必须小于数值容差 1e-7
        assert!(res.cohesion_ratio < 1e-12);
        assert!(res.tide_stress_ratio < 1e-12);
        assert!(res.inertial_number_rel_diff < 1e-12);
        assert!(res.friction_abs_diff < 1e-12);
        assert!(res.flow_stress_rel_diff < 1e-6);
    }
}
