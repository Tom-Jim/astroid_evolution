//! 投影有效摩擦系数与保守标量摩擦角
//! 严格依照 docs/software.md 第 9.5 节 与 docs/mathstrict.md 第 8.3、9.5 节

/// 计算用于几何投影的有效摩擦系数 mu_eff
///
/// 公式依据 docs/software.md 第 9.5 节 与 docs/mathstrict.md 式 (8332):
/// mu_eff = (c + mu_proj * p + ||T_tide||_F / sqrt(2)) / p_*
///
/// 其中：
/// - c: 宏观黏聚力 (Pa)
/// - mu_proj: 投影基准摩擦系数
/// - p: 静水正压力 (Pa)
/// - t_tide_frobenius: 潮汐背应力张量的 Frobenius 范数 ||T_tide||_F
/// - p_star: 低压正则化有效压力 p_* = p + p_c (Pa)
///
/// 严格提示：该标量摩擦角是对张量屈服约束的充分（保守）闭合。
#[inline]
pub fn effective_projection_mu(
    cohesion: f64,
    mu_proj: f64,
    pressure: f64,
    t_tide_frobenius: f64,
    p_star: f64,
) -> f64 {
    if p_star <= 1e-14 {
        return 0.0;
    }
    let numerator = cohesion + mu_proj * pressure.max(0.0) + t_tide_frobenius / std::f64::consts::SQRT_2;
    (numerator / p_star).max(0.0)
}

/// 计算标量摩擦角 phi = arctan(mu_eff)
#[inline]
pub fn friction_angle(mu_eff: f64) -> f64 {
    mu_eff.max(0.0).atan()
}

/// 计算摩擦角正弦平方 sin^2(phi) = mu_eff^2 / (1 + mu_eff^2)
#[inline]
pub fn sin2_friction_angle(mu_eff: f64) -> f64 {
    let mu2 = mu_eff * mu_eff;
    mu2 / (1.0 + mu2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_friction_angle() {
        let c = 0.0;
        let mu_proj = 0.5;
        let p = 100.0;
        let t_tide_f = 0.0;
        let p_star = p;

        let mu_eff = effective_projection_mu(c, mu_proj, p, t_tide_f, p_star);
        assert!((mu_eff - 0.5).abs() < 1e-14);

        let phi = friction_angle(mu_eff);
        assert!((phi - 0.5_f64.atan()).abs() < 1e-14);

        let sin2 = sin2_friction_angle(mu_eff);
        // 0.25 / 1.25 = 0.2
        assert!((sin2 - 0.2).abs() < 1e-14);
    }
}
