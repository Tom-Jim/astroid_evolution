//! 剪切率到摩擦角闭合与坡度约束检查
//! 严格依照 docs/software.md 第 9.5、10.6 节 与 docs/mathstrict.md 第 8.3、9 节

/// 计算有效投影摩擦系数 mu_eff
///
/// 公式依据 docs/software.md 第 9.5 节 与 docs/mathstrict.md 式 (8332):
/// mu_eff = (c + mu_proj * p + ||T_tide||_F / sqrt(2)) / p_*
///
/// 其中 p_* = p + p_c 是低压正则化分母。若 p_* <= 0，返回 0.0。
#[inline]
pub fn effective_projection_mu(
    cohesion: f64,
    mu_proj: f64,
    pressure: f64,
    t_tide_frobenius_norm: f64,
    p_star: f64,
) -> f64 {
    if p_star <= 1e-14 {
        return 0.0;
    }
    let numerator = cohesion + mu_proj * pressure + t_tide_frobenius_norm / std::f64::consts::SQRT_2;
    (numerator / p_star).max(0.0)
}

/// 计算标量保守摩擦角 phi = arctan(mu_eff)
#[inline]
pub fn friction_angle(mu_eff: f64) -> f64 {
    mu_eff.max(0.0).atan()
}

/// 计算 sin^2(phi) = mu_eff^2 / (1 + mu_eff^2)
/// 用于避免计算三角函数，直接供 QP 约束投影检查
#[inline]
pub fn sin2_friction_angle(mu_eff: f64) -> f64 {
    let mu2 = mu_eff * mu_eff;
    mu2 / (1.0 + mu2)
}

/// 检查某表面节点处有效引力与法向量构成的坡度角是否违背休止角约束
///
/// 坡度角定义：sin^2(theta_slope) = |g_eff x n|^2 / (|g_eff|^2 * |n|^2)
/// 约束条件：sin^2(theta_slope) <= sin^2(phi)
///
/// 返回值：
/// (is_violated, sin2_slope, sin2_phi)
pub fn check_slope_violation(
    g_eff: [f64; 3],
    normal: [f64; 3],
    phi: f64,
) -> (bool, f64, f64) {
    let cross = [
        g_eff[1] * normal[2] - g_eff[2] * normal[1],
        g_eff[2] * normal[0] - g_eff[0] * normal[2],
        g_eff[0] * normal[1] - g_eff[1] * normal[0],
    ];

    let cross_norm_sq = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    let g_norm_sq = g_eff[0] * g_eff[0] + g_eff[1] * g_eff[1] + g_eff[2] * g_eff[2];
    let n_norm_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];

    let denom = g_norm_sq * n_norm_sq;
    let sin2_slope = if denom > 1e-20 {
        (cross_norm_sq / denom).min(1.0)
    } else {
        0.0
    };

    let sin_phi = phi.sin();
    let sin2_phi = sin_phi * sin_phi;

    let is_violated = sin2_slope > sin2_phi + 1e-12;

    (is_violated, sin2_slope, sin2_phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_projection_mu_and_angle() {
        let c = 5.0; // Pa
        let mu_proj = 0.6;
        let p = 100.0; // Pa
        let t_tide_norm = 10.0;
        let p_c = 10.0;
        let p_star = p + p_c; // 110.0

        let mu_eff = effective_projection_mu(c, mu_proj, p, t_tide_norm, p_star);
        let expected_num = 5.0 + 0.6 * 100.0 + 10.0 / std::f64::consts::SQRT_2;
        let expected_mu = expected_num / 110.0;

        assert!((mu_eff - expected_mu).abs() < 1e-12);

        let phi = friction_angle(mu_eff);
        assert!((phi - expected_mu.atan()).abs() < 1e-12);

        let sin2 = sin2_friction_angle(mu_eff);
        assert!((sin2 - phi.sin().powi(2)).abs() < 1e-12);
    }

    #[test]
    fn test_slope_violation() {
        // g 沿 -z 方向，法向量 n 倾斜
        let g = [0.0, 0.0, -9.8];
        let normal_aligned = [0.0, 0.0, 1.0]; // 垂直表面，slope = 0
        let phi = 0.5; // ~28.6 deg

        let (violated, sin2_slope, _) = check_slope_violation(g, normal_aligned, phi);
        assert!(!violated);
        assert!(sin2_slope < 1e-12);

        // 倾斜 45 度的法向，slope = 45 deg > 28.6 deg
        let normal_steep = [1.0, 0.0, 1.0];
        let (violated_steep, sin2_slope_steep, sin2_phi) = check_slope_violation(g, normal_steep, phi);
        assert!(violated_steep);
        assert!((sin2_slope_steep - 0.5).abs() < 1e-12); // sin^2(45) = 0.5
        assert!(sin2_slope_steep > sin2_phi);
    }
}
