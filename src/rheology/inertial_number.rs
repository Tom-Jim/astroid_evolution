//! 经典与修正惯性数 (Inertial Number) 计算
//! 严格依照 docs/software.md 第 9.2 节 与 docs/mathstrict.md 第 4.1、4.4 节

/// 计算传统无黏聚颗粒体系的经典惯性数
///
/// I = d * dot_gamma / sqrt(p / rho_p)
///
/// 若 p <= 0 或 rho_p <= 0，返回 0.0。
#[inline]
pub fn classic_inertial_number(
    grain_diameter: f64,
    shear_rate: f64,
    pressure: f64,
    grain_density: f64,
) -> f64 {
    if pressure <= 1e-14 || grain_density <= 1e-14 || grain_diameter <= 0.0 {
        return 0.0;
    }
    let vel_scale = (pressure / grain_density).sqrt();
    grain_diameter * shear_rate.max(0.0) / vel_scale
}

/// 计算考虑微观颗粒内聚力影响的修正惯性数
///
/// 公式依据 docs/software.md 第 9.2 节 与 docs/mathstrict.md 式 (3816):
/// I_c = d * dot_gamma / sqrt(p_* / rho_p)
///
/// 其中 p_* = p + p_c 为有效正压力。
#[inline]
pub fn modified_inertial_number(
    grain_diameter: f64,
    shear_rate: f64,
    p_star: f64,
    grain_density: f64,
) -> f64 {
    if p_star <= 1e-14 || grain_density <= 1e-14 || grain_diameter <= 0.0 {
        return 0.0;
    }
    let vel_scale = (p_star / grain_density).sqrt();
    grain_diameter * shear_rate.max(0.0) / vel_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inertial_numbers() {
        let d = 0.01;      // 1 cm
        let dot_gamma = 5.0; // 5 s^-1
        let p = 1000.0;    // 1 kPa
        let rho_p = 2500.0;// 2.5 g/cm^3
        let p_c = 100.0;   // 100 Pa
        let p_star = p + p_c;

        let i_cl = classic_inertial_number(d, dot_gamma, p, rho_p);
        let expected_i_cl = 0.01 * 5.0 / (1000.0 / 2500.0_f64).sqrt();
        assert!((i_cl - expected_i_cl).abs() < 1e-12);

        let i_c = modified_inertial_number(d, dot_gamma, p_star, rho_p);
        let expected_i_c = 0.01 * 5.0 / (1100.0 / 2500.0_f64).sqrt();
        assert!((i_c - expected_i_c).abs() < 1e-12);

        // 当 p_c -> 0 时，I_c -> I_cl
        let i_c_zero_cohesion = modified_inertial_number(d, dot_gamma, p, rho_p);
        assert!((i_c_zero_cohesion - i_cl).abs() < 1e-14);
    }
}
