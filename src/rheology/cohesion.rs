//! 内聚压力与有效正压力计算
//! 严格依照 docs/software.md 第 9.2 节 与 docs/mathstrict.md 第 4.3 节

/// 计算微观颗粒接触诱导的等效内聚压力 p_c
///
/// 公式依据 docs/software.md 第 9.2 节 与 docs/mathstrict.md 式 (3805):
/// p_c = C_c * (Z * nu * F_H) / d^2
///
/// 参数：
/// - c_c: 内聚几何常数 (无量纲)
/// - z: 平均配位数
/// - nu: 密实度 / 固体体积分数
/// - cohesion_force: 单个接触内聚力 F_H (N)
/// - grain_diameter: 颗粒粒径 d (m)
#[inline]
pub fn cohesive_pressure(
    c_c: f64,
    z: f64,
    nu: f64,
    cohesion_force: f64,
    grain_diameter: f64,
) -> f64 {
    if grain_diameter <= 0.0 {
        return 0.0;
    }
    let d2 = grain_diameter * grain_diameter;
    c_c * (z * nu * cohesion_force) / d2
}

/// 计算正则化后的有效正压力 p_* = p + p_c
#[inline]
pub fn effective_pressure(pressure: f64, p_c: f64) -> f64 {
    (pressure + p_c).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohesive_pressure() {
        let c_c = 1.0;
        let z = 6.0;
        let nu = 0.6;
        let f_h = 1e-7; // 100 nN
        let d = 1e-3;   // 1 mm

        // p_c = 1.0 * (6.0 * 0.6 * 1e-7) / (1e-6) = 3.6 * 1e-7 / 1e-6 = 0.36 Pa
        let pc = cohesive_pressure(c_c, z, nu, f_h, d);
        assert!((pc - 0.36).abs() < 1e-12);

        let p_star = effective_pressure(10.0, pc);
        assert!((p_star - 10.36).abs() < 1e-12);
    }
}
