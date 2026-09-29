//! 联合屈服准则与屈服面函数
//! 严格依照 docs/software.md 第 9.4 节 与 docs/mathstrict.md 第 4.6 节

/// 计算包含潮汐背应力的微观-宏观联合屈服函数 F_comb
///
/// 公式依据 docs/software.md 第 9.4 节 与 docs/mathstrict.md 式 (3870):
/// F_comb = sqrt(1/2 (s - T_tide) : (s - T_tide)) - c - mu_y * p
///
/// 物理含义：
/// - F_comb <= 0: 应力处于弹性/未屈服区域，介质表现为准刚性；
/// - F_comb > 0: 超过屈服极限，触发塑性剪切与重力流滑移。
#[inline]
pub fn combined_yield_function(
    stress_dev: &[[f64; 3]; 3],
    t_tide: &[[f64; 3]; 3],
    cohesion: f64,
    mu_y: f64,
    pressure: f64,
) -> f64 {
    let mut contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let diff = stress_dev[i][j] - t_tide[i][j];
            contraction += diff * diff;
        }
    }
    let effective_shear_stress = (0.5 * contraction).max(0.0).sqrt();
    effective_shear_stress - cohesion - mu_y * pressure.max(0.0)
}

/// 判别是否达到屈服状态 (F_comb > 0)
#[inline]
pub fn is_yielded(f_comb: f64) -> bool {
    f_comb > 0.0
}

/// 经典 Drucker-Prager / Mohr-Coulomb 屈服函数 (无潮汐背应力极限 T_tide = 0)
#[inline]
pub fn classic_yield_function(
    stress_dev: &[[f64; 3]; 3],
    cohesion: f64,
    mu_y: f64,
    pressure: f64,
) -> f64 {
    let zero_tide = [[0.0; 3]; 3];
    combined_yield_function(stress_dev, &zero_tide, cohesion, mu_y, pressure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_condition() {
        let p = 1000.0;
        let c = 50.0;
        let mu_y = 0.4;
        let yield_threshold = c + mu_y * p; // 450.0 Pa

        // 构造等效应力等于 400 Pa 的偏应力张量
        // sqrt(1/2 (s_11^2 + s_22^2)) = 400
        let s_below = [
            [400.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        // contraction = (400 * sqrt(2))^2 = 320000; sqrt(1/2 * 320000) = 400
        let f_below = classic_yield_function(&s_below, c, mu_y, p);
        assert!((f_below - (400.0 - yield_threshold)).abs() < 1e-10);
        assert!(!is_yielded(f_below));

        // 构造等效应力等于 500 Pa 的偏应力张量
        let s_above = [
            [500.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let f_above = classic_yield_function(&s_above, c, mu_y, p);
        assert!((f_above - (500.0 - yield_threshold)).abs() < 1e-10);
        assert!(is_yielded(f_above));
    }

    #[test]
    fn test_tide_counteraction() {
        let p = 1000.0;
        let c = 50.0;
        let mu_y = 0.4;

        let s = [
            [500.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        // 潮汐背应力与 s 同相，部分抵消流动偏应力
        let t_tide = [
            [200.0 * 2.0_f64.sqrt(), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];

        let f_comb = combined_yield_function(&s, &t_tide, c, mu_y, p);
        // 有效应力为 500 - 200 = 300 Pa < 450 Pa，不屈服
        assert!((f_comb - (300.0 - 450.0)).abs() < 1e-10);
        assert!(!is_yielded(f_comb));
    }
}
