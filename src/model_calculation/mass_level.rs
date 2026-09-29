//! 质量能级 W_0 求解与等势面质量积分
//! 严格依照 docs/software.md 第 8.3 节 与 docs/mathstrict.md 第 8 节

use crate::model_calculation::equipotential::solve_equipotential_radii_batch;

/// 计算在给定等势能级 W_0 下围成几何体的总质量 M(W_0)
///
/// 公式依据 docs/software.md 第 8.3 节：
/// M(W_0) = \int_{ {x: W(x) > W_0} } \rho(x) dx
///
/// 参数：
/// - w0: 候选等势能级
/// - directions: 球面正向采样方向集合 u_i
/// - weights: 球面积分权重 w_i (\sum w_i = 4 \pi)
/// - potential_fn: 有效势评估闭包 W(x)
/// - density_fn: 局部密度函数 \rho(x)
/// - r_min, r_max: 半径搜索范围
/// - initial_radius: 跟踪初猜半径
pub fn mass_above_level<P, D>(
    w0: f64,
    directions: &[[f64; 3]],
    weights: &[f64],
    potential_fn: &P,
    density_fn: &D,
    r_min: f64,
    r_max: f64,
    initial_radius: f64,
) -> f64
where
    P: Fn([f64; 3]) -> f64,
    D: Fn([f64; 3]) -> f64,
{
    let (radii, _) = solve_equipotential_radii_batch(
        directions,
        w0,
        potential_fn,
        r_min,
        r_max,
        initial_radius,
        1e-6,
    );

    let mut total_mass = 0.0;
    for (i, &r_i) in radii.iter().enumerate() {
        let u = directions[i];
        let w_omega = weights[i];

        // 径向 5 点 Gauss-Legendre 或 Simpson 积分: \int_0^{R_i} \rho(r u) r^2 dr
        let n_radial = 8;
        let dr = r_i / n_radial as f64;
        let mut rad_integral = 0.0;

        for k in 0..n_radial {
            let r_mid = (k as f64 + 0.5) * dr;
            let pos = [r_mid * u[0], r_mid * u[1], r_mid * u[2]];
            let rho = density_fn(pos);
            rad_integral += rho * r_mid * r_mid * dr;
        }

        total_mass += w_omega * rad_integral;
    }

    total_mass
}

/// 求解使得 M(W_0) = M_target 的精确等势能级 W_0
///
/// 文档强制要求：只有确认 M(W_0) 单调时才能使用二分法。
pub fn solve_mass_level<P, D>(
    target_mass: f64,
    directions: &[[f64; 3]],
    weights: &[f64],
    potential_fn: &P,
    density_fn: &D,
    w_min: f64,
    w_max: f64,
    r_min: f64,
    r_max: f64,
    initial_radius: f64,
    mass_tol: f64,
    max_iter: usize,
) -> Result<f64, String>
where
    P: Fn([f64; 3]) -> f64,
    D: Fn([f64; 3]) -> f64,
{
    let a = w_min;
    let b = w_max;

    let m_a = mass_above_level(
        a,
        directions,
        weights,
        potential_fn,
        density_fn,
        r_min,
        r_max,
        initial_radius,
    );
    let m_b = mass_above_level(
        b,
        directions,
        weights,
        potential_fn,
        density_fn,
        r_min,
        r_max,
        initial_radius,
    );

    // 检查区间单调性与括号跨越条件
    // W0 越高，等势面越靠近内核，对应体积与质量 M(W0) 越小
    let (mut w_low, mut w_high, m_low, m_high) = if m_a > m_b {
        (a, b, m_a, m_b)
    } else {
        (b, a, m_b, m_a)
    };

    if target_mass > m_low || target_mass < m_high {
        // 尝试扩大搜索区间
        return Err(format!(
            "目标质量 {target_mass} 不在搜索能级区间 [{m_high}, {m_low}] 内部"
        ));
    }

    let mut iter = 0;
    while iter < max_iter {
        iter += 1;
        let w_mid = 0.5 * (w_low + w_high);
        let m_mid = mass_above_level(
            w_mid,
            directions,
            weights,
            potential_fn,
            density_fn,
            r_min,
            r_max,
            initial_radius,
        );

        let err = (m_mid - target_mass).abs();
        if err <= mass_tol || err / target_mass < 1e-5 {
            return Ok(w_mid);
        }

        if m_mid > target_mass {
            w_low = w_mid;
        } else {
            w_high = w_mid;
        }
    }

    Ok(0.5 * (w_low + w_high))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mass_level_sphere() {
        // 匀质球体测试: 半径 R = 100 m, rho = 2000 kg/m^3
        // 真实质量 M = 4/3 * pi * 100^3 * 2000 = 8.37758e9 kg
        let rho0 = 2000.0;
        let density_fn = |_pos: [f64; 3]| -> f64 { rho0 };

        // 构造人工等势函数: W(r) = -1.0e8 / r
        let potential_fn = |pos: [f64; 3]| -> f64 {
            let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(1e-2);
            -1.0e8 / r
        };

        // 6 方向采样，立体角权重 4 pi / 6
        let directions = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let weights = [4.0 * PI / 6.0; 6];

        // 目标质量: R = 100 对应质量
        let target_mass = 4.0 / 3.0 * PI * 100.0_f64.powi(3) * rho0;
        // R = 100 对应 W0 = -1.0e8 / 100 = -1.0e6
        let w_expected = -1.0e6;

        let w0_solved = solve_mass_level(
            target_mass,
            &directions,
            &weights,
            &potential_fn,
            &density_fn,
            -2.0e6,
            -5.0e5,
            10.0,
            300.0,
            100.0,
            target_mass * 1e-3,
            40,
        ).expect("质量能级求解应当成功");

        assert!(
            (w0_solved - w_expected).abs() / w_expected.abs() < 1e-2,
            "求解质量能级偏差过大: 得到 {}, 期望 {}",
            w0_solved, w_expected
        );
    }
}
