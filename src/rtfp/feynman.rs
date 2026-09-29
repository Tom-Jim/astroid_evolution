//! Feynman 参数化移动中心径向表示
//! 严格依照 docs/software.md 第 7 节 与 docs/mathstrict.md 第 1.7 节
//!
//! A^{-1/2} B^{-alpha} = [Gamma(alpha + 1/2) / (Gamma(1/2) Gamma(alpha))]
//!                       * int_0^1 u^{-1/2} (1 - u)^{alpha - 1} [u A + (1 - u) B]^{-alpha - 1/2} du
//! 移动中心:
//!   z(u) = u x + (1 - u) c
//! 移动有效阻尼尺度:
//!   m^2(u) = (1 - u) a^2 + u (1 - u) ||x - c||^2   (其中 a = 1 / sigma)

use super::radial_integrals::{GAUSS_LEGENDRE_16_NODES, GAUSS_LEGENDRE_16_WEIGHTS};

/// 计算在给定 Feynman 参数 u in (0, 1) 处的移动中心 z(u)
#[inline]
pub fn moving_center(x: [f64; 3], c: [f64; 3], u: f64) -> [f64; 3] {
    [
        u * x[0] + (1.0 - u) * c[0],
        u * x[1] + (1.0 - u) * c[1],
        u * x[2] + (1.0 - u) * c[2],
    ]
}

/// 计算移动有效尺度平方 m^2(u) = (1 - u)/sigma^2 + u (1 - u) ||x - c||^2
#[inline]
pub fn moving_scale_squared(x: [f64; 3], c: [f64; 3], sigma: f64, u: f64) -> f64 {
    let dx = x[0] - c[0];
    let dy = x[1] - c[1];
    let dz = x[2] - c[2];
    let dist2 = dx * dx + dy * dy + dz * dz;

    let a2 = 1.0 / (sigma * sigma);
    (1.0 - u) * a2 + u * (1.0 - u) * dist2
}

/// 经变量代换 u = s^2 消除 u^{-1/2} 端点奇异性的高精度 Feynman 外层数值积分
/// int_0^1 u^{-1/2} (1 - u)^{alpha - 1} f(u) du = 2 int_0^1 (1 - s^2)^{alpha - 1} f(s^2) ds
pub fn integrate_feynman_parameter<F>(alpha: f64, mut integrand_f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    let mut sum = 0.0f64;
    // 在 s in [0, 1] 采用 16 点 Gauss-Legendre 求积 (mid = 0.5, h = 0.5)
    let h = 0.5;
    let mid = 0.5;

    for i in 0..16 {
        let node = GAUSS_LEGENDRE_16_NODES[i];
        let weight = GAUSS_LEGENDRE_16_WEIGHTS[i];
        let s = mid + h * node;
        let u = s * s;
        let factor = (1.0 - u).max(0.0).powf(alpha - 1.0);
        let f_val = integrand_f(u);

        sum += weight * 2.0 * factor * f_val;
    }

    sum * h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feynman_moving_center() {
        let x = [10.0, 20.0, 30.0];
        let c = [0.0, 0.0, 0.0];
        let z = moving_center(x, c, 0.5);
        assert_eq!(z, [5.0, 10.0, 15.0]);

        let m2 = moving_scale_squared(x, c, 0.1, 0.5);
        // a^2 = 1 / 0.01 = 100. (1-u)*a^2 = 50. u*(1-u)*dist2 = 0.25 * (100+400+900) = 0.25 * 1400 = 350.
        // m2 = 50 + 350 = 400.
        assert_eq!(m2, 400.0);
    }
}
