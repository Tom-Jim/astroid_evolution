//! 半拉格朗日反向特征线追溯与重心坐标插值
//! 严格依照 docs/software.md 第 10.4 节 与 docs/mathstrict.md 第 10 节
//!
//! 一阶反向特征线:
//!   X_d = X_i - Delta t * c_Gamma(X_i, t^n)

/// 计算一阶反向特征线出发点坐标
#[inline]
pub fn backtrace_characteristic_1st_order(
    x_arrival: [f64; 3],
    velocity: [f64; 3],
    dt: f64,
) -> [f64; 3] {
    [
        x_arrival[0] - dt * velocity[0],
        x_arrival[1] - dt * velocity[1],
        x_arrival[2] - dt * velocity[2],
    ]
}

/// 计算二阶中点反向特征线出发点坐标
/// 需要半步速度估计 velocity_half
#[inline]
pub fn backtrace_characteristic_2nd_order(
    x_arrival: [f64; 3],
    velocity_half: [f64; 3],
    dt: f64,
) -> [f64; 3] {
    [
        x_arrival[0] - dt * velocity_half[0],
        x_arrival[1] - dt * velocity_half[1],
        x_arrival[2] - dt * velocity_half[2],
    ]
}

/// 计算三维空间中点 P 投影在三角形 ABC 上的重心坐标 [lambda_a, lambda_b, lambda_c]
pub fn triangle_barycentric_coordinates(
    p: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
) -> [f64; 3] {
    let v0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let v2 = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];

    let d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

    let denom = (d00 * d11 - d01 * d01).max(1e-30);
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    [u, v, w]
}

/// 在三角形 ABC 内通过重心坐标对模态守恒状态 m in R^4 执行线性插值
pub fn barycentric_interpolate_modal_state(
    bary: [f64; 3],
    m_a: &[f64; 4],
    m_b: &[f64; 4],
    m_c: &[f64; 4],
) -> [f64; 4] {
    let mut m_interp = [0.0; 4];
    for k in 0..4 {
        m_interp[k] = bary[0] * m_a[k] + bary[1] * m_b[k] + bary[2] * m_c[k];
    }
    m_interp
}

// 别名提供，严格对应 docs/software.md 第 10.4 节命名契约
pub use backtrace_characteristic_1st_order as backtrace_surface_characteristic;
pub use backtrace_characteristic_2nd_order as backtrace_surface_characteristic_rk2;
pub use triangle_barycentric_coordinates as project_point_to_triangle_barycentric;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtrace_and_barycentric() {
        let x_i = [100.0, 50.0, 20.0];
        let vel = [10.0, -5.0, 2.0];
        let dt = 0.5;

        let x_d = backtrace_characteristic_1st_order(x_i, vel, dt);
        assert_eq!(x_d, [95.0, 52.5, 19.0]);

        // 重心坐标在顶点处的测试
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];

        let bary_a = triangle_barycentric_coordinates(a, a, b, c);
        assert!((bary_a[0] - 1.0).abs() < 1e-10);
        assert!(bary_a[1].abs() < 1e-10);
        assert!(bary_a[2].abs() < 1e-10);

        let mid = [0.3333333333333333, 0.3333333333333333, 0.0];
        let bary_mid = triangle_barycentric_coordinates(mid, a, b, c);
        assert!((bary_mid[0] - 0.3333333333333333).abs() < 1e-6);
        assert!((bary_mid[1] - 0.3333333333333333).abs() < 1e-6);
        assert!((bary_mid[2] - 0.3333333333333333).abs() < 1e-6);
    }
}
