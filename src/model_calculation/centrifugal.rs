//! 离心势、离心加速度与离心 Hessian 计算
//! 严格依照 docs/software.md 第 8.1、8.2 节 与 docs/mathstrict.md 第 8 节

/// 计算离心势能 U_c(x) = 1/2 ||Omega x x||^2
///
/// 对于自转角速度向量 Omega 和位置向量 x，
/// U_c(x) = 1/2 [ ||Omega||^2 ||x||^2 - (Omega · x)^2 ]
#[inline]
pub fn centrifugal_potential(omega: [f64; 3], pos: [f64; 3]) -> f64 {
    let cross = [
        omega[1] * pos[2] - omega[2] * pos[1],
        omega[2] * pos[0] - omega[0] * pos[2],
        omega[0] * pos[1] - omega[1] * pos[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2])
}

/// 计算离心加速度 a_c(x) = grad U_c(x) = -Omega x (Omega x x)
///
/// 严格对应 docs/software.md 第 8.2 节：
/// 当 Omega 沿 z 轴 (0, 0, Omega_z) 时，a_c = Omega_z^2 (x, y, 0)
#[inline]
pub fn centrifugal_acceleration(omega: [f64; 3], pos: [f64; 3]) -> [f64; 3] {
    let omega_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let omega_dot_x = omega[0] * pos[0] + omega[1] * pos[1] + omega[2] * pos[2];

    [
        omega_sq * pos[0] - omega_dot_x * omega[0],
        omega_sq * pos[1] - omega_dot_x * omega[1],
        omega_sq * pos[2] - omega_dot_x * omega[2],
    ]
}

/// 计算离心势能的 Hessian 矩阵: grad^2 U_c = ||Omega||^2 I - Omega (x) Omega
///
/// 当 Omega = (0, 0, Omega_z) 时，为 Omega_z^2 diag(1, 1, 0)
#[inline]
pub fn centrifugal_hessian(omega: [f64; 3]) -> [[f64; 3]; 3] {
    let omega_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    let mut h = [[0.0; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            h[i][j] = omega_sq * delta - omega[i] * omega[j];
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centrifugal_z_axis() {
        let omega = [0.0, 0.0, 2.0]; // 2 rad/s around z
        let pos = [3.0, 4.0, 5.0];

        // U_c = 1/2 * 2^2 * (3^2 + 4^2) = 0.5 * 4 * 25 = 50.0
        let uc = centrifugal_potential(omega, pos);
        assert!((uc - 50.0).abs() < 1e-12);

        // a_c = 2^2 * (3, 4, 0) = (12, 16, 0)
        let ac = centrifugal_acceleration(omega, pos);
        assert!((ac[0] - 12.0).abs() < 1e-12);
        assert!((ac[1] - 16.0).abs() < 1e-12);
        assert!((ac[2] - 0.0).abs() < 1e-12);

        // Hessian: diag(4, 4, 0)
        let h = centrifugal_hessian(omega);
        assert!((h[0][0] - 4.0).abs() < 1e-12);
        assert!((h[1][1] - 4.0).abs() < 1e-12);
        assert!((h[2][2] - 0.0).abs() < 1e-12);
        assert!((h[0][1]).abs() < 1e-12);
    }
}
