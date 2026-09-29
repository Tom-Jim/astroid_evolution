//! 有效引力场 (g_eff = ∇W) 与有效势 Hessian (H_W = ∇^2 W)
//! 严格依照 docs/software.md 第 8.2 节 与 docs/mathstrict.md 第 8 节

use crate::model_calculation::centrifugal::{centrifugal_acceleration, centrifugal_hessian};

/// 合成有效引力加速度 g_eff = grad W = g + a_c
///
/// 严格依照 docs/software.md 第 8.2 节：grad W = grad U + Omega^2 (x_1, x_2, 0)
#[inline]
pub fn compute_effective_gravity(
    gravitational_accel: [f64; 3],
    centrifugal_accel: [f64; 3],
) -> [f64; 3] {
    [
        gravitational_accel[0] + centrifugal_accel[0],
        gravitational_accel[1] + centrifugal_accel[1],
        gravitational_accel[2] + centrifugal_accel[2],
    ]
}

/// 在指定位置计算总有效引力加速度 g_eff(x)
#[inline]
pub fn compute_effective_gravity_at_point(
    gravitational_accel: [f64; 3],
    pos: [f64; 3],
    omega: [f64; 3],
) -> [f64; 3] {
    let a_c = centrifugal_acceleration(omega, pos);
    compute_effective_gravity(gravitational_accel, a_c)
}

/// 合成总有效势 Hessian: H_W = grad^2 W = H_U + grad^2 U_c
#[inline]
pub fn compute_effective_hessian(
    gravitational_hessian: &[[f64; 3]; 3],
    omega: [f64; 3],
) -> [[f64; 3]; 3] {
    let h_c = centrifugal_hessian(omega);
    let mut h_w = [[0.0; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            h_w[i][j] = gravitational_hessian[i][j] + h_c[i][j];
        }
    }
    h_w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_fields() {
        let g_u = [-5.0, -10.0, -2.0];
        let pos = [2.0, 3.0, 0.0];
        let omega = [0.0, 0.0, 1.5]; // Omega_z = 1.5, Omega^2 = 2.25

        // a_c = 2.25 * (2, 3, 0) = (4.5, 6.75, 0)
        // g_eff = (-5.0 + 4.5, -10.0 + 6.75, -2.0 + 0) = (-0.5, -3.25, -2.0)
        let g_eff = compute_effective_gravity_at_point(g_u, pos, omega);
        assert!((g_eff[0] - (-0.5)).abs() < 1e-12);
        assert!((g_eff[1] - (-3.25)).abs() < 1e-12);
        assert!((g_eff[2] - (-2.0)).abs() < 1e-12);

        let h_u = [
            [0.1, 0.05, 0.0],
            [0.05, 0.2, 0.0],
            [0.0, 0.0, -0.3],
        ];
        let h_w = compute_effective_hessian(&h_u, omega);
        assert!((h_w[0][0] - (0.1 + 2.25)).abs() < 1e-12);
        assert!((h_w[1][1] - (0.2 + 2.25)).abs() < 1e-12);
        assert!((h_w[2][2] - (-0.3)).abs() < 1e-12);
    }
}
