//! 自由表面几何度量、面积元与法向量
//! 严格依照 docs/software.md 第 10.2 节 与 docs/mathstrict.md 第 10.2 节
//!
//! 表面度量 Jacobian:
//!   J_Gamma = R^2 sin(theta) * sqrt(1 + (R_theta / R)^2 + (R_phi / (R sin(theta)))^2)

/// 表面度量状态
#[derive(Clone, Copy, Debug)]
pub struct SurfaceMetric {
    /// 表面极半径 R (米)
    pub radius: f64,
    /// 面积元因子 J_Gamma (m²)
    pub jacobian_gamma: f64,
    /// 表面外单位法向量 [n_x, n_y, n_z]
    pub normal: [f64; 3],
    /// 极向切向量 e_theta (未归一化)
    pub e_theta: [f64; 3],
    /// 纬向切向量 e_phi (未归一化)
    pub e_phi: [f64; 3],
    /// 第一基本形式矩阵 (度量张量) g_ab: [[g_tt, g_tp], [g_pt, g_pp]]
    pub g_metric: [[f64; 2]; 2],
    /// 逆度量张量 g^(ab)
    pub inv_g_metric: [[f64; 2]; 2],
}

/// 计算球极坐标下表面点的度量、面积元与外法向量
/// r_theta: dR / dtheta
/// r_phi: dR / dphi
pub fn compute_surface_metric(
    theta: f64,
    phi: f64,
    r: f64,
    r_theta: f64,
    r_phi: f64,
) -> SurfaceMetric {
    let sin_t = theta.sin().max(1e-6);
    let cos_t = theta.cos();
    let sin_p = phi.sin();
    let cos_p = phi.cos();

    // 嵌入空间中的三维位置向量 X(theta, phi)
    // X = R [sin_t cos_p, sin_t sin_p, cos_t]

    // 切向量 e_theta = dX / dtheta = R_theta u + R u_theta
    let u = [sin_t * cos_p, sin_t * sin_p, cos_t];
    let u_theta = [cos_t * cos_p, cos_t * sin_p, -sin_t];
    let u_phi = [-sin_t * sin_p, sin_t * cos_p, 0.0];

    let e_theta = [
        r_theta * u[0] + r * u_theta[0],
        r_theta * u[1] + r * u_theta[1],
        r_theta * u[2] + r * u_theta[2],
    ];

    let e_phi = [
        r_phi * u[0] + r * u_phi[0],
        r_phi * u[1] + r * u_phi[1],
        r_phi * u[2] + r * u_phi[2],
    ];

    // 法向量 n = e_theta x e_phi
    let mut nx = e_theta[1] * e_phi[2] - e_theta[2] * e_phi[1];
    let mut ny = e_theta[2] * e_phi[0] - e_theta[0] * e_phi[2];
    let mut nz = e_theta[0] * e_phi[1] - e_theta[1] * e_phi[0];
    let n_len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-30);

    nx /= n_len;
    ny /= n_len;
    nz /= n_len;

    // 面积元 J_Gamma = ||e_theta x e_phi||
    let jacobian_gamma = n_len;

    // 度量张量 g_ab
    let g_tt = e_theta[0] * e_theta[0] + e_theta[1] * e_theta[1] + e_theta[2] * e_theta[2];
    let g_tp = e_theta[0] * e_phi[0] + e_theta[1] * e_phi[1] + e_theta[2] * e_phi[2];
    let g_pp = e_phi[0] * e_phi[0] + e_phi[1] * e_phi[1] + e_phi[2] * e_phi[2];
    let g_metric = [[g_tt, g_tp], [g_tp, g_pp]];

    // 逆度量张量 g^(ab)
    let det_g = (g_tt * g_pp - g_tp * g_tp).max(1e-30);
    let inv_g_metric = [
        [g_pp / det_g, -g_tp / det_g],
        [-g_tp / det_g, g_tt / det_g],
    ];

    SurfaceMetric {
        radius: r,
        jacobian_gamma,
        normal: [nx, ny, nz],
        e_theta,
        e_phi,
        g_metric,
        inv_g_metric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_sphere_metric_degeneracy() {
        // 当 R 为常数球体 (R_theta = 0, R_phi = 0) 时
        // J_Gamma = R^2 sin(theta)
        let r0 = 450.0;
        let theta = PI / 3.0; // 60 度
        let phi = PI / 4.0;   // 45 度

        let metric = compute_surface_metric(theta, phi, r0, 0.0, 0.0);
        let expected_j = r0 * r0 * theta.sin();

        assert!(
            (metric.jacobian_gamma - expected_j).abs() < 1e-6,
            "球面度量 Jacobian 检验失败: 得到 {}, 期望 {}",
            metric.jacobian_gamma, expected_j
        );

        // 法向量应严格沿径向 u
        let u = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
        for i in 0..3 {
            assert!((metric.normal[i] - u[i]).abs() < 1e-10);
        }
    }
}
