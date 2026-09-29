//! 应变率张量、偏应变率与剪切率计算
//! 严格依照 docs/software.md 第 9.1 节 与 docs/mathstrict.md 第 4.1 节

/// 根据三维速度梯度张量 ∇v 计算对称应变率张量 D = 1/2 (∇v + (∇v)^T)
///
/// 其中 velocity_gradient[i][j] = ∂v_i / ∂x_j
#[inline]
pub fn compute_strain_rate(velocity_gradient: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut d = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = 0.5 * (velocity_gradient[i][j] + velocity_gradient[j][i]);
        }
    }
    d
}

/// 计算偏应变率张量 D' = D - 1/3 Tr(D) I
#[inline]
pub fn compute_deviatoric_strain_rate(d: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let tr_d = d[0][0] + d[1][1] + d[2][2];
    let mean_d = tr_d / 3.0;

    let mut d_prime = *d;
    d_prime[0][0] -= mean_d;
    d_prime[1][1] -= mean_d;
    d_prime[2][2] -= mean_d;
    d_prime
}

/// 严格根据三维张量双重收缩计算等效剪切率 dot_gamma = sqrt(2 D' : D')
///
/// 文档强制要求：严格使用 dot_gamma = sqrt(2 D' : D')，禁止在一般三维流动中直接使用局部简化式。
#[inline]
pub fn compute_shear_rate(d_prime: &[[f64; 3]; 3]) -> f64 {
    let mut double_contraction = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            double_contraction += d_prime[i][j] * d_prime[i][j];
        }
    }
    (2.0 * double_contraction).max(0.0).sqrt()
}

/// 从速度梯度直接计算等效剪切率的便捷函数
#[inline]
pub fn compute_shear_rate_from_vel_grad(velocity_gradient: &[[f64; 3]; 3]) -> f64 {
    let d = compute_strain_rate(velocity_gradient);
    let d_prime = compute_deviatoric_strain_rate(&d);
    compute_shear_rate(&d_prime)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_shear() {
        // 简单纯剪切流动: v_x = gamma_dot * y, v_y = 0, v_z = 0
        // 则 ∂v_x / ∂y = gamma0
        let gamma0 = 2.5;
        let mut vel_grad = [[0.0; 3]; 3];
        vel_grad[0][1] = gamma0;

        let d = compute_strain_rate(&vel_grad);
        assert!((d[0][1] - 0.5 * gamma0).abs() < 1e-14);
        assert!((d[1][0] - 0.5 * gamma0).abs() < 1e-14);
        assert!((d[0][0]).abs() < 1e-14);

        let d_prime = compute_deviatoric_strain_rate(&d);
        // Tr(D) = 0, 所以 D' == D
        assert!((d_prime[0][1] - 0.5 * gamma0).abs() < 1e-14);

        let shear_rate = compute_shear_rate(&d_prime);
        // D' : D' = (0.5 gamma0)^2 + (0.5 gamma0)^2 = 0.5 gamma0^2
        // sqrt(2 * D' : D') = sqrt(gamma0^2) = gamma0
        assert!((shear_rate - gamma0).abs() < 1e-12);
    }

    #[test]
    fn test_pure_dilation_zero_shear() {
        // 纯体积膨胀: v = (x, y, z), ∇v = I
        let vel_grad = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let d = compute_strain_rate(&vel_grad);
        let d_prime = compute_deviatoric_strain_rate(&d);

        // D' 应当全为 0
        for i in 0..3 {
            for j in 0..3 {
                assert!(d_prime[i][j].abs() < 1e-14);
            }
        }

        let shear_rate = compute_shear_rate(&d_prime);
        assert!(shear_rate < 1e-14);
    }
}
