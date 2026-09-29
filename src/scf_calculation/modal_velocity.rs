//! 自由表面物理流速与径向模态差分剪切率 F_rad
//! 严格依照 docs/software.md 第 10 节 与 docs/mathstrict.md 第 10.2 节

/// 评估 B_k 向量: B_k = grad_S q_k + (1 - k) q_k grad_S(log R)
///
/// 其中 grad_S(log R) = (1 / R) grad_S R
#[inline]
pub fn compute_b_k(
    grad_s_q_k: [f64; 2],
    q_k: f64,
    grad_s_r: [f64; 2],
    r: f64,
    k_order: usize, // k in {1, 2, 3, 4}
) -> [f64; 2] {
    let k = k_order as f64;
    let inv_r = if r > 1e-12 { 1.0 / r } else { 0.0 };
    let coeff = (1.0 - k) * q_k * inv_r;

    [
        grad_s_q_k[0] + coeff * grad_s_r[0],
        grad_s_q_k[1] + coeff * grad_s_r[1],
    ]
}

/// 计算自由表面物理流速合成切向分量 u_Gamma(omega, t) = R * sum_{k=1}^4 (k+2) B_k
pub fn compute_surface_tangential_velocity_2d(
    r: f64,
    grad_s_r: [f64; 2],
    q: &[f64; 4],
    grad_s_q: &[[f64; 2]; 4],
) -> [f64; 2] {
    let mut sum_b = [0.0; 2];
    for k in 1..=4 {
        let b_k = compute_b_k(grad_s_q[k - 1], q[k - 1], grad_s_r, r, k);
        let weight = (k + 2) as f64;
        sum_b[0] += weight * b_k[0];
        sum_b[1] += weight * b_k[1];
    }

    [r * sum_b[0], r * sum_b[1]]
}

/// 计算径向差分剪切率特征函数 F_rad
///
/// 公式依据 docs/mathstrict.md 式 (8260):
/// F_rad = || sum_{k=1}^4 (k + 2)(k - 1) B_k ||
///
/// 当 grad_S R = 0 (球面/局部冻结几何) 时，q_1 项严格消失，退化为:
/// F_rad|_(grad_S R = 0) = || 4 grad_S q_2 + 10 grad_S q_3 + 18 grad_S q_4 ||
pub fn compute_radial_shear_rate_f_rad(
    r: f64,
    grad_s_r: [f64; 2],
    q: &[f64; 4],
    grad_s_q: &[[f64; 2]; 4],
) -> f64 {
    let mut sum_vec = [0.0; 2];

    for k in 1..=4 {
        let b_k = compute_b_k(grad_s_q[k - 1], q[k - 1], grad_s_r, r, k);
        // (k + 2) * (k - 1):
        // k=1 -> 3 * 0 = 0
        // k=2 -> 4 * 1 = 4
        // k=3 -> 5 * 2 = 10
        // k=4 -> 6 * 3 = 18
        let weight = ((k + 2) * (k - 1)) as f64;
        sum_vec[0] += weight * b_k[0];
        sum_vec[1] += weight * b_k[1];
    }

    (sum_vec[0] * sum_vec[0] + sum_vec[1] * sum_vec[1]).sqrt()
}

/// 局部冻结几何简化形式: F_rad = || 4 grad_S q_2 + 10 grad_S q_3 + 18 grad_S q_4 ||
#[inline]
pub fn compute_f_rad_simplified(
    grad_s_q2: [f64; 2],
    grad_s_q3: [f64; 2],
    grad_s_q4: [f64; 2],
) -> f64 {
    let vx = 4.0 * grad_s_q2[0] + 10.0 * grad_s_q3[0] + 18.0 * grad_s_q4[0];
    let vy = 4.0 * grad_s_q2[1] + 10.0 * grad_s_q3[1] + 18.0 * grad_s_q4[1];
    (vx * vx + vy * vy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_rad_degeneracy() {
        let r = 100.0;
        let zero_grad_r = [0.0, 0.0];
        let q = [1.2, 0.5, -0.3, 0.1];
        let grad_s_q = [
            [1.0, 2.0],  // q_1 梯度
            [0.1, 0.2],  // q_2 梯度
            [-0.05, 0.1],// q_3 梯度
            [0.02, -0.01]// q_4 梯度
        ];

        let f_rad_full = compute_radial_shear_rate_f_rad(r, zero_grad_r, &q, &grad_s_q);
        let f_rad_simp = compute_f_rad_simplified(grad_s_q[1], grad_s_q[2], grad_s_q[3]);

        // 两者在 grad_S R = 0 处必须完全一致
        assert!((f_rad_full - f_rad_simp).abs() < 1e-12);

        // 验证 q_1 对 F_rad 确实无贡献
        let mut grad_s_q_alt = grad_s_q;
        grad_s_q_alt[0] = [100.0, -50.0]; // 改变 q_1 梯度
        let f_rad_alt = compute_radial_shear_rate_f_rad(r, zero_grad_r, &q, &grad_s_q_alt);
        assert!((f_rad_full - f_rad_alt).abs() < 1e-12);
    }
}
