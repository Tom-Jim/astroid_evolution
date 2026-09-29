//! K=4 径向 Galerkin 模态基函数
//! 严格依照 docs/software.md 第 10.1 节 与 docs/mathstrict.md 第 10.1 节
//!
//! 固定基函数:
//!   psi_1(zeta) = zeta^2
//!   psi_2(zeta) = zeta^3
//!   psi_3(zeta) = zeta^4
//!   psi_4(zeta) = zeta^5
//!   其中 zeta = r / R(theta, phi) in [0, 1]

/// K=4 阶基函数个数
pub const K4_ORDER: usize = 4;

/// 评估 4 个基函数在归一化半径 zeta in [0, 1] 处的值: [psi_1, psi_2, psi_3, psi_4]
#[inline]
pub fn basis_k4(zeta: f64) -> [f64; 4] {
    let z2 = zeta * zeta;
    let z3 = z2 * zeta;
    let z4 = z3 * zeta;
    let z5 = z4 * zeta;
    [z2, z3, z4, z5]
}

/// 评估 4 个基函数关于 zeta 的一阶导数: [psi'_1, psi'_2, psi'_3, psi'_4]
#[inline]
pub fn basis_k4_derivative(zeta: f64) -> [f64; 4] {
    let z = zeta;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z3 * z;
    [2.0 * z, 3.0 * z2, 4.0 * z3, 5.0 * z4]
}

/// 评估 4 个基函数关于 zeta 的二阶导数: [psi''_1, psi''_2, psi''_3, psi''_4]
#[inline]
pub fn basis_k4_second_derivative(zeta: f64) -> [f64; 4] {
    let z = zeta;
    let z2 = z * z;
    let z3 = z2 * z;
    [2.0, 6.0 * z, 12.0 * z2, 20.0 * z3]
}

/// 通过模态系数 q = [q_1, q_2, q_3, q_4] 评估物理场值: f(zeta) = sum_{k=1}^4 q_k psi_k(zeta)
#[inline]
pub fn evaluate_modal_field(q: &[f64; 4], zeta: f64) -> f64 {
    let psi = basis_k4(zeta);
    q[0] * psi[0] + q[1] * psi[1] + q[2] * psi[2] + q[3] * psi[3]
}

/// 评估物理场对 zeta 的一阶导数: f'(zeta) = sum_{k=1}^4 q_k psi'_k(zeta)
#[inline]
pub fn evaluate_modal_derivative(q: &[f64; 4], zeta: f64) -> f64 {
    let dpsi = basis_k4_derivative(zeta);
    q[0] * dpsi[0] + q[1] * dpsi[1] + q[2] * dpsi[2] + q[3] * dpsi[3]
}

/// 评估物理场对 zeta 的二阶导数: f''(zeta) = sum_{k=1}^4 q_k psi''_k(zeta)
#[inline]
pub fn evaluate_modal_second_derivative(q: &[f64; 4], zeta: f64) -> f64 {
    let d2psi = basis_k4_second_derivative(zeta);
    q[0] * d2psi[0] + q[1] * d2psi[1] + q[2] * d2psi[2] + q[3] * d2psi[3]
}

// 别名提供，方便跨模块灵活调用
pub use basis_k4_derivative as basis_k4_first_derivative;
pub use evaluate_modal_field as eval_modal_field;
pub use evaluate_modal_derivative as eval_modal_first_derivative;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_k4_values_and_derivatives() {
        let zeta = 0.5;
        let psi = basis_k4(zeta);
        assert_eq!(psi[0], 0.25);
        assert_eq!(psi[1], 0.125);
        assert_eq!(psi[2], 0.0625);
        assert_eq!(psi[3], 0.03125);

        // 导数差分验证
        let eps = 1e-6;
        let p_plus = basis_k4(zeta + eps);
        let p_minus = basis_k4(zeta - eps);
        let dpsi_ana = basis_k4_derivative(zeta);

        for k in 0..4 {
            let num = (p_plus[k] - p_minus[k]) / (2.0 * eps);
            assert!((dpsi_ana[k] - num).abs() < 1e-8);
        }

        // 二阶导数差分验证
        let dp_plus = basis_k4_derivative(zeta + eps);
        let dp_minus = basis_k4_derivative(zeta - eps);
        let d2psi_ana = basis_k4_second_derivative(zeta);

        for k in 0..4 {
            let num2 = (dp_plus[k] - dp_minus[k]) / (2.0 * eps);
            assert!((d2psi_ana[k] - num2).abs() < 1e-8);
        }
    }
}
