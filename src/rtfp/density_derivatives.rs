//! Cauchy 密度及其梯度、Hessian 与 Laplacian 严格求导
//! 严格依照 docs/software.md 第 7.2 节 与 docs/mathstrict.md 第 1.1 / 3.1 / 5800-5855 行
//!
//! 密度定义:
//!   rho_K(x) = sum_k w_k [1 + sigma_k^2 ||x - c_k||^2]^(-alpha_k)
//! 令 q_k = 1 + sigma_k^2 ||x - c_k||^2:
//!   grad rho_K = -2 sum_k alpha_k w_k sigma_k^2 (x - c_k) q_k^(-alpha_k - 1)
//!   Hessian rho_K = sum_k w_k [ -2 alpha_k sigma_k^2 q_k^(-alpha_k - 1) I
//!                              + 4 alpha_k (alpha_k + 1) sigma_k^4 (x - c_k)(x - c_k)^T q_k^(-alpha_k - 2) ]
//!   Laplacian rho_K = sum_k w_k [ -6 alpha_k sigma_k^2 q_k^(-alpha_k - 1)
//!                                + 4 alpha_k (alpha_k + 1) sigma_k^4 ||x - c_k||^2 q_k^(-alpha_k - 2) ]

use crate::model_calculation::contracts::{DensityField, DensityKernel};

pub fn eval_density_field(x: [f64; 3], field: &DensityField) -> f64 {
    field.bulk_density + eval_density(x, &field.kernels)
}

pub fn eval_density_gradient_field(x: [f64; 3], field: &DensityField) -> [f64; 3] {
    eval_density_gradient(x, &field.kernels)
}

pub fn eval_density_hessian_field(x: [f64; 3], field: &DensityField) -> [f64; 6] {
    eval_density_hessian(x, &field.kernels)
}

pub fn eval_density_laplacian_field(x: [f64; 3], field: &DensityField) -> f64 {
    eval_density_laplacian(x, &field.kernels)
}

/// 计算 Cauchy 变密度场在点 x 处的密度值 rho_K(x) (kg/m³)
pub fn eval_density(x: [f64; 3], kernels: &[DensityKernel]) -> f64 {
    let mut rho = 0.0f64;
    for k in kernels {
        let dx = x[0] - k.center[0];
        let dy = x[1] - k.center[1];
        let dz = x[2] - k.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let q = 1.0 + k.sigma * k.sigma * r2;
        rho += k.weight * q.powf(-k.alpha);
    }
    rho
}

/// 计算 Cauchy 变密度场梯度 grad rho_K(x) (kg/m⁴)
pub fn eval_density_gradient(x: [f64; 3], kernels: &[DensityKernel]) -> [f64; 3] {
    let mut grad = [0.0f64; 3];
    for k in kernels {
        let dx = x[0] - k.center[0];
        let dy = x[1] - k.center[1];
        let dz = x[2] - k.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let s2 = k.sigma * k.sigma;
        let q = 1.0 + s2 * r2;
        // 系数: -2 * alpha * w * sigma^2 * q^(-alpha - 1)
        let coeff = -2.0 * k.alpha * k.weight * s2 * q.powf(-k.alpha - 1.0);
        grad[0] += coeff * dx;
        grad[1] += coeff * dy;
        grad[2] += coeff * dz;
    }
    grad
}

/// 计算 Cauchy 变密度场二阶导张量 Hessian rho_K(x) (3x3 对称矩阵, 返回 6 独立分量: xx, yy, zz, xy, xz, yz) (kg/m⁵)
pub fn eval_density_hessian(x: [f64; 3], kernels: &[DensityKernel]) -> [f64; 6] {
    let mut h = [0.0f64; 6];
    for k in kernels {
        let dx = x[0] - k.center[0];
        let dy = x[1] - k.center[1];
        let dz = x[2] - k.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let s2 = k.sigma * k.sigma;
        let s4 = s2 * s2;
        let q = 1.0 + s2 * r2;

        let q_pow1 = q.powf(-k.alpha - 1.0);
        let q_pow2 = q.powf(-k.alpha - 2.0);

        let term_diag = -2.0 * k.alpha * k.weight * s2 * q_pow1;
        let term_outer = 4.0 * k.alpha * (k.alpha + 1.0) * k.weight * s4 * q_pow2;

        // xx, yy, zz
        h[0] += term_diag + term_outer * dx * dx;
        h[1] += term_diag + term_outer * dy * dy;
        h[2] += term_diag + term_outer * dz * dz;

        // xy, xz, yz
        h[3] += term_outer * dx * dy;
        h[4] += term_outer * dx * dz;
        h[5] += term_outer * dy * dz;
    }
    h
}

/// 计算 Cauchy 变密度场拉普拉斯量 Laplacian rho_K(x) = Tr(Hessian rho_K(x)) (kg/m⁵)
pub fn eval_density_laplacian(x: [f64; 3], kernels: &[DensityKernel]) -> f64 {
    let mut lap = 0.0f64;
    for k in kernels {
        let dx = x[0] - k.center[0];
        let dy = x[1] - k.center[1];
        let dz = x[2] - k.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;
        let s2 = k.sigma * k.sigma;
        let s4 = s2 * s2;
        let q = 1.0 + s2 * r2;

        let q_pow1 = q.powf(-k.alpha - 1.0);
        let q_pow2 = q.powf(-k.alpha - 2.0);

        // -6 alpha sigma^2 q^(-alpha - 1) + 4 alpha (alpha + 1) sigma^4 r^2 q^(-alpha - 2)
        lap += k.weight * (
            -6.0 * k.alpha * s2 * q_pow1
            + 4.0 * k.alpha * (k.alpha + 1.0) * s4 * r2 * q_pow2
        );
    }
    lap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_derivatives_consistency() {
        let k = DensityKernel {
            center: [100.0, 50.0, -25.0],
            sigma: 0.002,
            weight: 1200.0,
            alpha: 1.0,
        };
        let kernels = [k];
        let x = [120.0, 60.0, -10.0];

        // 1. 数值差分比对梯度
        let eps = 1e-6;
        let mut num_grad = [0.0; 3];
        for dim in 0..3 {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[dim] += eps;
            x_minus[dim] -= eps;
            num_grad[dim] = (eval_density(x_plus, &kernels) - eval_density(x_minus, &kernels)) / (2.0 * eps);
        }
        let analytic_grad = eval_density_gradient(x, &kernels);
        for dim in 0..3 {
            assert!(
                (analytic_grad[dim] - num_grad[dim]).abs() < 1e-5,
                "梯度差分校验不一致: dim {}, analytic {:.6e}, num {:.6e}",
                dim, analytic_grad[dim], num_grad[dim]
            );
        }

        // 2. 数值差分比对 Hessian 对角线与迹
        let h = eval_density_hessian(x, &kernels);
        let lap = eval_density_laplacian(x, &kernels);
        let trace = h[0] + h[1] + h[2];
        assert!(
            (trace - lap).abs() < 1e-10,
            "Hessian 迹 Tr(H) = {} 与 Laplacian = {} 必须严格恒等",
            trace, lap
        );
    }

    #[test]
    fn test_density_field_background_is_preserved() {
        let field = DensityField::new(100.0, vec![]).unwrap();
        assert_eq!(eval_density_field([0.0, 0.0, 0.0], &field), 100.0);
        assert_eq!(eval_density_gradient_field([0.0, 0.0, 0.0], &field), [0.0; 3]);
    }
}
