//! 球形有限部分 Hessian 计算与近场/远场互补积分
//! 严格依照 docs/software.md 第 7.4 节 与 docs/mathstrict.md 第 1.3 / 1.5 节
//!
//! H_{ij, ell0}^FP(x) = G rho(x) C_ij(x) ln(r0 / ell0)
//!                    + G int_{Gamma_x} T_ij(u) int_0^{r0} [ (rho(x - Ru) - rho(x)) / R ] dR dOmega
//!                    + G int_{S^2} T_ij(u) int_{E_x^far(u)} [ rho(x - Ru) / R ] dR dOmega
//!                    - (4 pi G / 3) vartheta_x rho(x) delta_ij

use std::f64::consts::PI;
use super::analytic::G_SI;
use super::density_derivatives::{eval_density, eval_density_gradient};
use super::radial_integrals::{GAUSS_LEGENDRE_16_NODES, GAUSS_LEGENDRE_16_WEIGHTS};
use super::ray_segments::RaySegment;
use super::tangent_cone::TangentCone;
use crate::model_calculation::contracts::DensityKernel;

/// 计算近场单条射线在 [0, r0] 上的差分核积分:
///   I_near = int_0^{r0} [ (rho(x - Ru) - rho(x)) / R ] dR
/// 在 R=0 端点处采用解析导数消除奇异性:
///   lim_{R->0} [ (rho(x - Ru) - rho(x)) / R ] = - grad rho(x) . u
pub fn eval_near_field_radial_integral(
    x: [f64; 3],
    u: [f64; 3],
    r0: f64,
    rho_x: f64,
    grad_rho_x: [f64; 3],
    kernels: &[DensityKernel],
) -> f64 {
    eval_near_field_radial_integral_with_background(x, u, r0, rho_x, grad_rho_x, kernels, 0.0)
}

pub fn eval_near_field_radial_integral_with_background(
    x: [f64; 3], u: [f64; 3], r0: f64, rho_x: f64,
    grad_rho_x: [f64; 3], kernels: &[DensityKernel], bulk_density: f64,
) -> f64 {
    let h = 0.5 * r0;
    let mid = 0.5 * r0;

    let grad_dot_u = grad_rho_x[0] * u[0] + grad_rho_x[1] * u[1] + grad_rho_x[2] * u[2];

    let mut sum = 0.0f64;
    for i in 0..16 {
        let node = GAUSS_LEGENDRE_16_NODES[i];
        let weight = GAUSS_LEGENDRE_16_WEIGHTS[i];
        let r = mid + h * node;

        let diff_ratio = if r > 1e-9 {
            let y = [x[0] - r * u[0], x[1] - r * u[1], x[2] - r * u[2]];
            let rho_y = bulk_density + eval_density(y, kernels);
            (rho_y - rho_x) / r
        } else {
            -grad_dot_u
        };

        sum += weight * diff_ratio;
    }

    sum * h
}

/// 计算远场单个交段 [r_min, r_max] 上的光滑积分:
///   I_far = int_{r_min}^{r_max} [ rho(x - Ru) / R ] dR
pub fn eval_far_field_radial_integral(
    x: [f64; 3],
    u: [f64; 3],
    seg: &RaySegment,
    kernels: &[DensityKernel],
) -> f64 {
    eval_far_field_radial_integral_with_background(x, u, seg, kernels, 0.0)
}

pub fn eval_far_field_radial_integral_with_background(
    x: [f64; 3], u: [f64; 3], seg: &RaySegment,
    kernels: &[DensityKernel], bulk_density: f64,
) -> f64 {
    if seg.r_max <= seg.r_min {
        return 0.0;
    }

    let h = 0.5 * (seg.r_max - seg.r_min);
    let mid = 0.5 * (seg.r_max + seg.r_min);

    let mut sum = 0.0f64;
    for i in 0..16 {
        let node = GAUSS_LEGENDRE_16_NODES[i];
        let weight = GAUSS_LEGENDRE_16_WEIGHTS[i];
        let r = mid + h * node;
        if r <= 1e-12 {
            continue;
        }

        let y = [x[0] - r * u[0], x[1] - r * u[1], x[2] - r * u[2]];
        let rho_y = bulk_density + eval_density(y, kernels);

        sum += weight * (rho_y / r);
    }

    sum * h
}

/// 组装完整的球形有限部分 Hessian 张量 H_{ij, ell0}^FP(x) (6 独立分量: xx, yy, zz, xy, xz, yz)
pub fn assemble_finite_part_hessian(
    x: [f64; 3],
    cone: &TangentCone,
    far_segments_per_dir: &[Vec<RaySegment>],
    dirs: &[([f64; 3], f64)],
    kernels: &[DensityKernel],
    r0: f64,
    ell0: f64,
) -> [f64; 6] {
    assemble_finite_part_hessian_with_background(x, cone, far_segments_per_dir, dirs, kernels, r0, ell0, 0.0)
}

pub fn assemble_finite_part_hessian_with_background(
    x: [f64; 3], cone: &TangentCone, far_segments_per_dir: &[Vec<RaySegment>],
    dirs: &[([f64; 3], f64)], kernels: &[DensityKernel], r0: f64, ell0: f64,
    bulk_density: f64,
) -> [f64; 6] {
    let rho_x = bulk_density + eval_density(x, kernels);
    let grad_x = eval_density_gradient(x, kernels);

    let mut h = [0.0f64; 6];

    // 1. 对数项: G rho(x) C_ij(x) ln(r0 / ell0)
    let log_ratio = (r0 / ell0).ln();
    let log_factor = G_SI * rho_x * log_ratio;
    for k in 0..6 {
        h[k] += log_factor * cone.cone_tensor[k];
    }

    // 2. 近场无奇异差分角向积分: G int_{Gamma_x} T_ij(u) I_near(u) dOmega
    for (dir_idx, (u, omega)) in dirs.iter().enumerate() {
        if !cone.in_cone_mask[dir_idx] {
            continue;
        }

        let i_near = eval_near_field_radial_integral_with_background(x, *u, r0, rho_x, grad_x, kernels, bulk_density);

        let t_xx = 3.0 * u[0] * u[0] - 1.0;
        let t_yy = 3.0 * u[1] * u[1] - 1.0;
        let t_zz = 3.0 * u[2] * u[2] - 1.0;
        let t_xy = 3.0 * u[0] * u[1];
        let t_xz = 3.0 * u[0] * u[2];
        let t_yz = 3.0 * u[1] * u[2];

        let weight = G_SI * omega * i_near;
        h[0] += weight * t_xx;
        h[1] += weight * t_yy;
        h[2] += weight * t_zz;
        h[3] += weight * t_xy;
        h[4] += weight * t_xz;
        h[5] += weight * t_yz;
    }

    // 3. 远场角向积分: G int_{S^2} T_ij(u) [ sum_segs I_far ] dOmega
    for (dir_idx, (u, omega)) in dirs.iter().enumerate() {
        let segs = &far_segments_per_dir[dir_idx];
        if segs.is_empty() {
            continue;
        }

        let mut i_far_total = 0.0f64;
        for seg in segs {
            i_far_total += eval_far_field_radial_integral_with_background(x, *u, seg, kernels, bulk_density);
        }

        let t_xx = 3.0 * u[0] * u[0] - 1.0;
        let t_yy = 3.0 * u[1] * u[1] - 1.0;
        let t_zz = 3.0 * u[2] * u[2] - 1.0;
        let t_xy = 3.0 * u[0] * u[1];
        let t_xz = 3.0 * u[0] * u[2];
        let t_yz = 3.0 * u[1] * u[2];

        let weight = G_SI * omega * i_far_total;
        h[0] += weight * t_xx;
        h[1] += weight * t_yy;
        h[2] += weight * t_zz;
        h[3] += weight * t_xy;
        h[4] += weight * t_xz;
        h[5] += weight * t_yz;
    }

    // 4. 接触分布对角项: - (4 pi G / 3) vartheta_x rho(x) delta_ij
    let contact_term = (4.0 * PI * G_SI / 3.0) * cone.solid_angle_fraction * rho_x;
    h[0] -= contact_term;
    h[1] -= contact_term;
    h[2] -= contact_term;

    h
}

/// 尺度变换恒等式核验: H_{ell1} - H_{ell0} = G rho(x) C_ij(x) ln(ell0 / ell1)
pub fn scale_transform_difference(
    rho_x: f64,
    cone: &TangentCone,
    ell0: f64,
    ell1: f64,
) -> [f64; 6] {
    let factor = G_SI * rho_x * (ell0 / ell1).ln();
    [
        factor * cone.cone_tensor[0],
        factor * cone.cone_tensor[1],
        factor * cone.cone_tensor[2],
        factor * cone.cone_tensor[3],
        factor * cone.cone_tensor[4],
        factor * cone.cone_tensor[5],
    ]
}
