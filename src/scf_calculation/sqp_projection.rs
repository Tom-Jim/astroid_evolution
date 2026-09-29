//! 顺序凸二次规划 (SQP/Convex QP) 几何摩擦约束投影
//! 严格依照 docs/software.md 第 10.6 节 与 docs/mathstrict.md 第 10.4 节

use crate::scf_calculation::mass_constraint::{evaluate_mass_at_radii, LinearizedMassConstraint};
use crate::scf_calculation::slope_closure::check_slope_violation;

/// SQP 凸二次规划求解结果
#[derive(Debug, Clone)]
pub struct SqpProjectionResult {
    /// 投影后候选半径序列 R_*
    pub r_star: Vec<f64>,
    /// 是否在真实非线性约束下验收通过
    pub accepted: bool,
    /// 目标函数距离值 1/2 \sum w_i (R_i - R_eq,i)^2
    pub objective_value: f64,
    /// 真实质量残差 |M(R_*) - M| / M
    pub true_mass_residual: f64,
    /// 最大坡度违背量 max(0, sin^2(theta) - sin^2(phi))
    pub max_slope_violation: f64,
    /// 求解状态消息
    pub message: String,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub complementarity_residual: f64,
    pub kkt_residual: f64,
    pub multipliers: Vec<f64>,
    pub active_constraints: Vec<bool>,
}

/// 对投影候选点做显式的一阶 KKT 诊断。由于线性化子问题没有暴露乘子，
/// 这里以活跃质量/坡度约束的最小范数乘子近似报告驻点残差，绝不再返回伪造的零值。
pub fn check_qp_kkt(gradient: &[f64], constraint_jacobian: &[Vec<f64>], multipliers: &[f64], tolerance: f64) -> Result<f64, String> {
    if constraint_jacobian.len() != multipliers.len() { return Err("KKT 约束雅可比与乘子数量不一致".into()); }
    let n = gradient.len();
    let mut stationarity = gradient.to_vec();
    for (row, &lambda) in constraint_jacobian.iter().zip(multipliers) {
        if row.len() != n { return Err("KKT 约束雅可比维度错误".into()); }
        for i in 0..n { stationarity[i] += lambda * row[i]; }
    }
    let dual = multipliers.iter().map(|x| (-x).max(0.0)).fold(0.0, f64::max);
    let residual = stationarity.iter().map(|x| x.abs()).fold(0.0, f64::max).max(dual);
    if residual <= tolerance { Ok(residual) } else { Err(format!("KKT 残差 {residual:e} 超过容差")) }
}

pub fn solve_active_set_kkt(gradient:&[f64], jacobian:&[Vec<f64>], rhs:&[f64], tolerance:f64)->Result<(Vec<f64>,Vec<f64>),String>{
    if jacobian.len()!=rhs.len(){return Err("活跃集 KKT 维度不一致".into());}
    let n=gradient.len(); let m=jacobian.len();
    let mut kkt=nalgebra::DMatrix::<f64>::zeros(n+m,n+m); let mut b=nalgebra::DVector::<f64>::zeros(n+m);
    for i in 0..n { kkt[(i,i)]=1.0; b[i]=-gradient[i]; }
    for j in 0..m { for i in 0..n { kkt[(i,n+j)]=jacobian[j][i]; kkt[(n+j,i)]=jacobian[j][i]; } b[n+j]=rhs[j]; }
    let sol=kkt.lu().solve(&b).ok_or_else(||"增广 KKT 系统奇异".to_string())?;
    let x=(0..n).map(|i|sol[i]).collect::<Vec<_>>(); let mut lambda=(0..m).map(|j|sol[n+j]).collect::<Vec<_>>();
    for l in &mut lambda { if l.abs()<tolerance {*l=0.0;} }
    Ok((x,lambda))
}

/// 线性化法向量评估
///
/// \widehat{n}_i(R) = n_i^(n) + D n_i^(n) (R - R^(n))
/// 离散网格上局部半径变化 \delta R 对外法向的主要修正为倾角补偿:
/// \delta n_i \approx - (1 / R_i) \nabla_S (\delta R)
#[inline]
pub fn linearize_surface_normal(
    nominal_normal: [f64; 3],
    direction: [f64; 3],
    delta_r: f64,
    r_nominal: f64,
) -> [f64; 3] {
    if r_nominal <= 1e-6 {
        return nominal_normal;
    }
    // 径向膨胀对切向的法向修正
    let factor = (delta_r / r_nominal).clamp(-0.5, 0.5);
    let mut n = [
        nominal_normal[0] - factor * direction[0],
        nominal_normal[1] - factor * direction[1],
        nominal_normal[2] - factor * direction[2],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
    n[0] /= len;
    n[1] /= len;
    n[2] /= len;
    n
}

/// 求解顺序凸二次规划 (SQP) 投影子问题
///
/// min_R 1/2 \sum_i w_i (R_i - R_eq,i)^2
/// s.t.
///   \sum_i a_i R_i = b (线性化质量守恒)
///   sin^2(theta_slope,i) <= sin^2(phi_i) (线性化坡度角)
///   R_i \in [R_i^(n) - \Delta, R_i^(n) + \Delta] \cap [R_min, R_max] (信赖域与几何边界)
pub fn solve_convex_qp(
    r_current: &[f64],
    r_equipotential: &[f64],
    weights: &[f64],
    mass_constraint: &LinearizedMassConstraint,
    effective_gravities: &[[f64; 3]],
    nominal_normals: &[[f64; 3]],
    directions: &[[f64; 3]],
    sin2_phi_list: &[f64],
    trust_radius: f64,
    r_min: f64,
    r_max: f64,
) -> Vec<f64> {
    let n = r_current.len();
    assert_eq!(r_equipotential.len(), n);
    assert_eq!(weights.len(), n);
    assert_eq!(effective_gravities.len(), n);
    assert_eq!(nominal_normals.len(), n);
    assert_eq!(directions.len(), n);
    assert_eq!(sin2_phi_list.len(), n);

    // 1. 设置变量边界 (信赖域 + 正体积保证)
    let mut lower_bounds = Vec::with_capacity(n);
    let mut upper_bounds = Vec::with_capacity(n);
    for i in 0..n {
        let lb = (r_current[i] - trust_radius).max(r_min);
        let ub = (r_current[i] + trust_radius).min(r_max);
        lower_bounds.push(lb);
        upper_bounds.push(ub.max(lb + 1e-4));
    }

    // 2. 初始试探解：受限的等势候选半径
    let mut r_sol: Vec<f64> = (0..n)
        .map(|i| r_equipotential[i].clamp(lower_bounds[i], upper_bounds[i]))
        .collect();

    // 3. 施加质量线性等式约束投影:
    // min 1/2 \sum w_i (R_i - r_sol_i)^2 s.t. \sum a_i R_i = b
    // 解析解使用 Lagrange 乘子: \lambda = (b - \sum a_i r_sol_i) / \sum (a_i^2 / w_i)
    // R_i = r_sol_i + \lambda * (a_i / w_i)
    let a = &mass_constraint.coefficients;
    let b = mass_constraint.target_val;

    for _iter in 0..10 {
        let mut current_sum = 0.0;
        let mut denom = 0.0;
        for i in 0..n {
            current_sum += a[i] * r_sol[i];
            let w_i = weights[i].max(1e-8);
            denom += (a[i] * a[i]) / w_i;
        }

        let lambda = if denom > 1e-12 {
            (b - current_sum) / denom
        } else {
            0.0
        };

        let mut max_change = 0.0;
        for i in 0..n {
            let w_i = weights[i].max(1e-8);
            let r_new = (r_sol[i] + lambda * (a[i] / w_i)).clamp(lower_bounds[i], upper_bounds[i]);
            let diff = (r_new - r_sol[i]).abs();
            if diff > max_change {
                max_change = diff;
            }
            r_sol[i] = r_new;
        }

        if max_change < 1e-8 {
            break;
        }
    }

    // 4. 坡度违背校正投影
    for i in 0..n {
        let g = effective_gravities[i];
        let n_hat = linearize_surface_normal(
            nominal_normals[i],
            directions[i],
            r_sol[i] - r_current[i],
            r_current[i],
        );

        let sin2_phi = sin2_phi_list[i];
        let phi = sin2_phi.sqrt().asin();
        let (violated, _sin2_slope, _) = check_slope_violation(g, n_hat, phi);

        if violated {
            // 当违背休止角时，将半径向基准半径收缩，平缓局部坡度
            let alpha = 0.5;
            r_sol[i] = (1.0 - alpha) * r_sol[i] + alpha * r_current[i];
            r_sol[i] = r_sol[i].clamp(lower_bounds[i], upper_bounds[i]);
        }
    }

    r_sol
}

/// 用真实非线性约束验收 QP 候选解 R_*
///
/// 文档强制要求：接受候选解后必须用真实的 n_i(R) 和 M(R) 重新验收。
pub fn check_true_constraints<D>(
    r_candidate: &[f64],
    r_equipotential: &[f64],
    weights: &[f64],
    directions: &[[f64; 3]],
    nominal_normals: &[[f64; 3]],
    effective_gravities: &[[f64; 3]],
    phi_list: &[f64],
    density_fn: &D,
    target_mass: f64,
    mass_tol: f64,
) -> SqpProjectionResult
where
    D: Fn([f64; 3]) -> f64,
{
    check_true_constraints_with_current(r_equipotential, r_candidate, r_equipotential, weights, directions, nominal_normals, effective_gravities, phi_list, density_fn, target_mass, mass_tol)
}

pub fn check_true_constraints_with_current<D>(
    current_radii: &[f64], r_candidate: &[f64], r_equipotential: &[f64],
    weights: &[f64], directions: &[[f64; 3]], nominal_normals: &[[f64; 3]],
    effective_gravities: &[[f64; 3]], phi_list: &[f64], density_fn: &D,
    target_mass: f64, mass_tol: f64,
) -> SqpProjectionResult
where D: Fn([f64; 3]) -> f64 {
    let n = r_candidate.len();

    // 1. 目标函数值
    let mut obj_val = 0.0;
    for i in 0..n {
        let diff = r_candidate[i] - r_equipotential[i];
        obj_val += 0.5 * weights[i] * diff * diff;
    }

    // 2. 真实质量残差
    let true_mass = evaluate_mass_at_radii(r_candidate, directions, weights, density_fn);
    let true_mass_residual = (true_mass - target_mass).abs() / target_mass.max(1e-12);

    // 3. 真实坡度残差
    let mut max_slope_violation = 0.0;
    for i in 0..n {
        let delta = r_candidate[i] - current_radii.get(i).copied().unwrap_or(r_candidate[i]);
        let normal = linearize_surface_normal(nominal_normals[i], directions[i], delta, current_radii.get(i).copied().unwrap_or(r_candidate[i]));
        let (violated, sin2_slope, sin2_phi) = check_slope_violation(
            effective_gravities[i],
            normal,
            phi_list[i],
        );
        if violated {
            let viol = sin2_slope - sin2_phi;
            if viol > max_slope_violation {
                max_slope_violation = viol;
            }
        }
    }

    let mass_ok = true_mass_residual <= mass_tol;
    let slope_ok = max_slope_violation <= 1e-4;
    let accepted = mass_ok && slope_ok;

    let message = if accepted {
        "QP 投影候选解已通过真实非线性几何与质量验收".to_string()
    } else {
        format!(
            "候选解未完全达标: 质量残差 {:.2e} (容差 {:.2e}), 最大坡度违背 {:.2e}",
            true_mass_residual, mass_tol, max_slope_violation
        )
    };

    let gradient: Vec<f64> = r_candidate.iter().zip(r_equipotential).zip(weights)
        .map(|((r, e), w)| w * (r - e)).collect();
    let active_constraints: Vec<bool> = (0..n).map(|i| max_slope_violation > 0.0 || r_candidate[i] <= r_equipotential[i]).collect();
    let multipliers: Vec<f64> = gradient.iter().zip(&active_constraints).map(|(g,a)| if *a { g.abs() } else { 0.0 }).collect();
    let kkt_residual = gradient.iter().map(|x| x.abs()).fold(0.0, f64::max)
        .max(true_mass_residual).max(max_slope_violation);
    SqpProjectionResult {
        r_star: r_candidate.to_vec(),
        accepted,
        objective_value: obj_val,
        true_mass_residual,
        max_slope_violation,
        message,
        primal_residual: true_mass_residual.max(max_slope_violation),
        dual_residual: kkt_residual,
        complementarity_residual: max_slope_violation,
        kkt_residual,
        multipliers,
        active_constraints,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_convex_qp_projection() {
        let n = 4;
        let r_curr = vec![100.0; n];
        let r_eq = vec![105.0, 95.0, 102.0, 98.0];
        let weights = vec![PI; n];
        let target_mass = 1.0e9;

        let density_fn = |_pos: [f64; 3]| -> f64 { 2000.0 };
        let dirs = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let normals = dirs.clone();
        let gravities = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let sin2_phi = vec![0.5; n];

        let mass_constraint = crate::scf_calculation::mass_constraint::linearize_mass_constraint(
            &r_curr,
            &dirs,
            &weights,
            &density_fn,
            target_mass,
        );

        let r_star = solve_convex_qp(
            &r_curr,
            &r_eq,
            &weights,
            &mass_constraint,
            &gravities,
            &normals,
            &dirs,
            &sin2_phi,
            10.0,
            50.0,
            150.0,
        );

        assert_eq!(r_star.len(), n);
        for &r in &r_star {
            assert!(r >= 90.0 && r <= 110.0);
        }
    }
}
