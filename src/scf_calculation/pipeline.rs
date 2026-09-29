//! 完整 SCF (Self-Consistent Field) 几何自洽场管线编排
//! 严格依照 docs/software.md 第 11 节 与 docs/mathstrict.md 第 10.4 节
//!
//! 闭环链条：
//!   q^(n) -> m^(n+1/2) -> q^(n+1/2) -> dot_gamma -> I_c -> phi^(n) -> R_*^(n) -> R^(n+1) -> (g, H)

use crate::model_calculation::equipotential::solve_equipotential_radii_batch;
use crate::model_calculation::mass_level::solve_mass_level;
use crate::rheology::cohesion::{cohesive_pressure, effective_pressure};
use crate::rheology::friction_angle::{effective_projection_mu, friction_angle, sin2_friction_angle};
use crate::rheology::friction_law::mu_dynamic;
use crate::rheology::inertial_number::modified_inertial_number;
use crate::rheology::state::{RheologyParams, RheologyState};
use crate::scf_calculation::conservation::conservative_modal_update_batch;
use crate::scf_calculation::conservative_state::{
    compute_conservative_states_batch, recover_physical_states_batch,
};
use crate::scf_calculation::convergence::{
    check_scf_convergence, compute_geometry_residual, compute_mass_residual,
    compute_slope_residual, compute_transport_residual, ScfResiduals, ScfTolerances,
};
use crate::scf_calculation::galerkin_matrices::{assemble_mass_matrix, assemble_mass_matrix_inverse};
use crate::scf_calculation::mass_constraint::{evaluate_mass_at_radii, linearize_mass_constraint};
use crate::rheology::strain_rate::{compute_deviatoric_strain_rate, compute_shear_rate, compute_strain_rate};
use crate::scf_calculation::sqp_projection::{check_true_constraints_with_current, solve_convex_qp};
use crate::scf_calculation::under_relaxation::apply_under_relaxation;
use crate::error_analysis::{check_slope_constraint, check_transport_conservation};

/// SCF 单步迭代输入数据与冻结状态
pub struct ScfStepInput<'a> {
    pub current_radii: &'a [f64],
    pub current_modal_q: &'a [[f64; 4]],
    pub directions: &'a [[f64; 3]],
    pub weights: &'a [f64],
    pub nominal_normals: &'a [[f64; 3]],
    pub effective_gravities: &'a [[f64; 3]],
    pub t_tide_norms: &'a [f64],
    pub rheology_state: &'a RheologyState,
    pub rheology_params: &'a RheologyParams,
    pub target_mass: f64,
    pub relaxation_lambda: f64,
    pub trust_radius: f64,
}

/// SCF 单步迭代输出结果
#[derive(Debug, Clone)]
pub struct ScfStepOutput {
    /// 迭代后新半径序列 R^(n+1)
    pub next_radii: Vec<f64>,
    /// 输运后新物理模态序列 q^(n+1)
    pub next_modal_q: Vec<[f64; 4]>,
    /// 候选等势半径序列 R_eq
    pub r_equipotential: Vec<f64>,
    /// QP 投影中间解 R_*
    pub r_star: Vec<f64>,
    /// 表面物理摩擦角序列 phi
    pub phi_list: Vec<f64>,
    /// 各项收敛残差诊断
    pub residuals: ScfResiduals,
    /// 是否满足收敛判据
    pub is_converged: bool,
    /// 诊断说明信息
    pub diagnostics: Vec<String>,
}

/// 执行单步完整 SCF 循环迭代
pub fn run_scf_iteration_step<P, D>(
    input: &ScfStepInput,
    potential_fn: &P,
    density_fn: &D,
    tols: &ScfTolerances,
) -> Result<ScfStepOutput, String>
where
    P: Fn([f64; 3]) -> f64,
    D: Fn([f64; 3]) -> f64,
{
    let n = input.current_radii.len();
    let m_mat = assemble_mass_matrix();
    let inv_m = assemble_mass_matrix_inverse();

    // 1. 物理模态转守恒模态 m = M q
    let m_curr = compute_conservative_states_batch(&m_mat, input.current_modal_q);

    // 2. 冻结几何上的半拉格朗日模态更新与曲面守恒修正
    // (此处在离散球面采用单位 Jacobian，支持无源纯对流推进)
    let gamma_from_radius: Vec<f64> = input.current_radii.iter().map(|r| r * r).collect();
    let mapping_jacobians: Vec<f64> = input.current_radii.iter().map(|r| {
        let rr = r.abs().max(1e-12);
        // 球面参数化下的面积尺度 dA ~ r^2；保留正值并避免退化放大。
        (rr * rr).recip().min(1.0e12)
    }).collect();
    let m_transported = conservative_modal_update_batch(
        &gamma_from_radius,
        &gamma_from_radius,
        &mapping_jacobians,
        &m_curr,
        None,
        0.01,
    );
    let q_half = recover_physical_states_batch(&inv_m, &m_transported);

    // 3. 流变闭合: 表面剪切率 -> I_c -> mu_d -> mu_eff -> phi
    let mut phi_list = Vec::with_capacity(n);
    let mut sin2_phi_list = Vec::with_capacity(n);

    let p_c = cohesive_pressure(
        input.rheology_params.c_c,
        input.rheology_state.coordination_number,
        input.rheology_state.solid_fraction,
        input.rheology_params.cohesion_force,
        input.rheology_state.grain_diameter,
    );
    let p_star = effective_pressure(input.rheology_state.pressure, p_c);

    for i in 0..n {
        // 利用 K=4 径向模态差分特征评估局部剪切率
        let grad_q2 = [0.01 * q_half[i][1], 0.0];
        let grad_q3 = [0.005 * q_half[i][2], 0.0];
        let grad_q4 = [0.002 * q_half[i][3], 0.0];
        let mut velocity_gradient = [[0.0; 3]; 3];
        velocity_gradient[0][1] = grad_q2[0] + grad_q3[0] + grad_q4[0];
        velocity_gradient[1][0] = grad_q2[1] + grad_q3[1] + grad_q4[1];
        velocity_gradient[2][0] = grad_q3[0] + grad_q4[0];
        let d = compute_strain_rate(&velocity_gradient);
        let d_prime = compute_deviatoric_strain_rate(&d);
        let shear_rate = compute_shear_rate(&d_prime).max(1e-12);

        let i_c = modified_inertial_number(
            input.rheology_state.grain_diameter,
            shear_rate,
            p_star,
            input.rheology_state.grain_density,
        );
        let mu_d = mu_dynamic(
            i_c,
            input.rheology_params.mu_s,
            input.rheology_params.mu_2,
            input.rheology_params.i_0,
        );

        let mu_eff = effective_projection_mu(
            input.rheology_params.cohesion_macro,
            input.rheology_params.mu_proj.max(mu_d),
            input.rheology_state.pressure,
            input.t_tide_norms[i],
            p_star,
        );
        let phi = friction_angle(mu_eff);
        let s2 = sin2_friction_angle(mu_eff);

        phi_list.push(phi);
        sin2_phi_list.push(s2);
    }

    // 4. 质量能级 W_0 求解与等势候选面 R_eq
    let w0 = solve_mass_level(
        input.target_mass,
        input.directions,
        input.weights,
        potential_fn,
        density_fn,
        -1e7,
        1e7,
        1.0,
        2000.0,
        input.current_radii[0],
        input.target_mass * 1e-3,
        50,
    ).map_err(|e| format!("质量能级求解失败: {e}"))?;

    let (r_eq, _) = solve_equipotential_radii_batch(
        input.directions,
        w0,
        potential_fn,
        1.0,
        2000.0,
        input.current_radii[0],
        1e-4,
    );

    // 5. 线性化质量约束与 SQP/QP 投影
    let mass_constraint = linearize_mass_constraint(
        input.current_radii,
        input.directions,
        input.weights,
        density_fn,
        input.target_mass,
    );

    let r_star = solve_convex_qp(
        input.current_radii,
        &r_eq,
        input.weights,
        &mass_constraint,
        input.effective_gravities,
        input.nominal_normals,
        input.directions,
        &sin2_phi_list,
        input.trust_radius,
        10.0,
        1000.0,
    );

    // 6. 欠松弛更新: R^(n+1) = (1 - lambda) R^(n) + lambda R_*
    let next_radii = apply_under_relaxation(
        input.current_radii,
        &r_star,
        input.relaxation_lambda,
    );

    // 7. 残差诊断与收敛性检查
    let geom_res = compute_geometry_residual(&next_radii, input.current_radii);
    let actual_mass = evaluate_mass_at_radii(&next_radii, input.directions, input.weights, density_fn);
    let mass_res = compute_mass_residual(actual_mass, input.target_mass);

    let mut sin2_slopes = Vec::with_capacity(n);
    for i in 0..n {
        let (_, s2, _) = crate::scf_calculation::slope_closure::check_slope_violation(
            input.effective_gravities[i],
            input.nominal_normals[i],
            phi_list[i],
        );
        sin2_slopes.push(s2);
    }
    let slope_res = compute_slope_residual(&sin2_slopes, &sin2_phi_list);
    let trans_res = compute_transport_residual(&m_transported, &m_curr);
    let transport_before: Vec<f64> = m_curr.iter().map(|m| m.iter().sum()).collect();
    let transport_after: Vec<f64> = m_transported.iter().map(|m| m.iter().sum()).collect();
    check_transport_conservation(&transport_before, &transport_after, tols.transport_tol)
        .map_err(|e| format!("输运守恒审计失败: {e}"))?;

    // 检验真实约束
    let true_eval = check_true_constraints_with_current(
        input.current_radii,
        &r_star,
        &r_eq,
        input.weights,
        input.directions,
        input.nominal_normals,
        input.effective_gravities,
        &phi_list,
        density_fn,
        input.target_mass,
        tols.mass_tol,
    );
    if !true_eval.accepted {
        return Err(format!("SQP 候选解未通过真实约束验收: {}", true_eval.message));
    }
    check_slope_constraint(&sin2_slopes, &sin2_phi_list, tols.slope_tol)
        .map_err(|e| format!("坡度约束审计失败: {e}"))?;

    let residuals = ScfResiduals {
        geometry_residual: geom_res,
        mass_residual: mass_res,
        slope_residual: slope_res,
        transport_residual: trans_res,
        kkt_residual: true_eval.kkt_residual,
    };

    let (is_converged, diagnostics) = check_scf_convergence(&residuals, tols);

    Ok(ScfStepOutput {
        next_radii,
        next_modal_q: q_half,
        r_equipotential: r_eq,
        r_star,
        phi_list,
        residuals,
        is_converged,
        diagnostics,
    })
}

/// 严格的外层 SCF 驱动器：每轮只提交上一轮已验收的状态，达到收敛或
/// 明确返回最大迭代次数错误；UI 层可在 callback 中处理暂停/取消。
pub fn run_scf_until_converged<P, D, F>(
    input: ScfStepInput<'_>,
    potential_fn: &P,
    density_fn: &D,
    tols: &ScfTolerances,
    max_iterations: usize,
    mut callback: F,
) -> Result<ScfStepOutput, String>
where
    P: Fn([f64; 3]) -> f64,
    D: Fn([f64; 3]) -> f64,
    F: FnMut(usize, &ScfStepOutput) -> Result<(), String>,
{
    if max_iterations == 0 { return Err("SCF 最大迭代次数必须大于零".into()); }
    let mut radii = input.current_radii.to_vec();
    let mut modal_q = input.current_modal_q.to_vec();
    for iteration in 0..max_iterations {
        let step = ScfStepInput {
            current_radii: &radii, current_modal_q: &modal_q,
            directions: input.directions, weights: input.weights,
            nominal_normals: input.nominal_normals,
            effective_gravities: input.effective_gravities,
            t_tide_norms: input.t_tide_norms,
            rheology_state: input.rheology_state, rheology_params: input.rheology_params,
            target_mass: input.target_mass, relaxation_lambda: input.relaxation_lambda,
            trust_radius: input.trust_radius,
        };
        let output = run_scf_iteration_step(&step, potential_fn, density_fn, tols)?;
        callback(iteration, &output)?;
        if output.is_converged { return Ok(output); }
        radii = output.next_radii;
        modal_q = output.next_modal_q;
    }
    Err(format!("SCF 在 {max_iterations} 轮内未收敛"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_scf_single_step() {
        let n = 6;
        let r0 = 100.0_f64;
        let radii = vec![r0; n];
        let q_init = vec![[1.0, 0.0, 0.0, 0.0]; n];
        let weights = vec![4.0 * PI / n as f64; n];

        let dirs = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let normals = dirs.clone();
        let gravities = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];
        let tides = vec![0.0; n];

        let rho0 = 2000.0;
        let density_fn = |_pos: [f64; 3]| -> f64 { rho0 };
        let potential_fn = |pos: [f64; 3]| -> f64 {
            let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(1.0);
            -1.0e8 / r
        };

        let target_mass = 4.0 / 3.0 * PI * r0.powi(3) * rho0;
        let state = RheologyState::new(1000.0, 0.6, 6.0, 0.01, rho0).unwrap();
        let params = RheologyParams::default();

        let input = ScfStepInput {
            current_radii: &radii,
            current_modal_q: &q_init,
            directions: &dirs,
            weights: &weights,
            nominal_normals: &normals,
            effective_gravities: &gravities,
            t_tide_norms: &tides,
            rheology_state: &state,
            rheology_params: &params,
            target_mass,
            relaxation_lambda: 0.3,
            trust_radius: 5.0,
        };

        let tols = ScfTolerances::default();
        let output = run_scf_iteration_step(&input, &potential_fn, &density_fn, &tols)
            .expect("SCF 步迭代成功");

        assert_eq!(output.next_radii.len(), n);
        assert_eq!(output.next_modal_q.len(), n);
        assert!(output.residuals.geometry_residual >= 0.0);
    }
}
