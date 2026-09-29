//! 阶段 6 完整集成测试套件：自洽场 (SCF) 闭环、质量能级、等势候选面与 SQP/QP 投影
//! 严格依照 docs/software.md 第 8、10.6、11、15.6 节 与 docs/mathstrict.md 第 8、10.4 节
//!
//! 覆盖阶段 6 推荐实现顺序的 5 个关键步骤：
//! 1. 质量能级 M(W_0) 连续性、单调性与二分求解
//! 2. 等势候选面 W(R_eq u) = W_0 径向连续跟踪与分支选择
//! 3. 顺序凸二次规划 (SQP/QP) 几何摩擦约束投影与真实非线性验收
//! 4. 欠松弛 R^(n+1) = (1 - \lambda) R^(n) + \lambda R_* 与自适应阻尼
//! 5. 几何、质量、坡度、输运、KKT 多物理场残差诊断与收敛性判定

use astroid_evolution::model_calculation::equipotential::{
    solve_equipotential_radii_batch, solve_equipotential_radius,
};
use astroid_evolution::model_calculation::mass_level::{mass_above_level, solve_mass_level};
use astroid_evolution::model_calculation::potential::compute_effective_potential_at_point;
use astroid_evolution::rheology::state::{RheologyParams, RheologyState};
use astroid_evolution::scf_calculation::convergence::{
    check_scf_convergence, ScfTolerances,
};
use astroid_evolution::scf_calculation::mass_constraint::linearize_mass_constraint;
use astroid_evolution::scf_calculation::pipeline::{
    run_scf_iteration_step, ScfStepInput,
};
use astroid_evolution::scf_calculation::sqp_projection::{
    check_true_constraints, solve_convex_qp,
};
use astroid_evolution::scf_calculation::under_relaxation::{
    apply_under_relaxation, AdaptiveRelaxationController,
};
use std::f64::consts::PI;

// ============================================================================
// 步骤 1：质量能级 M(W_0) 连续性、单调性与二分求解
// ============================================================================
#[test]
fn test_step1_mass_level_monotonicity_and_bisection() {
    let rho0 = 2200.0; // 密度 2200 kg/m^3
    let density_fn = |_pos: [f64; 3]| -> f64 { rho0 };

    // 人工有效引力势: W(x) = -GM / r + 1/2 \omega^2 (x^2 + y^2)
    let gm = 1.0e8;
    let potential_fn = |pos: [f64; 3]| -> f64 {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(1.0);
        -gm / r
    };

    // 正八面体 6 方向采样
    let directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let weights = [4.0 * PI / 6.0; 6];

    // 1. 验证严格单调性：W_0 越高，等势面越小，包络质量 M(W_0) 越小
    let w_low = -2.0e6;
    let w_mid = -1.0e6;
    let w_high = -0.5e6;

    let m_low = mass_above_level(w_low, &directions, &weights, &potential_fn, &density_fn, 10.0, 1000.0, 50.0);
    let m_mid = mass_above_level(w_mid, &directions, &weights, &potential_fn, &density_fn, 10.0, 1000.0, 100.0);
    let m_high = mass_above_level(w_high, &directions, &weights, &potential_fn, &density_fn, 10.0, 1000.0, 200.0);

    assert!(m_low < m_mid, "W = -2e6 对应半径 r=50m, 质量应当小于 r=100m");
    assert!(m_mid < m_high, "W = -1e6 对应半径 r=100m, 质量应当小于 r=200m");

    // 2. 二分求解给定目标质量对应能级 W_0
    let target_r = 120.0_f64;
    let target_mass = 4.0 / 3.0 * PI * target_r.powi(3) * rho0;
    let expected_w0 = -gm / target_r;

    let solved_w0 = solve_mass_level(
        target_mass,
        &directions,
        &weights,
        &potential_fn,
        &density_fn,
        -1.5e6,
        -0.5e6,
        10.0,
        500.0,
        120.0,
        target_mass * 1e-3,
        50,
    ).expect("质量能级求解应当成功");

    assert!(
        (solved_w0 - expected_w0).abs() / expected_w0.abs() < 1e-2,
        "质量能级 W0 求解偏差过大: 得到 {}, 期望 {}",
        solved_w0, expected_w0
    );
}

// ============================================================================
// 步骤 2：等势候选面径向连续跟踪与分支选择
// ============================================================================
#[test]
fn test_step2_equipotential_candidate_surface_tracking() {
    let gm = 5.0e7;
    let omega_z = 0.001; // 慢自转
    let omega = [0.0, 0.0, omega_z];

    let pot_fn = |pos: [f64; 3]| -> f64 {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt().max(0.1);
        let u_grav = -gm / r;
        compute_effective_potential_at_point(u_grav, pos, omega)
    };

    let w0 = -5.0e5; // 目标等势值，标称球半径 ~ 100 m

    // 1. 单方向根存在性与残差验证
    let dir_eq = [1.0, 0.0, 0.0];
    let res_single = solve_equipotential_radius(
        dir_eq,
        w0,
        &pot_fn,
        10.0,
        500.0,
        100.0,
        1e-6,
        64,
    );

    assert!(res_single.success, "单方向求根应当成功");
    assert!(res_single.residual / w0.abs() < 1e-5, "等势相对残差必须小于容差");
    assert!(res_single.radius > 90.0 && res_single.radius < 110.0);

    // 2. 连续采样方向批量跟踪与回退测试
    let n_pts = 12;
    let mut dirs = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let angle = 2.0 * PI * (i as f64) / (n_pts as f64);
        dirs.push([angle.cos(), angle.sin(), 0.0]);
    }

    let (radii, failed_count) = solve_equipotential_radii_batch(
        &dirs,
        w0,
        &pot_fn,
        10.0,
        500.0,
        res_single.radius,
        1e-6,
    );

    assert_eq!(radii.len(), n_pts);
    assert_eq!(failed_count, 0, "所有采样方向均应成功跟踪到等势面");

    // 由于轴对称性，赤道圆周上所有半径应当高度一致
    for &r in &radii {
        assert!((r - radii[0]).abs() < 1e-3, "赤道等势面半径应当对称一致");
    }
}

// ============================================================================
// 步骤 3：顺序凸二次规划 (SQP/QP) 几何摩擦约束投影与真实非线性验收
// ============================================================================
#[test]
fn test_step3_sqp_qp_projection_and_true_constraint_acceptance() {
    let n = 6;
    let r_nominal = 100.0_f64;
    let r_current = vec![r_nominal; n];
    // 假设等势面略有起伏
    let r_eq = vec![104.0, 96.0, 102.0, 98.0, 103.0, 97.0];
    let weights = vec![4.0 * PI / n as f64; n];
    let rho0 = 2500.0;
    let density_fn = |_pos: [f64; 3]| -> f64 { rho0 };

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
        [-9.8, 0.0, 0.0],
        [9.8, 0.0, 0.0],
        [0.0, -9.8, 0.0],
        [0.0, 9.8, 0.0],
        [0.0, 0.0, -9.8],
        [0.0, 0.0, 9.8],
    ];
    let phi_list = vec![0.55; n]; // 休止角 ~ 31.5 度
    let sin2_phi_list: Vec<f64> = phi_list.iter().map(|&p: &f64| p.sin().powi(2)).collect();

    let target_mass = 4.0 / 3.0 * PI * r_nominal.powi(3) * rho0;

    // 1. 质量约束一阶线性化
    let mass_constraint = linearize_mass_constraint(
        &r_current,
        &dirs,
        &weights,
        &density_fn,
        target_mass,
    );
    assert!((mass_constraint.current_mass - target_mass).abs() / target_mass < 1e-4);

    // 2. 求解 SQP/QP 凸优化投影
    let r_star = solve_convex_qp(
        &r_current,
        &r_eq,
        &weights,
        &mass_constraint,
        &gravities,
        &normals,
        &dirs,
        &sin2_phi_list,
        10.0, // 信赖域 10 m
        50.0,
        200.0,
    );

    assert_eq!(r_star.len(), n);
    for i in 0..n {
        assert!(r_star[i] >= 90.0 && r_star[i] <= 110.0, "信赖域必须得到严格满足");
    }

    // 3. 真实非线性约束重新验收
    let eval_result = check_true_constraints(
        &r_star,
        &r_eq,
        &weights,
        &dirs,
        &normals,
        &gravities,
        &phi_list,
        &density_fn,
        target_mass,
        1e-2, // 质量容差 1%
    );

    assert!(eval_result.accepted, "真实非线性几何验收应当通过: {}", eval_result.message);
    assert!(eval_result.true_mass_residual < 1e-2);
    assert!(eval_result.max_slope_violation <= 1e-4);
}

// ============================================================================
// 步骤 4：欠松弛迭代与自适应阻尼
// ============================================================================
#[test]
fn test_step4_under_relaxation_and_adaptive_damping() {
    let r_curr = vec![100.0, 100.0, 100.0];
    let r_star = vec![110.0, 90.0, 105.0];

    // 1. 固定松弛因子 lambda = 0.3
    let lambda = 0.3;
    let r_next = apply_under_relaxation(&r_curr, &r_star, lambda);
    // r0 = 0.7 * 100 + 0.3 * 110 = 103.0
    // r1 = 0.7 * 100 + 0.3 * 90 = 97.0
    // r2 = 0.7 * 100 + 0.3 * 105 = 101.5
    assert!((r_next[0] - 103.0).abs() < 1e-12);
    assert!((r_next[1] - 97.0).abs() < 1e-12);
    assert!((r_next[2] - 101.5).abs() < 1e-12);

    // 2. 自适应阻尼控制器测试
    let mut controller = AdaptiveRelaxationController::new(0.25, 0.05, 0.8);
    // 连续收敛步：逐步提升 lambda
    controller.update(1.0e-2);
    controller.update(8.0e-3);
    controller.update(5.0e-3);
    assert!(controller.lambda > 0.25, "单调收敛时应提高松弛步长");

    // 残差震荡反弹步：快速施加强阻尼
    let before_cut = controller.lambda;
    controller.update(7.0e-3);
    assert!(controller.lambda < before_cut, "残差震荡时必须主动收缩松弛步长");
    assert!(controller.lambda >= controller.lambda_min);
}

// ============================================================================
// 步骤 5：几何、质量、坡度、输运、KKT 收敛判定与完整循环测试
// ============================================================================
#[test]
fn test_step5_scf_convergence_and_full_pipeline_step() {
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
        relaxation_lambda: 0.25,
        trust_radius: 5.0,
    };

    let tols = ScfTolerances {
        geometry_tol: 0.1, // 宽容度测试单步
        mass_tol: 0.05,
        slope_tol: 0.05,
        transport_tol: 0.05,
        kkt_tol: 0.05,
    };

    // 执行单步完整循环
    let step_out = run_scf_iteration_step(&input, &potential_fn, &density_fn, &tols)
        .expect("SCF 完整循环单步执行应当成功");

    assert_eq!(step_out.next_radii.len(), n);
    assert_eq!(step_out.next_modal_q.len(), n);
    assert_eq!(step_out.phi_list.len(), n);

    // 验证残差物理合理性
    assert!(step_out.residuals.geometry_residual >= 0.0);
    assert!(step_out.residuals.mass_residual >= 0.0);
    assert!(step_out.residuals.slope_residual >= 0.0);
    assert!(step_out.residuals.transport_residual >= 0.0);

    // 判定收敛诊断接口
    let (converged, diag) = check_scf_convergence(&step_out.residuals, &tols);
    assert_eq!(converged, step_out.is_converged);
    if !converged {
        assert!(!diag.is_empty(), "未收敛时必须提供详细原因诊断");
    }
}
