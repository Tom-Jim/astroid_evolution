//! 阶段 5 完整集成测试套件：流变学与 K=4 径向 Galerkin 输运闭包
//! 严格依照 docs/software.md 第 9、10、15.5 节 与 docs/mathstrict.md 第 4、8、9、10 节
//!
//! 覆盖阶段 5 推荐实现顺序的 6 个关键步骤：
//! 1. Galerkin 基函数 (K=4)
//! 2. M, K1, K2 矩阵装配与解析逆
//! 3. m = M q 守恒变量状态转换
//! 4. 半拉格朗日输运与曲面 Jacobian 守恒修正
//! 5. 联合流变闭合、本构方程与经典极限退化
//! 6. 摩擦角 phi 与坡度闭合约束

use astroid_evolution::rheology::cohesion::{cohesive_pressure, effective_pressure};
use astroid_evolution::rheology::constitutive::{equivalent_flow_stress, flow_stress, flow_viscosity};
use astroid_evolution::rheology::degeneracy::check_degeneracy_to_classic;
use astroid_evolution::rheology::friction_angle::{effective_projection_mu, friction_angle, sin2_friction_angle};
use astroid_evolution::rheology::friction_law::mu_dynamic;
use astroid_evolution::rheology::inertial_number::modified_inertial_number;
use astroid_evolution::rheology::r#yield::combined_yield_function;
use astroid_evolution::rheology::state::{RheologyParams, RheologyState};
use astroid_evolution::rheology::strain_rate::{
    compute_deviatoric_strain_rate, compute_shear_rate, compute_strain_rate,
};
use astroid_evolution::rheology::tide_stress::{equivalent_stress, frobenius_norm, tidal_backstress};
use astroid_evolution::scf_calculation::conservation::{
    conservative_modal_update, conservative_modal_update_batch, surface_map_jacobian_from_grad,
};
use astroid_evolution::scf_calculation::conservative_state::{
    compute_conservative_modal_state, compute_conservative_states_batch,
    recover_physical_modal_state, recover_physical_states_batch,
};
use astroid_evolution::scf_calculation::galerkin::{
    basis_k4, basis_k4_first_derivative, basis_k4_second_derivative, eval_modal_field,
    eval_modal_first_derivative,
};
use astroid_evolution::scf_calculation::galerkin_matrices::{
    assemble_first_stiffness, assemble_first_stiffness_rational, assemble_mass_matrix,
    assemble_mass_matrix_inverse, assemble_mass_matrix_inverse_rational,
    assemble_mass_matrix_rational, assemble_second_stiffness, assemble_second_stiffness_rational,
    invert_mass_matrix,
};
use astroid_evolution::scf_calculation::semi_lagrangian::{
    backtrace_surface_characteristic, backtrace_surface_characteristic_rk2,
    barycentric_interpolate_modal_state, project_point_to_triangle_barycentric,
};
use astroid_evolution::scf_calculation::slope_closure::check_slope_violation;
use astroid_evolution::scf_calculation::surface_metric::compute_surface_metric;
use num_rational::Ratio;

// ============================================================================
// 步骤 1：Galerkin 基函数 (K=4)
// ============================================================================
#[test]
fn test_step1_galerkin_basis_k4_comprehensive() {
    // 1. 基函数值验证: psi_k(zeta) = zeta^(k+1), k=1..4 -> zeta^2, zeta^3, zeta^4, zeta^5
    let zeta_half = 0.5;
    let psi_half = basis_k4(zeta_half);
    assert!((psi_half[0] - 0.25).abs() < 1e-14); // 0.5^2
    assert!((psi_half[1] - 0.125).abs() < 1e-14); // 0.5^3
    assert!((psi_half[2] - 0.0625).abs() < 1e-14); // 0.5^4
    assert!((psi_half[3] - 0.03125).abs() < 1e-14); // 0.5^5

    // 边界条件: zeta = 0 处恒为 0，zeta = 1 处恒为 1
    let psi_0 = basis_k4(0.0);
    assert_eq!(psi_0, [0.0, 0.0, 0.0, 0.0]);
    let psi_1 = basis_k4(1.0);
    assert_eq!(psi_1, [1.0, 1.0, 1.0, 1.0]);

    // 2. 一阶导数验证: psi'_k(zeta) = (k+1) * zeta^k -> 2*zeta, 3*zeta^2, 4*zeta^3, 5*zeta^4
    let dpsi_half = basis_k4_first_derivative(zeta_half);
    assert!((dpsi_half[0] - 2.0 * 0.5).abs() < 1e-14);
    assert!((dpsi_half[1] - 3.0 * 0.25).abs() < 1e-14);
    assert!((dpsi_half[2] - 4.0 * 0.125).abs() < 1e-14);
    assert!((dpsi_half[3] - 5.0 * 0.0625).abs() < 1e-14);

    // 3. 二阶导数验证: psi''_k(zeta) = k*(k+1) * zeta^(k-1) -> 2, 6*zeta, 12*zeta^2, 20*zeta^3
    let d2psi_half = basis_k4_second_derivative(zeta_half);
    assert!((d2psi_half[0] - 2.0).abs() < 1e-14);
    assert!((d2psi_half[1] - 6.0 * 0.5).abs() < 1e-14);
    assert!((d2psi_half[2] - 12.0 * 0.25).abs() < 1e-14);
    assert!((d2psi_half[3] - 20.0 * 0.125).abs() < 1e-14);

    // 4. 模态场叠加函数测试
    let q = [1.0, 2.0, -1.0, 0.5];
    let val_at_one = eval_modal_field(&q, 1.0);
    // f(1) = 1*1 + 2*1 - 1*1 + 0.5*1 = 2.5
    assert!((val_at_one - 2.5).abs() < 1e-14);

    let dval_at_one = eval_modal_first_derivative(&q, 1.0);
    // f'(1) = 1*2 + 2*3 - 1*4 + 0.5*5 = 2 + 6 - 4 + 2.5 = 6.5
    assert!((dval_at_one - 6.5).abs() < 1e-14);
}

// ============================================================================
// 步骤 2：M, K1, K2 矩阵装配与解析逆
// ============================================================================
#[test]
fn test_step2_galerkin_matrices_exact_assembly_and_inversion() {
    let m_rat = assemble_mass_matrix_rational();
    let k1_rat = assemble_first_stiffness_rational();
    let k2_rat = assemble_second_stiffness_rational();
    let inv_m_rat = assemble_mass_matrix_inverse_rational();

    // 1. 严格对比 mathstrict.md 第 10.1 节表格解析有理数值
    assert_eq!(m_rat[0][0], Ratio::new(1, 7));
    assert_eq!(m_rat[1][1], Ratio::new(1, 9));
    assert_eq!(m_rat[2][2], Ratio::new(1, 11));
    assert_eq!(m_rat[3][3], Ratio::new(1, 13));

    assert_eq!(k1_rat[0][0], Ratio::new(4, 5));
    assert_eq!(k1_rat[1][1], Ratio::new(9, 7));
    assert_eq!(k1_rat[2][2], Ratio::new(16, 9));
    assert_eq!(k1_rat[3][3], Ratio::new(25, 11));

    assert_eq!(k2_rat[0][0], Ratio::new(4, 3));
    assert_eq!(k2_rat[1][1], Ratio::new(36, 5));
    assert_eq!(k2_rat[2][2], Ratio::new(144, 7));
    assert_eq!(k2_rat[3][3], Ratio::new(400, 9));

    // 2. 严格对称性验证: A_ij = A_ji
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(m_rat[i][j], m_rat[j][i]);
            assert_eq!(k1_rat[i][j], k1_rat[j][i]);
            assert_eq!(k2_rat[i][j], k2_rat[j][i]);
            assert_eq!(inv_m_rat[i][j], inv_m_rat[j][i]);
        }
    }

    // 3. 有理数乘法 M * M^(-1) 严格恒等为单位阵 I
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = Ratio::new(0, 1);
            for k in 0..4 {
                sum = sum + m_rat[i][k] * inv_m_rat[k][j];
            }
            let expected = if i == j { Ratio::new(1, 1) } else { Ratio::new(0, 1) };
            assert_eq!(sum, expected, "M * M^(-1) 有理数检验失败于 ({i}, {j})");
        }
    }

    // 4. 浮点矩阵转换与求逆接口测试
    let m_float = assemble_mass_matrix();
    let k1_float = assemble_first_stiffness();
    let k2_float = assemble_second_stiffness();
    let inv_m_float = assemble_mass_matrix_inverse();

    assert!((m_float[0][0] - 1.0 / 7.0).abs() < 1e-15);
    assert!((k1_float[0][0] - 0.8).abs() < 1e-15);
    assert!((k2_float[0][0] - 4.0 / 3.0).abs() < 1e-15);

    let inv_computed = invert_mass_matrix(&m_float).expect("质量矩阵求逆应当成功");
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(inv_computed[i][j], inv_m_float[i][j]);
        }
    }
}

// ============================================================================
// 步骤 3：m = M q 守恒变量状态转换
// ============================================================================
#[test]
fn test_step3_conservative_modal_state_transformations() {
    let m_mat = assemble_mass_matrix();
    let inv_m = assemble_mass_matrix_inverse();

    // 任意物理模态测试向量
    let q_test_vectors = vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.25, -0.42, 0.88, 0.15],
        [-2.5, 3.1, -1.8, 0.95],
    ];

    for q_orig in &q_test_vectors {
        // 映射为守恒变量 m = M q
        let m = compute_conservative_modal_state(&m_mat, q_orig);
        // 恢复物理模态 q_rec = M^(-1) m
        let q_rec = recover_physical_modal_state(&inv_m, &m);

        for k in 0..4 {
            assert!(
                (q_orig[k] - q_rec[k]).abs() < 1e-8,
                "模态 {k} 往返恢复误差过大: 原始 {}, 恢复 {}",
                q_orig[k], q_rec[k]
            );
        }
    }

    // 批量测试
    let m_batch = compute_conservative_states_batch(&m_mat, &q_test_vectors);
    let q_recovered_batch = recover_physical_states_batch(&inv_m, &m_batch);
    assert_eq!(q_recovered_batch.len(), q_test_vectors.len());

    for (q_orig, q_rec) in q_test_vectors.iter().zip(q_recovered_batch.iter()) {
        for k in 0..4 {
            assert!((q_orig[k] - q_rec[k]).abs() < 1e-8);
        }
    }
}

// ============================================================================
// 步骤 4：半拉格朗日输运与反向特征线
// ============================================================================
#[test]
fn test_step4_semi_lagrangian_transport_and_conservation() {
    // 1. 一阶特征反向追踪: X_d = X_i - dt * c_Gamma
    let node = [10.0, 5.0, 2.0];
    let vel = [1.0, -0.5, 0.2];
    let dt = 0.5;

    let x_d_1 = backtrace_surface_characteristic(node, vel, dt);
    assert!((x_d_1[0] - (10.0 - 0.5 * 1.0)).abs() < 1e-14);
    assert!((x_d_1[1] - (5.0 - 0.5 * (-0.5))).abs() < 1e-14);
    assert!((x_d_1[2] - (2.0 - 0.5 * 0.2)).abs() < 1e-14);

    // 2. 二阶 RK2 反向追踪
    let vel_half = [1.1, -0.4, 0.25];
    let x_d_2 = backtrace_surface_characteristic_rk2(node, vel_half, dt);
    assert!((x_d_2[0] - (10.0 - 0.5 * 1.1)).abs() < 1e-14);

    // 3. 三角形投影与重心坐标插值
    let v0 = [0.0, 0.0, 0.0];
    let v1 = [1.0, 0.0, 0.0];
    let v2 = [0.0, 1.0, 0.0];
    let p_test = [0.25, 0.25, 0.0];

    let bary = project_point_to_triangle_barycentric(p_test, v0, v1, v2);
    // 重心坐标: u=0.25, v=0.25, w=0.5
    assert!((bary[0] - 0.5).abs() < 1e-12);
    assert!((bary[1] - 0.25).abs() < 1e-12);
    assert!((bary[2] - 0.25).abs() < 1e-12);

    let modal_v0 = [1.0, 2.0, 3.0, 4.0];
    let modal_v1 = [2.0, 4.0, 6.0, 8.0];
    let modal_v2 = [3.0, 6.0, 9.0, 12.0];
    let m_interp = barycentric_interpolate_modal_state(bary, &modal_v0, &modal_v1, &modal_v2);

    // m_k = 0.5 * v0 + 0.25 * v1 + 0.25 * v2
    // m_0 = 0.5*1 + 0.25*2 + 0.25*3 = 0.5 + 0.5 + 0.75 = 1.75
    assert!((m_interp[0] - 1.75).abs() < 1e-12);

    // 4. 曲面映射 Jacobian 与模态守恒更新
    let da_d_da = [[1.2, 0.1], [-0.05, 0.9]];
    let j_map = surface_map_jacobian_from_grad(&da_d_da);
    let expected_j_map = (1.2 * 0.9 - 0.1 * (-0.05_f64)).abs();
    assert!((j_map - expected_j_map).abs() < 1e-14);

    let j_gamma_next = 1.5;
    let j_gamma_dep = 1.4;
    let m_dep = [2.0, 3.0, 4.0, 5.0];
    let source = [0.1, -0.2, 0.3, 0.0];

    let m_next = conservative_modal_update(
        j_gamma_next,
        j_gamma_dep,
        j_map,
        &m_dep,
        Some(&source),
        dt,
    );
    let expected_weight = (j_map * j_gamma_dep) / j_gamma_next;
    for k in 0..4 {
        let expected_val = expected_weight * m_dep[k] + dt * source[k];
        assert!((m_next[k] - expected_val).abs() < 1e-12);
    }

    // 批量守恒更新验证
    let batch_res = conservative_modal_update_batch(
        &[j_gamma_next],
        &[j_gamma_dep],
        &[j_map],
        &[m_dep],
        Some(&[source]),
        dt,
    );
    assert_eq!(batch_res.len(), 1);
    assert_eq!(batch_res[0], m_next);

    // 曲面度量 J_Gamma 计算一致性
    let metric = compute_surface_metric(1.0, 0.5, 100.0, 0.0, 0.0);
    assert!(metric.jacobian_gamma > 0.0);
}

// ============================================================================
// 步骤 5：联合流变闭合、本构方程与经典极限退化
// ============================================================================
#[test]
fn test_step5_rheology_constitutive_and_asymptotic_degeneracy() {
    let params = RheologyParams::default();
    let state = RheologyState::new(
        1000.0, // p = 1 kPa
        0.58,   // nu = 58%
        6.2,    // Z = 6.2
        0.005,  // d = 5 mm
        2600.0, // rho_p = 2600 kg/m^3
    ).expect("合法状态参数");

    // 1. 应变率张量与严格剪切率计算
    let vel_grad = [
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let d = compute_strain_rate(&vel_grad);
    let d_prime = compute_deviatoric_strain_rate(&d);
    let shear_rate = compute_shear_rate(&d_prime);
    assert!((shear_rate - 3.0).abs() < 1e-12);

    // 2. 内聚压力与有效压力
    let p_c = cohesive_pressure(params.c_c, state.coordination_number, state.solid_fraction, 1e-6, state.grain_diameter);
    let p_star = effective_pressure(state.pressure, p_c);
    assert!(p_c > 0.0);
    assert!(p_star > state.pressure);

    // 3. 修正惯性数与动态摩擦系数
    let i_c = modified_inertial_number(state.grain_diameter, shear_rate, p_star, state.grain_density);
    let mu_d = mu_dynamic(i_c, params.mu_s, params.mu_2, params.i_0);
    assert!(mu_d > params.mu_s);
    assert!(mu_d < params.mu_2);

    // 4. 潮汐背应力与屈服函数
    let hessian_w = [
        [1.0, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -0.5],
    ];
    let t_tide = tidal_backstress(params.c_t, state.grain_density, state.solid_fraction, state.grain_diameter, &hessian_w);
    let t_norm = frobenius_norm(&t_tide);
    let t_eq = equivalent_stress(&t_tide);
    assert!((t_norm / std::f64::consts::SQRT_2 - t_eq).abs() < 1e-12);

    // 5. 偏应力与本构流动
    let eta = flow_viscosity(params.cohesion_macro, mu_d, state.pressure, shear_rate, params.gamma_dot_eps);
    let tau_flow = flow_stress(eta, &d_prime);
    let tau_eq = equivalent_flow_stress(&tau_flow);
    // 偏流动等效应力必须与 (c + mu_d * p) 高精度一致
    assert!((tau_eq - (params.cohesion_macro + mu_d * state.pressure)).abs() < 1e-5);

    // 6. 联合屈服准则
    let f_comb = combined_yield_function(&tau_flow, &t_tide, params.cohesion_macro, params.mu_y, state.pressure);
    assert!(f_comb.is_finite());

    // 7. 渐近退化检验 (docs/mathstrict.md 第 4.7-4.16 节)
    // 当 F_H -> 0, T_tide -> 0, c -> 0 时，联合模型严格退化为经典 mu(I) 模型
    let zero_hessian = [[0.0; 3]; 3];
    let degen = check_degeneracy_to_classic(
        state.pressure,
        state.solid_fraction,
        state.coordination_number,
        state.grain_diameter,
        state.grain_density,
        0.0, // F_H = 0
        params.c_c,
        params.c_t,
        &zero_hessian,
        &d_prime,
        shear_rate,
        params.mu_s,
        params.mu_2,
        params.i_0,
        0.0, // c = 0
        1e-8,
    );

    assert!(degen.cohesion_ratio < 1e-14, "内聚比率应当精确归零");
    assert!(degen.tide_stress_ratio < 1e-14, "潮汐应力比率应当精确归零");
    assert!(degen.inertial_number_rel_diff < 1e-14, "惯性数相对偏差应当小于 1e-14");
    assert!(degen.friction_abs_diff < 1e-14, "摩擦系数偏差应当小于 1e-14");
    assert!(degen.flow_stress_rel_diff < 1e-6, "流动应力退化偏差应当小于 1e-6");
}

// ============================================================================
// 步骤 6：摩擦角 phi 与坡度闭合约束
// ============================================================================
#[test]
fn test_step6_friction_angle_and_slope_closure() {
    let c = 10.0; // 10 Pa
    let mu_proj = 0.45;
    let p = 500.0; // 500 Pa
    let t_norm = 15.0; // 15 Pa
    let p_star = p + 50.0; // p_* = 550 Pa

    // 1. 有效投影摩擦系数计算
    let mu_eff = effective_projection_mu(c, mu_proj, p, t_norm, p_star);
    let expected_num = 10.0 + 0.45 * 500.0 + 15.0 / std::f64::consts::SQRT_2;
    let expected_mu = expected_num / 550.0;
    assert!((mu_eff - expected_mu).abs() < 1e-14);

    // 2. 保守标量摩擦角
    let phi = friction_angle(mu_eff);
    assert!((phi - expected_mu.atan()).abs() < 1e-14);

    let sin2_phi = sin2_friction_angle(mu_eff);
    assert!((sin2_phi - phi.sin().powi(2)).abs() < 1e-12);

    // 3. 坡度闭合约束测试
    // 有效引力向量沿 -z 方向: g_eff = (0, 0, -g)
    let g_eff = [0.0, 0.0, -10.0];

    // 情况 A：平缓地表 (法向倾角 10 度 < 休止角 phi ~ 23 度)
    let normal_gentle = [10.0_f64.to_radians().sin(), 0.0, 10.0_f64.to_radians().cos()];
    let (violated_gentle, sin2_gentle, _) = check_slope_violation(g_eff, normal_gentle, phi);
    assert!(!violated_gentle, "平缓坡面不应当违背摩擦角约束");
    assert!(sin2_gentle < sin2_phi);

    // 情况 B：陡峭崖面 (法向倾角 45 度 > 休止角 phi ~ 23 度)
    let normal_steep = [45.0_f64.to_radians().sin(), 0.0, 45.0_f64.to_radians().cos()];
    let (violated_steep, sin2_steep, _) = check_slope_violation(g_eff, normal_steep, phi);
    assert!(violated_steep, "陡峭坡面应当触发违背约束");
    assert!(sin2_steep > sin2_phi);
}
