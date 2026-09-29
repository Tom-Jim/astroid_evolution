//! 阶段 2：CPU 严格参考解 完备测试套件
//! 严格依照 docs/software.md 第 15 节 与 docs/mathstrict.md 第 1 节

use std::f64::consts::PI;
use std::path::Path;

use astroid_evolution::file_support::load_project_toml;
use astroid_evolution::model_calculation::contracts::{DensityKernel, FieldBranch};
use astroid_evolution::rtfp::analytic::G_SI;
use astroid_evolution::rtfp::bvh::build_flattened_bvh;
use astroid_evolution::rtfp::cpu_reference::{CpuReferenceSolver, CpuSolverOptions};
use astroid_evolution::rtfp::density_derivatives::{
    eval_density, eval_density_gradient, eval_density_hessian, eval_density_laplacian,
};
use astroid_evolution::rtfp::feynman::{integrate_feynman_parameter, moving_center, moving_scale_squared};
use astroid_evolution::rtfp::finite_part::scale_transform_difference;
use astroid_evolution::rtfp::quadrature;
use astroid_evolution::rtfp::radial_integrals::{
    alpha_one_gradient, alpha_one_hessian, alpha_one_hessian_finite_part, alpha_one_potential,
    fixed_quadrature_triplet,
};
use astroid_evolution::rtfp::residual_channels::{ResidualChannel, ResidualChannelSelector};
use astroid_evolution::rtfp::tangent_cone::compute_tangent_cone;

#[test]
fn test_step1_cauchy_density_and_derivatives() {
    let kernels = vec![
        DensityKernel {
            center: [0.0, 0.0, 0.0],
            sigma: 0.002,
            weight: 1200.0,
            alpha: 1.0,
        },
        DensityKernel {
            center: [100.0, 50.0, -50.0],
            sigma: 0.004,
            weight: 300.0,
            alpha: 1.5,
        },
    ];

    let x = [80.0, 40.0, 20.0];

    // 1. 密度值正定性
    let rho = eval_density(x, &kernels);
    assert!(rho > 0.0);

    // 2. 梯度差分验证
    let grad_ana = eval_density_gradient(x, &kernels);
    let eps = 1e-6;
    for dim in 0..3 {
        let mut xp = x;
        let mut xm = x;
        xp[dim] += eps;
        xm[dim] -= eps;
        let num_deriv = (eval_density(xp, &kernels) - eval_density(xm, &kernels)) / (2.0 * eps);
        assert!(
            (grad_ana[dim] - num_deriv).abs() < 1e-5,
            "梯度检验失败: dim {dim}, ana {:.6e}, num {:.6e}",
            grad_ana[dim], num_deriv
        );
    }

    // 3. Hessian 对称性与迹恒等式 Tr(Hessian) == Laplacian
    let h = eval_density_hessian(x, &kernels);
    let lap = eval_density_laplacian(x, &kernels);
    let trace = h[0] + h[1] + h[2];
    assert!(
        (trace - lap).abs() < 1e-9,
        "Laplacian 与 Hessian 迹不匹配: trace {trace}, lap {lap}"
    );
}

#[test]
fn test_step2_alpha_one_and_radial_integrals() {
    let b = 25.0;
    let d = 30.0;
    let a0 = b * b + d * d;
    let r0 = 10.0;
    let r1 = 80.0;

    // 1. alpha=1 解析原函数 vs 16 点 Gauss-Legendre
    let iu_ana = alpha_one_potential(b, d, r0, r1);
    let ig_ana = alpha_one_gradient(b, d, r0, r1);
    let ih_ana = alpha_one_hessian(a0, b, d, r0, r1);

    let (iu_num, ig_num, ih_num) = fixed_quadrature_triplet(1.0, b, d, r0, r1);

    assert!((iu_ana - iu_num).abs() < 1e-9);
    assert!((ig_ana - ig_num).abs() < 1e-9);
    assert!((ih_ana - ih_num).abs() < 1e-9);

    // 2. 端点 R0 = 0 处有限部分径向积分
    let ell0 = 1.0;
    let ih_fp = alpha_one_hessian_finite_part(a0, b, d, r1, ell0);
    assert!(ih_fp.is_finite());

    // 3. alpha != 1 的求积稳定性
    let (iu_15, ig_15, ih_15) = fixed_quadrature_triplet(1.5, b, d, r0, r1);
    assert!(iu_15 > 0.0 && ig_15 > 0.0 && ih_15 > 0.0);
}

#[test]
fn test_step3_external_classical_rtfp() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).unwrap();
    let bvh = build_flattened_bvh(&loaded.mesh);

    let solver = CpuReferenceSolver::new(
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        CpuSolverOptions {
            directions: 288,
            near_field_r0: 0.01,
            ell0: 1.0,
        },
    );

    // 取外部近区观测点 (Ryugu 半径 ~450 m, 600 m 位于外部近区)
    let p_ext = [0.0, 0.0, 600.0];
    let sample = solver.compute_field_at_point(p_ext);

    assert_eq!(sample.branch, FieldBranch::ExternalClassical);
    assert!(sample.potential > 0.0, "外部引力势能必须严格为正");

    // 加速度 g 必须指向实体内部 (即 -z 方向)
    assert!(sample.gravity[2] < 0.0, "重力加速度必须指向质心");

    // 外部无源真空点拉普拉斯方程 Tr(H) = 0
    let h_frobenius = (sample.hessian6[0].powi(2)
        + sample.hessian6[1].powi(2)
        + sample.hessian6[2].powi(2)
        + 2.0 * (sample.hessian6[3].powi(2) + sample.hessian6[4].powi(2) + sample.hessian6[5].powi(2))).sqrt();

    let relative_trace = sample.trace.abs() / h_frobenius;
    assert!(
        relative_trace < 1e-3,
        "外部点拉普拉斯迹误差过大: Tr(H) = {:.3e}, ||H||_F = {:.3e}, rel = {:.3e}",
        sample.trace, h_frobenius, relative_trace
    );
}

#[test]
fn test_step4_finite_part_and_tangent_cone() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).unwrap();
    let bvh = build_flattened_bvh(&loaded.mesh);

    let dirs = quadrature::directions(288);

    // 1. 实体内部质心点 [0, 0, 0] (网格半径 ~450m, probe 10m 严格在内部)
    let p_center = [0.0, 0.0, 0.0];
    let cone = compute_tangent_cone(p_center, &bvh, &loaded.mesh, &dirs, 10.0);

    // 内部点的立体角必须接近 1 (4 pi)
    assert!(
        (cone.solid_angle_fraction - 1.0).abs() < 0.05,
        "内部点 vartheta 应接近 1.0，当前为 {}",
        cone.solid_angle_fraction
    );

    // 内部球对称切锥张量 C_ij 必须接近 0
    for k in 0..6 {
        assert!(cone.cone_tensor[k].abs() < 0.5);
    }

    // 2. 尺度变换不变性测试: H_{ell1} - H_{ell0} == G rho C ln(ell0 / ell1)
    let rho_center = eval_density(p_center, &loaded.kernels);
    let diff = scale_transform_difference(rho_center, &cone, 1.0, 2.0);
    // 内部 C ≈ 0，差值应接近 0
    for val in diff {
        assert!(val.abs() < 1e-12);
    }
}

#[test]
fn test_step5_cpu_reference_solver_full_pipeline() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).unwrap();
    let bvh = build_flattened_bvh(&loaded.mesh);

    let solver = CpuReferenceSolver::new(
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        CpuSolverOptions {
            directions: 288,
            near_field_r0: 10.0,
            ell0: 1.0,
        },
    );

    // 批次计算: 1 个外部点、1 个内部点
    let test_points = vec![
        [0.0, 0.0, 600.0],  // 外部
        [0.0, 0.0, 0.0],    // 内部质心
    ];

    let samples = solver.compute_field_batch(&test_points);
    assert_eq!(samples.len(), 2);

    // 外部点校验
    assert_eq!(samples[0].branch, FieldBranch::ExternalClassical);
    assert!(samples[0].potential > 0.0);
    assert!(samples[0].gravity[2] < 0.0);

    // 内部点校验: 泊松迹恒等式 Tr(H) = -4 pi G vartheta rho
    assert_eq!(samples[1].branch, FieldBranch::InteriorFinitePart);
    let rho_int = eval_density(test_points[1], &loaded.kernels);
    let expected_trace = -4.0 * PI * G_SI * 1.0 * rho_int;
    let trace_err = (samples[1].trace - expected_trace).abs() / expected_trace.abs();
    println!(
        "内部点迹校验: 实际 Tr(H) = {:.5e}, 泊松理论值 = {:.5e}, 相对误差 = {:.3e}",
        samples[1].trace, expected_trace, trace_err
    );
    assert!(trace_err < 0.1, "内部点有限部分泊松迹相对误差应在容差范围内");
}

#[test]
fn test_step6_residual_channels_and_feynman() {
    // 1. 残差通道互斥性
    let selector = ResidualChannelSelector::new(ResidualChannel::DirectQuadrature);
    assert!(selector.validate_disjoint(ResidualChannel::CarlsonSymmetric).is_ok());
    assert!(selector.validate_disjoint(ResidualChannel::DirectQuadrature).is_err());

    // 2. Feynman 参数化移动中心
    let x = [100.0, 0.0, 0.0];
    let c = [0.0, 0.0, 0.0];
    let z_half = moving_center(x, c, 0.5);
    assert_eq!(z_half, [50.0, 0.0, 0.0]);

    let m2 = moving_scale_squared(x, c, 0.01, 0.5);
    assert!(m2 > 0.0);

    // 3. Feynman 数值求积测试: int_0^1 u^{-1/2} du = 2
    let int_val = integrate_feynman_parameter(1.0, |_u| 1.0);
    assert!((int_val - 2.0).abs() < 1e-12);
}
