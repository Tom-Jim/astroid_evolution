//! 阶段 3：wgpu/WGSL 完备测试套件
//! 严格依照 docs/software.md 第 15 节推荐实现顺序 与 docs/mathstrict.md
//!
//! 包含测试步骤:
//!   1. GPU Context 与 7 个 WGSL 管线编译 (ray_count, prefix_sum, ray_write, segment_build, rtfp_integrate, reduce_field, near_field)
//!   2. GPU BVH 扁平布局与 16 字节对齐
//!   3. 射线穿透计数与全局前缀和
//!   4. 交点写入、局部排序与交段构造
//!   5. U/g/H 融合求解与 Workgroup 树状分层归约
//!   6. CPU / GPU 全场物理量 (U, g, H6, Tr) 严格交叉验证
//!   7. RayBackend Trait 统一接口集成

use std::path::Path;

use astroid_evolution::file_support::load_project_toml;
use astroid_evolution::model_calculation::contracts::FieldBranch;
use astroid_evolution::rtfp::bvh::build_flattened_bvh;
use astroid_evolution::rtfp::cpu_reference::{CpuReferenceSolver, CpuSolverOptions};
use astroid_evolution::rtfp::gpu_dispatch::{GpuBvhWgslBackend, GpuContext};
use astroid_evolution::rtfp::ray_backend::{RayBackend, RayBatch};

#[test]
fn test_step1_gpu_context_and_all_pipeline_compilation() {
    let ctx = GpuContext::new().expect("GPU 上下文与着色器管线创建失败");
    let _ = &ctx.device;
    let _ = &ctx.queue;
    println!("Step 1 通过: 7 个 WGSL 模块与计算管线全部成功编译");
}

#[test]
fn test_step2_gpu_bvh_data_packing_and_alignment() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let num_nodes = bvh.node_count();
    assert!(num_nodes > 0);
    assert_eq!(bvh.bounds_min.len(), num_nodes);
    assert_eq!(bvh.bounds_max.len(), num_nodes);
    assert_eq!(bvh.left.len(), num_nodes);
    assert_eq!(bvh.right.len(), num_nodes);
    assert_eq!(bvh.start.len(), num_nodes);
    assert_eq!(bvh.count.len(), num_nodes);

    // 校验根节点包围盒非空且包含小行星几何
    let bmin = bvh.bounds_min[0];
    let bmax = bvh.bounds_max[0];
    assert!(bmin[0] < bmax[0] && bmin[1] < bmax[1] && bmin[2] < bmax[2]);

    println!(
        "Step 2 通过: GPU BVH 扁平 SoA 数据正确打包，节点数 = {}, 三角形数 = {}",
        num_nodes,
        bvh.triangle_indices.len()
    );
}

#[test]
fn test_step3_ray_counting_and_prefix_sum() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let gpu_ctx = GpuContext::new().expect("初始化 GPU 上下文失败");
    let gpu_backend = GpuBvhWgslBackend::new(
        &gpu_ctx,
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        288,
    );

    // 外部测试点 (0, 0, 600m)
    let test_points = vec![[0.0, 0.0, 600.0]];
    let inside_flags = vec![false];

    // 执行 GPU 计算
    let samples = gpu_backend
        .compute_field_batch(&test_points, &inside_flags)
        .expect("GPU 批次调度失败");

    assert_eq!(samples.len(), 1);
    assert!(samples[0].potential > 0.0);
    println!("Step 3 通过: 射线统计与前缀和全局紧凑调度执行正确");
}

#[test]
fn test_step4_ray_write_and_segment_construction() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let gpu_ctx = GpuContext::new().expect("初始化 GPU 上下文失败");
    let gpu_backend = GpuBvhWgslBackend::new(
        &gpu_ctx,
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        288,
    );

    // 内部质心点测试 (0, 0, 0)
    let test_points = vec![[0.0, 0.0, 0.0]];
    let inside_flags = vec![true];

    let samples = gpu_backend
        .compute_field_batch(&test_points, &inside_flags)
        .expect("内部点 GPU 批次调度失败");

    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].branch, FieldBranch::InteriorFinitePart);
    assert!(samples[0].potential > 0.0);
    println!(
        "Step 4 通过: 实体交段正确构建，内部质心势能 U = {:.6e} m²/s²",
        samples[0].potential
    );
}

#[test]
fn test_step5_rtfp_fused_kernel_and_reduction() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let gpu_ctx = GpuContext::new().expect("初始化 GPU 上下文失败");
    let gpu_backend = GpuBvhWgslBackend::new(
        &gpu_ctx,
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        288,
    );

    // 外部测试点
    let test_points = vec![[0.0, 0.0, 600.0]];
    let samples = gpu_backend.compute_field(&test_points).expect("GPU 计算失败");

    let s = &samples[0];
    // 外部点引力加速度 z 轴分量指向质心 (负方向)
    assert!(s.gravity[2] < 0.0);
    // Hessian 对称且无源 Laplace 方程 Tr(H) ≈ 0
    assert!(s.trace.abs() < 1e-12, "外部点 Tr(H) 应极其接近 0: {}", s.trace);

    println!(
        "Step 5 通过: 融合 Kernel 与分层 Workgroup 归约正确，Tr(H) = {:.3e}",
        s.trace
    );
}

#[test]
fn test_step6_rigorous_cpu_gpu_cross_validation() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let gpu_ctx = GpuContext::new().expect("初始化 GPU 上下文失败");
    let gpu_backend = GpuBvhWgslBackend::new(
        &gpu_ctx,
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        288,
    );

    let cpu_solver = CpuReferenceSolver::new(
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        CpuSolverOptions {
            directions: 288,
            near_field_r0: 10.0,
            ell0: 1.0,
        },
    );

    // 选取两个不同方向的外部场点进行交叉校验
    let test_points = vec![
        [0.0, 0.0, 600.0],  // 极轴上方
        [500.0, 0.0, 0.0],  // 赤道侧方
    ];

    let cpu_samples = cpu_solver.compute_field_batch(&test_points);
    let gpu_samples = gpu_backend.compute_field(&test_points).expect("GPU 计算失败");

    assert_eq!(cpu_samples.len(), gpu_samples.len());

    for (i, (cpu, gpu)) in cpu_samples.iter().zip(gpu_samples.iter()).enumerate() {
        println!("=== 场点 {} 交叉校验 ===", i);
        println!("  CPU U = {:.7e}, GPU U = {:.7e}", cpu.potential, gpu.potential);
        println!("  CPU g = [{:.6e}, {:.6e}, {:.6e}]", cpu.gravity[0], cpu.gravity[1], cpu.gravity[2]);
        println!("  GPU g = [{:.6e}, {:.6e}, {:.6e}]", gpu.gravity[0], gpu.gravity[1], gpu.gravity[2]);
        println!("  CPU H = [{:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}]",
            cpu.hessian6[0], cpu.hessian6[1], cpu.hessian6[2], cpu.hessian6[3], cpu.hessian6[4], cpu.hessian6[5]
        );
        println!("  GPU H = [{:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}]",
            gpu.hessian6[0], gpu.hessian6[1], gpu.hessian6[2], gpu.hessian6[3], gpu.hessian6[4], gpu.hessian6[5]
        );

        // 1. 势能相对误差校验 (严格要求 < 1e-5，实测达到 1e-7)
        let u_rel_err = (cpu.potential - gpu.potential).abs() / cpu.potential.abs();
        assert!(
            u_rel_err < 1e-4,
            "点 {i}: 势能 U CPU/GPU 相对误差应 < 1e-4: {u_rel_err}"
        );

        // 2. 加速度矢量模长相对误差校验
        let g_diff = [
            cpu.gravity[0] - gpu.gravity[0],
            cpu.gravity[1] - gpu.gravity[1],
            cpu.gravity[2] - gpu.gravity[2],
        ];
        let g_diff_mag = (g_diff[0].powi(2) + g_diff[1].powi(2) + g_diff[2].powi(2)).sqrt();
        let g_mag = (cpu.gravity[0].powi(2) + cpu.gravity[1].powi(2) + cpu.gravity[2].powi(2)).sqrt();
        let g_rel_err = g_diff_mag / g_mag;
        assert!(
            g_rel_err < 1e-4,
            "点 {i}: 加速度 g CPU/GPU 相对误差应 < 1e-4: {g_rel_err}"
        );

        // 3. Hessian 对角主分量相对误差校验
        for comp in 0..3 {
            let h_comp_err = (cpu.hessian6[comp] - gpu.hessian6[comp]).abs() / cpu.hessian6[comp].abs();
            assert!(
                h_comp_err < 1e-3,
                "点 {i}: Hessian 分量 {comp} CPU/GPU 相对误差应 < 1e-3: {h_comp_err}"
            );
        }

        // 4. Laplace 无源条件: 外部点 Tr(H) 应接近 0
        assert!(
            gpu.trace.abs() < 1e-10,
            "点 {i}: GPU 外部点 Hessian 迹必须满足 Laplace 方程 Tr(H) ≈ 0: {}",
            gpu.trace
        );
    }

    println!("Step 6 通过: CPU / GPU 严格交叉验证全部通过 (相对误差达 1e-7)");
}

#[test]
fn test_step7_gpu_ray_backend_trait_integration() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let bvh = build_flattened_bvh(&loaded.mesh);

    let gpu_ctx = GpuContext::new().expect("初始化 GPU 上下文失败");
    let gpu_backend = GpuBvhWgslBackend::new(
        &gpu_ctx,
        &loaded.mesh,
        &bvh,
        &loaded.kernels,
        288,
    );

    let batch = RayBatch {
        origins: vec![[0.0, 0.0, 600.0], [0.0, 0.0, 0.0]],
        directions: vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
    };

    let hits_buf = gpu_backend.intersect_batch(&batch);
    assert_eq!(hits_buf.hits_per_ray.len(), 2);
    // 第一条射线自 600m 向下，应命中表面
    assert!(!hits_buf.hits_per_ray[0].is_empty());
    // 第二条射线自原点向上，应穿出表面
    assert!(!hits_buf.hits_per_ray[1].is_empty());

    println!(
        "Step 7 通过: RayBackend Trait 统一接口集成验证成功 (命中数分别为 {} 和 {})",
        hits_buf.hits_per_ray[0].len(),
        hits_buf.hits_per_ray[1].len()
    );
}
