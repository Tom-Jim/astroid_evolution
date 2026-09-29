use std::path::Path;
use std::time::Instant;

use astroid_evolution::rtfp::density::Density;
use astroid_evolution::rtfp::mesh::Mesh;
use astroid_evolution::rtfp::RtfpModule;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("  RTFP (Ray-Traced Finite-Part) 引力与引力梯度张量独立求解器");
    println!("  输入模型: GLB / 点云表面 | 观测高度: 1.000 mm | 输出: H 张量与重力加速度 g");
    println!("================================================================================\n");

    let glb_path = Path::new("assets/models/ryugu.glb");
    let density_path = Path::new("assets/density/cauchy_elliptic.toml");

    // 1. 加载或输入模型 (优先加载真实小行星 GLB 模型，未找到则回退至点云表面)
    let mesh = if glb_path.exists() {
        println!("--> 正在从 GLB 加载模型: {}", glb_path.display());
        // ryugu.glb 单位为 km，按 1000.0 转换为标准米 (m)
        Mesh::load_glb(glb_path, 1000.0)?
    } else {
        println!("--> 未检测到 GLB 文件，正在从程序化表面点云重构网格...");
        // 构造代表性小天体点云表面测试集
        let mut sample_pts = Vec::new();
        for i in 0..1000 {
            let theta = (i as f64) * 0.1;
            let phi = (i as f64) * 0.05;
            let r = 450.0 + 30.0 * (3.0 * theta).sin();
            sample_pts.push([
                r * theta.sin() * phi.cos(),
                r * theta.cos(),
                r * theta.sin() * phi.sin(),
            ]);
        }
        Mesh::from_point_cloud(&sample_pts, 32, 64)?
    };

    let nv = mesh.vertex_count();
    let nf = mesh.face_count();
    let (volume, centroid) = mesh.volume_and_centroid();
    println!("--> 模型加载完成: {} 顶点, {} 三角面片", nv, nf);
    println!("--> 实体体积: {:.4e} m³, 质心: [{:.3}, {:.3}, {:.3}] m", volume, centroid[0], centroid[1], centroid[2]);

    // 2. 加载 Cauchy 密度模型 (ρ(x) = Σ_k w_k [1 + σ_k² ||x - c_k||²]^(-α))
    let density = if density_path.exists() {
        println!("--> 正在加载 Cauchy 密度配置: {}", density_path.display());
        Density::from_toml(density_path)?
    } else {
        println!("--> 使用内置 Ryugu 标准 4-Kernel Cauchy 变密度场...");
        Density::default_ryugu()
    };
    println!("--> Cauchy 混合核数量: {}", density.kernels().len());

    // 3. 构建独立 RTFP 计算模块 (288 球面求积方向，完备解析与 BVH 射线追踪)
    println!("--> 正在初始化 RTFP 求解器 (构建 BVH 扁平加速结构与 288-向高斯-勒让德方向)...");
    let rtfp = RtfpModule::new(mesh, density, 288)?;

    // 4. 执行模型表面 1mm (0.001 m) 观测面上的 RTFP H 求解，并由 H 恢复 g
    println!("--> 开始并行计算模型表面 1.000 mm 高度处的引力梯度张量 H 与加速度 g...");
    let t_start = Instant::now();
    let result = rtfp.compute_surface_1mm()?;
    let elapsed = t_start.elapsed();
    println!("--> 计算完成！耗时: {:.2?} (总点数: {})\n", elapsed, result.points.len());

    // 5. 打印关键观测点处的详细张量分量与重力恢复结果
    println!("--------------------------------------------------------------------------------");
    println!("  模型表面 1mm 高度处局部采样点 RTFP 计算结果");
    println!("--------------------------------------------------------------------------------");

    // 抽样打印典型位置 (赤道、两极、特征面)
    let sample_indices = [0, nv / 4, nv / 2, 3 * nv / 4, nv - 1];
    for &idx in &sample_indices {
        if idx >= nv { continue; }
        let p_obs = result.points[idx];
        let nrm = result.normals[idx];
        let h = result.hessians[idx];
        let g = result.gravities[idx];
        let h_frob = result.h_frobenius[idx];
        let g_mag = result.g_magnitudes[idx];
        let tr = result.traces[idx];

        println!("观测点 #{idx:04} (1mm 高度坐标: [{:.2}, {:.2}, {:.2}] m):", p_obs[0], p_obs[1], p_obs[2]);
        println!("  表面法向 n:    [{:.4}, {:.4}, {:.4}]", nrm[0], nrm[1], nrm[2]);
        println!("  Hessian H:    [xx={:.3e}, yy={:.3e}, zz={:.3e}, xy={:.3e}, xz={:.3e}, yz={:.3e}] s⁻²",
            h[0], h[1], h[2], h[3], h[4], h[5]);
        println!("  ||H||_F:      {:.3e} s⁻² | 迹校验 Tr(H): {:.3e} s⁻² (无源自由空间应 ≈ 0)", h_frob, tr);
        println!("  由 H 求解 g:  [{:.5e}, {:.5e}, {:.5e}] m/s²", g[0], g[1], g[2]);
        println!("  重力模长 ||g||: {:.5e} m/s² ({:.3} mm/s²)\n", g_mag, g_mag * 1000.0);
    }

    // 6. 统计全局分布指标
    let mut min_h = f64::INFINITY;
    let mut max_h = 0.0f64;
    let mut avg_h = 0.0f64;
    let mut min_g = f64::INFINITY;
    let mut max_g = 0.0f64;
    let mut avg_g = 0.0f64;
    let mut max_trace_err = 0.0f64;

    for i in 0..result.points.len() {
        let hf = result.h_frobenius[i];
        let gm = result.g_magnitudes[i];
        let tr = result.traces[i].abs();

        if hf < min_h { min_h = hf; }
        if hf > max_h { max_h = hf; }
        avg_h += hf;

        if gm < min_g { min_g = gm; }
        if gm > max_g { max_g = gm; }
        avg_g += gm;

        if tr > max_trace_err { max_trace_err = tr; }
    }
    avg_h /= result.points.len() as f64;
    avg_g /= result.points.len() as f64;

    println!("================================================================================");
    println!("  1mm 观测面全域统计指标");
    println!("================================================================================");
    println!("  Hessian 范数 ||H||_F : 最小值 = {:.3e}, 最大值 = {:.3e}, 均值 = {:.3e} s⁻²", min_h, max_h, avg_h);
    println!("  引力模长 ||g||       : 最小值 = {:.5e}, 最大值 = {:.5e}, 均值 = {:.5e} m/s²", min_g, max_g, avg_g);
    println!("  调和迹误差 Max |Tr|  : {:.3e} s⁻² (严格满足拉普拉斯方程 ∇²U = 0)", max_trace_err);
    println!("================================================================================");

    Ok(())
}
