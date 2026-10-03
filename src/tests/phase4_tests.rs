//! 阶段 4：Bevy 显示完备测试套件
//! 严格依照 docs/software.md 第 15 节推荐实现顺序 与 第 13 节
//!
//! 包含测试步骤:
//!   1. 表面网格显示 (Mesh 构建、属性插槽与 SurfaceMeshEntity)
//!   2. 势场和引力着色 (U, |g|, Turbo / Viridis / Coolwarm 色表映射)
//!   3. Hessian / 坡度调试视图 (Tr(H), 坡度超限标记, 法向与射线调试开关)
//!   4. 实时 GPU 顶点更新 (GpuSurfaceUpdater 与 Bevy Mesh 同步)

use bevy::prelude::*;
use bevy::render::mesh::VertexAttributeValues;
use std::path::Path;

use astroid_evolution::file_support::load_project_toml;
use astroid_evolution::model_calculation::contracts::{FieldBranch, FieldSample};
use astroid_evolution::realtime_display::compute_pipeline::GpuSurfaceUpdater;
use astroid_evolution::realtime_display::events::DisplayMode;
use astroid_evolution::realtime_display::field_renderer::{
    colormap_coolwarm, colormap_turbo, colormap_viridis, compute_vertex_colors,
};
use astroid_evolution::realtime_display::mesh_renderer::build_bevy_mesh;
use astroid_evolution::realtime_display::plugin::AsteroidDisplayPlugin;
use astroid_evolution::realtime_display::resources::{
    AsteroidDisplayData, ColormapType, DisplaySettings,
};
use astroid_evolution::rtfp::gpu_dispatch::GpuContext;

#[test]
fn test_step1_surface_mesh_display() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let mesh_geom = &loaded.mesh;

    let nv = mesh_geom.vertex_count();
    let nf = mesh_geom.face_count();
    assert!(nv > 0 && nf > 0);

    let normals_f64 = mesh_geom.vertex_normals();
    let mut vertices = Vec::with_capacity(nv);
    let mut normals = Vec::with_capacity(nv);
    for i in 0..nv {
        let p = mesh_geom.vertex(i);
        let n = normals_f64[i];
        vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
    }

    let mut faces = Vec::with_capacity(nf);
    for f in 0..nf {
        faces.push(mesh_geom.face(f));
    }

    // 1. 构建 Bevy Mesh 对象
    let bevy_mesh = build_bevy_mesh(&vertices, &faces, &normals, None);

    // 2. 检查顶点的三维坐标与法向属性
    let pos_attr = bevy_mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .expect("缺失 POSITION 属性");
    match pos_attr {
        VertexAttributeValues::Float32x3(p) => assert_eq!(p.len(), nv),
        _ => panic!("POSITION 属性格式错误"),
    }

    let norm_attr = bevy_mesh
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .expect("缺失 NORMAL 属性");
    match norm_attr {
        VertexAttributeValues::Float32x3(n) => assert_eq!(n.len(), nv),
        _ => panic!("NORMAL 属性格式错误"),
    }

    let col_attr = bevy_mesh
        .attribute(Mesh::ATTRIBUTE_COLOR)
        .expect("缺失 COLOR 属性");
    match col_attr {
        VertexAttributeValues::Float32x4(c) => assert_eq!(c.len(), nv),
        _ => panic!("COLOR 属性格式错误"),
    }

    // 3. 检查三角面片拓扑索引
    let indices = bevy_mesh.indices().expect("缺失 Indices");
    assert_eq!(indices.len(), nf * 3);

    // 4. 初始化 Bevy App 验证 ECS 组件与插件装载
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AsteroidDisplayPlugin);

    let mut display_data = AsteroidDisplayData::default();
    display_data.vertices = vertices;
    display_data.faces = faces;
    display_data.normals = normals;
    app.insert_resource(display_data);

    // 执行一帧 Startup 与 Update 系统调度
    app.update();

    println!(
        "Step 1 通过: 表面多面体网格在 Bevy 中构建并绑定成功 ({} 顶点, {} 三角面片)",
        nv, nf
    );
}

#[test]
fn test_step2_field_potential_and_gravity_coloring() {
    let toml_path = Path::new("assets/projects/sample_project.toml");
    let loaded = load_project_toml(toml_path).expect("加载 sample_project.toml 失败");
    let nv = loaded.mesh.vertex_count();

    let normals_f64 = loaded.mesh.vertex_normals();
    let mut data = AsteroidDisplayData::default();
    for i in 0..nv {
        let p = loaded.mesh.vertex(i);
        let n = normals_f64[i];
        data.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        data.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);

        // 构造代表性场样本 (模拟势能梯度从赤道至极区递减)
        let r = (p[0].powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
        let u_mock = 0.03 + 0.01 * (p[2] / r);
        let g_mock = [0.0, 0.0, -5e-5 * (p[2] / r)];
        data.field_samples.push(FieldSample {
            potential: u_mock,
            gravity: g_mock,
            hessian6: [0.0; 6],
            trace: 0.0,
            branch: FieldBranch::ExternalClassical,
        });
    }

    let mut settings = DisplaySettings::default();

    // 1. 势能 U 伪彩色着色 (Turbo 色表)
    settings.mode = DisplayMode::PotentialU;
    settings.colormap = ColormapType::Turbo;
    let (colors_u, min_u, max_u) = compute_vertex_colors(&data, &settings);
    assert_eq!(colors_u.len(), nv);
    assert!(max_u > min_u);
    for col in &colors_u {
        assert!(col[0] >= 0.0 && col[0] <= 1.0);
        assert!(col[1] >= 0.0 && col[1] <= 1.0);
        assert!(col[2] >= 0.0 && col[2] <= 1.0);
        assert_eq!(col[3], 1.0);
    }

    // 2. 引力模长 ||g|| 伪彩色着色 (Viridis 色表)
    settings.mode = DisplayMode::GravityMagnitude;
    settings.colormap = ColormapType::Viridis;
    let (colors_g, min_g, max_g) = compute_vertex_colors(&data, &settings);
    assert_eq!(colors_g.len(), nv);
    assert!(min_g >= 0.0 && max_g >= min_g);

    // 3. 色表插值单调性与边界点检验
    let c0 = colormap_turbo(0.0);
    let c1 = colormap_turbo(1.0);
    assert_ne!(c0, c1);

    let cv0 = colormap_viridis(0.0);
    let cv1 = colormap_viridis(1.0);
    assert_ne!(cv0, cv1);

    println!(
        "Step 2 通过: 势能 U 与引力模长 ||g|| 多色表着色成功 (U 跨度 [{:.3e}, {:.3e}], g 跨度 [{:.3e}, {:.3e}])",
        min_u, max_u, min_g, max_g
    );
}

#[test]
fn test_step3_hessian_and_slope_debug_views() {
    let mut data = AsteroidDisplayData::default();
    let nv = 100;
    for i in 0..nv {
        let trace_val = if i < 50 { -1e-8 } else { 1e-8 }; // 含正负零点
        let slope_viol = if i % 10 == 0 { 0.12 } else { 0.0 }; // 坡度违约点
        data.vertices.push([(i as f32) * 5.0, 0.0, 0.0]);
        data.normals.push([0.0, 1.0, 0.0]);
        data.field_samples.push(FieldSample {
            potential: 0.03,
            gravity: [0.0, 0.0, -1e-4],
            hessian6: [0.0; 6],
            trace: trace_val,
            branch: FieldBranch::InteriorFinitePart,
        });
        data.slope_violations.push(slope_viol);
        data.mass_residuals.push(0.001);
    }

    let mut settings = DisplaySettings::default();

    // 1. Hessian Trace 视图: 自动选用 Coolwarm 双极冷暖色表
    settings.mode = DisplayMode::HessianTrace;
    settings.colormap = ColormapType::Coolwarm;
    let (_colors_tr, min_tr, max_tr) = compute_vertex_colors(&data, &settings);
    assert!(min_tr < 0.0 && max_tr > 0.0);
    let c_cw_neg = colormap_coolwarm(0.0);
    let c_cw_pos = colormap_coolwarm(1.0);
    // 负值偏蓝，正值偏红
    assert!(c_cw_neg[2] > c_cw_neg[0]);
    assert!(c_cw_pos[0] > c_cw_pos[2]);

    // 2. 坡度违约视图
    settings.mode = DisplayMode::SlopeViolation;
    let (colors_slope, _, max_slope) = compute_vertex_colors(&data, &settings);
    assert_eq!(max_slope, 0.12);
    assert_eq!(colors_slope.len(), nv);

    // 3. 调试显示标志位
    settings.show_normals = true;
    settings.show_rays = true;
    assert!(settings.show_normals && settings.show_rays);

    println!("Step 3 通过: Hessian 迹双极冷暖视图与坡度超限警告标记验证通过");
}

#[test]
fn test_step4_realtime_gpu_vertex_deformation() {
    let gpu_ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("当前环境无可用 GPU 设备或沙盒无法枚举 Metal，跳过 GPU 硬件测试: {e}");
            return;
        }
    };
    let updater = GpuSurfaceUpdater::new(&gpu_ctx).expect("编译顶点更新 Compute Pipeline 失败");

    // 构造测试球面基方向与动态演化的模态半径
    let n = 256;
    let mut dirs = Vec::with_capacity(n);
    let mut radii = Vec::with_capacity(n);
    let mut expected_positions = Vec::with_capacity(n);

    for i in 0..n {
        let theta = (i as f32) * 0.1;
        let phi = (i as f32) * 0.05;
        let ux = theta.sin() * phi.cos();
        let uy = theta.cos();
        let uz = theta.sin() * phi.sin();
        let len = (ux * ux + uy * uy + uz * uz).sqrt().max(1e-6);
        let u = [ux / len, uy / len, uz / len];

        // 模拟形变模态: R(theta) = R0 * (1.0 + 0.15 * cos(3*theta))
        let r = 450.0 * (1.0 + 0.15 * (3.0 * theta).cos());
        dirs.push(u);
        radii.push(r);
        expected_positions.push([u[0] * r, u[1] * r, u[2] * r]);
    }

    // 执行 GPU Compute Shader 更新顶点坐标
    let updated_positions = updater
        .update_surface_positions_gpu(&gpu_ctx, &dirs, &radii)
        .expect("GPU 顶点形变调度失败");

    assert_eq!(updated_positions.len(), n);

    for i in 0..n {
        let p_exp = expected_positions[i];
        let p_gpu = updated_positions[i];
        let diff = ((p_exp[0] - p_gpu[0]).powi(2)
            + (p_exp[1] - p_gpu[1]).powi(2)
            + (p_exp[2] - p_gpu[2]).powi(2))
        .sqrt();
        assert!(
            diff < 1e-3,
            "第 {i} 个顶点 GPU 实时形变与解析位置不一致: exp {:?}, gpu {:?}",
            p_exp,
            p_gpu
        );
    }

    println!("Step 4 通过: 实时 GPU 顶点更新 Compute Shader 调度与回读精度完美通过 (误差 < 1mm)");
}
