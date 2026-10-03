//! 阶段 7 完整集成测试套件：WASM 浏览器运行、WebGPU 硬件探测、资源加载、兼容性检查与 CPU Fallback
//! 严格依照 docs/software.md 第 14、15.7 节 与 ryugu_cauchy_gravity 实现
//!
//! 覆盖阶段 7 推荐实现顺序的 5 个关键步骤：
//! 1. WebGPU capability probe (硬件能力探测与降级原因判定)
//! 2. 浏览器资源加载 (远程 Fetch、文件拖放 Drag & Drop 点云解析、MIME 检测)
//! 3. WGSL 兼容性检查 (着色器语法验证、f64 双精度禁用检查、计算入口点扫描)
//! 4. CPU fallback (计算分发器平滑降级与 Staging 缓冲区类型转换回读)
//! 5. 浏览器显示和交互 (Canvas 视口管理、Retina HiDPI 缩放、WasmSession 控制生命周期)

use astroid_evolution::wasm_browser_display::asset_fetch::{
    AssetType, detect_asset_type, fetch_asset_bytes, fetch_asset_text,
    parse_dropped_xyz_point_cloud,
};
use astroid_evolution::wasm_browser_display::canvas::CanvasViewport;
use astroid_evolution::wasm_browser_display::capability_probe::{
    FallbackReason, check_wgsl_compatibility, probe_webgpu_support,
};
use astroid_evolution::wasm_browser_display::readback::cast_readback_slice;
use astroid_evolution::wasm_browser_display::wasm_entry::{ColorMode, ViewerCommand, WasmSession};
use astroid_evolution::wasm_browser_display::web_gpu::{ComputeDispatcher, SolverBackendMode};

// ============================================================================
// 步骤 1：WebGPU capability probe (硬件能力探测与降级原因判定)
// ============================================================================
#[test]
fn test_step1_webgpu_capability_probe() {
    let report = probe_webgpu_support();

    // 在原生测试环境中，报告应当如实反映系统能力或提供降级原因
    if report.supported {
        assert!(report.max_compute_workgroup_size_x > 0);
        assert!(report.max_storage_buffer_binding_size > 0);
        assert!(!report.adapter_info.is_empty());
    } else {
        assert!(report.fallback_reason.is_some());
    }

    // 验证所有降级原因的描述文本完备性
    let reasons = [
        FallbackReason::NoWebGpu,
        FallbackReason::AdapterNotFound,
        FallbackReason::DeviceLost,
        FallbackReason::LimitsExceeded,
        FallbackReason::ShaderCompilationFailed,
        FallbackReason::UserForcedCpu,
    ];

    for reason in &reasons {
        let desc = reason.description();
        assert!(!desc.is_empty());
    }
}

// ============================================================================
// 步骤 2：浏览器资源加载 (类型检测、拖放点云解析、配置读取)
// ============================================================================
#[test]
fn test_step2_browser_resource_loading() {
    // 1. 测试文件扩展名类型判断
    assert_eq!(detect_asset_type("model.glb"), AssetType::GlbModel);
    assert_eq!(detect_asset_type("model.GLTF"), AssetType::GlbModel);
    assert_eq!(detect_asset_type("config.toml"), AssetType::ConfigToml);
    assert_eq!(detect_asset_type("parameters.json"), AssetType::ConfigToml);
    assert_eq!(detect_asset_type("compute.wgsl"), AssetType::WgslShader);
    assert_eq!(detect_asset_type("asteroid.xyz"), AssetType::PointCloud);
    assert_eq!(detect_asset_type("lidar.txt"), AssetType::PointCloud);

    // 2. 测试拖放/上传的 XYZ 散乱点云文本解析
    let xyz_content = b"# Asteroid surface lidar samples\n\
        100.5 200.25 -50.125\n\
        \n\
        // Second line with comma delimiter\n\
        110.0, 210.5, -45.0\n\
        0.0  0.0  0.0\n";

    let points = parse_dropped_xyz_point_cloud(xyz_content).expect("Point cloud parsing failed");
    assert_eq!(points.len(), 3);
    assert_eq!(points[0], [100.5, 200.25, -50.125]);
    assert_eq!(points[1], [110.0, 210.5, -45.0]);
    assert_eq!(points[2], [0.0, 0.0, 0.0]);

    // 3. 测试非法点云数据的错误拦截
    assert!(parse_dropped_xyz_point_cloud(b"").is_err());
    assert!(parse_dropped_xyz_point_cloud(b"# only comments").is_err());
    assert!(parse_dropped_xyz_point_cloud(b"1.0 2.0").is_err()); // 缺少 Z 分量

    // 4. 测试本地资源加载
    let future = async {
        let shader_bytes = fetch_asset_bytes("src/scf_calculation/shaders/asteroid_scf.wgsl").await;
        assert!(shader_bytes.is_ok());

        let shader_text = fetch_asset_text("src/scf_calculation/shaders/asteroid_scf.wgsl").await;
        assert!(shader_text.is_ok());
        assert!(shader_text.unwrap().contains("@compute"));
    };
    pollster::block_on(future);
}

// ============================================================================
// 步骤 3：WGSL 兼容性检查 (着色器语法验证、f64 双精度禁用检查、入口点扫描)
// ============================================================================
#[test]
fn test_step3_wgsl_compatibility_checks() {
    // 1. 标准合法计算着色器：使用 f32 与 @compute
    let valid_wgsl = r#"
        struct Uniforms {
            omega: f32,
            gamma: f32,
        };
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(64, 1, 1)
        fn main_compute(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            data[idx] = data[idx] * u.omega;
        }

        fn helper_func(val: f32) -> f32 {
            return val * 2.0;
        }
    "#;

    let valid_report = check_wgsl_compatibility(valid_wgsl);
    assert!(valid_report.compatible);
    assert!(!valid_report.contains_f64_keyword);
    assert!(
        valid_report
            .entry_points
            .contains(&"main_compute".to_string())
    );
    assert!(
        valid_report
            .entry_points
            .contains(&"helper_func".to_string())
    );

    // 2. 非法双精度着色器：WebGPU WGSL 不支持 f64，应被检出并标记警告
    let invalid_wgsl = r#"
        @compute @workgroup_size(32)
        fn compute_f64() {
            let x: f64 = 1.0;
        }
    "#;

    let invalid_report = check_wgsl_compatibility(invalid_wgsl);
    assert!(!invalid_report.compatible);
    assert!(invalid_report.contains_f64_keyword);
    assert!(!invalid_report.notes.is_empty());

    // 3. 验证项目中真实的 WGSL 着色器源码均兼容 WebGPU 规范 (无 f64 关键字)
    let project_shaders = [
        "src/scf_calculation/shaders/asteroid_scf.wgsl",
        "src/rtfp/shaders/ray_count.wgsl",
        "src/rtfp/shaders/rtfp_integrate.wgsl",
    ];

    for path in &project_shaders {
        if let Ok(source) = std::fs::read_to_string(path) {
            let report = check_wgsl_compatibility(&source);
            assert!(
                report.compatible,
                "着色器 {} 包含不兼容 WGSL 语法的结构: {:?}",
                path, report.notes
            );
        }
    }
}

// ============================================================================
// 步骤 4：CPU fallback (计算分发器平滑降级与 Staging 缓冲区类型转换回读)
// ============================================================================
#[test]
fn test_step4_cpu_fallback_and_readback() {
    // 1. 创建强制 CPU 参考解模式的调度器
    let dispatcher = ComputeDispatcher::create_cpu_only();
    assert!(!dispatcher.is_gpu_active());
    assert_eq!(
        dispatcher.mode,
        SolverBackendMode::CpuFallback(FallbackReason::UserForcedCpu)
    );

    // 2. 模拟 Staging 缓冲区到 CPU 切片的零拷贝安全转换 (bytemuck Pod)
    let original_values: Vec<f32> = vec![1.23, -4.56, 7.89, 0.0, 42.0];
    let raw_bytes: &[u8] = bytemuck::cast_slice(&original_values);

    let casted_vec: Vec<f32> = cast_readback_slice(raw_bytes).expect("Readback slice cast failed");
    assert_eq!(casted_vec, original_values);

    // 3. 验证非对齐/残缺字节切片能被安全拦截
    let malformed_bytes = &raw_bytes[0..raw_bytes.len() - 1]; // 长度不是 sizeof(f32)=4 的整数倍
    assert!(cast_readback_slice::<f32>(malformed_bytes).is_err());
}

// ============================================================================
// 步骤 5：浏览器显示和交互 (Canvas 视口管理、Retina HiDPI 缩放、WasmSession 生命周期)
// ============================================================================
#[test]
fn test_step5_browser_display_and_interaction() {
    // 1. 验证 Canvas 视口与 Retina (DPR=2.0) 像素缩放
    let css_w = 960.0;
    let css_h = 540.0;
    let dpr = 2.0;

    let viewport = CanvasViewport::from_css_size(css_w, css_h, dpr);
    assert_eq!(viewport.width, 1920); // 物理像素
    assert_eq!(viewport.height, 1080);
    assert_eq!(viewport.device_pixel_ratio, 2.0);
    assert!((viewport.aspect_ratio() - 16.0 / 9.0).abs() < 1e-6);
    assert!((viewport.css_width() - css_w).abs() < 1e-6);
    assert!((viewport.css_height() - css_h).abs() < 1e-6);

    // 2. 验证 WasmSession 完整控制生命周期
    let mut session = WasmSession::new();
    assert!(!session.is_running());
    assert_eq!(session.current_step(), 0);
    assert_eq!(session.selected_model(), "Ryugu");

    // 启动与单步推进
    session.run();
    assert!(session.is_running());

    let step_1 = session.step();
    let step_2 = session.step();
    assert_eq!(step_1, 1);
    assert_eq!(step_2, 2);
    assert_eq!(session.current_step(), 2);

    // 暂停
    session.pause();
    assert!(!session.is_running());
    assert_eq!(session.current_step(), 2);

    // 模型切换
    session.select_model("Phobos");
    assert_eq!(session.selected_model(), "Phobos");

    // 重置
    session.reset();
    assert!(!session.is_running());
    assert_eq!(session.current_step(), 0);

    // 3. 验证 JSON 硬件能力导出
    let cap_json = session.get_capability_json();
    assert!(cap_json.contains("\"supported\":"));
    assert!(cap_json.contains("\"adapter_info\":"));
    assert!(cap_json.contains("\"max_compute_workgroup_size_x\":"));

    // 4. 验证交互控制指令与颜色模式序列化兼容性
    let cmd = ViewerCommand::SetColorMode(ColorMode::EffectivePotential);
    let cmd_toml = toml::to_string(&cmd).unwrap_or_default();
    assert!(cmd_toml.contains("EffectivePotential"));
}
