//! 项目配置 TOML 加载器
//! 严格依照 docs/software.md 第 5 节实现 load_project_toml() 闭环流水线

use std::fs;
use std::path::Path;

use super::input_validation::{estimate_initial_mass_field, validate_project_config};
use super::point_cloud::PointCloud;
use super::schema::ProjectConfig;
use super::surface_reconstruction::{reconstruct_surface, ReconstructionMethod};
use super::unit_system::UnitSystem;
use crate::model_calculation::contracts::{DensityField, DensityKernel};
use crate::rtfp::geometry_validation::{validate_closed_polyhedron, GeometryValidationReport};
use crate::rtfp::mesh::Mesh;

/// 成功加载并验证后的项目上下文
pub struct LoadedProject {
    pub config: ProjectConfig,
    pub units: UnitSystem,
    pub mesh: Mesh,
    pub point_cloud: Option<PointCloud>,
    pub kernels: Vec<DensityKernel>,
    pub density_field: DensityField,
    pub initial_mass: f64,
    pub validation_report: GeometryValidationReport,
}

/// 按照 9 项严格规则加载并检验项目 TOML 配置文件
pub fn load_project_toml(path: &Path) -> Result<LoadedProject, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("无法读取配置文件 '{}': {e}", path.display()))?;

    let config: ProjectConfig = toml::from_str(&content)
        .map_err(|e| format!("TOML 配置解析失败: {e}"))?;

    // 1. 显式单位系统初始化
    let units = UnitSystem::new(&config.project.length_unit, &config.project.density_unit)?;

    // 基础配置检验 (sigma > 0, alpha > 0, galerkin_order == 4, finite_part_scale > 0)
    if let Err(errors) = validate_project_config(&config, &units) {
        return Err(format!("项目配置参数验证不通过:\n  - {}", errors.join("\n  - ")));
    }

    // 转换所有 Cauchy 混合核至标准 SI 单位
    let mut kernels = Vec::with_capacity(config.density.kernels.len());
    for k in &config.density.kernels {
        let (c_si, sig_si, w_si, alpha) = units.convert_cauchy_kernel(k.center, k.sigma, k.weight, k.alpha);
        kernels.push(DensityKernel {
            center: c_si,
            sigma: sig_si,
            weight: w_si,
            alpha,
        });
    }
    let bulk_density = config.density.mean_density
        .map(|v| units.to_si_density(v))
        .unwrap_or(0.0);
    let density_field = DensityField::new(bulk_density, kernels.clone())?;

    // 加载或重建几何多面体网格
    let mut loaded_point_cloud = None;
    let mesh = if let Some(ref pc_cfg) = config.point_cloud {
        let pc_path = Path::new(&pc_cfg.path);
        // 4. 点云非空检查
        let pc = PointCloud::load_from_file(pc_path, &units)?;
        let method = ReconstructionMethod::parse(&pc_cfg.reconstruction, pc_cfg.alpha);
        let m = reconstruct_surface(&pc, &method, pc_cfg.closed_surface_required)?;
        loaded_point_cloud = Some(pc);
        m
    } else if let Some(ref mesh_cfg) = config.mesh {
        let m_path = Path::new(&mesh_cfg.path);
        let scale = mesh_cfg.scale.unwrap_or(1.0) * units.length.to_meters_scale();
        Mesh::load_glb(m_path, scale)?
    } else {
        return Err("未指定任何几何模型输入".to_string());
    };

    // 封闭多面体严格几何核验
    let val_report = validate_closed_polyhedron(&mesh);
    if !val_report.is_valid {
        let err_msg = val_report.errors.join("; ");
        let warn_msg = val_report.warnings.join("; ");
        return Err(format!("网格几何校验失败: {err_msg}; {warn_msg}"));
    }

    // 6. 实体内部与边界的 rho(x) >= 0 采样验证
    let mut sample_points = mesh.observation_points(0.0);
    let (_, centroid) = mesh.volume_and_centroid();
    sample_points.push(centroid);
    for f in 0..mesh.face_count().min(256) {
        let [i0, i1, i2] = mesh.face(f);
        let a = mesh.vertex(i0 as usize);
        let b = mesh.vertex(i1 as usize);
        let c = mesh.vertex(i2 as usize);
        sample_points.push([
            (centroid[0] + a[0] + b[0] + c[0]) * 0.25,
            (centroid[1] + a[1] + b[1] + c[1]) * 0.25,
            (centroid[2] + a[2] + b[2] + c[2]) * 0.25,
        ]);
    }
    super::input_validation::validate_density_field_non_negative(&density_field, &sample_points)?;

    // 7. M = int_V rho dx 的初始质量估计
    let initial_mass = estimate_initial_mass_field(&density_field, &mesh);

    Ok(LoadedProject {
        config,
        units,
        mesh,
        point_cloud: loaded_point_cloud,
        kernels,
        density_field,
        initial_mass,
        validation_report: val_report,
    })
}
