//! 输入严格性检查与前置验证
//! 严格依照 docs/software.md 第 5 节的 9 大必备检验规则

use super::schema::ProjectConfig;
use super::unit_system::UnitSystem;
use crate::model_calculation::contracts::{DensityField, DensityKernel};
use crate::rtfp::density_derivatives::eval_density_field;
use crate::rtfp::mesh::Mesh;

/// 配置项验证报告
#[derive(Clone, Debug, Default)]
pub struct InputValidationReport {
    pub is_valid: bool,
    pub mass_estimate_kg: f64,
    pub min_sampled_density: f64,
    pub max_sampled_density: f64,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// 校验项目配置完整性与基本参数
pub fn validate_project_config(
    config: &ProjectConfig,
    _units: &UnitSystem,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // 2. sigma > 0 & 3. alpha > 0
    if config.density.kernels.is_empty() && config.density.mean_density.unwrap_or(0.0) <= 0.0 {
        errors.push("密度配置未包含任何 Cauchy 核 (density.kernels 为空)".to_string());
    }

    for (i, k) in config.density.kernels.iter().enumerate() {
        if k.sigma <= 0.0 {
            errors.push(format!("第 {} 个核 sigma 必须 > 0，当前为 {}", i, k.sigma));
        }
        if k.alpha <= 0.0 {
            errors.push(format!("第 {} 个核 alpha 必须 > 0，当前为 {}", i, k.alpha));
        }
    }

    // 4. 输入路径检查
    if config.point_cloud.is_none() && config.mesh.is_none() {
        errors.push("未指定几何输入源 (必须提供 [point_cloud] 或 [mesh])".to_string());
    }

    // 8. finite_part_scale > 0
    if config.solver.finite_part_scale <= 0.0 {
        errors.push(format!(
            "求解器有限部分标度 finite_part_scale 必须 > 0，当前为 {}",
            config.solver.finite_part_scale
        ));
    }

    if config.solver.near_field_radius <= 0.0 {
        errors.push(format!(
            "求解器近场半径 near_field_radius 必须 > 0，当前为 {}",
            config.solver.near_field_radius
        ));
    }

    // 9. galerkin_order == 4 检查
    if config.solver.galerkin_order != 4 {
        errors.push(format!(
            "当前版本仅严格实现 K=4 径向 Galerkin 降阶模型，当前请求阶数为 {}",
            config.solver.galerkin_order
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// 验证 Cauchy 混合密度在多面体实体范围内的非负性 (采样验证)
/// rho_K(x) >= 0 对几乎所有 x 属于 V
pub fn validate_density_non_negative(
    kernels: &[DensityKernel],
    sample_points: &[[f64; 3]],
) -> Result<(f64, f64), String> {
    if sample_points.is_empty() {
        return Err("无采样点可用于密度非负性验证".to_string());
    }

    let mut min_rho = f64::INFINITY;
    let mut max_rho = f64::NEG_INFINITY;

    for (idx, &p) in sample_points.iter().enumerate() {
        let mut rho = 0.0f64;
        for k in kernels {
            let dx = p[0] - k.center[0];
            let dy = p[1] - k.center[1];
            let dz = p[2] - k.center[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let factor = 1.0 + k.sigma * k.sigma * r2;
            rho += k.weight * factor.powf(-k.alpha);
        }

        if rho < min_rho {
            min_rho = rho;
        }
        if rho > max_rho {
            max_rho = rho;
        }

        // 物理密度必须非负
        if rho < -1e-6 {
            return Err(format!(
                "采样点 #{idx} 处密度为负值 (rho = {:.4e} kg/m³，位置: [{:.2}, {:.2}, {:.2}])，违背物理非负假设",
                rho, p[0], p[1], p[2]
            ));
        }
    }

    Ok((min_rho, max_rho))
}

pub fn validate_density_field_non_negative(
    field: &DensityField,
    sample_points: &[[f64; 3]],
) -> Result<(f64, f64), String> {
    if sample_points.is_empty() { return Err("无采样点可用于密度验证".into()); }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (i, &p) in sample_points.iter().enumerate() {
        let rho = eval_density_field(p, field);
        lo = lo.min(rho); hi = hi.max(rho);
        if rho < -1e-10 { return Err(format!("采样点 #{i} 密度为负: {rho:e}")); }
    }
    Ok((lo, hi))
}

/// 估算多面体实体内部的初始总质量: M = int_V rho(x) dx
/// 采用四面体剖分 + 4 点数值求积
pub fn estimate_initial_mass(kernels: &[DensityKernel], mesh: &Mesh) -> f64 {
    let nf = mesh.face_count();
    let mut total_mass = 0.0f64;

    // 四面体对称 4 点高斯积分权重与重心坐标
    let a = 0.5854101966249685;
    let b = 0.1381966011250105;
    let weights = [0.25, 0.25, 0.25, 0.25];
    let bary = [
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a],
    ];

    let v0_origin = [0.0f64; 3];

    for f in 0..nf {
        let [i0, i1, i2] = mesh.face(f);
        let p1 = mesh.vertex(i0 as usize);
        let p2 = mesh.vertex(i1 as usize);
        let p3 = mesh.vertex(i2 as usize);

        // 顶点到原点构成的四面体带符号体积: 1/6 * det([p1, p2, p3])
        let cross = [
            p2[1] * p3[2] - p2[2] * p3[1],
            p2[2] * p3[0] - p2[0] * p3[2],
            p2[0] * p3[1] - p2[1] * p3[0],
        ];
        let det6 = p1[0] * cross[0] + p1[1] * cross[1] + p1[2] * cross[2];
        let tet_vol = det6 / 6.0;

        let verts = [v0_origin, p1, p2, p3];

        let mut tet_mass = 0.0f64;
        for q in 0..4 {
            let [c0, c1, c2, c3] = bary[q];
            let q_pt = [
                c0 * verts[0][0] + c1 * verts[1][0] + c2 * verts[2][0] + c3 * verts[3][0],
                c0 * verts[0][1] + c1 * verts[1][1] + c2 * verts[2][1] + c3 * verts[3][1],
                c0 * verts[0][2] + c1 * verts[1][2] + c2 * verts[2][2] + c3 * verts[3][2],
            ];

            let mut rho = 0.0f64;
            for k in kernels {
                let dx = q_pt[0] - k.center[0];
                let dy = q_pt[1] - k.center[1];
                let dz = q_pt[2] - k.center[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                let factor = 1.0 + k.sigma * k.sigma * r2;
                rho += k.weight * factor.powf(-k.alpha);
            }
            tet_mass += weights[q] * rho;
        }

        total_mass += tet_vol * tet_mass;
    }

    total_mass
}

pub fn estimate_initial_mass_field(field: &DensityField, mesh: &Mesh) -> f64 {
    let kernel_mass = estimate_initial_mass(&field.kernels, mesh);
    field.bulk_density * mesh.volume_and_centroid().0 + kernel_mass
}
