//! 标量场 (U, |g|, Tr(H), 坡度, 质量) 伪彩色映射与更新系统
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;
use bevy::render::mesh::{Mesh, VertexAttributeValues};

use super::events::DisplayMode;
use super::resources::{AsteroidDisplayData, AsteroidMeshHandles, ColormapType, DisplaySettings};

/// Google Turbo 高动态范围感知均匀色表拟合多项式
pub fn colormap_turbo(t: f32) -> [f32; 4] {
    let x = t.clamp(0.0, 1.0);
    let r = (0.1357 + x * (4.61539 - x * (42.6603 - x * (132.131 - x * (152.55 - x * 61.386)))))
        .clamp(0.0, 1.0);
    let g = (0.0914 + x * (2.19419 + x * (4.84297 - x * (14.185 - x * (4.2773 - x * 11.295)))))
        .clamp(0.0, 1.0);
    let b = (0.1067 + x * (12.5925 - x * (60.1818 - x * (109.07 - x * (88.506 - x * 26.81)))))
        .clamp(0.0, 1.0);
    [r, g, b, 1.0]
}

/// Viridis 感知均匀色表
pub fn colormap_viridis(t: f32) -> [f32; 4] {
    let x = t.clamp(0.0, 1.0);
    let r = (-0.0007 + x * (0.4137 + x * (-0.0469 + x * (1.8159 - x * 1.1866)))).clamp(0.0, 1.0);
    let g = (0.0036 + x * (1.3323 + x * (-0.7161 + x * (1.0827 - x * 0.7042)))).clamp(0.0, 1.0);
    let b = (0.2827 + x * (0.8351 + x * (-1.5583 + x * (1.8617 - x * 0.4214)))).clamp(0.0, 1.0);
    [r, g, b, 1.0]
}

/// 双极性 Coolwarm 冷暖色表 (适合迹 Tr(H) 等含正负零点的数据)
pub fn colormap_coolwarm(t: f32) -> [f32; 4] {
    let x = t.clamp(0.0, 1.0);
    if x < 0.5 {
        let f = x * 2.0;
        let r = 0.23 + 0.77 * f;
        let g = 0.30 + 0.70 * f;
        let b = 0.75 + 0.25 * f;
        [r, g, b, 1.0]
    } else {
        let f = (x - 0.5) * 2.0;
        let r = 1.0 - 0.30 * f;
        let g = 1.0 - 0.85 * f;
        let b = 1.0 - 0.85 * f;
        [r, g, b, 1.0]
    }
}

/// 真实小行星碳质岩石灰度色表
pub fn colormap_rock(t: f32) -> [f32; 4] {
    let base = 0.15 + 0.15 * t.clamp(0.0, 1.0);
    [base * 1.05, base * 1.0, base * 0.95, 1.0]
}

/// 根据色表枚举计算归一化标量颜色
pub fn sample_colormap(t: f32, cmap: ColormapType) -> [f32; 4] {
    match cmap {
        ColormapType::Turbo => colormap_turbo(t),
        ColormapType::Viridis => colormap_viridis(t),
        ColormapType::Coolwarm => colormap_coolwarm(t),
        ColormapType::Inferno => colormap_turbo(t),
        ColormapType::Grayscale => colormap_rock(t),
    }
}

/// 从场样本与诊断量提取当前模式的目标标量值
pub fn extract_scalar_value(
    data: &AsteroidDisplayData,
    idx: usize,
    mode: DisplayMode,
) -> f32 {
    match mode {
        DisplayMode::SurfaceRealistic => 1.0,
        DisplayMode::PotentialU => {
            if idx < data.field_samples.len() {
                data.field_samples[idx].potential as f32
            } else {
                0.0
            }
        }
        DisplayMode::GravityMagnitude => {
            if idx < data.field_samples.len() {
                let g = data.field_samples[idx].gravity;
                (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt() as f32
            } else {
                0.0
            }
        }
        DisplayMode::HessianTrace => {
            if idx < data.field_samples.len() {
                data.field_samples[idx].trace as f32
            } else {
                0.0
            }
        }
        DisplayMode::SlopeViolation => {
            if idx < data.slope_violations.len() {
                data.slope_violations[idx]
            } else {
                0.0
            }
        }
        DisplayMode::MassResidual => {
            if idx < data.mass_residuals.len() {
                data.mass_residuals[idx]
            } else {
                0.0
            }
        }
    }
}

/// 计算小行星网格所有顶点的颜色数组与当前极值范围
pub fn compute_vertex_colors(
    data: &AsteroidDisplayData,
    settings: &DisplaySettings,
) -> (Vec<[f32; 4]>, f32, f32) {
    let n = data.vertices.len();
    if n == 0 {
        return (Vec::new(), 0.0, 1.0);
    }

    if settings.mode == DisplayMode::SurfaceRealistic {
        let colors = vec![[0.22, 0.20, 0.18, 1.0]; n];
        return (colors, 0.0, 1.0);
    }

    let mut scalars = Vec::with_capacity(n);
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;

    for i in 0..n {
        let val = extract_scalar_value(data, i, settings.mode);
        scalars.push(val);
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    let actual_min = settings.custom_min.unwrap_or(min_val);
    let actual_max = settings.custom_max.unwrap_or(max_val);
    let range = if (actual_max - actual_min).abs() > 1e-12 {
        actual_max - actual_min
    } else {
        1.0
    };

    let mut colors = Vec::with_capacity(n);
    for val in scalars {
        let norm_t = ((val - actual_min) / range).clamp(0.0, 1.0);
        colors.push(sample_colormap(norm_t, settings.colormap));
    }

    (colors, actual_min, actual_max)
}

/// Bevy ECS 系统: 响应显示模式变化与场数据更新，刷新 Mesh 顶点颜色
pub fn update_field_colors_system(
    data: Res<AsteroidDisplayData>,
    settings: Res<DisplaySettings>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    handles: Option<Res<AsteroidMeshHandles>>,
) {
    if !data.is_changed() && !settings.is_changed() {
        return;
    }

    let (Some(mut meshes), Some(handles)) = (meshes, handles) else {
        return;
    };

    let Some(mut mesh) = meshes.get_mut(&handles.mesh) else {
        return;
    };

    let (colors, _min_v, _max_v) = compute_vertex_colors(&data, &settings);
    if colors.len() == data.vertices.len() && !colors.is_empty() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colors));
    }
}
