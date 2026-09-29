//! 几何与射线调试渲染 (法向、射线采样、坡度超限标记)
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;

use super::events::DisplayMode;
use super::resources::{AsteroidDisplayData, DisplaySettings};

/// 法向调试实体标记 Component
#[derive(Component, Debug, Default, Clone)]
pub struct NormalDebugEntity;

/// 射线调试实体标记 Component
#[derive(Component, Debug, Default, Clone)]
pub struct RayDebugEntity;

/// 误差曲线与图表实体标记 Component
#[derive(Component, Debug, Default, Clone)]
pub struct ErrorGraphEntity;

/// Bevy Gizmo 调试绘制系统: 法向量箭头、射线采样束与坡度超限警告
pub fn debug_render_gizmos_system(
    gizmos: Option<Gizmos>,
    data: Res<AsteroidDisplayData>,
    settings: Res<DisplaySettings>,
) {
    let Some(mut gizmos) = gizmos else {
        return;
    };
    let nv = data.vertices.len();
    if nv == 0 {
        return;
    }

    // 1. 表面外法向量箭头绘制 (抽样步长防止拥挤)
    if settings.show_normals && !data.normals.is_empty() {
        let step = (nv / 120).max(1);
        let scale = settings.normal_scale;
        let arrow_color = Color::srgb(0.2, 0.8, 1.0); // 浅蓝青色

        for i in (0..nv).step_by(step) {
            let p = Vec3::from_array(data.vertices[i]);
            let n = Vec3::from_array(data.normals[i]).normalize_or_zero();
            let tip = p + n * scale;
            gizmos.arrow(p, tip, arrow_color);
        }
    }

    // 2. 引力计算射线采样可视化
    if settings.show_rays && !data.field_samples.len() == 0 {
        let step = (nv / 20).max(1);
        let ray_scale = settings.ray_scale;
        let ray_color = Color::srgba(1.0, 0.8, 0.2, 0.5); // 半透明金黄

        for i in (0..nv).step_by(step) {
            let p = Vec3::from_array(data.vertices[i]);
            // 可视化局域引力加速度方向
            if i < data.field_samples.len() {
                let g = data.field_samples[i].gravity;
                let g_vec = Vec3::new(g[0] as f32, g[1] as f32, g[2] as f32);
                let g_dir = g_vec.normalize_or_zero();
                gizmos.arrow(p, p + g_dir * ray_scale, ray_color);
            }
        }
    }

    // 3. 坡度约束违反警告标记
    if settings.mode == DisplayMode::SlopeViolation && !data.slope_violations.is_empty() {
        let warn_color = Color::srgb(1.0, 0.1, 0.1); // 亮红警告
        for i in 0..nv {
            if i < data.slope_violations.len() && data.slope_violations[i] > 0.05 {
                let p = Vec3::from_array(data.vertices[i]);
                gizmos.sphere(p, 5.0, warn_color);
            }
        }
    }
}
