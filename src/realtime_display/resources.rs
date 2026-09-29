//! Bevy 运行时 Resource 定义
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;
use super::events::DisplayMode;
use crate::model_calculation::contracts::FieldSample;

/// 伪彩色色表枚举
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Reflect)]
pub enum ColormapType {
    /// Google Turbo 色表 (高动态范围、感知均匀)
    #[default]
    Turbo,
    /// Viridis 色表
    Viridis,
    /// Inferno 色表
    Inferno,
    /// 蓝-白-红双极冷暖色表 (适合迹 Tr(H) 正负分布)
    Coolwarm,
    /// 真实岩石灰度
    Grayscale,
}

/// 全局显示与可视化配置 Resource
#[derive(Resource, Clone, Debug)]
pub struct DisplaySettings {
    /// 当前场量显示模式
    pub mode: DisplayMode,
    /// 当前色表
    pub colormap: ColormapType,
    /// 是否渲染表面法向量调试线
    pub show_normals: bool,
    /// 是否渲染射线追踪调试线
    pub show_rays: bool,
    /// 是否渲染线框
    pub show_wireframe: bool,
    /// 法向箭头缩放长度 (米)
    pub normal_scale: f32,
    /// 射线缩放长度 (米)
    pub ray_scale: f32,
    /// 颜色映射下界 (为 None 时自动归一化)
    pub custom_min: Option<f32>,
    /// 颜色映射上界
    pub custom_max: Option<f32>,
    /// 自动色标动态范围
    pub auto_range: bool,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            mode: DisplayMode::SurfaceRealistic,
            colormap: ColormapType::Turbo,
            show_normals: false,
            show_rays: false,
            show_wireframe: false,
            normal_scale: 15.0,
            ray_scale: 200.0,
            custom_min: None,
            custom_max: None,
            auto_range: true,
        }
    }
}

/// 小行星表面状态与全场计算结果 Resource (ECS 共享存储)
#[derive(Resource, Clone, Debug, Default)]
pub struct AsteroidDisplayData {
    /// 顶点位置集合 [x, y, z] (米)
    pub vertices: Vec<[f32; 3]>,
    /// 三角形面片拓扑
    pub faces: Vec<[u32; 3]>,
    /// 顶点法向
    pub normals: Vec<[f32; 3]>,
    /// 顶点极半径
    pub radii: Vec<f32>,
    /// 表面每个顶点的引力场样本 (U, g, H, trace)
    pub field_samples: Vec<FieldSample>,
    /// 坡度约束违反量
    pub slope_violations: Vec<f32>,
    /// 质量局部残差
    pub mass_residuals: Vec<f32>,
}

/// 场景渲染网格与材质句柄 Resource
#[derive(Resource, Clone, Debug)]
pub struct AsteroidMeshHandles {
    pub mesh: Handle<Mesh>,
    pub material: Handle<StandardMaterial>,
}
