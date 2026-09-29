//! Bevy 显示事件定义
//! 依照 docs/software.md 第 13 节

use bevy::prelude::*;

/// 场量与诊断可视化模式
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Reflect)]
pub enum DisplayMode {
    /// 表面几何真实渲染
    #[default]
    SurfaceRealistic,
    /// 势能 U(x) 伪彩色分布
    PotentialU,
    /// 引力模长 ||g(x)|| 伪彩色分布
    GravityMagnitude,
    /// 引力梯度迹 Tr(H) (自由空间无源与内部泊松源)
    HessianTrace,
    /// 坡度约束违反率 / 摩擦角限制
    SlopeViolation,
    /// 质量守恒误差分布
    MassResidual,
}

/// 切换可视化模式事件
#[derive(Event, Clone, Copy, Debug)]
pub struct ChangeDisplayModeEvent {
    pub mode: DisplayMode,
}

/// 表面网格顶点实时更新事件
#[derive(Event, Clone, Debug)]
pub struct SurfaceMeshUpdatedEvent {
    pub vertex_count: usize,
}

/// 场量计算完成与重新着色事件
#[derive(Event, Clone, Debug)]
pub struct FieldUpdatedEvent {
    pub mode: DisplayMode,
    pub min_val: f32,
    pub max_val: f32,
}
