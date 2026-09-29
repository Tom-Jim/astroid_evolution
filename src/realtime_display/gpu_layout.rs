//! WGSL 严格内存对齐结构
//! 依照 docs/software.md 第 12 节 与 第 13 节

use bytemuck::{Pod, Zeroable};

/// 渲染与显示 Uniform 数据 (256 字节对齐安全)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DisplayUniformsGpu {
    /// 观察-投影变换矩阵 (4x4, 64 字节)
    pub view_proj: [f32; 16],
    /// 摄像机视点位置 (vec4<f32>, 16 字节)
    pub camera_pos: [f32; 4],
    /// 光照方向 (vec4<f32>, 16 字节)
    pub light_dir: [f32; 4],
    /// 基础颜色 (vec4<f32>, 16 字节)
    pub base_color: [f32; 4],
    /// 标量场映射参数: [min_val, max_val, mode, colormap_type]
    pub field_params: [f32; 4],
    /// 调试参数: [normal_scale, ray_scale, wireframe_flag, padding]
    pub debug_params: [f32; 4],
    /// 预留拓展填充 (64 字节)
    pub _reserved: [f32; 16],
}

impl Default for DisplayUniformsGpu {
    fn default() -> Self {
        let mut id = [0.0f32; 16];
        id[0] = 1.0;
        id[5] = 1.0;
        id[10] = 1.0;
        id[15] = 1.0;
        Self {
            view_proj: id,
            camera_pos: [0.0, 0.0, 1500.0, 1.0],
            light_dir: [0.577, 0.577, 0.577, 0.0],
            base_color: [0.75, 0.72, 0.68, 1.0], // Ryugu 碳质小行星灰褐色基调
            field_params: [0.0, 1.0, 0.0, 0.0],
            debug_params: [10.0, 50.0, 0.0, 0.0],
            _reserved: [0.0; 16],
        }
    }
}

/// 顶点渲染数据 (vec4 内存对齐)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VertexGpu {
    /// 顶点位置 [x, y, z, 1.0]
    pub position: [f32; 4],
    /// 顶点法向 [nx, ny, nz, 0.0]
    pub normal: [f32; 4],
    /// 顶点颜色 [r, g, b, a]
    pub color: [f32; 4],
    /// 标量场采样与属性 [U, |g|, Tr(H), slope]
    pub attributes: [f32; 4],
}

/// 调试线段顶点 (法向箭头、引力射线)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DebugLineVertexGpu {
    pub position: [f32; 4],
    pub color: [f32; 4],
}
