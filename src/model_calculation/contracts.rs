//! 跨模块统一数据契约
//! 严格依照 docs/software.md 第 6 节 定义

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// 几何网格状态
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryState {
    /// 顶点坐标 [x, y, z] (米)
    pub vertices: Vec<[f32; 3]>,
    /// 三角面片拓扑索引 (基于 0)
    pub faces: Vec<[u32; 3]>,
    /// 顶点外法向量
    pub normals: Vec<[f32; 3]>,
    /// 顶点极半径 ||v - centroid||
    pub radii: Vec<f32>,
    /// 多面体闭合体积 (m³)
    pub volume: f64,
    /// 质心坐标 (m)
    pub centroid: [f64; 3],
}

/// Cauchy 密度核定义:
/// rho(y) = sum_k w_k [1 + sigma_k^2 ||y - c_k||^2]^(-alpha_k)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct DensityKernel {
    /// 核中心坐标 c_k (m)
    pub center: [f64; 3],
    /// 尺度逆长 sigma_k (m⁻¹), sigma_k > 0
    pub sigma: f64,
    /// 密度权重 w_k (kg/m³)
    pub weight: f64,
    /// 幂次指数 alpha_k, alpha_k > 0
    pub alpha: f64,
}

/// 完整密度场契约。`bulk_density` 是均匀背景项，核集合是空间变化项。
/// 所有严格求解器都应使用此结构，避免把均匀密度误当成空核集合。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DensityField {
    pub bulk_density: f64,
    pub kernels: Vec<DensityKernel>,
}

impl DensityField {
    pub fn new(bulk_density: f64, kernels: Vec<DensityKernel>) -> Result<Self, String> {
        if !bulk_density.is_finite() || bulk_density < 0.0 {
            return Err("均匀背景密度必须是有限非负数".into());
        }
        if kernels.iter().any(|k| !k.sigma.is_finite() || k.sigma <= 0.0
            || !k.alpha.is_finite() || k.alpha <= 0.0
            || !k.weight.is_finite() || !k.center.iter().all(|v| v.is_finite())) {
            return Err("Cauchy 密度核包含非法参数".into());
        }
        Ok(Self { bulk_density, kernels })
    }
}

/// 场点计算结果与数值分支
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldSample {
    /// 引力势 U (m²/s²)
    pub potential: f64,
    /// 引力加速度 g = grad U (m/s²)
    pub gravity: [f64; 3],
    /// 引力梯度张量 Hessian U (6 独立分量: xx, yy, zz, xy, xz, yz) (s⁻²)
    pub hessian6: [f64; 6],
    /// 张量迹 Tr(H) = Delta U = -4 pi G vartheta_x rho(x) (s⁻²)
    pub trace: f64,
    /// 场点对应的求解分支
    pub branch: FieldBranch,
}

/// 数学分支枚举 (严格区分经典导数与有限部分)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldBranch {
    /// 外部真空点 (经典 Newton/Feynman, Tr(H) = 0)
    ExternalClassical,
    /// 内部光滑点 (球形有限部分, Tr(H) = -4 pi G rho)
    InteriorFinitePart,
    /// 三角面片内部点 (半球有限部分, Tr(H) = -2 pi G rho)
    BoundaryFinitePart,
    /// 边棱点 (二面角切锥有限部分, Tr(H) = -4 pi G vartheta rho)
    EdgeFinitePart,
    /// 顶点 (多面体立体角切锥有限部分)
    VertexFinitePart,
}

/// Galerkin K=4 表面径向模态状态
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModalState {
    /// 物理模态系数 q
    pub q: Vec<[f32; 4]>,
    /// 守恒变量 m = M q
    pub m: Vec<[f32; 4]>,
    /// 模态质量矩阵 M (4x4)
    pub mass_matrix: [[f32; 4]; 4],
}

/// 流变学状态变量
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RheologyState {
    pub pressure: f32,
    pub density: f32,
    pub packing_fraction: f32,
    pub coordination: f32,
    pub shear_rate: f32,
    pub inertial_number: f32,
    pub friction_angle: f32,
}

/// SCF 迭代全局诊断量
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ScfDiagnostics {
    pub geometry_residual: f64,
    pub mass_residual: f64,
    pub slope_residual: f64,
    pub transport_residual: f64,
    pub kkt_residual: f64,
}

/// GPU 内存对齐的 Cauchy 核结构 (16 字节对齐 vec4)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct KernelGpu {
    /// [c_x, c_y, c_z, sigma]
    pub center_sigma: [f32; 4],
    /// [w, alpha, padding_0, padding_1]
    pub weight_alpha: [f32; 4],
}

impl DensityKernel {
    pub fn to_gpu(&self) -> KernelGpu {
        KernelGpu {
            center_sigma: [
                self.center[0] as f32,
                self.center[1] as f32,
                self.center[2] as f32,
                self.sigma as f32,
            ],
            weight_alpha: [
                self.weight as f32,
                self.alpha as f32,
                0.0,
                0.0,
            ],
        }
    }
}
