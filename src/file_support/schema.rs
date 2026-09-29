//! TOML 配置文件数据结构 Schema
//! 严格依照 docs/software.md 第 5 节 定义

use serde::{Deserialize, Serialize};

/// 项目总配置根节点
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project: ProjectMeta,
    pub point_cloud: Option<PointCloudConfig>,
    pub mesh: Option<MeshConfig>,
    pub rotation: Option<RotationConfig>,
    #[serde(default)]
    pub solver: SolverConfig,
    pub density: DensityConfig,
}

/// 项目元信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub name: String,
    #[serde(default = "default_length_unit")]
    pub length_unit: String,
    #[serde(default = "default_density_unit")]
    pub density_unit: String,
}

fn default_length_unit() -> String {
    "m".to_string()
}

fn default_density_unit() -> String {
    "kg/m3".to_string()
}

/// 点云输入配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointCloudConfig {
    pub path: String,
    #[serde(default)]
    pub has_normals: bool,
    #[serde(default = "default_reconstruction")]
    pub reconstruction: String,
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default = "default_true")]
    pub closed_surface_required: bool,
}

fn default_reconstruction() -> String {
    "spherical_grid".to_string()
}

fn default_true() -> bool {
    true
}

/// 网格输入配置 (如 GLB / OBJ)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshConfig {
    pub path: String,
    #[serde(default = "default_scale")]
    pub scale: Option<f64>,
}

fn default_scale() -> Option<f64> {
    Some(1.0)
}

/// 旋转配置 (角速度矢量)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotationConfig {
    #[serde(default)]
    pub omega: [f64; 3],
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            omega: [0.0, 0.0, 0.0],
        }
    }
}

/// 求解器参数配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverConfig {
    #[serde(default = "default_galerkin_order")]
    pub galerkin_order: usize,
    #[serde(default = "default_rt_directions")]
    pub rt_directions: usize,
    #[serde(default = "default_radial_quadrature")]
    pub radial_quadrature: usize,
    #[serde(default = "default_near_field_radius")]
    pub near_field_radius: f64,
    #[serde(default = "default_finite_part_scale")]
    pub finite_part_scale: f64,
    #[serde(default = "default_under_relaxation")]
    pub under_relaxation: f64,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            galerkin_order: default_galerkin_order(),
            rt_directions: default_rt_directions(),
            radial_quadrature: default_radial_quadrature(),
            near_field_radius: default_near_field_radius(),
            finite_part_scale: default_finite_part_scale(),
            under_relaxation: default_under_relaxation(),
            max_iterations: default_max_iterations(),
        }
    }
}

fn default_galerkin_order() -> usize { 4 }
fn default_rt_directions() -> usize { 288 }
fn default_radial_quadrature() -> usize { 16 }
fn default_near_field_radius() -> f64 { 0.01 }
fn default_finite_part_scale() -> f64 { 1.0 }
fn default_under_relaxation() -> f64 { 0.5 }
fn default_max_iterations() -> usize { 1000 }

/// 密度场配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensityConfig {
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub mean_density: Option<f64>,
    #[serde(default)]
    pub alpha_default: Option<f64>,
    pub kernels: Vec<KernelItemConfig>,
}

/// 单个 Cauchy 密度核配置项
/// 支持 center / c, weight / w 等别名
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelItemConfig {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(alias = "c")]
    pub center: [f64; 3],
    #[serde(alias = "w")]
    pub weight: f64,
    pub sigma: f64,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
}

fn default_alpha() -> f64 {
    1.0
}
