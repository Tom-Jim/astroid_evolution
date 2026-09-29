//! SCF 与 Galerkin 模态计算模块
//! 包含 K=4 径向基函数、矩阵装配、守恒状态转换、半拉格朗日输运、度量计算、坡度闭合、
//! 质量约束线性化、SQP/QP 几何投影、欠松弛阻尼、收敛判据及 SCF 完整闭环管线

pub mod conservation;
pub mod conservative_state;
pub mod convergence;
pub mod galerkin;
pub mod galerkin_matrices;
pub mod galerkin_k8;
pub mod mass_constraint;
pub mod modal_velocity;
pub mod pipeline;
pub mod semi_lagrangian;
pub mod slope_closure;
pub mod sqp_projection;
pub mod surface_metric;
pub mod under_relaxation;
pub mod surface_state;
pub mod source_update;
pub mod mesh_update;
pub mod state;
pub mod engine;
