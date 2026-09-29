//! 统一场计算后端契约。CPU、GPU 与未来硬件后端都必须返回同一组 U/g/H/分支结果。
use crate::model_calculation::contracts::{DensityField, FieldSample};
use super::cpu_reference::{CpuReferenceSolver, CpuSolverOptions};
use super::bvh::FlattenedBvh;
use super::mesh::Mesh;
use super::gpu_dispatch::GpuBvhWgslBackend;

pub trait FieldBackend: Send + Sync {
    fn evaluate(&self, points: &[[f64; 3]], inside: &[bool]) -> Result<Vec<FieldSample>, String>;
    fn name(&self) -> &'static str;
    fn validate_samples(&self, samples: &[FieldSample]) -> Result<(), String> {
        if samples.iter().any(|s| {
            !s.potential.is_finite() || s.gravity.iter().any(|v| !v.is_finite())
                || s.hessian6.iter().any(|v| !v.is_finite()) || !s.trace.is_finite()
        }) { return Err(format!("{} 返回非有限场量", self.name())); }
        for s in samples {
            if matches!(s.branch, crate::model_calculation::contracts::FieldBranch::ExternalClassical) && s.trace.abs() > 1e-7 {
                return Err(format!("{} 外部场点 trace 不满足调和条件: {}", self.name(), s.trace));
            }
        }
        Ok(())
    }
}

pub struct CpuFieldBackend<'a> { solver: CpuReferenceSolver<'a> }

impl<'a> CpuFieldBackend<'a> {
    pub fn new(mesh: &'a Mesh, bvh: &'a FlattenedBvh, density: &'a DensityField, options: CpuSolverOptions) -> Self {
        Self { solver: CpuReferenceSolver::new_with_density_field(mesh, bvh, density, options) }
    }
}

impl FieldBackend for CpuFieldBackend<'_> {
    fn evaluate(&self, points: &[[f64; 3]], inside: &[bool]) -> Result<Vec<FieldSample>, String> {
        if points.len() != inside.len() { return Err("场点与 inside 标记数量不一致".into()); }
        let out: Vec<FieldSample> = points.iter().map(|p| self.solver.compute_field_at_point(*p)).collect();
        self.validate_samples(&out)?;
        Ok(out)
    }
    fn name(&self) -> &'static str { "cpu-f64-reference" }
}

impl FieldBackend for GpuBvhWgslBackend<'_> {
    fn evaluate(&self, points: &[[f64; 3]], inside: &[bool]) -> Result<Vec<FieldSample>, String> {
        let out = self.compute_field_batch(points, inside)?;
        self.validate_samples(&out)?;
        Ok(out)
    }
    fn name(&self) -> &'static str { "gpu-wgsl-f32" }
}
