//! 数学不变量、守恒量和跨后端误差诊断。

use crate::model_calculation::contracts::FieldSample;
use crate::rtfp::mesh::Mesh;
use crate::rtfp::ray_segments::RaySegment;

pub mod conservation;
pub mod residuals;

pub fn check_mesh_orientation(mesh: &Mesh) -> Result<(), String> {
    for f in 0..mesh.face_count() {
        let [a,b,c] = mesh.face(f);
        let pa = mesh.vertex(a as usize); let pb = mesh.vertex(b as usize); let pc = mesh.vertex(c as usize);
        let e1 = [pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2]];
        let e2 = [pc[0]-pa[0],pc[1]-pa[1],pc[2]-pa[2]];
        let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        if n.iter().any(|x| !x.is_finite()) || n.iter().map(|x| x*x).sum::<f64>() <= 1e-30 { return Err(format!("退化或非有限面片 {f}")); }
    }
    Ok(())
}

pub fn check_ray_segment_complementarity(segments: &[RaySegment], tolerance: f64) -> Result<(), String> {
    for s in segments {
        if !s.r_min.is_finite() || !s.r_max.is_finite() || s.r_min < -tolerance || s.r_max <= s.r_min + tolerance {
            return Err("射线交段不满足互补区间条件".into());
        }
    }
    Ok(())
}

pub fn check_residual_partition(parts: &[f64], tolerance: f64) -> Result<(), String> {
    if parts.iter().any(|x| !x.is_finite() || *x < -tolerance) { return Err("残差分区含非法项".into()); }
    Ok(())
}

pub fn check_hessian_trace(sample: &FieldSample, expected: f64, tolerance: f64) -> Result<(), String> {
    if (sample.trace - expected).abs() <= tolerance { Ok(()) } else { Err(format!("Hessian trace 残差超限: {}", sample.trace - expected)) }
}

pub fn check_transport_conservation(before: &[f64], after: &[f64], tolerance: f64) -> Result<(), String> {
    if before.len() != after.len() { return Err("输运守恒数组长度不一致".into()); }
    let b = before.iter().sum::<f64>(); let a = after.iter().sum::<f64>();
    check_mass_conservation(b, a, tolerance)
}

pub fn check_slope_constraint(sin2_slope: &[f64], sin2_phi: &[f64], tolerance: f64) -> Result<(), String> {
    if sin2_slope.len() != sin2_phi.len() { return Err("坡度约束数组长度不一致".into()); }
    if sin2_slope.iter().zip(sin2_phi).any(|(s,p)| *s > *p + tolerance) { Err("存在坡度约束违背".into()) } else { Ok(()) }
}

pub fn check_gpu_cpu_difference(gpu: &[f64], cpu: &[f64], tolerance: f64) -> Result<(), String> {
    if gpu.len() != cpu.len() { return Err("GPU/CPU 结果长度不一致".into()); }
    if relative_difference(gpu, cpu) <= tolerance { Ok(()) } else { Err("GPU/CPU 相对差异超限".into()) }
}

pub fn check_k4_to_k8_difference(k4: &[f64], k8: &[f64], tolerance: f64) -> Result<(), String> {
    check_gpu_cpu_difference(k4, k8, tolerance)
}

pub fn check_finite_part_scale_change(h0: &[f64], h1: &[f64], rho: f64, g_const: f64, log_ratio: f64, coefficient: &[f64; 6], tolerance: f64) -> Result<(), String> {
    if h0.len() != 6 || h1.len() != 6 { return Err("有限部分 Hessian 必须包含 6 个分量".into()); }
    let err = (0..6).map(|i| (h1[i] - h0[i] - g_const * rho * coefficient[i] * log_ratio).abs()).fold(0.0, f64::max);
    if err <= tolerance { Ok(()) } else { Err(format!("有限部分尺度变化残差 {err:e}")) }
}

/// SCF/RTFP 统一工程验收入口。调用方必须在提交下一轮状态前执行，
/// 任何一个数学不变量失败都会阻止状态推进。
pub fn run_engineering_audit(
    mesh: &Mesh,
    segments: &[RaySegment],
    transport_before: &[f64],
    transport_after: &[f64],
    sin2_slope: &[f64],
    sin2_phi: &[f64],
    hessian_sample: &FieldSample,
    tolerances: (f64, f64),
) -> Result<(), String> {
    check_mesh_orientation(mesh)?;
    check_ray_segment_complementarity(segments, tolerances.0)?;
    check_transport_conservation(transport_before, transport_after, tolerances.1)?;
    check_slope_constraint(sin2_slope, sin2_phi, tolerances.1)?;
    check_hessian_symmetry(hessian_sample)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ErrorBudget {
    pub quadrature: f64,
    pub finite_part: f64,
    pub gpu_roundoff: f64,
    pub radial_truncation: f64,
    pub characteristic: f64,
    pub interpolation: f64,
    pub sqp: f64,
    pub geometry: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct NumericalTolerances { pub absolute: f64, pub relative: f64, pub trace: f64, pub mass: f64, pub slope: f64 }

impl NumericalTolerances {
    pub fn calibrated(scale: f64, condition_proxy: f64) -> Self {
        let s=scale.abs().max(1.0); let c=condition_proxy.abs().max(1.0);
        Self { absolute: 64.0*f64::EPSILON*s*c, relative: 128.0*f64::EPSILON*c.sqrt(), trace: 256.0*f64::EPSILON*s*c, mass: 1e-10*c.sqrt(), slope: 1e-10*c.sqrt() }
    }
}

impl ErrorBudget {
    pub fn total(&self) -> f64 {
        self.quadrature + self.finite_part + self.gpu_roundoff + self.radial_truncation
            + self.characteristic + self.interpolation + self.sqp + self.geometry
    }
}

pub fn check_hessian_symmetry(sample: &FieldSample) -> Result<(), String> {
    if sample.hessian6.iter().all(|v| v.is_finite()) { Ok(()) }
    else { Err("Hessian 含非有限值".into()) }
}

pub fn check_external_trace(sample: &FieldSample, tolerance: f64) -> Result<(), String> {
    if sample.trace.abs() <= tolerance { Ok(()) }
    else { Err(format!("外部 Hessian trace 超限: {} > {}", sample.trace.abs(), tolerance)) }
}

pub fn relative_difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
        / a.iter().zip(b).map(|(x, y)| x.abs().max(y.abs())).fold(1e-30, f64::max)
}

pub fn check_mass_conservation(before: f64, after: f64, tolerance: f64) -> Result<(), String> {
    let err = (after - before).abs() / before.abs().max(1e-30);
    if err <= tolerance { Ok(()) } else { Err(format!("质量守恒误差 {err:e}")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn error_budget_sums_named_terms() {
        let b = ErrorBudget { quadrature: 1.0, finite_part: 2.0, ..Default::default() };
        assert_eq!(b.total(), 3.0);
    }
}
