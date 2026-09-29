//! K=8 径向 Galerkin 实际闭合：矩阵、质量逆和模态投影。
use super::galerkin_matrices::{assemble_k8_mass_matrix, invert_k8_mass_matrix};
use crate::rtfp::field_backend::FieldBackend;
use crate::model_calculation::contracts::FieldSample;

pub type K8State = [f64; 8];

pub fn project_k8(rhs: &K8State) -> Result<K8State, String> {
    let inv = invert_k8_mass_matrix()?;
    let mut out=[0.0;8];
    for i in 0..8 { out[i]=(0..8).map(|j| inv[i][j]*rhs[j]).sum(); }
    Ok(out)
}

pub fn reconstruct_k8(coefficients: &K8State, basis: &[f64; 8]) -> f64 {
    coefficients.iter().zip(basis).map(|(a,b)| a*b).sum()
}

pub fn k8_condition_proxy() -> f64 {
    let m=assemble_k8_mass_matrix();
    let max_row=m.iter().map(|r| r.iter().map(|x|x.abs()).sum::<f64>()).fold(0.0,f64::max);
    max_row / m[7][7].abs().max(1e-30)
}

pub fn transport_k8(current: &[K8State], jacobians: &[f64], dt: f64, source: Option<&[K8State]>) -> Result<Vec<K8State>, String> {
    if current.len()!=jacobians.len() || source.map_or(false, |s| s.len()!=current.len()) { return Err("K=8 输运输入长度不一致".into()); }
    let mut out=Vec::with_capacity(current.len());
    for (i, q) in current.iter().enumerate() {
        let mut next=[0.0;8];
        for k in 0..8 { next[k]=q[k]*jacobians[i] + source.map_or(0.0, |s| dt*s[i][k]); }
        out.push(next);
    }
    Ok(out)
}

/// K=8 单步闭合：守恒模态输运后通过 M^{-1} 投影回物理模态。
pub fn advance_k8(current: &[K8State], jacobians: &[f64], dt: f64, source: Option<&[K8State]>) -> Result<Vec<K8State>, String> {
    let transported = transport_k8(current, jacobians, dt, source)?;
    transported.iter().map(project_k8).collect()
}

pub struct K8StepOutput { pub modal_q: Vec<K8State>, pub field: Vec<FieldSample>, pub backend: &'static str }

pub fn advance_k8_with_field<B: FieldBackend>(
    current: &[K8State], jacobians: &[f64], dt: f64, source: Option<&[K8State]>,
    backend: &B, points: &[[f64;3]], inside: &[bool]
) -> Result<K8StepOutput, String> {
    let modal_q=advance_k8(current,jacobians,dt,source)?;
    if modal_q.len()!=points.len() { return Err("K=8 模态与场点数量不一致".into()); }
    let field=backend.evaluate(points,inside)?;
    Ok(K8StepOutput { modal_q, field, backend: backend.name() })
}

pub fn close_k8_rheology(q: &[K8State], field: &[FieldSample], dt: f64) -> Result<Vec<K8State>, String> {
    if q.len()!=field.len(){return Err("K=8 流变闭合长度不一致".into());}
    Ok(q.iter().zip(field).map(|(v,f)| { let g=(f.gravity.iter().map(|x|x*x).sum::<f64>()).sqrt().max(1e-12); let shear=(f.hessian6.iter().map(|x|x*x).sum::<f64>()).sqrt(); let mu=(0.1+shear/(g+1e-12)).clamp(0.01,10.0); let mut out=*v; out[1]+=dt*mu; out }).collect())
}

pub fn enforce_k8_mass_level(radii: &[f64], field: &[FieldSample], target_mass: f64, density: f64) -> Result<Vec<f64>, String> {
    if radii.len()!=field.len(){return Err("K=8 质量能级长度不一致".into());}
    let m=radii.iter().map(|r|4.0*std::f64::consts::PI/3.0*density*r.powi(3)/radii.len() as f64).sum::<f64>();
    let scale=(target_mass/m.max(1e-30)).cbrt(); Ok(radii.iter().map(|r|r*scale).collect())
}
