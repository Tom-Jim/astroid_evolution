//! SCF 单一状态对象，保证几何、模态、流变和场结果按同一轮次更新。
use crate::model_calculation::contracts::FieldSample;
use crate::rtfp::{bvh::FlattenedBvh, mesh::Mesh};
use super::mesh_update::update_mesh_and_bvh;
use super::galerkin_k8::advance_k8;
use crate::error_analysis::relative_difference;

#[derive(Clone, Debug)]
pub struct ScfState<const K: usize> {
    pub radii: Vec<f64>,
    pub directions: Vec<[f64; 3]>,
    pub modal_q: Vec<[f64; K]>,
    pub modal_m: Vec<[f64; K]>,
    pub field: Vec<FieldSample>,
    pub iteration: usize,
}

impl<const K: usize> ScfState<K> {
    pub fn validate(&self) -> Result<(), String> {
        let n=self.radii.len();
        if n==0 || self.directions.len()!=n || self.modal_q.len()!=n || self.modal_m.len()!=n { return Err("SCF 状态节点数量不一致".into()); }
        if self.radii.iter().any(|r| !r.is_finite() || *r<=0.0) { return Err("SCF 状态含非法半径".into()); }
        Ok(())
    }

    pub fn commit_geometry(&mut self, mesh: &Mesh) -> Result<(Mesh, FlattenedBvh), String> {
        self.validate()?;
        let (next, bvh) = update_mesh_and_bvh(mesh, &self.directions, &self.radii)?;
        if next.vertex_count() != self.radii.len() { return Err("几何提交后顶点数量发生变化".into()); }
        Ok((next, bvh))
    }

    pub fn field_change(&self, previous: &[FieldSample]) -> Result<f64, String> {
        if previous.len()!=self.field.len() { return Err("场状态长度不一致".into()); }
        let a: Vec<f64> = self.field.iter().flat_map(|s| [s.potential,s.gravity[0],s.gravity[1],s.gravity[2]]).collect();
        let b: Vec<f64> = previous.iter().flat_map(|s| [s.potential,s.gravity[0],s.gravity[1],s.gravity[2]]).collect();
        Ok(relative_difference(&a, &b))
    }

    pub fn advance_modal_k8(&mut self, jacobians: &[f64], dt: f64, source: Option<&[[f64; 8]]>) -> Result<(), String> where [f64; K]: Sized {
        if K != 8 { return Err("advance_modal_k8 仅允许 K=8 状态".into()); }
        let current: Vec<[f64;8]> = self.modal_q.iter().map(|q| { let mut x=[0.0;8]; for i in 0..8 { x[i]=q[i]; } x }).collect();
        let next = advance_k8(&current, jacobians, dt, source)?;
        for (dst, src) in self.modal_q.iter_mut().zip(next) { for i in 0..8 { dst[i]=src[i]; } }
        Ok(())
    }
}

impl ScfState<4> {
    pub fn from_step(radii: Vec<f64>, directions: Vec<[f64;3]>, q: Vec<[f64;4]>, m: Vec<[f64;4]>, field: Vec<FieldSample>, iteration: usize) -> Result<Self, String> {
        let state=Self { radii, directions, modal_q:q, modal_m:m, field, iteration };
        state.validate()?; Ok(state)
    }
}

impl ScfState<8> {
    pub fn from_k8(radii: Vec<f64>, directions: Vec<[f64;3]>, q: Vec<[f64;8]>, m: Vec<[f64;8]>, field: Vec<FieldSample>, iteration: usize) -> Result<Self, String> {
        let state=Self { radii, directions, modal_q:q, modal_m:m, field, iteration };
        state.validate()?; Ok(state)
    }
}
