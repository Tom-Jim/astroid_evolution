//! SCF 主流程编排器：状态、场后端和几何索引必须以同一轮次提交。
use crate::rtfp::field_backend::FieldBackend;
use crate::rtfp::mesh::Mesh;
use crate::rtfp::bvh::FlattenedBvh;
use crate::model_calculation::contracts::FieldSample;
use super::state::ScfState;
use super::mesh_update::update_mesh_and_bvh;

pub struct ScfEngine<B> { pub backend: B, pub mesh: Mesh, pub bvh: FlattenedBvh, pub state: ScfState<4> }

pub struct ScfEngineK8<B> { pub backend: B, pub mesh: Mesh, pub bvh: FlattenedBvh, pub state: ScfState<8> }

impl<B: FieldBackend> ScfEngine<B> {
    pub fn evaluate_field(&mut self, points: &[[f64;3]], inside: &[bool]) -> Result<Vec<FieldSample>, String> {
        let field=self.backend.evaluate(points, inside)?;
        self.state.field=field.clone();
        Ok(field)
    }

    pub fn commit_radii(&mut self, radii: Vec<f64>, iteration: usize) -> Result<(), String> {
        if radii.len()!=self.state.radii.len() { return Err("SCF 几何提交节点数不一致".into()); }
        let (mesh,bvh)=update_mesh_and_bvh(&self.mesh, &self.state.directions, &radii)?;
        self.mesh=mesh; self.bvh=bvh; self.state.radii=radii; self.state.iteration=iteration;
        self.state.validate()
    }

    pub fn run_iterations<F>(&mut self, points: &[[f64;3]], inside: &[bool], max_iterations: usize, mut update: F) -> Result<(), String>
    where F: FnMut(&mut ScfState<4>, &[FieldSample]) -> Result<Vec<f64>, String> {
        if max_iterations==0 { return Err("SCF 最大迭代次数必须大于零".into()); }
        for it in 0..max_iterations {
            let field=self.evaluate_field(points,inside)?;
            let next=update(&mut self.state,&field)?;
            self.commit_radii(next,it+1)?;
        }
        Ok(())
    }
}

impl<B: FieldBackend> ScfEngineK8<B> {
    pub fn evaluate_field(&mut self, points:&[[f64;3]], inside:&[bool])->Result<Vec<FieldSample>,String>{
        let f=self.backend.evaluate(points,inside)?; self.state.field=f.clone(); Ok(f)
    }
    pub fn commit_radii(&mut self, radii:Vec<f64>, iteration:usize)->Result<(),String>{
        if radii.len()!=self.state.radii.len(){return Err("K=8 几何提交节点数不一致".into());}
        let (mesh,bvh)=update_mesh_and_bvh(&self.mesh,&self.state.directions,&radii)?;
        self.mesh=mesh; self.bvh=bvh; self.state.radii=radii; self.state.iteration=iteration; self.state.validate()
    }

    pub fn run_iterations<F>(&mut self, points:&[[f64;3]], inside:&[bool], max_iterations:usize, mut update:F)->Result<(),String>
    where F: FnMut(&mut ScfState<8>, &[FieldSample])->Result<Vec<f64>,String> {
        if max_iterations==0{return Err("K=8 SCF 最大迭代次数必须大于零".into());}
        for it in 0..max_iterations { let field=self.evaluate_field(points,inside)?; let next=update(&mut self.state,&field)?; self.commit_radii(next,it+1)?; }
        Ok(())
    }
}
