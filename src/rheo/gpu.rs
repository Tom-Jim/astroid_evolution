//! Driver of the double-single rheology chain (`rheo/shader/chain.wesl`). One
//! thread per node; the mode tables are read uniformly by every lane.

use super::basis::{self, MODE_DS, ds_pair};
use super::model::{NodeBatch, StarSurface, VelocityField, local_frame};
use super::params::{Material, MuProjMode};
use crate::common::error::SimError;
use crate::common::math::DVec3;

/// DS values written per node.
pub const OUT_DS: usize = 48;
const NODE_DS: usize = 19;

#[derive(Clone, Debug)]
pub struct NodeResult {
    pub radius: f64,
    pub normal: DVec3,
    /// `sqrt(det g)`: surface area per unit solid angle.
    pub area_density: f64,
    pub velocity: DVec3,
    pub e1: DVec3,
    pub e2: DVec3,
    /// `g_ab` in the chart basis.
    pub metric: [f64; 3],
    /// `D_ab` in the chart basis `(D11, D12, D22)`.
    pub d_ab: [f64; 3],
    pub d_nn: f64,
    pub d_na: [f64; 2],
    pub shear: f64,
    pub inertial: f64,
    pub mu_d: f64,
    pub eta: f64,
    pub tau_eq: f64,
    pub p_tilde: f64,
    pub p_star: f64,
    pub p_ref: f64,
    pub mu_proj: f64,
    pub mu_eff: f64,
    pub phi: f64,
    pub sin2_phi: f64,
    pub sin_phi: f64,
    pub sin_slope: f64,
    pub slack: f64,
    pub overhang_margin: f64,
    pub tide_frob: f64,
    pub v_r: f64,
    pub v_n: f64,
    pub s0: f64,
    pub tg: f64,
    pub iterations: u32,
    pub converged: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChainStats {
    pub max_iterations: u32,
    pub unconverged: u32,
    pub invalid: u32,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct ChainResult {
    pub nodes: Vec<NodeResult>,
    pub stats: ChainStats,
}

fn ds_value(words: &[f32], i: usize) -> f64 {
    f64::from(words[2 * i]) + f64::from(words[2 * i + 1])
}

fn vec3(words: &[f32], i: usize) -> DVec3 {
    DVec3::new(
        ds_value(words, i),
        ds_value(words, i + 1),
        ds_value(words, i + 2),
    )
}

fn unpack(words: &[f32]) -> NodeResult {
    NodeResult {
        radius: ds_value(words, 0),
        normal: vec3(words, 1),
        area_density: ds_value(words, 4),
        velocity: vec3(words, 5),
        e1: vec3(words, 8),
        e2: vec3(words, 11),
        metric: [
            ds_value(words, 43),
            ds_value(words, 44),
            ds_value(words, 45),
        ],
        d_ab: [
            ds_value(words, 14),
            ds_value(words, 15),
            ds_value(words, 16),
        ],
        d_nn: ds_value(words, 17),
        d_na: [ds_value(words, 18), ds_value(words, 19)],
        shear: ds_value(words, 20),
        inertial: ds_value(words, 21),
        mu_d: ds_value(words, 22),
        eta: ds_value(words, 23),
        tau_eq: ds_value(words, 24),
        p_tilde: ds_value(words, 25),
        p_star: ds_value(words, 26),
        p_ref: ds_value(words, 27),
        mu_proj: ds_value(words, 28),
        mu_eff: ds_value(words, 29),
        phi: ds_value(words, 30),
        sin2_phi: ds_value(words, 31),
        sin_slope: ds_value(words, 32),
        slack: ds_value(words, 33),
        overhang_margin: ds_value(words, 34),
        tide_frob: ds_value(words, 35),
        v_r: ds_value(words, 36),
        v_n: ds_value(words, 37),
        s0: ds_value(words, 38),
        tg: ds_value(words, 39),
        iterations: ds_value(words, 40) as u32,
        converged: ds_value(words, 41) > 0.5,
        sin_phi: ds_value(words, 42),
    }
}

fn pack_nodes(batch: &NodeBatch) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch.len() * NODE_DS * 2);
    let mut push = |x: f64| out.extend_from_slice(&ds_pair(x));
    for i in 0..batch.len() {
        let w = batch.dirs[i]
            .try_normalize()
            .expect("directions validated as unit vectors");
        let (t1, t2) = local_frame(w);
        for v in [w, t1, t2, batch.gravity[i]] {
            push(v.x);
            push(v.y);
            push(v.z);
        }
        for h in batch.hessian_w[i] {
            push(h);
        }
        push(batch.pressure[i]);
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    node_count: u32,
    l_max: u32,
    max_iter: u32,
    proj_dynamic: u32,
    pad0: u32,
    pad1: u32,
    d_p: [f32; 2],
    rho_p: [f32; 2],
    mu_s: [f32; 2],
    mu_2: [f32; 2],
    i0: [f32; 2],
    coh: [f32; 2],
    p_c: [f32; 2],
    gamma_eps: [f32; 2],
    p_min: [f32; 2],
    rho_bulk: [f32; 2],
    h_s: [f32; 2],
    tide_coef: [f32; 2],
    mu_y: [f32; 2],
    tol: [f32; 2],
    eps_overhang: [f32; 2],
}

fn gpu_params(m: &Material, nodes: usize, l_max: usize) -> GpuParams {
    let s = &m.spec;
    GpuParams {
        node_count: nodes as u32,
        l_max: l_max as u32,
        max_iter: m.max_iter,
        proj_dynamic: u32::from(m.mode == MuProjMode::Dynamic),
        pad0: 0,
        pad1: 0,
        d_p: ds_pair(s.d_p),
        rho_p: ds_pair(s.rho_p),
        mu_s: ds_pair(s.mu_s),
        mu_2: ds_pair(s.mu_2),
        i0: ds_pair(s.i_0),
        coh: ds_pair(s.cohesion_surface),
        p_c: ds_pair(m.p_c),
        gamma_eps: ds_pair(s.gamma_eps),
        p_min: ds_pair(s.p_min),
        rho_bulk: ds_pair(m.rho_bulk),
        h_s: ds_pair(s.h_s),
        tide_coef: ds_pair(m.tide_coef),
        mu_y: ds_pair(m.mu_y),
        tol: ds_pair(m.tol),
        eps_overhang: ds_pair(s.eps_overhang),
    }
}

/// Host-side tables for one evaluation.
struct Tables {
    modes: Vec<f32>,
    recur: Vec<f32>,
    l_max: usize,
}

fn tables(surface: &StarSurface, velocity: &VelocityField) -> Result<Tables, SimError> {
    let l_max = surface.l_max.max(velocity.l_max);
    let rows = basis::mode_sums(surface, velocity, l_max)?;
    debug_assert_eq!(rows.first().map(|r| r.len()), Some(MODE_DS));
    Ok(Tables {
        modes: basis::mode_table(&rows),
        recur: basis::recurrence_table(l_max),
        l_max,
    })
}

/// Handle to the GPU chain (`rheo/shader/chain.wesl`).
pub struct RheoGpu;

impl RheoGpu {
    pub fn new() -> Result<Self, SimError> {
        crate::gpu::gpu().map_err(SimError::gpu)?;
        Ok(Self)
    }

    pub fn device_name(&self) -> String {
        crate::gpu::gpu()
            .map(|g| g.adapter.clone())
            .unwrap_or_default()
    }

    /// Run the whole chain on the nodes of `batch`.
    pub fn evaluate(
        &self,
        material: &Material,
        surface: &StarSurface,
        velocity: &VelocityField,
        batch: &NodeBatch,
    ) -> Result<ChainResult, SimError> {
        batch.validate()?;
        if batch.is_empty() {
            return Ok(ChainResult {
                nodes: Vec::new(),
                stats: ChainStats::default(),
            });
        }
        let t = {
            let _span = crate::scf::timing::span("rheology_tables");
            tables(surface, velocity)?
        };
        let packed = pack_nodes(batch);
        let params = gpu_params(material, batch.len(), t.l_max);
        let started = crate::common::time::wall_seconds();
        let k = &crate::gpu::kernels::RHEO_CHAIN;
        let mut job = crate::gpu::Job::new();
        let nodes = job.input(&packed);
        let params = job.value(&params);
        let modes = job.input(&t.modes);
        let recur = job.input(&t.recur);
        let out = job.zeros::<f32>(batch.len() * OUT_DS * 2);
        let stats = job.zeros::<u32>(3);
        job.run(
            k,
            k.groups(batch.len()),
            &[nodes, params, modes, recur, out, stats],
        );
        let ro = job.read_all::<f32>(out, batch.len() * OUT_DS * 2);
        let rs = job.read_all::<u32>(stats, 3);
        let back = job.submit().map_err(SimError::gpu)?;
        let words = back.get::<f32>(ro);
        let raw = back.get::<u32>(rs);
        let nodes = words.chunks_exact(OUT_DS * 2).map(unpack).collect();
        Ok(ChainResult {
            nodes,
            stats: ChainStats {
                max_iterations: raw[0],
                unconverged: raw[1],
                invalid: raw[2],
                seconds: crate::common::time::wall_seconds() - started,
            },
        })
    }
}
