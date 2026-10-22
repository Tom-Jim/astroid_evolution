//! TOML project of `rheo-cli`: a closed, self-contained case. Every field is required.

use std::f64::consts::PI;

use serde::Deserialize;
use serde_json::{Value, json};

use super::basis::K;
use super::gpu::{ChainResult, RheoGpu};
use super::model::{Environment, NodeBatch, StarSurface, VelocityField, fibonacci_directions};
use super::params::{Material, MaterialSpec, SolverSpec};
use crate::common::error::SimError;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceSpec {
    /// Mean radius, the `(0,0)` mode divided by `sqrt(4 pi)`.
    mean_radius: f64,
    /// `[l, m, r_lm]` for `l >= 1`.
    modes: Vec<[f64; 3]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VelocitySpec {
    /// `[l, m, k, q]`.
    poloidal: Vec<[f64; 4]>,
    /// `[l, m, k, s]`.
    toroidal: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvironmentSpec {
    /// `g_eff = -k x + omega^2 x_perp`.
    gravity_k: f64,
    omega: f64,
    p_ext: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeSpec {
    count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectFile {
    material: MaterialSpec,
    solver: SolverSpec,
    surface: SurfaceSpec,
    velocity: VelocitySpec,
    environment: EnvironmentSpec,
    nodes: NodeSpec,
}

pub struct Project {
    pub material: Material,
    pub surface: StarSurface,
    pub velocity: VelocityField,
    pub environment: Environment,
    pub node_count: usize,
}

fn integer(name: &str, v: f64) -> Result<i64, SimError> {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1.0e6 {
        Ok(v as i64)
    } else {
        Err(SimError::invalid_input(format!(
            "{name} must be an integer, got {v}"
        )))
    }
}

impl Project {
    pub fn parse(text: &str) -> Result<Self, SimError> {
        let file: ProjectFile = toml::from_str(text)
            .map_err(|e| SimError::model_not_closed(format!("project file: {e}")))?;
        let material = Material::new(file.material, file.solver)?;
        if !(file.surface.mean_radius.is_finite() && file.surface.mean_radius > 0.0) {
            return Err(SimError::geometry("mean_radius must be positive"));
        }
        let mut entries = vec![(0, 0, file.surface.mean_radius * (4.0 * PI).sqrt())];
        for [l, m, v] in &file.surface.modes {
            let l = integer("surface l", *l)?;
            if l < 1 {
                return Err(SimError::invalid_input("surface modes start at l = 1"));
            }
            entries.push((l as usize, integer("surface m", *m)? as i32, *v));
        }
        let surface = StarSurface::from_entries(&entries)?;
        let mut poloidal = Vec::new();
        let mut toroidal = Vec::new();
        for (src, dst) in [
            (&file.velocity.poloidal, &mut poloidal),
            (&file.velocity.toroidal, &mut toroidal),
        ] {
            for [l, m, k, v] in src {
                let k = integer("velocity k", *k)?;
                if !(0..K as i64).contains(&k) {
                    return Err(SimError::invalid_input(format!(
                        "velocity k = {k} outside 0..{K}"
                    )));
                }
                dst.push((
                    integer("velocity l", *l)?.max(0) as usize,
                    integer("velocity m", *m)? as i32,
                    k as usize,
                    *v,
                ));
            }
        }
        let velocity = VelocityField::from_entries(&poloidal, &toroidal)?;
        let e = file.environment;
        if !(e.gravity_k.is_finite() && e.omega.is_finite() && e.p_ext.is_finite()) {
            return Err(SimError::model_not_closed("environment must be finite"));
        }
        if file.nodes.count == 0 {
            return Err(SimError::invalid_input("nodes.count must be positive"));
        }
        Ok(Self {
            material,
            surface,
            velocity,
            environment: Environment {
                gravity_k: e.gravity_k,
                omega: e.omega,
                p_ext: e.p_ext,
            },
            node_count: file.nodes.count,
        })
    }

    pub fn nodes(&self) -> NodeBatch {
        self.environment
            .nodes(&self.surface, fibonacci_directions(self.node_count))
    }

    /// Run the chain; a non-converged or invalid node is an error, never a result.
    pub fn run(&self, gpu: &RheoGpu) -> Result<(NodeBatch, ChainResult), SimError> {
        let nodes = self.nodes();
        let result = gpu.evaluate(&self.material, &self.surface, &self.velocity, &nodes)?;
        if result.stats.invalid > 0 {
            return Err(SimError::geometry(format!(
                "{} nodes have a non-positive radius or degenerate metric",
                result.stats.invalid
            )));
        }
        if result.stats.unconverged > 0 {
            return Err(SimError::rheology_not_converged(format!(
                "{} of {} nodes did not reach the frozen-viscosity fixed point within {} iterations",
                result.stats.unconverged,
                result.nodes.len(),
                self.material.max_iter
            )));
        }
        Ok((nodes, result))
    }
}

fn summary(values: impl Iterator<Item = f64>) -> Value {
    let (mut lo, mut hi, mut sum, mut n) = (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
    for v in values {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v;
        n += 1;
    }
    json!({ "min": lo, "mean": sum / n.max(1) as f64, "max": hi })
}

pub fn report(project: &Project, device: &str, result: &ChainResult) -> Value {
    let n = &result.nodes;
    json!({
        "device": device,
        "nodes": n.len(),
        "mu_proj_mode": project.material.spec.mu_proj_mode,
        "gpu_seconds": result.stats.seconds,
        "max_fixed_point_iterations": result.stats.max_iterations,
        "radius": summary(n.iter().map(|x| x.radius)),
        "shear_rate": summary(n.iter().map(|x| x.shear)),
        "inertial_number": summary(n.iter().map(|x| x.inertial)),
        "mu_d": summary(n.iter().map(|x| x.mu_d)),
        "eta": summary(n.iter().map(|x| x.eta)),
        "mu_eff": summary(n.iter().map(|x| x.mu_eff)),
        "phi_deg": summary(n.iter().map(|x| x.phi.to_degrees())),
        "sin2_phi": summary(n.iter().map(|x| x.sin2_phi)),
        "slope_slack": summary(n.iter().map(|x| x.slack)),
        "overhang_violations": n.iter().filter(|x| x.overhang_margin < 0.0).count(),
        "slack_positive": n.iter().filter(|x| x.slack > 0.0).count(),
        "verification": "f64 reference and acceptance checks are in `rheo-cli check`",
    })
}

pub fn csv(nodes: &NodeBatch, result: &ChainResult) -> String {
    let mut out = String::from(
        "wx,wy,wz,radius,nx,ny,nz,shear,inertial,mu_d,eta,mu_proj,mu_eff,phi,sin2_phi,sin_slope,slack,overhang_margin,d_nn,d_n1,d_n2\n",
    );
    for (d, n) in nodes.dirs.iter().zip(&result.nodes) {
        out.push_str(&format!(
            "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}\n",
            d.x, d.y, d.z, n.radius, n.normal.x, n.normal.y, n.normal.z, n.shear, n.inertial,
            n.mu_d, n.eta, n.mu_proj, n.mu_eff, n.phi, n.sin2_phi, n.sin_slope, n.slack,
            n.overhang_margin, n.d_nn, n.d_na[0], n.d_na[1]
        ));
    }
    out
}
