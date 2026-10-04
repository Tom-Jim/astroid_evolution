//! Picard closure on a frozen general-mesh geometry.
//! Viscosity is held fixed inside each Stokes solve and is not a single explicit guess.

use crate::common::math::DVec3;
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::rheology::closure::flow_viscosity;
use crate::scf_simulator::rheology::friction_law::mu_dynamic;
use crate::scf_simulator::rheology::inertial_number::modified_inertial_number;
use crate::scf_simulator::rheology::picard::apply_under_relaxation;

use super::pressure::{PressureProblem, PressureSolution, solve_pressure};
use super::stokes::{StokesInput, StokesSolution, solve_stokes, tet_strain};
use super::volume::TetMesh;

#[derive(Clone, Debug)]
pub struct PicardMaterial {
    pub cohesion: f64,
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    pub grain_diameter: f64,
    pub rho_grain: f64,
    pub p_min: f64,
    pub gamma_eps: f64,
    pub relaxation: f64,
    pub iterations: usize,
}

#[derive(Clone, Debug)]
pub struct PicardState {
    pub velocity: Vec<DVec3>,
    pub bubble: Vec<DVec3>,
    pub eta: Vec<f64>,
    pub pressure: PressureSolution,
    pub stokes: StokesSolution,
    pub shear_rate: Vec<f64>,
    pub inertial: Vec<f64>,
    pub mu: Vec<f64>,
    pub residual: f64,
    pub converged: bool,
    pub gamma_eps_used: f64,
}

pub struct PicardLoad<'a> {
    pub density: &'a [f64],
    pub gravity: &'a [DVec3],
    pub tide_divergence: &'a [DVec3],
    pub vertex_mass: &'a [f64],
    pub force: &'a [DVec3],
}

pub fn picard(
    mesh: &TetMesh,
    material: &PicardMaterial,
    load: &PicardLoad<'_>,
) -> Result<PicardState, ScfError> {
    let nt = mesh.tets.len();
    let mut velocity = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut bubble = vec![DVec3::ZERO; nt];
    let mut eta = vec![material.cohesion.max(material.p_min) / material.gamma_eps.max(1e-12); nt];
    let mut last_pressure = None;
    let mut last_stokes = None;
    let mut shear_rate = vec![0.0; nt];
    let mut inertial = vec![0.0; nt];
    let mut mu = vec![material.mu_s; nt];
    let mut residual = f64::INFINITY;
    let iterations = material.iterations.max(1);
    let mut gamma_used = material.gamma_eps;
    for step in 0..iterations {
        let exponent = (iterations - 1 - step) as i32;
        gamma_used = material.gamma_eps * 10f64.powi(exponent.min(3));
        let flow_divergence = if step == 0 {
            vec![DVec3::ZERO; mesh.quad_points.len()]
        } else {
            flow_div(mesh, &velocity, &bubble, &eta)
        };
        let pressure = solve_pressure(
            mesh,
            &PressureProblem {
                density: load.density,
                gravity: load.gravity,
                tide_divergence: load.tide_divergence,
                flow_divergence: &flow_divergence,
                boundary_pressure: &[],
                p_min: material.p_min,
                velocity_is_initial: step == 0,
            },
        )?;
        let tet_pressure = average_pressure(mesh, &pressure.tilde);
        for tet in 0..nt {
            let strain = tet_strain(mesh, &velocity, &bubble, tet);
            let shear = (2.0 * dev_dot(&strain)).sqrt();
            shear_rate[tet] = shear;
            let p_tilde = tet_pressure[tet].max(material.p_min);
            inertial[tet] = modified_inertial_number(
                material.grain_diameter,
                shear,
                p_tilde,
                material.rho_grain.max(material.p_min),
            );
            mu[tet] = mu_dynamic(inertial[tet], material.mu_s, material.mu_2, material.i_0);
            eta[tet] = flow_viscosity(material.cohesion, mu[tet], p_tilde, shear, gamma_used)
                .max(material.p_min);
        }
        let stokes = solve_stokes(
            mesh,
            &StokesInput {
                eta: &eta,
                force: load.force,
                vertex_mass: load.vertex_mass,
                boundary_stress: None,
            },
        )?;
        let flat_old = flatten(&velocity, &bubble);
        let flat_new = flatten(&stokes.vertex_velocity, &stokes.bubble);
        let relaxed = apply_under_relaxation(&flat_old, &flat_new, material.relaxation);
        residual = increment(&flat_old, &relaxed);
        unflatten(&relaxed, &mut velocity, &mut bubble);
        last_pressure = Some(pressure);
        last_stokes = Some(stokes);
    }
    let pressure =
        last_pressure.ok_or_else(|| ScfError::picard_not_converged("Picard did not start"))?;
    let stokes =
        last_stokes.ok_or_else(|| ScfError::picard_not_converged("Stokes did not start"))?;
    let tol = 1.0e-3;
    Ok(PicardState {
        velocity,
        bubble,
        eta,
        pressure,
        stokes,
        shear_rate,
        inertial,
        mu,
        residual,
        converged: residual.is_finite() && residual <= tol,
        gamma_eps_used: gamma_used,
    })
}

fn average_pressure(mesh: &TetMesh, nodal: &[f64]) -> Vec<f64> {
    mesh.tets
        .iter()
        .map(|tet| {
            tet.iter()
                .map(|vertex| nodal[*vertex as usize])
                .sum::<f64>()
                / 4.0
        })
        .collect()
}

fn flow_div(mesh: &TetMesh, velocity: &[DVec3], bubble: &[DVec3], eta: &[f64]) -> Vec<DVec3> {
    let mut nodal = vec![DVec3::ZERO; mesh.vertices.len()];
    let mut weight = vec![0.0; mesh.vertices.len()];
    for tet in 0..mesh.tets.len() {
        let strain = tet_strain(mesh, velocity, bubble, tet);
        let mut tau_div_proxy = DVec3::ZERO;
        tau_div_proxy.x = 2.0 * eta[tet] * (strain[0][0] + strain[0][1] + strain[0][2]);
        tau_div_proxy.y = 2.0 * eta[tet] * (strain[1][0] + strain[1][1] + strain[1][2]);
        tau_div_proxy.z = 2.0 * eta[tet] * (strain[2][0] + strain[2][1] + strain[2][2]);
        for vertex in mesh.tets[tet] {
            nodal[vertex as usize] = nodal[vertex as usize] + tau_div_proxy;
            weight[vertex as usize] += 1.0;
        }
    }
    for index in 0..nodal.len() {
        if weight[index] > 0.0 {
            nodal[index] = nodal[index] * (1.0 / weight[index]);
        }
    }
    mesh.quad_points
        .iter()
        .enumerate()
        .map(|(sample, _)| {
            let tet = mesh.tets[mesh.quad_tet[sample]];
            (nodal[tet[0] as usize]
                + nodal[tet[1] as usize]
                + nodal[tet[2] as usize]
                + nodal[tet[3] as usize])
                * 0.25
        })
        .collect()
}

fn dev_dot(strain: &[[f64; 3]; 3]) -> f64 {
    let trace = (strain[0][0] + strain[1][1] + strain[2][2]) / 3.0;
    let mut sum = 0.0;
    for row in 0..3 {
        for col in 0..3 {
            let value = strain[row][col] - if row == col { trace } else { 0.0 };
            sum += value * value;
        }
    }
    sum
}

fn flatten(velocity: &[DVec3], bubble: &[DVec3]) -> Vec<f64> {
    let mut out = Vec::with_capacity((velocity.len() + bubble.len()) * 3);
    for value in velocity.iter().chain(bubble.iter()) {
        out.push(value.x);
        out.push(value.y);
        out.push(value.z);
    }
    out
}

fn unflatten(flat: &[f64], velocity: &mut [DVec3], bubble: &mut [DVec3]) {
    for (index, slot) in velocity.iter_mut().enumerate() {
        *slot = DVec3::new(flat[3 * index], flat[3 * index + 1], flat[3 * index + 2]);
    }
    let offset = velocity.len() * 3;
    for (index, slot) in bubble.iter_mut().enumerate() {
        *slot = DVec3::new(
            flat[offset + 3 * index],
            flat[offset + 3 * index + 1],
            flat[offset + 3 * index + 2],
        );
    }
}

fn increment(previous: &[f64], next: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (left, right) in previous.iter().zip(next) {
        let diff = left - right;
        num += diff * diff;
        den += right * right;
    }
    num.sqrt() / den.sqrt().max(1e-30)
}
