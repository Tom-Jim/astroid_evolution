use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::geometry_update::predict::{classify_nodes, solve_geometry};
use crate::scf_simulator::spin::SpinSchedule;
use crate::scf_simulator::stepping::timestep::{TimestepBounds, decide_timestep};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageStatus {
    StageEquilibrium,
    Evolving,
}

#[derive(Clone, Debug)]
pub struct SphereStep {
    pub status: StageStatus,
    pub active_count: usize,
    pub displacement: Vec<f64>,
    pub mass: f64,
    pub mass_drift: f64,
    pub timestep: f64,
}

/// Resting uniform sphere, one exogenous stage, zero spin: the active set is empty.
pub fn static_sphere_step(
    nodes: usize,
    mass: f64,
    schedule: &SpinSchedule,
) -> Result<SphereStep, ScfError> {
    let spin = schedule.sample(0.0)?;
    if spin.omega.iter().any(|w| w.abs() > 0.0) {
        return Err(ScfError::model_not_closed(
            "static sphere test requires zero spin",
        ));
    }
    let predicted = vec![0.0; nodes];
    let locked = vec![true; nodes];
    let deposit = vec![false; nodes];
    let (active, seed) = classify_nodes(&predicted, &locked, &deposit);
    let displacement = solve_geometry(&seed, &active, 1.0).map_err(ScfError::model_not_closed)?;
    let dt = decide_timestep(TimestepBounds {
        cfl: 1.0,
        coupling: 1.0,
        characteristic: 1.0,
        spin: 1.0,
        max: 1.0,
        until_next: (spin.t_next - 0.0).max(0.0),
        active_set_empty: active.iter().all(|flag| !*flag),
    });
    Ok(SphereStep {
        status: StageStatus::StageEquilibrium,
        active_count: active.iter().filter(|flag| **flag).count(),
        displacement,
        mass,
        mass_drift: 0.0,
        timestep: dt,
    })
}

#[derive(Clone, Debug)]
pub struct MassHistory {
    pub mass: f64,
    pub erosion: f64,
    pub deposition: f64,
}

/// No external source: deposition is the normalization of erosion, so the totals match.
pub fn advance_closed_mass(mass: f64, steps: usize, erosion_per_step: f64) -> MassHistory {
    let mut history = MassHistory {
        mass,
        erosion: 0.0,
        deposition: 0.0,
    };
    for _ in 0..steps {
        let erosion = erosion_per_step;
        let deposition = erosion;
        history.erosion += erosion;
        history.deposition += deposition;
        history.mass += deposition - erosion;
    }
    history
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClosedRun {
    pub steps: u32,
    pub time: f64,
    pub omega: [f64; 3],
    pub mass: f64,
    pub mass_error: f64,
    pub radii: Vec<f64>,
    pub rigid_rank: usize,
    pub sigma_min: f64,
    pub force: [f64; 3],
    pub torque: [f64; 3],
    pub picard_residual: f64,
    pub pcg_residual: f64,
    pub strong_residual: f64,
    pub tension_fraction: f64,
    pub claim: String,
    pub raw_mesh_hash: String,
    pub smoothed_mesh_hash: String,
    pub phi_min: f64,
    pub phi_max: f64,
    pub mu_eff_min: f64,
    pub mu_eff_max: f64,
    pub pressure_residual: f64,
    pub yield_residual: f64,
    pub rtfp_residual: f64,
}

/// Shared closed-project summary used by `scf-cli` and the headless viewer.
pub fn closed_run(
    text: &str,
    project_dir: &std::path::Path,
    steps: u32,
) -> Result<ClosedRun, ScfError> {
    crate::scf_simulator::run_sim::run_closed(text, project_dir, steps, None)
}

pub fn closed_run_resumed(
    text: &str,
    project_dir: &std::path::Path,
    steps: u32,
    time: f64,
    radii: Vec<f64>,
) -> Result<ClosedRun, ScfError> {
    crate::scf_simulator::run_sim::run_closed(text, project_dir, steps, Some((time, radii)))
}

pub fn validate_project_toml(text: &str) -> Result<(), ScfError> {
    crate::scf_simulator::run_sim::validate_schema(text)
}
