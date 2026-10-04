//! Inputs for the general-mesh Tier2 campaign. The radial chart is not part of this job.
use std::path::PathBuf;

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::scf_simulator::accept::step_checks::ScfTolerances;
use crate::scf_simulator::checkpoint::MeshCheckpoint;
use crate::scf_simulator::geometry_update::star::StarReport;
use crate::scf_simulator::spin::SpinSchedule;

pub struct MeshOptions {
    pub max_steps: Option<u32>,
    pub until_t: Option<f64>,
    pub out_dir: Option<PathBuf>,
    pub resume: Option<MeshCheckpoint>,
    pub stable_geometry_tol: f64,
    pub stable_field_tol: f64,
    pub stable_steps: u32,
    pub checkpoint: Option<PathBuf>,
}

pub struct MeshJob {
    pub raw_hash: String,
    pub mesh: TriMesh,
    pub smoothed_hash: String,
    pub star: StarReport,
    pub star_min_cos: f64,
    pub require_star: bool,
    pub slope: crate::scf_simulator::geometry_update::smooth::SlopeLaw,
    pub crater: crate::scf_simulator::geometry_update::smooth::CraterMask,
    pub initial_volume: f64,
    pub initial_crater: Option<crate::scf_simulator::geometry_update::smooth::CraterMeasure>,
    pub density: CauchyDensity,
    pub density_hash: String,
    pub density_unit: String,
    pub mass_target: f64,
    pub spin: SpinSchedule,
    pub cohesion: f64,
    pub mu_y: f64,
    pub mu_s: f64,
    pub mu_2: f64,
    pub i_0: f64,
    pub grain_diameter: f64,
    pub p_min: f64,
    pub delta_y: f64,
    pub erosion_coefficient: f64,
    pub external_mass_flux: f64,
    pub stress_scale: f64,
    pub l_max: usize,
    pub directions: usize,
    pub radial_degree: usize,
    pub gamma: f64,
    pub g: f64,
    pub shear_floor: f64,
    pub near_radius: f64,
    pub ell0: f64,
    pub under_relaxation: f64,
    pub picard_iterations: usize,
    pub tolerances: ScfTolerances,
    pub trust_radius: f64,
    pub cfl: f64,
    pub coupling: f64,
    pub characteristic: f64,
    pub spin_dt: f64,
    pub dt_max: f64,
    pub raw_mesh: TriMesh,
    pub require_convergence: bool,
    pub rho_bulk: f64,
    pub rho_grain: f64,
}
