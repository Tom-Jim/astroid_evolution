//! Closed quasi-static step from an explicit project file.
//!
//! The order is the production chain: RTFP, the four sample roles, Poisson
//! pressure, Stokes–Picard, yield, rigid core and slip, mode C, the timestep,
//! the Clarabel trust projection, outer acceptance, realignment, then a second
//! field evaluation on the accepted geometry. Every numeric input is read from
//! the project. A missing field is `ModelNotClosed` (10).

use std::path::Path;

use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::api::{FieldMask, PointRole, PointSet};
use crate::rtfp_solver::{RtfpOptions, bind_and_eval};
use crate::scf_simulator::accept::step_checks::{
    ScfResiduals, ScfTolerances, check_scf_convergence, compute_geometry_residual,
    compute_mass_residual, compute_slope_residual, compute_transport_residual, residual_within,
};
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::geometry_update::accept::{
    audit_radial_mesh, finite_gravitational_field, vertex_areas, vertex_unit_normals,
};
use crate::scf_simulator::geometry_update::mass::{
    evaluate_mass_at_radii, solid_angle_weights, surface_mass_jacobian,
};
use crate::scf_simulator::geometry_update::predict::{partition_nodes, predicted_radial_increment};
use crate::scf_simulator::geometry_update::realign::realign_body;
use crate::scf_simulator::geometry_update::socp::{
    FrozenIncrement, OuterVerdict, normalized_normal_jacobian, project_trusted_increment,
};
use crate::scf_simulator::quadrature::classify_grid::classify_shells;
use crate::scf_simulator::quadrature::volume_points::volume_points;
use crate::rheo_friction::closure::{
    equivalent_flow_stress, flow_stress, flow_viscosity, inertial_number, mu_of_inertial,
};
use crate::scf_simulator::spin::SpinSchedule;
use crate::scf_simulator::stepping::timestep::{TimestepBounds, decide_timestep};
use crate::scf_simulator::stokes::picard_loop::{GravSample, PicardMaterial, picard_solve};
use crate::scf_simulator::stokes::rigid::assemble_rigid_constraint;
use crate::scf_simulator::yielding::core_fit::{fit_core_motion, slip_velocity};
use crate::scf_simulator::yielding::core_graph::rigid_core;
use crate::scf_simulator::yielding::yield_fn::{classic_yield_function, is_yielded};

use super::simulation::ClosedRun;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Representation {
    RadialStar,
    GeneralMesh,
}

struct Project {
    raw_mesh: TriMesh,
    mesh: TriMesh,
    density: CauchyDensity,
    mass_target: f64,
    spin: SpinSchedule,
    cohesion: f64,
    mu_y: f64,
    #[allow(dead_code)]
    eta_c: f64,
    p_min: f64,
    delta_y: f64,
    erosion_coefficient: f64,
    external_mass_flux: f64,
    stress_scale: f64,
    l_max: usize,
    directions: usize,
    radial_degree: usize,
    gamma: f64,
    g: f64,
    shear_floor: f64,
    near_radius: f64,
    ell0: f64,
    under_relaxation: f64,
    picard_iterations: usize,
    require_convergence: bool,
    tolerances: ScfTolerances,
    trust_radius: f64,
    cfl: f64,
    coupling: f64,
    characteristic: f64,
    spin_dt: f64,
    dt_max: f64,
    representation: Representation,
    #[allow(dead_code)]
    require_star: bool,
    star_min_cos: f64,
    slope: crate::scf_simulator::geometry_update::smooth::SlopeRequest,
    crater: crate::scf_simulator::geometry_update::smooth::CraterMask,
    #[allow(dead_code)]
    initial_volume: f64,
    #[allow(dead_code)]
    initial_crater: Option<crate::scf_simulator::geometry_update::smooth::CraterMeasure>,
    raw_hash: String,
    smoothed_hash: String,
    #[allow(dead_code)]
    star: crate::scf_simulator::geometry_update::star::StarReport,
    vertex_records: Vec<crate::scf_simulator::geometry_update::smooth::VertexRecord>,
    #[allow(dead_code)]
    density_hash: String,
    #[allow(dead_code)]
    density_unit: String,
    mu_s: f64,
    mu_2: f64,
    i_0: f64,
    grain_diameter: f64,
}

pub fn spin_at(
    text: &str,
    project_dir: &Path,
    time: f64,
) -> Result<crate::scf_simulator::spin::SpinSample, ScfError> {
    let project = parse_project(text, project_dir, false)?;
    project.spin.sample(time)
}

pub fn validate_schema(text: &str) -> Result<(), ScfError> {
    parse_project(text, Path::new("."), false).map(|_| ())
}

/// Schema plus geometry I/O. `scf-cli validate` and `scf-cli run-sim` both enter here.
pub fn validate_at(text: &str, project_dir: &Path) -> Result<(), ScfError> {
    closed_project(text, project_dir).map(|_| ())
}

pub fn historical_review(
    text: &str,
    project_dir: &Path,
) -> Result<crate::scf_simulator::history_review::HistoryReport, ScfError> {
    let project = closed_project(text, project_dir)?;
    let value: toml::Value =
        toml::from_str(text).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    let density_table = section(&value, "density")?;
    let material = section(&value, "material")?;
    let bulk = sourced_record(density_table, "bulk_density", "kg/m^3")?;
    let porosity = sourced_record(density_table, "porosity", "1")?;
    let mu = sourced_record(material, "mu_s", "1")?;
    let cohesion = sourced_record(material, "cohesion", "Pa")?;
    let sample = project.spin.sample(0.0)?;
    let mask = &project.crater;
    crate::scf_simulator::history_review::review(
        &crate::scf_simulator::history_review::ReviewRequest {
            raw: &project.raw_mesh,
            smoothed: &project.mesh,
            density: &project.density,
            mass_target: project.mass_target,
            mass_tolerance: project.tolerances.mass_tol,
            bulk_density: bulk.value,
            bulk_uncertainty: bulk.uncertainty,
            porosity: porosity.value,
            porosity_uncertainty: porosity.uncertainty,
            grain_diameter: project.slope.law.grain_diameter.value,
            grain_uncertainty: project.slope.law.grain_diameter.uncertainty,
            mu_s: project.mu_s,
            mu_uncertainty: mu.uncertainty,
            mu_2: project.mu_2,
            cohesion: project.cohesion,
            cohesion_uncertainty: cohesion.uncertainty,
            spin_rate: project.slope.law.spin_rate.value,
            spin_uncertainty: project.slope.law.spin_rate.uncertainty,
            timestep: project.slope.law.timestep.value,
            timestep_uncertainty: project.slope.law.timestep.uncertainty,
            calibration: project.slope.law.calibration.value,
            calibration_uncertainty: project.slope.law.calibration.uncertainty,
            nominal_bulk: bulk.value,
            omega: sample.omega,
            gravitational_constant: project.g,
            steps_executed: 0,
            mask,
            vertices: &project.vertex_records,
            coefficients: project_coefficients(&value, &project.slope.law)?,
            field: crate::rtfp_solver::RtfpOptions {
                directions: project.directions.max(8),
                r0: project.near_radius,
                ell0: project.ell0,
                g: project.g,
                quadrature_nodes: project.radial_degree.max(2),
            },
            parameter_source: project.slope.law.calibration.source.clone(),
        },
    )
}

fn closed_project(text: &str, project_dir: &Path) -> Result<Project, ScfError> {
    parse_project(text, project_dir, true)
}

pub fn run_closed(
    text: &str,
    project_dir: &Path,
    steps: u32,
    resume: Option<(f64, Vec<f64>)>,
) -> Result<ClosedRun, ScfError> {
    run_with(
        text,
        project_dir,
        Some(steps),
        None,
        None,
        resume,
        None,
        None,
        None,
        None,
        0,
    )
}

pub fn run_with(
    text: &str,
    project_dir: &Path,
    max_steps: Option<u32>,
    until_t: Option<f64>,
    out_dir: Option<&Path>,
    resume: Option<(f64, Vec<f64>)>,
    resume_mesh: Option<crate::scf_simulator::checkpoint::MeshCheckpoint>,
    checkpoint: Option<&Path>,
    stable_geometry_tol: Option<f64>,
    stable_field_tol: Option<f64>,
    stable_steps: u32,
) -> Result<ClosedRun, ScfError> {
    let project = closed_project(text, project_dir)?;
    if project.representation == Representation::GeneralMesh {
        return dispatch_general_mesh(
            text,
            &project,
            max_steps,
            until_t,
            out_dir,
            resume_mesh,
            checkpoint,
            stable_geometry_tol,
            stable_field_tol,
            stable_steps,
        );
    }
    admit_radial_star(&project.mesh, &project.density, project.star_min_cos)?;
    let steps = max_steps.unwrap_or(if until_t.is_some() { u32::MAX } else { 1 });
    let mut radii: Vec<f64> = project.mesh.vertices.iter().map(|v| v.length()).collect();
    let mut time = 0.0;
    if let Some((restored_time, restored_radii)) = resume {
        if !restored_time.is_finite() || restored_time < 0.0 || restored_radii.len() != radii.len()
        {
            return Err(ScfError::model_not_closed(
                "checkpoint time or radius count does not match the mesh",
            ));
        }
        if restored_radii
            .iter()
            .any(|radius| !radius.is_finite() || *radius <= 0.0)
        {
            return Err(ScfError::model_not_closed(
                "checkpoint radius is not finite and positive",
            ));
        }
        time = restored_time;
        radii = restored_radii;
    }
    let directions: Vec<DVec3> = project
        .mesh
        .vertices
        .iter()
        .map(|v| {
            v.try_normalize()
                .ok_or_else(|| ScfError::model_not_closed("a mesh vertex lies at the origin"))
        })
        .collect::<Result<_, _>>()?;
    let faces = project.mesh.faces.clone();
    let mut density = project.density.clone();
    let mut omega = [0.0; 3];
    let mut mass = 0.0;
    let mut mass_error = 0.0;
    let mut rigid_rank = 0;
    let mut sigma_min = 0.0;
    let mut force = [0.0; 3];
    let mut torque = [0.0; 3];
    let mut velocity_coeff = Vec::new();
    let mut picard_residual = 0.0;
    let mut pcg_residual = 0.0;
    let mut strong_residual = 0.0;
    let mut tension_fraction = 0.0;
    let mut phi_min = f64::NAN;
    let mut phi_max = f64::NAN;
    let mut mu_min = f64::NAN;
    let mut mu_max = f64::NAN;
    let mut yield_residual = f64::NAN;
    let mut rtfp_residual = f64::NAN;
    let evaluations = steps.max(1);
    for step in 0..evaluations {
        if until_t.is_some_and(|end| time >= end) {
            break;
        }
        let advance = step < steps;
        let sample = project.spin.sample(time)?;
        omega = sample.omega;
        if let Err(detail) = audit_radial_mesh(&directions, &radii, &faces) {
            return Err(ScfError::model_not_closed(detail));
        }
        let mesh = mesh_from_radii(&directions, &radii, &faces)?;
        let forward = evaluate_fields(&project, &density, &mesh, sample.omega)?;
        let gravity: Vec<GravSample> = forward
            .volume_position
            .iter()
            .zip(&forward.volume_g)
            .map(|(position, g_eff)| GravSample {
                position: *position,
                g_eff: *g_eff,
            })
            .collect();
        let mechanics = picard_solve(
            project.l_max,
            4,
            &mesh,
            &density,
            &gravity,
            PicardMaterial {
                cohesion: project.cohesion,
                mu: project.mu_y,
                gamma_eps: project.shear_floor,
                p_min: project.p_min,
                gravity: project.g,
                tide_coefficient: 0.0,
                omega_z: sample.omega[2],
                relaxation: project.under_relaxation,
                aug_lagrange: project.gamma,
            },
            project.picard_iterations,
            velocity_coeff,
        )?;
        velocity_coeff = mechanics.coefficients.clone();
        picard_residual = mechanics.picard_residual;
        if project.require_convergence && !residual_within(picard_residual, 1.0e-3) {
            return Err(ScfError::rheology_not_converged(format!(
                "picard_residual {picard_residual}"
            )));
        }
        pcg_residual = mechanics.pcg_residual;
        strong_residual = mechanics.strong_residual;
        tension_fraction = mechanics.tension_fraction;
        let rigid_samples: Vec<([f64; 3], [f64; 3], f64)> = forward
            .volume_position
            .iter()
            .zip(&forward.volume_g)
            .zip(&forward.volume_weight)
            .map(|((position, g_eff), weight)| (*position, *g_eff, *weight))
            .collect();
        let rigid = assemble_rigid_constraint(project.l_max, &mesh, &density, &rigid_samples)?;
        rigid_rank = rigid.rank;
        sigma_min = rigid.sigma_min;
        force = rigid.force;
        torque = rigid.torque;
        eprintln!(
            "rank(B_rb) {} sigma_min(B_rb) {} force {} {} {} torque {} {} {}",
            rigid.rank,
            rigid.sigma_min,
            rigid.force[0],
            rigid.force[1],
            rigid.force[2],
            rigid.torque[0],
            rigid.torque[1],
            rigid.torque[2]
        );
        let friction = nodal_phi(&project, &mesh, &density, &mechanics)?;
        let (yielded, step_yield) = yield_flags(&project, &friction, &mechanics)?;
        phi_min = friction.phi.iter().copied().fold(f64::INFINITY, f64::min);
        phi_max = friction
            .phi
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        mu_min = friction.mu.iter().copied().fold(f64::INFINITY, f64::min);
        mu_max = friction
            .mu
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        yield_residual = step_yield;
        let neighbors = vertex_neighbors(&faces, radii.len());
        let node_mass = vertex_masses(&mesh, &density)?;
        let core = rigid_core(&yielded, &neighbors, &node_mass)?;
        let positions: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| v.to_array()).collect();
        let slip = relative_slip(&positions, &mechanics.vertex_velocity, &core)?;
        let (erosion, deposition) = mode_c(&project, &yielded, &slip, &node_mass);
        let active_empty = !yielded.iter().any(|flag| *flag);
        let dt = decide_timestep(TimestepBounds {
            cfl: project.cfl,
            coupling: project.coupling,
            characteristic: project.characteristic,
            spin: project.spin_dt,
            max: project.dt_max,
            until_next: (sample.t_next - time).max(0.0),
            active_set_empty: active_empty,
        });
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ScfError::model_not_closed(
                "timestep is not a positive finite number",
            ));
        }
        let net_flux: Vec<f64> = deposition
            .iter()
            .zip(&erosion)
            .map(|(inflow, outflow)| inflow - outflow)
            .collect();
        let (locked, deposit, movable) =
            partition_nodes(&core, &net_flux).map_err(ScfError::model_not_closed)?;
        let areas = vertex_areas(&mesh);
        let normals = vertex_unit_normals(&mesh).map_err(ScfError::model_not_closed)?;
        let angle_weights =
            solid_angle_weights(&mesh.vertices, &faces).map_err(ScfError::model_not_closed)?;
        let mut d_pred = Vec::with_capacity(radii.len());
        let mut mass_jacobian = Vec::with_capacity(radii.len());
        for i in 0..radii.len() {
            let position = directions[i] * radii[i];
            let rho_k = density.rho(position);
            if !(rho_k.is_finite() && rho_k > 0.0) {
                return Err(ScfError::model_not_closed(
                    "rho_K on the surface is not positive",
                ));
            }
            mass_jacobian.push(surface_mass_jacobian(angle_weights[i], rho_k, radii[i]));
            d_pred.push(
                predicted_radial_increment(
                    movable[i],
                    deposit[i],
                    DVec3::from_array(normals[i]),
                    directions[i],
                    dt,
                    DVec3::from_array(slip[i]),
                    net_flux[i],
                    areas[i],
                    if density.bulk > 0.0 {
                        density.bulk
                    } else {
                        rho_k
                    },
                    rho_k,
                )
                .map_err(ScfError::model_not_closed)?,
            );
        }
        let direction_arrays: Vec<[f64; 3]> = directions.iter().map(|dir| dir.to_array()).collect();
        let ray_mass =
            evaluate_mass_at_radii(&radii, &direction_arrays, &angle_weights, &|point| {
                density.rho(DVec3::from_array(point))
            });
        let problem = FrozenIncrement {
            d_pred: d_pred.clone(),
            locked: locked.clone(),
            deposit: deposit.clone(),
            objective_weight: angle_weights,
            mass_jacobian,
            current_mass: ray_mass,
            target_mass: project.mass_target,
            volume_jacobian: None,
            current_volume: mesh.signed_volume(),
            target_volume: mesh.signed_volume(),
            gravity: forward.vertex_g.clone(),
            normals,
            normal_jacobian: normalized_normal_jacobian(&mesh, &directions, &radii),
            phi: friction.phi.clone(),
        };
        let projection = project_trusted_increment(
            &problem,
            &radii,
            project.trust_radius,
            |candidate, solution| {
                if !solution.kkt_residual.is_finite() {
                    return OuterVerdict::Reject("KKT residual is not finite".into());
                }
                if let Err(detail) = audit_radial_mesh(&directions, candidate, &faces) {
                    return OuterVerdict::Reject(detail);
                }
                let candidate_mesh = match mesh_from_radii(&directions, candidate, &faces) {
                    Ok(mesh) => mesh,
                    Err(err) => return OuterVerdict::Reject(err.detail),
                };
                if let Err(err) = admit_radial_star(&candidate_mesh, &density, project.star_min_cos)
                {
                    return OuterVerdict::Reject(err.detail);
                }
                let mass = if candidate
                    .iter()
                    .zip(&radii)
                    .all(|(next, current)| next.to_bits() == current.to_bits())
                {
                    forward.mass
                } else {
                    match evaluate_fields(&project, &density, &candidate_mesh, sample.omega) {
                        Ok(field) => field.mass,
                        Err(err) => return OuterVerdict::Abort(err.detail),
                    }
                };
                if residual_within(
                    compute_mass_residual(mass, project.mass_target),
                    project.tolerances.mass_tol,
                ) {
                    OuterVerdict::Accept
                } else {
                    OuterVerdict::Reject("nonlinear mass acceptance failed".into())
                }
            },
        )?;
        for i in 0..radii.len() {
            if locked[i] && projection.solution.displacement[i].to_bits() != 0.0_f64.to_bits() {
                return Err(ScfError::model_not_closed("a locked node left d = 0"));
            }
            if deposit[i] && projection.solution.displacement[i].to_bits() != d_pred[i].to_bits() {
                return Err(ScfError::model_not_closed("a deposit node left d_pred"));
            }
        }
        let next_radii = projection.radii;
        let kkt_residual = projection.solution.kkt_residual;
        let candidate = mesh_from_radii(&directions, &next_radii, &faces)?;
        let frame = realign_body(&candidate, &directions, &density, 1.0e-6, 1.0e-8)?;
        let realigned_radii = if frame.applied {
            density = frame.density;
            frame.radii
        } else {
            next_radii
        };
        let accepted_mesh = mesh_from_radii(&directions, &realigned_radii, &faces)?;
        admit_radial_star(&accepted_mesh, &density, project.star_min_cos)?;
        let final_field = evaluate_fields(&project, &density, &accepted_mesh, sample.omega)?;
        audit_gravitational_field(&project, &density, &accepted_mesh)?;
        let geometry_residual = compute_geometry_residual(&realigned_radii, &radii);
        let mass_residual = compute_mass_residual(final_field.mass, project.mass_target);
        let slope_residual = slope_residual(&accepted_mesh, &friction.phi, &final_field);
        let transport_residual =
            compute_transport_residual(&surface_modes(&deposition), &surface_modes(&erosion));
        rtfp_residual = finite_rtfp_residual(forward.rtfp_residual, final_field.rtfp_residual)?;
        eprintln!(
            "model radial_star tier1_initial 1 tier1_candidate 1 U {} gradient {} face_hessian {} volume_gradient {} volume_hessian {} rtfp_residual {:.6e} pressure_residual {:.6e} yield_residual {:.6e} phi_min {:.6e} phi_max {:.6e} kkt_residual {:.6e}",
            forward.potential_samples,
            forward.gradient_samples,
            forward.face_hessian_samples,
            forward.volume_gradient_samples,
            forward.volume_hessian_samples,
            rtfp_residual,
            strong_residual,
            yield_residual,
            phi_min,
            phi_max,
            kkt_residual
        );
        let residuals = ScfResiduals {
            geometry_residual,
            mass_residual,
            slope_residual,
            transport_residual,
            kkt_residual,
        };
        let (ok, reasons) = check_scf_convergence(&residuals, &project.tolerances);
        if !ok {
            return Err(ScfError::model_not_closed(reasons.join("; ")));
        }
        mass = final_field.mass;
        mass_error = mass_residual;
        if advance {
            radii = realigned_radii;
            time += dt;
            if frame.applied {
                velocity_coeff.clear();
            }
        }
    }
    if !rtfp_residual.is_finite() {
        return Err(ScfError::model_not_closed(
            "rtfp_residual is not finite; stability was not assessed",
        ));
    }
    Ok(ClosedRun {
        steps,
        time,
        omega,
        mass,
        mass_error,
        rigid_rank,
        sigma_min,
        force,
        torque,
        picard_residual,
        pcg_residual,
        strong_residual,
        tension_fraction,
        radii,
        claim: "step_accepted".into(),
        raw_mesh_hash: project.raw_hash.clone(),
        smoothed_mesh_hash: project.smoothed_hash.clone(),
        phi_min,
        phi_max,
        mu_eff_min: mu_min,
        mu_eff_max: mu_max,
        pressure_residual: strong_residual,
        yield_residual,
        rtfp_residual,
    })
}

struct Forward {
    mass: f64,
    rtfp_residual: f64,
    potential_samples: usize,
    gradient_samples: usize,
    face_hessian_samples: usize,
    volume_gradient_samples: usize,
    volume_hessian_samples: usize,
    volume_position: Vec<[f64; 3]>,
    volume_weight: Vec<f64>,
    #[allow(dead_code)]
    volume_rho: Vec<f64>,
    volume_g: Vec<[f64; 3]>,
    vertex_g: Vec<[f64; 3]>,
    surface_g: Vec<[f64; 3]>,
    surface_normal: Vec<[f64; 3]>,
}

fn evaluate_fields(
    project: &Project,
    density: &CauchyDensity,
    mesh: &TriMesh,
    omega: [f64; 3],
) -> Result<Forward, ScfError> {
    let options = RtfpOptions {
        directions: project.directions,
        r0: project.near_radius,
        ell0: project.ell0,
        g: project.g,
        quadrature_nodes: project.radial_degree.max(2),
    };
    let r_scale = mesh
        .vertices
        .iter()
        .map(|v| v.length())
        .fold(0.0_f64, f64::max);
    if !r_scale.is_finite() || r_scale <= 0.0 {
        return Err(ScfError::model_not_closed("mesh radius is not positive"));
    }
    let unit = volume_points(project.directions, project.radial_degree);
    let volume: Vec<DVec3> = unit
        .iter()
        .map(|node| {
            DVec3::new(
                node.position[0] * r_scale,
                node.position[1] * r_scale,
                node.position[2] * r_scale,
            )
        })
        .collect();
    let shells = classify_shells(project.radial_degree);
    let mut classify = Vec::new();
    for &zeta in &shells {
        if zeta <= 0.0 {
            continue;
        }
        classify.push(DVec3::new(r_scale * zeta, 0.0, 0.0));
        classify.push(DVec3::new(0.0, r_scale * zeta, 0.0));
        classify.push(DVec3::new(0.0, 0.0, r_scale * zeta));
    }
    let faces: Vec<DVec3> = (0..mesh.faces.len())
        .map(|f| {
            let [a, b, c] = mesh.triangle(f);
            (a + b + c) * (1.0 / 3.0)
        })
        .collect();
    let vertices = mesh.vertices.clone();
    let field = FieldMask {
        potential: true,
        gradient: true,
        hessian: true,
    };
    let volume_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &volume,
        PointRole::VolumeQuad,
        field,
    )?;
    let _classify_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &classify,
        PointRole::ClassifyGrid,
        FieldMask::ALL,
    )?;
    let face_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &faces,
        PointRole::SurfaceFace,
        field,
    )?;
    let vertex_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &vertices,
        PointRole::SurfaceVertex,
        FieldMask {
            potential: true,
            gradient: true,
            hessian: false,
        },
    )?;
    let mut mass = 0.0;
    let mut volume_position = Vec::new();
    let mut volume_weight = Vec::new();
    let mut volume_rho = Vec::new();
    let mut volume_g = Vec::new();
    let mut rtfp_residual = 0.0_f64;
    let mut volume_hessian_samples = 0usize;
    for (i, node) in unit.iter().enumerate() {
        let x = volume[i];
        if volume_field.theta[i] < 0.5 {
            continue;
        }
        let hessian = volume_field
            .hessian
            .get(i)
            .copied()
            .ok_or_else(|| ScfError::model_not_closed("Tier1 volume Hessian is missing"))?;
        if hessian.iter().any(|value| !value.is_finite())
            || !volume_field.gradient[i].is_finite()
            || !volume_field.potential[i].is_finite()
        {
            return Err(ScfError::model_not_closed(
                "Tier1 volume potential, gradient, or Hessian is not finite",
            ));
        }
        let rho = density.rho(x);
        if !(rho.is_finite() && rho >= 0.0) {
            return Err(ScfError::model_not_closed(
                "density on a volume sample is not finite and non-negative",
            ));
        }
        let trace = hessian[0] + hessian[1] + hessian[2];
        let poisson = trace + 4.0 * std::f64::consts::PI * project.g * volume_field.theta[i] * rho;
        let scale = (4.0 * std::f64::consts::PI * project.g * rho)
            .abs()
            .max(trace.abs())
            .max(1e-30);
        let relative = poisson.abs() / scale;
        if !relative.is_finite() {
            return Err(ScfError::model_not_closed(
                "RTFP Poisson residual is not finite",
            ));
        }
        rtfp_residual = rtfp_residual.max(relative);
        volume_hessian_samples += 1;
        let weight = rho * node.weight * r_scale * r_scale * r_scale;
        mass += weight;
        let g_eff = centrifugal(volume_field.gradient[i], x, omega);
        volume_position.push([x.x, x.y, x.z]);
        volume_weight.push(weight);
        volume_rho.push(rho);
        volume_g.push([g_eff.x, g_eff.y, g_eff.z]);
    }
    if volume_hessian_samples == 0 {
        return Err(ScfError::model_not_closed(
            "Tier1 volume gradient/Hessian has no interior sample",
        ));
    }
    let mut vertex_g = Vec::new();
    for (i, point) in vertices.iter().enumerate() {
        if !vertex_field.potential[i].is_finite() || !vertex_field.gradient[i].is_finite() {
            return Err(ScfError::model_not_closed(
                "Tier1 vertex potential or gradient is not finite",
            ));
        }
        let g_eff = centrifugal(vertex_field.gradient[i], *point, omega);
        vertex_g.push([g_eff.x, g_eff.y, g_eff.z]);
    }
    if mass <= 0.0 || !mass.is_finite() {
        return Err(ScfError::model_not_closed(
            "RTFP interior mass is not positive",
        ));
    }
    let mut surface_g = Vec::new();
    let mut surface_normal = Vec::new();
    for (f, point) in faces.iter().enumerate() {
        let hessian =
            face_field.hessian.get(f).copied().ok_or_else(|| {
                ScfError::model_not_closed("Tier1 surface-face Hessian is missing")
            })?;
        if !face_field.potential[f].is_finite()
            || !face_field.gradient[f].is_finite()
            || hessian.iter().any(|value| !value.is_finite())
        {
            return Err(ScfError::model_not_closed(
                "Tier1 surface-face potential, gradient, or Hessian is not finite",
            ));
        }
        let g_eff = centrifugal(face_field.gradient[f], *point, omega);
        let normal = mesh
            .face_normal(f)
            .ok_or_else(|| ScfError::model_not_closed("a mesh face is degenerate"))?;
        surface_g.push([g_eff.x, g_eff.y, g_eff.z]);
        surface_normal.push([normal.x, normal.y, normal.z]);
    }
    Ok(Forward {
        mass,
        rtfp_residual,
        potential_samples: vertex_field.potential.len(),
        gradient_samples: vertex_field.gradient.len() + face_field.gradient.len() + volume_g.len(),
        face_hessian_samples: face_field.hessian.len(),
        volume_gradient_samples: volume_g.len(),
        volume_hessian_samples,
        volume_position,
        volume_weight,
        volume_rho,
        volume_g,
        vertex_g,
        surface_g,
        surface_normal,
    })
}

fn centrifugal(gradient: DVec3, x: DVec3, omega: [f64; 3]) -> DVec3 {
    let w = DVec3::new(omega[0], omega[1], omega[2]);
    let w2 = w.length_squared();
    let along = if w2 == 0.0 { 0.0 } else { x.dot(w) / w2 };
    let parallel = w * along;
    let perp = x - parallel;
    gradient + perp * w2
}

fn eval_role(
    _project: &Project,
    density: &CauchyDensity,
    options: &RtfpOptions,
    mesh: &TriMesh,
    points: &[DVec3],
    role: PointRole,
    want: FieldMask,
) -> Result<crate::rtfp_solver::api::FieldBatch, ScfError> {
    if points.is_empty() {
        return Err(ScfError::model_not_closed(
            "a field role has no sample points",
        ));
    }
    bind_and_eval(
        options.clone(),
        mesh,
        density.clone(),
        &PointSet::from_points(points),
        role,
        want,
    )
    .map_err(|err| ScfError::model_not_closed(err.detail))
}

struct NodalPhi {
    phi: Vec<f64>,
    mu: Vec<f64>,
    pressure: Vec<f64>,
    shear: Vec<f64>,
}

fn nodal_phi(
    project: &Project,
    mesh: &TriMesh,
    density: &CauchyDensity,
    mechanics: &crate::scf_simulator::stokes::picard_loop::PicardOutput,
) -> Result<NodalPhi, ScfError> {
    let n = mesh.vertices.len();
    if mechanics.vertex_pressure.len() != n || mechanics.vertex_strain.len() != n {
        return Err(ScfError::rheology_not_converged(
            "pressure or shear rate was not mapped onto every surface node",
        ));
    }
    let mut phi = Vec::with_capacity(n);
    let mut mu = Vec::with_capacity(n);
    let mut pressure = Vec::with_capacity(n);
    let mut shear = Vec::with_capacity(n);
    for i in 0..n {
        let p = mechanics.vertex_pressure[i];
        let rate = &mechanics.vertex_strain[i];
        if !p.is_finite() || rate.iter().any(|value| !value.is_finite()) {
            return Err(ScfError::rheology_not_converged(
                "a nodal pressure or strain tensor is not finite",
            ));
        }
        let gamma = (2.0 * rate.iter().map(|value| value * value).sum::<f64>()).sqrt();
        let rho = density.rho(mesh.vertices[i]);
        if !(rho.is_finite() && rho > 0.0) {
            return Err(ScfError::model_not_closed(
                "nodal density for the inertial number is not positive",
            ));
        }
        let mu_i = mu_of_inertial(
            inertial_number(project.grain_diameter, gamma, p.max(project.p_min), rho),
            project.mu_s,
            project.mu_2,
            project.i_0,
        );
        phi.push(mu_i.max(0.0).atan());
        mu.push(mu_i);
        pressure.push(p.max(project.p_min));
        shear.push(gamma);
    }
    Ok(NodalPhi {
        phi,
        mu,
        pressure,
        shear,
    })
}

fn yield_flags(
    project: &Project,
    friction: &NodalPhi,
    mechanics: &crate::scf_simulator::stokes::picard_loop::PicardOutput,
) -> Result<(Vec<bool>, f64), ScfError> {
    let mut flags = Vec::with_capacity(friction.phi.len());
    let mut yield_residual = 0.0_f64;
    for i in 0..friction.phi.len() {
        let mut tensor = [[0.0; 3]; 3];
        let rate = &mechanics.vertex_strain[i];
        for row in 0..3 {
            for col in 0..3 {
                tensor[row][col] = 0.5 * (rate[row * 3 + col] + rate[col * 3 + row]);
            }
        }
        let p = friction.pressure[i];
        let eta = flow_viscosity(
            project.cohesion,
            friction.mu[i],
            p,
            friction.shear[i],
            project.shear_floor,
        );
        let tau = flow_stress(eta, &tensor);
        let tau_y = project.cohesion + friction.mu[i] * p;
        let f_delta = equivalent_flow_stress(&tau) - (1.0 - project.delta_y) * tau_y;
        let classic = classic_yield_function(&tau, project.cohesion, friction.mu[i], p);
        if !classic.is_finite() || !f_delta.is_finite() {
            return Err(ScfError::model_not_closed("yield residual is not finite"));
        }
        yield_residual = yield_residual.max(classic.max(0.0));
        flags.push(is_yielded(f_delta.max(classic)));
    }
    Ok((flags, yield_residual))
}

fn vertex_neighbors(faces: &[[u32; 3]], n: usize) -> Vec<Vec<usize>> {
    let mut neighbors = vec![Vec::new(); n];
    for face in faces {
        let ids = [face[0] as usize, face[1] as usize, face[2] as usize];
        for a in 0..3 {
            for b in 0..3 {
                if a != b && !neighbors[ids[a]].contains(&ids[b]) {
                    neighbors[ids[a]].push(ids[b]);
                }
            }
        }
    }
    neighbors
}

fn vertex_masses(mesh: &TriMesh, density: &CauchyDensity) -> Result<Vec<f64>, ScfError> {
    let mut mass = vec![0.0; mesh.vertices.len()];
    for f in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(f);
        let area = (b - a).cross(c - a).length() * 0.5;
        let centroid = (a + b + c) * (1.0 / 3.0);
        let rho = density.rho(centroid).max(0.0);
        let share = rho * area / 3.0;
        for index in mesh.faces[f] {
            mass[index as usize] += share;
        }
    }
    if mass.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(ScfError::model_not_closed(
            "a surface node has no positive mass",
        ));
    }
    Ok(mass)
}

fn relative_slip(
    positions: &[[f64; 3]],
    velocity: &[[f64; 3]],
    core: &[bool],
) -> Result<Vec<[f64; 3]>, ScfError> {
    let mut points = Vec::new();
    let mut speeds = Vec::new();
    for (i, locked) in core.iter().enumerate() {
        if *locked {
            points.push(positions[i]);
            speeds.push(*velocity.get(i).unwrap_or(&[0.0; 3]));
        }
    }
    if points.is_empty() {
        return Ok(vec![[0.0; 3]; positions.len()]);
    }
    let motion = fit_core_motion(&points, &speeds)?;
    Ok(positions
        .iter()
        .zip(velocity)
        .map(|(point, speed)| slip_velocity(*point, *speed, motion))
        .collect())
}

fn mode_c(
    project: &Project,
    yielded: &[bool],
    slip: &[[f64; 3]],
    node_mass: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut erosion = vec![0.0; node_mass.len()];
    for i in 0..node_mass.len() {
        if yielded.get(i).copied().unwrap_or(false) {
            let speed =
                (slip[i][0] * slip[i][0] + slip[i][1] * slip[i][1] + slip[i][2] * slip[i][2])
                    .sqrt();
            erosion[i] = project.erosion_coefficient * project.stress_scale * speed * node_mass[i];
        }
    }
    let eroded = erosion.iter().sum::<f64>();
    let incoming = eroded + project.external_mass_flux;
    let weight: f64 = node_mass.iter().sum();
    let deposition = node_mass.iter().map(|m| incoming * m / weight).collect();
    (erosion, deposition)
}

fn audit_gravitational_field(
    project: &Project,
    density: &CauchyDensity,
    mesh: &TriMesh,
) -> Result<(), ScfError> {
    let options = RtfpOptions {
        directions: project.directions,
        r0: project.near_radius,
        ell0: project.ell0,
        g: project.g,
        quadrature_nodes: project.radial_degree.max(2),
    };
    let faces: Vec<DVec3> = (0..mesh.faces.len())
        .map(|face| {
            let [a, b, c] = mesh.triangle(face);
            (a + b + c) * (1.0 / 3.0)
        })
        .collect();
    let face_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &faces,
        PointRole::SurfaceFace,
        FieldMask::ALL,
    )?;
    let face_gradient: Vec<[f64; 3]> = face_field
        .gradient
        .iter()
        .map(|gradient| gradient.to_array())
        .collect();
    if !finite_gravitational_field(&face_field.potential, &face_gradient, &face_field.hessian) {
        return Err(ScfError::model_not_closed(
            "U, grad U or the Hessian on the accepted surface is not finite",
        ));
    }
    let vertex_field = eval_role(
        project,
        density,
        &options,
        mesh,
        &mesh.vertices,
        PointRole::SurfaceVertex,
        FieldMask {
            potential: true,
            gradient: true,
            hessian: false,
        },
    )?;
    if vertex_field
        .potential
        .iter()
        .any(|value| !value.is_finite())
        || vertex_field
            .gradient
            .iter()
            .any(|gradient| !gradient.is_finite())
    {
        return Err(ScfError::model_not_closed(
            "U or grad U on an accepted vertex is not finite",
        ));
    }
    Ok(())
}

fn slope_residual(mesh: &TriMesh, phi: &[f64], forward: &Forward) -> f64 {
    let mut sin2_slope = Vec::new();
    let mut sin2_phi = Vec::new();
    for (i, g) in forward.surface_g.iter().enumerate() {
        let ids = mesh.faces[i];
        let angle = (phi[ids[0] as usize] + phi[ids[1] as usize] + phi[ids[2] as usize]) / 3.0;
        let g = DVec3::from_array(*g);
        let n = DVec3::from_array(forward.surface_normal[i]);
        let denom = g.length_squared() * n.length_squared();
        let slope = if denom > 1e-20 {
            (g.cross(n).length_squared() / denom).min(1.0)
        } else {
            0.0
        };
        sin2_slope.push(slope);
        sin2_phi.push(angle.sin().powi(2));
    }
    compute_slope_residual(&sin2_slope, &sin2_phi)
}

fn surface_modes(values: &[f64]) -> Vec<[f64; 4]> {
    values.iter().map(|v| [*v, 0.0, 0.0, 0.0]).collect()
}

fn finite_rtfp_residual(initial: f64, candidate: f64) -> Result<f64, ScfError> {
    if !initial.is_finite() || !candidate.is_finite() {
        return Err(ScfError::model_not_closed(
            "rtfp_residual is not finite; stability was not assessed",
        ));
    }
    Ok(initial.max(candidate))
}

fn dispatch_general_mesh(
    text: &str,
    project: &Project,
    max_steps: Option<u32>,
    until_t: Option<f64>,
    out_dir: Option<&Path>,
    resume_mesh: Option<crate::scf_simulator::checkpoint::MeshCheckpoint>,
    checkpoint: Option<&Path>,
    stable_geometry_tol: Option<f64>,
    stable_field_tol: Option<f64>,
    stable_steps: u32,
) -> Result<ClosedRun, ScfError> {
    let parsed: toml::Value =
        toml::from_str(text).unwrap_or(toml::Value::Table(Default::default()));
    let rho_bulk = project
        .density
        .kernels
        .first()
        .map(|kernel| kernel.weight)
        .unwrap_or(0.0)
        .max(project.density.bulk);
    let rho_grain = parsed
        .get("density")
        .and_then(|table| table.get("grain_density"))
        .and_then(|record| record.get("value"))
        .and_then(|value| value.as_float())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(rho_bulk);
    let job = crate::scf_simulator::mesh_loop::MeshJob {
        raw_hash: project.raw_hash.clone(),
        mesh: project.mesh.clone(),
        smoothed_hash: project.smoothed_hash.clone(),
        star: project.star.clone(),
        star_min_cos: project.star_min_cos,
        require_star: project.require_star,
        slope: project.slope.law.clone(),
        crater: project.crater.clone(),
        initial_volume: project.initial_volume,
        initial_crater: project.initial_crater.clone(),
        density: project.density.clone(),
        density_hash: project.density_hash.clone(),
        density_unit: project.density_unit.clone(),
        mass_target: project.mass_target,
        spin: project.spin.clone(),
        cohesion: project.cohesion,
        mu_y: project.mu_y,
        mu_s: project.mu_s,
        mu_2: project.mu_2,
        i_0: project.i_0,
        grain_diameter: project.grain_diameter,
        p_min: project.p_min,
        delta_y: project.delta_y,
        erosion_coefficient: project.erosion_coefficient,
        external_mass_flux: project.external_mass_flux,
        stress_scale: project.stress_scale,
        l_max: project.l_max,
        directions: project.directions,
        radial_degree: project.radial_degree,
        gamma: project.gamma,
        g: project.g,
        shear_floor: project.shear_floor,
        near_radius: project.near_radius,
        ell0: project.ell0,
        under_relaxation: project.under_relaxation,
        picard_iterations: project.picard_iterations,
        tolerances: project.tolerances,
        trust_radius: project.trust_radius,
        cfl: project.cfl,
        coupling: project.coupling,
        characteristic: project.characteristic,
        spin_dt: project.spin_dt,
        dt_max: project.dt_max,
        raw_mesh: project.raw_mesh.clone(),
        require_convergence: project.require_convergence,
        rho_bulk,
        rho_grain,
    };
    let mesh_options = crate::scf_simulator::mesh_loop::MeshOptions {
        max_steps,
        until_t,
        out_dir: out_dir.map(|path| path.to_path_buf()),
        resume: resume_mesh,
        stable_geometry_tol: stable_geometry_tol.unwrap_or(project.tolerances.geometry_tol),
        stable_field_tol: stable_field_tol.unwrap_or(project.tolerances.geometry_tol),
        stable_steps,
        checkpoint: checkpoint.map(|path| path.to_path_buf()),
    };
    crate::scf_simulator::general_mesh::execute(&job, &mesh_options)
}

fn admit_radial_star(
    mesh: &TriMesh,
    density: &CauchyDensity,
    star_min_cos: f64,
) -> Result<(), ScfError> {
    let report =
        crate::scf_simulator::geometry_update::star::audit_star(mesh, density, star_min_cos)?;
    if !report.shaped || report.unique_crossing_count != 1 {
        return Err(ScfError::model_not_closed(format!(
            "radial_star refused after the coalesced star audit: {}",
            report.detail
        )));
    }
    if !(report.star_margin.is_finite() && report.star_margin >= star_min_cos) {
        return Err(ScfError::model_not_closed(format!(
            "radial_star n·e_r {} is below c0 {}",
            report.star_margin, star_min_cos
        )));
    }
    if !(report.radial_error.is_finite() && report.radial_error <= 1.0e-3) {
        return Err(ScfError::model_not_closed(format!(
            "radial_star radial error {} is not a finite chart residual",
            report.radial_error
        )));
    }
    let scale = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.length())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mass_center = DVec3::from_array(report.mass_centroid);
    if mass_center.length() > 1.0e-2 * scale || report.volume_mass_delta > 1.0e-2 * scale {
        let normals = vertex_unit_normals(mesh).map_err(ScfError::model_not_closed)?;
        let mut origin_margin = f64::INFINITY;
        for (vertex, normal) in mesh.vertices.iter().zip(&normals) {
            let Some(direction) = vertex.try_normalize() else {
                origin_margin = -1.0;
                break;
            };
            origin_margin = origin_margin.min(DVec3::from_array(*normal).dot(direction));
        }
        return Err(ScfError::model_not_closed(format!(
            "radial_star mass centroid is not the radial origin; star_margin {origin_margin}"
        )));
    }
    Ok(())
}

fn mesh_from_radii(
    directions: &[DVec3],
    radii: &[f64],
    faces: &[[u32; 3]],
) -> Result<TriMesh, ScfError> {
    let vertices = directions
        .iter()
        .zip(radii)
        .map(|(dir, r)| *dir * *r)
        .collect();
    TriMesh::try_new(vertices, faces.to_vec()).map_err(|err| ScfError::model_not_closed(err.detail))
}

fn parse_project(text: &str, project_dir: &Path, load_mesh: bool) -> Result<Project, ScfError> {
    let value: toml::Value =
        toml::from_str(text).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    audit_document(&value)?;
    let geometry = section(&value, "geometry")?;
    let locator = geometry_locator(geometry)?;
    let representation = match geometry
        .get("representation")
        .and_then(|item| item.as_str())
    {
        None | Some("radial_star") | Some("star_radial") => Representation::RadialStar,
        Some("general_mesh") | Some("mesh") => Representation::GeneralMesh,
        Some(other) => {
            return Err(ScfError::model_not_closed(format!(
                "geometry.representation '{other}' is not radial_star or general_mesh"
            )));
        }
    };
    let geometry_unit = match geometry.get("length_unit") {
        None => None,
        Some(toml::Value::String(unit)) => Some(unit.clone()),
        Some(_) => {
            return Err(ScfError::model_not_closed(
                "geometry.length_unit is not a string",
            ));
        }
    };
    let reconstruct = match geometry.get("reconstruct") {
        None => None,
        Some(toml::Value::String(name)) => Some(match name.as_str() {
            "convex_hull" => crate::rtfp_solver::io::Reconstruct::ConvexHull,
            "radial_raycast" => crate::rtfp_solver::io::Reconstruct::RadialRaycast,
            other => {
                return Err(ScfError::model_not_closed(format!(
                    "geometry.reconstruct '{other}' is not convex_hull or radial_raycast"
                )));
            }
        }),
        Some(_) => {
            return Err(ScfError::model_not_closed(
                "geometry.reconstruct is not a string",
            ));
        }
    };
    if representation == Representation::GeneralMesh {
        for role in ["raw_input_mesh", "smoothed_initial_mesh", "evolving_mesh"] {
            let table = geometry
                .get(role)
                .and_then(|item| item.as_table())
                .ok_or_else(|| ScfError::model_not_closed(format!("missing geometry.{role}")))?;
            if table.get("role").and_then(|item| item.as_str()) != Some(role) {
                return Err(ScfError::model_not_closed(format!(
                    "geometry.{role}.role must be \"{role}\""
                )));
            }
        }
        if value.get("smoothing").is_none() {
            return Err(ScfError::model_not_closed(
                "mesh representation requires an explicit [smoothing] section",
            ));
        }
    }
    let density_table = section(&value, "density")?;
    reject_require_nonnegative_flag(density_table)?;
    let bulk = require_f64(density_table, "bulk", "density")?;
    if bulk < 0.0 {
        return Err(ScfError::model_not_closed(
            "density.bulk must be non-negative",
        ));
    }
    let (mass_target, mass_uncertainty) = mass_target_of(density_table)?;
    let kernel_values = density_table
        .get("kernels")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ScfError::model_not_closed("missing density.kernels"))?;
    if kernel_values.is_empty() {
        return Err(ScfError::model_not_closed("missing density.kernels"));
    }
    let mut kernels = Vec::new();
    for (i, kernel) in kernel_values.iter().enumerate() {
        if kernel.as_table().is_none() {
            return Err(ScfError::model_not_closed(format!(
                "illegal density kernel at density.kernels[{i}]"
            )));
        }
        reject_kernel_kind(kernel, i)?;
        let center = vec3(kernel, "center", &format!("density.kernels[{i}]"))?;
        let weight = require_f64(kernel, "weight", &format!("density.kernels[{i}]"))?;
        let sigma = require_f64(kernel, "sigma", &format!("density.kernels[{i}]"))?;
        let alpha = require_f64(kernel, "alpha", &format!("density.kernels[{i}]"))?;
        audit_kernel_provenance(kernel, i, center, weight, sigma, alpha)?;
        kernels.push(
            CauchyKernel::try_new(
                DVec3::new(center[0], center[1], center[2]),
                weight,
                sigma,
                alpha,
            )
            .map_err(|err| {
                ScfError::model_not_closed(format!("illegal density kernel: {}", err.detail))
            })?,
        );
    }
    let density = CauchyDensity::try_new(bulk, kernels)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    let project_unit = section(&value, "project")?
        .get("density_unit")
        .and_then(|item| item.as_str())
        .unwrap_or("kg/m3")
        .to_string();
    let material = section(&value, "material")?;
    let (cohesion, mu_y, eta_c, p_min, delta_y) = parse_material(material)?;
    let mode_c = section(&value, "mode_c")?;
    let (erosion_coefficient, external_mass_flux, stress_scale) = parse_mode_c(mode_c)?;
    let galerkin = section(&value, "galerkin")?;
    let l_max = require_count(galerkin, "l_max", "galerkin", 1)?;
    let directions = require_count(galerkin, "directions", "galerkin", 8)?;
    let radial_degree = require_count(galerkin, "radial_degree", "galerkin", 2)?;
    let near_radius = positive(galerkin, "near_radius", "galerkin")?;
    let ell0 = positive(galerkin, "ell0", "galerkin")?;
    let stokes = section(&value, "stokes")?;
    let gamma_record = sourced_record(stokes, "gamma", "1")?;
    let gamma_source = required_text(stokes.get("gamma").expect("gamma"), "source", "gamma")?;
    if !gamma_source.to_ascii_lowercase().contains("augmented") {
        return Err(ScfError::model_not_closed(
            "stokes.gamma must identify the augmented-Lagrange penalty",
        ));
    }
    let shear_record = sourced_record(stokes, "shear_floor", "1/s")?;
    let gamma = gamma_record.value;
    let g = require_f64(stokes, "g", "stokes")?;
    let shear_floor = shear_record.value;
    let gamma_eps = sourced_record(material, "gamma_eps", "1/s")?;
    if gamma <= 0.0 || g <= 0.0 || shear_floor <= 0.0 {
        return Err(ScfError::model_not_closed(
            "stokes gamma, g, and shear_floor must be positive",
        ));
    }
    if (shear_floor - gamma_eps.value).abs() > shear_record.uncertainty + gamma_eps.uncertainty {
        return Err(ScfError::model_not_closed(
            "stokes.shear_floor disagrees with material.gamma_eps",
        ));
    }
    let picard = section(&value, "picard")?;
    let under_relaxation = require_f64(picard, "under_relaxation", "picard")?;
    let picard_iterations = require_count(picard, "max_iterations", "picard", 1)?;
    let require_convergence = match picard.get("require_convergence") {
        None => false,
        Some(toml::Value::Boolean(flag)) => *flag,
        Some(_) => {
            return Err(ScfError::model_not_closed(
                "picard.require_convergence must be a boolean",
            ));
        }
    };
    if !(0.0 < under_relaxation && under_relaxation <= 1.0) {
        return Err(ScfError::model_not_closed(
            "picard under_relaxation must be in (0, 1] and max_iterations >= 1",
        ));
    }
    let accept = section(&value, "accept")?;
    let tolerances = ScfTolerances {
        geometry_tol: positive(accept, "geometry_tol", "accept")?,
        mass_tol: positive(accept, "mass_tol", "accept")?,
        slope_tol: positive(accept, "slope_tol", "accept")?,
        transport_tol: positive(accept, "transport_tol", "accept")?,
        kkt_tol: positive(accept, "kkt_tol", "accept")?,
    };
    let relative_mass = mass_uncertainty / mass_target;
    if (tolerances.mass_tol - relative_mass).abs() > 1.0e-12 * relative_mass.max(1.0) {
        return Err(ScfError::model_not_closed(
            "accept.mass_tol must equal the sourced mass uncertainty divided by mass_target",
        ));
    }
    let trust_radius = positive(accept, "trust_radius", "accept")?;
    let cfl = positive(accept, "cfl", "accept")?;
    let coupling = positive(accept, "coupling", "accept")?;
    let characteristic = positive(accept, "characteristic", "accept")?;
    let spin_dt = positive(accept, "spin", "accept")?;
    let dt_max = positive(accept, "dt_max", "accept")?;
    let rotation = vec3(section(&value, "rotation")?, "omega", "rotation")?;
    let spin = parse_spin(section(&value, "spin_schedule")?)?;
    let mu_s = material_coefficient(material, "mu_s", "1")?;
    let mu_2 = material_coefficient(material, "mu_2", "1")?;
    let i_0 = material_coefficient(material, "i_0", "1")?;
    let grain_diameter = material_coefficient(material, "grain_diameter", "m")?;
    if i_0 <= 0.0 || grain_diameter <= 0.0 {
        return Err(ScfError::model_not_closed(
            "material.i_0 and material.grain_diameter must be positive",
        ));
    }
    let (require_star, star_min_cos, slope) = parse_smoothing(&value, g)?;
    let spin_gap =
        (rotation[0].hypot(rotation[1]).hypot(rotation[2]) - slope.law.spin_rate.value).abs();
    if spin_gap > slope.law.spin_rate.uncertainty {
        return Err(ScfError::model_not_closed(
            "rotation.omega is outside the sourced smoothing.spin_rate uncertainty",
        ));
    }
    if (dt_max - slope.law.timestep.value).abs() > slope.law.timestep.uncertainty {
        return Err(ScfError::model_not_closed(
            "accept.dt_max is outside the sourced smoothing.timestep uncertainty",
        ));
    }
    audit_material_calibration(
        &material,
        &slope.law,
        mu_s,
        mu_2,
        i_0,
        grain_diameter,
        cohesion,
        eta_c,
    )?;
    audit_density_calibration(&density_table, &density, &slope.law)?;
    audit_spin_horizon(section(&value, "spin_schedule")?, &spin, &slope.law)?;
    let raw = if load_mesh {
        read_mesh(
            &project_dir.join(locator),
            geometry_unit.as_deref(),
            reconstruct,
        )?
    } else {
        TriMesh::unit_cube()
    };
    let raw_hash = crate::rtfp_solver::io::mesh_hash(&raw);
    let (mesh, smoothed_hash, star, crater, vertex_records) = if load_mesh
        && representation == Representation::GeneralMesh
    {
        let crater = crate::scf_simulator::geometry_update::smooth::load_crater_mask(
            &slope,
            &raw,
            project_dir,
        )?;
        let (mass_now, _) = crate::rtfp_solver::volume::closed_mass_moments(&raw, &density)
            .map_err(ScfError::model_not_closed)?;
        let room = mass_uncertainty - (mass_now - mass_target).abs();
        let relative_room = if mass_now.abs() > 0.0 {
            (room / mass_now.abs()).max(0.0)
        } else {
            0.0
        };
        let relaxed = crate::scf_simulator::geometry_update::smooth::relax_slopes(
            &raw,
            &density,
            &slope,
            &crater,
            rotation,
            tolerances.mass_tol.min(relative_room),
        )?;
        let hash = crate::rtfp_solver::io::mesh_hash(&relaxed.mesh);
        // Star shape is recorded for the diagnostic. It does not refuse general-mesh RTFP.
        let _ = require_star;
        let star = crate::scf_simulator::geometry_update::star::audit_star(
            &relaxed.mesh,
            &density,
            star_min_cos,
        )?;
        (relaxed.mesh, hash, star, relaxed.mask, relaxed.vertices)
    } else {
        (
            raw.clone(),
            raw_hash.clone(),
            crate::scf_simulator::geometry_update::star::StarReport::radial_representation(),
            crate::scf_simulator::geometry_update::smooth::CraterMask::unmasked(raw.vertices.len()),
            Vec::new(),
        )
    };
    let initial_volume = mesh.signed_volume();
    let initial_crater =
        crate::scf_simulator::geometry_update::smooth::crater_measure(&mesh, &crater);
    require_nonnegative_density(&density, &mesh)?;
    if load_mesh && representation == Representation::GeneralMesh {
        audit_enclosed_mass(&mesh, &density, mass_target, mass_uncertainty)?;
    }
    let density_hash = hash_density(&density, &project_unit);
    Ok(Project {
        raw_mesh: raw.clone(),
        mesh,
        density,
        mass_target,
        spin,
        cohesion,
        mu_y,
        eta_c,
        p_min,
        delta_y,
        erosion_coefficient,
        external_mass_flux,
        stress_scale,
        l_max,
        directions,
        radial_degree,
        gamma,
        g,
        shear_floor,
        near_radius,
        ell0,
        under_relaxation,
        picard_iterations,
        require_convergence,
        tolerances,
        trust_radius,
        cfl,
        coupling,
        characteristic,
        spin_dt,
        dt_max,
        representation,
        require_star,
        star_min_cos,
        slope,
        crater,
        initial_volume,
        initial_crater,
        raw_hash,
        smoothed_hash,
        star,
        vertex_records,
        density_hash,
        density_unit: project_unit,
        mu_s,
        mu_2,
        i_0,
        grain_diameter,
    })
}

fn parse_spin(table: &toml::Value) -> Result<SpinSchedule, ScfError> {
    let kind = table
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ScfError::model_not_closed("missing spin_schedule.kind"))?;
    match kind {
        "stages" => {
            let stages = table
                .get("stages")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ScfError::model_not_closed("missing spin_schedule.stages"))?;
            if stages.is_empty() {
                return Err(ScfError::model_not_closed("spin stages are empty"));
            }
            let mut out = Vec::new();
            for (i, stage) in stages.iter().enumerate() {
                out.push(crate::scf_simulator::spin::SpinStage {
                    t_end: require_f64(stage, "t_end", &format!("spin_schedule.stages[{i}]"))?,
                    omega: vec3(stage, "omega", &format!("spin_schedule.stages[{i}]"))?,
                });
            }
            Ok(SpinSchedule::Stages(out))
        }
        "linear" => Ok(SpinSchedule::Linear {
            omega0: vec3(table, "omega0", "spin_schedule")?,
            omega1: vec3(table, "omega1", "spin_schedule")?,
            t0: require_f64(table, "t0", "spin_schedule")?,
            t1: require_f64(table, "t1", "spin_schedule")?,
        }),
        "table" => {
            let times = table
                .get("times")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ScfError::model_not_closed("missing spin_schedule.times"))?;
            let omegas = table
                .get("omegas")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ScfError::model_not_closed("missing spin_schedule.omegas"))?;
            let times = times
                .iter()
                .map(|v| finite_number(v, "spin_schedule.times"))
                .collect::<Result<Vec<_>, _>>()?;
            let mut parsed = Vec::new();
            for (i, omega) in omegas.iter().enumerate() {
                let label = format!("spin_schedule.omegas[{i}]");
                parsed.push(if omega.get("omega").is_some() {
                    vec3(omega, "omega", &label)?
                } else {
                    number_triple(omega, &label)?
                });
            }
            Ok(SpinSchedule::Table {
                times,
                omegas: parsed,
            })
        }
        _ => Err(ScfError::model_not_closed(
            "spin_schedule.kind must be stages, linear, or table",
        )),
    }
}

fn load_obj(text: &str) -> Result<TriMesh, ScfError> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x = parse_token(parts.next(), "obj vertex")?;
                let y = parse_token(parts.next(), "obj vertex")?;
                let z = parse_token(parts.next(), "obj vertex")?;
                vertices.push(DVec3::new(x, y, z));
            }
            Some("f") => {
                let mut ids = [0u32; 3];
                for slot in &mut ids {
                    let token = parts
                        .next()
                        .ok_or_else(|| ScfError::model_not_closed("obj face is not a triangle"))?;
                    let id = token
                        .split('/')
                        .next()
                        .unwrap_or(token)
                        .parse::<i64>()
                        .map_err(|_| {
                            ScfError::model_not_closed("obj face index is not an integer")
                        })?;
                    if id <= 0 {
                        return Err(ScfError::model_not_closed(
                            "obj face index must be positive",
                        ));
                    }
                    *slot = u32::try_from(id - 1).map_err(|_| {
                        ScfError::model_not_closed("obj face index is out of range")
                    })?;
                }
                if parts.next().is_some() {
                    return Err(ScfError::model_not_closed("obj face is not a triangle"));
                }
                faces.push(ids);
            }
            _ => {}
        }
    }
    TriMesh::try_new(vertices, faces).map_err(|err| ScfError::model_not_closed(err.detail))
}

fn parse_token(token: Option<&str>, what: &str) -> Result<f64, ScfError> {
    let value = token
        .ok_or_else(|| ScfError::model_not_closed(format!("{what} is incomplete")))?
        .parse::<f64>()
        .map_err(|_| ScfError::model_not_closed(format!("{what} is not numeric")))?;
    if !value.is_finite() {
        return Err(ScfError::model_not_closed(format!("{what} is not finite")));
    }
    Ok(value)
}

fn section<'a>(value: &'a toml::Value, name: &str) -> Result<&'a toml::Value, ScfError> {
    value
        .get(name)
        .ok_or_else(|| ScfError::model_not_closed(format!("missing {name}")))
}

fn require_f64(table: &toml::Value, key: &str, section_name: &str) -> Result<f64, ScfError> {
    match table.get(key) {
        None => Err(ScfError::model_not_closed(format!(
            "missing {section_name}.{key}"
        ))),
        Some(value) => finite_number(value, &format!("{section_name}.{key}")),
    }
}

fn finite_number(value: &toml::Value, label: &str) -> Result<f64, ScfError> {
    let number = value
        .as_float()
        .or_else(|| value.as_integer().map(|i| i as f64))
        .ok_or_else(|| ScfError::model_not_closed(format!("{label} is not numeric")))?;
    if !number.is_finite() {
        return Err(ScfError::model_not_closed(format!("{label} is not finite")));
    }
    Ok(number)
}

fn require_count(
    table: &toml::Value,
    key: &str,
    section_name: &str,
    min: usize,
) -> Result<usize, ScfError> {
    let number = require_f64(table, key, section_name)?;
    if number.fract() != 0.0 || number < min as f64 || number > u32::MAX as f64 {
        return Err(ScfError::model_not_closed(format!(
            "{section_name}.{key} must be an integer >= {min}"
        )));
    }
    Ok(number as usize)
}

fn positive(table: &toml::Value, key: &str, section_name: &str) -> Result<f64, ScfError> {
    let number = require_f64(table, key, section_name)?;
    if number <= 0.0 {
        return Err(ScfError::model_not_closed(format!(
            "{section_name}.{key} must be positive"
        )));
    }
    Ok(number)
}

fn number_triple(value: &toml::Value, label: &str) -> Result<[f64; 3], ScfError> {
    let Some(values) = value.as_array() else {
        return Err(ScfError::model_not_closed(format!(
            "{label} is not numeric"
        )));
    };
    triple_from(values, label)
}

fn vec3(table: &toml::Value, key: &str, section_name: &str) -> Result<[f64; 3], ScfError> {
    let label = format!("{section_name}.{key}");
    match table.get(key) {
        None => Err(ScfError::model_not_closed(format!("missing {label}"))),
        Some(value) => number_triple(value, &label),
    }
}

fn triple_from(values: &[toml::Value], label: &str) -> Result<[f64; 3], ScfError> {
    if values.len() != 3 {
        return Err(ScfError::model_not_closed(format!(
            "{label} must have 3 components"
        )));
    }
    let mut out = [0.0; 3];
    for (slot, value) in out.iter_mut().zip(values) {
        *slot = finite_number(value, label)?;
    }
    Ok(out)
}

fn geometry_locator(geometry: &toml::Value) -> Result<&str, ScfError> {
    let path = path_string(geometry, "path")?;
    let source = path_string(geometry, "source")?;
    path.or(source)
        .ok_or_else(|| ScfError::model_not_closed("missing geometry.path or geometry.source"))
}

fn path_string<'a>(table: &'a toml::Value, key: &str) -> Result<Option<&'a str>, ScfError> {
    match table.get(key) {
        None => Ok(None),
        Some(toml::Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() {
                Ok(None)
            } else {
                Ok(Some(text))
            }
        }
        Some(_) => Err(ScfError::model_not_closed(format!(
            "geometry.{key} is not a path"
        ))),
    }
}

pub fn load_mesh_file(path: &Path) -> Result<TriMesh, ScfError> {
    read_mesh(path, None, None)
}

fn read_mesh(
    path: &Path,
    length_unit: Option<&str>,
    reconstruct: Option<crate::rtfp_solver::io::Reconstruct>,
) -> Result<TriMesh, ScfError> {
    if !path.is_file() {
        return Err(ScfError::model_not_closed(format!(
            "geometry file does not exist: {}",
            path.display()
        )));
    }
    let bytes = std::fs::read(path)
        .map_err(|err| ScfError::model_not_closed(format!("geometry file: {err}")))?;
    if bytes.is_empty() {
        return Err(ScfError::model_not_closed("geometry file is empty"));
    }
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(extension.as_str(), "glb" | "gltf") {
        let unit = length_unit
            .ok_or_else(|| ScfError::model_not_closed("GLB import needs geometry.length_unit"))?;
        let document =
            crate::rtfp_solver::io::load_mesh_file(crate::rtfp_solver::io::ImportRequest {
                path,
                length_unit: Some(unit),
                reconstruct,
            })
            .map_err(|err| ScfError::model_not_closed(err.detail))?;
        return Ok(document.mesh);
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| ScfError::model_not_closed("geometry file has an unrecognized format"))?;
    if text.trim().is_empty() {
        return Err(ScfError::model_not_closed("geometry file is empty"));
    }
    let has_vertex = text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("v ") || line.starts_with("v\t")
    });
    if !has_vertex {
        return Err(ScfError::model_not_closed(
            "geometry file has an unrecognized format",
        ));
    }
    let has_face = text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("f ") || line.starts_with("f\t")
    });
    if !has_face {
        let strategy = reconstruct.ok_or_else(|| {
            ScfError::model_not_closed(
                "vertex input has no faces; pass geometry.reconstruct = \"convex_hull\" or \"radial_raycast\"",
            )
        })?;
        let unit = length_unit.unwrap_or("1");
        let document =
            crate::rtfp_solver::io::load_mesh_file(crate::rtfp_solver::io::ImportRequest {
                path,
                length_unit: Some(unit),
                reconstruct: Some(strategy),
            })
            .map_err(|err| ScfError::model_not_closed(err.detail))?;
        return Ok(document.mesh);
    }
    let _ = length_unit;
    load_obj(text)
}

fn parse_smoothing(
    value: &toml::Value,
    gravity: f64,
) -> Result<
    (
        bool,
        f64,
        crate::scf_simulator::geometry_update::smooth::SlopeRequest,
    ),
    ScfError,
> {
    let Some(table) = value.get("smoothing") else {
        return Err(ScfError::model_not_closed(
            "missing [smoothing]; slope relaxation needs sourced material parameters",
        ));
    };
    crate::scf_simulator::geometry_update::smooth::parse_slope_request(table, gravity)
}

fn hash_density(density: &CauchyDensity, unit: &str) -> String {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(unit.as_bytes());
    hasher.update(&density.bulk.to_le_bytes());
    for kernel in &density.kernels {
        for value in [
            kernel.center.x,
            kernel.center.y,
            kernel.center.z,
            kernel.weight,
            kernel.sigma,
            kernel.alpha,
        ] {
            hasher.update(&value.to_le_bytes());
        }
    }
    format!("{:08x}", hasher.finalize())
}

fn mass_target_of(table: &toml::Value) -> Result<(f64, f64), ScfError> {
    let key = if table.get("mass_target").is_some() {
        "mass_target"
    } else if table.get("total_mass_target").is_some() {
        "total_mass_target"
    } else {
        return Err(ScfError::model_not_closed("missing density mass target"));
    };
    let mass_target = require_f64(table, key, "density")?;
    if mass_target <= 0.0 {
        return Err(ScfError::model_not_closed(
            "density mass target must be positive",
        ));
    }
    let record = sourced_record(table, "mass", "kg")?;
    if (record.value - mass_target).abs() > 0.0 {
        return Err(ScfError::model_not_closed(
            "density.mass_target and density.mass.value disagree",
        ));
    }
    if record.uncertainty < 0.0 || record.uncertainty >= mass_target {
        return Err(ScfError::model_not_closed(
            "density.mass.uncertainty must lie in [0, mass_target)",
        ));
    }
    Ok((mass_target, record.uncertainty))
}

struct SourcedRecord {
    value: f64,
    uncertainty: f64,
    calibration: String,
    provenance: String,
}

fn sourced_record(table: &toml::Value, name: &str, unit: &str) -> Result<SourcedRecord, ScfError> {
    let item = table.get(name).ok_or_else(|| {
        ScfError::model_not_closed(format!(
            "missing {name}; a source, unit, and uncertainty are required"
        ))
    })?;
    let owner = name;
    let value = required_number(item, "value", owner)?;
    let declared = required_text(item, "unit", owner)?;
    let uncertainty = required_number(item, "uncertainty", owner)?;
    let source = required_text(item, "source", owner)?;
    let calibration = required_text(item, "calibration", owner)?;
    if declared != unit {
        return Err(ScfError::model_not_closed(format!(
            "{owner}.unit must be {unit}"
        )));
    }
    if source.trim().is_empty() || calibration.trim().is_empty() {
        return Err(ScfError::model_not_closed(format!(
            "{owner} is missing its source or calibration"
        )));
    }
    let provenance = crate::scf_simulator::geometry_update::smooth::classify_provenance(
        item.get("provenance").and_then(|entry| entry.as_str()),
        &source,
        &calibration,
    )?;
    if !uncertainty.is_finite() || uncertainty < 0.0 {
        return Err(ScfError::model_not_closed(format!(
            "{owner}.uncertainty must be finite and non-negative"
        )));
    }
    Ok(SourcedRecord {
        value,
        uncertainty,
        calibration,
        provenance,
    })
}

fn project_coefficients(
    value: &toml::Value,
    law: &crate::scf_simulator::geometry_update::smooth::SlopeLaw,
) -> Result<Vec<crate::scf_simulator::history_review::Coefficient>, ScfError> {
    let mut out = Vec::new();
    for item in [
        &law.grain_diameter,
        &law.friction_angle,
        &law.critical_slope,
        &law.spin_rate,
        &law.gravitational_constant,
        &law.timestep,
        &law.calibration,
    ] {
        out.push(crate::scf_simulator::history_review::Coefficient {
            name: item.name.clone(),
            value: item.value,
            uncertainty: item.uncertainty,
            unit: item.unit.clone(),
            provenance: item.provenance.clone(),
        });
    }
    let density = section(value, "density")?;
    let material = section(value, "material")?;
    for (table, name, unit) in [
        (density, "mass", "kg"),
        (density, "bulk_density", "kg/m^3"),
        (density, "grain_density", "kg/m^3"),
        (density, "porosity", "1"),
        (material, "cohesion", "Pa"),
        (material, "mu_s", "1"),
        (material, "mu_y", "1"),
        (material, "mu_2", "1"),
        (material, "i_0", "1"),
        (material, "grain_diameter", "m"),
        (material, "central_pressure", "Pa"),
        (material, "eta_c", "Pa*s"),
        (material, "p_min", "Pa"),
        (material, "delta_y", "1"),
        (material, "gamma_eps", "1/s"),
    ] {
        let record = sourced_record(table, name, unit)?;
        out.push(crate::scf_simulator::history_review::Coefficient {
            name: name.into(),
            value: record.value,
            uncertainty: record.uncertainty,
            unit: unit.into(),
            provenance: record.provenance,
        });
    }
    if let Some(mode) = value.get("mode_c") {
        for (name, unit) in [
            ("erosion_coefficient", "1"),
            ("external_mass_flux", "kg/s"),
            ("stress_scale", "1"),
        ] {
            let record = sourced_record(mode, name, unit)?;
            out.push(crate::scf_simulator::history_review::Coefficient {
                name: name.into(),
                value: record.value,
                uncertainty: record.uncertainty,
                unit: unit.into(),
                provenance: record.provenance,
            });
        }
    }
    if let Some(stokes) = value.get("stokes") {
        for (name, unit) in [("gamma", "1"), ("shear_floor", "1/s")] {
            if stokes.get(name).is_some() {
                let record = sourced_record(stokes, name, unit)?;
                out.push(crate::scf_simulator::history_review::Coefficient {
                    name: name.into(),
                    value: record.value,
                    uncertainty: record.uncertainty,
                    unit: unit.into(),
                    provenance: record.provenance,
                });
            }
        }
    }
    Ok(out)
}

fn required_number(table: &toml::Value, key: &str, owner: &str) -> Result<f64, ScfError> {
    table
        .get(key)
        .and_then(|item| {
            item.as_float()
                .or_else(|| item.as_integer().map(|n| n as f64))
        })
        .filter(|value| value.is_finite())
        .ok_or_else(|| ScfError::model_not_closed(format!("missing {owner}.{key}")))
}

fn required_text(table: &toml::Value, key: &str, owner: &str) -> Result<String, ScfError> {
    table
        .get(key)
        .and_then(|item| item.as_str())
        .map(|text| text.to_string())
        .ok_or_else(|| ScfError::model_not_closed(format!("missing {owner}.{key}")))
}

fn audit_kernel_provenance(
    kernel: &toml::Value,
    index: usize,
    center: [f64; 3],
    weight: f64,
    sigma: f64,
    alpha: f64,
) -> Result<(), ScfError> {
    let owner = format!("density.kernels[{index}]");
    let source = required_text(kernel, "source", &owner)?;
    let calibration = required_text(kernel, "calibration", &owner)?;
    if source.trim().is_empty() || calibration.trim().is_empty() {
        return Err(ScfError::model_not_closed(format!(
            "{owner} is missing its source or calibration"
        )));
    }
    let center_unit = required_text(kernel, "center_unit", &owner)?;
    let sigma_unit = required_text(kernel, "sigma_unit", &owner)?;
    let weight_unit = required_text(kernel, "weight_unit", &owner)?;
    let alpha_unit = required_text(kernel, "alpha_unit", &owner)?;
    if center_unit != "m" || sigma_unit != "1/m" || weight_unit != "kg/m^3" || alpha_unit != "1" {
        return Err(ScfError::model_not_closed(format!(
            "{owner} units must be center m, sigma 1/m, weight kg/m^3, alpha 1"
        )));
    }
    let center_uncertainty = required_number(kernel, "center_uncertainty", &owner)?;
    let sigma_uncertainty = required_number(kernel, "sigma_uncertainty", &owner)?;
    let weight_uncertainty = required_number(kernel, "weight_uncertainty", &owner)?;
    let alpha_uncertainty = required_number(kernel, "alpha_uncertainty", &owner)?;
    if center_uncertainty < 0.0
        || sigma_uncertainty < 0.0
        || weight_uncertainty < 0.0
        || alpha_uncertainty < 0.0
    {
        return Err(ScfError::model_not_closed(format!(
            "{owner} uncertainties must be non-negative"
        )));
    }
    if center.iter().any(|value| !value.is_finite()) || !(sigma > 0.0) || !(alpha > 0.0) {
        return Err(ScfError::model_not_closed(format!(
            "{owner} center, sigma, and alpha are not a usable SI kernel"
        )));
    }
    let _ = weight;
    Ok(())
}

fn audit_material_calibration(
    material: &toml::Value,
    law: &crate::scf_simulator::geometry_update::smooth::SlopeLaw,
    mu_s: f64,
    mu_2: f64,
    i_0: f64,
    grain_diameter: f64,
    cohesion: f64,
    eta_c: f64,
) -> Result<(), ScfError> {
    let mu_s_record = sourced_record(material, "mu_s", "1")?;
    let mu_y_record = sourced_record(material, "mu_y", "1")?;
    let mu_2_record = sourced_record(material, "mu_2", "1")?;
    let i_0_record = sourced_record(material, "i_0", "1")?;
    let grain = sourced_record(material, "grain_diameter", "m")?;
    let cohesion_record = sourced_record(material, "cohesion", "Pa")?;
    let eta = sourced_record(material, "eta_c", "Pa*s")?;
    let pressure = sourced_record(material, "central_pressure", "Pa")?;
    let mu_y = mu_y_record.value;
    for (name, scalar, record) in [
        ("mu_s", mu_s, &mu_s_record),
        ("mu_y", mu_y, &mu_y_record),
        ("mu_2", mu_2, &mu_2_record),
        ("i_0", i_0, &i_0_record),
        ("grain_diameter", grain_diameter, &grain),
        ("cohesion", cohesion, &cohesion_record),
        ("eta_c", eta_c, &eta),
    ] {
        if (scalar - record.value).abs() > 0.0 {
            return Err(ScfError::model_not_closed(format!(
                "material.{name} and its sourced value disagree"
            )));
        }
    }
    if (grain.value - law.grain_diameter.value).abs()
        > grain.uncertainty + law.grain_diameter.uncertainty
    {
        return Err(ScfError::model_not_closed(
            "material.grain_diameter is outside the sourced smoothing grain diameter",
        ));
    }
    if (mu_y - mu_s).abs() > mu_y_record.uncertainty + mu_s_record.uncertainty {
        return Err(ScfError::model_not_closed(
            "material.mu_y is outside the sourced mu_s uncertainty",
        ));
    }
    let rate_gap = (mu_2 - mu_s).abs();
    if rate_gap > mu_s_record.uncertainty + mu_2_record.uncertainty
        && !i_0_record
            .calibration
            .to_ascii_lowercase()
            .contains("inertial")
    {
        return Err(ScfError::model_not_closed(
            "mu_2 differs from mu_s without an inertial-number calibration for i_0",
        ));
    }
    if pressure.value <= 0.0 {
        return Err(ScfError::model_not_closed(
            "material.central_pressure must be positive",
        ));
    }
    let floor = sourced_record(material, "p_min", "Pa")?;
    if floor.value <= 0.0 || floor.value >= 1.0e-3 * pressure.value {
        return Err(ScfError::model_not_closed(
            "material.p_min must stay positive and below 0.001 of the measured central pressure",
        ));
    }
    let rate_floor = sourced_record(material, "gamma_eps", "1/s")?;
    if rate_floor.value <= 0.0 || rate_floor.value >= 0.1 * law.spin_rate.value {
        return Err(ScfError::model_not_closed(
            "material.gamma_eps must stay positive and below 0.1 of the measured spin rate",
        ));
    }
    let softening = sourced_record(material, "delta_y", "1")?;
    if softening.value != 0.0 {
        return Err(ScfError::model_not_closed(
            "no Ryugu softening measurement; material.delta_y must be 0",
        ));
    }
    let predicted = mu_s * pressure.value / law.spin_rate.value;
    if (eta_c - predicted).abs() > eta.uncertainty {
        return Err(ScfError::model_not_closed(
            "material.eta_c is outside mu_s * central_pressure / spin_rate",
        ));
    }
    Ok(())
}

fn audit_density_calibration(
    table: &toml::Value,
    density: &CauchyDensity,
    _law: &crate::scf_simulator::geometry_update::smooth::SlopeLaw,
) -> Result<(), ScfError> {
    let bulk_density = sourced_record(table, "bulk_density", "kg/m^3")?;
    let grain = sourced_record(table, "grain_density", "kg/m^3")?;
    let porosity = sourced_record(table, "porosity", "1")?;
    if grain.value <= 0.0 {
        return Err(ScfError::model_not_closed(
            "density.grain_density must be positive",
        ));
    }
    let center = density
        .kernels
        .first()
        .map(|kernel| kernel.center)
        .unwrap_or(DVec3::ZERO);
    let rho = density.rho(center);
    if (rho - bulk_density.value).abs() > bulk_density.uncertainty {
        return Err(ScfError::model_not_closed(
            "the density at the first kernel center is outside the sourced bulk density",
        ));
    }
    let predicted = 1.0 - bulk_density.value / grain.value;
    if (porosity.value - predicted).abs() > porosity.uncertainty {
        return Err(ScfError::model_not_closed(
            "density.porosity is outside 1 - bulk_density/grain_density",
        ));
    }
    Ok(())
}

fn audit_spin_horizon(
    table: &toml::Value,
    spin: &crate::scf_simulator::spin::SpinSchedule,
    law: &crate::scf_simulator::geometry_update::smooth::SlopeLaw,
) -> Result<(), ScfError> {
    let crate::scf_simulator::spin::SpinSchedule::Stages(stages) = spin else {
        return Ok(());
    };
    let holds_measured = stages.iter().all(|stage| {
        let rate = stage.omega[0].hypot(stage.omega[1]).hypot(stage.omega[2]);
        (rate - law.spin_rate.value).abs() <= law.spin_rate.uncertainty
    });
    if !holds_measured {
        return Ok(());
    }
    let horizon = sourced_record(table, "time", "s")?;
    let last = stages.last().map(|stage| stage.t_end).unwrap_or(0.0);
    if (last - horizon.value).abs() > horizon.uncertainty {
        return Err(ScfError::model_not_closed(
            "spin_schedule stage t_end is outside the sourced time",
        ));
    }
    Ok(())
}

fn audit_enclosed_mass(
    mesh: &TriMesh,
    density: &CauchyDensity,
    mass_target: f64,
    mass_uncertainty: f64,
) -> Result<(), ScfError> {
    let (mass, _) = crate::rtfp_solver::volume::closed_mass_moments(mesh, density)
        .map_err(|detail| ScfError::model_not_closed(detail))?;
    let gap = (mass - mass_target).abs();
    if gap > mass_uncertainty {
        return Err(ScfError::model_not_closed(format!(
            "enclosed mass {mass} kg differs from mass_target {mass_target} kg by {gap} kg, outside the sourced uncertainty {mass_uncertainty} kg; density and the mesh were not rescaled"
        )));
    }
    Ok(())
}

fn material_coefficient(material: &toml::Value, name: &str, unit: &str) -> Result<f64, ScfError> {
    Ok(sourced_record(material, name, unit)?.value)
}

fn read_eta(value: &toml::Value) -> Result<f64, ScfError> {
    if value.as_table().is_some() {
        required_number(value, "value", "eta_c")
    } else {
        finite_number(value, "material.eta_c")
    }
}

fn reject_require_nonnegative_flag(table: &toml::Value) -> Result<(), ScfError> {
    if let Some(flag) = table.get("require_nonnegative") {
        if flag.as_bool().is_none() {
            return Err(ScfError::model_not_closed(
                "density.require_nonnegative must be a boolean",
            ));
        }
    }
    Ok(())
}

fn reject_kernel_kind(kernel: &toml::Value, index: usize) -> Result<(), ScfError> {
    if let Some(kind) = kernel.get("kind").or_else(|| kernel.get("type")) {
        let name = kind.as_str().unwrap_or("");
        if !matches!(name, "cauchy" | "cauchy_kernel") {
            return Err(ScfError::model_not_closed(format!(
                "illegal density kernel at density.kernels[{index}]"
            )));
        }
    }
    Ok(())
}

fn require_nonnegative_density(density: &CauchyDensity, mesh: &TriMesh) -> Result<(), ScfError> {
    let mut points = vec![DVec3::ZERO];
    points.extend(mesh.vertices.iter().copied());
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        points.push((a + b + c) * (1.0 / 3.0));
        points.push((a + b + c) * (1.0 / 6.0));
    }
    for point in points {
        let rho = density.rho(point);
        if !rho.is_finite() || rho < 0.0 {
            return Err(ScfError::model_not_closed(format!(
                "density is negative at ({}, {}, {}); require_nonnegative failed",
                point.x, point.y, point.z
            )));
        }
    }
    Ok(())
}

fn parse_material(material: &toml::Value) -> Result<(f64, f64, f64, f64, f64), ScfError> {
    let cohesion = material_coefficient(material, "cohesion", "Pa")?;
    let mu_s = material_coefficient(material, "mu_s", "1")?;
    let mu_y = material_coefficient(material, "mu_y", "1")?;
    let mu_2 = material_coefficient(material, "mu_2", "1")?;
    let eta = material_viscosity(material)?;
    let gamma_eps = sourced_record(material, "gamma_eps", "1/s")?.value;
    let p_min = sourced_record(material, "p_min", "Pa")?.value;
    let delta_y = sourced_record(material, "delta_y", "1")?.value;
    if cohesion < 0.0 || mu_s < 0.0 || mu_y < 0.0 || mu_2 < 0.0 {
        return Err(ScfError::model_not_closed(
            "material cohesion, mu_s, mu_y, and mu_2 must be non-negative",
        ));
    }
    if eta <= 0.0 || gamma_eps <= 0.0 || p_min <= 0.0 {
        return Err(ScfError::model_not_closed(
            "material eta, gamma_eps, and p_min must be positive",
        ));
    }
    if !(0.0..1.0).contains(&delta_y) {
        return Err(ScfError::model_not_closed(
            "material delta_y must lie in [0, 1)",
        ));
    }
    if let Some(value) = material.get("young_modulus") {
        let young = finite_number(value, "material.young_modulus")?;
        if young <= 0.0 {
            return Err(ScfError::model_not_closed(
                "material.young_modulus must be positive",
            ));
        }
    }
    if let Some(value) = material.get("poisson_ratio") {
        let nu = finite_number(value, "material.poisson_ratio")?;
        if !(-1.0 < nu && nu < 0.5) {
            return Err(ScfError::model_not_closed(
                "material.poisson_ratio must lie in (-1, 1/2)",
            ));
        }
    }
    if let Some(value) = material.get("hardening_modulus") {
        let hardening = finite_number(value, "material.hardening_modulus")?;
        if hardening < 0.0 {
            return Err(ScfError::model_not_closed(
                "material.hardening_modulus must be non-negative",
            ));
        }
    }
    if let Some(value) = material.get("flow_rule") {
        if value
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .is_none()
        {
            return Err(ScfError::model_not_closed(
                "material.flow_rule must be a name",
            ));
        }
    }
    Ok((cohesion, mu_y, eta, p_min, delta_y))
}

fn parse_mode_c(table: &toml::Value) -> Result<(f64, f64, f64), ScfError> {
    let erosion = sourced_record(table, "erosion_coefficient", "1")?;
    let flux = sourced_record(table, "external_mass_flux", "kg/s")?;
    let scale = sourced_record(table, "stress_scale", "1")?;
    if erosion.value != 0.0 || flux.value != 0.0 {
        return Err(ScfError::model_not_closed(
            "no closed Ryugu mass-loss measurement; erosion_coefficient and external_mass_flux stay 0",
        ));
    }
    if scale.value != 1.0 {
        return Err(ScfError::model_not_closed(
            "mode_c.stress_scale has no measured normalization and stays 1",
        ));
    }
    Ok((erosion.value, flux.value, scale.value))
}

fn material_viscosity(material: &toml::Value) -> Result<f64, ScfError> {
    let eta = material
        .get("eta")
        .map(|value| finite_number(value, "material.eta"))
        .transpose()?;
    let eta_c = material.get("eta_c").map(read_eta).transpose()?;
    let chosen = match (eta, eta_c) {
        (None, None) => return Err(ScfError::model_not_closed("missing material.eta")),
        (Some(eta), None) => eta,
        (None, Some(eta_c)) | (Some(_), Some(eta_c)) => eta_c,
    };
    if chosen <= 0.0 || eta.is_some_and(|value| value <= 0.0) {
        return Err(ScfError::model_not_closed(
            "material eta, gamma_eps, and p_min must be positive",
        ));
    }
    Ok(chosen)
}

fn audit_document(value: &toml::Value) -> Result<(), ScfError> {
    audit_node(None, value)
}

fn audit_node(key: Option<&str>, value: &toml::Value) -> Result<(), ScfError> {
    if let Some(number) = value.as_float() {
        if !number.is_finite() {
            return Err(ScfError::model_not_closed(format!(
                "{} is not finite",
                node_label(key)
            )));
        }
    }
    if let Some(text) = value.as_str() {
        if is_nonfinite_literal(text) {
            return Err(ScfError::model_not_closed(format!(
                "{} is not finite",
                node_label(key)
            )));
        }
    }
    if let Some(items) = value.as_array() {
        for item in items {
            if item.as_str().is_some() {
                return Err(ScfError::model_not_closed(format!(
                    "{} is not numeric",
                    node_label(key)
                )));
            }
            audit_node(key, item)?;
        }
    }
    if let Some(table) = value.as_table() {
        for (child, child_value) in table {
            check_order_key(child, child_value)?;
            audit_node(Some(child), child_value)?;
        }
    }
    Ok(())
}

fn check_order_key(key: &str, value: &toml::Value) -> Result<(), ScfError> {
    let name = key.to_ascii_lowercase();
    if !matches!(name.as_str(), "l_max" | "k" | "k_p" | "galerkin_order") {
        return Ok(());
    }
    let Some(number) = value
        .as_float()
        .or_else(|| value.as_integer().map(|i| i as f64))
    else {
        return Err(ScfError::model_not_closed(format!(
            "{key} must be an integer >= 1"
        )));
    };
    if !number.is_finite() || number.fract() != 0.0 || number < 1.0 || number > u32::MAX as f64 {
        return Err(ScfError::model_not_closed(format!(
            "{key} must be an integer >= 1"
        )));
    }
    Ok(())
}

fn is_nonfinite_literal(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "nan" | "+nan" | "-nan" | "inf" | "+inf" | "-inf" | "infinity" | "+infinity" | "-infinity"
    )
}

fn node_label(key: Option<&str>) -> &str {
    key.unwrap_or("value")
}
