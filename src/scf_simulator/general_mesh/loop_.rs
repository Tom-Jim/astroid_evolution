//! Production general-mesh cycle: Tier1, pressure, Stokes–Picard, yield,
//! friction, a three-dimensional increment, and a second Tier1 on the candidate.
//! Radial Galerkin is not on this path.

use std::fs;
use std::path::Path;

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::api::{FieldBatch, FieldMask, PointRole, PointSet};
use crate::rtfp_solver::field::audit::{ErrorReport, production_rtfp_status};
use crate::rtfp_solver::radial::gauss_legendre::{
    FaceRule, coarser_face_rule, face_rule_for_directions, rules_differ,
};
use crate::rtfp_solver::{RtfpOptions, bind_and_eval_ray_rule};
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::mesh_io::write_glb;
use crate::scf_simulator::mesh_loop::{MeshJob, MeshOptions};
use crate::rheo_friction::closure::tidal_backstress;
use crate::scf_simulator::simulation::ClosedRun;
use crate::scf_simulator::stepping::timestep::{TimestepBounds, decide_timestep};

use super::geometry::{analyze, audit_candidate, displace, mass_row, star_margin};
use super::geometry_update::{
    VectorIncrement, align_frame, predict_displacement, solve_increment, trust_scale,
};
use super::rheology::{PicardLoad, PicardMaterial, picard};
use super::stokes::sample_velocity;
use super::volume::{TetMesh, integrate_density, tetrahedralize};
use super::yielding::{classify, surface_friction};

pub fn execute(job: &MeshJob, options: &MeshOptions) -> Result<ClosedRun, ScfError> {
    let mut mesh = if let Some(saved) = &options.resume {
        if saved.vertices.len() != job.mesh.vertices.len()
            || saved.faces.len() != job.mesh.faces.len()
        {
            return Err(ScfError::model_not_closed(
                "checkpoint vertex or face count does not match the general mesh",
            ));
        }
        TriMesh::try_new(saved.vertices.clone(), saved.faces.clone())
            .map_err(|err| ScfError::model_not_closed(err.detail))?
    } else {
        job.mesh.clone()
    };
    let mut density = job.density.clone();
    let mut time = options
        .resume
        .as_ref()
        .map(|saved| saved.time)
        .unwrap_or(0.0);
    if let Some(dir) = &options.out_dir {
        fs::create_dir_all(dir).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
        write_pair(dir, "raw_input", &job.raw_mesh)?;
        write_pair(dir, "smoothed_initial", &job.mesh)?;
    }
    let steps = options.max_steps.unwrap_or(if options.until_t.is_some() {
        u32::MAX
    } else {
        1
    });
    let mut stable_run = 0u32;
    let mut claim = "not_converged".to_string();
    let mut report = empty_report(job, time);
    let mut series = Vec::new();
    let mut diagnostics = Vec::new();
    for step in 0..steps {
        if options.until_t.is_some_and(|end| time >= end) {
            break;
        }
        let spin = job.spin.sample(time)?;
        let outcome = geological_step(job, &mesh, &density, time, &spin.omega, step)?;
        diagnostics.push(outcome.diagnostic.clone());
        series.push(outcome.series.clone());
        report = outcome.run.clone();
        if outcome.accepted {
            mesh = outcome.mesh;
            density = outcome.density;
            time = outcome.time;
            if let Some(dir) = &options.out_dir {
                let snap = dir.join(format!("snap_{:06}", step + 1));
                fs::create_dir_all(&snap)
                    .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
                write_pair(&snap, "mesh", &mesh)?;
                fs::write(snap.join("state.json"), outcome.series.to_string())
                    .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
            }
            if outcome.stable_ready {
                stable_run += 1;
            } else {
                stable_run = 0;
            }
            claim = if options.stable_steps > 0 && stable_run >= options.stable_steps {
                "stable".to_string()
            } else {
                "step_accepted".to_string()
            };
        } else {
            stable_run = 0;
            claim = "not_converged".to_string();
            if outcome.hard_error.is_some() {
                break;
            }
        }
        if claim == "stable" {
            break;
        }
    }
    report.claim = claim.clone();
    report.time = time;
    if let Some(dir) = &options.out_dir {
        let mut lines = String::new();
        for row in &series {
            lines.push_str(&row.to_string());
            lines.push('\n');
        }
        fs::write(dir.join("series.jsonl"), lines)
            .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
        let mut document = serde_json::json!({
            "model": "general_mesh",
            "calls": diagnostics,
            "status": claim,
            "final_stable": false,
            "verification_level": if claim == "stable" { "passed" } else { "not_converged" },
            "star_diagnostic": {
                "shaped": job.star.shaped,
                "star_margin": job.star.star_margin,
                "require_star": job.require_star,
                "gates_rtfp": false,
            },
            "raw_mesh_hash": job.raw_hash,
            "smoothed_mesh_hash": job.smoothed_hash,
        });
        if claim == "stable" {
            write_pair(dir, "final", &mesh)?;
            document["final_stable"] = serde_json::json!(true);
            document["verification_level"] = serde_json::json!("passed");
            document["status"] = serde_json::json!("stable");
        } else if dir.join("final.obj").exists() || dir.join("final.glb").exists() {
            return Err(ScfError::model_not_closed(
                "a non-stable general-mesh run tried to keep a final mesh",
            ));
        }
        fs::write(
            dir.join("diagnostics.json"),
            serde_json::to_string_pretty(&document).unwrap_or_default(),
        )
        .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    }
    println!("model general_mesh");
    println!(
        "final_stable {}",
        if report.claim == "stable" {
            "true"
        } else {
            "false"
        }
    );
    Ok(report)
}

struct StepOutcome {
    accepted: bool,
    stable_ready: bool,
    hard_error: Option<String>,
    mesh: TriMesh,
    density: CauchyDensity,
    time: f64,
    run: ClosedRun,
    series: serde_json::Value,
    diagnostic: serde_json::Value,
}

fn geological_step(
    job: &MeshJob,
    mesh: &TriMesh,
    density: &CauchyDensity,
    time: f64,
    omega: &[f64; 3],
    step: u32,
) -> Result<StepOutcome, ScfError> {
    let geometry = analyze(mesh)?;
    let volume = tetrahedralize(mesh)?;
    let current_hash = crate::rtfp_solver::io::mesh_hash(mesh);
    let first = evaluate_tier1(job, mesh, density, &volume, step as u64 * 2 + 1)?;
    eprintln!(
        "tier1_call {} mesh_hash {} density_hash {} points {} status {} potential {} gradient {} hessian {}",
        first.id,
        first.mesh_hash,
        first.density_hash,
        first.points,
        first.status,
        fmt_opt(first.potential_error),
        fmt_opt(first.gradient_error),
        fmt_opt(first.hessian_error)
    );
    let samples = field_samples(mesh, &volume);
    let gravity = effective_gravity(job, mesh, density, &volume, &samples, *omega)?;
    let vertex_mass = lumped_mass(&volume, density);
    let tet_force: Vec<DVec3> = (0..volume.tets.len())
        .map(|tet| gravity.g_tet[tet] * gravity.rho_tet[tet])
        .collect();
    let quad_density: Vec<f64> = volume
        .quad_points
        .iter()
        .map(|point| density.rho(*point))
        .collect();
    let quad_gravity = map_quad(&volume, &gravity.g_tet);
    let tide_div = map_quad(&volume, &gravity.tide_div);
    let picard_state = picard(
        &volume,
        &PicardMaterial {
            cohesion: job.cohesion,
            mu_s: job.mu_s,
            mu_2: job.mu_2,
            i_0: job.i_0,
            grain_diameter: job.grain_diameter,
            rho_grain: job.rho_grain,
            p_min: job.p_min,
            gamma_eps: job.shear_floor,
            relaxation: job.under_relaxation,
            iterations: job.picard_iterations,
        },
        &PicardLoad {
            density: &quad_density,
            gravity: &quad_gravity,
            tide_divergence: &tide_div,
            vertex_mass: &vertex_mass,
            force: &tet_force,
        },
    )?;
    eprintln!(
        "stokes_linear {:.6e} divergence {:.6e} rigid {:.6e} pressure {:.6e} picard {:.6e} iterations {}",
        picard_state.stokes.linear_residual,
        picard_state.stokes.divergence_residual,
        picard_state.stokes.rigid_residual,
        picard_state.pressure.residual,
        picard_state.residual,
        picard_state.stokes.iterations
    );
    let surface_velocity =
        surface_speed(&volume, &picard_state.velocity, &picard_state.bubble, mesh);
    let tet_pressure = picard_state
        .pressure
        .tilde
        .iter()
        .enumerate()
        .map(|(vertex, _)| picard_state.pressure.tilde[vertex])
        .take(volume.tets.len().min(picard_state.pressure.tilde.len()))
        .collect::<Vec<_>>();
    let pressure_on_tet = average_on_tets(&volume, &picard_state.pressure.tilde);
    let tide = vec![[[0.0; 3]; 3]; volume.tets.len()];
    let yield_report = classify(
        &volume,
        &picard_state.velocity,
        &picard_state.bubble,
        &picard_state.eta,
        &pressure_on_tet,
        &tide,
        job.cohesion,
        job.mu_y,
        mesh.vertices.len(),
    )?;
    let (surface_yield, surface_core) =
        classify_surface(mesh, &volume, &yield_report.yielded, &yield_report.core);
    let mut slip = vec![DVec3::ZERO; mesh.vertices.len()];
    for (index, speed) in surface_velocity.iter().enumerate() {
        slip[index] = *speed;
        if surface_core[index] {
            slip[index] = DVec3::ZERO;
        }
    }
    let max_slip = slip
        .iter()
        .map(|value| value.length())
        .fold(0.0_f64, f64::max);
    let deposit = vec![false; mesh.vertices.len()];
    let movable: Vec<bool> = surface_core
        .iter()
        .zip(&surface_yield)
        .map(|(core, yield_flag)| !core && *yield_flag)
        .collect();
    let locked: Vec<bool> = movable.iter().map(|flag| !*flag).collect();
    let (phi, mu_face, _) = surface_friction(
        &mesh.faces,
        &volume,
        &picard_state.velocity,
        &picard_state.bubble,
        &pressure_on_tet,
        &movable,
        job.mu_s,
        job.mu_2,
        job.i_0,
        job.grain_diameter,
        job.rho_grain,
        &geometry,
    )?;
    let phi_finite: Vec<f64> = phi
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    let phi_min = phi_finite.iter().copied().fold(f64::INFINITY, f64::min);
    let phi_max = phi_finite.iter().copied().fold(0.0_f64, f64::max);
    let mu_min = mu_face.iter().copied().fold(f64::INFINITY, f64::min);
    let mu_max = mu_face.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rho_surface: Vec<f64> = mesh
        .vertices
        .iter()
        .map(|vertex| density.rho(*vertex).max(job.p_min))
        .collect();
    let dt = decide_timestep(TimestepBounds {
        cfl: job.cfl,
        coupling: job.coupling,
        characteristic: job.characteristic,
        spin: job.spin_dt,
        max: job.dt_max,
        until_next: 1.0e9,
        active_set_empty: movable.iter().all(|flag| !*flag),
    });
    let chi: Vec<f64> = surface_yield
        .iter()
        .map(|flag| if *flag { 1.0 } else { 0.0 })
        .collect();
    let delta_h = vec![0.0; mesh.vertices.len()];
    let d_pred = predict_displacement(
        &chi,
        dt,
        &slip,
        &geometry.vertex_normal,
        &delta_h,
        job.rho_bulk,
        &rho_surface,
        &locked,
    )?;
    let (mass_now, _, _) = integrate_density(&volume, |point| density.rho(point));
    let coeff = mass_row(&geometry, &rho_surface)?;
    let mut displacement = d_pred.clone();
    let mut kkt = f64::NAN;
    let mut clarabel = false;
    let mut accepted_mesh = None;
    let mut accepted_density = None;
    let mut second = None;
    let mut hard = None;
    let mut trust_rejects = 0u32;
    let length_scale = mesh
        .vertices
        .iter()
        .map(|v| v.length())
        .fold(1.0_f64, f64::max)
        .max(1.0);
    loop {
        let trust = match trust_scale(job.trust_radius, trust_rejects) {
            Ok(trust) => trust,
            Err(err) => {
                hard = Some(err.detail);
                break;
            }
        };
        let slope_node = movable.clone();
        let solved = match solve_increment(
            &VectorIncrement {
                d_pred: d_pred.clone(),
                locked: locked.clone(),
                deposit: deposit.clone(),
                weight: geometry.dual_area.clone(),
                mass_coeff: coeff.clone(),
                current_mass: mass_now,
                target_mass: mass_now,
                gravity: gravity.g_surface.clone(),
                normal: geometry.vertex_normal.clone(),
                normal_jacobian: geometry.normal_jacobian.clone(),
                phi: phi.clone(),
                slope_node,
            },
            trust,
        ) {
            Ok(solved) => solved,
            Err(err) => {
                hard = Some(err.detail);
                break;
            }
        };
        kkt = solved.kkt_residual;
        clarabel = solved.clarabel_called;
        displacement = solved.displacement;
        let candidate = match displace(mesh, &displacement) {
            Ok(candidate) => candidate,
            Err(_) => {
                trust_rejects += 1;
                continue;
            }
        };
        let (aligned, aligned_density, centroid_residual, _) =
            match align_frame(&candidate, density) {
                Ok(value) => value,
                Err(_) => (candidate, density.clone(), f64::NAN, f64::NAN),
            };
        let _ = centroid_residual;
        let aligned_volume = match tetrahedralize(&aligned) {
            Ok(value) => value,
            Err(_) => {
                trust_rejects += 1;
                continue;
            }
        };
        let (mass_next, _, _) =
            integrate_density(&aligned_volume, |point| aligned_density.rho(point));
        let audit = audit_candidate(
            mesh,
            &aligned,
            &displacement,
            mass_now,
            mass_next,
            job.mass_target.abs().max(1.0),
            trust.max(length_scale),
            &job.crater.protection,
        );
        let surface_mass_change: f64 = coeff
            .iter()
            .zip(&displacement)
            .map(|(row, step)| row.dot(*step))
            .sum();
        let mass_ok =
            surface_mass_change.abs() <= job.tolerances.mass_tol * mass_now.abs().max(1.0);
        let volume_now = mesh.signed_volume();
        let volume_next = aligned.signed_volume();
        let volume_ok = volume_next > 0.0
            && (volume_next - volume_now).abs()
                <= job.tolerances.mass_tol * volume_now.abs().max(1e-30);
        if !audit.topology_valid
            || audit.self_intersections > 0
            || audit.flipped_faces > 0
            || !mass_ok
            || !volume_ok
        {
            trust_rejects += 1;
            continue;
        }
        let tier = evaluate_tier1(
            job,
            &aligned,
            &aligned_density,
            &aligned_volume,
            step as u64 * 2 + 2,
        )?;
        eprintln!(
            "tier1_call {} mesh_hash {} density_hash {} points {} status {} potential {} gradient {} hessian {}",
            tier.id,
            tier.mesh_hash,
            tier.density_hash,
            tier.points,
            tier.status,
            fmt_opt(tier.potential_error),
            fmt_opt(tier.gradient_error),
            fmt_opt(tier.hessian_error)
        );
        second = Some(tier);
        accepted_mesh = Some(aligned);
        accepted_density = Some(aligned_density);
        break;
    }
    let second = second
        .unwrap_or_else(|| missing_tier(step as u64 * 2 + 2, &current_hash, &job.density_hash));
    let hashes_differ = second.mesh_hash != current_hash;
    let tier_ok = first.status == "PASS" && second.status == "PASS";
    let picard_ok =
        picard_state.converged || !job.require_convergence && picard_state.residual.is_finite();
    let picard_for_stable = picard_state.converged;
    let geometry_residual = displacement
        .iter()
        .map(|step| step.length())
        .fold(0.0_f64, f64::max)
        / length_scale;
    let field_residual = if first.gradient_error.is_some() && second.gradient_error.is_some() {
        first
            .gradient_error
            .unwrap_or(0.0)
            .max(second.gradient_error.unwrap_or(0.0))
    } else {
        f64::NAN
    };
    let mass_residual = (mass_now - job.mass_target).abs() / job.mass_target.abs().max(1.0);
    let volume_residual =
        (geometry.volume - job.initial_volume).abs() / job.initial_volume.abs().max(1e-30);
    let accepted = accepted_mesh.is_some()
        && tier_ok
        && picard_for_stable
        && hashes_differ
        && hard.is_none()
        && geometry_residual
            <= job
                .tolerances
                .geometry_tol
                .max(job.trust_radius / length_scale);
    let _ = (tet_pressure, picard_ok);
    let stable_ready = accepted
        && mass_residual <= job.tolerances.mass_tol
        && picard_state.stokes.divergence_residual <= job.tolerances.kkt_tol.max(1e-4)
        && picard_state.pressure.residual <= job.tolerances.kkt_tol.max(1e-2)
        && yield_report.fit_residual.is_finite();
    let geometry_ok = accepted_mesh.is_some();
    let next_mesh = accepted_mesh.unwrap_or_else(|| mesh.clone());
    let next_density = accepted_density.unwrap_or_else(|| density.clone());
    let next_time = if accepted { time + dt } else { time };
    let eta_char = picard_state.eta.iter().sum::<f64>() / picard_state.eta.len().max(1) as f64;
    let omega_mag = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let ekman_inverse =
        job.rho_bulk * omega_mag * length_scale * length_scale / eta_char.max(1e-30);
    let epsilon_euler = max_slip * dt / length_scale;
    let run = ClosedRun {
        steps: step + 1,
        time: next_time,
        omega: *omega,
        mass: mass_now,
        mass_error: mass_residual,
        radii: mesh.vertices.iter().map(|vertex| vertex.length()).collect(),
        rigid_rank: picard_state.stokes.rigid_rank,
        sigma_min: picard_state.stokes.sigma_min,
        force: [
            picard_state.stokes.rigid_multiplier[0],
            picard_state.stokes.rigid_multiplier[1],
            picard_state.stokes.rigid_multiplier[2],
        ],
        torque: [
            picard_state.stokes.rigid_multiplier[3],
            picard_state.stokes.rigid_multiplier[4],
            picard_state.stokes.rigid_multiplier[5],
        ],
        picard_residual: picard_state.residual,
        pcg_residual: picard_state.stokes.linear_residual,
        strong_residual: picard_state.stokes.divergence_residual,
        tension_fraction: picard_state.pressure.tension_fraction,
        claim: if accepted {
            "step_accepted"
        } else {
            "not_converged"
        }
        .to_string(),
        raw_mesh_hash: job.raw_hash.clone(),
        smoothed_mesh_hash: job.smoothed_hash.clone(),
        phi_min: if phi_min.is_finite() {
            phi_min
        } else {
            f64::NAN
        },
        phi_max: if phi_max.is_finite() {
            phi_max
        } else {
            f64::NAN
        },
        mu_eff_min: if mu_min.is_finite() { mu_min } else { f64::NAN },
        mu_eff_max: if mu_max.is_finite() { mu_max } else { f64::NAN },
        pressure_residual: picard_state.pressure.residual,
        yield_residual: yield_report.f_delta.iter().copied().fold(0.0_f64, f64::max),
        rtfp_residual: second
            .gradient_error
            .or(first.gradient_error)
            .unwrap_or(f64::NAN),
    };
    let series = serde_json::json!({
        "step": step,
        "accepted": accepted,
        "mesh_hash": current_hash,
        "candidate_mesh_hash": second.mesh_hash,
        "hashes_differ": hashes_differ,
        "geometry_residual": json_num(geometry_residual),
        "field_residual": json_num(field_residual),
        "mass_residual": json_num(mass_residual),
        "volume_residual": json_num(volume_residual),
        "momentum_residual": json_num(picard_state.stokes.rigid_residual),
        "divergence_residual": json_num(picard_state.stokes.divergence_residual),
        "pressure_residual": json_num(picard_state.pressure.residual),
        "yield_residual": json_num(run.yield_residual),
        "slope_residual": if clarabel { json_num(kkt) } else { serde_json::Value::Null },
        "rtfp_residual": json_num(run.rtfp_residual),
        "topology_valid": geometry_ok,
        "self_intersections": if geometry_ok { serde_json::json!(0) } else { serde_json::json!(null) },
        "picard_residual": json_num(picard_state.residual),
        "socp_residual": if clarabel { json_num(kkt) } else { serde_json::Value::Null },
        "clarabel_called": clarabel,
        "ekman_inverse": json_num(ekman_inverse),
        "epsilon_euler": json_num(epsilon_euler),
        "core_mass": json_num(yield_report.core_mass),
        "core_volume_fraction": json_num(yield_report.core_volume_fraction),
        "yield_volume_fraction": json_num(yield_report.yield_volume_fraction),
        "mobile_area_fraction": json_num(yield_report.mobile_area_fraction),
        "fit_residual": json_num(yield_report.fit_residual),
        "max_slip": json_num(max_slip),
        "tension_fraction": json_num(picard_state.pressure.tension_fraction),
        "star_margin": json_num(star_margin(mesh, DVec3::ZERO)),
        "time": next_time,
    });
    let diagnostic = serde_json::json!({
        "tier1_initial": first.to_json(),
        "tier1_candidate": second.to_json(),
        "pressure_range": [picard_state.pressure.p_min_value, picard_state.pressure.p_max],
        "tilde_range": [picard_state.pressure.tilde_min, picard_state.pressure.tilde_max],
        "initial_velocity": picard_state.pressure.initial_velocity,
        "stokes_iterations": picard_state.stokes.iterations,
        "diagonal_condition": picard_state.stokes.diagonal_condition,
        "sigma_min": picard_state.stokes.sigma_min,
        "rigid_rank": picard_state.stokes.rigid_rank,
        "velocity_finite": picard_state.stokes.velocity_finite,
        "pressure_finite": picard_state.stokes.pressure_finite,
        "hard_error": hard,
    });
    Ok(StepOutcome {
        accepted,
        stable_ready,
        hard_error: hard,
        mesh: next_mesh,
        density: next_density,
        time: next_time,
        run,
        series,
        diagnostic,
    })
}

struct FieldSamples {
    faces: Vec<DVec3>,
    volume: Vec<DVec3>,
    classify: Vec<DVec3>,
}

struct GravityField {
    g_tet: Vec<DVec3>,
    g_surface: Vec<DVec3>,
    rho_tet: Vec<f64>,
    tide_div: Vec<DVec3>,
}

struct Tier1Call {
    id: u64,
    mesh_hash: String,
    density_hash: String,
    points: usize,
    near: f64,
    fine: String,
    coarse: String,
    potential_error: Option<f64>,
    gradient_error: Option<f64>,
    hessian_error: Option<f64>,
    status: String,
}

impl Tier1Call {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "tier1_call_id": self.id,
            "mesh_hash": self.mesh_hash,
            "density_hash": self.density_hash,
            "points": self.points,
            "near_radius": self.near,
            "fine_rule": self.fine,
            "coarse_rule": self.coarse,
            "potential_error": self.potential_error,
            "gradient_error": self.gradient_error,
            "hessian_error": self.hessian_error,
            "status": self.status,
        })
    }
}

fn evaluate_tier1(
    job: &MeshJob,
    mesh: &TriMesh,
    density: &CauchyDensity,
    volume: &TetMesh,
    id: u64,
) -> Result<Tier1Call, ScfError> {
    let samples = field_samples(mesh, volume);
    let options = RtfpOptions {
        directions: job.directions.max(8),
        r0: job.near_radius,
        ell0: job.ell0,
        g: job.g,
        quadrature_nodes: job.radial_degree.max(2),
    };
    let fine = face_rule_for_directions(options.directions);
    let coarse = coarser_face_rule(fine);
    let mut potential = None;
    let mut gradient = None;
    let mut hessian = None;
    let mut points = 0usize;
    let mut completed = 0usize;
    for (points_set, role, want) in [
        (
            samples.faces.clone(),
            PointRole::SurfaceFace,
            FieldMask::ALL,
        ),
        (
            samples.volume.clone(),
            PointRole::VolumeQuad,
            FieldMask {
                potential: true,
                gradient: true,
                hessian: false,
            },
        ),
        (
            samples.classify.clone(),
            PointRole::ClassifyGrid,
            FieldMask::ALL,
        ),
    ] {
        if points_set.is_empty() {
            continue;
        }
        points += points_set.len();
        eprintln!(
            "tier1_eval id {id} role {role:?} points {} faces {}",
            points_set.len(),
            mesh.faces.len()
        );
        let (fine_batch, coarse_batch) = eval_pair(
            &options,
            fine,
            coarse,
            mesh,
            density,
            &points_set,
            role,
            want,
        )?;
        completed += points_set.len();
        potential = max_opt(
            potential,
            component_gap(
                &fine_batch.potential,
                &coarse_batch.potential,
                want.potential,
            ),
        );
        gradient = max_opt(
            gradient,
            vector_gap(&fine_batch.gradient, &coarse_batch.gradient, want.gradient),
        );
        hessian = max_opt(
            hessian,
            sym_gap(&fine_batch.hessian, &coarse_batch.hessian, want.hessian),
        );
    }
    let report = ErrorReport {
        points_completed: completed,
        points_requested: points,
        rules_differ: rules_differ(fine, coarse),
        max_potential: potential,
        max_gradient: gradient,
        max_hessian: hessian,
        want: FieldMask::ALL,
    };
    let status = if potential.is_none() && gradient.is_none() && hessian.is_none() {
        "NOT_ASSESSED"
    } else {
        production_rtfp_status(&report)
    };
    Ok(Tier1Call {
        id,
        mesh_hash: crate::rtfp_solver::io::mesh_hash(mesh),
        density_hash: job.density_hash.clone(),
        points,
        near: job.near_radius,
        fine: format!(
            "degree {} tol {:.3e} limit {}",
            fine.degree, fine.tol, fine.limit
        ),
        coarse: format!(
            "degree {} tol {:.3e} limit {}",
            coarse.degree, coarse.tol, coarse.limit
        ),
        potential_error: potential,
        gradient_error: gradient,
        hessian_error: hessian,
        status: status.to_string(),
    })
}

fn effective_gravity(
    job: &MeshJob,
    mesh: &TriMesh,
    density: &CauchyDensity,
    volume: &TetMesh,
    samples: &FieldSamples,
    omega: [f64; 3],
) -> Result<GravityField, ScfError> {
    let options = RtfpOptions {
        directions: job.directions.max(8),
        r0: job.near_radius,
        ell0: job.ell0,
        g: job.g,
        quadrature_nodes: job.radial_degree.max(2),
    };
    let fine = face_rule_for_directions(options.directions);
    let volume_field = eval_one(
        &options,
        fine,
        mesh,
        density,
        &samples.volume,
        PointRole::VolumeQuad,
        FieldMask {
            potential: false,
            gradient: true,
            hessian: false,
        },
    )?;
    let classify_field = eval_one(
        &options,
        fine,
        mesh,
        density,
        &samples.classify,
        PointRole::ClassifyGrid,
        FieldMask {
            potential: false,
            gradient: false,
            hessian: true,
        },
    )?;
    let mut g_samples = Vec::new();
    for (index, point) in samples.volume.iter().enumerate() {
        g_samples.push(centrifugal(volume_field.gradient[index], *point, omega));
    }
    let mut g_tet = vec![DVec3::ZERO; volume.tets.len()];
    let mut rho_tet = vec![0.0; volume.tets.len()];
    let mut tide_div = vec![DVec3::ZERO; volume.tets.len()];
    for tet in 0..volume.tets.len() {
        let center = centroid_tet(volume, tet);
        let nearest = nearest_index(&samples.volume, center);
        g_tet[tet] = g_samples[nearest];
        rho_tet[tet] = density.rho(center);
        let hessian =
            classify_field.hessian[nearest.min(classify_field.hessian.len().saturating_sub(1))];
        let h_w = [
            [hessian[0] + omega[2] * omega[2], hessian[3], hessian[4]],
            [hessian[3], hessian[1] + omega[2] * omega[2], hessian[5]],
            [hessian[4], hessian[5], hessian[2]],
        ];
        let tide = tidal_backstress(1.0, job.rho_grain, 1.0, job.grain_diameter, &h_w);
        tide_div[tet] = DVec3::new(
            tide[0][0] + tide[0][1] + tide[0][2],
            tide[1][0] + tide[1][1] + tide[1][2],
            tide[2][0] + tide[2][1] + tide[2][2],
        ) * (1.0 / job.initial_volume.abs().cbrt().max(1.0));
    }
    let mut g_surface = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let nearest = nearest_index(&samples.volume, *vertex);
        g_surface.push(g_samples[nearest]);
    }
    Ok(GravityField {
        g_tet,
        g_surface,
        rho_tet,
        tide_div,
    })
}

fn centrifugal(gradient: DVec3, point: DVec3, omega: [f64; 3]) -> DVec3 {
    let omega_sq = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2];
    gradient + DVec3::new(point.x, point.y, 0.0) * omega_sq
}

fn field_samples(mesh: &TriMesh, volume: &TetMesh) -> FieldSamples {
    let face_budget = if mesh.faces.len() > 64 {
        1
    } else {
        mesh.faces.len().min(4)
    };
    let tet_budget = if mesh.faces.len() > 64 {
        1
    } else {
        volume.tets.len().min(4)
    };
    let mut faces = Vec::new();
    for face in (0..mesh.faces.len()).step_by((mesh.faces.len() / face_budget.max(1)).max(1)) {
        let [a, b, c] = mesh.triangle(face);
        faces.push((a + b + c) * (1.0 / 3.0));
        if faces.len() == face_budget {
            break;
        }
    }
    let mut volume_points = Vec::new();
    for tet in (0..volume.tets.len()).step_by((volume.tets.len() / tet_budget.max(1)).max(1)) {
        volume_points.push(centroid_tet(volume, tet));
        if volume_points.len() == tet_budget {
            break;
        }
    }
    FieldSamples {
        faces,
        classify: volume_points.clone(),
        volume: volume_points,
    }
}

fn eval_pair(
    options: &RtfpOptions,
    fine: FaceRule,
    coarse: FaceRule,
    mesh: &TriMesh,
    density: &CauchyDensity,
    points: &[DVec3],
    role: PointRole,
    want: FieldMask,
) -> Result<(FieldBatch, FieldBatch), ScfError> {
    let fine_batch = eval_one(options, fine, mesh, density, points, role, want)?;
    let coarse_batch = eval_one(options, coarse, mesh, density, points, role, want)?;
    Ok((fine_batch, coarse_batch))
}

fn eval_one(
    options: &RtfpOptions,
    rule: FaceRule,
    mesh: &TriMesh,
    density: &CauchyDensity,
    points: &[DVec3],
    role: PointRole,
    want: FieldMask,
) -> Result<FieldBatch, ScfError> {
    bind_and_eval_ray_rule(
        options.clone(),
        rule,
        mesh,
        density.clone(),
        &PointSet::from_points(points),
        role,
        want,
    )
    .map_err(|err| ScfError::model_not_closed(err.detail))
}

fn lumped_mass(volume: &TetMesh, density: &CauchyDensity) -> Vec<f64> {
    let mut mass = vec![0.0; volume.vertices.len()];
    for (tet, &weight) in volume.volumes.iter().enumerate() {
        let center = centroid_tet(volume, tet);
        let rho = density.rho(center);
        for vertex in volume.tets[tet] {
            mass[vertex as usize] += rho * weight / 4.0;
        }
    }
    mass
}

fn surface_speed(
    volume: &TetMesh,
    velocity: &[DVec3],
    bubble: &[DVec3],
    mesh: &TriMesh,
) -> Vec<DVec3> {
    mesh.vertices
        .iter()
        .map(|vertex| {
            let tet = nearest_tet(volume, *vertex);
            sample_velocity(volume, velocity, bubble, tet, *vertex)
        })
        .collect()
}

fn classify_surface(
    mesh: &TriMesh,
    volume: &TetMesh,
    yielded: &[bool],
    core: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let mut surface_yield = vec![false; mesh.vertices.len()];
    let mut surface_core = vec![false; mesh.vertices.len()];
    for (index, vertex) in mesh.vertices.iter().enumerate() {
        let tet = nearest_tet(volume, *vertex);
        surface_yield[index] = yielded[tet];
        surface_core[index] = core[tet] && !yielded[tet];
    }
    (surface_yield, surface_core)
}

fn nearest_tet(volume: &TetMesh, point: DVec3) -> usize {
    let mut best = 0usize;
    let mut best_distance = f64::MAX;
    for tet in 0..volume.tets.len() {
        let distance = (centroid_tet(volume, tet) - point).length_squared();
        if distance < best_distance {
            best_distance = distance;
            best = tet;
        }
    }
    best
}

fn nearest_index(points: &[DVec3], query: DVec3) -> usize {
    let mut best = 0usize;
    let mut best_distance = f64::MAX;
    for (index, point) in points.iter().enumerate() {
        let distance = (*point - query).length_squared();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

fn centroid_tet(volume: &TetMesh, tet: usize) -> DVec3 {
    let p = volume.tets[tet].map(|index| volume.vertices[index as usize]);
    (p[0] + p[1] + p[2] + p[3]) * 0.25
}

fn map_quad(volume: &TetMesh, tet_values: &[DVec3]) -> Vec<DVec3> {
    volume.quad_tet.iter().map(|tet| tet_values[*tet]).collect()
}

fn average_on_tets(volume: &TetMesh, nodal: &[f64]) -> Vec<f64> {
    volume
        .tets
        .iter()
        .map(|tet| {
            tet.iter()
                .map(|vertex| nodal.get(*vertex as usize).copied().unwrap_or(0.0))
                .sum::<f64>()
                / 4.0
        })
        .collect()
}

fn write_pair(dir: &Path, name: &str, mesh: &TriMesh) -> Result<(), ScfError> {
    let document = crate::rtfp_solver::io::MeshDocument {
        mesh: mesh.clone(),
        length_unit: "m".into(),
        source_length_unit: "m".into(),
        mesh_hash: crate::rtfp_solver::io::mesh_hash(mesh),
        welded_positions: 0,
        dropped_degenerate: 0,
        triangulated_strips: 0,
        triangulated_fans: 0,
        source: name.into(),
    };
    crate::rtfp_solver::io::write_obj(&dir.join(format!("{name}.obj")), &document)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    write_glb(&dir.join(format!("{name}.glb")), mesh)
}

fn empty_report(job: &MeshJob, time: f64) -> ClosedRun {
    ClosedRun {
        steps: 0,
        time,
        omega: [0.0; 3],
        mass: job.mass_target,
        mass_error: f64::NAN,
        radii: Vec::new(),
        rigid_rank: 0,
        sigma_min: f64::NAN,
        force: [0.0; 3],
        torque: [0.0; 3],
        picard_residual: f64::NAN,
        pcg_residual: f64::NAN,
        strong_residual: f64::NAN,
        tension_fraction: f64::NAN,
        claim: "not_converged".into(),
        raw_mesh_hash: job.raw_hash.clone(),
        smoothed_mesh_hash: job.smoothed_hash.clone(),
        phi_min: f64::NAN,
        phi_max: f64::NAN,
        mu_eff_min: f64::NAN,
        mu_eff_max: f64::NAN,
        pressure_residual: f64::NAN,
        yield_residual: f64::NAN,
        rtfp_residual: f64::NAN,
    }
}

fn missing_tier(id: u64, mesh_hash: &str, density_hash: &str) -> Tier1Call {
    Tier1Call {
        id,
        mesh_hash: mesh_hash.to_string(),
        density_hash: density_hash.to_string(),
        points: 0,
        near: f64::NAN,
        fine: String::new(),
        coarse: String::new(),
        potential_error: None,
        gradient_error: None,
        hessian_error: None,
        status: "NOT_ASSESSED".into(),
    }
}

fn component_gap(fine: &[f64], coarse: &[f64], wanted: bool) -> Option<f64> {
    if !wanted || fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        let scale = left.abs().max(right.abs()).max(1e-30);
        max_gap = max_gap.max((left - right).abs() / scale);
    }
    Some(max_gap)
}

fn vector_gap(fine: &[DVec3], coarse: &[DVec3], wanted: bool) -> Option<f64> {
    if !wanted || fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        let scale = left.length().max(right.length()).max(1e-30);
        max_gap = max_gap.max((*left - *right).length() / scale);
    }
    Some(max_gap)
}

fn sym_gap(fine: &[[f64; 6]], coarse: &[[f64; 6]], wanted: bool) -> Option<f64> {
    if !wanted || fine.len() != coarse.len() || fine.is_empty() {
        return None;
    }
    let mut max_gap = 0.0_f64;
    for (left, right) in fine.iter().zip(coarse) {
        let mut scale = 1e-30_f64;
        let mut gap = 0.0_f64;
        for index in 0..6 {
            if !left[index].is_finite() || !right[index].is_finite() {
                return None;
            }
            scale = scale.max(left[index].abs()).max(right[index].abs());
            gap = gap.max((left[index] - right[index]).abs());
        }
        max_gap = max_gap.max(gap / scale);
    }
    Some(max_gap)
}

fn max_opt(current: Option<f64>, next: Option<f64>) -> Option<f64> {
    match (current, next) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn json_num(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else {
        serde_json::Value::Null
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.6e}"),
        _ => "unknown".into(),
    }
}

pub fn gate_accepts(tier_status: &str, picard_converged: bool, residuals_known: bool) -> bool {
    tier_status == "PASS" && picard_converged && residuals_known
}
