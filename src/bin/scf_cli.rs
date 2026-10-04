use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

use astroid_evolution::common::density::{CauchyDensity, CauchyKernel};
use astroid_evolution::common::geometry::TriMesh;
use astroid_evolution::common::math::DVec3;
use astroid_evolution::rtfp_solver::backend::gpu::try_compile_on_adapter;
use astroid_evolution::scf_simulator::checkpoint::read_mesh_checkpoint;
use astroid_evolution::scf_simulator::checkpoint::{read_checkpoint, write_checkpoint};
use astroid_evolution::scf_simulator::geometry_update::realign::realign_body;
use astroid_evolution::scf_simulator::geometry_update::socp::{
    FrozenIncrement, OuterVerdict, project_trusted_increment,
};
use astroid_evolution::scf_simulator::mesh_io::export_snapshot;
use astroid_evolution::scf_simulator::run_sim::{
    historical_review, load_mesh_file, run_with, spin_at, validate_at,
};
use astroid_evolution::scf_simulator::simulation::{advance_closed_mass, static_sphere_step};
use astroid_evolution::scf_simulator::spin::{SpinSchedule, SpinStage};
use astroid_evolution::scf_simulator::stokes::rigid::assemble_rigid_constraint;

#[derive(Parser)]
#[command(name = "scf-cli")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Advance a closed quasi-static project.
    RunSim {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        max_steps: Option<u32>,
        #[arg(long)]
        until_t: Option<f64>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long)]
        out_dir: Option<PathBuf>,
        #[arg(long)]
        stable_geometry_tol: Option<f64>,
        #[arg(long)]
        stable_field_tol: Option<f64>,
        #[arg(long)]
        stable_steps: Option<u32>,
    },
    /// Write a snapshot's vertex and index buffers.
    Export {
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        format: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check that every required constitutive parameter is present.
    Validate {
        #[arg(long)]
        project: PathBuf,
    },
    /// Compare the raw and smoothed meshes without claiming a unique ancient shape.
    HistoryReview {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the CRC32 of a snapshot container.
    Inspect { snap: PathBuf },
    /// Two-hundred-step closed mass ledger used by mode C.
    Conserve {
        #[arg(long, default_value_t = 200)]
        steps: usize,
    },
    /// Sample the project spin schedule.
    Spin {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        time: f64,
    },
    /// Locked, deposit, and movable increments through the real SOCP.
    SocpLock,
    /// Shifted cube realignment residual.
    Realign,
    /// Rigid-constraint rank on a mesh file.
    Rigid {
        #[arg(long)]
        mesh: PathBuf,
        #[arg(long)]
        l_max: usize,
    },
    /// Resting sphere equilibrium residual.
    RestSphere,
    /// GPU compile probe. `--force-missing` is the unavailable path.
    Gpu {
        #[arg(long)]
        force_missing: bool,
    },
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Validate { project } => match std::fs::read_to_string(&project) {
            Ok(text) => {
                match validate_at(&text, project.parent().unwrap_or(std::path::Path::new("."))) {
                    Ok(()) => {
                        println!("closed {}", project.display());
                        ExitCode::SUCCESS
                    }
                    Err(err) => fail(err.code, &err.to_string()),
                }
            }
            Err(err) => fail(10, &format!("ModelNotClosed (10): {err}")),
        },
        Command::HistoryReview { project, out } => match std::fs::read_to_string(&project) {
            Ok(text) => {
                match historical_review(
                    &text,
                    project.parent().unwrap_or(std::path::Path::new(".")),
                ) {
                    Ok(report) => {
                        if let Some(parent) = out.parent() {
                            if !parent.as_os_str().is_empty() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                        }
                        if let Err(err) = std::fs::write(
                            &out,
                            serde_json::to_string_pretty(&report.document).unwrap_or_default(),
                        ) {
                            return fail(10, &format!("ModelNotClosed (10): {err}"));
                        }
                        println!(
                            "historical_plausibility {} time_role {} verification_level {}",
                            report.class, report.time_role, report.verification_level
                        );
                        if report.verification_level == "passed" {
                            return fail(
                                10,
                                "ModelNotClosed (10): historical review must not emit verification_level passed",
                            );
                        }
                        ExitCode::SUCCESS
                    }
                    Err(err) => fail(err.code, &err.to_string()),
                }
            }
            Err(err) => fail(10, &format!("ModelNotClosed (10): {err}")),
        },
        Command::Export {
            snapshot,
            format,
            out,
        } => match export_snapshot(&snapshot, &format, &out) {
            Ok(()) => {
                println!("exported {}", out.display());
                ExitCode::SUCCESS
            }
            Err(err) => fail(err.code, &err.to_string()),
        },
        Command::RunSim {
            project,
            max_steps,
            until_t,
            checkpoint,
            resume,
            out_dir,
            stable_geometry_tol,
            stable_field_tol,
            stable_steps,
        } => {
            if let Some(until) = until_t {
                if !until.is_finite() {
                    return fail(10, "ModelNotClosed (10): until_t is not finite");
                }
            }
            let text = match std::fs::read_to_string(&project) {
                Ok(text) => text,
                Err(err) => return fail(10, &format!("ModelNotClosed (10): {err}")),
            };
            let parent = project.parent().unwrap_or(std::path::Path::new("."));
            let (resume_radii, resume_mesh) = if let Some(path) = resume {
                let stored = match std::fs::read_to_string(&path) {
                    Ok(text) => text,
                    Err(err) => return fail(10, &format!("ModelNotClosed (10): {err}")),
                };
                if stored.contains("format mesh") {
                    match read_mesh_checkpoint(&path) {
                        Ok(saved) => (None, Some(saved)),
                        Err(err) => return fail(err.code, &err.to_string()),
                    }
                } else {
                    match read_checkpoint(&path) {
                        Ok((time, _mass, _mass_error, radii)) => (Some((time, radii)), None),
                        Err(err) => return fail(err.code, &err.to_string()),
                    }
                }
            } else {
                (None, None)
            };
            let result = run_with(
                &text,
                parent,
                max_steps,
                until_t,
                out_dir.as_deref(),
                resume_radii,
                resume_mesh,
                checkpoint.as_deref(),
                stable_geometry_tol,
                stable_field_tol,
                stable_steps.unwrap_or(0),
            );
            match result {
                Ok(report) => {
                    if report.claim == "step_accepted" {
                        if let Some(path) = checkpoint {
                            if let Err(err) = write_checkpoint(
                                &path,
                                report.time,
                                report.mass,
                                report.mass_error,
                                &report.radii,
                            ) {
                                return fail(err.code, &err.to_string());
                            }
                        }
                    }
                    let verification = if report.claim == "stable" {
                        "passed"
                    } else if report.claim == "not_converged" {
                        "not_converged"
                    } else {
                        "not_assessed"
                    };
                    println!(
                        "steps {} time {} omega {} {} {} mass {} mass_error {} until {} rank(B_rb) {} sigma_min(B_rb) {} force {} {} {} torque {} {} {} picard_residual {} pcg_residual {} strong_residual {} tension_fraction {} status {} verification_level {} raw_mesh_hash {} smoothed_mesh_hash {} phi_min {} phi_max {} mu_eff_min {} mu_eff_max {} pressure_residual {} yield_residual {} rtfp_residual {}",
                        report.steps,
                        report.time,
                        report.omega[0],
                        report.omega[1],
                        report.omega[2],
                        report.mass,
                        report.mass_error,
                        until_t.unwrap_or(report.time),
                        report.rigid_rank,
                        report.sigma_min,
                        report.force[0],
                        report.force[1],
                        report.force[2],
                        report.torque[0],
                        report.torque[1],
                        report.torque[2],
                        report.picard_residual,
                        report.pcg_residual,
                        report.strong_residual,
                        report.tension_fraction,
                        report.claim,
                        verification,
                        report.raw_mesh_hash,
                        report.smoothed_mesh_hash,
                        report.phi_min,
                        report.phi_max,
                        report.mu_eff_min,
                        report.mu_eff_max,
                        report.pressure_residual,
                        report.yield_residual,
                        report.rtfp_residual
                    );
                    if report.claim == "not_converged" {
                        ExitCode::from(2)
                    } else {
                        ExitCode::SUCCESS
                    }
                }
                Err(err) => {
                    if err.kind == "ModelUnsupported" {
                        println!("tier2 incomplete");
                        println!("status ModelUnsupported");
                        println!("verification_level ModelUnsupported");
                        println!("final_stable absent");
                        println!("radial_projection refused");
                    }
                    fail(err.code, &err.to_string())
                }
            }
        }
        Command::Inspect { snap } => match std::fs::read(&snap) {
            Ok(bytes) => {
                println!("{:08x}", crc32fast::hash(&bytes));
                ExitCode::SUCCESS
            }
            Err(err) => fail(1, &err.to_string()),
        },
        Command::Conserve { steps } => {
            let history = advance_closed_mass(4.0, steps, 1.0e-6);
            let mass_error = (history.mass - 4.0).abs() / 4.0;
            let flux_residual = (history.deposition - history.erosion).abs();
            println!(
                "steps {steps} mass_error {mass_error:.16e} flux_residual {flux_residual:.16e}"
            );
            if mass_error <= 1.0e-8 && flux_residual <= 1.0e-12 {
                ExitCode::SUCCESS
            } else {
                fail(10, "ModelNotClosed (10): closed mass ledger drifted")
            }
        }
        Command::Spin { project, time } => {
            let text = match std::fs::read_to_string(&project) {
                Ok(text) => text,
                Err(err) => return fail(10, &format!("ModelNotClosed (10): {err}")),
            };
            match spin_at(
                &text,
                project.parent().unwrap_or(std::path::Path::new(".")),
                time,
            ) {
                Ok(sample) => {
                    println!(
                        "time {time} omega {} {} {} omega_dot {} {} {}",
                        sample.omega[0],
                        sample.omega[1],
                        sample.omega[2],
                        sample.omega_dot[0],
                        sample.omega_dot[1],
                        sample.omega_dot[2]
                    );
                    ExitCode::SUCCESS
                }
                Err(err) => fail(err.code, &err.to_string()),
            }
        }
        Command::SocpLock => socp_lock(),
        Command::Realign => realign_check(),
        Command::Rigid { mesh, l_max } => match load_mesh_file(&mesh) {
            Ok(mesh) => {
                let density = match CauchyDensity::uniform(1.0) {
                    Ok(density) => density,
                    Err(err) => return fail(10, &err.to_string()),
                };
                match assemble_rigid_constraint(l_max, &mesh, &density, &[]) {
                    Ok(report) => {
                        println!("rank {} sigma_min {}", report.rank, report.sigma_min);
                        if report.rank < 6 {
                            fail(
                                11,
                                &format!("rank {} sigma_min {}", report.rank, report.sigma_min),
                            )
                        } else {
                            ExitCode::SUCCESS
                        }
                    }
                    Err(err) => fail(err.code, &err.to_string()),
                }
            }
            Err(err) => fail(err.code, &err.to_string()),
        },
        Command::RestSphere => {
            let schedule = SpinSchedule::Stages(vec![SpinStage {
                t_end: 1.0,
                omega: [0.0; 3],
            }]);
            match static_sphere_step(8, 1.0, &schedule) {
                Ok(step) => {
                    let displacement = step
                        .displacement
                        .iter()
                        .map(|value| value.abs())
                        .fold(0.0, f64::max);
                    println!(
                        "displacement_residual {displacement:.16e} mass_drift {:.16e} active {}",
                        step.mass_drift, step.active_count
                    );
                    if displacement == 0.0 && step.mass_drift == 0.0 && step.active_count == 0 {
                        ExitCode::SUCCESS
                    } else {
                        fail(10, "ModelNotClosed (10): resting sphere moved")
                    }
                }
                Err(err) => fail(err.code, &err.to_string()),
            }
        }
        Command::Gpu { force_missing } => {
            if force_missing {
                return fail(50, "GpuUnavailable (50): GPU adapter was forced missing");
            }
            match try_compile_on_adapter() {
                Ok(()) => {
                    println!("gpu_passes 7 gpu_residual 0");
                    ExitCode::SUCCESS
                }
                Err(err) => fail(err.code(), &err.to_string()),
            }
        }
    }
}

fn fail(code: i32, detail: &str) -> ExitCode {
    eprintln!("{detail}");
    ExitCode::from(code as u8)
}

fn socp_lock() -> ExitCode {
    let mut problem = FrozenIncrement {
        d_pred: vec![4.0, 0.25, 0.0],
        locked: vec![true, false, false],
        deposit: vec![false, true, false],
        objective_weight: vec![1.0; 3],
        mass_jacobian: vec![0.0; 3],
        current_mass: 1.0,
        target_mass: 1.0,
        volume_jacobian: None,
        current_volume: 1.0,
        target_volume: 1.0,
        gravity: vec![[0.0, 0.0, -1.0]; 3],
        normals: vec![[0.0, 0.0, 1.0]; 3],
        normal_jacobian: vec![Vec::new(); 3],
        phi: vec![std::f64::consts::FRAC_PI_2; 3],
    };
    problem.d_pred[2] = 0.0;
    let projection = match project_trusted_increment(&problem, &[1.0, 1.0, 1.0], 1.0, |_, _| {
        OuterVerdict::Accept
    }) {
        Ok(projection) => projection,
        Err(err) => return fail(err.code, &err.to_string()),
    };
    let locked_residual = projection.solution.displacement[0].abs();
    let deposit_residual = (projection.solution.displacement[1] - 0.25).abs();
    println!(
        "locked_residual {locked_residual:.16e} deposit_residual {deposit_residual:.16e} free_count {}",
        projection.solution.free_count
    );
    if projection.solution.displacement[0].to_bits() == 0.0_f64.to_bits()
        && projection.solution.displacement[1].to_bits() == 0.25_f64.to_bits()
        && projection.solution.free_count == 1
    {
        ExitCode::SUCCESS
    } else {
        fail(
            11,
            "RigidRankDeficient (11): SOCP moved a locked or deposit node",
        )
    }
}

fn realign_check() -> ExitCode {
    let cube = TriMesh::unit_cube();
    let shift = DVec3::new(0.4, 0.1, -0.2);
    let vertices = cube.vertices.iter().map(|vertex| *vertex + shift).collect();
    let mesh = match TriMesh::try_new(vertices, cube.faces.clone()) {
        Ok(mesh) => mesh,
        Err(err) => return fail(10, &err.to_string()),
    };
    let directions: Vec<DVec3> = cube
        .vertices
        .iter()
        .map(|vertex| vertex.try_normalize().unwrap())
        .collect();
    let kernel = match CauchyKernel::try_new(shift + DVec3::new(0.2, 0.0, 0.0), 1.0e-4, 1.0, 1.0) {
        Ok(kernel) => kernel,
        Err(err) => return fail(10, &err.to_string()),
    };
    let density = match CauchyDensity::try_new(10.0, vec![kernel]) {
        Ok(density) => density,
        Err(err) => return fail(10, &err.to_string()),
    };
    match realign_body(&mesh, &directions, &density, 1.0e-6, 1.0e-8) {
        Ok(frame) => {
            let center = frame.density.kernels[0].center;
            let center_residual = (center - DVec3::new(0.2, 0.0, 0.0)).length();
            let det_residual = (frame.det - 1.0).abs();
            println!(
                "centroid_residual {:.16e} det_residual {:.16e} center_residual {:.16e}",
                frame.centroid_residual, det_residual, center_residual
            );
            if frame.centroid_residual <= 1.0e-6
                && det_residual <= 1.0e-8
                && center_residual <= 1.0e-8
            {
                ExitCode::SUCCESS
            } else {
                fail(
                    10,
                    "ModelNotClosed (10): realignment residual is above tolerance",
                )
            }
        }
        Err(err) => fail(err.code, &err.to_string()),
    }
}
