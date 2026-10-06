use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use astroid_evolution::rheo_friction::RheoGpu;
use astroid_evolution::scf_simulator::error::ScfError;
use astroid_evolution::scf_simulator::evolve::{
    self, RunOptions, State, StepRecord, compose_frame,
};
use astroid_evolution::scf_simulator::forward::Forward;
use astroid_evolution::scf_simulator::frame::{self, departure};
use astroid_evolution::scf_simulator::io::{
    Snapshot, initial_body, read_snapshot, write_obj, write_snapshot, write_vertices_csv,
};
use astroid_evolution::scf_simulator::mechanics;
use astroid_evolution::scf_simulator::params::Project;
use astroid_evolution::scf_simulator::relax::relax;
use astroid_evolution::scf_simulator::spin::SpinSchedule;

#[derive(Parser)]
#[command(
    name = "scf-cli",
    about = "Tier 2: quasi-static SCF evolution (Tier 1 gravity + rheo_friction)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check that the project is closed: every constant present and in range.
    Validate {
        #[arg(long)]
        project: PathBuf,
    },
    /// Sample a spin schedule.
    Spin {
        #[arg(long)]
        spin: String,
        #[arg(long)]
        time: f64,
    },
    /// Supercritical-slope relaxation with crater protection, then frame alignment.
    Relax {
        #[arg(long)]
        project: PathBuf,
        /// Spin rate (rad/s) of the relaxation.
        #[arg(long)]
        omega: f64,
        #[arg(long)]
        out: PathBuf,
        /// Also write an OBJ of the relaxed polyhedron.
        #[arg(long)]
        obj: Option<PathBuf>,
    },
    /// Rigidly re-align a snapshot (or the initial body) to the principal-axis frame.
    Realign {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        snapshot: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Quasi-static evolution under an exogenous spin schedule.
    RunSim {
        #[arg(long)]
        project: PathBuf,
        /// `stages:T1@w1,..` | `const:w@T` | `linear:w0,rate,T` | `table:t@w,..` | `table-file:PATH`
        #[arg(long)]
        spin: String,
        /// Start from this snapshot (restores time, step, frame and velocity).
        #[arg(long)]
        resume: Option<PathBuf>,
        /// Relax the initial body (at the first spin rate) before the run.
        #[arg(long)]
        relax_first: bool,
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(long)]
        until: Option<f64>,
        #[arg(long)]
        out_dir: PathBuf,
        /// Write `checkpoint.json` every N steps.
        #[arg(long, default_value_t = 10)]
        checkpoint_every: usize,
        #[arg(long)]
        quiet: bool,
    },
    /// One mechanics pass (Stokes, classification, friction chain) on a state; JSON report.
    Inspect {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        snapshot: Option<PathBuf>,
        #[arg(long)]
        omega: f64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Export a snapshot as OBJ or per-vertex CSV.
    Export {
        #[arg(long)]
        snapshot: PathBuf,
        /// `obj` | `csv`
        #[arg(long)]
        format: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Report the Metal device used by `rheo_friction`.
    Gpu,
}

fn fail(err: &ScfError) -> ExitCode {
    eprintln!("{err}");
    ExitCode::from(err.code().clamp(1, 255) as u8)
}

fn load(project: &Path) -> Result<Project, ScfError> {
    Project::load(project)
}

fn gpu() -> Result<RheoGpu, ScfError> {
    RheoGpu::new().map_err(ScfError::from)
}

fn snapshot_of(state: &State, level: usize) -> Snapshot {
    Snapshot::from_body(
        &state.body,
        level,
        state.t,
        state.step,
        state.omega,
        state.frame_rotation,
        state.frame_translation,
        state.velocity.clone(),
    )
}

fn state_from(snap: &Snapshot) -> Result<State, ScfError> {
    Ok(State {
        body: snap.to_body()?,
        level: snap.level,
        t: snap.time,
        step: snap.step,
        omega: snap.omega,
        frame_rotation: snap.frame_rotation,
        frame_translation: astroid_evolution::common::math::DVec3::from_array(
            snap.frame_translation,
        ),
        velocity: snap.velocity.clone(),
    })
}

fn initial_state(project: &Project, snapshot: Option<&Path>) -> Result<State, ScfError> {
    match snapshot {
        Some(p) => {
            let snap = read_snapshot(p)?;
            if snap.level != project.body.level {
                return Err(ScfError::model_not_closed(format!(
                    "snapshot level {} differs from body.level {}",
                    snap.level, project.body.level
                )));
            }
            state_from(&snap)
        }
        None => {
            let (state, _) = evolve::start(project, initial_body(project)?)?;
            Ok(state)
        }
    }
}

fn line(r: &StepRecord) -> String {
    format!(
        "n={:<5} t={:.5e} adv={:.3e} w={:.4e} {:<17} bound={:<14} picard={:<3} yield={:.3} mobile={:<5} \
         sup={:<4} slack={:.2e} dM={:.1e} trust={:.1e}/{} frame={}",
        r.step,
        r.t,
        r.advance,
        r.omega,
        r.kind,
        r.dt_bound,
        r.picard_iterations,
        r.yielded_fraction,
        r.n_mobile,
        r.n_supercritical_faces,
        r.max_slack,
        r.mass_residual,
        r.trust,
        r.trust_retries,
        if r.frame_applied { "yes" } else { "no" },
    )
}

fn run(cli: Cli) -> Result<(), ScfError> {
    match cli.command {
        Command::Validate { project } => {
            let p = load(&project)?;
            println!(
                "closed {}  level={} g={:e} l_surface={} l_velocity={}",
                project.display(),
                p.body.level,
                p.g,
                p.stokes.l_surface,
                p.stokes.l_velocity
            );
        }
        Command::Spin { spin, time } => {
            let s = SpinSchedule::parse(&spin)?;
            s.validate()?;
            let v = s.sample(time);
            println!(
                "t={time} omega={:e} omega_dot={:e} t_next={:e} t_end={:e}",
                v.omega,
                v.omega_dot,
                v.t_next,
                s.t_end()
            );
        }
        Command::Relax {
            project,
            omega,
            out,
            obj,
        } => {
            let p = load(&project)?;
            let gpu = gpu()?;
            let (state, _) = evolve::start(&p, initial_body(&p)?)?;
            let (body, rep) = relax(&p, &gpu, &state.body, omega)?;
            let (body, frep, _) = frame::realign(&body, None, p.g, &p.geometry, false)?;
            println!(
                "relax: converged={} stalled={} protection={} locked={} protected={} preserved_supercritical={} slack {:.3e} -> {:.3e} in {} sweeps",
                rep.converged,
                rep.stalled,
                rep.identification,
                rep.n_locked,
                rep.n_protected,
                rep.preserved_supercritical,
                rep.slack_before,
                rep.slack_after,
                rep.sweeps.len()
            );
            for s in &rep.sweeps {
                println!(
                    "  sweep {:<3} slack={:.3e} super={:<5} trust={:.2e} retries={} socp_it={} dM={:.2e} dmax={:.2e}",
                    s.sweep,
                    s.max_slack,
                    s.n_supercritical,
                    s.trust,
                    s.trust_retries,
                    s.socp_iterations,
                    s.mass_residual,
                    s.max_displacement
                );
            }
            println!(
                "frame: applied={} angle={:.3e} deg",
                frep.applied, frep.angle_deg
            );
            let mut next = state;
            next.body = body;
            next.omega = omega;
            let (rot, tr) = compose_frame(next.frame_rotation, next.frame_translation, &frep);
            next.frame_rotation = rot;
            next.frame_translation = tr;
            write_snapshot(&out, &snapshot_of(&next, p.body.level))?;
            if let Some(o) = obj {
                write_obj(&o, &next.body)?;
            }
        }
        Command::Realign {
            project,
            snapshot,
            out,
        } => {
            let p = load(&project)?;
            let state = initial_state(&p, snapshot.as_deref())?;
            let (body, rep, _) = frame::realign(&state.body, None, p.g, &p.geometry, true)?;
            println!(
                "realign: passes={} angle={:.4e} deg  dM/M={:.2e} dV/V={:.2e} max dR/R={:.2e}",
                rep.passes,
                rep.angle_deg,
                rep.mass_change,
                rep.volume_change,
                rep.max_radius_change
            );
            println!(
                "  before: centroid={:.2e} offdiag={:.2e} order={:.2e}\n  after:  centroid={:.2e} offdiag={:.2e} order={:.2e}",
                rep.before.centroid,
                rep.before.off_diagonal,
                rep.before.ordering,
                rep.after.centroid,
                rep.after.off_diagonal,
                rep.after.ordering
            );
            let mut next = state;
            next.body = body;
            let (rot, tr) = compose_frame(next.frame_rotation, next.frame_translation, &rep);
            next.frame_rotation = rot;
            next.frame_translation = tr;
            write_snapshot(&out, &snapshot_of(&next, p.body.level))?;
        }
        Command::RunSim {
            project,
            spin,
            resume,
            relax_first,
            max_steps,
            until,
            out_dir,
            checkpoint_every,
            quiet,
        } => {
            let p = load(&project)?;
            let schedule = SpinSchedule::parse(&spin)?;
            schedule.validate()?;
            let gpu = gpu()?;
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| ScfError::io(format!("{}: {e}", out_dir.display())))?;
            let (mut state, mut fwd) = match &resume {
                Some(path) => {
                    let st = initial_state(&p, Some(path))?;
                    let f = Forward::bind(&st.body, p.g)?;
                    (st, f)
                }
                None => evolve::start(&p, initial_body(&p)?)?,
            };
            if relax_first && resume.is_none() {
                let omega0 = schedule.sample(0.0).omega;
                let (body, rep) = relax(&p, &gpu, &state.body, omega0)?;
                println!(
                    "relax-first: converged={} slack {:.3e} -> {:.3e} ({} sweeps)",
                    rep.converged,
                    rep.slack_before,
                    rep.slack_after,
                    rep.sweeps.len()
                );
                let (body, _, f) = frame::realign(&body, None, p.g, &p.geometry, false)?;
                state.body = body;
                fwd = f;
            }
            let dep = departure(
                &fwd.moments,
                state.body.radius.iter().sum::<f64>() / state.body.radius.len() as f64,
                p.geometry.eps_deg,
            );
            println!(
                "start: t={:.5e} mass={:.6e} kg  frame departure centroid={:.1e} offdiag={:.1e}",
                state.t, fwd.moments.mass, dep.centroid, dep.off_diagonal
            );
            let level = p.body.level;
            let log_path = out_dir.join("steps.jsonl");
            let mut log = String::new();
            let ck = out_dir.join("checkpoint.json");
            let opts = RunOptions { max_steps, until };
            let mut obs = |r: &StepRecord, s: &State| -> Result<(), ScfError> {
                if !quiet {
                    println!("{}", line(r));
                }
                log.push_str(&serde_json::to_string(r).map_err(|e| ScfError::io(e.to_string()))?);
                log.push('\n');
                if checkpoint_every > 0 && s.step % checkpoint_every == 0 {
                    write_snapshot(&ck, &snapshot_of(s, level))?;
                    std::fs::write(&log_path, &log)
                        .map_err(|e| ScfError::io(format!("{}: {e}", log_path.display())))?;
                }
                Ok(())
            };
            let result = evolve::run(&p, &gpu, &schedule, state, fwd, &opts, &mut obs);
            std::fs::write(&log_path, &log)
                .map_err(|e| ScfError::io(format!("{}: {e}", log_path.display())))?;
            let result = result?;
            write_snapshot(
                &out_dir.join("final.json"),
                &snapshot_of(&result.state, level),
            )?;
            write_obj(&out_dir.join("final.obj"), &result.state.body)?;
            println!(
                "{}: t={:.6e} steps={} -> {}",
                if result.complete {
                    "complete"
                } else {
                    "stopped"
                },
                result.state.t,
                result.records.len(),
                out_dir.display()
            );
        }
        Command::Inspect {
            project,
            snapshot,
            omega,
            out,
        } => {
            let p = load(&project)?;
            let gpu = gpu()?;
            let state = initial_state(&p, snapshot.as_deref())?;
            let mut fwd = Forward::bind(&state.body, p.g)?;
            let m = mechanics::evaluate(&p, &gpu, &state.body, &mut fwd, omega, None)?;
            let dep = departure(
                &fwd.moments,
                state.body.radius.iter().sum::<f64>() / state.body.radius.len() as f64,
                p.geometry.eps_deg,
            );
            let area = frame::vertex_areas(&state.body);
            let rho_total: f64 = (0..area.len()).map(|i| area[i] * state.body.rho_s[i]).sum();
            let report = serde_json::json!({
                "omega": omega,
                "mass": fwd.moments.mass,
                "volume": fwd.moments.volume,
                "centroid": fwd.moments.centroid.to_array(),
                "frame_departure": {
                    "centroid": dep.centroid, "off_diagonal": dep.off_diagonal, "ordering": dep.ordering
                },
                "surface_fit_error": m.fit_error,
                "picard": {
                    "iterations": m.sol.picard_iterations, "eps_picard": m.sol.eps_picard,
                    "eps_momentum": m.sol.eps_momentum, "eps_lambda": m.sol.eps_lambda,
                    "eps_rot": m.sol.eps_rot, "eta_lo": m.sol.eta_lo, "eta_hi": m.sol.eta_hi,
                    "tensile_fraction": m.sol.tensile_fraction, "gamma_eps": m.sol.gamma_eps,
                },
                "classification": {
                    "yielded_fraction": m.class.yielded_fraction,
                    "mobile_area_fraction": m.class.mobile_area_fraction,
                    "core_mass_fraction": m.class.core_mass_fraction,
                    "core_degenerate": m.class.core_degenerate,
                    "core_empty": m.class.core_empty,
                    "core_creep": m.class.core_creep,
                    "plastic_dissipation": m.class.plastic_dissipation,
                    "inverse_ekman": m.class.inverse_ekman,
                    "n_mobile": m.class.movable.iter().filter(|x| **x).count(),
                },
                "friction_chain": {
                    "nodes": m.rheo.len(),
                    "unconverged": m.rheo_stats.unconverged,
                    "max_iterations": m.rheo_stats.max_iterations,
                    "seconds": m.rheo_stats.seconds,
                    "max_slack": m.rheo.iter().map(|n| n.slack).fold(0.0, f64::max),
                    "supercritical_faces": m.rheo.iter().filter(|n| n.slack > p.relief.tol_slack).count(),
                    "min_phi": m.rheo.iter().map(|n| n.phi).fold(f64::INFINITY, f64::min),
                    "max_phi": m.rheo.iter().map(|n| n.phi).fold(0.0, f64::max),
                },
                "rho_s_total": rho_total,
            });
            let text =
                serde_json::to_string_pretty(&report).map_err(|e| ScfError::io(e.to_string()))?;
            println!("{text}");
            if let Some(o) = out {
                std::fs::write(&o, text)
                    .map_err(|e| ScfError::io(format!("{}: {e}", o.display())))?;
            }
        }
        Command::Export {
            snapshot,
            format,
            out,
        } => {
            let body = read_snapshot(&snapshot)?.to_body()?;
            match format.as_str() {
                "obj" => write_obj(&out, &body)?,
                "csv" => write_vertices_csv(&out, &body)?,
                other => {
                    return Err(ScfError::model_not_closed(format!(
                        "export format must be obj or csv, got {other}"
                    )));
                }
            }
        }
        Command::Gpu => {
            println!("rheo_friction device: {}", gpu()?.device_name());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(&e),
    }
}
