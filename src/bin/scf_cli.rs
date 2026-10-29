use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use astroid_evolution::common::error::SimError;
use astroid_evolution::scf::evolve::{
    self, RunOptions, State, StepFields, StepRecord, compose_frame,
};
use astroid_evolution::scf::forward::Forward;
use astroid_evolution::scf::forward::Loading;
use astroid_evolution::scf::frame::{self, departure};
use astroid_evolution::scf::io::{
    Snapshot, initial_body, read_snapshot, write_obj, write_snapshot, write_vertices_csv,
};
use astroid_evolution::scf::mechanics;
use astroid_evolution::scf::params::Project;
use astroid_evolution::scf::relax::relax;
use astroid_evolution::scf::spin::SpinSchedule;
use astroid_evolution::scf::timing::{self, Aggregate};

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
        /// Evaluate the deep state on every step instead of reusing it within the lag budget.
        #[arg(long)]
        no_lag: bool,
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
        /// Also report the unnormalized C20 and C22 at this reference radius (m).
        #[arg(long)]
        r_ref: Option<f64>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Also write the per-point fields (vertices, faces, volume points) as JSON.
        #[arg(long)]
        fields: Option<PathBuf>,
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
    /// Reconstruct the pre-evolution state of a tidally decaying moon: fit the
    /// present density to the mass and C20, then iterate X0 <- X0 + (X_p - F(X0))
    /// over a synchronous orbit that decays to `a_now` at `--years`.
    History {
        #[arg(long)]
        project: PathBuf,
        /// Spin schedule from the start of the history to today (any `--spin` grammar).
        #[arg(long)]
        spin: String,
        /// Fit the present density to this mass (kg) and unnormalized C20 at `--r-ref` (m)
        /// with a central core; without them the project density is used as it is.
        #[arg(long, requires_all = ["c20", "r_ref", "core_sigma"])]
        mass: Option<f64>,
        #[arg(long)]
        c20: Option<f64>,
        #[arg(long)]
        r_ref: Option<f64>,
        /// Inverse width (1/m) and exponent of the central core.
        #[arg(long)]
        core_sigma: Option<f64>,
        #[arg(long, default_value_t = 2.0)]
        core_alpha: f64,
        #[arg(long, default_value_t = 4)]
        iterations: usize,
        /// RMS radius misfit (m) that ends the iteration.
        #[arg(long, default_value_t = 0.5)]
        tol: f64,
        /// Damping β of the fixed-point update, in (0, 1].
        #[arg(long, default_value_t = 1.0)]
        damping: f64,
        #[arg(long)]
        out_dir: PathBuf,
        /// Also write the project with the fitted present-day density here.
        #[arg(long)]
        write_project: Option<PathBuf>,
    },
}

fn initial_state(project: &Project, snapshot: Option<&Path>) -> Result<State, SimError> {
    match snapshot {
        Some(p) => {
            let snap = read_snapshot(p)?;
            if project.body.shape == "ellipsoid" && snap.level != project.body.level {
                return Err(SimError::model_not_closed(format!(
                    "snapshot level {} differs from body.level {}",
                    snap.level, project.body.level
                )));
            }
            snap.to_state()
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

fn run(cli: Cli) -> Result<(), SimError> {
    match cli.command {
        Command::Validate { project } => {
            let p = Project::load(&project)?;
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
            let p = Project::load(&project)?;
            let (state, _) = evolve::start(&p, initial_body(&p)?)?;
            let (body, rep) = relax(&p, &state.body, Loading { omega, tide: 0.0 })?;
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
                    "  sweep {:<3} slack={:.3e} super={:<5} active={:<5} trust={:.2e} retries={} socp_it={} socp={:.2}s dM={:.2e} dmax={:.2e}",
                    s.sweep,
                    s.max_slack,
                    s.n_supercritical,
                    s.n_active,
                    s.trust,
                    s.trust_retries,
                    s.socp_iterations,
                    s.socp_seconds,
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
            write_snapshot(&out, &Snapshot::from_state(&next))?;
            if let Some(o) = obj {
                write_obj(&o, &next.body)?;
            }
        }
        Command::Realign {
            project,
            snapshot,
            out,
        } => {
            let p = Project::load(&project)?;
            let state = initial_state(&p, snapshot.as_deref())?;
            let (body, rep, _) = frame::realign(&state.body, None, p.g, &p.geometry, true)?;
            println!(
                "realign: passes={} angle={:.4e} deg  dM/M={:.2e} dV/V={:.2e}",
                rep.passes,
                rep.angle_deg,
                rep.mass_change,
                rep.volume_change
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
            write_snapshot(&out, &Snapshot::from_state(&next))?;
        }
        Command::RunSim {
            project,
            spin,
            resume,
            relax_first,
            max_steps,
            until,
            no_lag,
            out_dir,
            checkpoint_every,
            quiet,
        } => {
            let p = Project::load(&project)?;
            let schedule = SpinSchedule::parse(&spin)?;
            schedule.validate()?;
            std::fs::create_dir_all(&out_dir)
                .map_err(|e| SimError::io(format!("{}: {e}", out_dir.display())))?;
            timing::set_phase("startup");
            let wall = std::time::Instant::now();
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
                let (body, rep) = relax(
                    &p,
                    &state.body,
                    Loading {
                        omega: omega0,
                        tide: schedule.tide(0.0),
                    },
                )?;
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
                state.body.r_mean(),
                p.geometry.eps_deg,
            );
            println!(
                "start: t={:.5e} mass={:.6e} kg  frame departure centroid={:.1e} offdiag={:.1e}",
                state.t, fwd.moments.mass, dep.centroid, dep.off_diagonal
            );
            let log_path = out_dir.join("steps.jsonl");
            let mut log = String::new();
            let ck = out_dir.join("checkpoint.json");
            let n_vertices = state.body.topo.n_vertices();
            let n_faces = state.body.topo.n_faces();
            let opts = RunOptions { max_steps, until, lag: !no_lag };
            timing::set_phase("step");
            let mut obs =
                |r: &StepRecord, s: &State, _fields: &StepFields| -> Result<(), SimError> {
                    if !quiet {
                        println!("{}", line(r));
                    }
                    log.push_str(&serde_json::to_string(r)?);
                    log.push('\n');
                    if checkpoint_every > 0 && s.step.is_multiple_of(checkpoint_every) {
                        write_snapshot(&ck, &Snapshot::from_state(s))?;
                        std::fs::write(&log_path, &log)
                            .map_err(|e| SimError::io(format!("{}: {e}", log_path.display())))?;
                    }
                    Ok(())
                };
            let result = evolve::run(&p, &schedule, state, fwd, &opts, &mut obs);
            let samples = timing::drain();
            let wall_seconds = wall.elapsed().as_secs_f64();
            if !samples.is_empty() {
                write_timing(
                    &out_dir.join("timing.json"),
                    &project,
                    &spin,
                    n_vertices,
                    n_faces,
                    wall_seconds,
                    &samples,
                    result.as_ref().ok(),
                )?;
            }
            std::fs::write(&log_path, &log)
                .map_err(|e| SimError::io(format!("{}: {e}", log_path.display())))?;
            let result = result?;
            write_snapshot(
                &out_dir.join("final.json"),
                &Snapshot::from_state(&result.state),
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
        Command::History {
            project,
            mass,
            c20,
            r_ref,
            core_sigma,
            core_alpha,
            spin,
            iterations,
            tol,
            damping,
            out_dir,
            write_project,
        } => {
            use astroid_evolution::scf::history::{self, GravityTarget, InversionOptions};
            let p = Project::load(&project)?;
            std::fs::create_dir_all(&out_dir)?;
            let (state, _) = evolve::start(&p, initial_body(&p)?)?;
            let (present, fit) = match (mass, c20, r_ref, core_sigma) {
                (Some(mass), Some(c20), Some(r_ref), Some(sigma)) => {
                    let target = GravityTarget {
                        mass,
                        c20,
                        r_ref,
                        sigma,
                        alpha: core_alpha,
                    };
                    let (present, fit) = history::fit_core(&state.body, p.g, &target)?;
                    println!(
                        "present density: bulk {:.1} kg/m3, core weight {:.1}, centre {:.1} kg/m3; C20 homogeneous {:.4} -> {:.4}, C22 {:.4}",
                        fit.bulk,
                        fit.weight,
                        fit.centre_density,
                        fit.homogeneous.c20,
                        fit.fitted.c20,
                        fit.fitted.c22
                    );
                    if let Some(path) = &write_project {
                        write_present_project(&project, &p, &fit, &target, path)?;
                        println!("present project -> {}", path.display());
                    }
                    (present, Some(fit))
                }
                _ => (state.body, None),
            };
            let schedule = SpinSchedule::parse(&spin)?;
            schedule.validate()?;
            // The present state opens a future run, so its clock starts at zero.
            write_snapshot(
                &out_dir.join("present.json"),
                &Snapshot::at_rest(&present, p.body.level),
            )?;
            write_obj(&out_dir.join("present.obj"), &present)?;
            let mut obs = |k: usize, r: &StepRecord, _s: &State, _f: &StepFields| {
                println!("iter {k} {}", line(r));
                Ok(())
            };
            let inv = history::invert(
                &p,
                &schedule,
                &present,
                &InversionOptions {
                    max_iterations: iterations,
                    tol,
                    damping,
                },
                &mut obs,
            )?;
            for st in &inv.steps {
                println!(
                    "iteration {}: misfit rms {:.3} m max {:.3} m, departure {:.3} m, {} steps ({} geometry), {:.1}s",
                    st.iteration,
                    st.rms,
                    st.max,
                    st.departure,
                    st.steps,
                    st.geometry_steps,
                    st.seconds
                );
            }
            write_snapshot(
                &out_dir.join("initial.json"),
                &Snapshot::at_rest(&inv.initial, p.body.level),
            )?;
            write_obj(&out_dir.join("initial.obj"), &inv.initial)?;
            let report = serde_json::json!({
                "spin": spin,
                "fit": fit,
                "iterations": inv.steps,
                "converged": inv.converged,
            });
            std::fs::write(
                out_dir.join("history.json"),
                serde_json::to_string_pretty(&report)?,
            )?;
            println!(
                "history: converged={} -> {}",
                inv.converged,
                out_dir.display()
            );
        }
        Command::Inspect {
            project,
            snapshot,
            omega,
            r_ref,
            out,
            fields,
        } => {
            let p = Project::load(&project)?;
            let state = initial_state(&p, snapshot.as_deref())?;
            let mut fwd = Forward::bind(&state.body, p.g)?;
            let m = mechanics::evaluate(
                &mut Default::default(),
                &p,
                &state.body,
                &mut fwd,
                Loading { omega, tide: 0.0 },
                None,
                Default::default(),
            )?;
            let dep = departure(
                &fwd.moments,
                state.body.r_mean(),
                p.geometry.eps_deg,
            );
            let inward: Vec<f64> = m
                .g_face
                .iter()
                .zip(&m.rheo)
                .map(|(g, node)| -g.dot(node.normal))
                .collect();
            let area = state.body.vertex_areas();
            let rho_total: f64 = (0..area.len()).map(|i| area[i] * state.body.rho_s[i]).sum();
            let report = serde_json::json!({
                "omega": omega,
                "mass": fwd.moments.mass,
                "volume": fwd.moments.volume,
                "centroid": fwd.moments.centroid.to_array(),
                "frame_departure": {
                    "centroid": dep.centroid, "off_diagonal": dep.off_diagonal, "ordering": dep.ordering
                },
                "nitsche_c": m.ops.c_nitsche,
                "volume_points": m.ops.vol.pts.len(),
                "cut_rays": m.ops.vol.cut_rays,
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
                // Binding of the surface: −g_eff·n at the face centres (positive inward). A
                // non-positive value is a face the spin would shed (§2.8b, lift-off).
                "surface_binding": {
                    "min_inward": inward.iter().copied().fold(f64::INFINITY, f64::min),
                    "mean_inward": inward.iter().sum::<f64>() / inward.len().max(1) as f64,
                    "unbound_faces": inward.iter().filter(|g| **g <= 0.0).count(),
                },
                "rho_s_total": rho_total,
                "degree2": r_ref.map(|r| astroid_evolution::scf::history::degree2(&fwd.moments, r)),
            });
            let text = serde_json::to_string_pretty(&report)?;
            println!("{text}");
            if let Some(path) = fields {
                let v3 = |v: &[astroid_evolution::common::math::DVec3]| -> Vec<[f64; 3]> {
                    v.iter().map(|x| x.to_array()).collect()
                };
                let faces = state.body.face_geometry();
                let dump = serde_json::json!({
                    "vertices": {
                        "position": v3(&state.body.positions()),
                        "f_delta": m.class.f_delta,
                        "v_sl": v3(&m.class.v_sl),
                        "movable": m.class.movable,
                    },
                    "faces": {
                        "centre": faces.iter().map(|f| f.0.to_array()).collect::<Vec<_>>(),
                        "g_eff": v3(&m.g_face),
                        "phi": m.rheo.iter().map(|n| n.phi).collect::<Vec<_>>(),
                        "slack": m.rheo.iter().map(|n| n.slack).collect::<Vec<_>>(),
                        "shear": m.rheo.iter().map(|n| n.shear).collect::<Vec<_>>(),
                        "p_tilde": m.rheo.iter().map(|n| n.p_tilde).collect::<Vec<_>>(),
                        "velocity": m.rheo.iter().map(|n| n.velocity.to_array()).collect::<Vec<_>>(),
                        "stress": m.surface_stress,
                    },
                    "volume": {
                        "point": v3(&m.ops.vol.pts),
                        "weight": m.ops.vol.w,
                        "eta": m.sol.eta,
                        "p_tilde": m.sol.p_tilde,
                    },
                    "stokes": {
                        "u": m.sol.u,
                        "lambda": m.sol.lambda,
                    },
                });
                std::fs::write(&path, serde_json::to_string(&dump)?)
                    .map_err(|e| SimError::io(format!("{}: {e}", path.display())))?;
            }
            if let Some(o) = out {
                std::fs::write(&o, text)
                    .map_err(|e| SimError::io(format!("{}: {e}", o.display())))?;
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
                    return Err(SimError::model_not_closed(format!(
                        "export format must be obj or csv, got {other}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn write_timing(
    path: &Path,
    project: &Path,
    spin: &str,
    n_vertices: usize,
    n_faces: usize,
    wall_seconds: f64,
    samples: &[Aggregate],
    result: Option<&astroid_evolution::scf::evolve::RunResult>,
) -> Result<(), SimError> {
    let total: f64 = samples.iter().map(|s| s.seconds).sum();
    let group_seconds = |group: &str| -> f64 {
        samples
            .iter()
            .filter(|s| timing::group(s.name) == group)
            .map(|s| s.seconds)
            .sum()
    };
    let modules: Vec<_> = samples
        .iter()
        .map(|s| {
            serde_json::json!({
                "phase": s.phase,
                "module": s.name,
                "seconds": s.seconds,
                "calls": s.calls,
                "share": if total > 0.0 { s.seconds / total } else { 0.0 },
                "group": timing::group(s.name),
            })
        })
        .collect();
    let steps = result.map(|r| {
        r.records
            .iter()
            .map(|rec| {
                serde_json::json!({
                    "step": rec.step,
                    "kind": rec.kind,
                    "t": rec.t,
                    "dt": rec.dt,
                    "dt_bound": rec.dt_bound,
                    "omega": rec.omega,
                    "n_mobile": rec.n_mobile,
                    "n_deposit": rec.n_deposit,
                    "n_locked": rec.n_locked,
                    "n_supercritical_faces": rec.n_supercritical_faces,
                    "picard_iterations": rec.picard_iterations,
                    "socp_iterations": rec.socp_iterations,
                    "trust_retries": rec.trust_retries,
                    "rheo_seconds": rec.rheo_seconds,
                    "rheo_unconverged": rec.rheo_unconverged,
                    "slope_slack": rec.slope_slack,
                    "mass_slack": rec.mass_slack,
                    "max_displacement": rec.max_displacement,
                })
            })
            .collect::<Vec<_>>()
    });
    let report = serde_json::json!({
        "project": project.display().to_string(),
        "spin": spin,
        "vertices": n_vertices,
        "faces": n_faces,
        "wall_seconds": wall_seconds,
        "accounted_seconds": total,
        "groups": {
            "RTFP": group_seconds("RTFP"),
            "rheology": group_seconds("rheology"),
            "outer_socp": group_seconds("outer_socp"),
            "other": group_seconds("other"),
        },
        "modules": modules,
        "steps": steps,
    });
    let text = serde_json::to_string_pretty(&report)?;
    std::fs::write(path, &text).map_err(|e| SimError::io(format!("{}: {e}", path.display())))?;
    println!("timing report -> {}", path.display());
    println!(
        "groups: RTFP {:.3}s   rheology {:.3}s   outer SOCP {:.3}s   other {:.3}s   wall {:.3}s",
        group_seconds("RTFP"),
        group_seconds("rheology"),
        group_seconds("outer_socp"),
        group_seconds("other"),
        wall_seconds
    );
    println!("modules, longest first:");
    for (i, s) in samples.iter().enumerate() {
        println!(
            "  {:>2}. {:>8.3}s  {:>3} calls  {:<8} {}",
            i + 1,
            s.seconds,
            s.calls,
            s.phase,
            s.name
        );
    }
    Ok(())
}

/// The project text with `[density]` replaced by the fitted matrix and core. The core
/// sits at the centre of mass of the uniform body, written in the mesh input frame.
fn write_present_project(
    source: &Path,
    p: &Project,
    fit: &astroid_evolution::scf::history::CoreFit,
    target: &astroid_evolution::scf::history::GravityTarget,
    out: &Path,
) -> Result<(), SimError> {
    use astroid_evolution::io::{ImportRequest, load_mesh_file};
    use astroid_evolution::scf::io::mesh_body;
    let mesh_path = p.resolve(
        p.body
            .mesh
            .as_deref()
            .ok_or_else(|| SimError::model_not_closed("history needs a mesh body"))?,
    );
    let doc = load_mesh_file(ImportRequest {
        path: &mesh_path,
        length_unit: p.body.mesh_unit.as_deref(),
        reconstruct: None,
    })?;
    let (mut body, origin) = mesh_body(p, &doc.mesh.vertices, doc.mesh.faces.clone())?;
    body.kernels.clear();
    body.bulk = 1.0;
    let c = Forward::bind(&body, p.g)?.moments.centroid + origin;
    let text = std::fs::read_to_string(source)?;
    let start = text
        .find("[density]")
        .ok_or_else(|| SimError::model_not_closed("no [density] section"))?;
    let end = text[start + 1..]
        .find("\n[")
        .map_or(text.len(), |i| start + 2 + i);
    let density = format!(
        "[density]\n# Fitted by scf-cli history: mass {:.4e} kg, C20 {:.4} at {} m (homogeneous {:.4}).\nbulk = {:.3}\nkernels = [\n  {{ center = [{:.3}, {:.3}, {:.3}], weight = {:.3}, sigma = {:e}, alpha = {:?} }},\n]\n\n",
        target.mass,
        fit.fitted.c20,
        target.r_ref,
        fit.homogeneous.c20,
        fit.bulk,
        c.x,
        c.y,
        c.z,
        fit.weight,
        target.sigma,
        target.alpha
    );
    let mut out_text = format!("{}{}{}", &text[..start], density, &text[end..]);
    // Keep the mesh reachable from the new location.
    if let (Some(rel), Some(dir)) = (p.body.mesh.as_deref(), out.parent()) {
        let abs = std::fs::canonicalize(&mesh_path)?;
        let dir = std::fs::canonicalize(if dir.as_os_str().is_empty() {
            Path::new(".")
        } else {
            dir
        })?;
        let r = pathdiff(&abs, &dir);
        out_text = out_text.replace(&format!("mesh = \"{rel}\""), &format!("mesh = \"{r}\""));
    }
    Ok(std::fs::write(out, out_text)?)
}

/// `target` relative to `base` (both absolute).
fn pathdiff(target: &Path, base: &Path) -> String {
    let t: Vec<_> = target.components().collect();
    let b: Vec<_> = base.components().collect();
    let common = t.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<String> = vec!["..".into(); b.len() - common];
    parts.extend(
        t[common..]
            .iter()
            .map(|c| c.as_os_str().to_string_lossy().into_owned()),
    );
    parts.join("/")
}

fn main() -> ExitCode {
    SimError::exit(run(Cli::parse()))
}
