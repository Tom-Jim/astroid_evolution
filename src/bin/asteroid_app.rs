use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use bevy::prelude::*;
use clap::{Args, Parser, Subcommand};

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::pipeline::{Evolver, SurfaceFrame, mesh_summary};
use astroid_evolution::asteroid_app::resources::AppConfig;
use astroid_evolution::scf_simulator::params::Project;
use astroid_evolution::scf_simulator::spin::SpinSchedule;

#[derive(Parser)]
#[command(
    name = "asteroid-app",
    about = "Tier 3: show the Tier 2 body for a project mesh at constant spin"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct Common {
    #[arg(long, default_value = "assets/projects/ryugu_cauchy.toml")]
    project: PathBuf,
    /// Spin rate (rad/s) of the default `const:` schedule.
    #[arg(long, default_value_t = 4.986655e-4)]
    omega: f64,
    /// Duration (s) of the default `const:` schedule.
    #[arg(long, default_value_t = 3.15576e13)]
    duration: f64,
    /// Full schedule (`stages:`, `const:`, `linear:`, `table:`, `table-file:`); overrides --omega/--duration.
    #[arg(long)]
    spin: Option<String>,
    #[arg(long)]
    max_steps: Option<usize>,
}

#[derive(Subcommand)]
enum Command {
    /// Window: loads the project mesh into Tier 2 and draws each new body.
    View(Common),
    /// Headless: the same Tier 2 run, writing the body OBJ after each step.
    Run {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        out_dir: PathBuf,
        /// Write every N-th step (the first body and the last body are always written).
        #[arg(long, default_value_t = 1)]
        every: usize,
    },
    /// Load the project and report the mesh file that Tier 2 will read.
    Check {
        #[arg(long, default_value = "assets/projects/ryugu_cauchy.toml")]
        project: PathBuf,
    },
}

struct Setup {
    text: String,
    dir: PathBuf,
    spin: String,
    project_path: String,
    mesh_path: String,
    mesh_unit: String,
}

fn tidy(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out
}

fn setup(common: &Common) -> Result<Setup, String> {
    let text = std::fs::read_to_string(&common.project)
        .map_err(|e| format!("{}: {e}", common.project.display()))?;
    let dir = common
        .project
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let project = Project::parse(&text, dir.clone()).map_err(|e| e.to_string())?;
    let summary = mesh_summary(&project)?;
    let spin = common.spin.clone().unwrap_or_else(|| {
        format!("const:{:.6e}@{:.5e}", common.omega, common.duration)
    });
    SpinSchedule::parse(&spin)
        .and_then(|s| s.validate().map(|_| s))
        .map_err(|e| e.to_string())?;
    Ok(Setup {
        text,
        dir,
        spin,
        project_path: tidy(&common.project).display().to_string(),
        mesh_path: tidy(&summary.path).display().to_string(),
        mesh_unit: summary.unit,
    })
}

fn config_from(setup: &Setup, max_steps: Option<usize>) -> AppConfig {
    AppConfig {
        project_text: setup.text.clone(),
        project_dir: setup.dir.clone(),
        project_path: setup.project_path.clone(),
        mesh_path: setup.mesh_path.clone(),
        mesh_unit: setup.mesh_unit.clone(),
        spin: setup.spin.clone(),
        max_steps,
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::View(common) => {
            let setup = setup(&common)?;
            App::new()
                .add_plugins(DefaultPlugins)
                .add_plugins(AsteroidPlugin {
                    config: config_from(&setup, common.max_steps),
                    windowed: true,
                })
                .run();
            Ok(())
        }
        Command::Run {
            common,
            out_dir,
            every,
        } => {
            let setup = setup(&common)?;
            std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
            let write = |name: String, frame: &SurfaceFrame| {
                frame
                    .write_obj(&out_dir.join(name))
                    .map_err(|e| e.to_string())
            };
            let evolver = Evolver::new(&setup.text, &setup.dir, &setup.spin)?;
            let first = evolver.frame();
            write(format!("tier2_{:06}.obj", first.step), &first)?;
            println!(
                "start: {} vertices, radius [{:.3e}, {:.3e}] m",
                first.vertices.len(),
                first.radius_min,
                first.radius_max
            );
            let mut last = None;
            let mut count = 0usize;
            let complete = evolver.run(
                common.max_steps,
                || Ok(()),
                |frame| {
                    count += 1;
                    println!(
                        "step {} t={:.6e} omega={:.6e} radius [{:.3e}, {:.3e}]",
                        frame.step, frame.time, frame.omega, frame.radius_min, frame.radius_max
                    );
                    if every > 0 && count % every == 0 {
                        write(format!("tier2_{:06}.obj", frame.step), &frame)?;
                    }
                    last = Some(frame);
                    Ok(())
                },
            )?;
            let last = last.unwrap_or(first);
            write("final.obj".into(), &last)?;
            println!(
                "{}: {count} steps -> {}",
                if complete { "complete" } else { "stopped" },
                out_dir.display()
            );
            Ok(())
        }
        Command::Check { project } => {
            let text = std::fs::read_to_string(&project)
                .map_err(|e| format!("{}: {e}", project.display()))?;
            let dir = project.parent().unwrap_or(Path::new(".")).to_path_buf();
            let parsed = Project::parse(&text, dir).map_err(|e| e.to_string())?;
            let summary = mesh_summary(&parsed)?;
            println!("project {}", tidy(&project).display());
            println!(
                "mesh {} ({})",
                tidy(&summary.path).display(),
                summary.unit
            );
            println!(
                "{} vertices, {} faces; tier 2 stores metres",
                summary.vertices, summary.faces
            );
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}
