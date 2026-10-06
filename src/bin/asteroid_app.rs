use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use bevy::prelude::*;
use clap::{Args, Parser, Subcommand};

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::pipeline::{Evolver, FusedFrame, load_original};
use astroid_evolution::asteroid_app::resources::AppConfig;
use astroid_evolution::scf_simulator::params::Project;
use astroid_evolution::scf_simulator::spin::SpinSchedule;

#[derive(Parser)]
#[command(
    name = "asteroid-app",
    about = "Tier 3: original model, live-fused with the Tier 2 evolution"
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
    /// Window: shows the original model; Space runs Tier 2 and fuses every step live.
    View(Common),
    /// Headless: the same pipeline without a window; writes the fused OBJ files.
    Run {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        out_dir: PathBuf,
        /// Write every N-th fused step (the first and the last frame are always written).
        #[arg(long, default_value_t = 1)]
        every: usize,
    },
    /// Load the project and the original model and report their sizes.
    Check {
        #[arg(long, default_value = "assets/projects/ryugu_cauchy.toml")]
        project: PathBuf,
    },
}

struct Setup {
    text: String,
    dir: PathBuf,
    spin: String,
}

fn setup(common: &Common) -> Result<Setup, String> {
    let text = std::fs::read_to_string(&common.project)
        .map_err(|e| format!("{}: {e}", common.project.display()))?;
    let dir = common
        .project
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    Project::parse(&text, dir.clone()).map_err(|e| e.to_string())?;
    let spin = common
        .spin
        .clone()
        .unwrap_or_else(|| format!("const:{}@{}", common.omega, common.duration));
    SpinSchedule::parse(&spin)
        .and_then(|s| s.validate().map(|_| s))
        .map_err(|e| e.to_string())?;
    Ok(Setup { text, dir, spin })
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::View(common) => {
            let s = setup(&common)?;
            App::new()
                .add_plugins(DefaultPlugins)
                .add_plugins(AsteroidPlugin {
                    config: AppConfig {
                        project_text: s.text,
                        project_dir: s.dir,
                        spin: s.spin,
                        max_steps: common.max_steps,
                    },
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
            let s = setup(&common)?;
            let project = Project::parse(&s.text, s.dir.clone()).map_err(|e| e.to_string())?;
            let original = Arc::new(load_original(&project)?);
            std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
            let write = |name: String, frame: &FusedFrame| {
                frame
                    .to_mesh(&original)
                    .write_obj(
                        &out_dir.join(name),
                        &format!(
                            "fused:step={}:dR=[{:.3e},{:.3e}]",
                            frame.step, frame.delta_min, frame.delta_max
                        ),
                    )
                    .map_err(|e| e.to_string())
            };
            let evolver = Evolver::new(&s.text, &s.dir, original.clone(), &s.spin)?;
            let first = evolver.frame()?;
            write(format!("fused_{:06}.obj", first.step), &first)?;
            println!(
                "start: {} vertices, dR [{:.3e}, {:.3e}] m",
                first.vertices.len(),
                first.delta_min,
                first.delta_max
            );
            let mut last = None;
            let mut count = 0usize;
            let complete = evolver.run(
                common.max_steps,
                || Ok(()),
                |frame| {
                    count += 1;
                    println!(
                        "step {} t={:.6e} omega={:.6e} dR [{:.3e}, {:.3e}]",
                        frame.step, frame.time, frame.omega, frame.delta_min, frame.delta_max
                    );
                    if every > 0 && count % every == 0 {
                        write(format!("fused_{:06}.obj", frame.step), &frame)?;
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
            let original = load_original(&parsed)?;
            println!(
                "original {} vertices, {} faces, unit {}",
                original.vertices.len(),
                original.faces.len(),
                original.length_unit
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
