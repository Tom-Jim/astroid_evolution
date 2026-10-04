use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

use astroid_evolution::rheo_friction::error::RheoError;
use astroid_evolution::rheo_friction::model::fibonacci_directions;
use astroid_evolution::rheo_friction::project::{self, Project};
use astroid_evolution::rheo_friction::selfcheck;
use astroid_evolution::rheo_friction::{Environment, RheoGpu};

#[derive(Parser)]
#[command(
    name = "rheo-cli",
    about = "Rheology and friction-angle chain (GPU, double-single)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check that a project is closed: every constant present and in range.
    Validate {
        #[arg(long)]
        project: PathBuf,
    },
    /// Run the full chain on the project nodes and write a JSON report.
    Run {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Per-node table (CSV).
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// Acceptance checks: rigid motion, trace, strain rate, closure in double-single against f64.
    Check,
    /// Throughput of the chain on a lumpy surface.
    Bench {
        #[arg(long, default_value_t = 200_000)]
        nodes: usize,
    },
    /// Report the Metal device the chain would use.
    Gpu,
}

fn fail(err: &RheoError) -> ExitCode {
    eprintln!("{err}");
    ExitCode::from(err.code().clamp(1, 255) as u8)
}

fn read_project(path: &PathBuf) -> Result<Project, RheoError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| RheoError::model_not_closed(format!("{}: {e}", path.display())))?;
    Project::parse(&text)
}

fn run(cli: Cli) -> Result<(), RheoError> {
    match cli.command {
        Command::Validate { project } => {
            read_project(&project)?;
            println!("closed {}", project.display());
        }
        Command::Run { project, out, csv } => {
            let p = read_project(&project)?;
            let gpu = RheoGpu::new()?;
            let (nodes, result) = p.run(&gpu)?;
            let report = project::report(&p, &gpu.device_name(), &result);
            let text = serde_json::to_string_pretty(&report).unwrap_or_default();
            match out {
                Some(path) => std::fs::write(&path, &text)
                    .map_err(|e| RheoError::invalid_input(format!("{}: {e}", path.display())))?,
                None => println!("{text}"),
            }
            if let Some(path) = csv {
                std::fs::write(&path, project::csv(&nodes, &result))
                    .map_err(|e| RheoError::invalid_input(format!("{}: {e}", path.display())))?;
            }
        }
        Command::Check => {
            let gpu = RheoGpu::new()?;
            let checks = selfcheck::run_all(&gpu)?;
            let mut failed = 0;
            for c in &checks {
                let ok = c.passed();
                failed += usize::from(!ok);
                println!(
                    "{} {:<64} {:.3e} <= {:.1e}",
                    if ok { "pass" } else { "FAIL" },
                    c.name,
                    c.value,
                    c.limit
                );
            }
            if failed > 0 {
                return Err(RheoError::invalid_input(format!("{failed} checks failed")));
            }
        }
        Command::Bench { nodes } => {
            let gpu = RheoGpu::new()?;
            let material = selfcheck::reference_material(1.0, "dynamic")?;
            let (surface, velocity) = (selfcheck::lumpy_surface(), selfcheck::flow_field());
            let env = Environment {
                gravity_k: 1.0e-5,
                omega: 2.0e-4,
                p_ext: 0.0,
            };
            let batch = env.nodes(&surface, fibonacci_directions(nodes));
            gpu.evaluate(&material, &surface, &velocity, &batch)?;
            let r = gpu.evaluate(&material, &surface, &velocity, &batch)?;
            println!(
                "{} nodes in {:.4} s on {} ({:.2e} nodes/s), max fixed-point iterations {}",
                nodes,
                r.stats.seconds,
                gpu.device_name(),
                nodes as f64 / r.stats.seconds,
                r.stats.max_iterations
            );
        }
        Command::Gpu => {
            let gpu = RheoGpu::new()?;
            println!("rheology chain ready on {}", gpu.device_name());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => fail(&err),
    }
}
