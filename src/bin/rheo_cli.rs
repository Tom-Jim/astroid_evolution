use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

use astroid_evolution::common::error::SimError;
use astroid_evolution::rheo::RheoGpu;
use astroid_evolution::rheo::project::{self, Project};

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
    /// Report the Metal device the chain would use.
    Gpu,
}

fn read_project(path: &PathBuf) -> Result<Project, SimError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| SimError::model_not_closed(format!("{}: {e}", path.display())))?;
    Project::parse(&text)
}

fn run(cli: Cli) -> Result<(), SimError> {
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
                    .map_err(|e| SimError::invalid_input(format!("{}: {e}", path.display())))?,
                None => println!("{text}"),
            }
            if let Some(path) = csv {
                std::fs::write(&path, project::csv(&nodes, &result))
                    .map_err(|e| SimError::invalid_input(format!("{}: {e}", path.display())))?;
            }
        }
        Command::Gpu => {
            let gpu = RheoGpu::new()?;
            println!("rheology chain ready on {}", gpu.device_name());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    SimError::exit(run(Cli::parse()))
}
