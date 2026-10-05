use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use clap::Parser;

use astroid_evolution::asteroid_app::driver::ScfPlugin;
use astroid_evolution::asteroid_app::playback::PlaybackPlugin;
use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::scf_simulator::io::initial_body;
use astroid_evolution::scf_simulator::params::Project;

#[derive(Parser)]
#[command(name = "asteroid-app")]
struct Cli {
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long)]
    playback: Option<PathBuf>,
    #[arg(long)]
    headless_test: bool,
    #[arg(long)]
    max_steps: Option<u32>,
    /// Spin rate used by the Tier 2 relaxation (rad/s).
    #[arg(long, default_value_t = 4.986655e-4)]
    omega: f64,
    /// Write the relaxed Tier 2 snapshot (the format is the same as `scf-cli relax`).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Write the relaxed OBJ alongside the viewer.
    #[arg(long)]
    obj: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.headless_test {
        let Some(project) = cli.project else {
            eprintln!("headless test needs --project");
            return ExitCode::from(2);
        };
        let text = match std::fs::read_to_string(&project) {
            Ok(text) => text,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(2);
            }
        };
        let parsed = match Project::parse(
            &text,
            project
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf(),
        ) {
            Ok(project) => project,
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(err.code().clamp(1, 255) as u8);
            }
        };
        if let Err(err) = initial_body(&parsed) {
            eprintln!("{err}");
            return ExitCode::from(err.code().clamp(1, 255) as u8);
        }
        println!(
            "model {} verification_level input_valid",
            if parsed.body.shape == "mesh" {
                "general_mesh"
            } else {
                "ellipsoid"
            }
        );
        return ExitCode::SUCCESS;
    }
    if let Some(path) = cli.playback {
        App::new()
            .add_plugins(DefaultPlugins)
            .add_plugins(AsteroidPlugin { windowed: true })
            .add_plugins(PlaybackPlugin { path })
            .run();
        return ExitCode::SUCCESS;
    }
    let Some(project) = cli.project else {
        eprintln!("pass --project or --playback");
        return ExitCode::from(2);
    };
    let text = match std::fs::read_to_string(&project) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("{err}");
            return ExitCode::from(2);
        }
    };
    if let Err(err) = Project::parse(
        &text,
        project
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf(),
    ) {
        eprintln!("{err}");
        return ExitCode::from(err.code().clamp(1, 255) as u8);
    }
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(AsteroidPlugin { windowed: true })
        .add_plugins(ScfPlugin {
            project_text: text,
            project_dir: project
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf(),
            omega: cli.omega,
            out: cli.out,
            obj: cli.obj,
        })
        .run();
    ExitCode::SUCCESS
}
