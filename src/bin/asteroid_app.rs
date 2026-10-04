use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use clap::Parser;

use astroid_evolution::asteroid_app::AsteroidPlugin;
use astroid_evolution::asteroid_app::driver::ScfPlugin;
use astroid_evolution::asteroid_app::playback::PlaybackPlugin;
use astroid_evolution::scf_simulator::simulation::closed_run;

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
        let steps = cli.max_steps.unwrap_or(3);
        if let Err(err) = closed_run(
            &text,
            project.parent().unwrap_or(std::path::Path::new(".")),
            steps,
        ) {
            eprintln!("{err}");
            return ExitCode::from(err.code as u8);
        }
        let exit = App::new()
            .add_plugins(MinimalPlugins)
            .add_plugins(AsteroidPlugin { windowed: false })
            .add_plugins(ScfPlugin {
                project_text: text,
                project_dir: project
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_path_buf(),
                max_steps: steps,
                exit_when_done: true,
            })
            .run();
        return if exit.is_success() {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
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
    if let Err(err) = closed_run(
        &text,
        project.parent().unwrap_or(std::path::Path::new(".")),
        0,
    ) {
        eprintln!("{err}");
        return ExitCode::from(err.code as u8);
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
            max_steps: cli.max_steps.unwrap_or(u32::MAX),
            exit_when_done: false,
        })
        .run();
    ExitCode::SUCCESS
}
