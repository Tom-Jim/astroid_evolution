use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;

use bevy::prelude::*;

use crate::app::pipeline::{Control, WorkerMessage};

#[derive(Component)]
pub struct SurfaceView;

#[derive(Component)]
pub struct RegolithView;

#[derive(Component)]
pub struct Hud;

/// Project the desktop window and the browser page open when none is given.
pub const DEFAULT_PROJECT: &str = "dist/.stage/assets/projects/ryugu.toml";
/// Directory the browser build serves as its root: `assets/` in it is the page's `assets/`.
pub const DATA_DIR: &str = "dist/.stage/";
/// Ryugu's YORP spin-down from 3.3 h (inside the 3.0–3.5 h window of the western-bulge
/// deformation, Hirabayashi et al. 2019) to today's 7.6326 h (Watanabe et al. 2019) at
/// −4.1e-7 °/day² (Cho et al. 2021): 9.92 Myr.
pub const DEFAULT_SPIN: &str = "linear:5.288877e-4,-9.585919e-19,3.131885e14";

/// Inputs of the viewer, written by the binary and repeated in the corner label.
#[derive(Resource, Clone)]
pub struct AppConfig {
    pub project_text: String,
    pub project_dir: PathBuf,
    /// Path of the project file, as given on the command line.
    pub project_path: String,
    /// Mesh file the project passes to Tier 2.
    pub mesh_path: String,
    /// Length unit declared for that mesh file (`km` for Ryugu). Tier 2 stores metres.
    pub mesh_unit: String,
    /// Spin schedule grammar of `scf-cli run-sim --spin`. The default is `const:`.
    pub spin: String,
    pub max_steps: Option<usize>,
    /// Snapshot to start from instead of the project body (desktop only).
    pub snapshot: Option<PathBuf>,
}

/// The imported mesh and the scale that puts its bounding radius at 1.
/// Tier 2 bodies reuse `scale` so a later change in size stays visible.
#[derive(Resource)]
pub struct Scene {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub center: [f64; 3],
    /// Metres to view units.
    pub scale: f64,
}

/// Largest distance from the Tier 2 origin, inverted. The body is already centred.
pub fn lock_scale(vertices: &[[f64; 3]]) -> f64 {
    let reach = vertices
        .iter()
        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
        .fold(0.0, f64::max)
        .max(1.0e-12);
    1.0 / reach
}

/// Imported file: subtract its bounding centre, keep axes, then scale.
pub fn display_imported(vertices: &[[f64; 3]], center: [f64; 3], scale: f64) -> Vec<[f32; 3]> {
    vertices
        .iter()
        .map(|v| {
            [
                ((v[0] - center[0]) * scale) as f32,
                ((v[1] - center[1]) * scale) as f32,
                ((v[2] - center[2]) * scale) as f32,
            ]
        })
        .collect()
}

/// Tier 2 positions are already centred. Axes stay as Tier 2 stored them.
pub fn display_positions(vertices: &[[f64; 3]], scale: f64) -> Vec<[f32; 3]> {
    vertices
        .iter()
        .map(|v| {
            [
                (v[0] * scale) as f32,
                (v[1] * scale) as f32,
                (v[2] * scale) as f32,
            ]
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The worker has not been started.
    Idle,
    Preparing,
    Running,
    Paused,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameStats {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    pub radius_min: f64,
    pub radius_max: f64,
    pub yielded_fraction: f64,
    pub max_slack: f64,
    pub rho_s_total: f64,
    pub max_displacement: f64,
    pub eps_momentum: f64,
    pub eps_lambda: f64,
    pub eps_picard: f64,
    pub picard_iterations: usize,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub mass_residual: f64,
    pub mobile_area_fraction: f64,
    pub dt: f64,
    pub dt_cfl: f64,
    pub dt_bound: &'static str,
}

#[derive(Resource)]
pub struct Evolution {
    pub phase: Phase,
    pub status: String,
    pub control: Arc<Control>,
    pub stats: Option<FrameStats>,
    pub receiver: Mutex<Option<Receiver<WorkerMessage>>>,
}

impl Default for Evolution {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            status: "waiting to start tier 2".into(),
            control: Arc::default(),
            stats: None,
            receiver: Mutex::new(None),
        }
    }
}
