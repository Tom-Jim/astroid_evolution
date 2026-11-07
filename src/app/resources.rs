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

/// The browser's wireframe: the surface edges as lines (`render::edge_mesh`).
#[derive(Component)]
pub struct WireView;


/// Project the desktop window and the browser page open when none is given.
pub const DEFAULT_PROJECT: &str = "dist/.stage/assets/projects/ryugu.toml";
/// Directory the browser build serves as its root: `assets/` in it is the page's `assets/`.
pub const DATA_DIR: &str = "dist/.stage/";
/// Ryugu's YORP spin-down from 3.40 h to today's 7.6326 h (Watanabe et al. 2019) at
/// −4.1e-7 °/day² (Cho et al. 2021): 9.41 Myr. 3.40 h is the fastest spin at which the
/// whole surface stays bound (inside the 3.0–3.5 h window of the western-bulge
/// deformation, Hirabayashi et al. 2019).
pub const DEFAULT_SPIN: &str = "linear:5.1333213e-4,-9.585919e-19,2.969610e14";

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

impl AppConfig {
    /// Read the project and check the spin schedule. The mesh is only located here:
    /// the scene and Tier 2 each read it once, when they need it.
    pub fn load(
        project: &std::path::Path,
        spin: &str,
        max_steps: Option<usize>,
        snapshot: Option<PathBuf>,
    ) -> Result<Self, String> {
        use crate::io::memfs::{normalize, read_to_string};
        let text = read_to_string(project)?;
        let dir = project
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        let parsed =
            crate::scf::params::Project::parse(&text, dir.clone()).map_err(|e| e.to_string())?;
        let mesh = parsed
            .body
            .mesh
            .as_deref()
            .ok_or("the project must use a mesh body")?;
        crate::scf::spin::SpinSchedule::parse(spin)
            .and_then(|s| s.validate())
            .map_err(|e| e.to_string())?;
        Ok(Self {
            project_text: text,
            project_dir: dir,
            project_path: normalize(project).display().to_string(),
            mesh_path: normalize(&parsed.resolve(mesh)).display().to_string(),
            mesh_unit: parsed.body.mesh_unit.clone().unwrap_or_else(|| "m".into()),
            spin: spin.into(),
            max_steps,
            snapshot,
        })
    }
}

impl AppConfig {
    /// Replace `section.key` values of the project text (the control layer's sliders).
    /// An integer key keeps an integer value; every key must already exist.
    pub fn apply_overrides(
        &mut self,
        overrides: &std::collections::BTreeMap<String, f64>,
    ) -> Result<(), String> {
        if overrides.is_empty() {
            return Ok(());
        }
        let mut table: toml::Table = self.project_text.parse().map_err(|e| format!("project: {e}"))?;
        for (path, value) in overrides {
            let (section, key) = path
                .split_once('.')
                .ok_or_else(|| format!("override {path} is not section.key"))?;
            let slot = table
                .get_mut(section)
                .and_then(|s| s.as_table_mut())
                .and_then(|s| s.get_mut(key))
                .ok_or_else(|| format!("override {path} names no project key"))?;
            *slot = match slot {
                toml::Value::Integer(_) => toml::Value::Integer(value.round() as i64),
                _ => toml::Value::Float(*value),
            };
        }
        self.project_text = toml::to_string(&table).map_err(|e| format!("project: {e}"))?;
        crate::scf::params::Project::parse(&self.project_text, self.project_dir.clone())
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// The worker has not been started.
    Idle,
    Preparing,
    Running,
    Paused,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
    pub eps_relax: f64,
    pub sweeps: usize,
    pub tensile_fraction: f64,
    pub inverse_ekman: f64,
    pub mass_residual: f64,
    pub mobile_area_fraction: f64,
    pub dt: f64,
    pub dt_cfl: f64,
    pub dt_bound: crate::app::snapshot::Tag,
    /// Largest principal Cauchy stress on the surface (Pa, tension positive).
    pub tension_max: f64,
    /// Runout of the eroded packets (§10.0.2): packets, mass-weighted mean distance (m),
    /// mean change of |latitude| (rad, negative equatorward), packets that hopped.
    pub runout_traced: usize,
    pub runout_distance: f64,
    pub runout_dlat: f64,
    pub runout_hops: usize,
}

impl FrameStats {
    pub fn of(s: &crate::app::snapshot::StepSnapshot) -> Self {
        Self {
            step: s.step,
            time: s.time,
            omega: s.omega,
            mass: s.mass,
            radius_min: s.radius_min,
            radius_max: s.radius_max,
            yielded_fraction: s.yielded_fraction,
            max_slack: s.max_slack,
            rho_s_total: s.rho_s_total,
            max_displacement: s.max_displacement,
            eps_momentum: s.audit.eps_momentum,
            eps_lambda: s.audit.eps_lambda,
            eps_relax: s.audit.eps_relax,
            sweeps: s.audit.sweeps,
            tensile_fraction: s.audit.tensile_fraction,
            inverse_ekman: s.audit.inverse_ekman,
            mass_residual: s.audit.mass_residual,
            mobile_area_fraction: s.audit.mobile_area_fraction,
            dt: s.audit.dt,
            dt_cfl: s.audit.dt_cfl,
            dt_bound: s.dt_bound,
            tension_max: s.tension_max,
            runout_traced: s.audit.runout_traced,
            runout_distance: s.audit.runout_distance,
            runout_dlat: s.audit.runout_dlat,
            runout_hops: s.audit.runout_hops,
        }
    }
}

#[derive(Resource)]
pub struct Evolution {
    pub phase: Phase,
    pub status: String,
    pub control: Arc<Control>,
    pub stats: Option<FrameStats>,
    pub receiver: Mutex<Option<Receiver<WorkerMessage>>>,
    /// The frames of a cached run not yet played (`app::cache`).
    pub replay: Option<crate::app::history::History>,
    /// The frames come from a cache file, not from a live computation.
    pub cached: bool,
}

impl Default for Evolution {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            status: "waiting to start tier 2".into(),
            control: Arc::default(),
            stats: None,
            receiver: Mutex::new(None),
            replay: None,
            cached: false,
        }
    }
}
