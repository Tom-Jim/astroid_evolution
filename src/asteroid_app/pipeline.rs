//! Tier 3 display feeder.
//!
//! The project mesh is handed to Tier 2 (`initial_body` → `start` → `run`) with the
//! viewer's spin schedule, which defaults to a constant rate. Each message is the
//! current Tier 2 body. Nothing here smooths, fuses, or maps that body back onto the
//! original file.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use crate::rheo_friction::RheoGpu;
use crate::rtfp_solver::io::{ImportRequest, load_mesh_file};
use crate::scf_simulator::body::Body;
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::evolve::{self, RunOptions, State};
use crate::scf_simulator::forward::Forward;
use crate::scf_simulator::io::initial_body;
use crate::scf_simulator::params::Project;
use crate::scf_simulator::spin::SpinSchedule;

/// Shared flags between the viewer and the worker thread.
#[derive(Default)]
pub struct Control {
    pub paused: AtomicBool,
    pub quit: AtomicBool,
}

/// The mesh Tier 2 is holding right now.
pub struct SurfaceFrame {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    /// Tier 2 vertex positions, in metres, in the Tier 2 frame.
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub radius_min: f64,
    pub radius_max: f64,
}

impl SurfaceFrame {
    /// Copy `body` as it stands. Vertex order and faces are Tier 2's.
    pub fn from_body(body: &Body, step: usize, time: f64, omega: f64, mass: f64) -> Self {
        let (mut radius_min, mut radius_max) = (f64::INFINITY, 0.0_f64);
        for radius in &body.radius {
            radius_min = radius_min.min(*radius);
            radius_max = radius_max.max(*radius);
        }
        if body.radius.is_empty() {
            radius_min = 0.0;
        }
        Self {
            step,
            time,
            omega,
            mass,
            vertices: body.positions().iter().map(|p| p.to_array()).collect(),
            faces: body.topo.faces.clone(),
            radius_min,
            radius_max,
        }
    }

    pub fn from_state(state: &State, mass: f64) -> Self {
        Self::from_body(&state.body, state.step, state.t, state.omega, mass)
    }

    pub fn write_obj(&self, path: &Path) -> Result<(), String> {
        let mut text = format!(
            "# tier2 step={} t={:.6e} omega={:.6e} radius=[{:.6e},{:.6e}] m\n",
            self.step, self.time, self.omega, self.radius_min, self.radius_max
        );
        for v in &self.vertices {
            text.push_str(&format!("v {:.9} {:.9} {:.9}\n", v[0], v[1], v[2]));
        }
        for f in &self.faces {
            text.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

pub enum WorkerMessage {
    Status(String),
    Frame(Box<SurfaceFrame>),
    Finished(String),
    Failed(String),
}

/// File the project will hand to Tier 2, before Tier 2 centres it.
pub struct MeshSummary {
    pub path: PathBuf,
    pub unit: String,
    pub vertices: usize,
    pub faces: usize,
}

/// The imported file in metres, with the view fit around its bounding centre.
///
/// Axes are the file's axes. glTF Y is up, which is already Bevy's up.
pub struct ViewFit {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub center: [f64; 3],
    pub scale: f64,
}

pub fn view_fit(project: &Project) -> Result<ViewFit, String> {
    let Some(mesh) = project.body.mesh.as_deref() else {
        return Err("the project must use body.shape = \"mesh\"".into());
    };
    let doc = load_mesh_file(ImportRequest {
        path: &project.resolve(mesh),
        length_unit: project.body.mesh_unit.as_deref(),
        reconstruct: None,
    })
    .map_err(|e| e.to_string())?;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for v in &doc.mesh.vertices {
        let p = v.to_array();
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let center = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let reach = doc
        .mesh
        .vertices
        .iter()
        .map(|v| {
            let p = v.to_array();
            (0..3)
                .map(|k| (p[k] - center[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
        .max(1.0e-12);
    Ok(ViewFit {
        vertices: doc.mesh.vertices.iter().map(|v| v.to_array()).collect(),
        faces: doc.mesh.faces,
        center,
        scale: 1.0 / reach,
    })
}

pub fn mesh_summary(project: &Project) -> Result<MeshSummary, String> {
    let Some(mesh) = project.body.mesh.as_deref() else {
        return Err("the project must use body.shape = \"mesh\"".into());
    };
    let path = project.resolve(mesh);
    let unit = project
        .body
        .mesh_unit
        .clone()
        .unwrap_or_else(|| "m".to_string());
    let doc = load_mesh_file(ImportRequest {
        path: &path,
        length_unit: project.body.mesh_unit.as_deref(),
        reconstruct: None,
    })
    .map_err(|e| e.to_string())?;
    Ok(MeshSummary {
        path,
        unit,
        vertices: doc.mesh.vertices.len(),
        faces: doc.mesh.faces.len(),
    })
}

/// Tier 2 state for one project and one spin schedule.
pub struct Evolver {
    project: Project,
    gpu: RheoGpu,
    schedule: SpinSchedule,
    state: State,
    forward: Forward,
}

impl Evolver {
    /// `initial_body` then `evolve::start`. The body is whatever Tier 2 built from the file.
    pub fn new(project_text: &str, project_dir: &Path, spin: &str) -> Result<Self, String> {
        let project =
            Project::parse(project_text, project_dir.to_path_buf()).map_err(|e| e.to_string())?;
        let schedule = SpinSchedule::parse(spin).map_err(|e| e.to_string())?;
        schedule.validate().map_err(|e| e.to_string())?;
        let body = initial_body(&project).map_err(|e| e.to_string())?;
        let (state, forward) = evolve::start(&project, body).map_err(|e| e.to_string())?;
        let gpu = RheoGpu::new().map_err(|e| e.to_string())?;
        Ok(Self {
            project,
            gpu,
            schedule,
            state,
            forward,
        })
    }

    /// The body Tier 2 holds before `run` advances it.
    pub fn frame(&self) -> SurfaceFrame {
        SurfaceFrame::from_state(&self.state, self.forward.moments.mass)
    }

    /// `evolve::run` with this schedule. `sink` receives the body after every accepted step.
    pub fn run(
        self,
        max_steps: Option<usize>,
        mut gate: impl FnMut() -> Result<(), String>,
        mut sink: impl FnMut(SurfaceFrame) -> Result<(), String>,
    ) -> Result<bool, String> {
        let Self {
            project,
            gpu,
            schedule,
            state,
            forward,
        } = self;
        let mut observer = |record: &evolve::StepRecord, state: &State| -> Result<(), ScfError> {
            gate().map_err(ScfError::io)?;
            sink(SurfaceFrame::from_state(state, record.mass)).map_err(ScfError::io)
        };
        let options = RunOptions {
            max_steps,
            until: None,
        };
        let result = evolve::run(
            &project,
            &gpu,
            &schedule,
            state,
            forward,
            &options,
            &mut observer,
        )
        .map_err(|e| e.to_string())?;
        Ok(result.complete)
    }
}

/// Run Tier 2 on a worker thread. The bounded channel keeps only the newest frames.
pub fn spawn(
    project_text: String,
    project_dir: PathBuf,
    spin: String,
    max_steps: Option<usize>,
    control: Arc<Control>,
) -> Receiver<WorkerMessage> {
    let (tx, rx) = mpsc::sync_channel(2);
    std::thread::Builder::new()
        .name("tier2-evolution".into())
        .stack_size(64 << 20)
        .spawn(move || {
            let send =
                |message: WorkerMessage| tx.send(message).map_err(|_| "viewer closed".to_string());
            let outcome = (|| -> Result<String, String> {
                send(WorkerMessage::Status(
                    "computing the first tier 2 step".into(),
                ))?;
                let evolver = Evolver::new(&project_text, &project_dir, &spin)?;
                send(WorkerMessage::Status("tier 2 running".into()))?;
                let gate = || {
                    while control.paused.load(Ordering::Relaxed) {
                        if control.quit.load(Ordering::Relaxed) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    if control.quit.load(Ordering::Relaxed) {
                        Err("cancelled".to_string())
                    } else {
                        Ok(())
                    }
                };
                let complete = evolver.run(max_steps, gate, |frame| {
                    send(WorkerMessage::Frame(Box::new(frame)))
                })?;
                Ok(if complete {
                    "schedule complete".into()
                } else {
                    "stopped at the step limit".into()
                })
            })();
            let _ = match outcome {
                Ok(message) => tx.send(WorkerMessage::Finished(message)),
                Err(message) => tx.send(WorkerMessage::Failed(message)),
            };
        })
        .expect("spawn the tier-2 worker thread");
    rx
}
