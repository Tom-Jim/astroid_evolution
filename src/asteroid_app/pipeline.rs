//! Tier 2 → fused-model pipeline used by the viewer.
//!
//! ```text
//! ryugu.glb ──► high (original, shown at start)
//!      │
//!      └─► sh_smooth::smooth (ancient preset) ─► tier-2 body ─► relax ─► run (quasi-static)
//!                                                                          │ every step
//!      fused = sh_smooth::fuse(high, tier-2 surface mapped back to the glb frame) ◄┘
//! ```
//!
//! Nothing here implements physics: smoothing, relaxation, evolution and fusion are the
//! existing library calls. This module only chains them and maps the tier-2 frame back to
//! the frame of the original mesh.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use nalgebra::{Matrix3, Vector3};

use crate::common::math::DVec3;
use crate::rheo_friction::RheoGpu;
use crate::rtfp_solver::io::{ImportRequest, load_mesh_file};
use crate::scf_simulator::body::Body;
use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::evolve::{self, RunOptions, State, compose_frame};
use crate::scf_simulator::forward::Forward;
use crate::scf_simulator::frame::realign;
use crate::scf_simulator::io::mesh_body;
use crate::scf_simulator::params::Project;
use crate::scf_simulator::relax::relax;
use crate::scf_simulator::spin::SpinSchedule;
use crate::sh_smooth::{Center, FuseOptions, Mesh, ShGpu, SmoothOptions, fuse, smooth};

/// Shared flags between the viewer and the worker thread.
#[derive(Default)]
pub struct Control {
    pub paused: AtomicBool,
    pub quit: AtomicBool,
}

/// One fused model: the original mesh plus the interpolated tier-2 displacement.
pub struct FusedFrame {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    /// Same vertex order and faces as the original mesh.
    pub vertices: Vec<[f64; 3]>,
    /// Signed radial displacement of each vertex relative to the original (m).
    pub radial_delta: Vec<f32>,
    pub delta_min: f64,
    pub delta_max: f64,
}

impl FusedFrame {
    /// The fused surface as a mesh with the original's faces.
    pub fn to_mesh(&self, original: &Mesh) -> Mesh {
        Mesh {
            vertices: self.vertices.clone(),
            faces: original.faces.clone(),
            length_unit: original.length_unit.clone(),
        }
    }
}

pub enum WorkerMessage {
    Status(String),
    Frame(Box<FusedFrame>),
    Finished(String),
    Failed(String),
}

/// The original model, read from the project's `body.mesh` (converted to metres).
pub fn load_original(project: &Project) -> Result<Mesh, String> {
    let Some(mesh) = project.body.mesh.as_deref() else {
        return Err("the project must use body.shape = \"mesh\"".into());
    };
    let path = project.resolve(mesh);
    let doc = load_mesh_file(ImportRequest {
        path: &path,
        length_unit: project.body.mesh_unit.as_deref(),
        reconstruct: None,
    })
    .map_err(|e| e.to_string())?;
    Ok(Mesh {
        vertices: doc.mesh.vertices.iter().map(|v| v.to_array()).collect(),
        faces: doc.mesh.faces.clone(),
        length_unit: doc.length_unit,
    })
}

/// Tier-2 body on the smoothed mesh's own vertices, as `initial_body` does for a file.
///
/// Returns the body and the vertex mean that was subtracted (tier-2 origin in the mesh frame).
/// SCF moves these vertices; they are not resampled onto an icosphere.
fn body_from_mesh(project: &Project, mesh: &Mesh) -> Result<(Body, DVec3), String> {
    let vertices: Vec<DVec3> = mesh
        .vertices
        .iter()
        .map(|v| DVec3::new(v[0], v[1], v[2]))
        .collect();
    mesh_body(project, &vertices, mesh.faces.clone()).map_err(|e| e.to_string())
}

/// Map tier-2 vertices back to the frame of the original mesh.
///
/// Tier 2 keeps `x_orig = Rᵀ x_cur + t` (§10.4.4 step 8b) relative to the mean-subtracted
/// mesh, so `x_mesh = Rᵀ x_cur + t + origin`.
pub fn to_mesh_frame(
    positions: &[DVec3],
    rotation: [[f64; 3]; 3],
    translation: DVec3,
    origin: DVec3,
) -> Vec<[f64; 3]> {
    let r = Matrix3::from_fn(|i, j| rotation[i][j]);
    let shift = Vector3::new(translation.x, translation.y, translation.z)
        + Vector3::new(origin.x, origin.y, origin.z);
    positions
        .iter()
        .map(|p| {
            let x = r.transpose() * Vector3::new(p.x, p.y, p.z) + shift;
            [x.x, x.y, x.z]
        })
        .collect()
}

/// `sh_smooth::fuse(high, coarse)` plus the per-vertex signed radial displacement.
pub fn fuse_frame(
    high: &Mesh,
    coarse: &Mesh,
    step: usize,
    time: f64,
    omega: f64,
    mass: f64,
) -> Result<FusedFrame, String> {
    let out = fuse(
        high,
        coarse,
        &FuseOptions {
            center: Center::Auto,
        },
    )
    .map_err(|e| e.to_string())?;
    let center = Vector3::from(out.report.center);
    let radial_delta = out
        .mesh
        .vertices
        .iter()
        .zip(&high.vertices)
        .map(|(fused, original)| {
            let dir = Vector3::from(*original) - center;
            let moved = Vector3::from(*fused) - Vector3::from(*original);
            dir.try_normalize(1.0e-12).map_or(0.0, |d| d.dot(&moved)) as f32
        })
        .collect();
    Ok(FusedFrame {
        step,
        time,
        omega,
        mass,
        vertices: out.mesh.vertices,
        radial_delta,
        delta_min: out.report.delta_min,
        delta_max: out.report.delta_max,
    })
}

/// Smoothed → relaxed tier-2 state that can be fused with the original at any time.
pub struct Evolver {
    project: Project,
    high: Arc<Mesh>,
    gpu: RheoGpu,
    schedule: SpinSchedule,
    origin: DVec3,
    state: State,
    forward: Forward,
}

impl Evolver {
    /// `ryugu.glb → smooth (ancient) → tier-2 body → start → relax → realign`.
    pub fn new(
        project_text: &str,
        project_dir: &Path,
        high: Arc<Mesh>,
        spin: &str,
    ) -> Result<Self, String> {
        let project =
            Project::parse(project_text, project_dir.to_path_buf()).map_err(|e| e.to_string())?;
        let schedule = SpinSchedule::parse(spin).map_err(|e| e.to_string())?;
        schedule.validate().map_err(|e| e.to_string())?;

        let sh_gpu = ShGpu::new().map_err(|e| e.to_string())?;
        let smoothed =
            smooth(&high, &SmoothOptions::ancient_ryugu(), &sh_gpu).map_err(|e| e.to_string())?;
        let (body, origin) = body_from_mesh(&project, &smoothed.mesh)?;

        let (mut state, _) = evolve::start(&project, body).map_err(|e| e.to_string())?;
        let gpu = RheoGpu::new().map_err(|e| e.to_string())?;
        let omega0 = schedule.sample(0.0).omega;
        let (relaxed, _) = relax(&project, &gpu, &state.body, omega0).map_err(|e| e.to_string())?;
        let (relaxed, report, fwd) = realign(&relaxed, None, project.g, &project.geometry, false)
            .map_err(|e| e.to_string())?;
        let (rotation, translation) =
            compose_frame(state.frame_rotation, state.frame_translation, &report);
        state.body = relaxed;
        state.omega = omega0;
        state.frame_rotation = rotation;
        state.frame_translation = translation;
        let forward = fwd;
        Ok(Self {
            project,
            high,
            gpu,
            schedule,
            origin,
            state,
            forward,
        })
    }

    /// Fuse the current tier-2 surface with the original mesh.
    pub fn frame(&self) -> Result<FusedFrame, String> {
        fused_from_state(
            &self.high,
            self.origin,
            &self.state,
            self.forward.moments.mass,
        )
    }

    /// Quasi-static evolution. `gate` runs before each step's fusion (pause / cancel),
    /// `sink` receives every fused frame. Returns whether the schedule completed.
    pub fn run(
        self,
        max_steps: Option<usize>,
        mut gate: impl FnMut() -> Result<(), String>,
        mut sink: impl FnMut(FusedFrame) -> Result<(), String>,
    ) -> Result<bool, String> {
        let Self {
            project,
            high,
            gpu,
            schedule,
            origin,
            state,
            forward,
        } = self;
        let mut observer = |record: &evolve::StepRecord, state: &State| -> Result<(), ScfError> {
            gate().map_err(ScfError::io)?;
            let frame =
                fused_from_state(&high, origin, state, record.mass).map_err(ScfError::io)?;
            sink(frame).map_err(ScfError::io)
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

fn fused_from_state(
    high: &Mesh,
    origin: DVec3,
    state: &State,
    mass: f64,
) -> Result<FusedFrame, String> {
    let coarse = Mesh {
        vertices: to_mesh_frame(
            &state.body.positions(),
            state.frame_rotation,
            state.frame_translation,
            origin,
        ),
        faces: state.body.topo.faces.clone(),
        length_unit: high.length_unit.clone(),
    };
    fuse_frame(high, &coarse, state.step, state.t, state.omega, mass)
}

/// Run the whole pipeline on a worker thread. The bounded channel gives back-pressure.
pub fn spawn(
    project_text: String,
    project_dir: std::path::PathBuf,
    high: Arc<Mesh>,
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
                    "smoothing the glb and relaxing the tier-2 body".into(),
                ))?;
                let evolver = Evolver::new(&project_text, &project_dir, high, &spin)?;
                send(WorkerMessage::Frame(Box::new(evolver.frame()?)))?;
                send(WorkerMessage::Status("evolving".into()))?;
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
