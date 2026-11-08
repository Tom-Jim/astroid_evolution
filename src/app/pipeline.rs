//! Tier 2 driver for the viewer.
//!
//! `evolve::run` runs on a thread of its own (`EvolutionPool`); its parallel loops
//! and the packing of a snapshot use the global Rayon pool. The Bevy thread only receives the
//! snapshot. A full channel blocks the solver, so a fast mesh does not drop steps.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::sync::{Condvar, Mutex};

use bevy::prelude::Resource;

use crate::app::resources::AppConfig;
use crate::app::snapshot::StepSnapshot;
use crate::common::error::SimError;
use crate::io::{ImportRequest, load_mesh_file};
use crate::scf::body::Body;
use crate::scf::evolve::{self, RunOptions, State};
use crate::scf::forward::Forward;
use crate::scf::io::initial_body;
use crate::scf::params::Project;
use crate::scf::spin::SpinSchedule;

/// Shared flags between the viewer and the evolution thread.
///
/// Pause waits on a condition variable, so a paused step does not occupy the
/// global Rayon pool with a sleep loop.
#[derive(Default)]
pub struct Control {
    pub paused: AtomicBool,
    pub quit: AtomicBool,
    pause: Mutex<()>,
    resume: Condvar,
}

impl Control {
    pub fn wait_if_paused(&self) -> Result<(), String> {
        let mut guard = self.pause.lock().unwrap();
        while self.paused.load(Ordering::SeqCst) && !self.quit.load(Ordering::SeqCst) {
            guard = self.resume.wait(guard).unwrap();
        }
        if self.quit.load(Ordering::SeqCst) {
            Err("cancelled".into())
        } else {
            Ok(())
        }
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::SeqCst);
        if !paused {
            self.resume.notify_all();
        }
        #[cfg(target_arch = "wasm32")]
        crate::web::set_pause_flag(paused);
    }

    pub fn request_quit(&self) {
        self.quit.store(true, Ordering::SeqCst);
        self.paused.store(false, Ordering::SeqCst);
        self.resume.notify_all();
        #[cfg(target_arch = "wasm32")]
        crate::web::set_quit_flag();
    }
}

/// The body Tier 2 holds before the first accepted step.
pub struct SurfaceFrame {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub radius_min: f64,
    pub radius_max: f64,
}

impl SurfaceFrame {
    pub fn from_body(body: &Body, step: usize, time: f64, omega: f64, mass: f64) -> Self {
        let (radius_min, radius_max) = crate::app::snapshot::radius_range(&body.positions);
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
        crate::common::obj::write(
            path,
            &format!(
                "# tier2 step={} t={:.6e} omega={:.6e} radius=[{:.6e},{:.6e}] m\n",
                self.step, self.time, self.omega, self.radius_min, self.radius_max
            ),
            &self.vertices,
            &self.faces,
        )
        .map_err(|e| e.to_string())
    }
}

pub enum WorkerMessage {
    Status(String),
    Frame(Box<StepSnapshot>),
    Finished(String),
    Failed(String),
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
    .map_err(|err| err.to_string())?;
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

/// Parsed project and the body `evolve::start` produced.
pub struct Prepared {
    pub project: Project,
    pub schedule: SpinSchedule,
    pub state: State,
    pub forward: Forward,
}

pub fn prepare(
    project_text: &str,
    project_dir: &Path,
    spin: &str,
    snapshot: Option<&Path>,
) -> Result<Prepared, String> {
    let project =
        Project::parse(project_text, project_dir.to_path_buf()).map_err(|e| e.to_string())?;
    let schedule = SpinSchedule::parse(spin).map_err(|e| e.to_string())?;
    schedule.validate().map_err(|e| e.to_string())?;
    let (state, forward) = match snapshot {
        Some(path) => {
            let state = crate::scf::io::read_snapshot(path)
                .and_then(|s| s.to_state())
                .map_err(|e| e.to_string())?;
            let forward = Forward::bind(&state.body, project.g).map_err(|e| e.to_string())?;
            (state, forward)
        }
        None => {
            let body = initial_body(&project).map_err(|e| e.to_string())?;
            evolve::start(&project, body).map_err(|e| e.to_string())?
        }
    };
    Ok(Prepared {
        project,
        schedule,
        state,
        forward,
    })
}

/// Run the schedule on the calling thread. `sink` receives one snapshot per accepted step.
pub fn run_prepared(
    prepared: Prepared,
    max_steps: Option<usize>,
    scale: f64,
    mut gate: impl FnMut() -> Result<(), String>,
    mut sink: impl FnMut(StepSnapshot) -> Result<(), String>,
) -> Result<bool, String> {
    let Prepared {
        project,
        schedule,
        state,
        forward,
    } = prepared;
    let slack_tol = project.relief.tol_slack as f32;
    let mut mark = crate::common::time::wall_seconds();
    let mut observer = |record: &evolve::StepRecord,
                        state: &State,
                        fields: &evolve::StepFields|
     -> Result<(), SimError> {
        let wall = crate::common::time::wall_seconds() - mark;
        gate().map_err(SimError::io)?;
        mark = crate::common::time::wall_seconds();
        let snapshot = {
            let _span = crate::scf::timing::span("frame_pack");
            crate::app::snapshot::pack(record, state, fields, scale, slack_tol, wall)
        };
        sink(snapshot).map_err(SimError::io)
    };
    let result = evolve::run(
        &project,
        &schedule,
        state,
        forward,
        &RunOptions {
            max_steps,
            until: None,
            lag: true,
        },
        &mut observer,
    )
    .map_err(|err| err.to_string())?;
    Ok(result.complete)
}

/// One thread for `evolve::run`, taking the runs in order.
///
/// It is not a Rayon worker, so the parallel loops of a step use the global pool
/// (every core); inside a one-thread pool they would all run on that thread. A window
/// close must not join a Metal step, so the thread is never joined and lives until
/// the process exits. The wasm viewer has no such thread; a Web Worker runs the same
/// `prepare` and `run_prepared`.
#[derive(Resource)]
pub struct EvolutionPool {
    #[cfg(not(target_arch = "wasm32"))]
    jobs: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
}

impl EvolutionPool {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (jobs, queue) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send>>();
            std::thread::Builder::new()
                .name("tier2-evolution".into())
                .stack_size(64 << 20)
                .spawn(move || {
                    for job in queue {
                        job();
                    }
                })
                .expect("tier 2 evolution thread");
            Self { jobs }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {}
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn spawn_job<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // The thread outlives every sender, so the send cannot fail.
        let _ = self.jobs.send(Box::new(job));
    }
}

impl Default for EvolutionPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Submit Tier 2 to `pool`. The channel holds two snapshots and then blocks.
pub fn spawn(
    config: &AppConfig,
    scale: f64,
    control: Arc<Control>,
    pool: &EvolutionPool,
) -> Receiver<WorkerMessage> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (control, pool);
        crate::web::spawn_evolution(config, scale)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let config = config.clone();
        let (tx, rx) = mpsc::sync_channel(2);
        pool.spawn_job(move || {
            let send =
                |message: WorkerMessage| tx.send(message).map_err(|_| "viewer closed".to_string());
            let outcome = (|| -> Result<String, String> {
                send(WorkerMessage::Status(
                    "computing the first tier 2 step".into(),
                ))?;
                let prepared = prepare(
                    &config.project_text,
                    &config.project_dir,
                    &config.spin,
                    config.snapshot.as_deref(),
                )?;
                let initial = crate::app::snapshot::pack_initial(
                    &prepared.state,
                    prepared.forward.moments.mass,
                    scale,
                    prepared.project.relief.tol_slack as f32,
                );
                send(WorkerMessage::Status("tier 2 running".into()))?;
                send(WorkerMessage::Frame(Box::new(initial)))?;
                let complete = run_prepared(
                    prepared,
                    config.max_steps,
                    scale,
                    || control.wait_if_paused(),
                    |snapshot| send(WorkerMessage::Frame(Box::new(snapshot))),
                )?;
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
        });
        rx
    }
}
