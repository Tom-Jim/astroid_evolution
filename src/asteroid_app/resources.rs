use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;

use bevy::prelude::*;

use crate::asteroid_app::pipeline::{Control, WorkerMessage};
use crate::sh_smooth::Mesh as SurfaceMesh;

/// Inputs of the viewer, written by the binary.
#[derive(Resource, Clone)]
pub struct AppConfig {
    pub project_text: String,
    pub project_dir: PathBuf,
    /// Spin schedule grammar of `scf-cli run-sim --spin`.
    pub spin: String,
    pub max_steps: Option<usize>,
}

/// The original model and the transform that puts it in the view (unit-ish radius).
#[derive(Resource)]
pub struct Scene {
    pub original: Arc<SurfaceMesh>,
    pub center: [f64; 3],
    pub scale: f64,
}

impl Scene {
    pub fn new(original: Arc<SurfaceMesh>) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in &original.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        let center = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let reach = original
            .vertices
            .iter()
            .map(|v| {
                (0..3)
                    .map(|k| (v[k] - center[k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max)
            .max(1.0e-12);
        Self {
            original,
            center,
            scale: 1.0 / reach,
        }
    }

    pub fn view_positions(&self, vertices: &[[f64; 3]]) -> Vec<[f32; 3]> {
        vertices
            .iter()
            .map(|v| {
                [
                    ((v[0] - self.center[0]) * self.scale) as f32,
                    ((v[1] - self.center[1]) * self.scale) as f32,
                    ((v[2] - self.center[2]) * self.scale) as f32,
                ]
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Showing the original model; Space starts the evolution.
    Idle,
    Preparing,
    Running,
    Paused,
    Done,
    Failed,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub step: usize,
    pub time: f64,
    pub omega: f64,
    pub mass: f64,
    pub delta_min: f64,
    pub delta_max: f64,
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
            status: "original model - press Space to evolve".into(),
            control: Arc::default(),
            stats: None,
            receiver: Mutex::new(None),
        }
    }
}
