//! Page boot, the evolution worker, and the snapshot channel back into Bevy.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};

use wasm_bindgen::prelude::*;

use crate::asteroid_app::pipeline::{self, WorkerMessage};
use crate::asteroid_app::resources::AppConfig;
use crate::asteroid_app::snapshot::StepSnapshot;
use crate::scf_simulator::params::Project;

use super::wire;

static SENDER: OnceLock<Mutex<Sender<WorkerMessage>>> = OnceLock::new();

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch, js_namespace = asteroid, js_name = gpuCall)]
    pub fn gpu_call(payload: &[u8]) -> Result<Vec<u8>, JsValue>;

    #[wasm_bindgen(js_namespace = asteroid, js_name = emit)]
    fn emit_message(kind: &str, payload: &str);

    #[wasm_bindgen(js_namespace = asteroid, js_name = startWorker)]
    fn start_worker(config: &str);

    #[wasm_bindgen(js_namespace = asteroid, js_name = note)]
    fn note(text: &str);

    #[wasm_bindgen(js_namespace = asteroid, js_name = setFlag)]
    fn set_flag(index: u32, value: u32);

    #[wasm_bindgen(js_namespace = asteroid, js_name = waitGate)]
    fn wait_gate_js() -> u32;
}

#[derive(serde::Deserialize)]
struct BootConfig {
    project: String,
    spin: String,
    #[serde(default)]
    max_steps: Option<usize>,
}

fn hook_panic() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
    }));
}

fn load_config(config_json: &str) -> Result<(AppConfig, f64), String> {
    let cfg: BootConfig =
        serde_json::from_str(config_json).map_err(|err| format!("viewer config: {err}"))?;
    let project_path = PathBuf::from(&cfg.project);
    let text = crate::rtfp_solver::io::memfs::read_to_string(&project_path)?;
    let dir = project_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let project = Project::parse(&text, dir.clone()).map_err(|err| err.to_string())?;
    let summary = pipeline::mesh_summary(&project)?;
    let scene = pipeline::view_fit(&project)?;
    Ok((
        AppConfig {
            project_text: text,
            project_dir: dir,
            project_path: cfg.project,
            mesh_path: summary.path.display().to_string(),
            mesh_unit: summary.unit,
            spin: cfg.spin,
            max_steps: cfg.max_steps,
        },
        scene.scale,
    ))
}

/// Decode the project mesh on this page's GPU before Bevy asks for it.
pub async fn prepare_viewer(config_json: &str) -> Result<(), String> {
    note("decoding the mesh on the GPU");
    let cfg: BootConfig =
        serde_json::from_str(config_json).map_err(|err| format!("viewer config: {err}"))?;
    let project_path = PathBuf::from(&cfg.project);
    let text = crate::rtfp_solver::io::memfs::read_to_string(&project_path)?;
    let dir = project_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let project = Project::parse(&text, dir).map_err(|err| err.to_string())?;
    if let Some(mesh) = project.body.mesh.as_deref() {
        crate::rtfp_solver::io::preload_viewer_mesh(
            &project.resolve(mesh),
            project.body.mesh_unit.as_deref(),
        )
        .await
        .map_err(|err| err.to_string())?;
    }
    Ok(())
}

/// Start Bevy on the page canvas. Assets are already in the memory table.
pub fn boot(config_json: &str) {
    hook_panic();
    match load_config(config_json) {
        Ok((config, _)) => crate::asteroid_app::run_viewer(config),
        Err(err) => {
            web_sys::console::error_1(&JsValue::from_str(&err));
            note(&err);
        }
    }
}

/// Hand the same project to the worker. Bevy keeps polling `rx`.
pub fn spawn_evolution(
    project_text: String,
    project_dir: PathBuf,
    spin: String,
    max_steps: Option<usize>,
    scale: f64,
) -> Receiver<WorkerMessage> {
    let (tx, rx) = mpsc::channel();
    let _ = SENDER.set(Mutex::new(tx));
    let config = serde_json::json!({
        "project_text": project_text,
        "project_dir": project_dir.display().to_string(),
        "spin": spin,
        "max_steps": max_steps,
        "scale": scale,
    });
    start_worker(&config.to_string());
    rx
}

#[derive(serde::Deserialize)]
struct WorkerConfig {
    project_text: String,
    project_dir: String,
    spin: String,
    max_steps: Option<usize>,
    scale: f64,
}

fn emit(kind: &str, payload: &str) {
    emit_message(kind, payload);
}

fn emit_frame(snapshot: &StepSnapshot) -> Result<(), String> {
    emit("frame", &wire::encode_snapshot(snapshot));
    Ok(())
}

/// Evolution entry on the worker instance.
pub fn worker_main(config_json: &str) {
    super::use_worker_mesh();
    hook_panic();
    let outcome = (|| -> Result<String, String> {
        let cfg: WorkerConfig =
            serde_json::from_str(config_json).map_err(|err| format!("worker config: {err}"))?;
        emit("status", "computing the first tier 2 step");
        let prepared = pipeline::prepare(
            &cfg.project_text,
            PathBuf::from(&cfg.project_dir).as_path(),
            &cfg.spin,
        )?;
        let initial = crate::asteroid_app::snapshot::pack_initial(
            &prepared.state,
            prepared.forward.moments.mass,
            cfg.scale,
            prepared.project.relief.tol_slack as f32,
        );
        emit("status", "tier 2 running");
        emit_frame(&initial)?;
        let complete = pipeline::run_prepared(
            prepared,
            cfg.max_steps,
            cfg.scale,
            || {
                if wait_gate_js() != 0 {
                    Err("cancelled".into())
                } else {
                    Ok(())
                }
            },
            |snapshot| emit_frame(&snapshot),
        )?;
        Ok(if complete {
            "schedule complete".into()
        } else {
            "stopped at the step limit".into()
        })
    })();
    match outcome {
        Ok(message) => emit("finished", &message),
        Err(message) => emit("failed", &message),
    }
}

/// The page received a worker message. Push it into the channel Bevy polls.
pub fn deliver(kind: &str, payload: &str) {
    let Some(slot) = SENDER.get() else {
        return;
    };
    let Ok(tx) = slot.lock() else {
        return;
    };
    let message = match kind {
        "status" => WorkerMessage::Status(payload.to_string()),
        "frame" => match wire::decode_snapshot(payload) {
            Ok(snapshot) => WorkerMessage::Frame(Box::new(snapshot)),
            Err(err) => WorkerMessage::Failed(err),
        },
        "finished" => WorkerMessage::Finished(payload.to_string()),
        "failed" => WorkerMessage::Failed(payload.to_string()),
        _ => return,
    };
    let _ = tx.send(message);
}

pub fn set_pause_flag(paused: bool) {
    set_flag(2, u32::from(paused));
}

pub fn set_quit_flag() {
    set_flag(3, 1);
    set_flag(2, 0);
}

pub fn call_gpu(payload: &[u8]) -> Result<Vec<u8>, String> {
    gpu_call(payload).map_err(js_error)
}

fn js_error(err: JsValue) -> String {
    if let Some(text) = err.as_string() {
        return text;
    }
    if let Ok(message) = js_sys::Reflect::get(&err, &JsValue::from_str("message")) {
        if let Some(text) = message.as_string() {
            if !text.is_empty() {
                return text;
            }
        }
    }
    format!("{err:?}")
}
