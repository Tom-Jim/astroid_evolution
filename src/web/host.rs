//! Page boot, the evolution worker, and the snapshot channel back into Bevy.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use crate::app::pipeline::{self, WorkerMessage};
use crate::app::resources::AppConfig;
use crate::app::snapshot::StepSnapshot;

use super::wire;

/// Channel of the current run; a restart replaces it.
static SENDER: Mutex<Option<Sender<WorkerMessage>>> = Mutex::new(None);

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

    #[wasm_bindgen(js_namespace = asteroid, js_name = status)]
    fn status_js(json: &str);

    #[wasm_bindgen(js_namespace = asteroid, js_name = setFlag)]
    fn set_flag(index: u32, value: u32);

    #[wasm_bindgen(js_namespace = asteroid, js_name = waitGate)]
    fn wait_gate_js() -> u32;

    #[wasm_bindgen(js_namespace = asteroid, js_name = download)]
    fn download(name: &str, bytes: &[u8]);
}

#[derive(serde::Deserialize)]
struct BootConfig {
    project: String,
    spin: String,
    #[serde(default)]
    snapshot: Option<String>,
    #[serde(default)]
    max_steps: Option<usize>,
}

fn hook_panic() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()));
    }));
}

fn load_config(config_json: &str) -> Result<AppConfig, String> {
    let cfg: BootConfig =
        serde_json::from_str(config_json).map_err(|err| format!("viewer config: {err}"))?;
    AppConfig::load(
        std::path::Path::new(&cfg.project),
        &cfg.spin,
        cfg.max_steps,
        cfg.snapshot.map(PathBuf::from),
    )
}

/// Check that the project and its mesh are installed before Bevy starts.
pub async fn prepare_viewer(config_json: &str) -> Result<(), String> {
    load_config(config_json).map(|_| ())
}

/// Start Bevy on the page canvas. Assets are already in the memory table.
pub fn boot(config_json: &str) {
    hook_panic();
    match load_config(config_json) {
        Ok(config) => {
            crate::app::bridge::set_sink(status_js);
            crate::app::bridge::set_saver(|name, bytes| {
                download(name, bytes);
                Ok(format!("downloaded {name}"))
            });
            crate::app::run_viewer(config)
        }
        Err(err) => {
            web_sys::console::error_1(&JsValue::from_str(&err));
            note(&err);
        }
    }
}

/// The cache file the page looks for before it computes a run (`app::cache::name`).
/// The project of the request is installed already.
pub fn cache_name(request_json: &str) -> Result<String, String> {
    let request: crate::app::bridge::RunRequest =
        serde_json::from_str(request_json).map_err(|err| format!("run request: {err}"))?;
    let mut config = AppConfig::load(
        std::path::Path::new(&request.project),
        &request.spin,
        request.max_steps,
        request.snapshot.map(PathBuf::from),
    )?;
    config.apply_overrides(&request.overrides)?;
    Ok(crate::app::cache::name(&config, std::path::Path::new("")))
}

/// The run request of a cache file, so the page can install what the run needs.
pub fn cache_header(bytes: &[u8]) -> Result<String, String> {
    let (header, _) = crate::app::cache::header(bytes)?;
    Ok(serde_json::json!({
        "project": header.project,
        "spin": header.spin,
        "snapshot": header.snapshot,
        "max_steps": header.max_steps,
    })
    .to_string())
}

/// Hand the same run to a new worker (the page stops the previous one). Bevy keeps
/// polling `rx`.
pub fn spawn_evolution(config: &AppConfig, scale: f64) -> Receiver<WorkerMessage> {
    let (tx, rx) = mpsc::channel();
    *SENDER.lock().unwrap() = Some(tx);
    let worker = serde_json::json!({
        "project_text": config.project_text,
        "project_dir": config.project_dir.display().to_string(),
        "spin": config.spin,
        "max_steps": config.max_steps,
        "snapshot": config.snapshot.as_ref().map(|p| p.display().to_string()),
        "scale": scale,
    });
    start_worker(&worker.to_string());
    rx
}

#[derive(serde::Deserialize)]
struct WorkerConfig {
    project_text: String,
    project_dir: String,
    spin: String,
    max_steps: Option<usize>,
    snapshot: Option<String>,
    scale: f64,
}

fn emit(kind: &str, payload: &str) {
    emit_message(kind, payload);
}

fn emit_frame(snapshot: &StepSnapshot) -> Result<(), String> {
    emit("frame", &wire::encode_snapshot(snapshot));
    Ok(())
}

/// The spans since the last frame, `[name, calls, seconds]` rows, when timing is on.
fn emit_timing() {
    use crate::scf::timing;
    if timing::active() {
        let rows: Vec<_> = timing::drain()
            .into_iter()
            .map(|a| serde_json::json!([a.name, a.calls, a.seconds]))
            .collect();
        emit("timing", &serde_json::Value::from(rows).to_string());
    }
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
            cfg.snapshot.as_deref().map(std::path::Path::new),
        )?;
        let initial = crate::app::snapshot::pack_initial(
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
            |snapshot| {
                let sent = {
                    let _span = crate::scf::timing::span("web_frame");
                    emit_frame(&snapshot)
                };
                emit_timing();
                sent
            },
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
    let slot = SENDER.lock().unwrap();
    let Some(tx) = slot.as_ref() else {
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
    // Only the evolution worker's own thread has the GPU bridge; a rayon pool thread does not.
    if rayon::current_thread_index().is_some() {
        return Err("a GPU call from a rayon pool thread: GPU work must stay on the worker thread".into());
    }
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
