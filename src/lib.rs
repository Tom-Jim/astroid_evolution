pub mod common;
pub mod gpu;

#[cfg(feature = "tier1")]
pub mod rtfp;

#[cfg(feature = "tier1")]
pub mod io;

#[cfg(feature = "rheo")]
pub mod rheo;

pub mod shsmooth;

#[cfg(feature = "tier2")]
pub mod scf;

#[cfg(feature = "tier3")]
pub mod app;

#[cfg(all(target_arch = "wasm32", feature = "bevy-web"))]
pub mod web;

/// wasm-bindgen entry points for the browser build, which Trunk builds from this
/// crate as a cdylib.
#[cfg(all(target_arch = "wasm32", feature = "bevy-web"))]
mod wasm_entry {
    use wasm_bindgen::prelude::*;

    /// Project and spin schedule the page opens without query parameters, the same
    /// run the desktop shell opens.
    #[wasm_bindgen]
    pub fn viewer_defaults() -> String {
        use crate::app::resources::{DATA_DIR, DEFAULT_PROJECT, DEFAULT_SPIN};
        serde_json::json!({
            "project": DEFAULT_PROJECT.strip_prefix(DATA_DIR).unwrap_or(DEFAULT_PROJECT),
            "spin": DEFAULT_SPIN,
        })
        .to_string()
    }

    #[wasm_bindgen]
    pub fn install_asset(path: &str, bytes: &[u8]) {
        crate::io::install_asset(std::path::Path::new(path), bytes.to_vec());
    }

    #[wasm_bindgen]
    pub async fn prepare_viewer(config_json: &str) -> Result<(), JsValue> {
        crate::web::prepare_viewer(config_json)
            .await
            .map_err(|err| JsValue::from_str(&err))
    }

    #[wasm_bindgen]
    pub fn boot(config_json: &str) {
        crate::web::boot(config_json);
    }

    /// Record module spans in this worker (the page's `?timing` query).
    #[wasm_bindgen]
    pub fn enable_timing() {
        crate::scf::timing::ensure_enabled();
    }

    #[wasm_bindgen]
    pub fn worker_main(config_json: &str) {
        crate::web::worker_main(config_json);
    }

    #[wasm_bindgen]
    pub async fn execute_gpu(request: &[u8]) -> Result<Vec<u8>, JsValue> {
        crate::web::execute_gpu(request)
            .await
            .map_err(|err| JsValue::from_str(&err))
    }

    /// One JSON command from the control panels (`app::bridge::Command`).
    #[wasm_bindgen]
    pub fn ui_command(json: &str) -> Result<(), JsValue> {
        crate::app::bridge::send(json).map_err(|err| JsValue::from_str(&err))
    }

    #[wasm_bindgen]
    pub fn deliver(kind: &str, payload: &str) {
        crate::web::deliver(kind, payload);
    }
}
