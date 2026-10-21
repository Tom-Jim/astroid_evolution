//! Wasm entry for the same Tier 3 viewer. The native `asteroid-app` binary is
//! unchanged. This crate only exists so the main library can stay an rlib.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn install_asset(path: &str, bytes: &[u8]) {
    astroid_evolution::rtfp_solver::io::install_asset(std::path::Path::new(path), bytes.to_vec());
}

#[wasm_bindgen]
pub async fn prepare_viewer(config_json: &str) -> Result<(), JsValue> {
    astroid_evolution::browser::prepare_viewer(config_json)
        .await
        .map_err(|err| JsValue::from_str(&err))
}

#[wasm_bindgen]
pub fn boot(config_json: &str) {
    astroid_evolution::browser::boot(config_json);
}

#[wasm_bindgen]
pub fn worker_main(config_json: &str) {
    astroid_evolution::browser::worker_main(config_json);
}

#[wasm_bindgen]
pub async fn execute_gpu(request: &[u8]) -> Result<Vec<u8>, JsValue> {
    astroid_evolution::browser::execute_gpu(request)
        .await
        .map_err(|err| JsValue::from_str(&err))
}

#[wasm_bindgen]
pub fn deliver(kind: &str, payload: &str) {
    astroid_evolution::browser::deliver(kind, payload);
}
