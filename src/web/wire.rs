//! Snapshot bytes (postcard, base64) between the worker instance and the page instance.
//! postcard keeps every f64 bit for bit, including the infinite step bounds that JSON
//! cannot carry; base64 fits the string channel of the worker.

use base64::Engine as _;

use crate::app::snapshot::StepSnapshot;

pub fn encode_snapshot(snapshot: &StepSnapshot) -> String {
    let bytes = postcard::to_stdvec(snapshot).unwrap_or_default();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn decode_snapshot(payload: &str) -> Result<StepSnapshot, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|err| format!("snapshot: {err}"))?;
    postcard::from_bytes(&bytes).map_err(|err| format!("snapshot: {err}"))
}
