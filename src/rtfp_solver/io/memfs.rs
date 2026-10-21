//! In-memory files for the wasm viewer.
//!
//! Native mesh loading keeps using `std::fs`. The browser installs the same
//! project and mesh bytes here before the first read, under the path
//! `Project::resolve` produces after `.` and `..` are collapsed.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

static FILES: Mutex<Option<HashMap<PathBuf, Vec<u8>>>> = Mutex::new(None);

fn table() -> std::sync::MutexGuard<'static, Option<HashMap<PathBuf, Vec<u8>>>> {
    FILES.lock().unwrap_or_else(|err| err.into_inner())
}

/// Collapse `.` and `..` so `assets/projects/../models/ryugu.glb` matches the
/// key the page installed.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(piece) => out.push(piece),
            Component::RootDir => out.push("/"),
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
        }
    }
    out
}

/// Replace the bytes stored for `path`. Later reads see this copy.
pub fn install(path: &Path, bytes: Vec<u8>) {
    let key = normalize(path);
    table().get_or_insert_with(HashMap::new).insert(key, bytes);
}

pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    let key = normalize(path);
    table()
        .as_ref()
        .and_then(|files| files.get(&key).cloned())
        .ok_or_else(|| format!("wasm asset {} is not loaded", key.display()))
}

pub fn read_to_string(path: &Path) -> Result<String, String> {
    let bytes = read(path)?;
    String::from_utf8(bytes).map_err(|err| format!("wasm asset {}: {err}", path.display()))
}
