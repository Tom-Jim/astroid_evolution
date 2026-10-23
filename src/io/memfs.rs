//! File reads for every target. The browser installs the project and mesh
//! bytes here before the first read, under the path `Project::resolve`
//! produces after `.` and `..` are collapsed; native builds fall back to disk.

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
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::Normal(piece) => out.push(piece),
            Component::RootDir => out.push("/"),
            Component::Prefix(prefix) => out.push(prefix.as_os_str()),
        }
    }
    out
}

/// A path that stays inside the directory it is joined to: relative, and no `..` that
/// climbs above that directory once `.` and `..` are collapsed. The viewer's run requests
/// name the project and the snapshot this way, relative to the data root.
pub fn relative_inside(path: &str) -> Result<PathBuf, String> {
    let raw = Path::new(path);
    if raw.is_absolute() || raw.has_root() {
        return Err(format!("path {path} must be relative to the data root"));
    }
    let norm = normalize(raw);
    match norm.components().next() {
        Some(Component::ParentDir) => Err(format!("path {path} leaves the data root")),
        Some(_) => Ok(norm),
        None => Err("empty path".to_string()),
    }
}

/// Replace the bytes stored for `path`. Later reads see this copy.
pub fn install(path: &Path, bytes: Vec<u8>) {
    let key = normalize(path);
    table().get_or_insert_with(HashMap::new).insert(key, bytes);
}

pub fn read(path: &Path) -> Result<Vec<u8>, String> {
    let key = normalize(path);
    if let Some(bytes) = table().as_ref().and_then(|files| files.get(&key).cloned()) {
        return Ok(bytes);
    }
    #[cfg(not(target_arch = "wasm32"))]
    return std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()));
    #[cfg(target_arch = "wasm32")]
    Err(format!("wasm asset {} is not loaded", key.display()))
}

pub fn read_to_string(path: &Path) -> Result<String, String> {
    let bytes = read(path)?;
    String::from_utf8(bytes).map_err(|err| format!("{}: {err}", path.display()))
}
