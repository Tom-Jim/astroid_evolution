//! Cached runs (`.aecache`): the viewer's stored timeline of a run, so that a run can be
//! played back without computing it.
//!
//! A file is the magic `AECACHE1`, the length (u32, little endian) and postcard bytes of
//! the header, then the deflated postcard bytes of the frames. The header is readable
//! without inflating the frames, so the page can fetch what a run needs before it
//! hands the file over. The browser looks for the cache of a request first and
//! computes only when there is none; the desktop computes. Both import and export.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::app::history::{History, Stored};
use crate::app::resources::AppConfig;

const MAGIC: &[u8; 8] = b"AECACHE1";
/// Directory of the cache files, relative to the data root.
pub const DIR: &str = "cache";
/// Frames kept in an exported file (evenly thinned; the first and newest are kept).
pub const MAX_FRAMES: usize = 480;
/// Seconds each cached frame takes on screen during playback.
pub const REPLAY_SECONDS: f32 = 0.35;
/// Largest inflated body accepted.
const MAX_BODY: usize = 2 << 30;

/// `path` relative to the data root `root` (the desktop loads absolute paths).
fn relative(root: &Path, path: &Path) -> String {
    let root = crate::io::memfs::normalize(root);
    let path = crate::io::memfs::normalize(path);
    path.strip_prefix(&root).unwrap_or(&path).display().to_string()
}

/// The run a file holds: the request (paths relative to the data root), and the
/// project text after the overrides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub project: String,
    pub spin: String,
    pub max_steps: Option<usize>,
    pub snapshot: Option<String>,
    pub project_text: String,
}

impl Header {
    pub fn of(config: &AppConfig, root: &Path) -> Self {
        Self {
            project: relative(root, Path::new(&config.project_path)),
            spin: config.spin.clone(),
            max_steps: config.max_steps,
            snapshot: config.snapshot.as_ref().map(|p| relative(root, p)),
            project_text: config.project_text.clone(),
        }
    }
}

/// The cache file of a run, relative to the data root: `cache/<project stem>-<key>.aecache`
/// with `key` the CRC32 of everything that defines the run. The browser and the
/// desktop give the same run the same name.
pub fn name(config: &AppConfig, root: &Path) -> String {
    let header = Header::of(config, root);
    let stem = Path::new(&header.project)
        .file_stem()
        .map_or_else(|| "run".to_string(), |s| s.to_string_lossy().into_owned());
    let mut h = crc32fast::Hasher::new();
    h.update(header.project_text.as_bytes());
    h.update(header.spin.as_bytes());
    h.update(&(header.max_steps.unwrap_or(0) as u64).to_le_bytes());
    if let Some(s) = &header.snapshot {
        h.update(s.as_bytes());
    }
    format!("{DIR}/{stem}-{:08x}.aecache", h.finalize())
}

pub fn encode(header: &Header, history: &History) -> Result<Vec<u8>, String> {
    let head = postcard::to_stdvec(header).map_err(|e| format!("cache: {e}"))?;
    let body = postcard::to_stdvec(&history.to_stored(MAX_FRAMES)).map_err(|e| format!("cache: {e}"))?;
    let mut out = MAGIC.to_vec();
    out.extend((head.len() as u32).to_le_bytes());
    out.extend(head);
    out.extend(miniz_oxide::deflate::compress_to_vec(&body, 6));
    Ok(out)
}

/// The header, and the deflated frames after it.
pub fn header(bytes: &[u8]) -> Result<(Header, &[u8]), String> {
    let rest = bytes
        .strip_prefix(MAGIC.as_slice())
        .ok_or("cache: not an .aecache file")?;
    let (len, rest) = rest.split_first_chunk::<4>().ok_or("cache: truncated file")?;
    let len = u32::from_le_bytes(*len) as usize;
    if rest.len() < len {
        return Err("cache: truncated file".into());
    }
    let (head, body) = rest.split_at(len);
    let header = postcard::from_bytes(head).map_err(|e| format!("cache: {e}"))?;
    Ok((header, body))
}

pub fn decode(bytes: &[u8]) -> Result<(Header, History), String> {
    let (header, body) = header(bytes)?;
    let raw = miniz_oxide::inflate::decompress_to_vec_with_limit(body, MAX_BODY)
        .map_err(|e| format!("cache: {e:?}"))?;
    let stored: Stored = postcard::from_bytes(&raw).map_err(|e| format!("cache: {e}"))?;
    if stored.is_empty() {
        return Err("cache: the file holds no frame".into());
    }
    Ok((header, History::from_stored(stored)?))
}
