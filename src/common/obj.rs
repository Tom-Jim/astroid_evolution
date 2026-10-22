//! Wavefront OBJ in the one dialect every module reads and writes: `v x y z`,
//! `f a b c` (1-based, negative indices count back), `#` comments, and a
//! `# length_unit` header.

use std::fmt::Write as _;
use std::path::Path;

use super::error::SimError;

pub struct Obj {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    /// Value of a `# length_unit` comment, when the file has one.
    pub length_unit: Option<String>,
    /// Polygons with more than three corners, fanned into triangles.
    pub fans: usize,
}

/// `strict` keeps the face list exactly: only triangles, no repeated corner, at
/// least one face. Otherwise polygons are fanned and a bare `x y z` line is a vertex.
pub fn parse(text: &str, strict: bool) -> Result<Obj, SimError> {
    let mut out = Obj {
        vertices: Vec::new(),
        faces: Vec::new(),
        length_unit: None,
        fans: 0,
    };
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let fail = |what: &str| SimError::model_not_closed(format!("line {}: {what}", n + 1));
        if let Some(unit) = line.strip_prefix("# length_unit ") {
            out.length_unit = Some(unit.trim().to_string());
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let head = parts.next().unwrap_or("");
        let bare = !strict && head.parse::<f64>().is_ok();
        if head == "v" || bare {
            let coords = if bare {
                std::iter::once(head).chain(parts).collect()
            } else {
                parts.collect::<Vec<_>>()
            };
            let c: Vec<f64> = coords
                .iter()
                .take(3)
                .map(|t| t.parse().unwrap_or(f64::NAN))
                .collect();
            if c.len() < 3 || c.iter().any(|v| !v.is_finite()) {
                return Err(fail("vertex is not three finite coordinates"));
            }
            out.vertices.push([c[0], c[1], c[2]]);
        } else if head == "f" {
            let nv = out.vertices.len() as i64;
            let ids = parts
                .map(|t| {
                    let id: i64 = t
                        .split('/')
                        .next()
                        .unwrap_or(t)
                        .parse()
                        .map_err(|_| fail("face index is not an integer"))?;
                    match id {
                        1.. if id <= nv => Ok((id - 1) as u32),
                        ..=-1 if id >= -nv => Ok((nv + id) as u32),
                        _ => Err(fail("face index is outside the vertex array")),
                    }
                })
                .collect::<Result<Vec<u32>, _>>()?;
            if ids.len() < 3 || (strict && ids.len() != 3) {
                return Err(fail(&format!(
                    "face has {} corners; only triangles keep the face count unchanged",
                    ids.len()
                )));
            }
            if strict && (ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0]) {
                return Err(fail(
                    "degenerate triangle with a repeated vertex; faces are not dropped",
                ));
            }
            out.fans += usize::from(ids.len() > 3);
            out.faces
                .extend((1..ids.len() - 1).map(|c| [ids[0], ids[c], ids[c + 1]]));
        }
    }
    if strict && (out.vertices.len() < 3 || out.faces.is_empty()) {
        return Err(SimError::model_not_closed(
            "mesh needs at least 3 vertices and 1 triangle",
        ));
    }
    Ok(out)
}

/// CRC-32 of the coordinates and indices: the `mesh_hash` of every written file.
pub fn hash(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> String {
    let mut hasher = crc32fast::Hasher::new();
    for c in vertices.iter().flatten() {
        hasher.update(&c.to_le_bytes());
    }
    for i in faces.iter().flatten() {
        hasher.update(&i.to_le_bytes());
    }
    format!("{:08x}", hasher.finalize())
}

/// `header` (comment lines, each ending in a newline), then the vertices and faces.
pub fn write(
    path: &Path,
    header: &str,
    vertices: impl IntoIterator<Item = impl std::borrow::Borrow<[f64; 3]>>,
    faces: &[[u32; 3]],
) -> Result<(), SimError> {
    let mut text = header.to_string();
    for v in vertices {
        let v = v.borrow();
        let _ = writeln!(text, "v {:.17} {:.17} {:.17}", v[0], v[1], v[2]);
    }
    for f in faces {
        let _ = writeln!(text, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, text)
        .map_err(|e| SimError::io(format!("cannot write {}: {e}", path.display())))
}
