//! Triangle-mesh import. Node transforms are applied. A vertex list without faces
//! stays a vertex list until the caller names a reconstruction.

use super::length_to_meters;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::error::RtfpError;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reconstruct {
    ConvexHull,
    RadialRaycast,
}

pub struct ImportRequest<'a> {
    pub path: &'a Path,
    pub length_unit: Option<&'a str>,
    pub reconstruct: Option<Reconstruct>,
}

#[derive(Clone)]
pub struct MeshDocument {
    pub mesh: TriMesh,
    /// Unit of the coordinates stored in `mesh`. Physical imports are converted to meters.
    pub length_unit: String,
    pub source_length_unit: String,
    pub mesh_hash: String,
    pub welded_positions: usize,
    pub dropped_degenerate: usize,
    pub triangulated_strips: usize,
    pub triangulated_fans: usize,
    pub source: String,
}

pub fn mesh_hash(mesh: &TriMesh) -> String {
    let mut hasher = crc32fast::Hasher::new();
    for vertex in &mesh.vertices {
        hasher.update(&vertex.x.to_le_bytes());
        hasher.update(&vertex.y.to_le_bytes());
        hasher.update(&vertex.z.to_le_bytes());
    }
    for face in &mesh.faces {
        for index in face {
            hasher.update(&index.to_le_bytes());
        }
    }
    format!("{:08x}", hasher.finalize())
}

fn read_bytes(path: &Path) -> Result<Vec<u8>, RtfpError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read(path).map_err(|err| {
            RtfpError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        super::memfs::read(path).map_err(|err| {
            RtfpError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })
    }
}

fn read_text(path: &Path) -> Result<String, RtfpError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::fs::read_to_string(path).map_err(|err| {
            RtfpError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        super::memfs::read_to_string(path).map_err(|err| {
            RtfpError::model_not_closed(format!("cannot read {}: {err}", path.display()))
        })
    }
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
struct ViewerMesh {
    key: String,
    document: MeshDocument,
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
static VIEWER_MESH: std::sync::Mutex<Option<ViewerMesh>> = std::sync::Mutex::new(None);

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn viewer_mesh_key(path: &Path, length_unit: Option<&str>) -> String {
    format!("{}|{}", path.display(), length_unit.unwrap_or(""))
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn cached_viewer_mesh(path: &Path, length_unit: Option<&str>) -> Option<MeshDocument> {
    let key = viewer_mesh_key(path, length_unit);
    VIEWER_MESH
        .lock()
        .ok()?
        .as_ref()
        .filter(|cached| cached.key == key)
        .map(|cached| cached.document.clone())
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn store_viewer_mesh(path: &Path, length_unit: Option<&str>, document: &MeshDocument) {
    if let Ok(mut slot) = VIEWER_MESH.lock() {
        *slot = Some(ViewerMesh {
            key: viewer_mesh_key(path, length_unit),
            document: document.clone(),
        });
    }
}

/// GPU-decode a glTF mesh on the page, then keep the document for the viewer.
/// The evolution worker decodes again through its own device gate.
#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
pub async fn preload_viewer_mesh(path: &Path, length_unit: Option<&str>) -> Result<(), RtfpError> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "glb" && ext != "gltf" {
        // The page has no worker GPU gate yet. Parsing here only feeds the
        // canvas; the evolution worker repeats the solid checks.
        let (unit, scale) = resolve_length(path, length_unit)?;
        let parsed = if ext == "obj" || ext == "xyz" {
            parse_obj(&read_text(path)?)?
        } else {
            return Err(RtfpError::model_not_closed(format!(
                "mesh extension '{ext}' is not obj, xyz, glb, or gltf"
            )));
        };
        let document = document_from_parsed(unit, scale, parsed, None, false)?;
        store_viewer_mesh(path, length_unit, &document);
        return Ok(());
    }
    let (unit, scale) = resolve_length(path, length_unit)?;
    let parsed = parse_gltf_gpu(path).await?;
    // The page cannot wait on the worker GPU gate. Solid checks run again on
    // the evolution worker, which can post those dispatches.
    let document = document_from_parsed(unit, scale, parsed, None, false)?;
    store_viewer_mesh(path, length_unit, &document);
    Ok(())
}

pub fn load_mesh_file(request: ImportRequest<'_>) -> Result<MeshDocument, RtfpError> {
    #[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
    if let Some(document) = cached_viewer_mesh(request.path, request.length_unit) {
        return Ok(document);
    }
    #[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
    let mesh_path = request.path;
    #[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
    let mesh_unit = request.length_unit;
    let document = load_mesh_file_uncached(request)?;
    #[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
    store_viewer_mesh(mesh_path, mesh_unit, &document);
    Ok(document)
}

fn resolve_length(path: &Path, length_unit: Option<&str>) -> Result<(String, f64), RtfpError> {
    let text_unit = declared_length_unit(path)?;
    let unit = match (length_unit, text_unit.as_deref()) {
        (Some(flag), Some(file)) if flag != file => {
            return Err(RtfpError::model_not_closed(format!(
                "length unit {flag} does not match the unit {file} declared in {}",
                path.display()
            )));
        }
        (Some(flag), _) => flag.to_string(),
        (None, Some(file)) => file.to_string(),
        (None, None) => {
            return Err(RtfpError::model_not_closed(
                "mesh length unit is missing; pass --length-unit or import a mesh that declares one",
            ));
        }
    };
    let scale = length_to_meters(&unit)?;
    Ok((unit, scale))
}

fn load_mesh_file_uncached(request: ImportRequest<'_>) -> Result<MeshDocument, RtfpError> {
    let (unit, scale) = resolve_length(request.path, request.length_unit)?;
    let ext = request
        .path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let parsed = match ext.as_str() {
        "obj" | "xyz" => parse_obj(&read_text(request.path)?),
        "glb" | "gltf" => parse_gltf(request.path),
        other => Err(RtfpError::model_not_closed(format!(
            "mesh extension '{other}' is not obj, xyz, glb, or gltf"
        ))),
    }?;
    document_from_parsed(unit, scale, parsed, request.reconstruct, true)
}

fn document_from_parsed(
    unit: String,
    scale: f64,
    parsed: ParsedMesh,
    reconstruct: Option<Reconstruct>,
    check_solid: bool,
) -> Result<MeshDocument, RtfpError> {
    let (vertices, faces, welded, dropped) = if parsed.faces.is_empty() {
        let strategy = reconstruct.ok_or_else(|| {
            RtfpError::model_not_closed(
                "ModelNotClosed: the input has vertices and no faces; pass --reconstruct convex_hull or --reconstruct radial_raycast",
            )
        })?;
        let mesh = match strategy {
            Reconstruct::ConvexHull => convex_hull(&parsed.vertices)?,
            Reconstruct::RadialRaycast => radial_raycast(&parsed.vertices)?,
        };
        (mesh.vertices, mesh.faces, 0, 0)
    } else {
        weld_and_drop(parsed.vertices, parsed.faces)?
    };
    let vertices = vertices
        .into_iter()
        .map(|v| DVec3::new(v.x * scale, v.y * scale, v.z * scale))
        .collect();
    let mesh = if check_solid {
        TriMesh::try_new(vertices, faces)?
    } else if vertices.len() < 4 || faces.len() < 4 {
        return Err(RtfpError::model_not_closed(
            "a solid mesh needs at least 4 vertices and 4 triangles",
        ));
    } else {
        TriMesh { vertices, faces }
    };
    let mesh_hash = mesh_hash(&mesh);
    let stored_unit = if unit == "1" || unit == "dimensionless" {
        unit.clone()
    } else {
        "m".to_string()
    };
    Ok(MeshDocument {
        mesh,
        length_unit: stored_unit,
        source_length_unit: unit,
        mesh_hash,
        welded_positions: welded,
        dropped_degenerate: dropped,
        triangulated_strips: parsed.strips,
        triangulated_fans: parsed.fans,
        source: parsed.source,
    })
}

struct ParsedMesh {
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
    strips: usize,
    fans: usize,
    source: String,
}

pub fn write_obj(path: &Path, document: &MeshDocument) -> Result<(), RtfpError> {
    let mut text = String::new();
    text.push_str("# astroid_evolution_mesh 1\n");
    text.push_str(&format!("# length_unit {}\n", document.length_unit));
    text.push_str(&format!(
        "# source_length_unit {}\n",
        document.source_length_unit
    ));
    text.push_str(&format!("# mesh_hash {}\n", document.mesh_hash));
    text.push_str(&format!("# source {}\n", document.source));
    text.push_str(&format!(
        "# welded_positions {}\n# dropped_degenerate {}\n# triangulated_strips {}\n# triangulated_fans {}\n",
        document.welded_positions,
        document.dropped_degenerate,
        document.triangulated_strips,
        document.triangulated_fans
    ));
    for vertex in &document.mesh.vertices {
        text.push_str(&format!(
            "v {:.17} {:.17} {:.17}\n",
            vertex.x, vertex.y, vertex.z
        ));
    }
    for face in &document.mesh.faces {
        text.push_str(&format!(
            "f {} {} {}\n",
            face[0] + 1,
            face[1] + 1,
            face[2] + 1
        ));
    }
    std::fs::write(path, text).map_err(|err| {
        RtfpError::model_not_closed(format!("cannot write {}: {err}", path.display()))
    })
}

fn declared_length_unit(path: &Path) -> Result<Option<String>, RtfpError> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "obj" {
        return Ok(None);
    }
    let text = read_text(path)?;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("# length_unit ") {
            return Ok(Some(rest.trim().to_string()));
        }
    }
    Ok(None)
}

fn parse_obj(text: &str) -> Result<ParsedMesh, RtfpError> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut fans = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let head = parts.next();
        if let Some(token) = head {
            if token != "v" && token != "f" && token.parse::<f64>().is_ok() {
                let mut coords = vec![token.parse::<f64>().unwrap_or(f64::NAN)];
                coords.extend(parts.map(|token| token.parse::<f64>().unwrap_or(f64::NAN)));
                if coords.len() < 3 || coords.iter().any(|value| !value.is_finite()) {
                    return Err(RtfpError::model_not_closed(
                        "vertex line is not three finite coordinates",
                    ));
                }
                vertices.push(DVec3::new(coords[0], coords[1], coords[2]));
                continue;
            }
        }
        match head {
            Some("v") => {
                let coords: Vec<f64> = parts
                    .map(|token| token.parse::<f64>().unwrap_or(f64::NAN))
                    .collect();
                if coords.len() < 3 || coords.iter().any(|value| !value.is_finite()) {
                    return Err(RtfpError::model_not_closed("obj vertex is not finite"));
                }
                vertices.push(DVec3::new(coords[0], coords[1], coords[2]));
            }
            Some("f") => {
                let mut ids = Vec::new();
                for token in parts {
                    let id: i64 =
                        token
                            .split('/')
                            .next()
                            .unwrap_or(token)
                            .parse()
                            .map_err(|_| {
                                RtfpError::model_not_closed("obj face index is not an integer")
                            })?;
                    if id == 0 || id < -((vertices.len()) as i64) {
                        return Err(RtfpError::model_not_closed(
                            "obj face index is out of range",
                        ));
                    }
                    let index = if id > 0 {
                        (id - 1) as u32
                    } else {
                        (vertices.len() as i64 + id) as u32
                    };
                    ids.push(index);
                }
                if ids.len() < 3 {
                    return Err(RtfpError::model_not_closed(
                        "obj face has fewer than 3 indices",
                    ));
                }
                if ids.len() > 3 {
                    fans += 1;
                    for corner in 1..ids.len() - 1 {
                        faces.push([ids[0], ids[corner], ids[corner + 1]]);
                    }
                } else {
                    faces.push([ids[0], ids[1], ids[2]]);
                }
            }
            _ => {}
        }
    }
    Ok(ParsedMesh {
        vertices,
        faces,
        strips: 0,
        fans,
        source: "obj".into(),
    })
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
struct GpuPrim {
    pos_count: u32,
    pos_offset: u32,
    pos_stride: u32,
    index_count: u32,
    index_offset: u32,
    index_stride: u32,
    index_width: u32,
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn gpu_prim_args(
    json: &serde_json::Value,
    primitive: &serde_json::Value,
) -> Result<Option<GpuPrim>, RtfpError> {
    let mode = primitive
        .get("mode")
        .and_then(|value| value.as_u64())
        .unwrap_or(4) as u32;
    if mode != 4 || primitive.get("indices").is_none() {
        return Ok(None);
    }
    let pos_index = primitive
        .get("attributes")
        .and_then(|value| value.get("POSITION"))
        .and_then(|value| value.as_u64())
        .ok_or_else(|| RtfpError::model_not_closed("missing POSITION"))?
        as usize;
    let pos = accessor(json, pos_index).map_err(RtfpError::model_not_closed)?;
    if pos.ty != "VEC3" || pos.component != 5126 {
        return Ok(None);
    }
    let index = primitive
        .get("indices")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| RtfpError::model_not_closed("primitive indices are missing"))?
        as usize;
    let idx = accessor(json, index).map_err(RtfpError::model_not_closed)?;
    if idx.ty != "SCALAR" {
        return Err(RtfpError::model_not_closed("index accessor is not SCALAR"));
    }
    let width = match idx.component {
        5121 => 1u32,
        5123 => 2,
        5125 => 4,
        _ => return Ok(None),
    };
    if idx.count % 3 != 0 {
        return Err(RtfpError::model_not_closed(
            "indexed triangle primitive length is not a multiple of 3",
        ));
    }
    Ok(Some(GpuPrim {
        pos_count: u32::try_from(pos.count)
            .map_err(|_| RtfpError::model_not_closed("too many mesh vertices"))?,
        index_count: u32::try_from(idx.count)
            .map_err(|_| RtfpError::model_not_closed("too many mesh indices"))?,
        pos_offset: u32::try_from(pos.offset)
            .map_err(|_| RtfpError::model_not_closed("mesh position offset does not fit"))?,
        index_offset: u32::try_from(idx.offset)
            .map_err(|_| RtfpError::model_not_closed("mesh index offset does not fit"))?,
        pos_stride: u32::try_from(pos.stride)
            .map_err(|_| RtfpError::model_not_closed("mesh position stride does not fit"))?,
        index_stride: u32::try_from(idx.stride.max(width as usize))
            .map_err(|_| RtfpError::model_not_closed("mesh index stride does not fit"))?,
        index_width: width,
    }))
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn read_primitive_gpu(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
    world: [f64; 16],
) -> Result<Option<(Vec<DVec3>, Vec<[u32; 3]>)>, RtfpError> {
    let Some(prim) = gpu_prim_args(json, primitive)? else {
        return Ok(None);
    };
    crate::browser::load_mesh_primitive(
        bin,
        prim.pos_count,
        prim.pos_offset,
        prim.pos_stride,
        prim.index_count,
        prim.index_offset,
        prim.index_stride,
        prim.index_width,
        world,
    )
    .map(Some)
    .map_err(RtfpError::model_not_closed)
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
async fn read_primitive_page(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
    world: [f64; 16],
) -> Result<Option<(Vec<DVec3>, Vec<[u32; 3]>)>, RtfpError> {
    let Some(prim) = gpu_prim_args(json, primitive)? else {
        return Ok(None);
    };
    crate::browser::load_mesh_primitive_local(
        bin,
        prim.pos_count,
        prim.pos_offset,
        prim.pos_stride,
        prim.index_count,
        prim.index_offset,
        prim.index_stride,
        prim.index_width,
        world,
    )
    .await
    .map(Some)
    .map_err(RtfpError::model_not_closed)
}

struct GltfScene {
    json: serde_json::Value,
    bin: Vec<u8>,
    worlds: Vec<[f64; 16]>,
}

fn open_gltf(path: &Path) -> Result<GltfScene, RtfpError> {
    let bytes = read_bytes(path)?;
    let (json, bin) = if path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .eq_ignore_ascii_case("glb")
    {
        decode_glb(&bytes)?
    } else {
        let json: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|err| RtfpError::model_not_closed(format!("glTF JSON is not valid: {err}")))?;
        let bin = external_buffer(path, &json)?;
        (json, bin)
    };
    let nodes = json
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no nodes array"))?;
    let worlds = world_matrices(nodes, &json)?;
    Ok(GltfScene { json, bin, worlds })
}

fn append_cpu_primitive(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
    world: [f64; 16],
    mesh_index: u64,
    primitive_index: usize,
    vertices: &mut Vec<DVec3>,
    faces: &mut Vec<[u32; 3]>,
) -> Result<TriKind, RtfpError> {
    let mode = primitive.get("mode").and_then(|v| v.as_u64()).unwrap_or(4) as u32;
    let positions = read_vec3(json, bin, primitive, "POSITION").map_err(|err| {
        RtfpError::model_not_closed(format!(
            "mesh {mesh_index} primitive {primitive_index}: {err}"
        ))
    })?;
    let indices = match primitive.get("indices") {
        Some(_) => read_indices(json, bin, primitive)?,
        None => (0..positions.len() as u32).collect(),
    };
    let base = vertices.len() as u32;
    for position in positions {
        vertices.push(transform_point(world, position));
    }
    let (triangles, kind) = triangulate(mode, &indices)?;
    for triangle in triangles {
        faces.push([triangle[0] + base, triangle[1] + base, triangle[2] + base]);
    }
    Ok(kind)
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
fn append_loaded(
    vertices: &mut Vec<DVec3>,
    faces: &mut Vec<[u32; 3]>,
    loaded: Vec<DVec3>,
    loaded_faces: Vec<[u32; 3]>,
) {
    let base = vertices.len() as u32;
    vertices.extend(loaded);
    for triangle in loaded_faces {
        faces.push([triangle[0] + base, triangle[1] + base, triangle[2] + base]);
    }
}

fn note_kind(kind: TriKind, strips: &mut usize, fans: &mut usize) {
    match kind {
        TriKind::Triangles => {}
        TriKind::Strip => *strips += 1,
        TriKind::Fan => *fans += 1,
    }
}

fn finish_parsed(
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
    strips: usize,
    fans: usize,
) -> Result<ParsedMesh, RtfpError> {
    if vertices.is_empty() {
        return Err(RtfpError::model_not_closed(
            "glTF scene contains no mesh vertices",
        ));
    }
    Ok(ParsedMesh {
        vertices,
        faces,
        strips,
        fans,
        source: "gltf".into(),
    })
}

fn parse_gltf(path: &Path) -> Result<ParsedMesh, RtfpError> {
    let scene = open_gltf(path)?;
    let nodes = scene
        .json
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no nodes array"))?;
    let meshes = scene
        .json
        .get("meshes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no meshes array"))?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut strips = 0usize;
    let mut fans = 0usize;
    for (node_index, node) in nodes.iter().enumerate() {
        let Some(mesh_index) = node.get("mesh").and_then(|v| v.as_u64()) else {
            continue;
        };
        let mesh = meshes.get(mesh_index as usize).ok_or_else(|| {
            RtfpError::model_not_closed(format!("node {node_index} references a missing mesh"))
        })?;
        let primitives = mesh
            .get("primitives")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RtfpError::model_not_closed(format!("mesh {mesh_index} has no primitives"))
            })?;
        for (primitive_index, primitive) in primitives.iter().enumerate() {
            #[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
            if crate::browser::worker_mesh()
                && let Some((loaded, loaded_faces)) = read_primitive_gpu(
                    &scene.json,
                    &scene.bin,
                    primitive,
                    scene.worlds[node_index],
                )?
            {
                append_loaded(&mut vertices, &mut faces, loaded, loaded_faces);
                continue;
            }
            let kind = append_cpu_primitive(
                &scene.json,
                &scene.bin,
                primitive,
                scene.worlds[node_index],
                mesh_index,
                primitive_index,
                &mut vertices,
                &mut faces,
            )?;
            note_kind(kind, &mut strips, &mut fans);
        }
    }
    finish_parsed(vertices, faces, strips, fans)
}

#[cfg(all(target_arch = "wasm32", any(feature = "tier3", feature = "bevy-web")))]
async fn parse_gltf_gpu(path: &Path) -> Result<ParsedMesh, RtfpError> {
    let scene = open_gltf(path)?;
    let nodes = scene
        .json
        .get("nodes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no nodes array"))?;
    let meshes = scene
        .json
        .get("meshes")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no meshes array"))?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut strips = 0usize;
    let mut fans = 0usize;
    for (node_index, node) in nodes.iter().enumerate() {
        let Some(mesh_index) = node.get("mesh").and_then(|v| v.as_u64()) else {
            continue;
        };
        let mesh = meshes.get(mesh_index as usize).ok_or_else(|| {
            RtfpError::model_not_closed(format!("node {node_index} references a missing mesh"))
        })?;
        let primitives = mesh
            .get("primitives")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                RtfpError::model_not_closed(format!("mesh {mesh_index} has no primitives"))
            })?;
        for (primitive_index, primitive) in primitives.iter().enumerate() {
            if let Some((loaded, loaded_faces)) =
                read_primitive_page(&scene.json, &scene.bin, primitive, scene.worlds[node_index])
                    .await?
            {
                append_loaded(&mut vertices, &mut faces, loaded, loaded_faces);
                continue;
            }
            let kind = append_cpu_primitive(
                &scene.json,
                &scene.bin,
                primitive,
                scene.worlds[node_index],
                mesh_index,
                primitive_index,
                &mut vertices,
                &mut faces,
            )?;
            note_kind(kind, &mut strips, &mut fans);
        }
    }
    finish_parsed(vertices, faces, strips, fans)
}

#[derive(Clone, Copy)]
enum TriKind {
    Triangles,
    Strip,
    Fan,
}

fn triangulate(mode: u32, indices: &[u32]) -> Result<(Vec<[u32; 3]>, TriKind), RtfpError> {
    match mode {
        4 => {
            if indices.len() % 3 != 0 {
                return Err(RtfpError::model_not_closed(
                    "indexed triangle primitive length is not a multiple of 3",
                ));
            }
            let faces = indices
                .chunks(3)
                .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                .collect();
            Ok((faces, TriKind::Triangles))
        }
        5 => {
            if indices.len() < 3 {
                return Err(RtfpError::model_not_closed(
                    "triangle strip has fewer than 3 indices",
                ));
            }
            let mut faces = Vec::new();
            for i in 0..indices.len() - 2 {
                let tri = if i % 2 == 0 {
                    [indices[i], indices[i + 1], indices[i + 2]]
                } else {
                    [indices[i + 1], indices[i], indices[i + 2]]
                };
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0] {
                    faces.push(tri);
                }
            }
            Ok((faces, TriKind::Strip))
        }
        6 => {
            if indices.len() < 3 {
                return Err(RtfpError::model_not_closed(
                    "triangle fan has fewer than 3 indices",
                ));
            }
            let mut faces = Vec::new();
            for i in 1..indices.len() - 1 {
                faces.push([indices[0], indices[i], indices[i + 1]]);
            }
            Ok((faces, TriKind::Fan))
        }
        other => Err(RtfpError::model_not_closed(format!(
            "primitive mode {other} is not triangles (4), a triangle strip (5), or a triangle fan (6); it was not triangulated"
        ))),
    }
}

fn decode_glb(bytes: &[u8]) -> Result<(serde_json::Value, Vec<u8>), RtfpError> {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err(RtfpError::model_not_closed("GLB magic is not glTF"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 2 {
        return Err(RtfpError::model_not_closed(format!(
            "GLB version {version} is not 2"
        )));
    }
    let mut offset = 12usize;
    let mut json = None;
    let mut bin = Vec::new();
    while offset + 8 <= bytes.len() {
        let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = &bytes[offset + 4..offset + 8];
        offset += 8;
        if offset + length > bytes.len() {
            return Err(RtfpError::model_not_closed(
                "GLB chunk exceeds the file length",
            ));
        }
        let chunk = &bytes[offset..offset + length];
        offset += length;
        if kind == b"JSON" {
            json = Some(serde_json::from_slice(chunk).map_err(|err| {
                RtfpError::model_not_closed(format!("GLB JSON chunk is not valid: {err}"))
            })?);
        } else if kind == b"BIN\0" {
            bin = chunk.to_vec();
        }
    }
    Ok((
        json.ok_or_else(|| RtfpError::model_not_closed("GLB has no JSON chunk"))?,
        bin,
    ))
}

fn external_buffer(path: &Path, json: &serde_json::Value) -> Result<Vec<u8>, RtfpError> {
    let buffers = json
        .get("buffers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| RtfpError::model_not_closed("glTF has no buffers"))?;
    if buffers.len() != 1 {
        return Err(RtfpError::model_not_closed(
            "this loader accepts one glTF buffer",
        ));
    }
    let Some(uri) = buffers[0].get("uri").and_then(|v| v.as_str()) else {
        return Err(RtfpError::model_not_closed("glTF buffer has no uri"));
    };
    if let Some(data) = uri.strip_prefix("data:") {
        let (_, payload) = data
            .split_once(",")
            .ok_or_else(|| RtfpError::model_not_closed("glTF data URI has no payload"))?;
        return base64_decode(payload);
    }
    let bin_path = path.parent().unwrap_or(Path::new(".")).join(uri);
    read_bytes(&bin_path)
}

fn base64_decode(text: &str) -> Result<Vec<u8>, RtfpError> {
    fn val(byte: u8) -> Result<u8, RtfpError> {
        match byte {
            b'A'..=b'Z' => Ok(byte - b'A'),
            b'a'..=b'z' => Ok(byte - b'a' + 26),
            b'0'..=b'9' => Ok(byte - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(RtfpError::model_not_closed("glTF data URI is not base64")),
        }
    }
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let a = val(bytes[i])?;
        let b = val(bytes[i + 1])?;
        let c = if bytes[i + 2] == b'=' {
            0
        } else {
            val(bytes[i + 2])?
        };
        let d = if bytes[i + 3] == b'=' {
            0
        } else {
            val(bytes[i + 3])?
        };
        out.push((a << 2) | (b >> 4));
        if bytes[i + 2] != b'=' {
            out.push((b << 4) | (c >> 2));
        }
        if bytes[i + 3] != b'=' {
            out.push((c << 6) | d);
        }
        i += 4;
    }
    Ok(out)
}

fn world_matrices(
    nodes: &[serde_json::Value],
    json: &serde_json::Value,
) -> Result<Vec<[f64; 16]>, RtfpError> {
    let mut parent = vec![None; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            for child in children {
                let child = child.as_u64().ok_or_else(|| {
                    RtfpError::model_not_closed("glTF child index is not an integer")
                })? as usize;
                if child >= nodes.len() || parent[child].is_some() {
                    return Err(RtfpError::model_not_closed(
                        "glTF node hierarchy has a missing or repeated child",
                    ));
                }
                parent[child] = Some(index);
            }
        }
    }
    let roots: Vec<usize> = if let Some(scene) = json.get("scene").and_then(|v| v.as_u64()) {
        let scenes = json
            .get("scenes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RtfpError::model_not_closed("glTF scene index has no scenes array"))?;
        let scene = scenes
            .get(scene as usize)
            .ok_or_else(|| RtfpError::model_not_closed("glTF scene index is out of range"))?;
        scene
            .get("nodes")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RtfpError::model_not_closed("glTF scene has no nodes"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(u64::MAX) as usize)
            .collect()
    } else {
        (0..nodes.len()).filter(|i| parent[*i].is_none()).collect()
    };
    let mut local = Vec::with_capacity(nodes.len());
    for node in nodes {
        local.push(local_matrix(node)?);
    }
    let mut world = vec![[0.0; 16]; nodes.len()];
    let mut seen = vec![false; nodes.len()];
    fn visit(
        index: usize,
        parent_world: [f64; 16],
        local: &[[f64; 16]],
        parent: &[Option<usize>],
        world: &mut [[f64; 16]],
        seen: &mut [bool],
    ) -> Result<(), RtfpError> {
        if seen[index] {
            return Err(RtfpError::model_not_closed("glTF node graph has a cycle"));
        }
        seen[index] = true;
        world[index] = mul_matrix(parent_world, local[index]);
        for (child, parent_index) in parent.iter().enumerate() {
            if *parent_index == Some(index) {
                visit(child, world[index], local, parent, world, seen)?;
            }
        }
        Ok(())
    }
    let identity = identity_matrix();
    for root in roots {
        if root >= nodes.len() {
            return Err(RtfpError::model_not_closed(
                "glTF scene node is out of range",
            ));
        }
        visit(root, identity, &local, &parent, &mut world, &mut seen)?;
    }
    for (index, done) in seen.iter().enumerate() {
        if !done && nodes[index].get("mesh").is_some() {
            return Err(RtfpError::model_not_closed(format!(
                "mesh node {index} is not reachable from the scene root, so its transform was not applied"
            )));
        }
    }
    Ok(world)
}

fn local_matrix(node: &serde_json::Value) -> Result<[f64; 16], RtfpError> {
    if let Some(matrix) = node.get("matrix").and_then(|v| v.as_array()) {
        if matrix.len() != 16 {
            return Err(RtfpError::model_not_closed(
                "glTF node matrix must have 16 entries",
            ));
        }
        let mut out = [0.0; 16];
        for (i, value) in matrix.iter().enumerate() {
            out[i] = value.as_f64().ok_or_else(|| {
                RtfpError::model_not_closed("glTF node matrix entry is not a number")
            })?;
        }
        if node.get("translation").is_some()
            || node.get("rotation").is_some()
            || node.get("scale").is_some()
        {
            return Err(RtfpError::model_not_closed(
                "glTF node sets both matrix and TRS",
            ));
        }
        return Ok(out);
    }
    let translation = optional_vec3(node, "translation")?.unwrap_or([0.0, 0.0, 0.0]);
    let rotation = optional_vec4(node, "rotation")?.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let scale = optional_vec3(node, "scale")?.unwrap_or([1.0, 1.0, 1.0]);
    if scale.iter().any(|v| !v.is_finite() || *v == 0.0) {
        return Err(RtfpError::model_not_closed(
            "glTF node scale is zero or not finite",
        ));
    }
    let mut matrix = quat_matrix(rotation);
    for col in 0..3 {
        for row in 0..3 {
            matrix[col * 4 + row] *= scale[col];
        }
    }
    matrix[12] = translation[0];
    matrix[13] = translation[1];
    matrix[14] = translation[2];
    Ok(matrix)
}

fn quat_matrix(q: [f64; 4]) -> [f64; 16] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    let mut m = [0.0; 16];
    m[0] = 1.0 - 2.0 * (yy + zz);
    m[4] = 2.0 * (xy - wz);
    m[8] = 2.0 * (xz + wy);
    m[1] = 2.0 * (xy + wz);
    m[5] = 1.0 - 2.0 * (xx + zz);
    m[9] = 2.0 * (yz - wx);
    m[2] = 2.0 * (xz - wy);
    m[6] = 2.0 * (yz + wx);
    m[10] = 1.0 - 2.0 * (xx + yy);
    m[15] = 1.0;
    m
}

fn identity_matrix() -> [f64; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

fn mul_matrix(parent: [f64; 16], local: [f64; 16]) -> [f64; 16] {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += parent[k * 4 + row] * local[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

fn transform_point(matrix: [f64; 16], point: DVec3) -> DVec3 {
    DVec3::new(
        matrix[0] * point.x + matrix[4] * point.y + matrix[8] * point.z + matrix[12],
        matrix[1] * point.x + matrix[5] * point.y + matrix[9] * point.z + matrix[13],
        matrix[2] * point.x + matrix[6] * point.y + matrix[10] * point.z + matrix[14],
    )
}

fn read_vec3(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
    attribute: &str,
) -> Result<Vec<DVec3>, String> {
    let index = primitive
        .get("attributes")
        .and_then(|v| v.get(attribute))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("missing {attribute}"))? as usize;
    let accessor = accessor(json, index)?;
    if accessor.ty != "VEC3" || accessor.component != 5126 {
        return Err(format!(
            "{attribute} must be a float VEC3, found {} component {}",
            accessor.ty, accessor.component
        ));
    }
    let mut out = Vec::with_capacity(accessor.count);
    for i in 0..accessor.count {
        let offset = accessor.offset + i * accessor.stride;
        if offset + 12 > bin.len() {
            return Err("accessor exceeds the buffer".into());
        }
        let x = f32::from_le_bytes(bin[offset..offset + 4].try_into().unwrap()) as f64;
        let y = f32::from_le_bytes(bin[offset + 4..offset + 8].try_into().unwrap()) as f64;
        let z = f32::from_le_bytes(bin[offset + 8..offset + 12].try_into().unwrap()) as f64;
        if ![x, y, z].iter().all(|v| v.is_finite()) {
            return Err("mesh vertex is not finite".into());
        }
        out.push(DVec3::new(x, y, z));
    }
    Ok(out)
}

fn read_indices(
    json: &serde_json::Value,
    bin: &[u8],
    primitive: &serde_json::Value,
) -> Result<Vec<u32>, RtfpError> {
    let index = primitive
        .get("indices")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| RtfpError::model_not_closed("primitive indices are missing"))?
        as usize;
    let accessor = accessor(json, index).map_err(RtfpError::model_not_closed)?;
    if accessor.ty != "SCALAR" {
        return Err(RtfpError::model_not_closed("index accessor is not SCALAR"));
    }
    let width = match accessor.component {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        other => {
            return Err(RtfpError::model_not_closed(format!(
                "index component type {other} is not unsigned byte, short, or int"
            )));
        }
    };
    let mut out = Vec::with_capacity(accessor.count);
    for i in 0..accessor.count {
        let offset = accessor.offset + i * accessor.stride.max(width);
        if offset + width > bin.len() {
            return Err(RtfpError::model_not_closed(
                "index accessor exceeds the buffer",
            ));
        }
        let value = match width {
            1 => bin[offset] as u32,
            2 => u16::from_le_bytes(bin[offset..offset + 2].try_into().unwrap()) as u32,
            _ => u32::from_le_bytes(bin[offset..offset + 4].try_into().unwrap()),
        };
        out.push(value);
    }
    Ok(out)
}

struct Accessor {
    offset: usize,
    stride: usize,
    count: usize,
    component: u32,
    ty: String,
}

fn accessor(json: &serde_json::Value, index: usize) -> Result<Accessor, String> {
    let accessors = json
        .get("accessors")
        .and_then(|v| v.as_array())
        .ok_or("missing accessors")?;
    let views = json
        .get("bufferViews")
        .and_then(|v| v.as_array())
        .ok_or("missing bufferViews")?;
    let accessor = accessors
        .get(index)
        .ok_or("accessor index is out of range")?;
    let view_index = accessor
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or("accessor has no bufferView")? as usize;
    let view = views
        .get(view_index)
        .ok_or("bufferView index is out of range")?;
    if view.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) != 0 {
        return Err("only buffer 0 is supported".into());
    }
    let view_offset = view.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let accessor_offset = accessor
        .get("byteOffset")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let count = accessor
        .get("count")
        .and_then(|v| v.as_u64())
        .ok_or("accessor has no count")? as usize;
    let component = accessor
        .get("componentType")
        .and_then(|v| v.as_u64())
        .ok_or("accessor has no componentType")? as u32;
    let ty = accessor
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let components = match ty.as_str() {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        _ => return Err(format!("accessor type {ty} is not supported")),
    };
    let component_size = match component {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        other => return Err(format!("component type {other} is not supported")),
    };
    let tight = components * component_size;
    let stride = view
        .get("byteStride")
        .and_then(|v| v.as_u64())
        .unwrap_or(tight as u64) as usize;
    if stride < tight {
        return Err("bufferView stride is smaller than the element".into());
    }
    Ok(Accessor {
        offset: view_offset + accessor_offset,
        stride,
        count,
        component,
        ty,
    })
}

fn optional_vec3(node: &serde_json::Value, key: &str) -> Result<Option<[f64; 3]>, RtfpError> {
    let Some(value) = node.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| RtfpError::model_not_closed(format!("glTF {key} is not an array")))?;
    if array.len() != 3 {
        return Err(RtfpError::model_not_closed(format!(
            "glTF {key} must have 3 entries"
        )));
    }
    let mut out = [0.0; 3];
    for (i, component) in array.iter().enumerate() {
        out[i] = component.as_f64().ok_or_else(|| {
            RtfpError::model_not_closed(format!("glTF {key} entry is not a number"))
        })?;
    }
    Ok(Some(out))
}

fn optional_vec4(node: &serde_json::Value, key: &str) -> Result<Option<[f64; 4]>, RtfpError> {
    let Some(value) = node.get(key) else {
        return Ok(None);
    };
    let array = value
        .as_array()
        .ok_or_else(|| RtfpError::model_not_closed(format!("glTF {key} is not an array")))?;
    if array.len() != 4 {
        return Err(RtfpError::model_not_closed(format!(
            "glTF {key} must have 4 entries"
        )));
    }
    let mut out = [0.0; 4];
    for (i, component) in array.iter().enumerate() {
        out[i] = component.as_f64().ok_or_else(|| {
            RtfpError::model_not_closed(format!("glTF {key} entry is not a number"))
        })?;
    }
    Ok(Some(out))
}

fn weld_and_drop(
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
) -> Result<(Vec<DVec3>, Vec<[u32; 3]>, usize, usize), RtfpError> {
    let mut map: HashMap<(u64, u64, u64), u32> = HashMap::new();
    let mut welded = Vec::new();
    let mut remap = vec![0u32; vertices.len()];
    for (index, vertex) in vertices.iter().enumerate() {
        if !vertex.is_finite() {
            return Err(RtfpError::model_not_closed("mesh vertex is not finite"));
        }
        let key = (vertex.x.to_bits(), vertex.y.to_bits(), vertex.z.to_bits());
        let slot = if let Some(existing) = map.get(&key) {
            *existing
        } else {
            let slot = welded.len() as u32;
            map.insert(key, slot);
            welded.push(*vertex);
            slot
        };
        remap[index] = slot;
    }
    let removed = vertices.len() - welded.len();
    let mut out = Vec::new();
    let mut dropped = 0usize;
    for face in faces {
        if face.iter().any(|index| *index as usize >= remap.len()) {
            return Err(RtfpError::model_not_closed(
                "mesh face index is outside the vertex array",
            ));
        }
        let tri = [
            remap[face[0] as usize],
            remap[face[1] as usize],
            remap[face[2] as usize],
        ];
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
            dropped += 1;
            continue;
        }
        out.push(tri);
    }
    Ok((welded, out, removed, dropped))
}

fn convex_hull(points: &[DVec3]) -> Result<TriMesh, RtfpError> {
    let mut unique = Vec::new();
    for point in points {
        if !point.is_finite() {
            return Err(RtfpError::model_not_closed("hull vertex is not finite"));
        }
        if unique
            .iter()
            .all(|kept: &DVec3| (*kept - *point).length() > 0.0)
        {
            unique.push(*point);
        }
    }
    if unique.len() < 4 {
        return Err(RtfpError::model_not_closed(
            "convex hull needs at least 4 non-coincident points",
        ));
    }
    let (i0, i1, i2, i3) = initial_tetra(&unique)?;
    let (i0, i1, i2, i3) = (i0 as u32, i1 as u32, i2 as u32, i3 as u32);
    let mut faces: Vec<[u32; 3]> = vec![[i0, i2, i1], [i0, i1, i3], [i0, i3, i2], [i1, i2, i3]];
    let volume = {
        let a = unique[i0 as usize];
        let b = unique[i1 as usize];
        let c = unique[i2 as usize];
        let d = unique[i3 as usize];
        (b - a).cross(c - a).dot(d - a)
    };
    if volume < 0.0 {
        for face in &mut faces {
            face.swap(1, 2);
        }
    }
    for (index, point) in unique.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&(index as u32)) {
            continue;
        }
        let mut visible = Vec::new();
        for (face_index, face) in faces.iter().enumerate() {
            let a = unique[face[0] as usize];
            let b = unique[face[1] as usize];
            let c = unique[face[2] as usize];
            let normal = (b - a).cross(c - a);
            if normal.dot(*point - a) > 1e-12 * normal.length().max(1.0) {
                visible.push(face_index);
            }
        }
        if visible.is_empty() {
            continue;
        }
        let mut edge_count: HashMap<(u32, u32), i32> = HashMap::new();
        for face_index in &visible {
            let face = faces[*face_index];
            for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                let key = if edge.0 < edge.1 {
                    (edge.0, edge.1)
                } else {
                    (edge.1, edge.0)
                };
                let sign = if edge.0 < edge.1 { 1 } else { -1 };
                *edge_count.entry(key).or_default() += sign;
            }
        }
        let horizon: Vec<(u32, u32)> = edge_count
            .into_iter()
            .filter_map(|((lo, hi), sign)| {
                if sign == 0 {
                    None
                } else if sign > 0 {
                    Some((lo, hi))
                } else {
                    Some((hi, lo))
                }
            })
            .collect();
        let mut next = Vec::new();
        for (face_index, face) in faces.iter().enumerate() {
            if !visible.contains(&face_index) {
                next.push(*face);
            }
        }
        for (start, end) in horizon {
            next.push([end, start, index as u32]);
        }
        faces = next;
    }
    TriMesh::try_new(unique, faces).map_err(Into::into)
}

fn initial_tetra(points: &[DVec3]) -> Result<(usize, usize, usize, usize), RtfpError> {
    let i0 = 0usize;
    let mut i1 = points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            (**a - points[0])
                .length()
                .partial_cmp(&(**b - points[0]).length())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    if (points[i1] - points[i0]).length() == 0.0 {
        return Err(RtfpError::model_not_closed(
            "convex hull points are coincident",
        ));
    }
    let mut i2 = None;
    let mut best = 0.0;
    for (index, point) in points.iter().enumerate() {
        let area = (points[i1] - points[i0])
            .cross(*point - points[i0])
            .length();
        if area > best {
            best = area;
            i2 = Some(index);
        }
    }
    let i2 = i2.ok_or_else(|| RtfpError::model_not_closed("convex hull points are colinear"))?;
    if best == 0.0 {
        return Err(RtfpError::model_not_closed(
            "convex hull points are colinear",
        ));
    }
    let mut i3 = None;
    let mut best_vol = 0.0;
    for (index, point) in points.iter().enumerate() {
        let volume = (points[i1] - points[i0])
            .cross(points[i2] - points[i0])
            .dot(*point - points[i0])
            .abs();
        if volume > best_vol {
            best_vol = volume;
            i3 = Some(index);
        }
    }
    let i3 = i3.ok_or_else(|| RtfpError::model_not_closed("convex hull points are coplanar"))?;
    if best_vol == 0.0 {
        return Err(RtfpError::model_not_closed(
            "convex hull points are coplanar",
        ));
    }
    if i1 == 0 {
        i1 = 1;
    }
    let _ = i0;
    Ok((0, i1, i2, i3))
}

fn radial_raycast(points: &[DVec3]) -> Result<TriMesh, RtfpError> {
    const N_LAT: usize = 8;
    const N_LON: usize = 16;
    if points.len() < 4 {
        return Err(RtfpError::model_not_closed(
            "radial raycast needs at least 4 points",
        ));
    }
    let mut centroid = DVec3::ZERO;
    for point in points {
        if !point.is_finite() {
            return Err(RtfpError::model_not_closed(
                "radial raycast point is not finite",
            ));
        }
        centroid = centroid + *point;
    }
    centroid = centroid * (1.0 / points.len() as f64);
    let bins = (N_LAT - 1) * N_LON + 2;
    let mut best: Vec<Option<(f64, DVec3)>> = vec![None; bins];
    for point in points {
        let offset = *point - centroid;
        let radius = offset.length();
        if radius == 0.0 {
            return Err(RtfpError::model_not_closed(
                "radial raycast point lies on the reconstruction center",
            ));
        }
        let dir = offset * (1.0 / radius);
        let theta = dir.z.clamp(-1.0, 1.0).acos();
        let band = theta / std::f64::consts::PI * (N_LAT as f64);
        let bin = if band < 0.5 {
            0
        } else if band > N_LAT as f64 - 0.5 {
            bins - 1
        } else {
            let lat = (band - 0.5).floor() as usize;
            let mut phi = dir.y.atan2(dir.x);
            if phi < 0.0 {
                phi += 2.0 * std::f64::consts::PI;
            }
            let lon =
                ((phi / (2.0 * std::f64::consts::PI)) * N_LON as f64).floor() as usize % N_LON;
            1 + lat.min(N_LAT - 2) * N_LON + lon
        };
        let replace = best[bin].map(|(kept, _)| radius > kept).unwrap_or(true);
        if replace {
            best[bin] = Some((radius, *point));
        }
    }
    let empty = best.iter().filter(|slot| slot.is_none()).count();
    if empty > 0 {
        return Err(RtfpError::model_not_closed(format!(
            "radial_raycast n_lat={N_LAT} n_lon={N_LON} left {empty} direction bins empty; the points were not replaced by a sphere"
        )));
    }
    let mut vertices = Vec::with_capacity(bins);
    for slot in best {
        vertices.push(slot.unwrap().1);
    }
    let rings = N_LAT - 1;
    let mut faces = Vec::new();
    let south = vertices.len() as u32 - 1;
    for lon in 0..N_LON {
        let a = 1 + lon as u32;
        let b = 1 + ((lon + 1) % N_LON) as u32;
        faces.push([0, a, b]);
    }
    for lat in 0..(rings - 1) {
        for lon in 0..N_LON {
            let row = 1 + lat * N_LON;
            let a = (row + lon) as u32;
            let b = (row + (lon + 1) % N_LON) as u32;
            let c = (row + N_LON + lon) as u32;
            let d = (row + N_LON + (lon + 1) % N_LON) as u32;
            faces.push([a, d, b]);
            faces.push([a, c, d]);
        }
    }
    let row = 1 + (rings - 1) * N_LON;
    for lon in 0..N_LON {
        let a = (row + lon) as u32;
        let b = (row + (lon + 1) % N_LON) as u32;
        faces.push([a, south, b]);
    }
    TriMesh::try_new(vertices, faces).map_err(Into::into)
}
