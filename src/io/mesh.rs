//! Triangle-mesh import (OBJ, XYZ point lists, glTF/GLB). Node transforms are
//! applied. A vertex list without faces stays a vertex list until the caller
//! names a reconstruction.

use std::collections::HashMap;
use std::path::Path;

use super::length_to_meters;
use crate::common::error::SimError;
use crate::common::geometry::TriMesh;
use crate::common::math::{DMat4, DVec3};

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

fn err(msg: impl Into<String>) -> SimError {
    SimError::model_not_closed(msg)
}

pub fn mesh_hash(mesh: &TriMesh) -> String {
    let vertices: Vec<[f64; 3]> = mesh.vertices.iter().map(|v| v.to_array()).collect();
    crate::common::obj::hash(&vertices, &mesh.faces)
}

/// File bytes: the disk natively, the in-memory asset table in the browser.
pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>, SimError> {
    super::memfs::read(path).map_err(|e| err(format!("cannot read {e}")))
}

fn read_text(path: &Path) -> Result<String, SimError> {
    String::from_utf8(read_bytes(path)?).map_err(|e| err(format!("{}: {e}", path.display())))
}

pub fn load_mesh_file(request: ImportRequest<'_>) -> Result<MeshDocument, SimError> {
    let path = request.path;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (parsed, declared) = match ext.as_str() {
        "obj" | "xyz" => parse_obj(&read_text(path)?)?,
        "glb" | "gltf" => (parse_gltf(path)?, None),
        other => {
            return Err(err(format!(
                "mesh extension '{other}' is not obj, xyz, glb, or gltf"
            )));
        }
    };
    let unit = match (request.length_unit, declared.as_deref()) {
        (Some(flag), Some(file)) if flag != file => {
            return Err(err(format!(
                "length unit {flag} does not match the unit {file} declared in {}",
                path.display()
            )));
        }
        (Some(u), _) | (None, Some(u)) => u.to_string(),
        (None, None) => {
            return Err(err(
                "mesh length unit is missing; pass --length-unit or import a mesh that declares one",
            ));
        }
    };
    let scale = length_to_meters(&unit)?;
    let (vertices, faces, welded, dropped) = if parsed.faces.is_empty() {
        let strategy = request.reconstruct.ok_or_else(|| {
            err("the input has vertices and no faces; pass --reconstruct convex_hull or --reconstruct radial_raycast")
        })?;
        let mesh = match strategy {
            Reconstruct::ConvexHull => convex_hull(&parsed.vertices)?,
            Reconstruct::RadialRaycast => radial_raycast(&parsed.vertices)?,
        };
        (mesh.vertices, mesh.faces, 0, 0)
    } else {
        weld_and_drop(parsed.vertices, parsed.faces)?
    };
    let mesh = TriMesh::try_new(vertices.into_iter().map(|v| v * scale).collect(), faces)?;
    let stored = if unit == "1" || unit == "dimensionless" {
        unit.clone()
    } else {
        "m".into()
    };
    Ok(MeshDocument {
        mesh_hash: mesh_hash(&mesh),
        mesh,
        length_unit: stored,
        source_length_unit: unit,
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

pub fn write_obj(path: &Path, document: &MeshDocument) -> Result<(), SimError> {
    let d = document;
    let header = format!(
        "# astroid_evolution_mesh 1\n# length_unit {}\n# source_length_unit {}\n# mesh_hash {}\n# source {}\n\
         # welded_positions {}\n# dropped_degenerate {}\n# triangulated_strips {}\n# triangulated_fans {}\n",
        d.length_unit,
        d.source_length_unit,
        d.mesh_hash,
        d.source,
        d.welded_positions,
        d.dropped_degenerate,
        d.triangulated_strips,
        d.triangulated_fans
    );
    crate::common::obj::write(
        path,
        &header,
        d.mesh.vertices.iter().map(|v| v.to_array()),
        &d.mesh.faces,
    )
}

/// OBJ vertices and faces (polygons fanned), or a bare `x y z` point list.
fn parse_obj(text: &str) -> Result<(ParsedMesh, Option<String>), SimError> {
    let obj = crate::common::obj::parse(text, false)?;
    let parsed = ParsedMesh {
        vertices: obj.vertices.into_iter().map(DVec3::from_array).collect(),
        faces: obj.faces,
        strips: 0,
        fans: obj.fans,
        source: "obj".into(),
    };
    Ok((parsed, obj.length_unit))
}

/// Every triangle primitive of the default scene (or every scene), in world
/// coordinates. Buffers come from the GLB blob, a data URI, or a sibling file.
fn parse_gltf(path: &Path) -> Result<ParsedMesh, SimError> {
    use base64::Engine as _;
    use gltf::mesh::Mode;
    let bytes = read_bytes(path)?;
    let gltf =
        gltf::Gltf::from_slice(&bytes).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut buffers = Vec::new();
    for buffer in gltf.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf
                .blob
                .clone()
                .ok_or_else(|| err("GLB has no binary chunk"))?,
            gltf::buffer::Source::Uri(uri) => match uri.strip_prefix("data:") {
                Some(rest) => {
                    let b64 = rest
                        .split_once(";base64,")
                        .map(|x| x.1)
                        .ok_or_else(|| err("data URI is not base64"))?;
                    base64::engine::general_purpose::STANDARD
                        .decode(b64)
                        .map_err(|e| err(format!("data URI: {e}")))?
                }
                None => read_bytes(&dir.join(uri))?,
            },
        };
        if data.len() < buffer.length() {
            return Err(err(format!(
                "buffer {} is shorter than declared",
                buffer.index()
            )));
        }
        buffers.push(data);
    }
    let mut out = ParsedMesh {
        vertices: Vec::new(),
        faces: Vec::new(),
        strips: 0,
        fans: 0,
        source: "gltf".into(),
    };
    let scenes: Vec<_> = gltf.default_scene().into_iter().collect();
    let scenes = if scenes.is_empty() {
        gltf.scenes().collect()
    } else {
        scenes
    };
    let mut stack: Vec<(gltf::Node, DMat4)> = scenes
        .iter()
        .flat_map(|s| s.nodes().map(|n| (n, DMat4::IDENTITY)))
        .collect();
    while let Some((node, parent)) = stack.pop() {
        let local = DMat4::from_cols_array_2d(&node.transform().matrix().map(|c| c.map(f64::from)));
        let world = parent * local;
        stack.extend(node.children().map(|c| (c, world)));
        let Some(mesh) = node.mesh() else { continue };
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let Some(pos) = reader.read_positions() else {
                continue;
            };
            let base = out.vertices.len() as u32;
            out.vertices.extend(pos.map(|p| {
                world.transform_point3(DVec3::new(p[0].into(), p[1].into(), p[2].into()))
            }));
            let n = out.vertices.len() as u32 - base;
            let idx: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..n).collect(),
            };
            if idx.iter().any(|&i| i >= n) {
                return Err(err("glTF index is outside its primitive"));
            }
            let tri = |a: u32, b: u32, c: u32| [base + a, base + b, base + c];
            match prim.mode() {
                Mode::Triangles => out
                    .faces
                    .extend(idx.chunks_exact(3).map(|t| tri(t[0], t[1], t[2]))),
                Mode::TriangleStrip => {
                    out.strips += 1;
                    out.faces.extend((0..idx.len().saturating_sub(2)).map(|k| {
                        if k % 2 == 0 {
                            tri(idx[k], idx[k + 1], idx[k + 2])
                        } else {
                            tri(idx[k + 1], idx[k], idx[k + 2])
                        }
                    }));
                }
                Mode::TriangleFan => {
                    out.fans += 1;
                    out.faces.extend(
                        (1..idx.len().saturating_sub(1)).map(|k| tri(idx[0], idx[k], idx[k + 1])),
                    );
                }
                // A point cloud keeps its vertices for a named reconstruction.
                Mode::Points => {}
                other => {
                    return Err(err(format!(
                        "glTF primitive mode {other:?} has no triangles"
                    )));
                }
            }
        }
    }
    if out.vertices.is_empty() {
        return Err(err(format!("{} has no mesh positions", path.display())));
    }
    Ok(out)
}

/// Merge bit-identical positions and drop triangles that collapse.
fn weld_and_drop(
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
) -> Result<(Vec<DVec3>, Vec<[u32; 3]>, usize, usize), SimError> {
    let mut map: HashMap<[u64; 3], u32> = HashMap::with_capacity(vertices.len());
    let mut welded = Vec::new();
    let mut remap = Vec::with_capacity(vertices.len());
    for v in &vertices {
        if !v.is_finite() {
            return Err(err("mesh vertex is not finite"));
        }
        let slot = *map
            .entry(v.to_array().map(f64::to_bits))
            .or_insert_with(|| {
                welded.push(*v);
                welded.len() as u32 - 1
            });
        remap.push(slot);
    }
    let removed = vertices.len() - welded.len();
    let total = faces.len();
    let mut out = Vec::with_capacity(total);
    for f in faces {
        let [Some(a), Some(b), Some(c)] = f.map(|i| remap.get(i as usize).copied()) else {
            return Err(err("mesh face index is outside the vertex array"));
        };
        if a != b && b != c && c != a {
            out.push([a, b, c]);
        }
    }
    let dropped = total - out.len();
    Ok((welded, out, removed, dropped))
}

/// Quickhull (`chull`), oriented outward.
fn convex_hull(points: &[DVec3]) -> Result<TriMesh, SimError> {
    if points.iter().any(|p| !p.is_finite()) {
        return Err(err("hull vertex is not finite"));
    }
    let pts: Vec<Vec<f64>> = points.iter().map(|p| p.to_array().to_vec()).collect();
    let hull = chull::ConvexHullWrapper::try_new(&pts, None)
        .map_err(|e| err(format!("convex hull: {e:?}")))?;
    let (verts, idx) = hull.vertices_indices();
    let vertices: Vec<DVec3> = verts.iter().map(|v| DVec3::new(v[0], v[1], v[2])).collect();
    let centre = vertices.iter().copied().sum::<DVec3>() / vertices.len() as f64;
    let faces = idx
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as u32);
            let (pa, pb, pc) = (
                vertices[a as usize],
                vertices[b as usize],
                vertices[c as usize],
            );
            if (pb - pa).cross(pc - pa).dot(pa - centre) < 0.0 {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();
    TriMesh::try_new(vertices, faces).map_err(Into::into)
}

/// The farthest point in each of 8 × 16 latitude–longitude bins about the
/// centroid, joined as a UV sphere. An empty bin is an error.
fn radial_raycast(points: &[DVec3]) -> Result<TriMesh, SimError> {
    const N_LAT: usize = 8;
    const N_LON: usize = 16;
    if points.len() < 4 || points.iter().any(|p| !p.is_finite()) {
        return Err(err("radial raycast needs at least 4 finite points"));
    }
    let centroid = points.iter().copied().sum::<DVec3>() / points.len() as f64;
    let bins = (N_LAT - 1) * N_LON + 2;
    let mut best: Vec<Option<(f64, DVec3)>> = vec![None; bins];
    for p in points {
        let offset = *p - centroid;
        let radius = offset.length();
        if radius == 0.0 {
            return Err(err(
                "radial raycast point lies on the reconstruction center",
            ));
        }
        let dir = offset / radius;
        let band = dir.z.clamp(-1.0, 1.0).acos() / std::f64::consts::PI * N_LAT as f64;
        let bin = if band < 0.5 {
            0
        } else if band > N_LAT as f64 - 0.5 {
            bins - 1
        } else {
            let lat = ((band - 0.5).floor() as usize).min(N_LAT - 2);
            let phi = dir.y.atan2(dir.x).rem_euclid(std::f64::consts::TAU);
            let lon = (phi / std::f64::consts::TAU * N_LON as f64).floor() as usize % N_LON;
            1 + lat * N_LON + lon
        };
        if best[bin].is_none_or(|(kept, _)| radius > kept) {
            best[bin] = Some((radius, *p));
        }
    }
    let empty = best.iter().filter(|s| s.is_none()).count();
    if empty > 0 {
        return Err(err(format!(
            "radial_raycast n_lat={N_LAT} n_lon={N_LON} left {empty} direction bins empty; the points were not replaced by a sphere"
        )));
    }
    let vertices: Vec<DVec3> = best.into_iter().map(|s| s.unwrap().1).collect();
    let south = vertices.len() as u32 - 1;
    let ring = |lat: usize, lon: usize| (1 + lat * N_LON + lon % N_LON) as u32;
    let mut faces = Vec::new();
    for lon in 0..N_LON {
        faces.push([0, ring(0, lon), ring(0, lon + 1)]);
        faces.push([ring(N_LAT - 2, lon), south, ring(N_LAT - 2, lon + 1)]);
        for lat in 0..N_LAT - 2 {
            let (a, b, c, d) = (
                ring(lat, lon),
                ring(lat, lon + 1),
                ring(lat + 1, lon),
                ring(lat + 1, lon + 1),
            );
            faces.push([a, d, b]);
            faces.push([a, c, d]);
        }
    }
    TriMesh::try_new(vertices, faces).map_err(Into::into)
}
