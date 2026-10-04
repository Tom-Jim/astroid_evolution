use std::fs;
use std::path::Path;

use crate::common::geometry::TriMesh;
use crate::scf_simulator::error::ScfError;

pub fn write_glb(path: &Path, mesh: &TriMesh) -> Result<(), ScfError> {
    let mut positions = Vec::with_capacity(mesh.vertices.len() * 12);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for (axis, value) in [vertex.x, vertex.y, vertex.z].into_iter().enumerate() {
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
            positions.extend_from_slice(&(value as f32).to_le_bytes());
        }
    }
    let index_offset = positions.len();
    let mut bin = positions;
    for face in &mesh.faces {
        for index in face {
            bin.extend_from_slice(&index.to_le_bytes());
        }
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\"}},\"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\"nodes\":[{{\"mesh\":0}}],\"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}},\"indices\":1,\"mode\":4}}]}}],\"accessors\":[{{\"bufferView\":0,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"min\":[{:.8e},{:.8e},{:.8e}],\"max\":[{:.8e},{:.8e},{:.8e}]}},{{\"bufferView\":1,\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\"}}],\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{},\"target\":34962}},{{\"buffer\":0,\"byteOffset\":{},\"byteLength\":{},\"target\":34963}}],\"buffers\":[{{\"byteLength\":{}}}]}}",
        mesh.vertices.len(),
        min[0],
        min[1],
        min[2],
        max[0],
        max[1],
        max[2],
        mesh.faces.len() * 3,
        mesh.vertices.len() * 12,
        index_offset,
        mesh.faces.len() * 12,
        bin.len()
    );
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(b"glTF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(total as u32).to_le_bytes());
    bytes.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"JSON");
    bytes.extend_from_slice(&json_bytes);
    bytes.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"BIN\0");
    bytes.extend_from_slice(&bin);
    fs::write(path, bytes).map_err(|err| ScfError::model_not_closed(err.to_string()))
}

pub fn export_snapshot(snapshot: &Path, format: &str, out: &Path) -> Result<(), ScfError> {
    let state_path = if snapshot.is_dir() {
        snapshot.join("state.json")
    } else {
        snapshot.to_path_buf()
    };
    let text = fs::read_to_string(&state_path)
        .map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    let vertices = value
        .get("vertices")
        .and_then(|item| item.as_array())
        .ok_or_else(|| ScfError::model_not_closed("snapshot has no vertex buffer"))?;
    let faces = value
        .get("faces")
        .and_then(|item| item.as_array())
        .ok_or_else(|| ScfError::model_not_closed("snapshot has no triangle index buffer"))?;
    let mut mesh_vertices = Vec::new();
    for vertex in vertices {
        let row = vertex
            .as_array()
            .ok_or_else(|| ScfError::model_not_closed("snapshot vertex is not a triple"))?;
        if row.len() != 3 {
            return Err(ScfError::model_not_closed(
                "snapshot vertex is not a triple",
            ));
        }
        mesh_vertices.push(crate::common::math::DVec3::new(
            row[0]
                .as_f64()
                .ok_or_else(|| ScfError::model_not_closed("snapshot coordinate is not numeric"))?,
            row[1]
                .as_f64()
                .ok_or_else(|| ScfError::model_not_closed("snapshot coordinate is not numeric"))?,
            row[2]
                .as_f64()
                .ok_or_else(|| ScfError::model_not_closed("snapshot coordinate is not numeric"))?,
        ));
    }
    let mut mesh_faces = Vec::new();
    for face in faces {
        let row = face
            .as_array()
            .ok_or_else(|| ScfError::model_not_closed("snapshot face is not a triple"))?;
        if row.len() != 3 {
            return Err(ScfError::model_not_closed("snapshot face is not a triple"));
        }
        let index = |item: &serde_json::Value| -> Result<u32, ScfError> {
            item.as_u64()
                .map(|value| value as u32)
                .ok_or_else(|| ScfError::model_not_closed("snapshot face index is not an integer"))
        };
        mesh_faces.push([index(&row[0])?, index(&row[1])?, index(&row[2])?]);
    }
    let mesh = TriMesh::try_new(mesh_vertices, mesh_faces)
        .map_err(|err| ScfError::model_not_closed(err.detail))?;
    match format {
        "obj" => {
            let document = crate::rtfp_solver::io::MeshDocument {
                mesh: mesh.clone(),
                length_unit: "m".into(),
                source_length_unit: "m".into(),
                mesh_hash: crate::rtfp_solver::io::mesh_hash(&mesh),
                welded_positions: 0,
                dropped_degenerate: 0,
                triangulated_strips: 0,
                triangulated_fans: 0,
                source: "snapshot".into(),
            };
            crate::rtfp_solver::io::write_obj(out, &document)
                .map_err(|err| ScfError::model_not_closed(err.detail))
        }
        "glb" => write_glb(out, &mesh),
        other => Err(ScfError::model_not_closed(format!(
            "export format '{other}' is not obj or glb"
        ))),
    }
}
