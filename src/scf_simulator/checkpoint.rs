//! Text checkpoint with a CRC so a resumed step cannot silently drop a radius.

use std::path::Path;

use crate::scf_simulator::error::ScfError;

pub fn write_checkpoint(
    path: &Path,
    time: f64,
    mass: f64,
    mass_error: f64,
    radii: &[f64],
) -> Result<(), ScfError> {
    if !time.is_finite() || !mass.is_finite() || !mass_error.is_finite() {
        return Err(ScfError::model_not_closed(
            "checkpoint scalars are not finite",
        ));
    }
    let radii_text = radii
        .iter()
        .map(|radius| {
            if !radius.is_finite() {
                return Err(ScfError::model_not_closed(
                    "checkpoint radius is not finite",
                ));
            }
            Ok(format!("{radius:.16e}"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .join(" ");
    let body = format!(
        "time {time:.16e}\nmass {mass:.16e}\nmass_error {mass_error:.16e}\nradii {radii_text}\n"
    );
    let crc = crc32fast::hash(body.as_bytes());
    let text = format!("{body}crc {crc:08x}\n");
    std::fs::write(path, text).map_err(|err| ScfError::model_not_closed(err.to_string()))
}

pub fn read_checkpoint(path: &Path) -> Result<(f64, f64, f64, Vec<f64>), ScfError> {
    let text =
        std::fs::read_to_string(path).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    let Some((body, crc_line)) = text.rsplit_once("crc ") else {
        return Err(ScfError::model_not_closed("checkpoint has no checksum"));
    };
    let stored = u32::from_str_radix(crc_line.trim(), 16)
        .map_err(|_| ScfError::model_not_closed("checkpoint checksum is not an integer"))?;
    if crc32fast::hash(body.as_bytes()) != stored {
        return Err(ScfError::model_not_closed(
            "checkpoint checksum does not match",
        ));
    }
    let mut time = None;
    let mut mass = None;
    let mut mass_error = None;
    let mut radii = None;
    for line in body.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("time") => time = Some(parse_finite(parts.next())?),
            Some("mass") => mass = Some(parse_finite(parts.next())?),
            Some("mass_error") => mass_error = Some(parse_finite(parts.next())?),
            Some("radii") => {
                radii = Some(
                    parts
                        .map(|token| parse_finite(Some(token)))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            _ => {}
        }
    }
    Ok((
        time.ok_or_else(|| ScfError::model_not_closed("checkpoint time is missing"))?,
        mass.ok_or_else(|| ScfError::model_not_closed("checkpoint mass is missing"))?,
        mass_error.ok_or_else(|| ScfError::model_not_closed("checkpoint mass_error is missing"))?,
        radii.ok_or_else(|| ScfError::model_not_closed("checkpoint radii are missing"))?,
    ))
}

#[derive(Clone, Debug)]
pub struct MeshCheckpoint {
    pub time: f64,
    pub mass: f64,
    pub mass_error: f64,
    pub input_mesh_hash: String,
    pub mesh_hash: String,
    pub density_hash: String,
    pub vertex_count: usize,
    pub face_count: usize,
    pub vertices: Vec<crate::common::math::DVec3>,
    pub faces: Vec<[u32; 3]>,
}

pub fn write_mesh_checkpoint(path: &Path, saved: &MeshCheckpoint) -> Result<(), ScfError> {
    let mut body = format!(
        "format mesh\ntime {:.16e}\nmass {:.16e}\nmass_error {:.16e}\ninput_mesh_hash {}\nmesh_hash {}\ndensity_hash {}\nvertices {}\n",
        saved.time,
        saved.mass,
        saved.mass_error,
        saved.input_mesh_hash,
        saved.mesh_hash,
        saved.density_hash,
        saved.vertices.len()
    );
    for vertex in &saved.vertices {
        body.push_str(&format!(
            "{:.16e} {:.16e} {:.16e}\n",
            vertex.x, vertex.y, vertex.z
        ));
    }
    body.push_str(&format!("faces {}\n", saved.faces.len()));
    for face in &saved.faces {
        body.push_str(&format!("{} {} {}\n", face[0], face[1], face[2]));
    }
    let crc = crc32fast::hash(body.as_bytes());
    std::fs::write(path, format!("{body}crc {crc:08x}\n"))
        .map_err(|err| ScfError::model_not_closed(err.to_string()))
}

pub fn read_mesh_checkpoint(path: &Path) -> Result<MeshCheckpoint, ScfError> {
    let text =
        std::fs::read_to_string(path).map_err(|err| ScfError::model_not_closed(err.to_string()))?;
    if !text.contains("format mesh") {
        return Err(ScfError::model_not_closed(
            "checkpoint is not a mesh checkpoint",
        ));
    }
    let Some((body, crc_line)) = text.rsplit_once("crc ") else {
        return Err(ScfError::model_not_closed("checkpoint has no checksum"));
    };
    let stored = u32::from_str_radix(crc_line.trim(), 16)
        .map_err(|_| ScfError::model_not_closed("checkpoint checksum is not an integer"))?;
    if crc32fast::hash(body.as_bytes()) != stored {
        return Err(ScfError::model_not_closed(
            "checkpoint checksum does not match",
        ));
    }
    let mut lines = body.lines();
    let mut time = None;
    let mut mass = None;
    let mut mass_error = None;
    let mut input_mesh_hash = None;
    let mut mesh_hash = None;
    let mut density_hash = None;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    while let Some(line) = lines.next() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("time") => time = Some(parse_finite(parts.next())?),
            Some("mass") => mass = Some(parse_finite(parts.next())?),
            Some("mass_error") => mass_error = Some(parse_finite(parts.next())?),
            Some("input_mesh_hash") => input_mesh_hash = parts.next().map(str::to_string),
            Some("mesh_hash") => mesh_hash = parts.next().map(str::to_string),
            Some("density_hash") => density_hash = parts.next().map(str::to_string),
            Some("vertices") => {
                let count: usize = parts.next().unwrap_or("0").parse().map_err(|_| {
                    ScfError::model_not_closed("checkpoint vertex count is invalid")
                })?;
                for _ in 0..count {
                    let row = lines.next().ok_or_else(|| {
                        ScfError::model_not_closed("checkpoint vertex is missing")
                    })?;
                    let mut xyz = row.split_whitespace();
                    vertices.push(crate::common::math::DVec3::new(
                        parse_finite(xyz.next())?,
                        parse_finite(xyz.next())?,
                        parse_finite(xyz.next())?,
                    ));
                }
            }
            Some("faces") => {
                let count: usize =
                    parts.next().unwrap_or("0").parse().map_err(|_| {
                        ScfError::model_not_closed("checkpoint face count is invalid")
                    })?;
                for _ in 0..count {
                    let row = lines
                        .next()
                        .ok_or_else(|| ScfError::model_not_closed("checkpoint face is missing"))?;
                    let mut ids = row.split_whitespace();
                    let parse_index = |token: Option<&str>| -> Result<u32, ScfError> {
                        token.unwrap_or("").parse().map_err(|_| {
                            ScfError::model_not_closed("checkpoint face index is invalid")
                        })
                    };
                    faces.push([
                        parse_index(ids.next())?,
                        parse_index(ids.next())?,
                        parse_index(ids.next())?,
                    ]);
                }
            }
            _ => {}
        }
    }
    Ok(MeshCheckpoint {
        time: time.ok_or_else(|| ScfError::model_not_closed("checkpoint time is missing"))?,
        mass: mass.ok_or_else(|| ScfError::model_not_closed("checkpoint mass is missing"))?,
        mass_error: mass_error
            .ok_or_else(|| ScfError::model_not_closed("checkpoint mass_error is missing"))?,
        input_mesh_hash: input_mesh_hash
            .ok_or_else(|| ScfError::model_not_closed("checkpoint input mesh hash is missing"))?,
        mesh_hash: mesh_hash
            .ok_or_else(|| ScfError::model_not_closed("checkpoint mesh hash is missing"))?,
        density_hash: density_hash
            .ok_or_else(|| ScfError::model_not_closed("checkpoint density hash is missing"))?,
        vertex_count: vertices.len(),
        face_count: faces.len(),
        vertices,
        faces,
    })
}

fn parse_finite(token: Option<&str>) -> Result<f64, ScfError> {
    let value = token
        .ok_or_else(|| ScfError::model_not_closed("checkpoint value is missing"))?
        .parse::<f64>()
        .map_err(|_| ScfError::model_not_closed("checkpoint value is not numeric"))?;
    if !value.is_finite() {
        return Err(ScfError::model_not_closed("checkpoint value is not finite"));
    }
    Ok(value)
}
