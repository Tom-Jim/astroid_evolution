//! Density files. A missing value is an error; it is not replaced by 1.

use super::{density_to_kg_m3, length_to_meters};
use crate::common::density::{CauchyDensity, CauchyKernel};
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;
use crate::rtfp_solver::rtfp::hw_rt::HardwareMesh;
use crate::rtfp_solver::error::RtfpError;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct DensityInput {
    /// `uniform` or `cauchy`.
    pub model: String,
    pub density_unit: String,
    pub length_unit: String,
    pub hash: String,
    pub density: CauchyDensity,
}

pub fn load_density(path: &Path) -> Result<DensityInput, RtfpError> {
    let text = std::fs::read_to_string(path).map_err(|err| {
        RtfpError::model_not_closed(format!("cannot read density {}: {err}", path.display()))
    })?;
    let value: toml::Value = toml::from_str(&text)
        .map_err(|err| RtfpError::model_not_closed(format!("density TOML is not valid: {err}")))?;
    let table = value
        .as_table()
        .ok_or_else(|| RtfpError::model_not_closed("density file must be a TOML table"))?;
    let density_unit = required_str(table, &["density_unit", "density_units"])?;
    let length_unit = required_str(table, &["length_unit", "length_units"])?;
    let length_scale = length_to_meters(length_unit)?;
    let density_scale = density_to_kg_m3(density_unit)?;
    let model_key = table
        .get("model")
        .or_else(|| table.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let density = if model_key == "uniform"
        || (table.contains_key("value") && !table.contains_key("kernels"))
    {
        let value = required_f64(table, &["value", "bulk", "rho"])?;
        uniform_density(value * density_scale)?
    } else {
        parse_cauchy(table, length_scale, density_scale)?
    };
    let model = if density.kernels.is_empty() { "uniform" } else { "cauchy" };
    let mut input = DensityInput {
        model: model.to_string(),
        density_unit: density_unit.to_string(),
        length_unit: length_unit.to_string(),
        hash: String::new(),
        density,
    };
    input.hash = density_hash(&input);
    Ok(input)
}

pub fn density_hash(input: &DensityInput) -> String {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(input.model.as_bytes());
    hasher.update(input.density_unit.as_bytes());
    hasher.update(input.length_unit.as_bytes());
    hasher.update(&input.density.bulk.to_le_bytes());
    for kernel in &input.density.kernels {
        for value in [
            kernel.center.x,
            kernel.center.y,
            kernel.center.z,
            kernel.weight,
            kernel.sigma,
            kernel.alpha,
        ] {
            hasher.update(&value.to_le_bytes());
        }
    }
    format!("{:08x}", hasher.finalize())
}

fn uniform_density(value: f64) -> Result<CauchyDensity, RtfpError> {
    if !(value.is_finite() && value > 0.0) {
        return Err(RtfpError::model_not_closed(
            "uniform density must be finite and positive; a missing value is not replaced by 1",
        ));
    }
    CauchyDensity::uniform(value).map_err(Into::into)
}

fn parse_cauchy(
    table: &toml::map::Map<String, toml::Value>,
    length_scale: f64,
    density_scale: f64,
) -> Result<CauchyDensity, RtfpError> {
    let alpha_default = table.get("alpha_default").and_then(|v| v.as_float());
    let bulk = if let Some(value) = table.get("bulk").or_else(|| table.get("bulk_density")) {
        value
            .as_float()
            .ok_or_else(|| RtfpError::model_not_closed("bulk density must be a number"))?
            * density_scale
    } else {
        // `mean_density` is a calibration note in the existing kernel files. It is not added on top of the kernels.
        0.0
    };
    let kernels_value = table.get("kernels").ok_or_else(|| {
        RtfpError::model_not_closed("Cauchy density file has no kernels array and no uniform value")
    })?;
    let kernels = kernels_value
        .as_array()
        .ok_or_else(|| RtfpError::model_not_closed("Cauchy kernels must be an array"))?;
    if kernels.is_empty() && bulk == 0.0 {
        return Err(RtfpError::model_not_closed(
            "Cauchy density has neither a positive bulk nor a kernel",
        ));
    }
    let mut parsed = Vec::with_capacity(kernels.len());
    for (index, kernel) in kernels.iter().enumerate() {
        let row = kernel
            .as_table()
            .ok_or_else(|| RtfpError::model_not_closed(format!("kernel {index} is not a table")))?;
        let center = vec3(row, &["center", "c"])
            .map_err(|err| RtfpError::model_not_closed(format!("kernel {index}: {err}")))?;
        let weight = row
            .get("weight")
            .or_else(|| row.get("w"))
            .and_then(|v| v.as_float())
            .ok_or_else(|| {
                RtfpError::model_not_closed(format!("kernel {index} is missing weight"))
            })?;
        let sigma = row.get("sigma").and_then(|v| v.as_float()).ok_or_else(|| {
            RtfpError::model_not_closed(format!("kernel {index} is missing sigma"))
        })?;
        let alpha = row
            .get("alpha")
            .and_then(|v| v.as_float())
            .or(alpha_default)
            .ok_or_else(|| {
                RtfpError::model_not_closed(format!(
                    "kernel {index} is missing alpha and the file has no alpha_default"
                ))
            })?;
        let center = DVec3::new(
            center[0] * length_scale,
            center[1] * length_scale,
            center[2] * length_scale,
        );
        let sigma = sigma / length_scale;
        parsed.push(
            CauchyKernel::try_new(center, weight * density_scale, sigma, alpha).map_err(|err| {
                RtfpError::model_not_closed(format!("kernel {index}: {}", err.detail))
            })?,
        );
    }
    CauchyDensity::try_new(bulk, parsed).map_err(Into::into)
}

pub fn positivity_points(
    mesh: &TriMesh,
    extra: &[DVec3],
) -> Result<Vec<(&'static str, DVec3)>, RtfpError> {
    let mut points = Vec::new();
    points.push(("interior", interior_sample(mesh)?));
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        points.push(("face", (a + b + c) * (1.0 / 3.0)));
    }
    for vertex in &mesh.vertices {
        points.push(("vertex", *vertex));
    }
    for point in extra {
        points.push(("near_surface", *point));
    }
    Ok(points)
}

pub fn assert_nonnegative(
    density: &CauchyDensity,
    points: &[(&str, DVec3)],
) -> Result<(), RtfpError> {
    for (role, point) in points {
        let rho = density.rho(*point);
        if !rho.is_finite() || rho < 0.0 {
            return Err(RtfpError::model_not_closed(format!(
                "density at {role} ({:.6e}, {:.6e}, {:.6e}) is {rho:.6e}; rho must be >= 0",
                point.x, point.y, point.z
            )));
        }
    }
    Ok(())
}

fn interior_sample(mesh: &TriMesh) -> Result<DVec3, RtfpError> {
    let scene = HardwareMesh::build(mesh)?;
    let mut min = mesh.vertices[0];
    let mut max = mesh.vertices[0];
    for vertex in &mesh.vertices {
        min.x = min.x.min(vertex.x);
        min.y = min.y.min(vertex.y);
        min.z = min.z.min(vertex.z);
        max.x = max.x.max(vertex.x);
        max.y = max.y.max(vertex.y);
        max.z = max.z.max(vertex.z);
    }
    let scale = (max - min).length().max(1e-12);
    let centroid = mesh
        .vertices
        .iter()
        .copied()
        .fold(DVec3::ZERO, |acc, v| acc + v)
        * (1.0 / mesh.vertices.len() as f64);
    if scene.inside(&[centroid])?[0] {
        return Ok(centroid);
    }
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let center = (a + b + c) * (1.0 / 3.0);
        let Some(normal) = (b - a).cross(c - a).try_normalize() else {
            continue;
        };
        for step in [1e-4, 1e-3, 1e-2] {
            let candidate = center - normal * (step * scale);
            if scene.inside(&[candidate])?[0] {
                return Ok(candidate);
            }
        }
    }
    Err(RtfpError::model_not_closed(
        "no interior sample was located for the density check",
    ))
}

fn required_str<'a>(
    table: &'a toml::map::Map<String, toml::Value>,
    keys: &[&str],
) -> Result<&'a str, RtfpError> {
    for key in keys {
        if let Some(value) = table.get(*key).and_then(|v| v.as_str()) {
            if !value.is_empty() {
                return Ok(value);
            }
        }
    }
    Err(RtfpError::model_not_closed(format!(
        "density file is missing {}",
        keys[0]
    )))
}

fn required_f64(
    table: &toml::map::Map<String, toml::Value>,
    keys: &[&str],
) -> Result<f64, RtfpError> {
    for key in keys {
        if let Some(value) = table.get(*key).and_then(|v| v.as_float()) {
            return Ok(value);
        }
    }
    Err(RtfpError::model_not_closed(format!(
        "density file is missing numeric {}",
        keys[0]
    )))
}

fn vec3(table: &toml::map::Map<String, toml::Value>, keys: &[&str]) -> Result<[f64; 3], String> {
    for key in keys {
        if let Some(value) = table.get(*key) {
            if let Some(array) = value.as_array() {
                if array.len() != 3 {
                    return Err(format!("{key} must have three components"));
                }
                let mut out = [0.0; 3];
                for (i, component) in array.iter().enumerate() {
                    out[i] = component
                        .as_float()
                        .ok_or_else(|| format!("{key} component {i} is not a number"))?;
                }
                return Ok(out);
            }
        }
    }
    Err(format!("missing {}", keys[0]))
}
