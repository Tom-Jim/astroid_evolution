//! Mesh and density inputs for the Tier 1 command line.
//!
//! Missing faces, units, or a density value stay as errors. Nothing in this module
//! substitutes a cube, a sphere, or a unit density.

pub mod aevb;
mod density;
pub mod memfs;
mod mesh;

pub use density::{
    DensityInput, assert_nonnegative, density_hash, load_density, positivity_points,
};
pub use mesh::{ImportRequest, MeshDocument, Reconstruct, load_mesh_file, mesh_hash, write_obj};

pub use memfs::install as install_asset;

use crate::common::error::SimError;
use crate::common::geometry::TriMesh;
use crate::common::math::DVec3;

pub fn length_to_meters(unit: &str) -> Result<f64, SimError> {
    match unit.trim() {
        "m" | "meter" | "meters" => Ok(1.0),
        "km" => Ok(1_000.0),
        "cm" => Ok(0.01),
        "mm" => Ok(0.001),
        "1" | "dimensionless" => Ok(1.0),
        // A positive number is the length of one file unit in metres: a model
        // whose scale is off is rescaled to the measured size.
        other => other
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| {
                SimError::model_not_closed(format!(
                    "length unit '{other}' is not m, km, cm, mm, 1, or metres per file unit"
                ))
            }),
    }
}

pub fn density_to_kg_m3(unit: &str) -> Result<f64, SimError> {
    match unit.trim() {
        "kg/m3" | "kg/m^3" => Ok(1.0),
        "g/cm3" | "g/cm^3" => Ok(1_000.0),
        "1" | "dimensionless" => Ok(1.0),
        other => Err(SimError::model_not_closed(format!(
            "density unit '{other}' is not one of kg/m3, g/cm3, or 1"
        ))),
    }
}

pub fn face_centroids(mesh: &TriMesh, max_points: Option<usize>) -> Vec<DVec3> {
    let count = max_points.unwrap_or(mesh.faces.len()).min(mesh.faces.len());
    (0..count)
        .map(|face| {
            let [a, b, c] = mesh.triangle(face);
            (a + b + c) * (1.0 / 3.0)
        })
        .collect()
}

pub fn vertex_points(mesh: &TriMesh, max_points: Option<usize>) -> Vec<DVec3> {
    let count = max_points
        .unwrap_or(mesh.vertices.len())
        .min(mesh.vertices.len());
    mesh.vertices[..count].to_vec()
}
