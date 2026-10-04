//! Outer geometry acceptance. A finite KKT residual is not an argument here.

use crate::common::geometry::{StarMesh, TriMesh};
use crate::common::math::DVec3;

/// Non-grazing star condition `n · e_r >= c_0`.
pub const STAR_MIN: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug)]
pub struct RadialAudit {
    pub volume: f64,
    pub star_margin: f64,
}

/// Positive volume, no self-intersection, and a star-shaped radial chart.
/// `TriMesh::try_new` owns the closed-manifold and intersection test.
pub fn audit_radial_mesh(
    directions: &[DVec3],
    radii: &[f64],
    faces: &[[u32; 3]],
) -> Result<RadialAudit, String> {
    if directions.len() != radii.len() {
        return Err("radius and direction counts differ".into());
    }
    for (radius, direction) in radii.iter().zip(directions) {
        if !radius.is_finite() || *radius <= 0.0 || !direction.is_finite() {
            return Err("a candidate radius or direction is not finite and positive".into());
        }
        let scale = direction.length();
        if !scale.is_finite() || (scale - 1.0).abs() > 1.0e-6 {
            return Err("a radial direction is not a unit vector".into());
        }
    }
    let vertices = directions
        .iter()
        .zip(radii)
        .map(|(direction, radius)| *direction * *radius)
        .collect();
    let mesh = TriMesh::try_new(vertices, faces.to_vec()).map_err(|err| err.detail)?;
    let volume = mesh.signed_volume();
    if !volume.is_finite() || volume <= 0.0 {
        return Err("candidate volume is not finite and positive".into());
    }
    let normals = vertex_unit_normals(&mesh)?;
    let mut star_margin = f64::INFINITY;
    for (normal, direction) in normals.iter().zip(directions) {
        let margin = DVec3::from_array(*normal).dot(*direction);
        if !margin.is_finite() || margin < STAR_MIN {
            return Err(format!("star_margin {margin}"));
        }
        star_margin = star_margin.min(margin);
    }
    Ok(RadialAudit {
        volume,
        star_margin,
    })
}

pub fn vertex_unit_normals(mesh: &TriMesh) -> Result<Vec<[f64; 3]>, String> {
    let mut acc = vec![DVec3::ZERO; mesh.vertices.len()];
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let area = (b - a).cross(c - a);
        for index in mesh.faces[face] {
            acc[index as usize] = acc[index as usize] + area;
        }
    }
    acc.into_iter()
        .map(|normal| {
            normal
                .try_normalize()
                .map(|unit| unit.to_array())
                .ok_or_else(|| "a vertex normal is degenerate".to_string())
        })
        .collect()
}

pub fn vertex_areas(mesh: &TriMesh) -> Vec<f64> {
    let mut area = vec![0.0; mesh.vertices.len()];
    for face in 0..mesh.faces.len() {
        let [a, b, c] = mesh.triangle(face);
        let share = (b - a).cross(c - a).length() * 0.5 / 3.0;
        for index in mesh.faces[face] {
            area[index as usize] += share;
        }
    }
    area
}

/// U, ∇U and the Hessian are accepted only when every component is finite.
pub fn finite_gravitational_field(
    potential: &[f64],
    gradient: &[[f64; 3]],
    hessian: &[[f64; 6]],
) -> bool {
    potential.iter().all(|value| value.is_finite())
        && gradient
            .iter()
            .all(|row| row.iter().all(|value| value.is_finite()))
        && hessian
            .iter()
            .all(|row| row.iter().all(|value| value.is_finite()))
}

pub fn update_radial_mesh(mesh: &StarMesh, radii: &[f64]) -> Result<StarMesh, String> {
    if radii.len() != mesh.len() {
        return Err("radial update node count differs from the star mesh".into());
    }
    if radii.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return Err("radius must be a finite positive length".into());
    }
    let mut next = mesh.clone();
    next.radius = radii.to_vec();
    let _ = next.realize(DVec3::ZERO)?;
    Ok(next)
}
