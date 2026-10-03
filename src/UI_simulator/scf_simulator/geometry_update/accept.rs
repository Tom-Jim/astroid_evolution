//! Commit a new radius vector on the single `StarMesh` component.

use crate::common::geometry::StarMesh;
use crate::common::math::DVec3;

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
