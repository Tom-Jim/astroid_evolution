//! Positive ray radii `R(ω̂)` from a chart origin. One direction array, not one entity per ray.

use crate::common::geometry::{FlattenedBvh, TriMesh};
use crate::common::math::DVec3;
use crate::rtfp_solver::error::RtfpError;
use crate::rtfp_solver::geometry::ray_query::intersect_bvh;

pub fn ray_radii(
    mesh: &TriMesh,
    bvh: &FlattenedBvh,
    origin: DVec3,
    dirs: &[DVec3],
) -> Result<Vec<f64>, RtfpError> {
    let mut out = Vec::with_capacity(dirs.len());
    for dir in dirs {
        let dir = dir.try_normalize().ok_or_else(|| {
            RtfpError::model_not_closed("ray direction must be a nonzero finite vector")
        })?;
        let hits = intersect_bvh(mesh, bvh, origin, dir);
        let t = hits
            .iter()
            .map(|h| h.t)
            .find(|t| *t > 0.0)
            .ok_or_else(|| RtfpError::geometry("ray found no positive intersection"))?;
        out.push(t);
    }
    Ok(out)
}
