//! SCF view of the Tier1 star chart. Radial projection is refused here.

use crate::common::density::CauchyDensity;
use crate::common::geometry::TriMesh;
use crate::scf_simulator::error::ScfError;

pub use crate::rtfp_solver::geometry::star_audit::StarReport;

pub fn audit_star(
    mesh: &TriMesh,
    density: &CauchyDensity,
    star_min_cos: f64,
) -> Result<StarReport, ScfError> {
    crate::rtfp_solver::geometry::star_audit::audit_star(mesh, density, star_min_cos)
        .map_err(|err| ScfError::model_not_closed(err.detail))
}

pub fn refuse_radial(report: &StarReport) -> ScfError {
    ScfError::geometry_invalid(format!(
        "smoothed mesh is not star-shaped ({}); radial projection was not applied; star_margin {}",
        report.detail, report.star_margin
    ))
}
