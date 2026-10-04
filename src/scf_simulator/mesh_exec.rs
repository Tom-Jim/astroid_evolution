//! General-mesh Tier2 entry.
//!
//! A non-star Stokes/SCF discretization is not implemented. This function does not
//! project the mesh onto direction×radius, and it does not write a stable final mesh.

use crate::scf_simulator::error::ScfError;
use crate::scf_simulator::simulation::ClosedRun;

pub const GENERAL_MESH_UNSUPPORTED: &str = "Tier2 general_mesh is not complete: ModelUnsupported. The non-star Stokes/SCF discretization is not implemented, so no stable final mesh is written. direction*radius Galerkin, radial Jacobian, and radial SOCP were not applied";

pub fn run_general_mesh() -> Result<ClosedRun, ScfError> {
    Err(ScfError::model_unsupported(GENERAL_MESH_UNSUPPORTED))
}
