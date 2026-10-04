//! Failure kinds of Tier 2. The exit code of `scf-cli` is `code()`.

use std::fmt;

#[derive(Clone, Debug)]
pub struct ScfError {
    pub kind: &'static str,
    pub detail: String,
}

impl ScfError {
    fn make(kind: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    /// A required constant or input is missing or out of range (§10.4.4 step 0).
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self::make("ModelNotClosed", detail)
    }

    /// The body is not an admissible star-shaped solid.
    pub fn geometry(detail: impl Into<String>) -> Self {
        Self::make("GeometryInvalid", detail)
    }

    /// `B_rb` lost rank, or the load is incompatible with the rigid modes (§8.5, A.3).
    pub fn rigid_rank(detail: impl Into<String>) -> Self {
        Self::make("RigidRankDeficient", detail)
    }

    /// The Picard iteration, or the weak momentum balance, did not converge (§2.15 a, b).
    pub fn rheology_not_converged(detail: impl Into<String>) -> Self {
        Self::make("RheologyNotConverged", detail)
    }

    /// The outer geometry acceptance failed at the trust-radius floor (§10.4.3).
    pub fn trust_exhausted(detail: impl Into<String>) -> Self {
        Self::make("TrustRadiusExhausted", detail)
    }

    pub fn solver(detail: impl Into<String>) -> Self {
        Self::make("SolverFailure", detail)
    }

    pub fn io(detail: impl Into<String>) -> Self {
        Self::make("Io", detail)
    }

    pub fn gpu(detail: impl Into<String>) -> Self {
        Self::make("GpuUnavailable", detail)
    }

    pub fn code(&self) -> i32 {
        match self.kind {
            "ModelNotClosed" => 10,
            "GeometryInvalid" => 11,
            "RigidRankDeficient" => 13,
            "RheologyNotConverged" => 20,
            "TrustRadiusExhausted" => 23,
            "SolverFailure" => 30,
            "Io" => 31,
            "GpuUnavailable" => 50,
            _ => 1,
        }
    }
}

impl fmt::Display for ScfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code(), self.detail)
    }
}

impl std::error::Error for ScfError {}

impl From<crate::rtfp_solver::RtfpError> for ScfError {
    fn from(e: crate::rtfp_solver::RtfpError) -> Self {
        match e.kind {
            "GpuUnavailable" => Self::gpu(e.detail),
            "GeometryInvalid" => Self::geometry(e.detail),
            _ => Self::model_not_closed(e.detail),
        }
    }
}

impl From<crate::rheo_friction::RheoError> for ScfError {
    fn from(e: crate::rheo_friction::RheoError) -> Self {
        match e.kind {
            "GpuUnavailable" => Self::gpu(e.detail),
            "RheologyNotConverged" => Self::rheology_not_converged(e.detail),
            "GeometryInvalid" => Self::geometry(e.detail),
            _ => Self::model_not_closed(e.detail),
        }
    }
}

impl From<crate::common::error::CommonError> for ScfError {
    fn from(e: crate::common::error::CommonError) -> Self {
        Self::model_not_closed(e.detail)
    }
}
