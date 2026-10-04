use std::fmt;

/// Failure of the rheology and friction-angle chain. The exit code of `rheo-cli` is `code()`.
#[derive(Clone, Debug)]
pub struct RheoError {
    pub kind: &'static str,
    pub detail: String,
}

impl RheoError {
    pub fn model_not_closed(detail: impl Into<String>) -> Self {
        Self {
            kind: "ModelNotClosed",
            detail: detail.into(),
        }
    }

    pub fn invalid_input(detail: impl Into<String>) -> Self {
        Self {
            kind: "InvalidInput",
            detail: detail.into(),
        }
    }

    pub fn gpu_unavailable(detail: impl Into<String>) -> Self {
        Self {
            kind: "GpuUnavailable",
            detail: detail.into(),
        }
    }

    pub fn rheology_not_converged(detail: impl Into<String>) -> Self {
        Self {
            kind: "RheologyNotConverged",
            detail: detail.into(),
        }
    }

    pub fn geometry(detail: impl Into<String>) -> Self {
        Self {
            kind: "GeometryInvalid",
            detail: detail.into(),
        }
    }

    pub fn code(&self) -> i32 {
        match self.kind {
            "ModelNotClosed" => 10,
            "GeometryInvalid" => 11,
            "InvalidInput" => 12,
            "RheologyNotConverged" => 40,
            "GpuUnavailable" => 50,
            _ => 1,
        }
    }
}

impl fmt::Display for RheoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}): {}", self.kind, self.code(), self.detail)
    }
}

impl std::error::Error for RheoError {}
